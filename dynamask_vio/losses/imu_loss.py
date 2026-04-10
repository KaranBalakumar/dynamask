"""IMU integration error loss and covariance calibration (NLL) loss.

Uses PyPose for numerically stable SO(3) Log map.
All loss functions force float32 to avoid AMP half-precision issues with
linalg operations.
"""

import torch

from ..models.preintegration import so3_log_map


def _rotation_angle(R: torch.Tensor) -> torch.Tensor:
    """Geodesic angle ‖Log(R)‖ for batch of rotation matrices [B, 3, 3] → [B]."""
    log_vec = so3_log_map(R)  # [B, 3]
    return log_vec.norm(dim=-1)  # [B]


def imu_integration_loss(pred_R: torch.Tensor, pred_v: torch.Tensor,
                          pred_p: torch.Tensor, gt_R: torch.Tensor,
                          gt_v: torch.Tensor, gt_p: torch.Tensor,
                          w_rot: float = 5.0, w_vel: float = 1.0,
                          w_pos: float = 1.0) -> torch.Tensor:
    """Supervise preintegrated motion against ground truth.

    Args:
        pred_R, gt_R: [B, 3, 3] rotation matrices
        pred_v, gt_v: [B, 3] velocity
        pred_p, gt_p: [B, 3] position
        w_rot/w_vel/w_pos: sub-weights

    Returns:
        scalar loss
    """
    # Force float32 — AMP autocast may have produced half-precision tensors
    pred_R = pred_R.float()
    gt_R = gt_R.float()
    pred_v = pred_v.float()
    gt_v = gt_v.float()
    pred_p = pred_p.float()
    gt_p = gt_p.float()

    finite_mask = (
        torch.isfinite(pred_R).all(dim=(-2, -1))
        & torch.isfinite(gt_R).all(dim=(-2, -1))
        & torch.isfinite(pred_v).all(dim=-1)
        & torch.isfinite(gt_v).all(dim=-1)
        & torch.isfinite(pred_p).all(dim=-1)
        & torch.isfinite(gt_p).all(dim=-1)
    )
    if not finite_mask.any():
        # Keep graph connection so AMP GradScaler still records inf checks.
        return pred_v.sum() * 0.0

    pred_R = pred_R[finite_mask].clamp(min=-1e3, max=1e3)
    gt_R = gt_R[finite_mask].clamp(min=-1e3, max=1e3)
    pred_v = pred_v[finite_mask]
    gt_v = gt_v[finite_mask]
    pred_p = pred_p[finite_mask]
    gt_p = gt_p[finite_mask]

    # Rotation error: geodesic distance on SO(3)
    R_err = pred_R.transpose(-1, -2) @ gt_R
    angle_err = _rotation_angle(R_err)  # [B]
    loss_R = angle_err.pow(2).mean()

    vel_res = (pred_v - gt_v).clamp(min=-1e3, max=1e3)
    pos_res = (pred_p - gt_p).clamp(min=-1e3, max=1e3)
    loss_v = vel_res.pow(2).mean()
    loss_p = pos_res.pow(2).mean()

    return w_rot * loss_R + w_vel * loss_v + w_pos * loss_p


def covariance_nll_loss(pred_R: torch.Tensor, pred_v: torch.Tensor,
                         pred_p: torch.Tensor, gt_R: torch.Tensor,
                         gt_v: torch.Tensor, gt_p: torch.Tensor,
                         Sigma: torch.Tensor,
                         logdet_min: float = -20.0) -> torch.Tensor:
    """Negative log-likelihood — teaches calibrated uncertainty.

    NLL = 0.5 * (e^T Σ^{-1} e + log|Σ|)

    **Error is detached** — following AirIMU (Qiu et al.) and Air-IO, which
    pass ``dist.detach()`` into ``diag_ln_cov_loss``.  The rationale: NLL's
    gradient w.r.t. error is ``Σ^{-1} @ err``.  At init Σ is O(1e-8), so
    Σ^{-1} is O(1e8), which creates runaway gradients through the IMU head
    (observed as inf gradient norms at step 0).  By detaching the error, NLL
    only trains the Σ head (to match observed error magnitudes) while the
    L2 ``imu_integration_loss`` trains the error itself — no feedback loop.

    Uses slogdet + linalg.solve for numerical stability (no explicit inverse)
    and a 1e-2 identity floor, which bounds Σ^{-1} at 100 — keeps Σ-head
    gradients bounded even before the noise corrector has warmed up.
    """
    # Force float32 — linalg ops (solve, slogdet) don't support half on CUDA
    pred_R = pred_R.float()
    gt_R = gt_R.float()
    pred_v = pred_v.float()
    gt_v = gt_v.float()
    pred_p = pred_p.float()
    gt_p = gt_p.float()
    Sigma = Sigma.float()

    finite_mask = (
        torch.isfinite(pred_R).all(dim=(-2, -1))
        & torch.isfinite(gt_R).all(dim=(-2, -1))
        & torch.isfinite(pred_v).all(dim=-1)
        & torch.isfinite(gt_v).all(dim=-1)
        & torch.isfinite(pred_p).all(dim=-1)
        & torch.isfinite(gt_p).all(dim=-1)
        & torch.isfinite(Sigma).all(dim=(-2, -1))
    )
    if not finite_mask.any():
        return pred_v.sum() * 0.0

    pred_R = pred_R[finite_mask].clamp(min=-1e3, max=1e3)
    gt_R = gt_R[finite_mask].clamp(min=-1e3, max=1e3)
    pred_v = pred_v[finite_mask]
    gt_v = gt_v[finite_mask]
    pred_p = pred_p[finite_mask]
    gt_p = gt_p[finite_mask]
    Sigma = Sigma[finite_mask]

    # 9-d error vector: (rot, vel, pos).
    # Detach — NLL trains Σ, not err (see docstring).
    with torch.no_grad():
        err_R = so3_log_map(gt_R.transpose(-1, -2) @ pred_R)  # [B, 3]
        err_v = (gt_v - pred_v).clamp(min=-1e3, max=1e3)
        err_p = (gt_p - pred_p).clamp(min=-1e3, max=1e3)
        err = torch.cat([err_R, err_v, err_p], dim=-1)  # [B, 9]

    # Regularise Sigma — 1e-2 identity floor.  Bounds Σ^{-1} ≤ 100, which is
    # the key number: it caps ‖dNLL/dΣ‖ ~ ‖Σ^{-1}‖² ‖err‖² ≤ 1e4 · ‖err‖²,
    # which stays finite even when the noise corrector predicts near-zero
    # variances early in training.
    eye9 = torch.eye(9, device=Sigma.device, dtype=Sigma.dtype)
    Sigma_reg = Sigma + 1e-2 * eye9

    # Log-determinant via slogdet, with a floor to stop the covariance term
    # from collapsing the total loss when the error becomes very small.
    sign, logabsdet = torch.linalg.slogdet(Sigma_reg)
    # If sign < 0 (shouldn't happen for PD matrix + reg), treat as zero
    logdet = torch.where(sign > 0, logabsdet, torch.zeros_like(logabsdet))
    logdet = logdet.clamp(min=logdet_min)

    # Mahalanobis distance via solve (avoids explicit inverse, better numerics)
    # Solves: Sigma_reg @ x = err  →  x = Sigma_reg^{-1} @ err
    sol = torch.linalg.solve(Sigma_reg, err.unsqueeze(-1))  # [B, 9, 1]
    mahal = (err.unsqueeze(-1) * sol).sum(dim=[-2, -1])      # [B]
    # Guard: tiny negative values can occur from numerical noise in solve
    mahal = mahal.clamp(min=0.0)

    loss = 0.5 * (mahal + logdet).mean()
    return loss
