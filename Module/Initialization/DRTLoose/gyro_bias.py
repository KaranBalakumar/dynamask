"""
Gyro-bias estimation for the DRT-loose initializer.

Solves: b_g* = argmin_{b_g}  Σ_ij  ρ( ‖r_ij(b_g)‖² )

where the SO(3) residual per consecutive keyframe pair is:

    r_ij(b_g) = Log( R_vis_ij^{-1} · R_BC^T · ΔR_imu_ij(b_g) · R_BC )

Using a direct iterative Gauss-Newton solve with Huber kernel (δ = 1e-2 rad),
initialised at b_g = 0 (or from a caller-supplied prior).

The first-order correction ΔR(b̄+δ) ≈ ΔR(b̄) · Exp(J_R_bg · δb_g) avoids
full reintegration per iteration.
"""

from __future__ import annotations

import torch
import pypose as pp

from .preintegration import PreintResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _so3_log_jacobian(r: torch.Tensor) -> torch.Tensor:
    """
    Left Jacobian inverse of SO3 Log map at r (3-vector).

    For |r| < 0.5 rad use identity approximation; otherwise use the exact
    formula:  J_log^{-1} = I + 0.5 [r]× + (1/|r|² - (1+cos|r|)/(2|r|sin|r|)) [r]×²

    Returns a (3,3) matrix such that:
        d(Log(Exp(r) · Exp(δ))) / dδ  |_{δ=0}  ≈  J_log^{-1}(r)
    """
    theta = r.norm()
    if theta.item() < 0.5:
        return torch.eye(3, dtype=r.dtype, device=r.device)

    sk = pp.vec2skew(r)  # (3,3) skew-symmetric [r]×
    c  = torch.cos(theta)
    s  = torch.sin(theta)
    t2 = theta * theta
    coeff = (1.0 / t2 - (1.0 + c) / (2.0 * theta * s))
    return torch.eye(3, dtype=r.dtype, device=r.device) + 0.5 * sk + coeff * (sk @ sk)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def rotation_residual(
    R_vis_ij: torch.Tensor,   # (3,3) visual relative rotation
    dR_imu_ij: torch.Tensor,  # (3,3) IMU preintegrated relative rotation (current b_g)
    R_BC: torch.Tensor,       # (3,3) IMU-to-camera rotation
) -> torch.Tensor:
    """
    Compute the 3-vector SO3 residual r = Log(R_vis^{-1} @ R_BC^T @ dR_imu @ R_BC).

    Parameters
    ----------
    R_vis_ij  : (3,3) visual relative rotation  R_ij from camera tracking.
    dR_imu_ij : (3,3) IMU preintegrated relative rotation ΔR_ij(b_g).
    R_BC      : (3,3) rotation from IMU body frame to camera frame.

    Returns
    -------
    r : (3,) SO3 logarithm of the rotation error.
    """
    # R_err = R_vis^{-1} @ R_BC^T @ dR_imu @ R_BC
    R_err = R_vis_ij.T @ R_BC.T @ dR_imu_ij @ R_BC        # (3,3)
    # Convert to SO3 LieTensor, take Log, return plain (3,) tensor
    r = pp.mat2SO3(R_err).Log().tensor()                   # (3,)
    return r


def solve_gyro_bias(
    preint_results: list[PreintResult],      # one per consecutive keyframe gap
    R_vis_list: list[torch.Tensor],          # list of (3,3) visual relative rotations R_vis_ij
    R_BC: torch.Tensor,                      # (3,3) IMU-to-camera extrinsic rotation
    b_g_init: torch.Tensor | None = None,   # (3,) initial guess; defaults to zeros
    huber_delta: float = 1e-2,
    max_iters: int = 50,
) -> tuple[torch.Tensor, bool]:
    """
    Estimate gyroscope bias via iterative Gauss-Newton with a Huber kernel.

    Uses a first-order correction to avoid full reintegration per iteration:
        ΔR(b̄+δ) ≈ ΔR(b̄) · Exp(J_R_bg · δb_g)

    Parameters
    ----------
    preint_results : N-1 preintegration results (one per consecutive kf gap).
    R_vis_list     : N-1 visual relative rotations R_vis_ij, each (3,3).
    R_BC           : (3,3) rotation from IMU body frame to camera frame.
    b_g_init       : (3,) initial gyro bias; zeros if None.
    huber_delta    : Huber kernel threshold [rad].
    max_iters      : maximum Gauss-Newton iterations.

    Returns
    -------
    (b_g_star, converged)
      b_g_star  : (3,) estimated gyro bias.
      converged : True if step norm dropped below 1e-8 before max_iters.
    """
    assert len(preint_results) == len(R_vis_list), (
        "preint_results and R_vis_list must have the same length"
    )

    dtype  = torch.float64
    device = R_BC.device

    R_BC = R_BC.to(dtype=dtype, device=device)

    b_g = (b_g_init.clone().to(dtype=dtype, device=device)
           if b_g_init is not None
           else torch.zeros(3, dtype=dtype, device=device))

    converged = False

    for _it in range(max_iters):
        JtWJ = torch.zeros(3, 3, dtype=dtype, device=device)
        JtWr = torch.zeros(3,    dtype=dtype, device=device)

        for pr, R_vis_ij in zip(preint_results, R_vis_list):
            R_vis_ij = R_vis_ij.to(dtype=dtype, device=device)

            # ---- Current ΔR with first-order bias correction ----------------
            # ΔR_log stored at preintegration bias (b_g=0 at integration time).
            # Correction: ΔR(b_g) ≈ ΔR(0) · Exp(J_R_bg · b_g)
            # (b_g appears with negative sign inside the integrator, but
            # J_R_bg already encodes that sign via the Forster eq-45 convention.)
            J_R_bg = pr.J_R_bg.to(dtype=dtype, device=device)  # (3,3)
            dR_log0 = pr.dR_log.to(dtype=dtype, device=device)  # (3,) log at stored bias

            # ΔR base rotation matrix
            dR_base = pp.so3(dR_log0).Exp().matrix().squeeze(0)   # (3,3)

            # bias-correction rotation Exp(J_R_bg · b_g)
            bias_corr_log = J_R_bg @ b_g                           # (3,)
            dR_bias_corr  = pp.so3(bias_corr_log).Exp().matrix().squeeze(0)  # (3,3)

            dR_ij = dR_base @ dR_bias_corr                         # (3,3)

            # ---- Residual ---------------------------------------------------
            r = rotation_residual(R_vis_ij, dR_ij, R_BC)           # (3,)
            r_norm = r.norm()

            # ---- Huber weight -----------------------------------------------
            w = min(1.0, huber_delta / r_norm.item()) if r_norm.item() > 1e-12 else 1.0

            # ---- Jacobian of residual wrt b_g --------------------------------
            # r = Log(R_vis^{-1} @ R_BC^T @ ΔR_ij @ R_BC)
            # ΔR_ij = dR_base @ Exp(J_R_bg · b_g)
            # d(ΔR_ij)/d(b_g) at current b_g = dR_base @ Exp(·) · J_R_bg
            #                                = dR_ij · J_R_bg  (right-perturbation)
            # Then:
            #   dr/d(b_g) = J_log^{-1}(r) · R_BC^T · dR_ij · J_R_bg
            #
            # (The R_vis^{-1} pulls out as a left factor and does not affect
            #  the right-perturbation Jacobian.)
            J_log_inv = _so3_log_jacobian(r)                       # (3,3)
            J_r = J_log_inv @ R_BC.T @ dR_ij @ J_R_bg              # (3,3)

            # ---- Accumulate weighted normal equations -----------------------
            JtWJ += w * (J_r.T @ J_r)
            JtWr += w * (J_r.T @ r)

        # ---- Solve: (J^T W J) δb_g = -J^T W r -----------------------------
        try:
            delta_b_g = torch.linalg.solve(JtWJ, -JtWr)
        except torch.linalg.LinAlgError:
            # Singular system – stop early
            break

        b_g = b_g + delta_b_g

        if delta_b_g.norm().item() < 1e-8:
            converged = True
            break

    return b_g, converged
