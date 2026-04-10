"""
Differentiable Two-Frame Bundle Adjustment — training-only module.

Zero learnable parameters. Provides geometric consistency signal for
self-supervised mask training via unrolled Gauss-Newton optimisation.

Key mechanisms (informed by DPVO, Teed et al., NeurIPS 2023):
  - SafeCholeskySolver: zero update + zero gradient on singular Hessian
  - GradientClip: per-element backward clamp to ±0.01
  - Hard outlier rejection (100px threshold)
  - Adaptive damping (ep decays from 10 to 1 over training)
  - Convergence check: detach pose if mean reproj > 50px
  - Depth validity check (Z > 0.2)

The BA is discarded at inference — the trained model never needs it.

Reference:
  DPVO ba.py:   CholeskySolver, block_solve damping formula
  DPVO ba_cuda.cu: Jacobian structure, outlier rejection, reprojection
  DPVO blocks.py:  GradClip (±0.01)
"""

import torch
import torch.nn as nn


class SafeCholeskySolver(torch.autograd.Function):
    """Differentiable Cholesky solve with failure recovery.

    If the Hessian is singular or near-singular, returns zero update
    in forward and zero gradient in backward. This prevents NaN
    propagation from degenerate correspondence configurations.

    Matches DPVO (dpvo/ba.py:12-37) — uses cholesky_ex (never throws).
    """

    @staticmethod
    def forward(ctx, H, b):
        """Solve H x = b via Cholesky decomposition.

        Args:
            H: [B, 6, 6] positive-definite Hessian
            b: [B, 6, 1] right-hand side (column vector)

        Returns:
            x: [B, 6, 1] solution (zero if Cholesky fails)
        """
        # cholesky_ex never throws — returns info != 0 on failure
        U, info = torch.linalg.cholesky_ex(H)

        if torch.any(info != 0):
            ctx.failed = True
            return torch.zeros_like(b)

        xs = torch.cholesky_solve(b, U)
        ctx.save_for_backward(U, xs)
        ctx.failed = False
        return xs

    @staticmethod
    def backward(ctx, grad_x):
        if ctx.failed:
            return None, None

        U, xs = ctx.saved_tensors
        dz = torch.cholesky_solve(grad_x, U)
        dH = -torch.matmul(xs, dz.transpose(-1, -2))
        return dH, dz


class _GradientClipFn(torch.autograd.Function):
    """Per-element gradient clamp: identity forward, clamp backward.

    Adopted from DPVO (dpvo/blocks.py:74-82).
    """

    @staticmethod
    def forward(ctx, x, clip_val):
        ctx.clip_val = clip_val
        return x

    @staticmethod
    def backward(ctx, grad_x):
        grad_x = torch.where(torch.isnan(grad_x),
                              torch.zeros_like(grad_x), grad_x)
        return grad_x.clamp(min=-ctx.clip_val, max=ctx.clip_val), None


def gradient_clip(x: torch.Tensor, clip_val: float = 0.01) -> torch.Tensor:
    """Apply per-element gradient clipping."""
    return _GradientClipFn.apply(x, clip_val)


def _skew(v: torch.Tensor) -> torch.Tensor:
    """Batch skew-symmetric matrix. v: [*, 3] -> [*, 3, 3]."""
    shape = v.shape[:-1]
    zero = torch.zeros(*shape, device=v.device, dtype=v.dtype)
    x, y, z = v[..., 0], v[..., 1], v[..., 2]
    return torch.stack([
        zero, -z, y,
        z, zero, -x,
        -y, x, zero,
    ], dim=-1).reshape(*shape, 3, 3)


def _so3_exp(omega: torch.Tensor) -> torch.Tensor:
    """Exponential map so(3) -> SO(3). omega: [B, 3] -> [B, 3, 3].

    Rodrigues formula with small-angle Taylor expansion for stability.
    Matches DPVO ba_cuda.cu:expSO3 (quaternion form) but using matrix form.
    """
    theta_sq = (omega * omega).sum(dim=-1, keepdim=True)  # [B, 1]
    theta = theta_sq.sqrt().clamp(min=1e-12)  # [B, 1]

    # Normalised axis
    axis = omega / theta  # [B, 3]
    K = _skew(axis)  # [B, 3, 3]

    theta = theta.unsqueeze(-1)  # [B, 1, 1]
    I = torch.eye(3, device=omega.device, dtype=omega.dtype).unsqueeze(0)

    # Small angle: R ≈ I + K*θ + K²*θ²/2  (first-order in θ)
    # Full: R = I + sin(θ)K + (1-cos(θ))K²
    sin_t = torch.sin(theta)
    cos_t = torch.cos(theta)

    R = I + sin_t * K + (1 - cos_t) * (K @ K)
    return R


def _so3_log(R: torch.Tensor) -> torch.Tensor:
    """Logarithmic map SO(3) -> so(3). R: [B, 3, 3] -> [B, 3].

    Inverse Rodrigues with numerical stability.
    """
    # cos(θ) = (tr(R) - 1) / 2
    trace = R[:, 0, 0] + R[:, 1, 1] + R[:, 2, 2]
    cos_theta = ((trace - 1.0) / 2.0).clamp(-1.0 + 1e-7, 1.0 - 1e-7)
    theta = torch.acos(cos_theta)  # [B]

    # Antisymmetric part of R
    vee = torch.stack([
        R[:, 2, 1] - R[:, 1, 2],
        R[:, 0, 2] - R[:, 2, 0],
        R[:, 1, 0] - R[:, 0, 1],
    ], dim=-1)  # [B, 3]

    # For small angles: ω ≈ vee/2
    # General: ω = θ/(2 sin θ) * vee
    small = theta.abs() < 1e-6
    factor = torch.where(
        small,
        0.5 + theta ** 2 / 12.0,
        theta / (2.0 * torch.sin(theta).clamp(min=1e-12)))

    return factor.unsqueeze(-1) * vee  # [B, 3]


def se3_exp_update(R: torch.Tensor, t: torch.Tensor,
                   delta_xi: torch.Tensor):
    """Update SE(3) pose via retraction (left-multiply by exp(δξ)).

    Convention: δξ = [δt(3), δω(3)] — translation first, then rotation.
    Matches DPVO ba_cuda.cu:retrSE3.

    new_pose = exp(δξ) ∘ old_pose
    R_new = δR @ R
    t_new = δR @ t + δt

    Args:
        R: [B, 3, 3] rotation
        t: [B, 3] translation
        delta_xi: [B, 6] tangent vector

    Returns:
        R_new [B, 3, 3], t_new [B, 3]
    """
    delta_t = delta_xi[:, :3]     # [B, 3]
    delta_omega = delta_xi[:, 3:]  # [B, 3]

    dR = _so3_exp(delta_omega)  # [B, 3, 3]
    R_new = dR @ R
    t_new = (dR @ t.unsqueeze(-1)).squeeze(-1) + delta_t

    return R_new, t_new


def triangulate_midpoint(p1: torch.Tensor, p2: torch.Tensor,
                         R: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """Triangulate 3D points via midpoint method.

    Camera 1 is at origin.
    Camera 2 has pose (R, t) meaning X_2 = R @ X_1 + t.
    So camera 2 origin in frame 1 = -R^T @ t.
    Ray from cam2 in frame 1 coords: d2_frame1 = R^T @ p2.

    Args:
        p1: [B, K, 3] bearing vectors in frame 1 (normalised image coords)
        p2: [B, K, 3] bearing vectors in frame 2
        R: [B, 3, 3] rotation (frame1 -> frame2)
        t: [B, 3] translation (frame1 -> frame2)

    Returns:
        X: [B, K, 3] 3D points in frame 1 coordinates
    """
    B, K, _ = p1.shape

    # Ray directions
    d1 = p1  # [B, K, 3]
    # Transform p2 from frame 2 to frame 1: R^T @ p2 for each point
    Rt = R.transpose(-1, -2)  # [B, 3, 3]
    d2 = torch.bmm(p2, Rt.transpose(-1, -2))  # p2 @ (R^T)^T = p2 @ R
    # Wait — we need R^T @ p2_i for each point.
    # p2 is [B, K, 3], R^T is [B, 3, 3]
    # (R^T @ p2_i^T)^T = p2_i @ R, so d2 = p2 @ R would give R^T applied
    # No: (R^T @ v)^T = v^T @ R. So for row vectors: d2 = p2 @ R is correct
    # for getting R^T @ p2 when p2 is stored as rows.
    d2 = torch.bmm(p2, R)  # [B, K, 3] = R^T applied to each row of p2

    # Camera 2 origin in frame 1 coords: c2 = -R^T @ t
    c2 = -(Rt @ t.unsqueeze(-1)).squeeze(-1)  # [B, 3]
    c2 = c2.unsqueeze(1).expand(-1, K, -1)    # [B, K, 3]

    # Closest point between rays (midpoint triangulation)
    # Ray 1: P = s * d1  (origin at 0)
    # Ray 2: P = c2 + r * d2
    d1d1 = (d1 * d1).sum(dim=-1, keepdim=True)  # [B, K, 1]
    d2d2 = (d2 * d2).sum(dim=-1, keepdim=True)
    d1d2 = (d1 * d2).sum(dim=-1, keepdim=True)

    denom = (d1d1 * d2d2 - d1d2 * d1d2).clamp(min=1e-8)

    d1c = (d1 * c2).sum(dim=-1, keepdim=True)
    d2c = (d2 * c2).sum(dim=-1, keepdim=True)

    s = (d2d2 * d1c - d1d2 * d2c) / denom
    r = (d1d2 * d1c - d1d1 * d2c) / denom

    # Midpoint of closest approach
    P1 = s * d1               # [B, K, 3] — point on ray 1
    P2 = c2 + r * d2          # [B, K, 3] — point on ray 2
    X = 0.5 * (P1 + P2)       # [B, K, 3] — midpoint

    return X


def compute_reprojection_jacobian(X_cam: torch.Tensor,
                                   K: torch.Tensor) -> torch.Tensor:
    """Compute Jacobian of reprojection w.r.t. 6-DoF pose delta.

    Convention: δξ = [δt(3), δω(3)].
    The perturbation model is: X_cam' = δR @ X_cam + δt
    So d(X_cam')/d(δt) = I, d(X_cam')/d(δω) = -[X_cam]_x.

    This matches DPVO's Jacobian structure in ba_cuda.cu
    (Jj for the target frame).

    Args:
        X_cam: [B, K, 3] 3D points in camera frame
        K: [B, 3, 3] camera intrinsic matrix

    Returns:
        J: [B, K, 2, 6] Jacobian
    """
    B, K_pts, _ = X_cam.shape

    X = X_cam[:, :, 0]  # [B, K]
    Y = X_cam[:, :, 1]
    Z = X_cam[:, :, 2]

    fx = K[:, 0, 0].unsqueeze(-1)  # [B, 1]
    fy = K[:, 1, 1].unsqueeze(-1)

    # Safe inverse depth (match DPVO: d = Z >= 0.2 ? 1/Z : 0)
    d = torch.where(Z.abs() > 0.2, 1.0 / Z, torch.zeros_like(Z))
    d2 = d * d

    # Jacobian of pixel coords w.r.t. SE(3) perturbation δξ = (δt, δω)
    # u = fx * X/Z + cx, v = fy * Y/Z + cy
    #
    # du/d(δt) = [fx*d, 0, -fx*X*d²]
    # du/d(δω) = [-fx*X*Y*d², fx*(1+X²*d²), -fx*Y*d]
    # dv/d(δt) = [0, fy*d, -fy*Y*d²]
    # dv/d(δω) = [-fy*(1+Y²*d²), fy*X*Y*d², fy*X*d]
    #
    # This matches DPVO ba_cuda.cu Jj structure.

    J = torch.zeros(B, K_pts, 2, 6, device=X_cam.device, dtype=X_cam.dtype)

    # du/dξ (row 0)
    J[:, :, 0, 0] = fx * d           # du/dtx
    J[:, :, 0, 1] = 0                # du/dty
    J[:, :, 0, 2] = -fx * X * d2     # du/dtz
    J[:, :, 0, 3] = -fx * X * Y * d2     # du/dωx
    J[:, :, 0, 4] = fx * (1 + X * X * d2)  # du/dωy
    J[:, :, 0, 5] = -fx * Y * d      # du/dωz

    # dv/dξ (row 1)
    J[:, :, 1, 0] = 0                # dv/dtx
    J[:, :, 1, 1] = fy * d           # dv/dty
    J[:, :, 1, 2] = -fy * Y * d2     # dv/dtz
    J[:, :, 1, 3] = -fy * (1 + Y * Y * d2)  # dv/dωx
    J[:, :, 1, 4] = fy * X * Y * d2  # dv/dωy
    J[:, :, 1, 5] = fy * X * d       # dv/dωz

    return J


def differentiable_ba(correspondences: torch.Tensor,
                      weights: torch.Tensor,
                      K: torch.Tensor,
                      R_init: torch.Tensor,
                      epoch: int,
                      phase2_start: int = 30,
                      n_iters: int = 3,
                      base_damping: float = 1e-4,
                      ep_initial: float = 10.0,
                      ep_final: float = 1.0,
                      ep_decay_epochs: int = 10,
                      outlier_threshold: float = 100.0,
                      convergence_threshold: float = 50.0) -> dict:
    """Estimate relative pose (R, t) from weighted correspondences.

    Fully differentiable — gradients flow back through correspondences and weights.

    Damping formula matches DPVO:
      DPVO ba.py block_solve:  A += (ep + lm * A) * I
      DPVO ba_cuda.cu:         S += I * (1e-4 * S + 1.0)
      Ours:                    H += (ep + base_damping * diag(H)) * I

    Args:
        correspondences: [B, K, 4] — (u, v, u', v') pixel coordinates
        weights: [B, K] — soft static confidence (1 - mask_prob)
        K: [B, 3, 3] — camera intrinsics
        R_init: [B, 3, 3] — IMU preintegrated rotation (warm start)
        epoch: current training epoch
        phase2_start: epoch at which Phase 2b started
        n_iters: number of Gauss-Newton iterations
        base_damping: multiplicative LM damping (DPVO uses 1e-4)
        ep_initial: additive damping at Phase 2b start (DPVO uses 10.0)
        ep_final: additive damping at steady state
        ep_decay_epochs: epochs to decay ep
        outlier_threshold: hard rejection threshold in pixels
        convergence_threshold: detach pose if mean reproj > this

    Returns:
        dict with:
            R: [B, 3, 3] estimated rotation
            t: [B, 3] estimated translation
            reproj_error: [B, K, 2] final reprojection errors
            converged: [B] float mask (1.0 if converged)
            mean_reproj: [B] mean reprojection error per sample
            n_inliers: [B] number of valid inlier points per sample
    """
    B, K_pts, _ = correspondences.shape
    device = correspondences.device
    dtype = correspondences.dtype

    # Adaptive damping schedule
    epochs_into_phase2 = max(0, epoch - phase2_start)
    if ep_decay_epochs > 0:
        decay_frac = min(1.0, epochs_into_phase2 / ep_decay_epochs)
    else:
        decay_frac = 1.0
    ep = ep_initial * (1.0 - decay_frac) + ep_final * decay_frac

    # Initialise pose: rotation from IMU, translation as unit forward
    R = R_init.clone()
    t = torch.zeros(B, 3, device=device, dtype=dtype)
    t[:, 2] = 1.0

    # Extract correspondences
    uv1 = correspondences[:, :, :2]   # [B, K, 2] frame t-1
    uv2 = correspondences[:, :, 2:4]  # [B, K, 2] frame t

    # Bearing vectors (normalised image coordinates)
    K_inv = torch.linalg.inv(K)
    ones = torch.ones(B, K_pts, 1, device=device, dtype=dtype)
    p1_h = torch.cat([uv1, ones], dim=-1)
    p2_h = torch.cat([uv2, ones], dim=-1)

    # p1 = K_inv @ [u, v, 1]^T for each point (row vectors)
    p1 = torch.bmm(p1_h, K_inv.transpose(-1, -2))  # [B, K, 3]
    p2 = torch.bmm(p2_h, K_inv.transpose(-1, -2))

    err = torch.zeros(B, K_pts, 2, device=device, dtype=dtype)
    n_inliers = torch.zeros(B, device=device, dtype=dtype)

    for i in range(n_iters):
        # Triangulate 3D points in frame 1
        X = triangulate_midpoint(p1, p2, R, t)  # [B, K, 3]

        # Transform to frame 2: X_2 = R @ X + t
        X_2 = (R @ X.transpose(-1, -2)).transpose(-1, -2) + t.unsqueeze(1)
        # X_2: [B, K, 3]

        Z = X_2[:, :, 2]  # [B, K]

        # Project to pixels: [u, v] = [fx*X/Z + cx, fy*Y/Z + cy]
        proj_2 = (K @ X_2.transpose(-1, -2)).transpose(-1, -2)  # [B, K, 3]
        proj_2_px = proj_2[:, :, :2] / proj_2[:, :, 2:3].clamp(min=1e-8)

        # Reprojection error
        err = proj_2_px - uv2  # [B, K, 2]

        # Hard outlier rejection + depth check
        # DPVO ba_cuda.cu: sqrt(rx*rx+ry*ry) < 128 && Z > 0.2
        err_norm = err.norm(dim=-1)  # [B, K]
        valid = (err_norm < outlier_threshold) & (Z > 0.2)
        valid_f = valid.float()
        n_inliers = valid_f.sum(dim=-1)  # [B]

        # Effective weights: soft mask confidence × hard validity
        w_eff = weights * valid_f  # [B, K]

        # Jacobian of reprojection w.r.t. pose
        J = compute_reprojection_jacobian(X_2, K)  # [B, K, 2, 6]

        # Build normal equations: H δξ = -g
        # H = J^T W J, g = J^T W e
        # J: [B, K, 2, 6], w_eff: [B, K], err: [B, K, 2]
        JtW = J.transpose(-1, -2) * w_eff.unsqueeze(-1).unsqueeze(-1)
        # JtW: [B, K, 6, 2]  (each [6,2] block weighted)
        H = torch.einsum("bkij,bkjl->bil", JtW, J)  # [B, 6, 6]
        g = torch.einsum("bkij,bkj->bi", JtW, err)   # [B, 6]

        # Damped normal equations (DPVO: S += I * (lm * S + ep))
        diag_H = H.diagonal(dim1=-2, dim2=-1)  # [B, 6]
        I6 = torch.eye(6, device=device, dtype=dtype).unsqueeze(0)
        H_damped = H + (base_damping * diag_H.unsqueeze(-1) + ep) * I6

        # Solve via SafeCholeskySolver
        delta_xi = SafeCholeskySolver.apply(H_damped, -g.unsqueeze(-1))
        delta_xi = delta_xi.squeeze(-1)  # [B, 6]

        # Per-element gradient clamp
        delta_xi = gradient_clip(delta_xi)

        # Update pose
        R, t = se3_exp_update(R, t, delta_xi)

    # Convergence check
    mean_err = err.norm(dim=-1).mean(dim=-1)  # [B]
    converged = (mean_err < convergence_threshold).float()

    # Detach pose for non-converged samples (zero gradient from L_pose)
    R = R * converged.view(B, 1, 1) + R.detach() * (1 - converged.view(B, 1, 1))
    t = t * converged.view(B, 1) + t.detach() * (1 - converged.view(B, 1))

    return {
        "R": R,
        "t": t,
        "reproj_error": err,
        "converged": converged,
        "mean_reproj": mean_err,
        "n_inliers": n_inliers,
    }


def select_static_correspondences(flow: torch.Tensor,
                                   mask: torch.Tensor,
                                   num_points: int = 256,
                                   mask_threshold: float = 0.3,
                                   min_points: int = 64):
    """Select static correspondences from predicted flow and mask.

    Selects pixels with low dynamic probability, extracts their flow-based
    correspondences, and returns soft weights for the BA.

    Args:
        flow: [B, 2, H, W] predicted optical flow at 1/8 resolution
        mask: [B, 1, H, W] dynamic probability (after sigmoid) at 1/8 res
        num_points: K — number of correspondences to select
        mask_threshold: pixels with mask < this are considered static
        min_points: minimum static points required (skip BA if fewer)

    Returns:
        correspondences: [B, K, 4] — (u, v, u', v') at 1/8 resolution scale
        weights: [B, K] — soft static confidence (1 - mask_prob)
        valid_batch: [B] bool — True if enough static points found
    """
    B, _, H, W = flow.shape
    device = flow.device
    dtype = flow.dtype

    mask_prob = mask.squeeze(1)  # [B, H, W]

    correspondences_list = []
    weights_list = []
    valid_list = []

    for b in range(B):
        # Find static pixels
        static_mask = mask_prob[b] < mask_threshold  # [H, W]
        static_idx = static_mask.nonzero(as_tuple=False)  # [N_static, 2]

        n_static = static_idx.shape[0]

        if n_static < min_points:
            correspondences_list.append(
                torch.zeros(num_points, 4, device=device, dtype=dtype))
            weights_list.append(
                torch.zeros(num_points, device=device, dtype=dtype))
            valid_list.append(False)
            continue

        # Subsample uniformly
        if n_static > num_points:
            perm = torch.randperm(n_static, device=device)[:num_points]
            static_idx = static_idx[perm]
        elif n_static < num_points:
            extra = num_points - n_static
            perm = torch.randint(0, n_static, (extra,), device=device)
            static_idx = torch.cat([static_idx, static_idx[perm]], dim=0)

        rows = static_idx[:, 0]  # y coords
        cols = static_idx[:, 1]  # x coords

        u1 = cols.to(dtype)
        v1 = rows.to(dtype)

        flow_u = flow[b, 0, rows, cols]
        flow_v = flow[b, 1, rows, cols]

        u2 = u1 + flow_u
        v2 = v1 + flow_v

        corr = torch.stack([u1, v1, u2, v2], dim=-1)  # [K, 4]
        w = 1.0 - mask_prob[b, rows, cols]  # [K]

        correspondences_list.append(corr)
        weights_list.append(w)
        valid_list.append(True)

    correspondences = torch.stack(correspondences_list, dim=0)  # [B, K, 4]
    weights = torch.stack(weights_list, dim=0)  # [B, K]
    valid_batch = torch.tensor(valid_list, device=device, dtype=torch.bool)

    return correspondences, weights, valid_batch
