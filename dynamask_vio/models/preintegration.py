"""
Differentiable IMU preintegration using PyPose.

Implements Forster et al. preintegration on SO(3) x R^6 with first-order
covariance propagation.  All operations are fully differentiable so gradients
flow back through the noise corrector.

Uses variable per-sample dt (real IMU is non-uniform).
"""

import torch
import torch.nn as nn
import pypose as pp
from pypose import SO3
from contextlib import nullcontext


class DifferentiablePreintegrator(nn.Module):
    """Pure-computation module (no learnable parameters).

    Input
    -----
    gyro   : [B, N, 3]  corrected gyroscope readings (rad/s)
    accel  : [B, N, 3]  corrected accelerometer readings (m/s^2)
    dt     : [B, N, 1]  per-sample time deltas (s)
    sigma2_g : [B, N, 3]  per-sample gyroscope noise variance
    sigma2_a : [B, N, 3]  per-sample accelerometer noise variance
    mask   : [B, N]     boolean, True = valid sample, False = padded

    Output (dict)
    ------
    delta_R  : [B, 3, 3]  preintegrated rotation matrix
    delta_v  : [B, 3]     preintegrated velocity change
    delta_p  : [B, 3]     preintegrated position change
    Sigma    : [B, 9, 9]  preintegration covariance  (rot, vel, pos)
    """

    def forward(self, gyro, accel, dt, sigma2_g, sigma2_a, mask):
        # PyPose SO3 ops are sensitive to mixed dtypes under AMP.
        # Run preintegration math in fp32 for stability and compatibility.
        gyro = torch.nan_to_num(gyro.float(), nan=0.0, posinf=0.0, neginf=0.0)
        accel = torch.nan_to_num(accel.float(), nan=0.0, posinf=0.0, neginf=0.0)
        dt = torch.nan_to_num(dt.float(), nan=0.0, posinf=0.0, neginf=0.0)
        sigma2_g = torch.nan_to_num(sigma2_g.float(), nan=1e-4, posinf=1.0, neginf=1e-6)
        sigma2_a = torch.nan_to_num(sigma2_a.float(), nan=1e-4, posinf=1.0, neginf=1e-6)
        mask = mask.bool()

        # Keep integration inputs in a physically plausible range to prevent
        # Exp/log singularities when mixed-precision activations spike.
        gyro = gyro.clamp(min=-200.0, max=200.0)
        accel = accel.clamp(min=-500.0, max=500.0)
        dt = dt.clamp(min=0.0, max=0.1)
        sigma2_g = sigma2_g.clamp(min=1e-6, max=1e2)
        sigma2_a = sigma2_a.clamp(min=1e-6, max=1e2)

        B, N, _ = gyro.shape
        device = gyro.device
        dtype = gyro.dtype

        # Initialise preintegrated quantities
        # delta_R as SO3 LieTensor (quaternion internally)
        delta_R = pp.identity_SO3(B, device=device, dtype=dtype)   # [B]
        delta_v = torch.zeros(B, 3, device=device, dtype=dtype)
        delta_p = torch.zeros(B, 3, device=device, dtype=dtype)

        # Covariance [B, 9, 9]: ordering (phi, vel, pos)
        Sigma = torch.zeros(B, 9, 9, device=device, dtype=dtype)

        for i in range(N):
            dt_i = dt[:, i, :]                  # [B, 1]
            gyro_i = gyro[:, i, :]              # [B, 3]
            accel_i = accel[:, i, :]            # [B, 3]
            valid = mask[:, i].unsqueeze(-1)    # [B, 1]
            # Use where() rather than multiply-by-zero, so invalid samples with
            # inf values cannot create inf*0 -> nan.
            dt_i = torch.where(valid, dt_i, torch.zeros_like(dt_i))
            gyro_i = torch.where(valid, gyro_i, torch.zeros_like(gyro_i))
            accel_i = torch.where(valid, accel_i, torch.zeros_like(accel_i))

            # --- rotation increment ---
            omega = gyro_i * dt_i               # [B, 3] angle-axis
            # Convert to SO3 via exponential map (pypose handles small-angle)
            dR_i = pp.so3(omega).Exp()          # [B] SO3

            # Current rotation matrix for velocity/position update
            R_mat = delta_R.matrix()            # [B, 3, 3]

            # Rotated acceleration
            Ra = torch.bmm(R_mat, accel_i.unsqueeze(-1)).squeeze(-1)  # [B, 3]

            # --- velocity and position increments ---
            delta_p = delta_p + delta_v * dt_i + 0.5 * Ra * dt_i.pow(2)
            delta_v = delta_v + Ra * dt_i

            # --- compose rotation ---
            delta_R = delta_R * dR_i

            # --- first-order covariance propagation ---
            # Build the per-step noise Jacobian and transition
            dt_s = dt_i.squeeze(-1)             # [B]
            I3 = torch.eye(3, device=device, dtype=dtype).unsqueeze(0).expand(B, -1, -1)

            # Skew-symmetric of Ra for Jacobian
            Ra_skew = _skew_symmetric(Ra)       # [B, 3, 3]

            # Transition matrix A_k  [B, 9, 9]
            A_k = torch.zeros(B, 9, 9, device=device, dtype=dtype)
            dR_mat = dR_i.matrix()              # [B, 3, 3]
            dR_mat_T = dR_mat.transpose(-1, -2)

            A_k[:, 0:3, 0:3] = dR_mat_T
            A_k[:, 3:6, 0:3] = -R_mat @ _skew_symmetric(accel_i) * dt_s.unsqueeze(-1).unsqueeze(-1)
            A_k[:, 3:6, 3:6] = I3
            A_k[:, 6:9, 0:3] = -0.5 * R_mat @ _skew_symmetric(accel_i) * (dt_s.pow(2)).unsqueeze(-1).unsqueeze(-1)
            A_k[:, 6:9, 3:6] = I3 * dt_s.unsqueeze(-1).unsqueeze(-1)
            A_k[:, 6:9, 6:9] = I3

            # Noise Jacobian B_k  [B, 9, 6]
            B_k = torch.zeros(B, 9, 6, device=device, dtype=dtype)
            B_k[:, 0:3, 0:3] = I3 * dt_s.unsqueeze(-1).unsqueeze(-1)
            B_k[:, 3:6, 3:6] = R_mat * dt_s.unsqueeze(-1).unsqueeze(-1)
            B_k[:, 6:9, 3:6] = 0.5 * R_mat * (dt_s.pow(2)).unsqueeze(-1).unsqueeze(-1)

            # Per-sample noise covariance Q_k  [B, 6, 6]
            sg = sigma2_g[:, i, :]              # [B, 3]
            sa = sigma2_a[:, i, :]              # [B, 3]
            Q_diag = torch.cat([sg, sa], dim=-1)  # [B, 6]
            Q_k = torch.diag_embed(Q_diag)     # [B, 6, 6]

            # Propagate:  Σ_{k+1} = A_k Σ_k A_k^T + B_k Q_k B_k^T
            Sigma_prop = A_k @ Sigma @ A_k.transpose(-1, -2) + B_k @ Q_k @ B_k.transpose(-1, -2)

            # For padded samples (valid=False) keep Sigma unchanged instead of
            # zeroing it — zeroing destroys all covariance accumulated so far.
            valid_mat = valid.unsqueeze(-1).expand_as(Sigma)  # [B, 9, 9]
            Sigma = torch.where(valid_mat, Sigma_prop, Sigma)

        # Convert final rotation to matrix form
        delta_R_mat = delta_R.matrix()          # [B, 3, 3]
        delta_R_mat = torch.nan_to_num(delta_R_mat, nan=0.0, posinf=0.0, neginf=0.0)
        bad_R = ~torch.isfinite(delta_R_mat).all(dim=(-2, -1))
        if bad_R.any():
            n_bad = int(bad_R.sum().item())
            eye = torch.eye(3, device=device, dtype=dtype).unsqueeze(0).expand(n_bad, -1, -1)
            delta_R_mat = delta_R_mat.clone()
            delta_R_mat[bad_R] = eye

        delta_v = torch.nan_to_num(delta_v, nan=0.0, posinf=0.0, neginf=0.0)
        delta_p = torch.nan_to_num(delta_p, nan=0.0, posinf=0.0, neginf=0.0)

        # Symmetrize covariance for numerical stability
        Sigma = torch.nan_to_num(Sigma, nan=0.0, posinf=1e6, neginf=-1e6)
        Sigma = 0.5 * (Sigma + Sigma.transpose(-1, -2))

        return {
            "delta_R": delta_R_mat,
            "delta_v": delta_v,
            "delta_p": delta_p,
            "Sigma": Sigma,
        }


def _skew_symmetric(v: torch.Tensor) -> torch.Tensor:
    """Batch skew-symmetric matrix from [B, 3] vectors → [B, 3, 3]."""
    B = v.shape[0]
    zero = torch.zeros(B, 1, device=v.device, dtype=v.dtype)
    x, y, z = v[:, 0:1], v[:, 1:2], v[:, 2:3]
    row0 = torch.cat([zero, -z, y], dim=-1)
    row1 = torch.cat([z, zero, -x], dim=-1)
    row2 = torch.cat([-y, x, zero], dim=-1)
    return torch.stack([row0, row1, row2], dim=1)  # [B, 3, 3]


def _project_so3(R: torch.Tensor) -> torch.Tensor:
    """Project a batch of matrices onto SO(3) via SVD (nearest rotation matrix).

    Handles float32 numerical drift that accumulates in matrix products.
    R: [..., 3, 3] → [..., 3, 3]
    """
    U, _, Vt = torch.linalg.svd(R)
    R_proj = U @ Vt
    # Fix determinant sign: det should be +1, not -1 (reflection)
    det = torch.linalg.det(R_proj)
    D = torch.ones(*R.shape[:-2], 3, device=R.device, dtype=R.dtype)
    D[..., 2] = det.sign()
    R_proj = (U * D.unsqueeze(-2)) @ Vt
    return R_proj


def so3_log_map(R: torch.Tensor) -> torch.Tensor:
    """Logarithmic map SO(3) → so(3).  R: [B, 3, 3] → [B, 3].

    Projects R onto SO(3) first to tolerate float32 drift in matrix products,
    then uses PyPose for the actual log map.  Everything runs in float32 with
    AMP disabled to avoid half-precision CUDA kernel errors.
    """
    if R.device.type == "cuda":
        amp_off_ctx = torch.autocast(device_type="cuda", enabled=False)
    else:
        amp_off_ctx = nullcontext()

    with amp_off_ctx:
        R = R.float()
        R = _project_so3(R)
        q = pp.mat2SO3(R)       # convert rotation matrix to SO3 quaternion
        out = q.Log().tensor()  # [B, 3] tangent vector

    return out
