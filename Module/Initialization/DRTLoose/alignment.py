"""
Linear alignment for the DRT-loose initializer.

Solves for [v_0, ..., v_{N-1}, s, g] (body-frame velocities at each keyframe,
scale, and gravity vector) from the preintegrated IMU deltas and the up-to-scale
camera translations.

Two-step process:
1. Linear seed: assemble A x = b and solve unconstrained via least-squares.
2. Constrained refine: normalize g to gravity_norm after the linear solve.

Reference:
  Dong-Si & Mourikis (2012); also DRT paper §5.6.
"""

from __future__ import annotations

import torch
from dataclasses import dataclass

from .preintegration import PreintResult


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class AlignmentResult:
    velocities: torch.Tensor    # (N, 3) body-frame velocities at each keyframe
    scale: float                # global scale for the up-to-scale translations
    gravity_world: torch.Tensor # (3,) gravity vector in world frame (after normalizing)
    success: bool
    reason: str | None          # None if success


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def normalize_gravity(g_vec: torch.Tensor, g_norm: float = 9.81007) -> torch.Tensor:
    """Normalize g_vec to have magnitude g_norm.

    Parameters
    ----------
    g_vec  : (3,) raw gravity estimate (unconstrained norm).
    g_norm : desired norm [m/s²]; defaults to standard gravity.

    Returns
    -------
    (3,) tensor with the same direction as g_vec and norm equal to g_norm.
    """
    return (g_vec / g_vec.norm().clamp_min(1e-12)) * g_norm


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def linear_alignment(
    preint_results: list[PreintResult],         # N-1 preintegration results
    translations_up_to_scale: torch.Tensor,     # (N, 3) from LiGT (t_0 = 0)
    rotations: list[torch.Tensor],              # N list of (3,3) camera rotations
    gravity_norm: float = 9.81007,
    t_BC: torch.Tensor | None = None,           # (3,) IMU→camera extrinsic translation
) -> AlignmentResult:
    """
    Assemble and solve the linear system from design doc §5.6.

    Unknowns  x = [v_0, v_1, ..., v_{N-1}, s, g_x, g_y, g_z]
              shape  (3N + 4,)

    For each consecutive keyframe pair (i, j=i+1) the two vector constraints are:

      Position constraint (3 eqs):
        ΔP_ij  =  R_i^T · (s · (t_j - t_i))  -  v_i · dt  -  0.5 · R_i^T · g · dt²

      Velocity constraint (3 eqs):
        ΔV_ij  =  R_i^T · v_j  -  v_i  -  R_i^T · g · dt

    RHS (b vector) includes IMU→camera extrinsic translation correction when provided:
        ΔP_corrected = ΔP + R_i^T @ R_j @ t_BC - t_BC

    Rearranged to place all unknowns on the LHS:

      Position (row-block 6k : 6k+3):
        [-I · dt,  0, ..., 0,  R_i^T · (t_j - t_i),  -0.5 · R_i^T · dt²] · x
            = ΔP_ij

        (block for v_i is -I * dt; block for s is R_i^T @ Δt_ij; block for g is -0.5 R_i^T dt²)

      Velocity (row-block 6k+3 : 6k+6):
        [-I,  R_i^T, 0, ..., 0,  0,  -R_i^T · dt] · x
            = ΔV_ij

        (block for v_i is -I; block for v_j is R_i^T; block for g is -R_i^T * dt)

    Parameters
    ----------
    preint_results            : list of N-1 PreintResult objects.
    translations_up_to_scale  : (N, 3) float64 tensor; t_0 = [0, 0, 0].
    rotations                 : list of N (3,3) float64 rotation matrices.
    gravity_norm              : magnitude of gravity [m/s²].
    t_BC                      : (3,) IMU→camera extrinsic translation; None to skip correction.

    Returns
    -------
    AlignmentResult with velocities, scale, gravity_world, success, reason.
    """
    N = len(rotations)
    if len(preint_results) != N - 1:
        return AlignmentResult(
            velocities=torch.zeros(N, 3, dtype=torch.float64),
            scale=1.0,
            gravity_world=torch.zeros(3, dtype=torch.float64),
            success=False,
            reason=f"Expected {N-1} preint_results for {N} keyframes, got {len(preint_results)}",
        )
    if N < 2:
        return AlignmentResult(
            velocities=torch.zeros(N, 3, dtype=torch.float64),
            scale=1.0,
            gravity_world=torch.zeros(3, dtype=torch.float64),
            success=False,
            reason="Need at least 2 keyframes for alignment",
        )

    dtype  = torch.float64
    device = translations_up_to_scale.device

    tpts = translations_up_to_scale.to(dtype=dtype, device=device)   # (N, 3)
    rots = [R.to(dtype=dtype, device=device) for R in rotations]     # list of (3,3)

    n_pairs = N - 1
    n_rows  = 6 * n_pairs          # 3 (pos) + 3 (vel) per pair
    n_cols  = 3 * N + 4            # 3N velocities + 1 scale + 3 gravity

    A = torch.zeros(n_rows, n_cols, dtype=dtype, device=device)
    b = torch.zeros(n_rows,         dtype=dtype, device=device)

    I3 = torch.eye(3, dtype=dtype, device=device)

    for k, pr in enumerate(preint_results):
        i = k          # "from" keyframe index
        j = k + 1      # "to"   keyframe index

        R_i = rots[i]                                           # (3,3)
        dt  = pr.sum_dt                                         # scalar [s]
        dt2 = dt * dt

        dP = pr.dP.to(dtype=dtype, device=device)               # (3,) ΔP in body frame
        dV = pr.dV.to(dtype=dtype, device=device)               # (3,) ΔV in body frame

        # Apply IMU→camera extrinsic translation correction (C++ reference does this)
        # dP_corrected = dP + R_i^T @ R_j @ t_BC - t_BC
        if t_BC is not None:
            R_j = rots[j]
            t_BC_body = t_BC.to(dtype=dtype, device=device)
            # R_i^T @ R_j @ t_BC - t_BC  (transformation from camera to body frame)
            dP = dP + (R_i.T @ R_j @ t_BC_body) - t_BC_body

        # Camera translation difference (up-to-scale, in camera/world frame)
        delta_t_cam = tpts[j] - tpts[i]                        # (3,)
        # C++ divides by 100 for numerical normalization
        R_i_T_delta_t = (R_i.T @ delta_t_cam) / 100.0         # (3,) rotate into body frame at i

        row_p = 6 * k       # position constraint rows [row_p : row_p+3]
        row_v = 6 * k + 3   # velocity constraint rows [row_v : row_v+3]

        col_vi = 3 * i      # column block for v_i
        col_vj = 3 * j      # column block for v_j
        col_s  = 3 * N      # column for scale
        col_g  = 3 * N + 1  # column block for gravity (3 cols: col_g, col_g+1, col_g+2)

        # ---------- Position constraint ----------------------------------
        # ΔP = R_i^T · (s · (t_j - t_i)) - v_i · dt - 0.5 · R_i^T · g · dt²
        # Rearranged:
        #   -v_i · dt + s · R_i^T Δt - 0.5 · R_i^T · dt² · g = ΔP
        # C++ reference multiplies gravity block by G.norm() for numerical stability
        A[row_p:row_p+3, col_vi:col_vi+3] = -I3 * dt
        A[row_p:row_p+3, col_s]           = R_i_T_delta_t      # (3,)
        A[row_p:row_p+3, col_g:col_g+3]   = -0.5 * R_i.T * dt2 * gravity_norm

        b[row_p:row_p+3] = dP

        # ---------- Velocity constraint ----------------------------------
        # ΔV = R_i^T · v_j - v_i - R_i^T · g · dt
        # Rearranged:
        #   -v_i + R_i^T · v_j - R_i^T · dt · g = ΔV
        A[row_v:row_v+3, col_vi:col_vi+3] = -I3
        A[row_v:row_v+3, col_vj:col_vj+3] = R_i.T
        A[row_v:row_v+3, col_g:col_g+3]   = -R_i.T * dt * gravity_norm

        b[row_v:row_v+3] = dV

    # ---- Matrix scaling (C++ reference does this for numerical stability) ----
    # Extract gravity block diagonal and scale by 1/mean(diag) to improve conditioning
    gravity_block = A[-(3 + 1):, -(3 + 1):]  # bottom-right 4x4 block (scale + gravity)
    gravity_g_block = gravity_block[1:, 1:]  # gravity 3x3 part
    mean_diag = (gravity_g_block[0, 0] + gravity_g_block[1, 1] + gravity_g_block[2, 2]) / 3.0
    if mean_diag > 1e-12:
        scale_factor = 1.0 / mean_diag
        A = A * scale_factor
        b = b * scale_factor

    # ---- Solve via least-squares ----------------------------------------
    try:
        result = torch.linalg.lstsq(A, b, rcond=1e-10)
        x = result.solution
    except Exception as exc:
        return AlignmentResult(
            velocities=torch.zeros(N, 3, dtype=dtype, device=device),
            scale=1.0,
            gravity_world=torch.zeros(3, dtype=dtype, device=device),
            success=False,
            reason=f"lstsq failed: {exc}",
        )

    if x is None or x.shape[0] != n_cols:
        return AlignmentResult(
            velocities=torch.zeros(N, 3, dtype=dtype, device=device),
            scale=1.0,
            gravity_world=torch.zeros(3, dtype=dtype, device=device),
            success=False,
            reason="lstsq returned unexpected solution shape",
        )

    # ---- Extract unknowns ------------------------------------------------
    velocities_flat = x[:3 * N]                                # (3N,)
    velocities      = velocities_flat.reshape(N, 3)            # (N, 3)
    scale_raw       = x[3 * N].item()                          # scalar
    # Compensate for /100 normalization in translation column (matching C++ reference)
    scale_raw       = scale_raw / 100.0                        # undo the /100 normalization
    g_unconstrained = x[3 * N + 1 : 3 * N + 4]               # (3,)

    # ---- Normalize gravity -----------------------------------------------
    gravity_world = normalize_gravity(g_unconstrained, gravity_norm)

    # ---- Check success ---------------------------------------------------
    g_norm_achieved = gravity_world.norm().item()
    reason = None

    if scale_raw <= 0.0:
        reason = f"Recovered scale is non-positive: {scale_raw:.6g}"
        success = False
    elif abs(g_norm_achieved - gravity_norm) > 0.5:
        reason = (
            f"Gravity norm ({g_norm_achieved:.4f}) deviates from expected "
            f"({gravity_norm:.4f}) before normalization; g_raw={g_unconstrained.norm().item():.4f}"
        )
        success = False
    else:
        success = True

    return AlignmentResult(
        velocities=velocities,
        scale=float(scale_raw),
        gravity_world=gravity_world,
        success=success,
        reason=reason,
    )
