"""
Linear alignment for the DRT-loose initializer.

Solves for [v_0, ..., v_{N-1}, s, g] (body-frame velocities at each keyframe,
scale, and gravity vector) from the preintegrated IMU deltas and the up-to-scale
camera translations.

Velocities are parametrised in the LOCAL (camera) frame of each keyframe,
matching the C++ reference (drtLooselyCoupled.cpp).  After the solve they
are rotated to world frame with  v_world_i = R_i @ v_local_i.

Two-step process:
1. Build tall matrix A_tall / b_tall (overdetermined system).
2. Convert to normal equations M = A_tall^T A_tall, m = -2 A_tall^T b_tall.
3. Apply gravity-norm constraint via gravityRefine (Lagrangian polynomial solve),
   matching the C++ gravityRefine in drtVioInit.cpp.
4. Rotate velocities local→world; apply rot0 (identity when cam_rotations[0]=I).

Reference:
  Dong-Si & Mourikis (2012); DRT paper §5.6; drtVioInit.cpp::gravityRefine.
"""

from __future__ import annotations

import numpy as np
import torch
from dataclasses import dataclass

from .preintegration import PreintResult


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class AlignmentResult:
    velocities: torch.Tensor    # (N, 3) world-frame velocities at each keyframe
    scale: float                # global scale for the up-to-scale translations
    gravity_world: torch.Tensor # (3,) gravity vector in world frame
    success: bool
    reason: str | None          # None if success


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def normalize_gravity(g_vec: torch.Tensor, g_norm: float = 9.81007) -> torch.Tensor:
    """Normalize g_vec to have magnitude g_norm."""
    return (g_vec / g_vec.norm().clamp_min(1e-12)) * g_norm


def _gravity_refine(
    M: torch.Tensor,
    m: torch.Tensor,
    Q: float,
    gravity_mag: float = 1.0,
) -> torch.Tensor | None:
    """Constrained solve: min x^T M x + m^T x + Q  s.t.  ||x[-3:]|| = gravity_mag.

    Direct port of gravityRefine() from drtVioInit.cpp.

    Parameters
    ----------
    M           : (n, n) positive-semi-definite normal matrix (already scaled).
    m           : (n,) linear coefficient = -2 * A^T b.
    Q           : scalar = b^T b.
    gravity_mag : norm of the gravity block; C++ calls with gravity_mag=1 (unit
                  sphere) and then multiplies result by G.norm() afterward.

    Returns
    -------
    (n,) float64 solution, or None if the constrained solve failed.
    """
    n = M.shape[0]
    q = n - 3   # size of non-gravity block (velocities + scale)

    M_np = M.double().numpy()
    m_np = m.double().numpy()

    # Schur complement decomposition (C++ notation matches)
    A_upper = 2.0 * M_np[:q, :q]           # (q, q)
    Bt      = 2.0 * M_np[q:, :q]           # (3, q)
    D       = 2.0 * M_np[q:, q:]           # (3, 3)

    try:
        A_inv = np.linalg.inv(A_upper)
    except np.linalg.LinAlgError:
        return None

    BtAi = Bt @ A_inv                       # (3, q)
    S    = D - BtAi @ Bt.T                 # (3, 3)  Schur complement

    det_S = np.linalg.det(S)
    try:
        inv_S = np.linalg.inv(S)
    except np.linalg.LinAlgError:
        return None
    Sa = det_S * inv_S                      # adj(S) = det(S) * S^{-1}
    U  = np.trace(S) * np.eye(3) - S       # cofactor-related matrix

    v1 = BtAi @ m_np[:q]                   # (3,)
    m2 = m_np[q:]                          # (3,)

    def _vXm(X: np.ndarray) -> float:
        Xm2 = X @ m2
        return float(v1 @ (X @ v1) - 2 * v1 @ Xm2 + m2 @ Xm2)

    c4 = 16.0 * _vXm(np.eye(3))
    c3 = 16.0 * _vXm(U)
    c2 = 4.0  * _vXm(2.0 * Sa + U @ U)
    c1 = 2.0  * _vXm(Sa @ U + U @ Sa)
    c0 =        _vXm(Sa @ Sa)

    # Characteristic polynomial coefficients of S (matches C++ t1/t2/t3)
    s00, s01, s02 = S[0, 0], S[0, 1], S[0, 2]
    s11, s12      = S[1, 1], S[1, 2]
    s22           = S[2, 2]
    t1 = s00 + s11 + s22
    t2 = s00*s11 + s00*s22 + s11*s22 - s01**2 - s02**2 - s12**2
    t3 = (s00*s11*s22 + 2*s01*s02*s12
          - s00*s12**2 - s11*s02**2 - s22*s01**2)

    # Degree-6 secular polynomial in Lagrange multiplier λ
    coeffs = np.array([
        64.,
        64. * t1,
        16. * (t1**2 + 2*t2),
        16. * (t1*t2 + t3),
        4.  * (t2**2 + 2*t1*t3),
        4.  * t3 * t2,
        t3**2,
    ])
    G2i = 1.0 / (gravity_mag ** 2)
    coeffs[2] -= c4 * G2i
    coeffs[3] -= c3 * G2i
    coeffs[4] -= c2 * G2i
    coeffs[5] -= c1 * G2i
    coeffs[6] -= c0 * G2i

    # Find real roots of the secular equation
    roots = np.roots(coeffs)
    real_mask  = np.abs(np.imag(roots)) < 1e-4 * (np.abs(roots) + 1.0)
    real_roots = np.real(roots[real_mask])
    if len(real_roots) == 0:
        real_roots = np.real(roots)   # fall back to all if none pass threshold

    # W: identity block on gravity rows/cols
    W_np = np.zeros((n, n))
    W_np[q:, q:] = np.eye(3)

    solution = None
    min_cost  = float('inf')

    for lam in real_roots:
        lam = float(lam)
        mat = 2.0 * M_np + 2.0 * lam * W_np
        try:
            x_ = -np.linalg.solve(mat, m_np)
        except np.linalg.LinAlgError:
            continue
        cost = float(x_ @ M_np @ x_ + m_np @ x_ + Q)
        if cost < min_cost:
            solution = x_.copy()
            min_cost = cost

    if solution is None:
        return None

    # Validate gravity norm constraint (C++ checks same condition)
    g_sq = float(solution[q:] @ solution[q:])
    if g_sq < 0 or abs(np.sqrt(max(g_sq, 0.0)) - gravity_mag) / gravity_mag > 1e-3:
        return None

    return torch.tensor(solution, dtype=torch.float64)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def linear_alignment(
    preint_results: list[PreintResult],         # N-1 preintegration results
    translations_up_to_scale: torch.Tensor,     # (N, 3) translations (t_0 = 0)
    rotations: list[torch.Tensor],              # N list of (3,3) camera rotations
    gravity_norm: float = 9.81007,
    t_BC: torch.Tensor | None = None,           # (3,) IMU→camera extrinsic translation
    metric: bool = False,                       # True → translations already metric
) -> AlignmentResult:
    """
    Assemble and solve the linear system matching drtLooselyCoupled.cpp::linearAlignment.

    Two modes:
    - metric=False (monocular): x = [v_0, ..., v_{N-1}, s, g]  shape (3N+4)
      Scale s couples translations (up-to-scale) to IMU deltas.
    - metric=True  (stereo):    x = [v_0, ..., v_{N-1}, g]     shape (3N+3)
      Translations are known (metric from stereo depth).  No scale unknown;
      the known metric position deltas move to the RHS.

    Velocities are in the LOCAL frame of each keyframe (matching C++ convention):
        position constraint (3 eqs per pair):
            -v_i * dt  +  scale_coeff * s  +  0.5 R_i^T g dt²  =  ΔP
        velocity constraint (3 eqs per pair):
            −v_i  +  R_i^T R_j v_j  +  R_i^T g dt  =  ΔV

    With metric=True, scale_coeff=0 and R_i^T(t_j−t_i) moves to RHS.

    After solving, velocities are rotated to world frame: v_world_i = R_i @ v_local_i.
    Then rot0 = R_0^T is applied (identity when cam_rotations[0]=I).

    Returns
    -------
    AlignmentResult with velocities (world frame), scale, gravity_world, success, reason.
    """
    N = len(rotations)
    mode_str = "METRIC (stereo)" if metric else "UP-TO-SCALE (monocular)"
    print(f"\n=== DRT linear_alignment [{mode_str}]: {N} frames ===")
    
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
    n_rows  = 6 * n_pairs
    # Metric: no scale unknown.  Monocular: 3N velocities + 1 scale + 3 gravity
    has_scale = not metric
    n_cols  = 3 * N + (4 if has_scale else 3)

    A_tall = torch.zeros(n_rows, n_cols, dtype=dtype, device=device)
    b_tall = torch.zeros(n_rows,         dtype=dtype, device=device)

    I3 = torch.eye(3, dtype=dtype, device=device)

    for k, pr in enumerate(preint_results):
        i = k
        j = k + 1

        R_i = rots[i]                                           # (3,3)
        R_j = rots[j]                                           # (3,3)
        dt  = pr.sum_dt                                         # scalar [s]
        dt2 = dt * dt

        dP = pr.dP.to(dtype=dtype, device=device)               # (3,) ΔP body frame
        dV = pr.dV.to(dtype=dtype, device=device)               # (3,) ΔV body frame

        # Camera translation difference (world/camera frame)
        delta_t_cam = tpts[j] - tpts[i]                         # (3,)

        # Extrinsic translation correction (matches C++ tmp_b position block)
        dP_corr = dP.clone()
        if t_BC is not None:
            t_BC_d = t_BC.to(dtype=dtype, device=device)
            dP_corr = dP_corr + (R_i.T @ R_j @ t_BC_d) - t_BC_d

        # Position RHS: for metric, subtract known R_i^T @ delta_t
        # For monocular, delta_t is up-to-scale and handled via scale unknown
        if metric:
            # Metric: -v_i*dt + 0.5*R_i^T*g*dt^2 = dP - R_i^T*delta_t
            b_pos = dP_corr - (R_i.T @ delta_t_cam)
            scale_coeff = None
        else:
            # Monocular: -v_i*dt + R_i^T*delta_t/100*s + 0.5*R_i^T*g*dt^2 = dP
            b_pos = dP_corr
            scale_coeff = (R_i.T @ delta_t_cam) / 100.0

        if k < 2:
            print(f"\nSegment {i}→{j}:")
            print(f"  delta_t = {delta_t_cam.tolist()}")
            if metric:
                print(f"  R_i^T @ delta_t (RHS subtract) = {(R_i.T @ delta_t_cam).tolist()}")
            else:
                print(f"  scale_coeff (R_i^T @ delta_t/100) = {scale_coeff.tolist()}")

        row_p = 6 * k       # position rows [row_p : row_p+3]
        row_v = 6 * k + 3   # velocity rows [row_v : row_v+3]

        col_vi = 3 * i      # column block for v_i (local frame)
        col_vj = 3 * j      # column block for v_j (local frame)
        col_g  = 3 * N + (1 if has_scale else 0)   # gravity block (after scale if present)

        # ---- Position constraint -----------------------------------------
        A_tall[row_p:row_p+3, col_vi:col_vi+3] = -I3 * dt
        if has_scale:
            col_s = 3 * N
            A_tall[row_p:row_p+3, col_s] = scale_coeff
        A_tall[row_p:row_p+3, col_g:col_g+3] = 0.5 * R_i.T * dt2 * gravity_norm
        b_tall[row_p:row_p+3] = b_pos

        # ---- Velocity constraint -----------------------------------------
        A_tall[row_v:row_v+3, col_vi:col_vi+3] = -I3
        A_tall[row_v:row_v+3, col_vj:col_vj+3] = R_i.T @ R_j
        A_tall[row_v:row_v+3, col_g:col_g+3] = R_i.T * dt * gravity_norm
        b_tall[row_v:row_v+3] = dV

    # ---- Normal equations -----------------------------------------------
    M        = A_tall.T @ A_tall
    m_vec    = -2.0 * (A_tall.T @ b_tall)
    Q_scalar = float(b_tall @ b_tall)

    # ---- Mean-diag normalization (gravity-gravity block) ----------------
    M_grav    = M[-3:, -3:]
    mean_diag = (M_grav[0, 0] + M_grav[1, 1] + M_grav[2, 2]) / 3.0

    print(f"\nBefore solve:")
    print(f"  M_grav diagonal = [{M_grav[0,0].item():.6e}, {M_grav[1,1].item():.6e}, {M_grav[2,2].item():.6e}]")
    print(f"  mean_diag = {mean_diag.item():.6e}")

    if mean_diag.abs() > 1e-12:
        sf       = 1.0 / mean_diag.item()
        print(f"  scale factor = {sf:.6e}")
        M        = M * sf
        m_vec    = m_vec * sf
        Q_scalar = Q_scalar * sf
    else:
        print(f"  scale factor = 1.0 (mean_diag too small)")

    # ---- Solve -----------------------------------------------------------
    # Metric case: system is well-determined; simple lstsq is sufficient.
    # Monocular case: use gravityRefine with unit-sphere constraint.
    if metric:
        try:
            result = torch.linalg.lstsq(M, -m_vec / 2.0, rcond=1e-10)
            x = result.solution
            constrained_ok = True
        except Exception as exc:
            return AlignmentResult(
                velocities=torch.zeros(N, 3, dtype=dtype, device=device),
                scale=1.0,
                gravity_world=torch.zeros(3, dtype=dtype, device=device),
                success=False,
                reason=f"metric lstsq failed: {exc}",
            )
    else:
        constrained_ok = False
        x = _gravity_refine(M.cpu(), m_vec.cpu(), Q_scalar, gravity_mag=1.0)
        if x is not None:
            constrained_ok = True

        if not constrained_ok:
            try:
                result = torch.linalg.lstsq(M, -m_vec / 2.0, rcond=1e-10)
                x = result.solution
            except Exception as exc:
                return AlignmentResult(
                    velocities=torch.zeros(N, 3, dtype=dtype, device=device),
                    scale=1.0,
                    gravity_world=torch.zeros(3, dtype=dtype, device=device),
                    success=False,
                    reason=f"lstsq fallback failed: {exc}",
                )

    if x is None or x.shape[0] != n_cols:
        return AlignmentResult(
            velocities=torch.zeros(N, 3, dtype=dtype, device=device),
            scale=1.0,
            gravity_world=torch.zeros(3, dtype=dtype, device=device),
            success=False,
            reason="solve returned unexpected shape",
        )

    x = x.to(dtype=dtype, device=device)

    # ---- Extract unknowns -----------------------------------------------
    vel_local_flat = x[:3 * N]                          # (3N,)  local-frame velocities
    vel_local      = vel_local_flat.reshape(N, 3)       # (N, 3)

    if has_scale:
        scale_raw = x[3 * N].item() / 100.0             # undo /100 normalization
        g_raw     = x[3 * N + 1 : 3 * N + 4]           # (3,) unit gravity
        print(f"\nAfter gravityRefine:")
        print(f"  x[3*N] (raw scale) = {x[3 * N].item():.6e}")
        print(f"  scale (scale/100) = {scale_raw:.6e}")
    else:
        scale_raw = 1.0                                   # metric: known scale
        g_raw     = x[3 * N : 3 * N + 3]                 # (3,) gravity from lstsq
        print(f"\nAfter lstsq solve:")
        print(f"  |g_raw| = {g_raw.norm().item():.4f}")
        # Normalize to expected gravity magnitude
        g_norm_raw = g_raw.norm().item()
        if g_norm_raw > 1e-6:
            g_raw = g_raw / g_norm_raw * gravity_norm
        else:
            g_raw = torch.tensor([0.0, 0.0, -gravity_norm], dtype=dtype, device=device)

    # ---- Rotate velocities local → world --------------------------------
    # C++: velocity[i] = rotation[i] * x.segment<3>(i*3)
    velocities_world = torch.stack([rots[i] @ vel_local[i] for i in range(N)])

    # ---- Apply rot0 = R_0^T (identity when cam_rotations[0]=I) ---------
    rot0 = rots[0].T
    velocities_world = torch.stack([rot0 @ v for v in velocities_world])

    # ---- Gravity vector -------------------------------------------------
    g_world_raw   = rot0 @ g_raw
    gravity_world = normalize_gravity(g_world_raw, gravity_norm)

    # ---- Success checks -------------------------------------------------
    reason  = None
    success = True

    if scale_raw <= 0.0:
        reason  = f"Recovered scale is non-positive: {scale_raw:.6g}"
        success = False

    return AlignmentResult(
        velocities=velocities_world,
        scale=float(scale_raw),
        gravity_world=gravity_world,
        success=success,
        reason=reason,
    )


def check_ltl_conditioning(LTL: torch.Tensor, max_cond: float = 1e8) -> bool:
    """Return True if LTL is well-conditioned (cond(LTL) <= max_cond)."""
    cond = torch.linalg.cond(LTL).item()
    return cond <= max_cond
