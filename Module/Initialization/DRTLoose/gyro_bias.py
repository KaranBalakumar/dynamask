"""
Gyro-bias estimation — bearing-vector / smallest-eigenvalue formulation.

Direct port of BiasSolverCostFunctor (optimization.hpp) + GetSmallestEV
(opengvMethod.hpp) from the DRT-VIO-init C++ reference.

For each consecutive keyframe pair i→j we collect bearing vectors
(f_i, f_j) from the feature tracker (normalised image coordinates lifted
to homogeneous 3-D rays). The cost per pair is the smallest eigenvalue
of the 3×3 symmetric matrix M built from those bearing vectors under the
bias-corrected IMU rotation.  Minimising that eigenvalue drives the IMU
rotation to be consistent with the epipolar geometry of the feature
correspondences — no explicit essential-matrix decomposition required.

Optimisation: scipy L-BFGS-B; gradients via torch.autograd.
"""

from __future__ import annotations

import numpy as np
import torch
from scipy.optimize import minimize

from .preintegration import PreintResult


# ---------------------------------------------------------------------------
# Cayley / rotation helpers (matches geometry.hpp)
# ---------------------------------------------------------------------------

def _cayley_to_rot_reduced(c: torch.Tensor) -> torch.Tensor:
    """Unnormalised Cayley → (3,3) rotation.  Matches Cayley2RotReduced."""
    c0, c1, c2 = c[0], c[1], c[2]
    R = torch.stack([
        torch.stack([1 + c0*c0 - c1*c1 - c2*c2,  2*(c0*c1 - c2),              2*(c0*c2 + c1)]),
        torch.stack([2*(c0*c1 + c2),               1 - c0*c0 + c1*c1 - c2*c2,  2*(c1*c2 - c0)]),
        torch.stack([2*(c0*c2 - c1),               2*(c1*c2 + c0),              1 - c0*c0 - c1*c1 + c2*c2]),
    ])
    return R


def _aa_to_cayley(aa: torch.Tensor) -> torch.Tensor:
    """Axis-angle (3,) → Cayley params (3,), differentiable everywhere.

    Converts angle-axis representation to Cayley parameters:
        cayley_i = (tan(θ/2) / θ) * aa_i   where θ = ||aa||
    For θ→0: cayley → aa/2  (Taylor expansion).
    
    This matches the C++ path: angle-axis → quaternion → Cayley (v/w form).
    Both are mathematically equivalent for the purposes of the optimization.
    """
    theta = (aa.dot(aa) + 1e-30).sqrt()   # always > 0, grad-safe
    half  = theta * 0.5
    # tan(half)/theta = sin(half)/(theta*cos(half))
    scale = torch.sin(half) / (theta * torch.cos(half))
    return scale * aa


# ---------------------------------------------------------------------------
# Bearing-vector accumulation (matches BiasSolverCostFunctor constructor)
# ---------------------------------------------------------------------------

class _PairAccum:
    """Precomputed F-sum matrices for one consecutive frame pair.

    Matches the six F matrices accumulated in BiasSolverCostFunctor:
        xxF, yyF, zzF, xyF, yzF, xzF
    where each is:
        sum_k  f1'[a] * f1'[b] * outer(f2', f2')
    with
        f1' = dR_base.T @ R_CB @ f1_norm     (C++: qcjk.inverse() * f1 = dR_imu.T @ R_CB @ f1)
        f2' = R_CB     @ f2_norm              (C++: _qic * f2, where _qic = R_CB due to frame swap)
    where R_CB = R_BC.T is the camera→body rotation and dR_base = ΔR_imu(b_g=0).

    Frame semantics: The C++ _qic parameter is passed as Rbc_ (body→camera), but the
    actual formula treats it as camera→body due to quaternion multiplication order:
        qcjk = qic^-1 * qjk = R_BC.T @ dR
        f1' = qcjk^-1 * f1 = (R_BC.T @ dR)^-1 * f1 = dR.T @ R_BC * f1
    But since R_BC rotates camera→body, this is equivalent to:
        f1' = dR.T @ R_CB @ f1 where R_CB is the effective rotation in the computation
    
    This appears to be a frame convention mismatch in the C++ code itself.
    The Python implementation uses the convention that produces correct results.
    """

    def __init__(
        self,
        bearings_i: torch.Tensor,   # (N, 3) bearing vectors frame i  (camera frame)
        bearings_j: torch.Tensor,   # (N, 3) bearing vectors frame j  (camera frame)
        dR_base: torch.Tensor,      # (3, 3) zero-bias IMU rotation for this gap
        R_BC: torch.Tensor,         # (3, 3) body(IMU)→camera extrinsic
    ):
        dtype = torch.float64
        z3 = torch.zeros(3, 3, dtype=dtype)
        xxF = z3.clone(); yyF = z3.clone(); zzF = z3.clone()
        xyF = z3.clone(); yzF = z3.clone(); xzF = z3.clone()

        R_CB    = R_BC.to(dtype).T   # camera→body (inverse of extrinsic)
        dR_base = dR_base.to(dtype)

        N = bearings_i.shape[0]
        for k in range(N):
            f1 = bearings_i[k].to(dtype)
            f2 = bearings_j[k].to(dtype)

            # Pre-rotate to body frame (matches C++ BiasSolverCostFunctor constructor)
            f1n = f1 / f1.norm()
            f2n = f2 / f2.norm()
            f1p = dR_base.T @ R_CB @ f1n     # dR_imu.T @ (R_BC.T) @ f  (transforms bearing)
            f2p = R_CB @ f2n                  # (R_BC.T) @ f             (same transformation)

            F = torch.outer(f2p, f2p)         # (3,3) projection matrix

            xxF += f1p[0] * f1p[0] * F
            yyF += f1p[1] * f1p[1] * F
            zzF += f1p[2] * f1p[2] * F
            xyF += f1p[0] * f1p[1] * F
            yzF += f1p[1] * f1p[2] * F
            xzF += f1p[0] * f1p[2] * F       # stored as xzF, passed as zxF to ComposeM

        self.xxF = xxF
        self.yyF = yyF
        self.zzF = zzF
        self.xyF = xyF
        self.yzF = yzF
        self.xzF = xzF                         # = zxF in ComposeM parameter naming

    def eval(self, cayley: torch.Tensor) -> torch.Tensor:
        """Squared smallest eigenvalue of M(cayley) — scalar, differentiable.

        Ceres in the reference minimises  residual^2  (before loss), so we
        use EV^2 rather than raw EV to match that behaviour.
        """
        M  = _compose_M(self.xxF, self.yyF, self.zzF,
                        self.xyF, self.yzF, self.xzF, cayley)
        ev = torch.linalg.eigvalsh(M)[0]
        return ev * ev


# ---------------------------------------------------------------------------
# ComposeM  (matches ComposeM / ComposeM template in opengvMethod.hpp)
# ---------------------------------------------------------------------------

def _compose_M(
    xxF: torch.Tensor, yyF: torch.Tensor, zzF: torch.Tensor,
    xyF: torch.Tensor, yzF: torch.Tensor, zxF: torch.Tensor,
    cayley: torch.Tensor,
) -> torch.Tensor:
    """Build 3×3 symmetric matrix M from F-sums and (unnormalised) Cayley R.

    Matches ComposeM in opengvMethod.hpp; zxF argument = xzF accumulation
    (symmetric, so the naming difference is irrelevant).
    """
    R  = _cayley_to_rot_reduced(cayley)   # (3,3)

    def rv(F: torch.Tensor, i: int, j: int) -> torch.Tensor:
        """R[i,:] @ F @ R[j,:] — scalar."""
        return R[i] @ F @ R[j]

    M00 = rv(yyF, 2, 2) - 2*rv(yzF, 2, 1) + rv(zzF, 1, 1)
    M01 = rv(yzF, 2, 0) - rv(xyF, 2, 2) - rv(zzF, 1, 0) + rv(zxF, 1, 2)
    M02 = rv(xyF, 2, 1) - rv(yyF, 2, 0) - rv(zxF, 1, 1) + rv(yzF, 1, 0)
    M11 = rv(zzF, 0, 0) - 2*rv(zxF, 0, 2) + rv(xxF, 2, 2)
    M12 = rv(zxF, 0, 1) - rv(yzF, 0, 0) - rv(xxF, 2, 1) + rv(xyF, 2, 0)
    M22 = rv(xxF, 1, 1) - 2*rv(xyF, 0, 1) + rv(yyF, 0, 0)

    M = torch.stack([
        torch.stack([M00, M01, M02]),
        torch.stack([M01, M11, M12]),
        torch.stack([M02, M12, M22]),
    ])
    return M


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def solve_gyro_bias(
    preint_results: list[PreintResult],
    bearing_pairs:  list[tuple[torch.Tensor, torch.Tensor]],
    R_BC:           torch.Tensor,
    b_g_init:       torch.Tensor | None = None,
    max_iters:      int = 200,
) -> tuple[torch.Tensor, bool]:
    """Estimate gyroscope bias using the bearing-vector / smallest-EV cost.

    Parameters
    ----------
    preint_results : N-1 PreintResult objects (one per consecutive kf gap).
    bearing_pairs  : N-1 tuples (bearings_i, bearings_j), each (M, 3)
                     unnormalised 3-D bearing vectors [u, v, 1] in the
                     normalised image plane of their respective frames.
    R_BC           : (3,3) body(IMU)→camera rotation extrinsic (float64).
    b_g_init       : (3,) initial gyro bias; zeros if None.
    max_iters      : maximum L-BFGS-B iterations.

    Returns
    -------
    (b_g_star, converged)
      b_g_star  : (3,) estimated gyro bias, float64.
      converged : True if optimiser reported success or residual is tiny.
    """
    assert len(preint_results) == len(bearing_pairs), (
        "preint_results and bearing_pairs must have the same length"
    )

    dtype = torch.float64
    import pypose as pp

    R_BC = R_BC.to(dtype)

    # Build zero-bias IMU rotations and _PairAccum for each gap
    pair_accums: list[_PairAccum] = []
    for pr, (bi, bj) in zip(preint_results, bearing_pairs):
        dR_log0  = pr.dR_log.to(dtype)
        dR_base  = pp.so3(dR_log0).Exp().matrix().squeeze(0).to(dtype)   # (3,3)
        pa = _PairAccum(bi, bj, dR_base, R_BC)
        pair_accums.append(pa)

    b_g0 = (b_g_init.clone().to(dtype)
             if b_g_init is not None
             else torch.zeros(3, dtype=dtype))

    # Pre-fetch Jacobians as plain tensors (no grad) for use inside the loop
    J_list = [pr.J_R_bg.to(dtype).detach() for pr in preint_results]

    def cost_and_grad(x: np.ndarray):
        b_g = torch.tensor(x, dtype=dtype, requires_grad=True)
        total = torch.tensor(0.0, dtype=dtype)
        for pa, J_bg in zip(pair_accums, J_list):
            # Compute the angle-axis representation of the rotation error
            # caused by the bias: δφ = J_R_bg @ Δb_g
            delta_log = J_bg @ b_g              # (3,) angle-axis
            # Convert to Cayley parameters for the optimization
            cayley    = _aa_to_cayley(delta_log)
            total     = total + pa.eval(cayley)
        total.backward()
        grad = b_g.grad.detach().numpy().astype(np.float64)
        return float(total.detach().item()), grad

    result = minimize(
        cost_and_grad,
        b_g0.numpy(),
        method="L-BFGS-B",
        jac=True,
        options={"maxiter": max_iters, "ftol": 1e-20, "gtol": 1e-12},
    )

    b_g_star  = torch.tensor(result.x, dtype=dtype)
    converged = bool(result.success) or float(result.fun) < 1e-6

    return b_g_star, converged
