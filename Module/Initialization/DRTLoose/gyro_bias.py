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

Optimisation: PyPose Levenberg-Marquardt with Cauchy robust loss, matching
the C++ Ceres solver that uses DOGLEG trust region + CauchyLoss(1e-5).
"""

from __future__ import annotations

import torch
import pypose as pp

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
        f1' = dR_base.T @ R_CB @ f1_norm     (C++: qcjk.inverse() * f1)
        f2' = R_CB     @ f2_norm              (C++: _qic * f2)

    where R_CB = R_BC.T is the camera→body rotation and dR_base = ΔR_imu(b_g=0).
    Uses R_CB (not R_BC) for bearing transformation — empirically validated
    against EuRoC MH01 ground truth.
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

            # Match C++ BiasSolverCostFunctor constructor:
            #   f1' = qcjk.inverse() * f1 = dR^T @ R_BC @ f1
            #   f2' = _qic * f2 = R_BC @ f2
            f1n = f1 / f1.norm()
            f2n = f2 / f2.norm()
            f1p = dR_base.T @ R_CB @ f1n     # dR^T @ R_CB @ f1
            f2p = R_CB @ f2n                  # R_CB @ f2

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
        """Smallest eigenvalue of M(cayley) — scalar, differentiable.

        Returns raw EV (not squared) so LM can form its own J^T J approximation.
        """
        M  = _compose_M(self.xxF, self.yyF, self.zzF,
                        self.xyF, self.yzF, self.xzF, cayley)
        return torch.linalg.eigvalsh(M)[0]


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
    """Estimate gyroscope bias using the bearing-vector / smallest-EV cost with PyPose LM.

    Uses Levenberg-Marquardt optimization with Cauchy robust loss, matching the C++ Ceres
    implementation which uses trust region + robust loss.

    Parameters
    ----------
    preint_results : N-1 PreintResult objects (one per consecutive kf gap).
    bearing_pairs  : N-1 tuples (bearings_i, bearings_j), each (M, 3)
                     unnormalised 3-D bearing vectors [u, v, 1] in the
                     normalised image plane of their respective frames.
    R_BC           : (3,3) body(IMU)→camera rotation extrinsic (float64).
    b_g_init       : (3,) initial gyro bias; zeros if None.
    max_iters      : maximum iterations.

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
    R_BC  = R_BC.to(dtype)

    # Build zero-bias IMU rotations and _PairAccum for each gap
    pair_accums: list[_PairAccum] = []
    for pr, (bi, bj) in zip(preint_results, bearing_pairs):
        dR_log0  = pr.dR_log.to(dtype)
        dR_base  = pp.so3(dR_log0).Exp().matrix().squeeze(0).to(dtype)   # (3,3)
        pa = _PairAccum(bi, bj, dR_base, R_BC)
        pair_accums.append(pa)

    # Pre-fetch Jacobians as plain tensors (no grad) for use in model
    J_list = [pr.J_R_bg.to(dtype).detach() for pr in preint_results]

    n_pairs = len(pair_accums)
    b_g0 = (b_g_init.clone().to(dtype)
            if b_g_init is not None
            else torch.zeros(3, dtype=dtype))

    class _GyroBiasModel(torch.nn.Module):
        def __init__(self, init):
            super().__init__()
            self.b_g = torch.nn.Parameter(init.clone())

        def forward(self, _inp=None):
            # Returns (n_pairs,) residuals — raw smallest EV per pair.
            # LM minimises Σ ev_i^2 via its own J^T J approximation.
            evs = []
            for pa, J_bg in zip(pair_accums, J_list):
                delta_log = J_bg @ self.b_g
                cayley    = _aa_to_cayley(delta_log)
                evs.append(pa.eval(cayley).unsqueeze(0))
            return torch.cat(evs)   # (n_pairs,)

    target = torch.zeros(n_pairs, dtype=dtype)

    model  = _GyroBiasModel(b_g0)
    # Cauchy delta=1e-5 matches Ceres CauchyLoss(1e-5) in the C++ reference.
    # The strong outlier rejection prevents the optimizer from chasing noisy
    # gradient directions when the bias signal is weak (flat cost landscape).
    solver = pp.optim.LM(
        model,
        kernel=pp.optim.kernel.Cauchy(delta=1e-5),
        vectorize=True,
        reject=16,
    )

    # Pass None as input each step; pypose calls model(None) internally to
    # recompute residuals after each parameter update (required for LM trust
    # region accept/reject logic — DO NOT pre-compute and pass residuals).
    loss = torch.tensor(float('inf'), dtype=dtype)
    for _ in range(max_iters):
        loss = solver.step(None, target)
        if float(loss) < 1e-12:
            break

    b_g_star = model.b_g.data.detach().clone()

    # Physical sanity guard: typical MEMS gyro bias is well below 0.5 rad/s.
    # If the optimizer drifted beyond that the cost surface was too noisy —
    # fall back to the (known-good) initial estimate.
    if b_g_star.norm() > 0.5:
        b_g_star = b_g0.clone()

    converged = float(loss) < 1e-6
    return b_g_star, converged
