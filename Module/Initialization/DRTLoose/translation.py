"""
LiGT translation recovery (up-to-scale) for the DRT-loose initializer.

Reference:
  Liu, Y., et al. "LiGT: Lightweight Global Translation Estimation from 3D
  Feature Tracks." — simplified epipolar-constraint variant used here.

All arithmetic is float64 to maintain numerical precision through the SVD.

Key conventions
---------------
- Keyframe 0 is the reference frame; its translation t_0 = 0 is excluded from
  the optimisation.
- t_flat returned by recover_translations() has shape (3*(N-1),) and corresponds
  to translations t_1, t_2, ..., t_{N-1} laid out consecutively.
- Bearing vectors are normalised: b = [u, v, 1] / ||[u, v, 1]||.
"""

from __future__ import annotations

import torch
from .tracks import FeatureTrack, select_base_views, _bearing


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _skew(v: torch.Tensor) -> torch.Tensor:
    """Build a 3×3 skew-symmetric matrix from a (3,) vector."""
    x, y, z = v[0], v[1], v[2]
    O = torch.zeros((), dtype=v.dtype, device=v.device)
    return torch.stack([
        torch.stack([O, -z,  y]),
        torch.stack([z,  O, -x]),
        torch.stack([-y, x,  O]),
    ])


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_LTL(
    tracks: list[FeatureTrack],
    rotations: list[torch.Tensor],
    n_keyframes: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build the LᵀL matrix for LiGT translation recovery.

    For each track with ≥ 2 observations, this function iterates over all
    ordered pairs of keyframes (ki, kj) within the track and adds an
    epipolar-style constraint:

        (skew(x_j) @ R_ij) @ t_j  -  skew(x_j) @ R_ij @ R_ji @ t_i  = 0

    where x_j is the bearing vector at kj and R_ij = R_j @ R_i^T.

    Simplified, the contribution to the linear system per pair is:
        A_j @ t_j  +  A_i @ t_i  =  0

    with  A_j = skew(x_j)   and   A_i = -skew(x_j) @ R_ij.

    Keyframe 0 is the reference (t_0 = 0) and its columns are dropped from
    the LTL system.

    Parameters
    ----------
    tracks      : list of FeatureTrack objects.
    rotations   : list of (3,3) float64 rotation matrices indexed by kf_idx.
    n_keyframes : total number of keyframes N (including the reference kf 0).

    Returns
    -------
    LTL   : (3*(N-1), 3*(N-1)) symmetric positive-semi-definite matrix.
    A_lr  : (P, 3) sign-disambiguation rows; one row per track that has a
            valid base pair.  Note: only the first row of A_j for the base
            pair (l, r) is stored, giving a (P, 3) matrix rather than
            (P, 3*N) — the sign constraint only needs the direction of the
            base-view coefficient.
    """
    N = n_keyframes
    size = 3 * (N - 1)
    LTL = torch.zeros(size, size, dtype=torch.float64)
    A_rows: list[torch.Tensor] = []

    for track in tracks:
        kf_indices = sorted(track.obs.keys())
        if len(kf_indices) < 2:
            continue

        l, r = select_base_views(track, rotations)

        for ii, ki in enumerate(kf_indices):
            for kj in kf_indices[ii + 1:]:
                # Relative rotation R_ij = R_j @ R_i^T
                if (rotations is not None
                        and ki < len(rotations)
                        and kj < len(rotations)
                        and rotations[ki] is not None
                        and rotations[kj] is not None):
                    R_ij = (rotations[kj].to(dtype=torch.float64)
                            @ rotations[ki].to(dtype=torch.float64).T)
                else:
                    R_ij = torch.eye(3, dtype=torch.float64)

                x_i = _bearing(track.obs[ki].to(dtype=torch.float64))
                x_j = _bearing(track.obs[kj].to(dtype=torch.float64))

                # Coefficient matrices for the linear constraint A_j @ t_j + A_i @ t_i = 0
                #   A_j = skew(x_j)
                #   A_i = -skew(x_j) @ R_ij
                A_j = _skew(x_j)                  # (3, 3)
                A_i = -(A_j @ R_ij)               # (3, 3)

                # Column offsets in the (3*(N-1)) system (kf 0 excluded)
                col_j = 3 * (kj - 1) if kj > 0 else None
                col_i = 3 * (ki - 1) if ki > 0 else None

                # Accumulate into LTL symmetrically
                if col_j is not None:
                    LTL[col_j:col_j + 3, col_j:col_j + 3] += A_j.T @ A_j
                if col_i is not None:
                    LTL[col_i:col_i + 3, col_i:col_i + 3] += A_i.T @ A_i
                if col_i is not None and col_j is not None:
                    LTL[col_i:col_i + 3, col_j:col_j + 3] += A_i.T @ A_j
                    LTL[col_j:col_j + 3, col_i:col_i + 3] += A_j.T @ A_i

                # Sign-disambiguation row: collect base-pair (l, r) contribution
                if ki == l and kj == r:
                    # Use first row of A_j as a sign indicator
                    A_rows.append(A_j[0].clone())  # shape (3,)

    if A_rows:
        A_lr = torch.stack(A_rows, dim=0)  # (P, 3)
    else:
        A_lr = torch.zeros(1, 3, dtype=torch.float64)

    return LTL, A_lr


def recover_translations(LTL: torch.Tensor) -> torch.Tensor:
    """Recover up-to-scale translations via the smallest right-singular vector of LTL.

    Parameters
    ----------
    LTL : (3*(N-1), 3*(N-1)) symmetric matrix built by build_LTL.

    Returns
    -------
    t_flat : (3*(N-1),) translation vector (up to global scale and sign).
    """
    _, _, Vt = torch.linalg.svd(LTL)
    return Vt[-1]  # row corresponding to the smallest singular value


def resolve_translation_sign(A_lr: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """Flip sign of t so that the majority of (A_lr @ t_fragment) entries are positive.

    For sign disambiguation we test A_lr @ t_fragment where t_fragment has the
    same number of columns as A_lr (either 3*(N-1) or 3, depending on how
    A_lr was built).

    Parameters
    ----------
    A_lr : (P, D) matrix of sign-constraint rows.
    t    : (D,) translation vector (up-to-scale, unknown sign).

    Returns
    -------
    t with the sign chosen so that the majority of (A_lr @ t) > 0.
    """
    judge = A_lr @ t          # (P,)
    pos = (judge > 0).sum().item()
    neg = judge.numel() - pos
    return t if pos >= neg else -t


def check_ltl_conditioning(LTL: torch.Tensor, max_cond: float = 1e8) -> bool:
    """Return True if LTL is well-conditioned (cond(LTL) <= max_cond).

    Uses torch.linalg.cond which defaults to the 2-norm condition number
    (ratio of largest to smallest singular value).

    Parameters
    ----------
    LTL      : square matrix to evaluate.
    max_cond : threshold above which the matrix is considered ill-conditioned.

    Returns
    -------
    True if cond(LTL) <= max_cond, False otherwise.
    """
    cond = torch.linalg.cond(LTL).item()
    return cond <= max_cond
