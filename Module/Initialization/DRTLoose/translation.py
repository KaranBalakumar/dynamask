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

    Direct port of drtLooselyCoupled::build_LTL.  For each track with ≥ 3
    observations, the function selects a base pair (l, r) with maximum parallax
    and then, for every other view i (i ≠ l), adds a three-view constraint
    involving the unknown global translations t_l, t_r, t_i.

    The resulting LTL system is (3(N−1), 3(N−1)) — keyframe 0 is the reference
    with t_0 ≡ 0 and its rows/cols are excluded.

    Parameters
    ----------
    tracks      : list of FeatureTrack objects.
    rotations   : list of (3,3) float64 rotation matrices indexed by kf_idx.
    n_keyframes : total number of keyframes N (including the reference kf 0).

    Returns
    -------
    LTL   : (3*(N-1), 3*(N-1)) symmetric positive-semi-definite matrix.
    A_lr  : (P, 3*N) sign-disambiguation matrix; one row per valid track.
    """
    N = n_keyframes                 # number of keyframes
    size = 3 * (N - 1)
    LTL = torch.zeros(size, size, dtype=torch.float64)
    A_lr_rows: list[torch.Tensor] = []

    for track in tracks:
        kf_indices = sorted(track.obs.keys())
        if len(kf_indices) < 3:
            continue

        l, r = select_base_views(track, rotations)

        for ki in kf_indices:
            if ki == l:
                continue       # skip base view l — only process non-base views

            i = ki
            # Fetch bearings (unit norm, camera frame)
            x_i = _bearing(track.obs[i].to(dtype=torch.float64))    # (3,)
            x_l = _bearing(track.obs[l].to(dtype=torch.float64))    # (3,)
            x_r = _bearing(track.obs[r].to(dtype=torch.float64))    # (3,)

            xi_cross = _skew(x_i)                                    # [x_i]×  (3,3)

            # Camera-frame relative rotations (matches C++ naming)
            R_i  = rotations[i].to(dtype=torch.float64)              # (3,3)
            R_l  = rotations[l].to(dtype=torch.float64)              # (3,3)
            R_r  = rotations[r].to(dtype=torch.float64)              # (3,3)

            R_cicl = R_i.T @ R_l    # R_Cl→Ci  = rotation from l to i    (3,3)
            R_crcl = R_r.T @ R_l    # R_Cl→Cr  = rotation from l to r    (3,3)

            # ---- a_lr row vector (1×3) -----------------------------------
            a_lr_tmp = _skew(R_crcl @ x_l) @ x_r                  # (3,)
            a_lr_t   = a_lr_tmp @ _skew(x_r)                       # (3,) row

            # A_lr sign-disambiguation rows (C++ stores per-track in A_lr)
            a_lr_l = a_lr_t @ R_r.T                                 # (3,)  for l_base
            a_lr_r = -a_lr_l                                         # (3,)  for r_base

            # ---- theta_lr (scalar) ---------------------------------------
            theta_lr_vec = _skew(x_r) @ R_crcl @ x_l               # (3,)
            theta_lr = theta_lr_vec.dot(theta_lr_vec)               # scalar

            # ---- Coefficient matrices ------------------------------------
            #   Coeff_B (@ r_base)   = [x_i]× · R_cicl · x_l · a_lr · R_rᵀ
            #   Coeff_C (@ i_view)   = θ_lr · [x_i]× · R_iᵀ
            #   Coeff_D (@ l_base)   = −(B + C)
            x_l_rot = R_cicl @ x_l                                   # (3,)
            xl_alr  = torch.outer(x_l_rot, a_lr_t)                   # (3,3)  x_l · a_lr
            Coeff_B = xi_cross @ (xl_alr @ R_r.T)                    # (3,3)

            Coeff_C = theta_lr * xi_cross @ R_i.T                    # (3,3)

            Coeff_D = -(Coeff_B + Coeff_C)                           # (3,3)

            # ---- tmp_LiGT_vec = [0 ... B@r ... C@i ... D@l ... 0] ------
            tmp = torch.zeros(3, 3 * N, dtype=torch.float64)
            tmp[:, 3*r : 3*r+3] = Coeff_B
            tmp[:, 3*i : 3*i+3] = Coeff_C
            tmp[:, 3*l : 3*l+3] = Coeff_D

            # ---- LTL accumulation (drop cols/rows for kf-0) -------------
            LTL_l_row = Coeff_D.T @ tmp     # (3, 3N)
            LTL_r_row = Coeff_B.T @ tmp     # (3, 3N)
            LTL_i_row = Coeff_C.T @ tmp     # (3, 3N)

            # Exclude first 3 columns (= kf-0) from each contribution
            if l > 0:
                LTL[3*l-3 : 3*l, :] += LTL_l_row[:, 3:]
            if r > 0:
                LTL[3*r-3 : 3*r, :] += LTL_r_row[:, 3:]
            if i > 0:
                LTL[3*i-3 : 3*i, :] += LTL_i_row[:, 3:]

            # ---- A_lr accumulation (sign-disambiguation) -----------------
            a_lr_full = torch.zeros(3 * N, dtype=torch.float64)
            a_lr_full[3*l : 3*l+3] = a_lr_l
            a_lr_full[3*r : 3*r+3] = a_lr_r
            A_lr_rows.append(a_lr_full)

    if A_lr_rows:
        A_lr = torch.stack(A_lr_rows, dim=0)                         # (P, 3*N)
    else:
        A_lr = torch.zeros(1, 3 * N, dtype=torch.float64)

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


def recover_translations_stereo(
    tracks: list[FeatureTrack],
    rotations: list[torch.Tensor],
    depth_maps: list[torch.Tensor],
    K: torch.Tensor,
    n_keyframes: int,
) -> torch.Tensor:
    """Recover metric camera translations from stereo depth + tracked features.

    For each consecutive keyframe pair (i, i+1), computes 3D points from
    stereo depth at both frames for every tracked feature, then estimates the
    relative camera translation via robust median of 3D displacements:

        t_rel = median_k ( X_{j,k} − R_{i→j} · X_{i,k} )

    Relative translations are chained to produce global translations with
    t_0 = 0.

    Parameters
    ----------
    tracks      : list of FeatureTrack objects.
    rotations   : list of (3,3) camera rotation matrices indexed by kf_idx.
    depth_maps  : list of (1, 1, H, W) or (1, H, W) stereo depth tensors,
                  one per keyframe.
    K           : (3,3) camera intrinsics matrix.
    n_keyframes : total number of keyframes N.

    Returns
    -------
    translations : (N, 3) metric translations in world/camera frame (t_0 = 0).
    """
    dtype = torch.float64
    N = n_keyframes
    translations = torch.zeros(N, 3, dtype=dtype)
    fx, fy = float(K[0, 0]), float(K[1, 1])
    cx, cy = float(K[0, 2]), float(K[1, 2])

    # Helper: bilinear sample depth at subpixel (px, py)
    def sample_depth(depth_map: torch.Tensor, px: float, py: float) -> float | None:
        H, W = depth_map.shape[-2], depth_map.shape[-1]
        # depth_map is (1,1,H,W) or (1,H,W)
        if depth_map.dim() == 4:
            d = depth_map[0, 0]
        elif depth_map.dim() == 3:
            d = depth_map[0]
        else:
            d = depth_map

        # Clamp to valid range for bilinear interpolation
        x0 = int(px)
        y0 = int(py)
        x1 = x0 + 1
        y1 = y0 + 1

        if x0 < 0 or x1 >= W or y0 < 0 or y1 >= H:
            return None

        wx = px - x0
        wy = py - y0

        d00 = d[y0, x0].item()
        d10 = d[y0, x1].item()
        d01 = d[y1, x0].item()
        d11 = d[y1, x1].item()

        # Skip if any corner is invalid
        if any(v <= 0 or v != v for v in [d00, d10, d01, d11]):
            return None

        d_top = d00 * (1 - wx) + d10 * wx
        d_bot = d01 * (1 - wx) + d11 * wx
        return d_top * (1 - wy) + d_bot * wy

    # For each consecutive keyframe pair, estimate relative translation
    for i in range(N - 1):
        j = i + 1
        R_rel = (rotations[j].to(dtype) @ rotations[i].to(dtype).T)  # R_i→j

        displacements: list[torch.Tensor] = []
        for track in tracks:
            if i not in track.obs or j not in track.obs:
                continue

            # Get bearing vectors (normalized image coords)
            uv_i = track.obs[i].to(dtype)  # (2,) [u, v]
            uv_j = track.obs[j].to(dtype)  # (2,) [u, v]

            # Convert to pixel coords
            px_i = uv_i[0].item() * fx + cx
            py_i = uv_i[1].item() * fy + cy
            px_j = uv_j[0].item() * fx + cx
            py_j = uv_j[1].item() * fy + cy

            # Sample depth at both frames
            d_i = sample_depth(depth_maps[i], px_i, py_i)
            d_j = sample_depth(depth_maps[j], px_j, py_j)
            if d_i is None or d_j is None:
                continue

            # 3D point in camera frame: X = d * [u, v, 1]
            X_i = d_i * torch.stack([uv_i[0], uv_i[1], torch.ones((), dtype=dtype)])
            X_j = d_j * torch.stack([uv_j[0], uv_j[1], torch.ones((), dtype=dtype)])

            # Displacement in camera frame j
            disp = X_j - R_rel @ X_i
            displacements.append(disp)

        if len(displacements) < 5:
            # Fall back to zero translation for this segment
            translations[j] = translations[i]
            continue

        # Robust median of displacements
        stacked = torch.stack(displacements)  # (M, 3)
        t_rel = stacked.median(dim=0).values

        # Chain: t_j = t_i + R_i @ t_rel  (R_i = rotations[i], maps camera i to world)
        translations[j] = translations[i] + rotations[i].to(dtype) @ t_rel

    # The absolute scale from sparse-stereo depth is unreliable — normalise so
    # the linear-alignment step can solve for scale from IMU+gravity constraints.
    # This preserves the stereo-derived shape/direction while avoiding depth bias.
    t_norm = translations.norm(dim=-1).max().item()
    if t_norm > 1e-10:
        translations = translations / t_norm

    return translations


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
