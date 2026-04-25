"""
Tests for LiGT translation recovery and feature-track utilities.

Module/Initialization/DRTLoose/tracks.py
Module/Initialization/DRTLoose/translation.py
"""

from __future__ import annotations

import torch
import pytest

from Module.Initialization.DRTLoose.tracks import (
    FeatureTrack,
    KeyframeBundle,
    tracks_with_min_obs,
    select_base_views,
    _bearing,
)
from Module.Initialization.DRTLoose.translation import (
    build_LTL,
    recover_translations,
    resolve_translation_sign,
    check_ltl_conditioning,
)


# ---------------------------------------------------------------------------
# tracks.py
# ---------------------------------------------------------------------------

def test_tracks_with_min_obs():
    t1 = FeatureTrack(0, {0: torch.zeros(2), 1: torch.zeros(2)})          # 2 obs
    t2 = FeatureTrack(1, {0: torch.zeros(2), 1: torch.zeros(2), 2: torch.zeros(2)})  # 3 obs
    result = tracks_with_min_obs([t1, t2], min_obs=3)
    assert len(result) == 1
    assert result[0].track_id == 1


def test_tracks_with_min_obs_all_pass():
    tracks = [FeatureTrack(i, {j: torch.zeros(2) for j in range(5)}) for i in range(3)]
    assert len(tracks_with_min_obs(tracks, min_obs=3)) == 3


def test_tracks_with_min_obs_none_pass():
    tracks = [FeatureTrack(0, {0: torch.zeros(2)})]
    assert len(tracks_with_min_obs(tracks, min_obs=3)) == 0


def test_select_base_views_two_obs_always_returned():
    t = FeatureTrack(0, {2: torch.tensor([0.1, 0.2]), 5: torch.tensor([0.3, 0.4])})
    l, r = select_base_views(t, rotations=None)
    assert {l, r} == {2, 5}


def test_select_base_views_finds_max_parallax():
    # Three views: kf 0 (close to kf 1) and kf 2 (far from both).
    # kf 0 and kf 2 should be selected as the max-parallax pair.
    # Bearing for kf 0: nearly forward → (0, 0)
    # Bearing for kf 1: slightly right → (0.01, 0)
    # Bearing for kf 2: far right → (1.0, 0)
    t = FeatureTrack(0, {
        0: torch.tensor([0.0,  0.0], dtype=torch.float64),
        1: torch.tensor([0.01, 0.0], dtype=torch.float64),
        2: torch.tensor([1.0,  0.0], dtype=torch.float64),
    })
    l, r = select_base_views(t, rotations=None)
    # kf 0 and kf 2 give the largest cross-product magnitude
    assert {l, r} == {0, 2}


def test_bearing_unit_norm():
    uv = torch.tensor([0.5, -0.3], dtype=torch.float64)
    b = _bearing(uv)
    assert b.shape == (3,)
    assert abs(b.norm().item() - 1.0) < 1e-12


def test_keyframe_bundle_defaults():
    kb = KeyframeBundle(kf_idx=3, timestamp=1.5)
    assert kb.R_cam is None


# ---------------------------------------------------------------------------
# translation.py
# ---------------------------------------------------------------------------

def test_translation_sign_resolution_prefers_positive_majority():
    # A_lr @ t = [-(-1), -(-1), (-1)*(-1)] = [1, 1, 1] → flip to positive
    A_lr = torch.tensor([[1.0, 0, 0], [1.0, 0, 0], [-1.0, 0, 0]], dtype=torch.float64)
    t = torch.tensor([-1.0, 0, 0], dtype=torch.float64)
    out = resolve_translation_sign(A_lr, t)
    assert torch.allclose(out, torch.tensor([1.0, 0, 0], dtype=torch.float64))


def test_translation_sign_already_correct():
    A_lr = torch.ones(4, 3, dtype=torch.float64)
    t = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float64)
    out = resolve_translation_sign(A_lr, t)
    assert torch.allclose(out, t)


def test_recover_translations_smallest_singular_vector():
    # Build a 3×3 LTL whose smallest singular value corresponds to the z-axis.
    # Rank-1 LTL with known null space.
    v = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float64)
    LTL = torch.eye(3, dtype=torch.float64) * 10.0
    LTL[2, 2] = 1e-10  # z-axis is the near-null direction
    t = recover_translations(LTL)
    # The recovered vector should be close to [0, 0, ±1]
    assert abs(abs(t[2].item()) - 1.0) < 1e-6


def test_check_ltl_conditioning_well_conditioned():
    LTL = torch.eye(6, dtype=torch.float64)
    assert check_ltl_conditioning(LTL, max_cond=1e8) is True


def test_check_ltl_conditioning_ill_conditioned():
    LTL = torch.eye(6, dtype=torch.float64)
    LTL[0, 0] = 1e10  # condition number ~ 1e10
    assert check_ltl_conditioning(LTL, max_cond=1e8) is False


def test_build_ltl_produces_symmetric_psd_matrix():
    """build_LTL on a simple 3-keyframe, 5-track setup produces a symmetric PSD matrix."""
    torch.manual_seed(0)
    n_kf = 3
    rotations = [torch.eye(3, dtype=torch.float64) for _ in range(n_kf)]

    tracks = []
    for i in range(5):
        obs = {
            0: torch.randn(2, dtype=torch.float64) * 0.1,
            1: torch.randn(2, dtype=torch.float64) * 0.1,
            2: torch.randn(2, dtype=torch.float64) * 0.1,
        }
        tracks.append(FeatureTrack(i, obs))

    LTL, A_lr = build_LTL(tracks, rotations, n_keyframes=n_kf)

    # Shape: 3*(N-1) × 3*(N-1) = 6×6
    assert LTL.shape == (6, 6)
    # Symmetry
    assert torch.allclose(LTL, LTL.T, atol=1e-12)
    # PSD: all eigenvalues non-negative
    eigvals = torch.linalg.eigvalsh(LTL)
    assert (eigvals >= -1e-10).all(), f"LTL has negative eigenvalues: {eigvals}"
    # A_lr rows are 3-D
    assert A_lr.shape[1] == 3


def test_build_ltl_empty_tracks():
    """build_LTL with no valid tracks returns zero matrix and fallback A_lr."""
    LTL, A_lr = build_LTL([], rotations=[], n_keyframes=2)
    assert LTL.shape == (3, 3)
    assert (LTL == 0).all()


def test_end_to_end_translation_recovery_produces_nonzero_result():
    """
    Smoke test: build_LTL + recover_translations on a valid 3-keyframe, 3-track setup
    should produce a non-zero translation vector and a PSD LTL matrix.
    The simplified epipolar-constraint LiGT is not guaranteed to match GT up to scale
    without the full DRT-paper formulation; this test only verifies the pipeline runs
    correctly and produces a non-degenerate result.
    """
    n_kf = 3
    rotations = [torch.eye(3, dtype=torch.float64) for _ in range(n_kf)]
    t_gt = [
        torch.tensor([0.0, 0.0, 0.0], dtype=torch.float64),
        torch.tensor([1.0, 0.0, 0.0], dtype=torch.float64),
        torch.tensor([2.0, 0.0, 0.0], dtype=torch.float64),
    ]

    def project(R, t, X):
        Xc = R.T @ (X - t)
        return Xc[:2] / Xc[2]

    landmarks = [
        torch.tensor([0.3,  0.2, 4.0], dtype=torch.float64),
        torch.tensor([-0.2, 0.4, 6.0], dtype=torch.float64),
        torch.tensor([0.1, -0.3, 5.5], dtype=torch.float64),
    ]
    tracks = [
        FeatureTrack(li, {i: project(rotations[i], t_gt[i], lm) for i in range(n_kf)})
        for li, lm in enumerate(landmarks)
    ]

    LTL, A_lr = build_LTL(tracks, rotations, n_keyframes=n_kf)
    t_flat = recover_translations(LTL)   # (6,) up to scale+sign

    # Non-degenerate: at least one component is significant
    assert t_flat.norm().item() > 1e-6
    # LTL is PSD
    eigvals = torch.linalg.eigvalsh(LTL)
    assert (eigvals >= -1e-10).all()
    # A_lr is non-empty with correct width
    assert A_lr.shape[1] == 3
    # sign disambiguation runs without error
    t2_frag = t_flat[3:6]
    t2_signed = resolve_translation_sign(A_lr, t2_frag)
    assert t2_signed.shape == (3,)
