"""
Unit tests for the DRT-loose quality gates (Module/Initialization/DRTLoose/quality.py).
"""

import torch
import pytest

from Module.Initialization.DRTLoose.quality import (
    check_avg_observation,
    check_acceleration_observability,
    check_ltl_conditioning_gate,
    check_positive_depth,
    check_gravity_consistency,
    check_state_finite,
    run_all_quality_gates,
    INSUFFICIENT_OBS,
    ILL_CONDITIONED,
    NEGATIVE_DEPTH,
    GRAVITY_BAD,
    NUMERIC,
    LOW_PARALLAX,
)
from Module.Initialization.DRTLoose.types import DRTInitConfig
from Module.Initialization.DRTLoose.tracks import FeatureTrack


# ---------------------------------------------------------------------------
# Gate 1: check_avg_observation
# ---------------------------------------------------------------------------

def test_check_avg_observation_pass():
    # 5 tracks each with 10 obs across 4 keyframes -> avg = 50/4 = 12.5 >= 10
    tracks = [
        FeatureTrack(track_id=i, obs={kf: torch.zeros(2) for kf in range(10)})
        for i in range(5)
    ]
    passed, reason = check_avg_observation(tracks, n_keyframes=4, min_avg_obs=10)
    assert passed
    assert reason is None


def test_check_avg_observation_fail():
    # 2 obs per track (1 track) across 10 keyframes -> avg = 2/10 = 0.2 < 30
    tracks = [FeatureTrack(track_id=0, obs={0: torch.zeros(2), 1: torch.zeros(2)})]
    passed, reason = check_avg_observation(tracks, n_keyframes=10, min_avg_obs=30)
    assert not passed
    assert reason == INSUFFICIENT_OBS


def test_check_avg_observation_exact_threshold():
    # Exactly at threshold: 300 obs across 10 keyframes -> avg = 30 = min_avg_obs
    tracks = [
        FeatureTrack(track_id=i, obs={kf: torch.zeros(2) for kf in range(30)})
        for i in range(10)
    ]
    # 10 tracks * 30 obs each = 300 total / 10 kf = 30 avg
    passed, reason = check_avg_observation(tracks, n_keyframes=10, min_avg_obs=30)
    assert passed


def test_check_avg_observation_empty_tracks():
    passed, reason = check_avg_observation([], n_keyframes=10, min_avg_obs=30)
    assert not passed
    assert reason == INSUFFICIENT_OBS


# ---------------------------------------------------------------------------
# Gate 2: check_acceleration_observability
# ---------------------------------------------------------------------------

def test_check_acceleration_observability_pass():
    # Dynamic motion: norms clearly different from gravity (9.81007)
    acc_norms = [12.5, 13.0, 14.2, 11.8, 15.0]
    passed, reason = check_acceleration_observability(acc_norms, gravity_norm=9.81007, eps=5e-3)
    assert passed
    assert reason is None


def test_check_acceleration_observability_fail_stationary():
    # All norms very close to gravity -> static
    g = 9.81007
    acc_norms = [g * (1 + 1e-5), g * (1 - 1e-5), g, g * 1.000001]
    passed, reason = check_acceleration_observability(acc_norms, gravity_norm=g, eps=5e-3)
    assert not passed
    assert reason == LOW_PARALLAX


def test_check_acceleration_observability_tensor_input():
    acc_norms = torch.tensor([12.5, 13.0, 14.2], dtype=torch.float64)
    passed, reason = check_acceleration_observability(acc_norms, gravity_norm=9.81007, eps=5e-3)
    assert passed


def test_check_acceleration_observability_empty():
    passed, reason = check_acceleration_observability([], gravity_norm=9.81007)
    assert not passed
    assert reason == LOW_PARALLAX


# ---------------------------------------------------------------------------
# Gate 3: check_ltl_conditioning_gate
# ---------------------------------------------------------------------------

def test_check_ltl_conditioning_gate_pass():
    # Well-conditioned: identity matrix has cond = 1
    LTL = torch.eye(6, dtype=torch.float64)
    passed, reason = check_ltl_conditioning_gate(LTL, max_cond=1e8)
    assert passed
    assert reason is None


def test_check_ltl_conditioning_gate_fail():
    # Ill-conditioned: large ratio between singular values
    LTL = torch.diag(torch.tensor([1e10, 1.0, 1.0, 1.0, 1.0, 1.0], dtype=torch.float64))
    passed, reason = check_ltl_conditioning_gate(LTL, max_cond=1e8)
    assert not passed
    assert reason == ILL_CONDITIONED


# ---------------------------------------------------------------------------
# Gate 4: check_positive_depth
# ---------------------------------------------------------------------------

def test_check_positive_depth_pass():
    # 8 out of 10 positive -> 0.8 >= 0.7
    depth_signs = [True] * 8 + [False] * 2
    passed, reason = check_positive_depth(depth_signs, min_ratio=0.7)
    assert passed
    assert reason is None


def test_check_positive_depth_fail():
    # Only 5 out of 10 positive -> 0.5 < 0.7
    depth_signs = [True] * 5 + [False] * 5
    passed, reason = check_positive_depth(depth_signs, min_ratio=0.7)
    assert not passed
    assert reason == NEGATIVE_DEPTH


def test_check_positive_depth_empty():
    passed, reason = check_positive_depth([], min_ratio=0.7)
    assert not passed
    assert reason == NEGATIVE_DEPTH


def test_check_positive_depth_all_positive():
    depth_signs = [True] * 20
    passed, reason = check_positive_depth(depth_signs, min_ratio=0.7)
    assert passed


def test_check_positive_depth_exact_threshold():
    # Exactly 70%: 7 out of 10
    depth_signs = [True] * 7 + [False] * 3
    passed, reason = check_positive_depth(depth_signs, min_ratio=0.7)
    assert passed


# ---------------------------------------------------------------------------
# Gate 5: check_gravity_consistency
# ---------------------------------------------------------------------------

def test_check_gravity_consistency_pass():
    g = torch.tensor([0., 0., 9.81], dtype=torch.float64)
    passed, reason = check_gravity_consistency(g, gravity_norm=9.81007, tol_rel=1e-3)
    # |9.81 - 9.81007| / 9.81007 ≈ 7e-6 < 1e-3 -> pass
    assert passed
    assert reason is None


def test_check_gravity_consistency_fail():
    g = torch.tensor([0., 0., 5.0], dtype=torch.float64)  # way off
    passed, reason = check_gravity_consistency(g, gravity_norm=9.81007, tol_rel=1e-3)
    assert not passed
    assert reason == GRAVITY_BAD


def test_check_gravity_consistency_exact_norm():
    g = torch.tensor([0., 0., 9.81007], dtype=torch.float64)
    passed, reason = check_gravity_consistency(g, gravity_norm=9.81007, tol_rel=1e-3)
    assert passed


def test_check_gravity_consistency_non_vertical():
    # Tilted gravity but correct magnitude
    import math
    g_norm = 9.81007
    g = torch.tensor([g_norm / math.sqrt(2), 0., g_norm / math.sqrt(2)], dtype=torch.float64)
    passed, reason = check_gravity_consistency(g, gravity_norm=g_norm, tol_rel=1e-3)
    assert passed


# ---------------------------------------------------------------------------
# Gate 6: check_state_finite
# ---------------------------------------------------------------------------

def test_check_state_finite_pass():
    R0 = torch.eye(3, dtype=torch.float64)
    v0 = torch.zeros(3, dtype=torch.float64)
    p0 = torch.zeros(3, dtype=torch.float64)
    b_g = torch.tensor([0.001, 0., 0.], dtype=torch.float64)
    b_a = torch.tensor([0., 0.01, 0.], dtype=torch.float64)
    passed, reason = check_state_finite(R0, v0, p0, b_g, b_a)
    assert passed
    assert reason is None


def test_check_state_finite_fail_nan():
    R0 = torch.full((3, 3), float('nan'), dtype=torch.float64)
    passed, reason = check_state_finite(
        R0, torch.zeros(3), torch.zeros(3),
        torch.zeros(3), torch.zeros(3),
    )
    assert not passed
    assert reason == NUMERIC


def test_check_state_finite_fail_inf():
    v0 = torch.tensor([float('inf'), 0., 0.], dtype=torch.float64)
    passed, reason = check_state_finite(
        torch.eye(3, dtype=torch.float64), v0, torch.zeros(3),
        torch.zeros(3), torch.zeros(3),
    )
    assert not passed
    assert reason == NUMERIC


def test_check_state_finite_fail_bg_too_large():
    b_g = torch.tensor([2.0, 0., 0.], dtype=torch.float64)  # > max_bg_norm=1.0
    passed, reason = check_state_finite(
        torch.eye(3, dtype=torch.float64), torch.zeros(3), torch.zeros(3),
        b_g, torch.zeros(3),
        max_bg_norm=1.0,
    )
    assert not passed
    assert reason == NUMERIC


def test_check_state_finite_fail_ba_too_large():
    b_a = torch.tensor([6.0, 0., 0.], dtype=torch.float64)  # > max_ba_norm=5.0
    passed, reason = check_state_finite(
        torch.eye(3, dtype=torch.float64), torch.zeros(3), torch.zeros(3),
        torch.zeros(3), b_a,
        max_ba_norm=5.0,
    )
    assert not passed
    assert reason == NUMERIC


# ---------------------------------------------------------------------------
# run_all_quality_gates integration
# ---------------------------------------------------------------------------

def _make_passing_inputs():
    """Build a minimal set of inputs that pass all 6 gates."""
    # Gate 1: tracks
    tracks = [
        FeatureTrack(track_id=i, obs={kf: torch.zeros(2) for kf in range(30)})
        for i in range(10)
    ]
    n_keyframes = 10
    # Gate 2: acc_norms — dynamic (12-15 m/s², mean ~13.4, far from 9.81)
    acc_norms = [12.5, 13.0, 14.2, 11.8, 15.0, 12.0, 13.5, 14.0, 12.8, 13.2]
    # Gate 3: well-conditioned LTL
    LTL = torch.eye(27, dtype=torch.float64)  # 3*(10-1)=27
    # Gate 4: depth signs
    depth_signs = [True] * 9 + [False] * 1
    # Gate 5: gravity
    g_vec = torch.tensor([0., 0., 9.81007], dtype=torch.float64)
    # Gate 6: finite state
    R0 = torch.eye(3, dtype=torch.float64)
    v0 = torch.zeros(3, dtype=torch.float64)
    p0 = torch.zeros(3, dtype=torch.float64)
    b_g = torch.tensor([0.001, 0., 0.], dtype=torch.float64)
    b_a = torch.tensor([0., 0.01, 0.], dtype=torch.float64)
    cfg = DRTInitConfig(quality_min_avg_obs=30)
    return tracks, n_keyframes, LTL, acc_norms, depth_signs, g_vec, R0, v0, p0, b_g, b_a, cfg


def test_run_all_quality_gates_pass():
    args = _make_passing_inputs()
    passed, reason = run_all_quality_gates(*args)
    assert passed
    assert reason is None


def test_run_all_quality_gates_fail_on_gate1():
    tracks, n_keyframes, LTL, acc_norms, depth_signs, g_vec, R0, v0, p0, b_g, b_a, cfg = _make_passing_inputs()
    # Override tracks to fail gate 1
    bad_tracks = [FeatureTrack(track_id=0, obs={0: torch.zeros(2)})]
    passed, reason = run_all_quality_gates(
        bad_tracks, n_keyframes, LTL, acc_norms, depth_signs,
        g_vec, R0, v0, p0, b_g, b_a, cfg,
    )
    assert not passed
    assert reason == INSUFFICIENT_OBS


def test_run_all_quality_gates_fail_on_gate5():
    tracks, n_keyframes, LTL, acc_norms, depth_signs, g_vec, R0, v0, p0, b_g, b_a, cfg = _make_passing_inputs()
    bad_g = torch.tensor([0., 0., 5.0], dtype=torch.float64)
    passed, reason = run_all_quality_gates(
        tracks, n_keyframes, LTL, acc_norms, depth_signs,
        bad_g, R0, v0, p0, b_g, b_a, cfg,
    )
    assert not passed
    assert reason == GRAVITY_BAD


def test_run_all_quality_gates_fail_on_gate6():
    tracks, n_keyframes, LTL, acc_norms, depth_signs, g_vec, R0, v0, p0, b_g, b_a, cfg = _make_passing_inputs()
    bad_R0 = torch.full((3, 3), float('nan'), dtype=torch.float64)
    passed, reason = run_all_quality_gates(
        tracks, n_keyframes, LTL, acc_norms, depth_signs,
        g_vec, bad_R0, v0, p0, b_g, b_a, cfg,
    )
    assert not passed
    assert reason == NUMERIC
