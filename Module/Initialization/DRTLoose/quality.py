"""
Quality gates for the DRT-loose initializer.

All six gates must pass for the bootstrap to succeed. Each returns
(passed: bool, reason: str | None) where reason is a failure code when passed=False.
"""

from __future__ import annotations

import torch

# Failure reason codes (match design doc §5.7)
LOW_PARALLAX       = "LOW_PARALLAX"
INSUFFICIENT_OBS   = "INSUFFICIENT_OBS"
ILL_CONDITIONED    = "ILL_CONDITIONED"
NEGATIVE_DEPTH     = "NEGATIVE_DEPTH"
GRAVITY_BAD        = "GRAVITY_BAD"
NUMERIC            = "NUMERIC"


def check_avg_observation(tracks, n_keyframes: int, min_avg_obs: int = 30) -> tuple[bool, str | None]:
    """Gate 1: avg observations per keyframe >= min_avg_obs."""
    total_obs = sum(len(t.obs) for t in tracks)
    avg = total_obs / max(n_keyframes, 1)
    if avg < min_avg_obs:
        return False, INSUFFICIENT_OBS
    return True, None


def check_acceleration_observability(
    acc_norms: list[float] | torch.Tensor,
    gravity_norm: float = 9.81007,
    eps: float = 5e-3,
) -> tuple[bool, str | None]:
    """Gate 2: IMU excitation check — not pure rotation/stationary.

    Passes if:
    - |mean(||a||) - G| / G > eps   AND
    - count(|||a_i|| - G| / G < eps) <= 1
    """
    if isinstance(acc_norms, torch.Tensor):
        norms = acc_norms.to(dtype=torch.float64).tolist()
    else:
        norms = list(acc_norms)

    if len(norms) == 0:
        return False, LOW_PARALLAX

    mean_norm = sum(norms) / len(norms)
    mean_deviation = abs(mean_norm - gravity_norm) / gravity_norm

    if mean_deviation <= eps:
        return False, LOW_PARALLAX

    # Count how many individual readings look static (within eps of gravity)
    static_count = sum(
        1 for a in norms if abs(a - gravity_norm) / gravity_norm < eps
    )
    if static_count > 1:
        return False, LOW_PARALLAX

    return True, None


def check_ltl_conditioning_gate(LTL: torch.Tensor, max_cond: float = 1e8) -> tuple[bool, str | None]:
    """Gate 3: LTL condition number <= max_cond."""
    cond = torch.linalg.cond(LTL).item()
    if cond > max_cond:
        return False, ILL_CONDITIONED
    return True, None


def check_positive_depth(depth_signs: list[bool], min_ratio: float = 0.7) -> tuple[bool, str | None]:
    """Gate 4: >= min_ratio of triangulated points at positive depth.

    depth_signs: list of bools (True = positive depth from base view).
    """
    if len(depth_signs) == 0:
        return False, NEGATIVE_DEPTH
    pos = sum(1 for s in depth_signs if s)
    ratio = pos / len(depth_signs)
    if ratio < min_ratio:
        return False, NEGATIVE_DEPTH
    return True, None


def check_gravity_consistency(
    g_vec: torch.Tensor,
    gravity_norm: float = 9.81007,
    tol_rel: float = 1e-3,
) -> tuple[bool, str | None]:
    """Gate 5: |||g|| - G| / G <= tol_rel."""
    g_norm = g_vec.norm().item()
    err = abs(g_norm - gravity_norm) / gravity_norm
    if err > tol_rel:
        return False, GRAVITY_BAD
    return True, None


def check_state_finite(
    R0: torch.Tensor,
    v0: torch.Tensor,
    p0: torch.Tensor,
    b_g: torch.Tensor,
    b_a: torch.Tensor,
    max_bg_norm: float = 1.0,
    max_ba_norm: float = 5.0,
) -> tuple[bool, str | None]:
    """Gate 6: all state tensors finite and bias norms within sanity bounds."""
    tensors = [R0, v0, p0, b_g, b_a]
    for t in tensors:
        if not torch.isfinite(t).all():
            return False, NUMERIC

    if b_g.norm().item() > max_bg_norm:
        return False, NUMERIC
    if b_a.norm().item() > max_ba_norm:
        return False, NUMERIC

    return True, None


def run_all_quality_gates(
    tracks,
    n_keyframes: int,
    LTL: torch.Tensor,
    acc_norms,
    depth_signs: list[bool],
    g_vec: torch.Tensor,
    R0: torch.Tensor,
    v0: torch.Tensor,
    p0: torch.Tensor,
    b_g: torch.Tensor,
    b_a: torch.Tensor,
    cfg,  # DRTInitConfig
) -> tuple[bool, str | None]:
    """Run all 6 gates in order. Returns (all_passed, first_failure_reason)."""
    # Gate 1: average observations
    passed, reason = check_avg_observation(tracks, n_keyframes, cfg.quality_min_avg_obs)
    if not passed:
        return False, reason

    # Gate 2: IMU excitation
    passed, reason = check_acceleration_observability(acc_norms, cfg.gravity_norm)
    if not passed:
        return False, reason

    # Gate 3: LTL conditioning
    passed, reason = check_ltl_conditioning_gate(LTL, cfg.quality_max_cond)
    if not passed:
        return False, reason

    # Gate 4: positive depth ratio
    passed, reason = check_positive_depth(depth_signs, cfg.quality_min_pos_depth_ratio)
    if not passed:
        return False, reason

    # Gate 5: gravity consistency
    passed, reason = check_gravity_consistency(g_vec, cfg.gravity_norm, cfg.quality_gravity_tol_rel)
    if not passed:
        return False, reason

    # Gate 6: state finiteness and bias sanity
    passed, reason = check_state_finite(R0, v0, p0, b_g, b_a)
    if not passed:
        return False, reason

    return True, None
