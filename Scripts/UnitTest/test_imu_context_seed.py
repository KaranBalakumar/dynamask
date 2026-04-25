"""Tests for IMUContext.seed_from_drt."""

import torch
import pypose as pp
import pytest

from Module.Initialization.DRTLoose.types import DRTInitResult


def make_success_drt() -> DRTInitResult:
    return DRTInitResult.success_result(
        R0=torch.eye(3, dtype=torch.float64),
        v0=torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64),
        p0=torch.tensor([0.1, 0.2, 0.3], dtype=torch.float64),
        b_g=torch.tensor([0.01, 0.02, 0.03], dtype=torch.float64),
        b_a=torch.tensor([0.1, 0.2, 0.3], dtype=torch.float64),
        g_W=torch.tensor([0.0, 0.0, 9.81007], dtype=torch.float64),
        scale=1.5,
    )


def make_minimal_ctx():
    """Return a minimal duck-typed IMUContext-like object for testing seed_from_drt."""
    from Module.Network.IMUContext.imu_context import IMUContext

    class MinimalCtx:
        _state = None
        _P = None
        _prev_cam_state = None
        gravity_world = torch.tensor([0., 0., 9.81], dtype=torch.float64)
        seed_from_drt = IMUContext.seed_from_drt

    return MinimalCtx()


def test_seed_from_drt_sets_velocity(monkeypatch):
    """seed_from_drt should set EKF state[3:6] to drt.v0."""
    ctx = make_minimal_ctx()
    drt = make_success_drt()
    ctx.seed_from_drt(drt)

    assert ctx._state is not None
    assert torch.allclose(ctx._state[3:6].float(), torch.tensor([1., 2., 3.]))


def test_seed_from_drt_sets_biases():
    ctx = make_minimal_ctx()
    drt = make_success_drt()
    ctx.seed_from_drt(drt)

    assert torch.allclose(ctx._state[9:12].float(), torch.tensor([0.01, 0.02, 0.03]))
    assert torch.allclose(ctx._state[12:15].float(), torch.tensor([0.1, 0.2, 0.3]))


def test_seed_from_drt_overwrites_gravity():
    ctx = make_minimal_ctx()
    drt = make_success_drt()
    ctx.seed_from_drt(drt)

    assert torch.allclose(ctx.gravity_world, torch.tensor([0., 0., 9.81007], dtype=torch.float64))


def test_seed_from_drt_sets_p_init():
    ctx = make_minimal_ctx()
    drt = make_success_drt()
    ctx.seed_from_drt(drt)

    assert ctx._P is not None
    assert ctx._P.shape == (15, 15)


def test_seed_from_drt_primes_prev_cam_state():
    ctx = make_minimal_ctx()
    drt = make_success_drt()
    ctx.seed_from_drt(drt)

    assert ctx._prev_cam_state is not None
    assert torch.allclose(ctx._prev_cam_state, ctx._state)


def test_seed_from_drt_raises_on_failure():
    ctx = make_minimal_ctx()
    bad = DRTInitResult.failure("ILL_CONDITIONED")
    with pytest.raises(AssertionError):
        ctx.seed_from_drt(bad)
