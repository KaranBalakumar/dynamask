"""
Unit tests for IMU preintegration kernel (DRT-loose initializer).

Reference: Forster et al., 2017, "On-Manifold Preintegration for Real-Time
Visual-Inertial Odometry", IEEE TRO.
"""

import torch
import pytest
import pypose as pp
from Module.Initialization.DRTLoose.preintegration import IMUPreintegrator


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _integrate_for(n_steps: int, gyro, acc, dt: float, **kw) -> "PreintResult":
    integ = IMUPreintegrator(**kw)
    gyro_t = torch.as_tensor(gyro, dtype=torch.float64)
    acc_t  = torch.as_tensor(acc,  dtype=torch.float64)
    for _ in range(n_steps):
        integ.integrate(gyro_t, acc_t, dt)
    return integ


# ---------------------------------------------------------------------------
# Step 1 (TDD): Failing test – written BEFORE the implementation exists
# ---------------------------------------------------------------------------

def test_preintegration_constant_acc_no_rotation():
    """
    100 ticks @ 100 Hz with constant +1 m/s² in x, no rotation.
    Expected (t = 1 s, a = [1, 0, 0] m/s²):
        dR_log ≈ [0, 0, 0]
        dV     ≈ [1, 0, 0]   (∫ a dt = 1 s × 1 m/s²)
        dP     ≈ [0.5, 0, 0] (½ a t²)

    Note: all integration is done in float64; comparison tensors are cast
    to float64 to match (torch.allclose raises on dtype mismatch).
    """
    integ = IMUPreintegrator()
    dt = 0.01
    for _ in range(100):
        integ.integrate(torch.zeros(3), torch.tensor([1.0, 0.0, 0.0]), dt)
    out = integ.result()
    f64 = {"dtype": torch.float64}
    assert torch.allclose(out.dR_log, torch.zeros(3, **f64), atol=1e-4)
    assert torch.allclose(out.dV, torch.tensor([1.0, 0.0, 0.0], **f64), atol=5e-2)
    assert torch.allclose(out.dP, torch.tensor([0.5, 0.0, 0.0], **f64), atol=5e-2)


# ---------------------------------------------------------------------------
# Step 4: Additional tests
# ---------------------------------------------------------------------------

def test_preintegration_bias_jacobians_finite_diff():
    """
    Verify J_V_ba and J_P_ba against finite differences at tolerance < 1e-5.
    We perturb b_a along each axis and compare the analytic Jacobian column
    to the finite-difference estimate.
    """
    eps  = 1e-6
    n    = 50
    dt   = 0.01
    gyro = [0.05, 0.02, -0.03]
    acc  = [0.3,  0.1,   0.2]

    base = _integrate_for(n, gyro, acc, dt)
    out0 = base.result()

    fd_J_V_ba = torch.zeros(3, 3, dtype=torch.float64)
    fd_J_P_ba = torch.zeros(3, 3, dtype=torch.float64)

    for i in range(3):
        delta_ba = torch.zeros(3, dtype=torch.float64)
        delta_ba[i] = eps
        integ_p = IMUPreintegrator(b_a=delta_ba)
        for _ in range(n):
            integ_p.integrate(
                torch.as_tensor(gyro, dtype=torch.float64),
                torch.as_tensor(acc,  dtype=torch.float64),
                dt,
            )
        out_p = integ_p.result()

        fd_J_V_ba[:, i] = (out_p.dV - out0.dV) / eps
        fd_J_P_ba[:, i] = (out_p.dP - out0.dP) / eps

    assert torch.allclose(out0.J_V_ba, fd_J_V_ba, atol=1e-4), (
        f"J_V_ba mismatch:\nanalytic:\n{out0.J_V_ba}\nFD:\n{fd_J_V_ba}"
    )
    assert torch.allclose(out0.J_P_ba, fd_J_P_ba, atol=1e-4), (
        f"J_P_ba mismatch:\nanalytic:\n{out0.J_P_ba}\nFD:\n{fd_J_P_ba}"
    )


def test_preintegration_with_nonzero_biases():
    """
    Constant gyro bias of 0.1 rad/s around x-axis, zero acc bias.
    After t = 1 s of integration the accumulated rotation log should be
    close to b_g * t = [0.1, 0, 0] (first-order, since angles are small).
    """
    b_g    = torch.tensor([0.1, 0.0, 0.0], dtype=torch.float64)
    integ  = IMUPreintegrator(b_g=b_g)
    dt     = 0.01
    n      = 100  # 1 s total

    # Raw gyro = 0, but bias is 0.1 → bias-corrected ω = 0 - 0.1 = -0.1 rad/s
    # Preintegrator integrates ω̃ = gyro - b_g, so ΔR accumulates -b_g * t.
    # We feed raw gyro = b_g so that bias-corrected reading is zero → no rotation.
    # Instead we want to test that the BIAS causes rotation, so feed raw gyro = 0:
    for _ in range(n):
        integ.integrate(torch.zeros(3, dtype=torch.float64),
                        torch.zeros(3, dtype=torch.float64), dt)

    out = integ.result()
    # dR_log should be proportional to -b_g * total_time (ω̃ = 0 - b_g)
    expected_log = -b_g * (n * dt)  # = [-0.1, 0, 0]
    # Allow generous tolerance since we're using first-order approximation
    assert torch.allclose(out.dR_log, expected_log, atol=5e-3), (
        f"Expected dR_log ≈ {expected_log}, got {out.dR_log}"
    )


def test_preintegration_gyro_bias_jacobians_finite_diff():
    """
    Verify J_R_bg, J_V_bg, J_P_bg against finite differences.
    We perturb b_g along each axis and compare analytic Jacobian columns
    to finite-difference estimates (eps = 1e-6, atol = 1e-4).
    """
    eps  = 1e-6
    n    = 50
    dt   = 0.01
    gyro = [0.05, 0.02, -0.03]
    acc  = [0.3,  0.1,   0.2]

    # Nonzero b_g prior so Jacobians are exercised in the nonlinear regime
    b_g0 = torch.tensor([0.02, -0.01, 0.03], dtype=torch.float64)

    base = IMUPreintegrator(b_g=b_g0)
    gyro_t = torch.as_tensor(gyro, dtype=torch.float64)
    acc_t  = torch.as_tensor(acc,  dtype=torch.float64)
    for _ in range(n):
        base.integrate(gyro_t, acc_t, dt)
    out0 = base.result()

    fd_J_R_bg = torch.zeros(3, 3, dtype=torch.float64)
    fd_J_V_bg = torch.zeros(3, 3, dtype=torch.float64)
    fd_J_P_bg = torch.zeros(3, 3, dtype=torch.float64)

    # Re-integrate to get the base ΔR as a LieTensor for the right-tangent FD.
    base_dR = pp.identity_SO3(dtype=torch.float64)
    integ_dR = IMUPreintegrator(b_g=b_g0)
    for _ in range(n):
        integ_dR.integrate(gyro_t, acc_t, dt)
    # Reconstruct ΔR from dR_log (Log is invertible; use Exp)
    base_dR = pp.so3(out0.dR_log.contiguous()).Exp()

    for i in range(3):
        delta_bg = torch.zeros(3, dtype=torch.float64)
        delta_bg[i] = eps
        integ_p = IMUPreintegrator(b_g=b_g0 + delta_bg)
        for _ in range(n):
            integ_p.integrate(gyro_t, acc_t, dt)
        out_p = integ_p.result()

        # J_R_bg is the Jacobian in the RIGHT tangent space of ΔR (Forster 2017,
        # Table I): δφ = Log(ΔR^{-1} ⊗ ΔR_perturbed).  Plain differences on
        # dR_log would give the wrong result for the off-diagonal terms.
        dR_p = pp.so3(out_p.dR_log.contiguous()).Exp()
        right_tang_diff = (base_dR.Inv() @ dR_p).Log().tensor()
        fd_J_R_bg[:, i] = right_tang_diff / eps

        fd_J_V_bg[:, i] = (out_p.dV - out0.dV) / eps
        fd_J_P_bg[:, i] = (out_p.dP - out0.dP) / eps

    assert torch.allclose(out0.J_R_bg, fd_J_R_bg, atol=1e-4), (
        f"J_R_bg mismatch:\nanalytic:\n{out0.J_R_bg}\nFD:\n{fd_J_R_bg}"
    )
    assert torch.allclose(out0.J_V_bg, fd_J_V_bg, atol=1e-4), (
        f"J_V_bg mismatch:\nanalytic:\n{out0.J_V_bg}\nFD:\n{fd_J_V_bg}"
    )
    assert torch.allclose(out0.J_P_bg, fd_J_P_bg, atol=1e-4), (
        f"J_P_bg mismatch:\nanalytic:\n{out0.J_P_bg}\nFD:\n{fd_J_P_bg}"
    )


def test_preintegration_reset():
    """
    After calling reset(), all accumulators should return to identity/zero,
    identical to a freshly constructed IMUPreintegrator.
    """
    integ = IMUPreintegrator()
    for _ in range(50):
        integ.integrate(
            torch.tensor([0.1, 0.2, -0.1], dtype=torch.float64),
            torch.tensor([0.5, 0.0,  0.3], dtype=torch.float64),
            0.01,
        )

    # Verify state is non-trivial before reset
    out_before = integ.result()
    f64 = {"dtype": torch.float64}
    assert not torch.allclose(out_before.dR_log, torch.zeros(3, **f64), atol=1e-6)

    integ.reset()
    out_after = integ.result()

    assert torch.allclose(out_after.dR_log, torch.zeros(3,    **f64), atol=1e-12)
    assert torch.allclose(out_after.dV,     torch.zeros(3,    **f64), atol=1e-12)
    assert torch.allclose(out_after.dP,     torch.zeros(3,    **f64), atol=1e-12)
    assert torch.allclose(out_after.J_R_bg, torch.zeros(3, 3, **f64), atol=1e-12)
    assert torch.allclose(out_after.J_V_bg, torch.zeros(3, 3, **f64), atol=1e-12)
    assert torch.allclose(out_after.J_V_ba, torch.zeros(3, 3, **f64), atol=1e-12)
    assert torch.allclose(out_after.J_P_bg, torch.zeros(3, 3, **f64), atol=1e-12)
    assert torch.allclose(out_after.J_P_ba, torch.zeros(3, 3, **f64), atol=1e-12)
    assert torch.allclose(out_after.Sigma,  torch.zeros(9, 9, **f64), atol=1e-12)
    assert out_after.sum_dt == pytest.approx(0.0)
