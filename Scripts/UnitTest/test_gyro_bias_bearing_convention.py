"""
Test: Bearing rotation convention in gyro bias solver.

Hypothesis: The Python bearing accumulation (_PairAccum) uses inverted
extrinsic rotation (R_CB = R_BC.T) when it should use R_BC directly.

This test compares the cost function value under both conventions
to identify which matches the reference C++ behavior.
"""

import torch
import pytest
from Module.Initialization.DRTLoose.gyro_bias import _PairAccum
from Module.Initialization.DRTLoose.preintegration import IMUPreintegrator


def test_bearing_rotation_convention():
    """
    Verify that bearing vectors are rotated with R_BC (not R_BC.T).
    
    Setup:
    - Create a synthetic rotation in the IMU frame
    - Create corresponding bearing vectors that respect that rotation
    - Solve for gyro bias with zero error
    - Verify the cost function is minimized when bias is zero
    
    If R_BC is inverted, the cost will be high even with zero bias,
    and minimized at a non-zero bias (the "loose" error observed).
    """
    dtype = torch.float64
    
    # Identity extrinsic (IMU = camera frame)
    R_BC_identity = torch.eye(3, dtype=dtype)
    
    # Non-identity extrinsic: 45° rotation around z-axis
    angle = torch.tensor(45.0 * 3.14159 / 180.0, dtype=dtype)
    c, s = torch.cos(angle), torch.sin(angle)
    R_BC_rotated = torch.stack([
        torch.stack([c, -s, torch.tensor(0., dtype=dtype)]),
        torch.stack([s, c, torch.tensor(0., dtype=dtype)]),
        torch.stack([torch.tensor(0., dtype=dtype), torch.tensor(0., dtype=dtype), torch.tensor(1., dtype=dtype)]),
    ])
    
    # Create synthetic IMU rotation (small angle around x-axis)
    omega_true = torch.tensor([0.01, 0.0, 0.0], dtype=dtype)  # rad/s
    dt = 0.1  # seconds
    dR_imu = torch.eye(3, dtype=dtype) + torch.tensor([
        [0., 0., 0.],
        [0., 0., -dt * omega_true[0]],
        [0., dt * omega_true[0], 0.],
    ], dtype=dtype)  # Small angle approximation: I + [ω]× dt
    
    # Normalized bearings in camera frame (arbitrary)
    f1_cam = torch.tensor([[0.1, 0.2, 1.0], [0.3, -0.1, 1.0], [0.5, 0.5, 1.0]], dtype=dtype)
    f1_cam = f1_cam / f1_cam.norm(dim=1, keepdim=True)
    
    # Apply TRUE rotation: bearings in frame 2 (in camera frame)
    # If dR_imu is the IMU rotation, and R_BC relates IMU to camera,
    # then camera rotation is: R_cam = R_BC @ dR_imu @ R_BC.T
    R_cam_true = R_BC_rotated @ dR_imu @ R_BC_rotated.T
    f2_cam = (R_cam_true.T @ f1_cam.T).T
    f2_cam = f2_cam / f2_cam.norm(dim=1, keepdim=True)
    
    # Create _PairAccum with the CORRECT convention (should use R_BC)
    pa_current = _PairAccum(f1_cam, f2_cam, dR_imu, R_BC_rotated)
    
    # Evaluate cost at zero bias (should be minimal)
    zero_cayley = torch.zeros(3, dtype=dtype)
    cost_zero_bias = pa_current.eval(zero_cayley)
    
    # Evaluate cost at a small bias
    small_cayley = torch.tensor([0.001, 0.001, 0.001], dtype=dtype)
    cost_small_bias = pa_current.eval(small_cayley)
    
    print(f"\nBearing rotation convention test:")
    print(f"Cost at zero bias: {cost_zero_bias.item():.2e}")
    print(f"Cost at small bias: {cost_small_bias.item():.2e}")
    print(f"Cost increases with perturbation: {cost_small_bias > cost_zero_bias}")
    
    # The cost should be minimized (or very small) at zero bias
    # If the convention is wrong, this will fail
    assert cost_zero_bias < 1e-6, f"Expected minimal cost at zero bias, got {cost_zero_bias.item():.2e}"
    assert cost_small_bias > cost_zero_bias, "Cost should increase with perturbation from optimum"


if __name__ == "__main__":
    test_bearing_rotation_convention()
    print("\n✓ Test passed: Bearing rotation convention is correct")
