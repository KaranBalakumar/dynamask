"""Verify analytic Jacobian ∂f_rigid/∂d against finite differences."""
import torch
import pypose as pp
import numpy as np


def compute_rigid_flow_and_jacobian(
    depth: torch.Tensor,      # (B, 1, H, W)
    K: torch.Tensor,          # (B, 3, 3)
    T_rel: torch.Tensor,      # (B, 4, 4)  SE(3) relative pose
):
    """Compute rigid flow f_rigid and analytic Jacobian J_d = ∂f_rigid/∂d.

    f_rigid = π(K · T_rel · X_cam) − uv
    where X_cam = d · K⁻¹ · [u,v,1]ᵀ

    Returns:
        f_rigid : (B, 2, H, W)  rigid flow in pixels
        J_d     : (B, 2, H, W)  ∂f_rigid/∂d   per pixel
    """
    B, _, H, W = depth.shape
    device, dtype = depth.device, depth.dtype
    K = K.to(dtype=dtype, device=device)

    fx, fy = K[:, 0, 0], K[:, 1, 1]
    cx, cy = K[:, 0, 2], K[:, 1, 2]

    # Unit ray r̂ = [(u−cx)/fx, (v−cy)/fy, 1]  per pixel
    ys, xs = torch.meshgrid(
        torch.arange(H, device=device, dtype=dtype),
        torch.arange(W, device=device, dtype=dtype), indexing="ij",
    )
    rx = (xs - cx.view(-1, 1, 1)) / fx.view(-1, 1, 1)   # (B, H, W)
    ry = (ys - cy.view(-1, 1, 1)) / fy.view(-1, 1, 1)   # (B, H, W)
    rz = torch.ones(B, H, W, device=device, dtype=dtype)

    R = T_rel[:, :3, :3]   # (B, 3, 3)
    t = T_rel[:, :3, 3]    # (B, 3)

    # X_next = d · R·r̂ + t
    Rr_x = (R[:, 0, 0].view(B,1,1) * rx + R[:, 0, 1].view(B,1,1) * ry + R[:, 0, 2].view(B,1,1) * rz)
    Rr_y = (R[:, 1, 0].view(B,1,1) * rx + R[:, 1, 1].view(B,1,1) * ry + R[:, 1, 2].view(B,1,1) * rz)
    Rr_z = (R[:, 2, 0].view(B,1,1) * rx + R[:, 2, 1].view(B,1,1) * ry + R[:, 2, 2].view(B,1,1) * rz)

    X_next_x = depth.squeeze(1) * Rr_x + t[:, 0].view(B, 1, 1)
    X_next_y = depth.squeeze(1) * Rr_y + t[:, 1].view(B, 1, 1)
    X_next_z = depth.squeeze(1) * Rr_z + t[:, 2].view(B, 1, 1)

    # Project
    Z_clamp = X_next_z.clamp_min(1e-10)
    u_proj = fx.view(-1,1,1) * X_next_x / Z_clamp + cx.view(-1,1,1)
    v_proj = fy.view(-1,1,1) * X_next_y / Z_clamp + cy.view(-1,1,1)

    f_rigid = torch.stack([u_proj - xs, v_proj - ys], dim=1)  # (B, 2, H, W)

    # --- Analytic Jacobian J_d = ∂f_rigid/∂d ---
    # ∂u'/∂d = fx * (Rr_x * Z - X * Rr_z) / Z²
    Z_sq = Z_clamp * Z_clamp
    du_dd = fx.view(-1,1,1) * (Rr_x * Z_clamp - X_next_x * Rr_z) / Z_sq
    dv_dd = fy.view(-1,1,1) * (Rr_y * Z_clamp - X_next_y * Rr_z) / Z_sq
    J_d = torch.stack([du_dd, dv_dd], dim=1)  # (B, 2, H, W)

    return f_rigid, J_d


def test_jacobian_finite_difference():
    """Verify analytic J_d against finite-difference Jacobian."""
    torch.manual_seed(42)
    B, H, W = 1, 60, 80
    device = torch.device("cpu")
    dtype = torch.float64

    # Random camera
    K = torch.tensor([[[320., 0, 320.], [0, 320., 240.], [0, 0, 1.]]], dtype=dtype)
    # Random depth
    depth = torch.rand(B, 1, H, W, dtype=dtype) * 10 + 1  # [1, 11] meters
    # Random pose
    T_rel = torch.eye(4, dtype=dtype).unsqueeze(0)
    T_rel[:, :3, 3] = torch.tensor([0.05, -0.02, 0.01], dtype=dtype)  # translation
    # Small random rotation
    import pypose as pp
    rot = pp.so3(torch.randn(1, 3, dtype=dtype) * 0.1).Exp().matrix()
    T_rel[:, :3, :3] = rot

    # Analytic
    f_rigid, J_d_analytic = compute_rigid_flow_and_jacobian(depth, K, T_rel)

    # Finite difference
    eps = 1e-6
    depth_plus = depth + eps
    f_rigid_plus, _ = compute_rigid_flow_and_jacobian(depth_plus, K, T_rel)
    J_d_fd = (f_rigid_plus - f_rigid) / eps

    # Compare
    diff = (J_d_analytic - J_d_fd).abs()
    rel_diff = diff / (J_d_fd.abs() + 1e-10)

    print(f"J_d analytic shape: {J_d_analytic.shape}")
    print(f"J_d finite-diff shape: {J_d_fd.shape}")
    print(f"Absolute diff: mean={diff.mean():.10f}, max={diff.max():.10f}")
    print(f"Relative diff: mean={rel_diff.mean():.10f}, max={rel_diff.max():.10f}")

    assert diff.max() < 5e-5, f"Jacobian mismatch! max diff={diff.max():.2e}"
    print("✓ Analytic Jacobian matches finite differences")


def test_jacobian_identity_pose():
    """With identity pose, f_rigid ≈ 0 and J_d ≈ 0."""
    B, H, W = 1, 30, 40
    dtype = torch.float64
    K = torch.tensor([[[320., 0, 320.], [0, 320., 240.], [0, 0, 1.]]], dtype=dtype)
    depth = torch.ones(B, 1, H, W, dtype=dtype) * 5.0
    T_id = torch.eye(4, dtype=dtype).unsqueeze(0)

    f_rigid, J_d = compute_rigid_flow_and_jacobian(depth, K, T_id)
    assert f_rigid.abs().max() < 1e-10, f"Identity pose should give zero rigid flow, got max={f_rigid.abs().max():.2e}"
    assert J_d.abs().max() < 1e-10, f"Identity pose should give zero Jacobian, got max={J_d.abs().max():.2e}"
    print("✓ Identity pose gives zero rigid flow and zero Jacobian")


def test_jacobian_pure_translation():
    """Pure forward translation. At optical center J_d=0, at edges J_d≠0."""
    B, H, W = 1, 120, 160
    dtype = torch.float64
    K = torch.tensor([[[160., 0, 80.], [0, 160., 60.], [0, 0, 1.]]], dtype=dtype)
    depth = torch.ones(B, 1, H, W, dtype=dtype) * 5.0
    T_rel = torch.eye(4, dtype=dtype).unsqueeze(0)
    T_rel[:, 2, 3] = 0.5  # forward Z

    f_rigid, J_d = compute_rigid_flow_and_jacobian(depth, K, T_rel)
    # Optical center at (cx=80, cy=60): J_d should be zero (pinhole ray invariant)
    jd_center = J_d[0, :, 60, 80]  # (2,) at pixel (80, 60)
    assert jd_center.abs().max() < 1e-10, f"J_d should be ~0 at optical center, got {jd_center}"
    print("✓ Pure forward translation: J_d ≈ 0 at optical center")
    # Corner: J_d should be non-zero
    jd_corner = J_d[0, :, 0, 0]
    assert jd_corner.abs().max() > 1e-8, f"J_d should be non-zero at corner, got {jd_corner}"
    print(f"✓ Pure forward translation: J_d at corner = {jd_corner.abs().max():.2e}")


if __name__ == "__main__":
    test_jacobian_identity_pose()
    test_jacobian_pure_translation()
    test_jacobian_finite_difference()
    print("\nAll tests passed.")
