"""Verify analytic Jacobians J_d and J_{f->X} against finite differences."""
import torch
import pypose as pp
import numpy as np


def compute_rigid_flow_and_jacobian(
    depth: torch.Tensor,      # (B, 1, H, W)
    K: torch.Tensor,          # (B, 3, 3)
    T_rel: torch.Tensor,      # (B, 4, 4)  SE(3) relative pose
):
    """Compute rigid flow f_rigid and analytic Jacobian J_d = df_rigid/dd."""
    B, _, H, W = depth.shape
    device, dtype = depth.device, depth.dtype
    K = K.to(dtype=dtype, device=device)

    fx, fy = K[:, 0, 0], K[:, 1, 1]
    cx, cy = K[:, 0, 2], K[:, 1, 2]

    ys, xs = torch.meshgrid(
        torch.arange(H, device=device, dtype=dtype),
        torch.arange(W, device=device, dtype=dtype), indexing="ij",
    )
    rx = (xs - cx.view(-1, 1, 1)) / fx.view(-1, 1, 1)
    ry = (ys - cy.view(-1, 1, 1)) / fy.view(-1, 1, 1)
    rz = torch.ones(B, H, W, device=device, dtype=dtype)

    R = T_rel[:, :3, :3]
    t = T_rel[:, :3, 3]

    Rr_x = (R[:, 0, 0].view(B,1,1) * rx + R[:, 0, 1].view(B,1,1) * ry + R[:, 0, 2].view(B,1,1) * rz)
    Rr_y = (R[:, 1, 0].view(B,1,1) * rx + R[:, 1, 1].view(B,1,1) * ry + R[:, 1, 2].view(B,1,1) * rz)
    Rr_z = (R[:, 2, 0].view(B,1,1) * rx + R[:, 2, 1].view(B,1,1) * ry + R[:, 2, 2].view(B,1,1) * rz)

    X_next_x = depth.squeeze(1) * Rr_x + t[:, 0].view(B, 1, 1)
    X_next_y = depth.squeeze(1) * Rr_y + t[:, 1].view(B, 1, 1)
    X_next_z = depth.squeeze(1) * Rr_z + t[:, 2].view(B, 1, 1)

    Z_clamp = X_next_z.clamp_min(1e-10)
    u_proj = fx.view(-1,1,1) * X_next_x / Z_clamp + cx.view(-1,1,1)
    v_proj = fy.view(-1,1,1) * X_next_y / Z_clamp + cy.view(-1,1,1)

    f_rigid = torch.stack([u_proj - xs, v_proj - ys], dim=1)

    Z_sq = Z_clamp * Z_clamp
    du_dd = fx.view(-1,1,1) * (Rr_x * Z_clamp - X_next_x * Rr_z) / Z_sq
    dv_dd = fy.view(-1,1,1) * (Rr_y * Z_clamp - X_next_y * Rr_z) / Z_sq
    J_d = torch.stack([du_dd, dv_dd], dim=1)

    return f_rigid, J_d


def test_jacobian_finite_difference():
    """Verify analytic J_d against finite-difference Jacobian."""
    torch.manual_seed(42)
    B, H, W = 1, 60, 80
    device = torch.device("cpu")
    dtype = torch.float64

    K = torch.tensor([[[320., 0, 320.], [0, 320., 240.], [0, 0, 1.]]], dtype=dtype)
    depth = torch.rand(B, 1, H, W, dtype=dtype) * 10 + 1
    T_rel = torch.eye(4, dtype=dtype).unsqueeze(0)
    T_rel[:, :3, 3] = torch.tensor([0.05, -0.02, 0.01], dtype=dtype)
    rot = pp.so3(torch.randn(1, 3, dtype=dtype) * 0.1).Exp().matrix()
    T_rel[:, :3, :3] = rot

    f_rigid, J_d_analytic = compute_rigid_flow_and_jacobian(depth, K, T_rel)

    eps = 1e-6
    depth_plus = depth + eps
    f_rigid_plus, _ = compute_rigid_flow_and_jacobian(depth_plus, K, T_rel)
    J_d_fd = (f_rigid_plus - f_rigid) / eps

    diff = (J_d_analytic - J_d_fd).abs()
    print(f"J_d FD: max diff={diff.max():.10f}")
    assert diff.max() < 5e-5, f"Jacobian mismatch! max diff={diff.max():.2e}"


def test_jacobian_identity_pose():
    """With identity pose, f_rigid = 0 and J_d = 0."""
    B, H, W = 1, 30, 40
    dtype = torch.float64
    K = torch.tensor([[[320., 0, 320.], [0, 320., 240.], [0, 0, 1.]]], dtype=dtype)
    depth = torch.ones(B, 1, H, W, dtype=dtype) * 5.0
    T_id = torch.eye(4, dtype=dtype).unsqueeze(0)

    f_rigid, J_d = compute_rigid_flow_and_jacobian(depth, K, T_id)
    assert f_rigid.abs().max() < 1e-10
    assert J_d.abs().max() < 1e-10


def test_jacobian_pure_translation():
    """Pure forward translation. At optical center J_d=0, at edges J_d!=0."""
    B, H, W = 1, 120, 160
    dtype = torch.float64
    K = torch.tensor([[[160., 0, 80.], [0, 160., 60.], [0, 0, 1.]]], dtype=dtype)
    depth = torch.ones(B, 1, H, W, dtype=dtype) * 5.0
    T_rel = torch.eye(4, dtype=dtype).unsqueeze(0)
    T_rel[:, 2, 3] = 0.5

    f_rigid, J_d = compute_rigid_flow_and_jacobian(depth, K, T_rel)
    jd_center = J_d[0, :, 60, 80]
    assert jd_center.abs().max() < 1e-10
    jd_corner = J_d[0, :, 0, 0]
    assert jd_corner.abs().max() > 1e-8


# ---- 2x3 Jacobian J_{f->X} tests ----

def test_jacobian_3d_finite_difference():
    """Verify analytic 2x3 Jacobian J_{f->X} against finite differences."""
    from Train.MatchingNet.loss import compute_rigid_flow_jacobian_3d

    torch.manual_seed(42)
    B, H, W = 1, 20, 30
    dtype = torch.float64

    K = torch.tensor([[[100., 0, 15.], [0, 100., 10.], [0, 0, 1.]]], dtype=dtype)
    depth = torch.rand(B, 1, H, W, dtype=dtype) * 5 + 1

    T_rel = torch.eye(4, dtype=dtype).unsqueeze(0)
    T_rel[:, :3, 3] = torch.tensor([0.03, -0.01, 0.05], dtype=dtype)
    rot = pp.so3(torch.randn(1, 3, dtype=dtype) * 0.1).Exp().matrix()
    T_rel[:, :3, :3] = rot

    _, J_3d, _ = compute_rigid_flow_jacobian_3d(depth, K, T_rel)

    # Finite difference: perturb 3D point, reproject, compute df/dX
    eps = 1e-6
    fx, fy = K[0, 0, 0].item(), K[0, 1, 1].item()
    cx, cy = K[0, 0, 2].item(), K[0, 1, 2].item()
    R, t_vec = T_rel[0, :3, :3], T_rel[0, :3, 3]

    # 3D points X = d * K^{-1} * [u,v,1]^T in camera frame
    ys, xs = torch.meshgrid(
        torch.arange(H, device=torch.device("cpu"), dtype=dtype),
        torch.arange(W, device=torch.device("cpu"), dtype=dtype), indexing="ij",
    )
    d_sq = depth.squeeze()
    X_pts = torch.stack([d_sq * (xs - cx) / fx, d_sq * (ys - cy) / fy, d_sq], dim=0)  # (3, H, W)

    J_fd = torch.zeros(2, 3, H, W, dtype=dtype)
    for comp in range(3):
        dX = torch.zeros_like(X_pts)
        dX[comp] = eps
        X_nom = X_pts.flatten(1)
        X_per = (X_pts + dX).flatten(1)

        # nom = R @ X + t
        Xp0 = (R @ X_nom + t_vec.unsqueeze(-1)).view(3, H, W)
        u0 = fx * Xp0[0] / Xp0[2].clamp_min(1e-10) + cx
        v0 = fy * Xp0[1] / Xp0[2].clamp_min(1e-10) + cy

        # pert = R @ (X+dX) + t
        Xp1 = (R @ X_per + t_vec.unsqueeze(-1)).view(3, H, W)
        u1 = fx * Xp1[0] / Xp1[2].clamp_min(1e-10) + cx
        v1 = fy * Xp1[1] / Xp1[2].clamp_min(1e-10) + cy

        J_fd[0, comp] = (u1 - u0) / eps
        J_fd[1, comp] = (v1 - v0) / eps

    diff = (J_3d[0] - J_fd).abs()
    print(f"J_3d FD: max diff={diff.max():.10f}")
    assert diff.max() < 1e-4, f"J_3d mismatch! max diff={diff.max():.2e}"


def test_jacobian_3d_identity_pose():
    """Identity pose: f_rigid=0 but J_3d!=0 (projection IS nonlinear in X)."""
    from Train.MatchingNet.loss import compute_rigid_flow_jacobian_3d

    B, H, W = 1, 15, 20
    dtype = torch.float64
    K = torch.tensor([[[100., 0, 10.], [0, 100., 7.5], [0, 0, 1.]]], dtype=dtype)
    depth = torch.ones(B, 1, H, W, dtype=dtype) * 3.0
    T_id = torch.eye(4, dtype=dtype).unsqueeze(0)

    f_rigid, J_3d, _ = compute_rigid_flow_jacobian_3d(depth, K, T_id)
    assert f_rigid.abs().max() < 1e-10, "identity pose must give zero rigid flow"
    assert J_3d.abs().max() > 0, "J_3d should be non-zero (projection is nonlinear in X)"

    # At optical center, J_3d[0, 0, cy, cx] = fx/depth (moving in X shifts u by fx/d)
    # and J_3d[0, 2, cy, cx] = 0 (moving along ray doesn't change projection at identity)
    cy_i, cx_i = 7, 10  # (cx=10, cy=7.5)
    assert abs(J_3d[0, 0, 0, cy_i, cx_i] - K[0,0,0].item() / 3.0) < 1e-10  # fx/d
    assert abs(J_3d[0, 0, 2, cy_i, cx_i]) < 1e-10  # Z component near 0 at opt center


if __name__ == "__main__":
    test_jacobian_identity_pose()
    test_jacobian_pure_translation()
    test_jacobian_finite_difference()
    test_jacobian_3d_identity_pose()
    test_jacobian_3d_finite_difference()
    print("\nAll tests passed.")
