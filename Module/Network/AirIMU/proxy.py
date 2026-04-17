from __future__ import annotations

import torch


def _pixel_grid(B: int, H: int, W: int, device: torch.device, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
    yy, xx = torch.meshgrid(
        torch.arange(H, device=device, dtype=dtype),
        torch.arange(W, device=device, dtype=dtype),
        indexing="ij",
    )
    x = xx.view(1, 1, H, W).repeat(B, 1, 1, 1)
    y = yy.view(1, 1, H, W).repeat(B, 1, 1, 1)
    return x, y


def _extract_intrinsics(K: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if K.ndim == 2:
        K = K.unsqueeze(0)
    fx = K[:, 0, 0]
    fy = K[:, 1, 1]
    cx = K[:, 0, 2]
    cy = K[:, 1, 2]
    return fx, fy, cx, cy


def build_imu_proxy(
    depth_t: torch.Tensor,
    K: torch.Tensor,
    flow_obs: torch.Tensor,
    delta_R_cam: torch.Tensor,
    delta_p_cam: torch.Tensor,
    Sigma_imu: torch.Tensor,
    cfg,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Construct 4-channel IMU proxy:
      [r_imu_norm, ||f_rigid_imu||, ||f_obs||, valid]
    """
    B, _, H, W = depth_t.shape
    device, dtype = depth_t.device, depth_t.dtype
    x, y = _pixel_grid(B, H, W, device, dtype)
    fx, fy, cx, cy = _extract_intrinsics(K.to(device=device, dtype=dtype))

    fxv = fx.view(B, 1, 1, 1)
    fyv = fy.view(B, 1, 1, 1)
    cxv = cx.view(B, 1, 1, 1)
    cyv = cy.view(B, 1, 1, 1)

    z = depth_t
    X = torch.cat(
        [
            (x - cxv) / fxv * z,
            (y - cyv) / fyv * z,
            z,
        ],
        dim=1,
    )
    X_flat = X.reshape(B, 3, -1)

    X1 = torch.bmm(delta_R_cam, X_flat) + delta_p_cam.unsqueeze(-1)
    X1 = X1.reshape(B, 3, H, W)

    z1 = X1[:, 2:3]
    x1 = fxv * (X1[:, 0:1] / z1.clamp(min=1e-6)) + cxv
    y1 = fyv * (X1[:, 1:2] / z1.clamp(min=1e-6)) + cyv

    f_rigid_x = x1 - x
    f_rigid_y = y1 - y
    f_rigid = torch.cat([f_rigid_x, f_rigid_y], dim=1)

    delta_f = flow_obs - f_rigid
    eps = 1e-8
    f_obs_norm = torch.linalg.vector_norm(flow_obs, dim=1, keepdim=True)
    f_rigid_norm = torch.linalg.vector_norm(f_rigid, dim=1, keepdim=True)
    delta_norm = torch.linalg.vector_norm(delta_f, dim=1, keepdim=True)

    sigma_u2 = Sigma_imu[:, 6, 6].view(B, 1, 1, 1)
    sigma_v2 = Sigma_imu[:, 7, 7].view(B, 1, 1, 1)
    sigma_px = torch.sqrt((sigma_u2 + sigma_v2).clamp(min=1e-12)) * fxv.clamp(min=1e-12)

    tau0 = float(getattr(cfg, "tau0", 0.5))
    alpha = float(getattr(cfg, "alpha", 0.25))
    tau = tau0 + alpha * sigma_px
    r_imu_norm = delta_norm / (tau + eps)

    d_min = float(getattr(cfg, "d_min", 0.1))
    d_max = float(getattr(cfg, "d_max", 80.0))
    eps_z = float(getattr(cfg, "eps_z", 1e-6))
    edgewidth = int(getattr(cfg, "edgewidth", 0))

    in_depth = (z >= d_min) & (z <= d_max)
    in_front = X1[:, 0:1] > eps_z  # NED forward axis (x)
    in_img = (x1 >= edgewidth) & (x1 < (W - edgewidth)) & (y1 >= edgewidth) & (y1 < (H - edgewidth))

    sigma_rot_cap = float(getattr(cfg, "sigma_rot_cap", 5.0))
    sigma_pos_cap = float(getattr(cfg, "sigma_pos_cap", 10.0))
    bad_sigma = (torch.diagonal(Sigma_imu[:, :3, :3], dim1=-2, dim2=-1).max(dim=1).values > sigma_rot_cap) | (
        torch.diagonal(Sigma_imu[:, 6:9, 6:9], dim1=-2, dim2=-1).max(dim=1).values > sigma_pos_cap
    )
    sigma_valid = (~bad_sigma).view(B, 1, 1, 1)

    valid = (in_depth & in_front & in_img & sigma_valid).to(dtype)

    r_imu_norm = r_imu_norm * valid
    f_rigid_norm = f_rigid_norm * valid
    f_obs_norm = f_obs_norm * valid

    proxy = torch.cat([r_imu_norm, f_rigid_norm, f_obs_norm, valid], dim=1)
    return proxy, valid

