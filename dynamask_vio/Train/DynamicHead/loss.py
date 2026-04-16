from __future__ import annotations

import torch
import torch.nn.functional as F


def _make_pixel_grid(height: int, width: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    ys, xs = torch.meshgrid(
        torch.arange(height, device=device, dtype=dtype),
        torch.arange(width, device=device, dtype=dtype),
        indexing="ij",
    )
    return torch.stack([xs, ys], dim=0).unsqueeze(0)


def _backproject(depth: torch.Tensor, K: torch.Tensor) -> torch.Tensor:
    b, _, h, w = depth.shape
    grid = _make_pixel_grid(h, w, depth.device, depth.dtype).repeat(b, 1, 1, 1)
    fx = K[:, 0, 0].view(b, 1, 1, 1)
    fy = K[:, 1, 1].view(b, 1, 1, 1)
    cx = K[:, 0, 2].view(b, 1, 1, 1)
    cy = K[:, 1, 2].view(b, 1, 1, 1)

    x = (grid[:, 0:1] - cx) / fx * depth
    y = (grid[:, 1:2] - cy) / fy * depth
    z = depth
    return torch.cat([x, y, z], dim=1)


def _project(xyz: torch.Tensor, K: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    b = xyz.size(0)
    x, y, z = xyz[:, 0:1], xyz[:, 1:2], xyz[:, 2:3].clamp(min=1e-6)
    fx = K[:, 0, 0].view(b, 1, 1, 1)
    fy = K[:, 1, 1].view(b, 1, 1, 1)
    cx = K[:, 0, 2].view(b, 1, 1, 1)
    cy = K[:, 1, 2].view(b, 1, 1, 1)
    u = fx * (x / z) + cx
    v = fy * (y / z) + cy
    return u, v


def _rigid_flow(depth: torch.Tensor, K: torch.Tensor, R_gt: torch.Tensor, t_gt: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    b, _, h, w = depth.shape
    xyz = _backproject(depth, K).reshape(b, 3, -1)
    xyz_t = torch.bmm(R_gt, xyz) + t_gt.unsqueeze(-1)
    xyz_t = xyz_t.reshape(b, 3, h, w)
    u1, v1 = _project(xyz_t, K)
    grid = _make_pixel_grid(h, w, depth.device, depth.dtype).repeat(b, 1, 1, 1)
    flow = torch.cat([u1 - grid[:, 0:1], v1 - grid[:, 1:2]], dim=1)
    in_bounds = (u1 >= 0) & (u1 <= (w - 1)) & (v1 >= 0) & (v1 <= (h - 1))
    return flow, in_bounds


def _flow_to_normalized_grid(flow: torch.Tensor) -> torch.Tensor:
    b, _, h, w = flow.shape
    base = _make_pixel_grid(h, w, flow.device, flow.dtype).repeat(b, 1, 1, 1)
    uv = base + flow
    x = (uv[:, 0] / max(w - 1, 1)) * 2 - 1
    y = (uv[:, 1] / max(h - 1, 1)) * 2 - 1
    return torch.stack([x, y], dim=-1)


def _warp_flow(flow: torch.Tensor, sample_flow: torch.Tensor) -> torch.Tensor:
    grid = _flow_to_normalized_grid(sample_flow)
    return F.grid_sample(flow, grid, mode="bilinear", padding_mode="zeros", align_corners=True)


def _focal_bce(pred: torch.Tensor, target: torch.Tensor, gamma: float, eps: float = 1e-6) -> torch.Tensor:
    pred = pred.clamp(min=eps, max=1.0 - eps)
    bce = -(target * torch.log(pred) + (1.0 - target) * torch.log(1.0 - pred))
    p_t = pred * target + (1.0 - pred) * (1.0 - target)
    return ((1.0 - p_t).pow(gamma)) * bce


def _edge_aware_smoothness(c_eff: torch.Tensor, image: torch.Tensor) -> torch.Tensor:
    grad_c_x = (c_eff[..., :, 1:] - c_eff[..., :, :-1]).abs()
    grad_c_y = (c_eff[..., 1:, :] - c_eff[..., :-1, :]).abs()
    grad_i_x = (image[..., :, 1:] - image[..., :, :-1]).abs().mean(dim=1, keepdim=True)
    grad_i_y = (image[..., 1:, :] - image[..., :-1, :]).abs().mean(dim=1, keepdim=True)
    return 0.5 * ((grad_c_x * torch.exp(-grad_i_x)).mean() + (grad_c_y * torch.exp(-grad_i_y)).mean())


def compose_factorized_loss(
    *,
    p_static: torch.Tensor,
    p_visible: torch.Tensor,
    flow: torch.Tensor,
    depth: torch.Tensor,
    image: torch.Tensor,
    K: torch.Tensor,
    R_gt: torch.Tensor,
    t_gt: torch.Tensor,
    cfg,
    flow_bwd: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    rigid_flow, in_bounds = _rigid_flow(depth=depth, K=K, R_gt=R_gt, t_gt=t_gt)
    residual = torch.norm(flow - rigid_flow, dim=1, keepdim=True)
    trans_norm = t_gt.norm(dim=1, keepdim=True).view(-1, 1, 1, 1)
    fx = K[:, 0, 0].view(-1, 1, 1, 1)
    tau_d = cfg.static.tau0 + cfg.static.alpha * (fx / depth.clamp(min=1e-3)) * trans_norm

    c_static_target = 1.0 - torch.sigmoid((residual - tau_d) / max(float(cfg.static.kappa), 1e-6))
    c_static_target = c_static_target.clamp(0.0, 1.0)

    depth_valid = (depth > cfg.d_min) & (depth < cfg.d_max)
    valid_mask = in_bounds & depth_valid
    valid = valid_mask.float()

    if flow_bwd is not None:
        warp_err = torch.norm(flow + _warp_flow(flow_bwd, flow), dim=1, keepdim=True)
        c_visible_target = 1.0 - torch.sigmoid(
            (warp_err - cfg.visible.tau_cyc_bce) / max(float(cfg.visible.kappa_cyc), 1e-6)
        )
        c_visible_target = c_visible_target.clamp(0.0, 1.0)
    else:
        c_visible_target = torch.ones_like(c_static_target)

    c_eff = (p_static * p_visible).clamp(0.0, 1.0)
    c_joint_target = (c_static_target * c_visible_target).clamp(0.0, 1.0)

    L_static = _focal_bce(p_static, c_static_target, gamma=float(cfg.static.focal_gamma))
    L_static = (L_static * valid).sum() / valid.sum().clamp(min=1.0)

    L_visible = F.binary_cross_entropy(
        p_visible.clamp(1e-6, 1.0 - 1e-6),
        c_visible_target,
        reduction="none",
    )
    L_visible = (L_visible * valid).sum() / valid.sum().clamp(min=1.0)

    L_joint = F.binary_cross_entropy(
        c_eff.clamp(1e-6, 1.0 - 1e-6),
        c_joint_target,
        reduction="none",
    )
    L_joint = (L_joint * valid).sum() / valid.sum().clamp(min=1.0)
    L_smooth = _edge_aware_smoothness(c_eff, image)

    total = torch.zeros((), dtype=p_static.dtype, device=p_static.device)
    if cfg.static.enabled:
        total = total + float(cfg.static.weight) * L_static
    if cfg.visible.enabled:
        total = total + float(cfg.visible.weight) * L_visible
    if cfg.joint.enabled:
        total = total + float(cfg.joint.weight) * L_joint
    if cfg.smooth.enabled:
        total = total + float(cfg.smooth.weight) * L_smooth

    return {
        "loss": total,
        "L_static": L_static,
        "L_visible": L_visible,
        "L_joint": L_joint,
        "L_smooth": L_smooth,
    }
