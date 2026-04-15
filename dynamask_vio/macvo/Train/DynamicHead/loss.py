from __future__ import annotations

from types import SimpleNamespace

import torch
import torch.nn.functional as F


def _pixel_grid(h: int, w: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    y, x = torch.meshgrid(
        torch.arange(h, device=device, dtype=dtype),
        torch.arange(w, device=device, dtype=dtype),
        indexing="ij",
    )
    return torch.stack([x, y], dim=0).unsqueeze(0)


def _masked_mean(x: torch.Tensor, mask: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    m = mask.to(dtype=x.dtype)
    return (x * m).sum() / m.sum().clamp(min=eps)


def _focal_bce_prob(pred: torch.Tensor, target: torch.Tensor, gamma: float = 2.0, eps: float = 1e-6) -> torch.Tensor:
    p = pred.clamp(min=eps, max=1.0 - eps)
    bce = -(target * torch.log(p) + (1.0 - target) * torch.log(1.0 - p))
    p_t = target * p + (1.0 - target) * (1.0 - p)
    return ((1.0 - p_t).clamp(min=eps).pow(gamma)) * bce


def _edge_aware_smoothness(c_pred: torch.Tensor, image: torch.Tensor) -> torch.Tensor:
    grad_c_x = (c_pred[:, :, :, 1:] - c_pred[:, :, :, :-1]).abs()
    grad_c_y = (c_pred[:, :, 1:, :] - c_pred[:, :, :-1, :]).abs()
    grad_i_x = (image[:, :, :, 1:] - image[:, :, :, :-1]).abs().mean(dim=1, keepdim=True)
    grad_i_y = (image[:, :, 1:, :] - image[:, :, :-1, :]).abs().mean(dim=1, keepdim=True)
    return 0.5 * ((grad_c_x * torch.exp(-grad_i_x)).mean() + (grad_c_y * torch.exp(-grad_i_y)).mean())


def rigid_flow(depth: torch.Tensor, K: torch.Tensor, R: torch.Tensor, t: torch.Tensor, eps: float = 1e-6):
    b, _, h, w = depth.shape
    grid = _pixel_grid(h, w, depth.device, depth.dtype).expand(b, -1, -1, -1)
    u, v = grid[:, 0], grid[:, 1]

    fx = K[:, 0, 0].view(b, 1, 1)
    fy = K[:, 1, 1].view(b, 1, 1)
    cx = K[:, 0, 2].view(b, 1, 1)
    cy = K[:, 1, 2].view(b, 1, 1)
    z = depth[:, 0].clamp(min=eps)
    x = (u - cx) / fx.clamp(min=eps) * z
    y = (v - cy) / fy.clamp(min=eps) * z

    pts = torch.stack([x, y, z], dim=1).view(b, 3, -1)
    pts_t = (R @ pts) + t.view(b, 3, 1)
    x_t = pts_t[:, 0].view(b, h, w)
    y_t = pts_t[:, 1].view(b, h, w)
    z_t = pts_t[:, 2].view(b, h, w).clamp(min=eps)

    u_t = fx * (x_t / z_t) + cx
    v_t = fy * (y_t / z_t) + cy
    uv_t = torch.stack([u_t, v_t], dim=1)
    flow = uv_t - grid

    m_infov = (u_t >= 0) & (u_t <= (w - 1)) & (v_t >= 0) & (v_t <= (h - 1))
    m_depth = depth[:, 0] > eps
    return flow, m_infov.unsqueeze(1), m_depth.unsqueeze(1)


def depth_adaptive_tau(depth: torch.Tensor, K: torch.Tensor, t_gt: torch.Tensor, tau0: float, alpha: float, eps: float = 1e-6):
    focal = 0.5 * (K[:, 0, 0] + K[:, 1, 1])
    t_norm = t_gt.norm(dim=-1)
    return tau0 + alpha * (focal.view(-1, 1, 1, 1) / depth.clamp(min=eps)) * t_norm.view(-1, 1, 1, 1)


def warp_with_flow(x: torch.Tensor, flow: torch.Tensor) -> torch.Tensor:
    b, _, h, w = x.shape
    grid = _pixel_grid(h, w, flow.device, flow.dtype).expand(b, -1, -1, -1)
    uv = grid + flow
    x_norm = 2.0 * (uv[:, 0] / max(w - 1, 1)) - 1.0
    y_norm = 2.0 * (uv[:, 1] / max(h - 1, 1)) - 1.0
    sample_grid = torch.stack([x_norm, y_norm], dim=-1)
    return F.grid_sample(x, sample_grid, mode="bilinear", padding_mode="zeros", align_corners=True)


def cycle_error(flow_fwd: torch.Tensor, flow_bwd: torch.Tensor) -> torch.Tensor:
    return (flow_fwd + warp_with_flow(flow_bwd, flow_fwd)).norm(dim=1, keepdim=True)


def compose_total_loss(
    *,
    c_pred: torch.Tensor,
    flow: torch.Tensor,
    depth: torch.Tensor,
    R_gt: torch.Tensor,
    t_gt: torch.Tensor,
    image: torch.Tensor,
    K: torch.Tensor,
    cfg: SimpleNamespace,
    flow_bwd: torch.Tensor | None = None,
    valid_depth_mask: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    f_rigid_gt, m_infov, m_depth = rigid_flow(depth, K, R_gt, t_gt)
    residual = (flow - f_rigid_gt).norm(dim=1, keepdim=True)
    tau = depth_adaptive_tau(depth, K, t_gt, tau0=float(cfg.dyn.tau0), alpha=float(cfg.dyn.alpha))

    d_soft = torch.sigmoid((residual - tau) / max(float(cfg.dyn.kappa), 1e-6))
    c_target = 1.0 - d_soft

    m_depth = m_depth & (depth > float(cfg.dyn.d_min)) & (depth < float(cfg.dyn.d_max))
    if flow_bwd is not None:
        cyc_err = cycle_error(flow, flow_bwd)
        m_cycle = cyc_err < float(cfg.dyn.tau_cyc_valid)
    else:
        cyc_err = torch.zeros_like(c_pred)
        m_cycle = torch.ones_like(c_pred, dtype=torch.bool)
    m_valid = m_infov & m_depth & m_cycle
    if valid_depth_mask is not None:
        m_valid = m_valid & valid_depth_mask.bool()

    out: dict[str, torch.Tensor] = {}
    total = torch.zeros((), device=c_pred.device, dtype=c_pred.dtype)

    if bool(cfg.dyn.enabled):
        l_dyn_map = _focal_bce_prob(c_pred, c_target, gamma=float(cfg.dyn.focal_gamma))
        l_dyn = _masked_mean(l_dyn_map, m_valid)
        total = total + float(cfg.dyn.weight) * l_dyn
        out["L_dyn"] = l_dyn

    if bool(cfg.smooth.enabled):
        l_smooth = _edge_aware_smoothness(c_pred, image)
        total = total + float(cfg.smooth.weight) * l_smooth
        out["L_smooth"] = l_smooth

    if bool(cfg.cyc.enabled):
        if flow_bwd is None:
            raise ValueError("loss.cyc.enabled=True requires backward flow for cycle supervision")
        c_cyc_target = 1.0 - torch.sigmoid(
            (cyc_err - float(cfg.cyc.tau_cyc_bce)) / max(float(cfg.cyc.kappa_cyc), 1e-6)
        )
        m_cyc_valid = m_infov & m_depth
        l_cyc_map = F.binary_cross_entropy(c_pred.clamp(1e-6, 1 - 1e-6), c_cyc_target, reduction="none")
        l_cyc = _masked_mean(l_cyc_map, m_cyc_valid)
        total = total + float(cfg.cyc.weight) * l_cyc
        out["L_cyc"] = l_cyc

    out["L_total"] = total
    out["residual_mean"] = _masked_mean(residual, m_infov & m_depth)
    out["c_target"] = c_target.detach()
    out["M_valid"] = m_valid.detach()
    return out


def compose_total_loss_v2(
    *,
    p_static: torch.Tensor,
    p_visible: torch.Tensor,
    c_eff: torch.Tensor,
    flow: torch.Tensor,
    depth: torch.Tensor,
    R_gt: torch.Tensor,
    t_gt: torch.Tensor,
    image: torch.Tensor,
    K: torch.Tensor,
    cfg: SimpleNamespace,
    flow_bwd: torch.Tensor,
    valid_depth_mask: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    f_rigid_gt, m_infov, m_depth = rigid_flow(depth, K, R_gt, t_gt)
    residual = (flow - f_rigid_gt).norm(dim=1, keepdim=True)
    tau = depth_adaptive_tau(depth, K, t_gt, tau0=float(cfg.static.tau0), alpha=float(cfg.static.alpha))
    c_static_target = 1.0 - torch.sigmoid((residual - tau) / max(float(cfg.static.kappa), 1e-6))

    cyc_err = cycle_error(flow, flow_bwd)
    c_visible_target = 1.0 - torch.sigmoid(
        (cyc_err - float(cfg.visible.tau_cyc_bce)) / max(float(cfg.visible.kappa_cyc), 1e-6)
    )
    c_joint_target = c_static_target * c_visible_target
    m_valid = m_infov & m_depth
    if valid_depth_mask is not None:
        m_valid = m_valid & valid_depth_mask.bool()

    l_static = _masked_mean(_focal_bce_prob(p_static, c_static_target, gamma=float(cfg.static.focal_gamma)), m_valid)
    l_visible = _masked_mean(
        F.binary_cross_entropy(p_visible.clamp(1e-6, 1 - 1e-6), c_visible_target, reduction="none"),
        m_valid,
    )
    l_joint = _masked_mean(
        F.binary_cross_entropy(c_eff.clamp(1e-6, 1 - 1e-6), c_joint_target, reduction="none"),
        m_valid,
    )
    l_smooth = _edge_aware_smoothness(c_eff, image)

    total = (
        float(cfg.static.weight) * l_static
        + float(cfg.visible.weight) * l_visible
        + float(cfg.joint.weight) * l_joint
        + float(cfg.smooth.weight) * l_smooth
    )
    return {
        "L_static": l_static,
        "L_visible": l_visible,
        "L_joint": l_joint,
        "L_smooth": l_smooth,
        "L_total": total,
        "c_joint_target": c_joint_target.detach(),
        "M_valid": m_valid.detach(),
    }
