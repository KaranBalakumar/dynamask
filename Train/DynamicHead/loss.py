from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pypose as pp
import torch
import torch.nn.functional as F

from Utility.Math import body2cam_se3


def _cfg_get(cfg: SimpleNamespace | dict[str, Any], key: str, default: Any) -> Any:
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


def _pixel_grid(B: int, H: int, W: int, device: torch.device, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
    yy, xx = torch.meshgrid(
        torch.arange(H, device=device, dtype=dtype),
        torch.arange(W, device=device, dtype=dtype),
        indexing="ij",
    )
    x = xx.view(1, 1, H, W).repeat(B, 1, 1, 1)
    y = yy.view(1, 1, H, W).repeat(B, 1, 1, 1)
    return x, y


def rigid_flow_from_motion(depth: torch.Tensor, K: torch.Tensor, R: torch.Tensor, t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    B, _, H, W = depth.shape
    x, y = _pixel_grid(B, H, W, device=depth.device, dtype=depth.dtype)
    fx = K[:, 0, 0].view(B, 1, 1, 1)
    fy = K[:, 1, 1].view(B, 1, 1, 1)
    cx = K[:, 0, 2].view(B, 1, 1, 1)
    cy = K[:, 1, 2].view(B, 1, 1, 1)

    z = depth.clamp(min=1e-6)
    X = torch.cat(
        [
            (x - cx) / fx * z,
            (y - cy) / fy * z,
            z,
        ],
        dim=1,
    )
    Xf = X.reshape(B, 3, -1)
    X1 = torch.bmm(R, Xf) + t.unsqueeze(-1)
    X1 = X1.reshape(B, 3, H, W)
    z1 = X1[:, 2:3]
    u1 = fx * (X1[:, 0:1] / z1.clamp(min=1e-6)) + cx
    v1 = fy * (X1[:, 1:2] / z1.clamp(min=1e-6)) + cy
    flow = torch.cat([u1 - x, v1 - y], dim=1)
    in_image = (u1 >= 0.0) & (u1 <= (W - 1)) & (v1 >= 0.0) & (v1 <= (H - 1))
    valid = (depth > 1e-6) & (z1 > 1e-6) & in_image
    return flow, valid


def relative_cam_motion_from_gt(
    gt_pose_t: pp.LieTensor,
    gt_pose_t1: pp.LieTensor,
    T_BS: pp.LieTensor | torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    T_rel_body = gt_pose_t.Inv() @ gt_pose_t1
    dR_body = T_rel_body.rotation().matrix()
    dp_body = T_rel_body.translation()
    return body2cam_se3(dR_body, dp_body, T_BS)


def cycle_consistency_mask(flow_fwd: torch.Tensor, flow_bwd: torch.Tensor, tau: float) -> torch.Tensor:
    B, _, H, W = flow_fwd.shape
    x, y = _pixel_grid(B, H, W, flow_fwd.device, flow_fwd.dtype)
    u = x + flow_fwd[:, :1]
    v = y + flow_fwd[:, 1:2]
    u_norm = 2.0 * (u / max(W - 1, 1)) - 1.0
    v_norm = 2.0 * (v / max(H - 1, 1)) - 1.0
    grid = torch.stack([u_norm.squeeze(1), v_norm.squeeze(1)], dim=-1)
    bwd_warp = F.grid_sample(flow_bwd, grid, mode="bilinear", padding_mode="zeros", align_corners=True)
    err = torch.linalg.vector_norm(flow_fwd + bwd_warp, dim=1, keepdim=True)
    in_image = (u >= 0.0) & (u <= (W - 1)) & (v >= 0.0) & (v <= (H - 1))
    return (err <= tau) & in_image


def confidence_targets(
    flow_obs: torch.Tensor,
    flow_rigid: torch.Tensor,
    valid: torch.Tensor,
    tau0: float,
    alpha: float,
    kappa: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    resid = torch.linalg.vector_norm(flow_obs - flow_rigid, dim=1, keepdim=True)
    rigid_mag = torch.linalg.vector_norm(flow_rigid, dim=1, keepdim=True)
    tau = tau0 + alpha * rigid_mag
    dyn_prob = torch.sigmoid((resid - tau) / max(kappa, 1e-6))
    c_star = 1.0 - dyn_prob
    return c_star.clamp(0.0, 1.0), valid


def _tv_loss(x: torch.Tensor) -> torch.Tensor:
    dx = (x[..., :, 1:] - x[..., :, :-1]).abs().mean()
    dy = (x[..., 1:, :] - x[..., :-1, :]).abs().mean()
    return dx + dy


def compute_dynamic_head_loss(
    logits_4: torch.Tensor,
    c_pred: torch.Tensor,
    flow_fwd: torch.Tensor,
    flow_bwd: torch.Tensor | None,
    depth_t1: torch.Tensor,
    K: torch.Tensor,
    gt_pose_t: pp.LieTensor,
    gt_pose_t1: pp.LieTensor,
    T_BS: pp.LieTensor | torch.Tensor,
    cfg: SimpleNamespace | dict[str, Any],
) -> tuple[torch.Tensor, dict[str, float]]:
    dR_cam, dp_cam = relative_cam_motion_from_gt(gt_pose_t, gt_pose_t1, T_BS)
    flow_rigid, geom_valid = rigid_flow_from_motion(depth_t1, K, dR_cam.to(depth_t1), dp_cam.to(depth_t1))
    c_star, valid = confidence_targets(
        flow_obs=flow_fwd,
        flow_rigid=flow_rigid,
        valid=geom_valid,
        tau0=float(_cfg_get(cfg, "tau0", 0.5)),
        alpha=float(_cfg_get(cfg, "alpha", 0.25)),
        kappa=float(_cfg_get(cfg, "kappa", 1.5)),
    )
    if flow_bwd is not None and bool(_cfg_get(cfg, "use_cycle", True)):
        cyc = cycle_consistency_mask(
            flow_fwd=flow_fwd,
            flow_bwd=flow_bwd,
            tau=float(_cfg_get(cfg, "tau_cyc_valid", 2.0)),
        )
        valid = valid & cyc

    valid_fine = valid.float()
    valid_mass_ds = F.interpolate(valid_fine, size=logits_4.shape[-2:], mode="bilinear", align_corners=False)
    c_star_num_ds = F.interpolate(c_star * valid_fine, size=logits_4.shape[-2:], mode="bilinear", align_corners=False)
    c_star_ds = c_star_num_ds / valid_mass_ds.clamp(min=1e-6)
    valid_ds = valid_mass_ds > 1e-6

    valid_f = valid_ds.float()
    if logits_4.shape[1] == 1:
        loss_map = F.binary_cross_entropy_with_logits(logits_4, c_star_ds, reduction="none")
        loss_dyn = (loss_map * valid_f).sum() / valid_f.sum().clamp(min=1.0)
        loss_visible = torch.tensor(0.0, device=logits_4.device, dtype=logits_4.dtype)
    else:
        static_loss = F.binary_cross_entropy_with_logits(logits_4[:, :1], c_star_ds, reduction="none")
        vis_target = valid_ds.float()
        vis_loss = F.binary_cross_entropy_with_logits(logits_4[:, 1:2], vis_target, reduction="none")
        loss_static = (static_loss * valid_f).sum() / valid_f.sum().clamp(min=1.0)
        loss_visible = vis_loss.mean()
        loss_dyn = loss_static + float(_cfg_get(cfg, "lambda_visible", 0.2)) * loss_visible

    loss_smooth = _tv_loss(c_pred)
    lambda_smooth = float(_cfg_get(cfg, "lambda_smooth", 0.03))
    total_loss = loss_dyn + lambda_smooth * loss_smooth

    with torch.no_grad():
        c_star_pred = F.interpolate(c_pred, size=c_star.shape[-2:], mode="bilinear", align_corners=False)
        mae = (c_star_pred - c_star).abs()[valid].mean() if valid.any() else torch.tensor(0.0, device=c_pred.device)
        hard_pred = (c_star_pred >= 0.5).float()
        hard_gt = (c_star >= 0.5).float()
        acc = (hard_pred == hard_gt)[valid].float().mean() if valid.any() else torch.tensor(0.0, device=c_pred.device)
        metrics = {
            "loss_total": float(total_loss.item()),
            "loss_dyn": float(loss_dyn.item()),
            "loss_visible": float(loss_visible.item()),
            "loss_smooth": float(loss_smooth.item()),
            "valid_ratio": float(valid.float().mean().item()),
            "target_mean": float(c_star[valid].mean().item()) if valid.any() else 0.0,
            "pred_mean": float(c_star_pred[valid].mean().item()) if valid.any() else 0.0,
            "mae": float(mae.item()),
            "acc@0.5": float(acc.item()),
        }
    return total_loss, metrics
