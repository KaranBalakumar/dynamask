from __future__ import annotations

from types import SimpleNamespace

import torch
import torch.nn.functional as F

from Module.Frontend.Frontend import StaticConfidence_FlowFormerCovFrontend
from .loss import compose_total_loss, compose_total_loss_v2
from .loop import _build_stereo_frame


@torch.no_grad()
def fit_temperature(
    frontend: StaticConfidence_FlowFormerCovFrontend,
    loader,
    loss_cfg: SimpleNamespace,
    device: torch.device,
    optimizer_steps: int = 40,
    t_min: float = 0.05,
    t_max: float = 5.0,
) -> float:
    frontend.eval()
    use_v2_loss = all(hasattr(loss_cfg, k) for k in ("static", "visible", "joint"))
    logits_all: list[torch.Tensor] = []
    targets_all: list[torch.Tensor] = []
    masks_all: list[torch.Tensor] = []

    for batch in loader:
        win_len = batch["img_prev_l"].shape[1]
        frontend.reset_stream()
        for t in range(win_len):
            frame_prev = _build_stereo_frame(batch, t, device, prev=True)
            frame_curr = _build_stereo_frame(batch, t, device, prev=False)
            depth_t, match_t = frontend.estimate_pair(frame_prev, frame_curr)

            flow_bwd = None
            if use_v2_loss or bool(loss_cfg.cyc.enabled):
                flow_bwd, _ = frontend.model.inference(frame_curr.imageL, frame_prev.imageL)

            if use_v2_loss:
                loss_out = compose_total_loss_v2(
                    p_static=getattr(match_t, "p_static").unsqueeze(1),
                    p_visible=getattr(match_t, "p_visible").unsqueeze(1),
                    c_eff=getattr(match_t, "static_conf").unsqueeze(1),
                    flow=match_t.flow,
                    depth=depth_t.depth,
                    R_gt=batch["gt_R_rel"][:, t].to(device),
                    t_gt=batch["gt_t_rel"][:, t].to(device),
                    image=frame_curr.imageL,
                    K=batch["K"][:, t].to(device),
                    cfg=loss_cfg,
                    flow_bwd=flow_bwd if flow_bwd is not None else torch.zeros_like(match_t.flow),
                    valid_depth_mask=batch["valid_depth_mask"][:, t].to(device),
                )
                targets_all.append(loss_out["c_joint_target"].detach().cpu())
            else:
                c_pred = getattr(match_t, "static_conf").unsqueeze(1)
                loss_out = compose_total_loss(
                    c_pred=c_pred,
                    flow=match_t.flow,
                    depth=depth_t.depth,
                    R_gt=batch["gt_R_rel"][:, t].to(device),
                    t_gt=batch["gt_t_rel"][:, t].to(device),
                    image=frame_curr.imageL,
                    K=batch["K"][:, t].to(device),
                    cfg=loss_cfg,
                    flow_bwd=flow_bwd,
                    valid_depth_mask=batch["valid_depth_mask"][:, t].to(device),
                )
                targets_all.append(loss_out["c_target"].detach().cpu())
            logits_all.append(getattr(match_t, "static_logit").unsqueeze(1).detach().cpu())
            masks_all.append(loss_out["M_valid"].detach().cpu())

    if not logits_all:
        return float(frontend.dynamic_head.T_calib.item())

    logits = torch.cat([x.reshape(-1) for x in logits_all], dim=0)
    targets = torch.cat([x.reshape(-1) for x in targets_all], dim=0)
    mask = torch.cat([x.reshape(-1) for x in masks_all], dim=0).bool()
    logits = logits[mask]
    targets = targets[mask]
    if logits.numel() < 64:
        return float(frontend.dynamic_head.T_calib.item())

    log_t = torch.tensor([float(torch.log(frontend.dynamic_head.T_calib.clamp(min=1e-4)).item())], requires_grad=True)
    opt = torch.optim.Adam([log_t], lr=0.05)
    for _ in range(max(int(optimizer_steps), 1)):
        opt.zero_grad()
        t = torch.exp(log_t).clamp(min=t_min, max=t_max)
        loss = F.binary_cross_entropy_with_logits(logits / t, targets)
        loss.backward()
        opt.step()

    t_new = float(torch.exp(log_t).clamp(min=t_min, max=t_max).item())
    frontend.dynamic_head.T_calib.fill_(t_new)
    return t_new
