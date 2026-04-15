from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Iterator

import torch

from Module.Frontend.Frontend import StaticConfidence_FlowFormerCovFrontend
from .loss import compose_total_loss, compose_total_loss_v2


@dataclass
class SequenceWindowSampler:
    dataset_len: int
    window_len: int
    stride: int

    def __iter__(self) -> Iterator[int]:
        yield from range(0, max(self.dataset_len - self.window_len + 1, 0), max(self.stride, 1))


def _build_stereo_frame(batch: dict[str, torch.Tensor], t: int, device: torch.device, *, prev: bool) -> SimpleNamespace:
    img_l = batch["img_prev_l"][:, t] if prev else batch["img_curr_l"][:, t]
    img_r = batch["img_prev_l"][:, t] if prev else batch["img_curr_r"][:, t]
    b, _, h, w = img_l.shape
    stereo = SimpleNamespace(
        imageL=img_l.to(device),
        imageR=img_r.to(device),
        K=batch["K"][:, t].to(device),
        baseline=batch["baseline"][:, t].to(device),
        height=h,
        width=w,
        frame_ns=int(batch["t_prev_ns"][0, t].item() if prev else batch["t_curr_ns"][0, t].item()),
        imu_window=batch["imu_window"][:, t].to(device),
        imu_mask=batch["imu_mask"][:, t].to(device),
    )
    return stereo


def run_window(
    frontend: StaticConfidence_FlowFormerCovFrontend,
    batch: dict[str, torch.Tensor],
    loss_cfg: SimpleNamespace,
    device: torch.device,
) -> tuple[torch.Tensor, dict[str, float]]:
    frontend.reset_stream()
    win_len = batch["img_prev_l"].shape[1]
    losses: list[torch.Tensor] = []
    meter: dict[str, float] = {}
    use_v2_loss = all(hasattr(loss_cfg, k) for k in ("static", "visible", "joint"))

    for t in range(win_len):
        frame_prev = _build_stereo_frame(batch, t, device, prev=True)
        frame_curr = _build_stereo_frame(batch, t, device, prev=False)
        depth_t, match_t = frontend.estimate_pair(frame_prev, frame_curr)

        flow_bwd = None
        if use_v2_loss or bool(loss_cfg.cyc.enabled):
            with torch.no_grad():
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
        losses.append(loss_out["L_total"])
        for k, v in loss_out.items():
            if k in {"c_target", "c_joint_target", "M_valid"}:
                continue
            meter[k] = meter.get(k, 0.0) + float(v.detach().item())

    total = torch.stack(losses).mean()
    denom = float(max(win_len, 1))
    meter = {k: v / denom for k, v in meter.items()}
    return total, meter
