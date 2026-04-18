from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import pypose as pp
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from DataLoader import DataFramePair, StereoInertialFrame
from Module.Network.AirIMU import build_imu_proxy
from Train.DynamicHead.loop import DynamicHeadTrainer
from Train.DynamicHead.loss import confidence_targets, cycle_consistency_mask, rigid_flow_from_motion, relative_cam_motion_from_gt
from Utility.Math import body2cam_se3


def _cfg_get(cfg, key: str, default):
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


@dataclass
class CalibrationResult:
    temperature: float
    temperature_visible: float | None
    nll: float
    ece: float


def _ece_binary(logits: torch.Tensor, labels: torch.Tensor, bins: int = 15) -> float:
    probs = torch.sigmoid(logits)
    bin_edges = torch.linspace(0.0, 1.0, bins + 1, device=probs.device)
    ece = torch.tensor(0.0, device=probs.device)
    for i in range(bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        m = (probs >= lo) & (probs < hi)
        if m.any():
            conf = probs[m].mean()
            acc = (probs[m] >= 0.5).float().eq(labels[m] >= 0.5).float().mean()
            ece = ece + (m.float().mean() * (conf - acc).abs())
    return float(ece.item())


@torch.no_grad()
def calibrate_temperature(
    trainer: DynamicHeadTrainer,
    loader: DataLoader[DataFramePair[StereoInertialFrame]],
    t_min: float = 0.5,
    t_max: float = 5.0,
    t_steps: int = 40,
    max_batches: int | None = None,
) -> CalibrationResult:
    logits_all: list[torch.Tensor] = []
    labels_all: list[torch.Tensor] = []
    vis_logits_all: list[torch.Tensor] = []
    vis_labels_all: list[torch.Tensor] = []

    loss_cfg = _cfg_get(trainer.cfg, "loss", {})
    use_cycle = bool(_cfg_get(trainer.cfg, "use_cycle", True)) and bool(_cfg_get(loss_cfg, "use_cycle", True))

    for bi, pair in enumerate(loader):
        if max_batches is not None and bi >= max_batches:
            break

        B = pair.cur.stereo.imageL.shape[0]
        depth_out_t = trainer.frontend.estimate_depth(pair.cur.stereo)
        _, match_out = trainer.frontend.estimate_pair(pair.cur.stereo, pair.nxt.stereo)
        flow_bwd = None
        if use_cycle:
            _, match_bwd = trainer.frontend.estimate_pair(pair.nxt.stereo, pair.cur.stereo)
            flow_bwd = match_bwd.flow.to(trainer.device)
        phi8 = trainer._extract_context(B).to(trainer.device, dtype=torch.float32)

        depth_t = depth_out_t.depth.to(trainer.device, dtype=torch.float32)
        flow_fwd = match_out.flow.to(trainer.device, dtype=torch.float32)
        h4 = torch.nn.functional.avg_pool2d(pair.cur.stereo.imageL.to(trainer.device, dtype=torch.float32), kernel_size=4, stride=4)

        imu_seq, bias_ref = trainer._prepare_imu(pair)
        imu_ctx, f_imu = trainer.imu_encoder(imu_seq, bias_ref)
        pre = imu_ctx["preint"]
        dR_cam_imu, dp_cam_imu = body2cam_se3(pre.delta_R, pre.delta_p, pair.cur.stereo.T_BS.to(trainer.device))
        proxy, _ = build_imu_proxy(
            depth_t=depth_t,
            K=pair.cur.stereo.K.to(trainer.device),
            flow_obs=flow_fwd,
            delta_R_cam=dR_cam_imu,
            delta_p_cam=dp_cam_imu,
            Sigma_imu=pre.Sigma,
            cfg=_cfg_get(trainer.cfg, "proxy", {}),
        )
        gt_pose_t = pair.cur.gt_pose
        gt_pose_t1 = pair.nxt.gt_pose
        if gt_pose_t is None or gt_pose_t1 is None:
            continue
        dR_cam, dp_cam = relative_cam_motion_from_gt(
            cast(pp.LieTensor, pp.SE3(gt_pose_t).to(trainer.device)),
            cast(pp.LieTensor, pp.SE3(gt_pose_t1).to(trainer.device)),
            pair.cur.stereo.T_BS.to(trainer.device),
        )
        flow_rigid, geom_valid = rigid_flow_from_motion(depth_t, pair.cur.stereo.K.to(trainer.device), dR_cam.to(depth_t), dp_cam.to(depth_t))
        c_star, valid = confidence_targets(
            flow_obs=flow_fwd,
            flow_rigid=flow_rigid,
            valid=geom_valid,
            tau0=float(_cfg_get(loss_cfg, "tau0", 0.5)),
            alpha=float(_cfg_get(loss_cfg, "alpha", 0.25)),
            kappa=float(_cfg_get(loss_cfg, "kappa", 1.5)),
        )
        if flow_bwd is not None:
            valid = valid & cycle_consistency_mask(flow_fwd, flow_bwd, tau=float(_cfg_get(loss_cfg, "tau_cyc_valid", 2.0)))

        head_out = trainer.head(phi8=phi8, h4=h4, z_hat=depth_t, e_raw=proxy[:, :1], f_imu=f_imu, h8_prev=None, return_probs=True)
        logits = head_out.logits_4[:, :1]
        c_star_ds = F.interpolate(c_star, size=logits.shape[-2:], mode="bilinear", align_corners=False)
        valid_ds = F.interpolate(valid.float(), size=logits.shape[-2:], mode="nearest") > 0.5
        if not valid_ds.any():
            continue
        logits_all.append(logits[valid_ds].flatten())
        labels_all.append(c_star_ds[valid_ds].flatten())
        if head_out.logits_4.shape[1] > 1:
            vis_logits_all.append(head_out.logits_4[:, 1:2].flatten())
            vis_labels_all.append(valid_ds.float().flatten())

    if not logits_all:
        return CalibrationResult(temperature=1.0, temperature_visible=None, nll=0.0, ece=0.0)

    logits_cat = torch.cat(logits_all, dim=0)
    labels_cat = torch.cat(labels_all, dim=0)
    T_grid = torch.linspace(t_min, t_max, t_steps, device=logits_cat.device)
    nlls = torch.stack([F.binary_cross_entropy_with_logits(logits_cat / t, labels_cat) for t in T_grid])
    i_best = int(torch.argmin(nlls).item())
    best_t = float(T_grid[i_best].item())
    temperature = trainer.head.get_buffer("temperature")
    temperature.copy_(torch.tensor(best_t, device=temperature.device, dtype=temperature.dtype))

    best_t_vis = None
    if vis_logits_all:
        v_logits = torch.cat(vis_logits_all, dim=0)
        v_labels = torch.cat(vis_labels_all, dim=0)
        nlls_vis = torch.stack([F.binary_cross_entropy_with_logits(v_logits / t, v_labels) for t in T_grid])
        i_vis = int(torch.argmin(nlls_vis).item())
        best_t_vis = float(T_grid[i_vis].item())
        temperature_visible = trainer.head.get_buffer("temperature_visible")
        temperature_visible.copy_(
            torch.tensor(best_t_vis, device=temperature_visible.device, dtype=temperature_visible.dtype)
        )

    ece = _ece_binary(logits_cat / temperature, labels_cat)
    return CalibrationResult(
        temperature=float(temperature.item()),
        temperature_visible=best_t_vis,
        nll=float(nlls[i_best].item()),
        ece=ece,
    )
