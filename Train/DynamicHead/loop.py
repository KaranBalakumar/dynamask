from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from DataLoader import DataFramePair, DynamicHeadTrainDataset, StereoInertialFrame
from Module.Frontend.Frontend import IFrontend
from Module.Network.AirIMU import IMUEncoder, build_imu_proxy
from Module.Network.DynamicHead import StaticConfidenceHead, build_head
from Train.DynamicHead.checks import assert_module_frozen, assert_module_trainable, freeze_module
from Train.DynamicHead.loss import compute_dynamic_head_loss
from Utility.Math import body2cam_se3


def _cfg_get(cfg: SimpleNamespace | dict[str, Any], key: str, default: Any) -> Any:
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


@dataclass
class TrainBatchResult:
    loss: float
    metrics: dict[str, float]


class DynamicHeadTrainer:
    def __init__(self, cfg: SimpleNamespace | dict[str, Any]) -> None:
        self.cfg = cfg
        self.device = torch.device(str(_cfg_get(cfg, "device", "cpu")))

        fe_cfg = _cfg_get(cfg, "frontend", None)
        if fe_cfg is None:
            raise ValueError("DynamicHead training requires frontend config")
        self.frontend: IFrontend = IFrontend.instantiate(fe_cfg.type, fe_cfg.args)
        if hasattr(self.frontend, "model"):
            freeze_module(self.frontend.model)  # type: ignore[arg-type]

        imu_cfg = _cfg_get(cfg, "imu_encoder", {})
        self.imu_encoder = IMUEncoder(imu_cfg).to(self.device)
        freeze_module(self.imu_encoder.corrector)
        assert_module_frozen(self.imu_encoder.corrector, "IMU corrector")
        assert_module_trainable(self.imu_encoder.head, "IMU feature MLP")

        self.head: StaticConfidenceHead = build_head(_cfg_get(cfg, "dynamic_head", {})).to(self.device)
        assert_module_trainable(self.head, "StaticConfidenceHead")

        params = list(self.head.parameters()) + list(self.imu_encoder.head.parameters())
        lr = float(_cfg_get(cfg, "lr", 3e-4))
        wd = float(_cfg_get(cfg, "weight_decay", 1e-4))
        self.optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=wd)

    def _extract_context(self, batch_size: int) -> torch.Tensor:
        model = getattr(self.frontend, "model", None)
        if model is None or getattr(model, "last_context", None) is None:
            raise RuntimeError("FlowFormerCov context not available; ensure frontend is FlowFormerCovFrontend")
        ctx = model.last_context
        if ctx.shape[0] == 2 * batch_size:
            return ctx[batch_size:].detach()
        if ctx.shape[0] == batch_size:
            return ctx.detach()
        raise RuntimeError(f"Unexpected context shape: {tuple(ctx.shape)} for batch={batch_size}")

    def _prepare_imu(self, pair: DataFramePair[StereoInertialFrame]) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        imu = pair.nxt.imu
        acc = imu.acc.to(self.device)
        gyro = imu.gyro.to(self.device)
        dt = (imu.time_ns[:, 1:, :] - imu.time_ns[:, :-1, :]).to(self.device).float() * 1e-9
        bias_ref = torch.zeros((acc.shape[0], 6), device=self.device, dtype=acc.dtype)
        return {"acc": acc, "gyro": gyro, "dt": dt}, bias_ref

    def train_step(self, pair: DataFramePair[StereoInertialFrame]) -> TrainBatchResult:
        self.optimizer.zero_grad(set_to_none=True)
        B = pair.cur.stereo.imageL.shape[0]

        with torch.inference_mode():
            depth_out_t = self.frontend.estimate_depth(pair.cur.stereo)
            _, match_out = self.frontend.estimate_pair(pair.cur.stereo, pair.nxt.stereo)
            flow_bwd = None
            if bool(_cfg_get(self.cfg, "use_cycle", True)):
                _, match_bwd = self.frontend.estimate_pair(pair.nxt.stereo, pair.cur.stereo)
                flow_bwd = match_bwd.flow.to(self.device)
            phi8 = self._extract_context(B).to(self.device, dtype=torch.float32)

        depth_t = depth_out_t.depth.to(self.device, dtype=torch.float32)
        flow_fwd = match_out.flow.to(self.device, dtype=torch.float32)
        cov = match_out.cov[:, :2].to(self.device, dtype=torch.float32) if match_out.cov is not None else torch.zeros_like(flow_fwd)
        h4 = torch.nn.functional.avg_pool2d(pair.cur.stereo.imageL.to(self.device, dtype=torch.float32), kernel_size=4, stride=4)

        imu_seq, bias_ref = self._prepare_imu(pair)
        imu_ctx, f_imu = self.imu_encoder(imu_seq, bias_ref)
        pre = imu_ctx["preint"]
        dR_cam, dp_cam = body2cam_se3(pre.delta_R, pre.delta_p, pair.cur.stereo.T_BS.to(self.device))
        proxy, _ = build_imu_proxy(
            depth_t=depth_t,
            K=pair.cur.stereo.K.to(self.device),
            flow_obs=flow_fwd,
            delta_R_cam=dR_cam,
            delta_p_cam=dp_cam,
            Sigma_imu=pre.Sigma,
            cfg=_cfg_get(self.cfg, "proxy", {}),
        )

        head_out = self.head(
            phi8=phi8,
            h4=h4,
            z_hat=depth_t,
            e_raw=proxy[:, :1],
            f_imu=f_imu,
            h8_prev=None,
            return_probs=True,
        )
        loss, metrics = compute_dynamic_head_loss(
            logits_4=head_out.logits_4,
            c_pred=head_out.c,
            flow_fwd=flow_fwd,
            flow_bwd=flow_bwd,
            depth_t1=depth_t,
            K=pair.cur.stereo.K.to(self.device, dtype=torch.float32),
            gt_pose_t=pair.cur.gt_pose.to(self.device),
            gt_pose_t1=pair.nxt.gt_pose.to(self.device),
            T_BS=pair.cur.stereo.T_BS.to(self.device),
            cfg=_cfg_get(self.cfg, "loss", {}),
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(list(self.head.parameters()) + list(self.imu_encoder.head.parameters()), max_norm=1.0)
        self.optimizer.step()
        metrics["lr"] = float(self.optimizer.param_groups[0]["lr"])
        return TrainBatchResult(loss=float(loss.item()), metrics=metrics)

    @torch.no_grad()
    def eval_step(self, pair: DataFramePair[StereoInertialFrame]) -> TrainBatchResult:
        B = pair.cur.stereo.imageL.shape[0]
        depth_out_t = self.frontend.estimate_depth(pair.cur.stereo)
        _, match_out = self.frontend.estimate_pair(pair.cur.stereo, pair.nxt.stereo)
        flow_bwd = None
        if bool(_cfg_get(self.cfg, "use_cycle", True)):
            _, match_bwd = self.frontend.estimate_pair(pair.nxt.stereo, pair.cur.stereo)
            flow_bwd = match_bwd.flow.to(self.device)
        phi8 = self._extract_context(B).to(self.device, dtype=torch.float32)

        depth_t = depth_out_t.depth.to(self.device, dtype=torch.float32)
        flow_fwd = match_out.flow.to(self.device, dtype=torch.float32)
        h4 = torch.nn.functional.avg_pool2d(pair.cur.stereo.imageL.to(self.device, dtype=torch.float32), kernel_size=4, stride=4)

        imu_seq, bias_ref = self._prepare_imu(pair)
        imu_ctx, f_imu = self.imu_encoder(imu_seq, bias_ref)
        pre = imu_ctx["preint"]
        dR_cam, dp_cam = body2cam_se3(pre.delta_R, pre.delta_p, pair.cur.stereo.T_BS.to(self.device))
        proxy, _ = build_imu_proxy(
            depth_t=depth_t,
            K=pair.cur.stereo.K.to(self.device),
            flow_obs=flow_fwd,
            delta_R_cam=dR_cam,
            delta_p_cam=dp_cam,
            Sigma_imu=pre.Sigma,
            cfg=_cfg_get(self.cfg, "proxy", {}),
        )
        head_out = self.head(phi8=phi8, h4=h4, z_hat=depth_t, e_raw=proxy[:, :1], f_imu=f_imu, h8_prev=None, return_probs=True)
        loss, metrics = compute_dynamic_head_loss(
            logits_4=head_out.logits_4,
            c_pred=head_out.c,
            flow_fwd=flow_fwd,
            flow_bwd=flow_bwd,
            depth_t1=depth_t,
            K=pair.cur.stereo.K.to(self.device, dtype=torch.float32),
            gt_pose_t=pair.cur.gt_pose.to(self.device),
            gt_pose_t1=pair.nxt.gt_pose.to(self.device),
            T_BS=pair.cur.stereo.T_BS.to(self.device),
            cfg=_cfg_get(self.cfg, "loss", {}),
        )
        return TrainBatchResult(loss=float(loss.item()), metrics=metrics)


def build_dataloader(cfg: SimpleNamespace | dict[str, Any], shuffle: bool) -> DataLoader[DataFramePair[StereoInertialFrame]]:
    dataset = DynamicHeadTrainDataset(_cfg_get(cfg, "dataset"))
    return DataLoader(
        dataset,
        batch_size=int(_cfg_get(cfg, "batch_size", 1)),
        shuffle=shuffle,
        num_workers=int(_cfg_get(cfg, "num_workers", 0)),
        collate_fn=DataFramePair.collate,
        drop_last=bool(_cfg_get(cfg, "drop_last", True)),
    )

