from __future__ import annotations

import torch
import typing as T
import pypose as pp

from dataclasses import dataclass
from types import SimpleNamespace

from ..Interface import DataFramePair, StereoInertialFrame
from ..SequenceBase import SequenceBase
from .VIODE import compute_camera_relative_pose


@dataclass(kw_only=True)
class DynamicHeadTrainPair(DataFramePair[StereoInertialFrame]):
    gt_R_rel: torch.Tensor
    gt_t_rel: torch.Tensor
    imu_window: torch.Tensor
    imu_mask: torch.Tensor


class DynamicHeadTrainDataset(SequenceBase[DynamicHeadTrainPair]):
    @classmethod
    def name(cls) -> str:
        return "DynamicHeadTrain"

    def __init__(self, cfg: SimpleNamespace, from_idx: int = 0, to_idx: int = -1) -> None:
        self.sequence = SequenceBase[StereoInertialFrame].instantiate(cfg.type, cfg.args).clip(
            start_idx=from_idx,
            end_idx=to_idx if to_idx != -1 else None,
        )
        if len(self.sequence) < 2:
            raise ValueError("DynamicHeadTrainDataset requires at least 2 stereo-inertial frames")
        super().__init__(len(self.sequence) - 1)

    def __getitem__(self, local_index: int) -> DynamicHeadTrainPair:
        i = self.get_index(local_index)
        cur = self.sequence[i]
        nxt = self.sequence[i + 1]
        if cur.gt_pose is None or nxt.gt_pose is None:
            raise ValueError("DynamicHeadTrainDataset requires gt_pose in source sequence")

        T_bc = cur.stereo.T_BS
        T_rel = compute_camera_relative_pose(cur.gt_pose, nxt.gt_pose, T_bc)
        T_rel_m = T_rel.matrix()
        gt_R_rel = T_rel_m[:, :3, :3]
        gt_t_rel = T_rel_m[:, :3, 3]

        imu_window, imu_mask = self._pack_imu_window(nxt)
        return DynamicHeadTrainPair(
            idx=[local_index],
            time_ns=cur.time_ns,
            cur=cur,
            nxt=nxt,
            gt_pose=cur.gt_pose,
            gt_R_rel=gt_R_rel,
            gt_t_rel=gt_t_rel,
            imu_window=imu_window,
            imu_mask=imu_mask,
        )

    @staticmethod
    def _pack_imu_window(frame: StereoInertialFrame) -> tuple[torch.Tensor, torch.Tensor]:
        imu = frame.imu
        acc = imu.acc
        gyro = imu.gyro
        time_ns = imu.time_ns
        if time_ns.dim() == 3:
            time_ns = time_ns.squeeze(-1)

        if time_ns.size(1) < 2:
            dt = torch.ones((time_ns.size(0), time_ns.size(1), 1), dtype=torch.float32, device=acc.device) * 1e-2
        else:
            dt_ns = torch.diff(time_ns, dim=1, prepend=time_ns[:, :1])
            dt = (dt_ns.float().unsqueeze(-1) / 1_000_000_000.0).clamp(min=1e-6)

        imu_window = torch.cat([acc, gyro, dt], dim=-1)
        imu_mask = torch.ones((*imu_window.shape[:2], 1), dtype=torch.bool, device=imu_window.device)
        return imu_window, imu_mask

    def transform_source(self, actions: list[T.Callable[[StereoInertialFrame], StereoInertialFrame]]):
        self.sequence = self.sequence.transform(actions)
        return self

    @classmethod
    def is_valid_config(cls, config: SimpleNamespace | None) -> None:
        SequenceBase.is_valid_config(config)
