from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import torch
import pypose as pp
from torch.utils.data import Dataset

from DataLoader import SequenceBase, StereoInertialFrame
from Utility.Extensions import ConfigTestable


class DynamicHeadTrainDataset(Dataset, ConfigTestable):
    """
    Windowed training dataset for static-confidence head.

    Returns a dictionary with time-major tensors for windowed BPTT:
      - img_prev_l, img_curr_l, img_curr_r: [T, 3, H, W]
      - K: [T, 3, 3], baseline: [T]
      - imu_window: [T, M, 7], imu_mask: [T, M]
      - gt_R_rel: [T, 3, 3], gt_t_rel: [T, 3]
      - valid_depth_mask: [T, 1, H, W]
    where T=window_len and M=imu_max_window_size.
    """

    def __init__(self, config: SimpleNamespace | dict[str, Any]):
        cfg = self.config_dict2ns(config)
        self.window_len = int(getattr(cfg, "window_len", 4))
        self.stride = int(getattr(cfg, "stride", self.window_len))
        self.imu_max_window_size = int(getattr(cfg, "imu_max_window_size", 64))
        if self.window_len < 1:
            raise ValueError("window_len should be >= 1")
        if self.stride < 1:
            raise ValueError("stride should be >= 1")

        source = getattr(cfg, "source")
        self.sequence = SequenceBase[StereoInertialFrame].instantiate(source.type, source.args)
        self.starts = list(range(0, max(len(self.sequence) - self.window_len, 0) + 1, self.stride))

    @staticmethod
    def config_dict2ns(cfg: SimpleNamespace | dict[str, Any]) -> SimpleNamespace:
        if isinstance(cfg, SimpleNamespace):
            return cfg
        return SequenceBase.config_dict2ns(cfg)

    @staticmethod
    def _pack_imu_window(frame: StereoInertialFrame, max_samples: int) -> tuple[torch.Tensor, torch.Tensor]:
        time_ns = frame.imu.time_ns[0, :, 0].to(dtype=torch.float32)
        acc = frame.imu.acc[0].to(dtype=torch.float32)
        gyro = frame.imu.gyro[0].to(dtype=torch.float32)
        t_rel = (time_ns - time_ns[0]) / 1e9
        imu = torch.cat([t_rel.unsqueeze(-1), acc, gyro], dim=-1)

        n = imu.size(0)
        if n >= max_samples:
            imu = imu[-max_samples:]
            mask = torch.ones((max_samples,), dtype=torch.bool)
        else:
            pad = torch.zeros((max_samples - n, 7), dtype=imu.dtype)
            imu = torch.cat([imu, pad], dim=0)
            mask = torch.cat(
                [torch.ones((n,), dtype=torch.bool), torch.zeros((max_samples - n,), dtype=torch.bool)],
                dim=0,
            )
        return imu, mask

    def __len__(self) -> int:
        return len(self.starts)

    @staticmethod
    def _pose_to_camera(frame: StereoInertialFrame) -> pp.LieTensor:
        if frame.gt_pose is None:
            raise RuntimeError("DynamicHeadTrainDataset requires gt_pose for relative pose supervision.")
        body_pose = pp.SE3(frame.gt_pose)
        t_bs = frame.stereo.T_BS
        if t_bs is None:
            return body_pose
        return body_pose @ pp.SE3(t_bs)

    @staticmethod
    def _valid_depth_mask(frame: StereoInertialFrame) -> torch.Tensor:
        if frame.stereo.gt_depth is not None:
            depth = frame.stereo.gt_depth[0].float()
            return torch.isfinite(depth).unsqueeze(0) & (depth > 0.0).unsqueeze(0)
        h, w = frame.stereo.height, frame.stereo.width
        return torch.ones((1, h, w), dtype=torch.bool)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        start = self.starts[idx]
        frames = [self.sequence[start + i] for i in range(self.window_len + 1)]

        img_prev_l = []
        img_curr_l = []
        img_curr_r = []
        K = []
        baseline = []
        imu_window = []
        imu_mask = []
        gt_R_rel = []
        gt_t_rel = []
        valid_depth_mask = []
        t_prev_ns = []
        t_curr_ns = []

        for t in range(self.window_len):
            prev_f = frames[t]
            curr_f = frames[t + 1]
            prev_cam = self._pose_to_camera(prev_f)
            curr_cam = self._pose_to_camera(curr_f)
            rel = prev_cam.Inv() @ curr_cam
            rel_R = rel.rotation().matrix()[0].float()
            rel_t = rel.translation()[0].float()
            imu_win, imu_valid = self._pack_imu_window(curr_f, self.imu_max_window_size)
            valid_mask = self._valid_depth_mask(curr_f)

            img_prev_l.append(prev_f.stereo.imageL[0].float())
            img_curr_l.append(curr_f.stereo.imageL[0].float())
            img_curr_r.append(curr_f.stereo.imageR[0].float())
            K.append(curr_f.stereo.K[0].float())
            baseline.append(curr_f.stereo.baseline[0].float())
            imu_window.append(imu_win)
            imu_mask.append(imu_valid)
            gt_R_rel.append(rel_R)
            gt_t_rel.append(rel_t)
            valid_depth_mask.append(valid_mask)
            t_prev_ns.append(torch.tensor(float(prev_f.stereo.time_ns[0]), dtype=torch.float32))
            t_curr_ns.append(torch.tensor(float(curr_f.stereo.time_ns[0]), dtype=torch.float32))

        return {
            "img_prev_l": torch.stack(img_prev_l, dim=0),
            "img_curr_l": torch.stack(img_curr_l, dim=0),
            "img_curr_r": torch.stack(img_curr_r, dim=0),
            "K": torch.stack(K, dim=0),
            "baseline": torch.stack(baseline, dim=0),
            "imu_window": torch.stack(imu_window, dim=0),
            "imu_mask": torch.stack(imu_mask, dim=0),
            "gt_R_rel": torch.stack(gt_R_rel, dim=0),
            "gt_t_rel": torch.stack(gt_t_rel, dim=0),
            "valid_depth_mask": torch.stack(valid_depth_mask, dim=0),
            "t_prev_ns": torch.stack(t_prev_ns, dim=0),
            "t_curr_ns": torch.stack(t_curr_ns, dim=0),
        }

    @classmethod
    def is_valid_config(cls, config: SimpleNamespace | None) -> None:
        cls._enforce_config_spec(config, {
            "window_len": lambda v: isinstance(v, int) and v > 0,
            "stride": lambda v: isinstance(v, int) and v > 0,
            "imu_max_window_size": lambda v: isinstance(v, int) and v > 0,
            "source": lambda v: v is not None,
        }, allow_excessive_cfg=True)
