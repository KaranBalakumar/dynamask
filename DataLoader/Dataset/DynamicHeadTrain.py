from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

import numpy as np
from torch.utils.data import Dataset

from ..Interface import DataFramePair, StereoInertialFrame
from ..SequenceBase import SequenceBase


def _cfg_get(cfg: SimpleNamespace | dict[str, Any], key: str, default: Any) -> Any:
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


def _optional_index(v: Any) -> int | None:
    if v is None:
        return None
    if isinstance(v, SimpleNamespace):
        return None
    return int(v)


@dataclass
class DynamicHeadDatasetStats:
    total_pairs: int
    valid_pairs: int


class DynamicHeadDatasetEmptyError(ValueError):
    def __init__(self, stats: DynamicHeadDatasetStats) -> None:
        self.stats = stats
        super().__init__(
            "DynamicHeadTrainDataset has zero valid training pairs "
            f"(total_pairs={stats.total_pairs}, valid_pairs={stats.valid_pairs}). "
            "Check that gt_pose exists for adjacent frames and IMU intervals are valid."
        )


class DynamicHeadTrainDataset(Dataset[DataFramePair[StereoInertialFrame]]):
    """
    Pairwise dataset for static-confidence-head training.
    Keeps only pairs that have GT body poses and a valid IMU interval in frame t+1.
    """

    @staticmethod
    def _metadata_gt_valid(gt_pose: Any, raw_idx: int) -> bool:
        if gt_pose is None:
            return False
        if isinstance(gt_pose, np.ndarray):
            return bool(np.isfinite(gt_pose[raw_idx]).all())
        return True

    @staticmethod
    def _metadata_imu_count(cam_time: np.ndarray, imu_time: np.ndarray, raw_idx: int) -> int:
        if raw_idx == 0:
            end = int(np.searchsorted(imu_time, cam_time[raw_idx], side="right"))
            return end
        start = int(np.searchsorted(imu_time, cam_time[raw_idx - 1], side="left"))
        end = int(np.searchsorted(imu_time, cam_time[raw_idx], side="right"))
        return max(0, end - start)

    def __init__(self, config: SimpleNamespace | dict[str, Any]) -> None:
        cfg = SequenceBase.config_dict2ns(config)
        seq_cfg = cfg.sequence
        sequence = SequenceBase.instantiate(seq_cfg.type, seq_cfg.args)
        start = int(_cfg_get(cfg, "start", 0))
        end = _optional_index(_cfg_get(cfg, "end", None))
        step = _optional_index(_cfg_get(cfg, "step", None))
        self.sequence = sequence.clip(start_idx=start, end_idx=end, step=step)

        self.valid_indices: list[int] = []
        has_metadata = all(hasattr(self.sequence, name) for name in ("cam_time", "imu_time", "gt_pose", "get_index"))
        if has_metadata:
            cam_time = np.asarray(getattr(self.sequence, "cam_time"))
            imu_time = np.asarray(getattr(self.sequence, "imu_time"))
            gt_pose = getattr(self.sequence, "gt_pose")
            for i in range(max(0, len(self.sequence) - 1)):
                cur_raw = int(self.sequence.get_index(i))
                nxt_raw = int(self.sequence.get_index(i + 1))
                if not self._metadata_gt_valid(gt_pose, cur_raw) or not self._metadata_gt_valid(gt_pose, nxt_raw):
                    continue
                if self._metadata_imu_count(cam_time, imu_time, nxt_raw) < 2:
                    continue
                self.valid_indices.append(i)
        else:
            for i in range(max(0, len(self.sequence) - 1)):
                cur = self.sequence[i]
                nxt = self.sequence[i + 1]
                if not isinstance(cur, StereoInertialFrame) or not isinstance(nxt, StereoInertialFrame):
                    continue
                if cur.gt_pose is None or nxt.gt_pose is None:
                    continue
                if nxt.imu.time_ns.shape[1] < 2:
                    continue
                self.valid_indices.append(i)

        self.stats = DynamicHeadDatasetStats(
            total_pairs=max(0, len(self.sequence) - 1),
            valid_pairs=len(self.valid_indices),
        )
        if self.stats.valid_pairs == 0:
            raise DynamicHeadDatasetEmptyError(self.stats)

    def __len__(self) -> int:
        return len(self.valid_indices)

    def __getitem__(self, index: int) -> DataFramePair[StereoInertialFrame]:
        i = self.valid_indices[index]
        cur = self.sequence[i]
        nxt = self.sequence[i + 1]
        assert isinstance(cur, StereoInertialFrame) and isinstance(nxt, StereoInertialFrame)
        return DataFramePair(
            idx=cur.idx,
            time_ns=cur.time_ns,
            gt_pose=cur.gt_pose,
            cur=cur,
            nxt=nxt,
        )
