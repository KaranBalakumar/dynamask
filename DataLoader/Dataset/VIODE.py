from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import h5py
import numpy as np
import pypose as pp
import torch
from numpy.typing import NDArray

from ..Interface import StereoData, IMUData, StereoInertialFrame
from ..SequenceBase import SequenceBase


class VIODE_StreamSequence(SequenceBase[StereoInertialFrame]):
    @classmethod
    def name(cls) -> str:
        return "VIODE_Stream"

    def __init__(self, config: SimpleNamespace | dict[str, Any]) -> None:
        cfg = self.config_dict2ns(config)
        self.path = Path(cfg.path)
        assert self.path.exists(), f"VIODE stream file does not exist: {self.path}"
        self.h5 = h5py.File(self.path, "r")

        self.cam_time: NDArray[np.int64] = np.asarray(self._dataset("stereo/time_ns")[:], dtype=np.int64)
        self.imu_time: NDArray[np.int64] = np.asarray(self._dataset("imu/time_ns")[:], dtype=np.int64)
        self.gt_pose: np.ndarray[Any, Any] | None = (
            np.asarray(self._dataset("stereo/gt_pose")[:]) if "stereo/gt_pose" in self.h5 else None
        )

        self.stereo_left = self._dataset("stereo/left")
        self.stereo_right = self._dataset("stereo/right")
        self.imu_acc_ds = self._dataset("imu/acc")
        self.imu_gyro_ds = self._dataset("imu/gyro")

        self.K = torch.from_numpy(np.asarray(self._dataset("calib/K")[:])).float().unsqueeze(0)
        self.T_BS = pp.SE3(torch.from_numpy(np.asarray(self._dataset("calib/T_BS")[:])).float().unsqueeze(0))
        self.baseline = torch.tensor([self._scalar_float("calib/baseline")], dtype=torch.float32)
        self.gravity = self._scalar_float("calib/gravity") if "calib/gravity" in self.h5 else 9.81

        left_shape = self.stereo_left.shape
        self.height, self.width = int(left_shape[1]), int(left_shape[2])

        super().__init__(len(self.cam_time))

    def __del__(self):
        try:
            if hasattr(self, "h5"):
                self.h5.close()
        except Exception:
            pass

    def _imu_range(self, index: int) -> tuple[int, int]:
        if index == 0:
            end = int(np.searchsorted(self.imu_time, self.cam_time[index], side="right"))
            return 0, end
        start = int(np.searchsorted(self.imu_time, self.cam_time[index - 1], side="left"))
        end = int(np.searchsorted(self.imu_time, self.cam_time[index], side="right"))
        return start, end

    def _dataset(self, path: str) -> h5py.Dataset:
        node = self.h5[path]
        if not isinstance(node, h5py.Dataset):
            raise TypeError(f"Expected HDF5 dataset at '{path}', got {type(node).__name__}")
        return node

    def _scalar_float(self, path: str) -> float:
        value = self._dataset(path)[()]
        return float(np.asarray(value).item())

    def __getitem__(self, local_index: int) -> StereoInertialFrame:
        index = self.get_index(local_index)
        left = np.asarray(self.stereo_left[index])
        right = np.asarray(self.stereo_right[index])

        left_t = torch.from_numpy(left).permute(2, 0, 1).unsqueeze(0).float() / 255.0
        right_t = torch.from_numpy(right).permute(2, 0, 1).unsqueeze(0).float() / 255.0

        i0, i1 = self._imu_range(index)
        imu_t = torch.from_numpy(self.imu_time[i0:i1]).long().view(1, -1, 1)
        imu_acc = torch.from_numpy(np.asarray(self.imu_acc_ds[i0:i1])).float().view(1, -1, 3)
        imu_gyro = torch.from_numpy(np.asarray(self.imu_gyro_ds[i0:i1])).float().view(1, -1, 3)

        t_ns = int(self.cam_time[index].item())
        gt_pose = None
        if self.gt_pose is not None:
            gt_pose_np = self.gt_pose[index]
            if np.isfinite(gt_pose_np).all():
                gt_pose = pp.SE3(torch.from_numpy(np.asarray(gt_pose_np)).float().unsqueeze(0))

        stereo = StereoData(
            T_BS=self.T_BS,
            K=self.K,
            baseline=self.baseline,
            time_ns=[t_ns],
            height=self.height,
            width=self.width,
            imageL=left_t,
            imageR=right_t,
        )
        imu = IMUData(
            T_BS=self.T_BS,
            time_ns=imu_t,
            gravity=[self.gravity],
            acc=imu_acc,
            gyro=imu_gyro,
        )

        return StereoInertialFrame(
            idx=[local_index],
            time_ns=[t_ns],
            gt_pose=gt_pose,
            stereo=stereo,
            imu=imu,
            gt_attitude=None,
        )

    @classmethod
    def is_valid_config(cls, config: SimpleNamespace | None) -> None:
        cls._enforce_config_spec(config, {
            "path": lambda s: isinstance(s, str),
        }, allow_excessive_cfg=True)
