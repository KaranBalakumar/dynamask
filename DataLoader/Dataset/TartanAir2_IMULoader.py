from pathlib import Path
from typing import cast

import numpy as np
import torch
import pypose as pp
from scipy.spatial.transform import Rotation

from DataLoader.Interface import IMUData, AttitudeData


class TartanAirV2IMULoader:
    """Load real IMU data from TartanAir v2 format.

    TartanAir v2 stores IMU data in ``<sequence>/imu/`` as npy files.
    IMU runs at 100 Hz, camera at 10 Hz.
    """

    def __init__(self, imu_dir: Path, gravity: float = 9.81) -> None:
        assert imu_dir.exists(), f"IMU directory does not exist: {imu_dir}"

        self.imu_dir = imu_dir
        self.gravity = gravity
        self.T_BS = pp.identity_SE3(1)  # Body -> Sensor transformation, (1, 7)

        # --- Load IMU data ------------------------------------------------
        acc = np.load(str(imu_dir / "acc.npy")).astype(np.float64)
        gyro = np.load(str(imu_dir / "gyro.npy")).astype(np.float64)
        imu_time = np.load(str(imu_dir / "imu_time.npy")).astype(np.float64)
        cam_time = np.load(str(imu_dir / "cam_time.npy")).astype(np.float64)

        self.acc = torch.tensor(acc, dtype=torch.float).unsqueeze(0)          # (1, N, 3)
        self.gyro = torch.tensor(gyro, dtype=torch.float).unsqueeze(0)        # (1, N, 3)
        self.imu_time = torch.tensor(imu_time * 1e9, dtype=torch.long).unsqueeze(0).unsqueeze(-1)  # (1, N, 1) ns
        self.cam_time = torch.tensor(cam_time * 1e9, dtype=torch.long)        # (M,) ns

        # --- Load ground truth (optional) ----------------------------------
        gt_files = ["ori_global.npy", "pos_global.npy", "vel_body.npy"]
        self.gt_available = all((imu_dir / f).exists() for f in gt_files)

        if self.gt_available:
            ori_global = np.load(str(imu_dir / "ori_global.npy")).astype(np.float64)
            pos_global = np.load(str(imu_dir / "pos_global.npy")).astype(np.float64)
            vel_body = np.load(str(imu_dir / "vel_body.npy")).astype(np.float64)

            # Convert Euler angles (xyz radians) to SO3 via scipy rotation
            r = Rotation.from_euler("xyz", ori_global, degrees=False)
            self.gt_rot = pp.euler2SO3(
                torch.from_numpy(r.as_euler("xyz", degrees=False)).float()
            ).unsqueeze(0)  # (1, N, 4)

            self.gt_pos = torch.tensor(pos_global, dtype=torch.float).unsqueeze(0)  # (1, N, 3)
            self.gt_vel = torch.tensor(vel_body, dtype=torch.float).unsqueeze(0)    # (1, N, 3)
        else:
            N = self.acc.size(1)
            self.gt_rot = pp.identity_SO3(1).repeat(1, N, 1)   # (1, N, 4)
            self.gt_pos = torch.zeros(1, N, 3, dtype=torch.float)
            self.gt_vel = torch.zeros(1, N, 3, dtype=torch.float)

        # --- Align camera timestamps to IMU indices -----------------------
        self._align_camera_time()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _align_camera_time(self) -> None:
        """For each camera timestamp ``t_k`` find::

            cam2imu_idx[k] = max{i | imu_time[i] <= t_k < imu_time[i+1]}

        The result is stored in ``self.cam2imu_idx``, padded with a
        past-the-end sentinel so that ``frameRangeQuery`` slices cleanly.
        """
        imu_t = self.imu_time[0, :, 0]  # (N,)  drop batch and trailing dim
        cam_t = self.cam_time            # (M,)

        # searchsorted(right=True) returns the insertion point *after* any
        # equal entries, i.e. the first index where imu_t > cam_t.
        idx = torch.searchsorted(imu_t, cam_t, right=True)  # (M,)

        # idx is in [0, N]; idx-1 gives the last imu index <= cam_t.
        # Clamp to [0, N-2] so that imu idx+1 is always in bounds.
        self.cam2imu_idx = torch.clamp(idx - 1, 0, len(imu_t) - 2)  # (M,)

        # Pad with a past-the-end sentinel for clean indexing in
        # frameRangeQuery when end_frame == len(self).
        sentinel = torch.tensor([self.acc.size(1)], dtype=torch.long)
        self.cam2imu_idx = torch.cat([self.cam2imu_idx, sentinel])  # (M+1,)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        """Number of camera frames (M)."""
        return self.cam_time.size(0)

    def frame_range_query(self, start_frame: int, end_frame: int) -> tuple[IMUData, AttitudeData]:
        """Retrieve IMU data spanning camera frames [start_frame, end_frame).

        Args:
            start_frame: start camera frame index (inclusive).
            end_frame:   end camera frame index (exclusive).

        Returns:
            Tuple of ``(IMUData, AttitudeData)`` with the IMU window
            between those two camera frames.
        """
        start_imu_idx = self.cam2imu_idx[start_frame].item()
        end_imu_idx = self.cam2imu_idx[end_frame].item()

        return IMUData(
            T_BS=self.T_BS,
            gravity=[self.gravity],
            time_ns=self.imu_time[:, start_imu_idx:end_imu_idx],
            acc=self.acc[:, start_imu_idx:end_imu_idx],
            gyro=self.gyro[:, start_imu_idx:end_imu_idx],
        ), AttitudeData(
            T_BS=self.T_BS,
            gravity=[self.gravity],
            time_ns=self.imu_time[:, start_imu_idx:end_imu_idx],
            gt_pos=self.gt_pos[:, start_imu_idx:end_imu_idx],
            gt_vel=self.gt_vel[:, start_imu_idx:end_imu_idx],
            gt_rot=cast(pp.LieTensor, self.gt_rot[:, start_imu_idx:end_imu_idx]),
            init_pos=self.gt_pos[:, start_imu_idx : start_imu_idx + 1],
            init_vel=self.gt_vel[:, start_imu_idx : start_imu_idx + 1],
            init_rot=cast(pp.LieTensor, self.gt_rot[:, start_imu_idx : start_imu_idx + 1]),
        )

    # Alias for API compatibility with v1's camelCase convention
    frameRangeQuery = frame_range_query

    def __getitem__(self, index: int) -> tuple[IMUData, AttitudeData]:
        return self.frame_range_query(index, index + 1)
