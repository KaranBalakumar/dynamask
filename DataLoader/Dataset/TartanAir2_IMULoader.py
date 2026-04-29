from pathlib import Path
from typing import cast

import numpy as np
import torch
import pypose as pp

from DataLoader.Interface import IMUData, AttitudeData


class TartanAirV2IMULoader:
    """Load real IMU data from TartanAir v2 format.

    TartanAir v2 stores IMU data in ``<sequence>/imu/`` as npy files.
    IMU runs at 100 Hz, camera at 10 Hz.
    """

    def __init__(self, imu_dir: Path, gravity: float = 9.81, fixed_imu_samples: int = 10) -> None:
        assert imu_dir.exists(), f"IMU directory does not exist: {imu_dir}"

        self.imu_dir = imu_dir
        self.gravity = gravity
        self.fixed_imu_samples = fixed_imu_samples  # 0 = use actual window size
        self.T_BS = pp.identity_SE3(1)

        # --- Load IMU data (required) ---------------------------------------
        for fname in ("acc.npy", "gyro.npy", "imu_time.npy", "cam_time.npy"):
            assert (imu_dir / fname).exists(), f"Required IMU file not found: {imu_dir / fname}"

        acc = np.load(str(imu_dir / "acc.npy"))
        gyro = np.load(str(imu_dir / "gyro.npy"))
        imu_time = np.load(str(imu_dir / "imu_time.npy"))
        cam_time = np.load(str(imu_dir / "cam_time.npy"))

        self.acc = torch.from_numpy(acc).float().unsqueeze(0)                      # (1, N, 3)
        self.gyro = torch.from_numpy(gyro).float().unsqueeze(0)                    # (1, N, 3)
        self.imu_time = torch.from_numpy(imu_time * 1e9).long().unsqueeze(0).unsqueeze(-1)  # (1, N, 1) ns
        self.cam_time = torch.from_numpy(cam_time * 1e9).long()                             # (M,) ns

        N = self.acc.size(1)
        assert self.gyro.size(1) == N and self.imu_time.size(1) == N, \
            f"IMU file length mismatch: acc={self.acc.size(1)} gyro={self.gyro.size(1)} imu_time={self.imu_time.size(1)}"

        # --- Load ground truth (optional) ------------------------------------
        gt_files = ["ori_global.npy", "pos_global.npy", "vel_body.npy"]
        self.gt_available = all((imu_dir / f).exists() for f in gt_files)

        if self.gt_available:
            ori_global = np.load(str(imu_dir / "ori_global.npy"))
            pos_global = np.load(str(imu_dir / "pos_global.npy"))
            vel_body = np.load(str(imu_dir / "vel_body.npy"))

            # Euler angles (xyz radians) → SO3
            self.gt_rot = pp.euler2SO3(
                torch.from_numpy(ori_global).float()
            ).unsqueeze(0)  # (1, N, 4)

            self.gt_pos = torch.from_numpy(pos_global).float().unsqueeze(0)  # (1, N, 3)
            self.gt_vel = torch.from_numpy(vel_body).float().unsqueeze(0)    # (1, N, 3)
        else:
            self.gt_rot = pp.identity_SO3(1).repeat(1, N, 1)   # (1, N, 4)
            self.gt_pos = torch.zeros(1, N, 3, dtype=torch.float)
            self.gt_vel = torch.zeros(1, N, 3, dtype=torch.float)

        # --- Align camera timestamps to IMU indices -------------------------
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
        N = imu_t.size(0)

        # searchsorted(right=True) returns the insertion point *after* any
        # equal entries, i.e. the first index where imu_t > cam_t.
        idx = torch.searchsorted(imu_t, cam_t, right=True)  # (M,) in [0, N]
        self.cam2imu_idx = torch.clamp(idx - 1, 0, N - 1)   # (M,) in [0, N-1]

        # Past-the-end sentinel so frameRangeQuery slices correctly
        # when end_frame == len(self).
        sentinel = torch.tensor([N], dtype=torch.long)
        self.cam2imu_idx = torch.cat([self.cam2imu_idx, sentinel])  # (M+1,)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        """Number of camera frames (M)."""
        return self.cam_time.size(0)

    def frame_range_query(self, start_frame: int, end_frame: int) -> tuple[IMUData, AttitudeData]:
        """Retrieve IMU data spanning camera frames [start_frame, end_frame)."""
        if self.fixed_imu_samples > 0:
            end_imu_idx = self.cam2imu_idx[end_frame].item()
            start_imu_idx = max(0, end_imu_idx - self.fixed_imu_samples)
            actual = end_imu_idx - start_imu_idx
            pad = self.fixed_imu_samples - actual
        else:
            start_imu_idx = self.cam2imu_idx[start_frame].item()
            end_imu_idx = self.cam2imu_idx[end_frame].item()
            pad = 0

        _slice = slice(start_imu_idx, end_imu_idx)

        imu = IMUData(
            T_BS=self.T_BS,
            gravity=[self.gravity],
            time_ns=self.imu_time[:, _slice],
            acc=self.acc[:, _slice],
            gyro=self.gyro[:, _slice],
        )
        att = AttitudeData(
            T_BS=self.T_BS,
            gravity=[self.gravity],
            time_ns=self.imu_time[:, _slice],
            gt_pos=self.gt_pos[:, _slice],
            gt_vel=self.gt_vel[:, _slice],
            gt_rot=cast(pp.LieTensor, self.gt_rot[:, _slice]),
            init_pos=self.gt_pos[:, start_imu_idx : start_imu_idx + 1],
            init_vel=self.gt_vel[:, start_imu_idx : start_imu_idx + 1],
            init_rot=cast(pp.LieTensor, self.gt_rot[:, start_imu_idx : start_imu_idx + 1]),
        )

        # Pad short windows (sequence starts) by repeating the first sample.
        # This ensures all IMU windows have the same size for batch collation.
        if pad > 0:
            _rep_first = lambda t: torch.cat([t[:, :1].repeat(1, pad, *([1]*(t.dim()-2))), t], dim=1)
            imu.acc = _rep_first(imu.acc)
            imu.gyro = _rep_first(imu.gyro)
            imu.time_ns = _rep_first(imu.time_ns)
            att.gt_pos = _rep_first(att.gt_pos)
            att.gt_vel = _rep_first(att.gt_vel)
            att.gt_rot = pp.LieTensor(_rep_first(att.gt_rot.tensor()), ltype=att.gt_rot.ltype)
            att.time_ns = _rep_first(att.time_ns)

        return imu, att

    # Alias for API compatibility with v1's camelCase convention
    frameRangeQuery = frame_range_query

    def __getitem__(self, index: int) -> tuple[IMUData, AttitudeData]:
        """Single-frame IMU access (used for the first keyframe)."""
        if index < 0 or index >= len(self):
            raise IndexError(f"Camera frame index {index} out of range [0, {len(self)})")
        return self.frame_range_query(index, index + 1)
