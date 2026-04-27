"""VIODE dataset loader from HDF5 files.

VIODE is a dynamic-scene stereo+IMU dataset with ground truth semantic
segmentation, odometry, and IMU.  Data is pre-converted from ROS bags to
HDF5 files by Scripts/convert_viode.py.

Each HDF5 file contains:
  cam0/images       (M, H, W, 3) uint8  — left stereo images
  cam1/images       (M, H, W, 3) uint8  — right stereo images
  cam0/times        (M,) int64          — left camera timestamps (ns)
  cam0/segmentation (M, H, W, 3) uint8  — semantic segmentation (left)
  imu/times         (N,) int64          — IMU timestamps (ns)
  imu/gyro          (N, 3) float32     — angular velocity (rad/s)
  imu/acc           (N, 3) float32     — linear acceleration (m/s²)
  odometry/times    (N,) int64          — GT odometry timestamps (ns)
  odometry/position (N, 3) float64     — GT position (world frame)
  odometry/orientation (N, 4) float64  — GT orientation (xyzw quaternion)
  odometry/velocity    (N, 3) float64  — GT linear velocity
  align/cam2imu_idx (M,) int64         — camera→IMU index mapping

  attrs: width, height, fx, fy, cx, cy, baseline
"""

from __future__ import annotations

import h5py
import numpy as np
import torch
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pypose as pp

from ..SequenceBase import SequenceBase
from ..Interface import (
    StereoData, StereoFrame, StereoInertialFrame,
    IMUData, AttitudeData,
)


class VIODESequence(SequenceBase[StereoInertialFrame]):
    """Load a single VIODE HDF5 sequence for dynGRU training."""

    @classmethod
    def name(cls) -> str:
        return "VIODE"

    def __init__(self, config: SimpleNamespace | dict[str, Any]):
        cfg = self.config_dict2ns(config)
        self.h5_path = Path(cfg.path)
        assert self.h5_path.exists(), f"HDF5 file not found: {self.h5_path}"

        self._h5: h5py.File | None = None
        self._open()

        # Camera intrinsics
        self.K = torch.tensor([[
            [self._h5.attrs['fx'], 0.0, self._h5.attrs['cx']],
            [0.0, self._h5.attrs['fy'], self._h5.attrs['cy']],
            [0.0, 0.0, 1.0],
        ]], dtype=torch.float32)
        self.baseline = float(self._h5.attrs['baseline'])
        self.height = int(self._h5.attrs['height'])
        self.width = int(self._h5.attrs['width'])

        # GT segmentation → dynamic mask (optionally)
        self.use_seg = getattr(cfg, "use_segmentation", True)

        super().__init__(self._h5['cam0/images'].shape[0] - 1)  # pairs
        self._close()  # close file handle to allow multiprocess pickling

    def _open(self):
        if self._h5 is None:
            self._h5 = h5py.File(self.h5_path, 'r')

    def _close(self):
        if self._h5 is not None:
            self._h5.close()
            self._h5 = None

    def __getstate__(self):
        state = self.__dict__.copy()
        state['_h5'] = None  # h5py objects cannot be pickled
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)

    def __del__(self):
        self._close()

    def _read_image(self, idx: int, cam: str) -> torch.Tensor:
        """Read and convert image to (1, 3, H, W) float32 [0,1]."""
        img = self._h5[f'{cam}/images'][idx]  # (H, W, 3) uint8
        return torch.from_numpy(img).float().permute(2, 0, 1).unsqueeze(0) / 255.0

    def _read_seg_mask(self, idx: int) -> torch.Tensor | None:
        """Build binary dynamic mask from semantic segmentation.

        VIODE segmentation encodes object IDs as RGB colors.  Static objects
        (road, building, etc.) get one set of colors, dynamic objects
        (vehicles, pedestrians) another.  We use a simplified heuristic:
        any non-black pixel in the segmentation image is potentially dynamic
        after masking out the static classes.

        Returns (1, 1, H, W) float32 mask: 1=dynamic, 0=static.
        """
        if not self.use_seg:
            return None
        seg = self._h5['cam0/segmentation'][idx]  # (H, W, 3) uint8
        # In VIODE/AirSim: black (0,0,0) = static background, non-black = object
        # This is approximate — full class mapping needs rgb_id.txt
        is_object = (seg.max(axis=-1) > 10).astype(np.float32)  # (H, W)
        return torch.from_numpy(is_object).unsqueeze(0).unsqueeze(0)  # (1,1,H,W)

    def _get_imu_window(self, frame_idx: int) -> tuple[IMUData, AttitudeData]:
        """Get IMU + odometry window between camera frames [idx, idx+1]."""
        cam2imu = self._h5['align/cam2imu_idx']
        start_imu = int(cam2imu[frame_idx])
        end_imu = int(cam2imu[frame_idx + 1]) if frame_idx + 1 < len(cam2imu) else int(cam2imu[frame_idx]) + 10

        # IMU
        gyro = torch.from_numpy(self._h5['imu/gyro'][start_imu:end_imu]).float().unsqueeze(0)
        acc  = torch.from_numpy(self._h5['imu/acc'][start_imu:end_imu]).float().unsqueeze(0)
        times_ns = torch.from_numpy(
            self._h5['imu/times'][start_imu:end_imu].astype(np.int64)
        ).unsqueeze(0).unsqueeze(-1)

        imu_data = IMUData(
            T_BS=pp.identity_SE3(1),
            time_ns=times_ns,
            gravity=[9.81],
            acc=acc,
            gyro=gyro,
        )

        # Odometry (GT attitude for EKF seeding)
        odom_times = self._h5['odometry/times'][:]
        odom_start = np.searchsorted(odom_times, self._h5['cam0/times'][frame_idx])
        odom_start = min(odom_start, len(self._h5['odometry/position']) - 1)

        pos = torch.from_numpy(self._h5['odometry/position'][odom_start:odom_start+1]).float().unsqueeze(0)  # (1,1,3)
        ori = torch.from_numpy(self._h5['odometry/orientation'][odom_start:odom_start+1]).float().unsqueeze(0)  # (1,1,4) xyzw
        vel = torch.from_numpy(self._h5['odometry/velocity'][odom_start:odom_start+1]).float().unsqueeze(0)  # (1,1,3)

        # Convert xyzw quaternion to SO3
        q = ori[0, 0]  # (4,) xyzw
        R_so3 = pp.euler2SO3(torch.zeros(1, 3)).to(ori.device)  # placeholder
        # Use scipy to convert xyzw → rotation matrix → SO3
        from scipy.spatial.transform import Rotation
        r = Rotation.from_quat(np.array([q[0], q[1], q[2], q[3]]))  # xyzw
        euler = r.as_euler('xyz', degrees=False)
        R_so3 = pp.euler2SO3(torch.from_numpy(euler).float().unsqueeze(0))

        att_data = AttitudeData(
            T_BS=pp.identity_SE3(1),
            time_ns=times_ns,
            gravity=[9.81],
            gt_pos=pos, gt_vel=vel,
            gt_rot=R_so3.unsqueeze(0),  # (1,1,4)
            init_pos=pos, init_vel=vel,
            init_rot=R_so3.unsqueeze(0),
        )

        return imu_data, att_data

    def _get_odom_pose(self, frame_idx: int) -> torch.Tensor:
        """Get SE(3) odometry pose for a camera frame as (7,) LieTensor."""
        cam_time = self._h5['cam0/times'][frame_idx]
        odom_times = self._h5['odometry/times'][:]
        odom_idx = np.searchsorted(odom_times, cam_time)
        odom_idx = min(odom_idx, len(odom_times) - 1)

        pos = self._h5['odometry/position'][odom_idx]       # (3,)
        ori = self._h5['odometry/orientation'][odom_idx]    # (4,) xyzw

        # Convert xyzw → SE3
        from scipy.spatial.transform import Rotation
        r = Rotation.from_quat(np.array([ori[0], ori[1], ori[2], ori[3]]))
        euler = r.as_euler('xyz', degrees=False)
        R = torch.from_numpy(r.as_matrix()).float()
        t = torch.from_numpy(pos).float()
        T_mat = torch.eye(4)
        T_mat[:3, :3] = R
        T_mat[:3, 3] = t
        return pp.mat2SE3(T_mat.unsqueeze(0)).squeeze(0)

    def __getitem__(self, local_index: int) -> StereoInertialFrame:
        self._open()
        idx = self.get_index(local_index)

        imgL = self._read_image(idx, 'cam0')
        imgR = self._read_image(idx, 'cam1')

        gt_depth = None  # VIODE does not provide GT depth
        gt_flow = None   # nor pre-computed optical flow
        flow_mask = None
        gt_dyn_mask = self._read_seg_mask(idx)

        stereo = StereoData(
            T_BS=pp.identity_SE3(1),
            K=self.K,
            baseline=torch.tensor([self.baseline]),
            time_ns=[int(self._h5['cam0/times'][idx])],
            height=self.height, width=self.width,
            imageL=imgL, imageR=imgR,
            gt_depth=gt_depth, gt_flow=gt_flow, flow_mask=flow_mask,
        )

        imu_data, att_data = self._get_imu_window(idx)
        gt_pose = self._get_odom_pose(idx)

        return StereoInertialFrame(
            idx=[local_index],
            stereo=stereo,
            imu=imu_data,
            gt_attitude=att_data,
            gt_pose=gt_pose,
            time_ns=[int(self._h5['cam0/times'][idx])],
        )

    @classmethod
    def is_valid_config(cls, config: SimpleNamespace | None) -> None:
        assert config is not None
        cls._enforce_config_spec(config, {
            "path": lambda s: isinstance(s, str),
        }, allow_excessive_cfg=True)
