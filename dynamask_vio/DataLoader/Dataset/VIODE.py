from __future__ import annotations

import cv2
import yaml
import torch
import numpy as np
import pypose as pp

from dataclasses import dataclass
from pathlib import Path
from bisect import bisect_left
from types import SimpleNamespace
from typing import Any

from ..Interface import StereoData, IMUData, StereoInertialFrame
from ..SequenceBase import SequenceBase


REQUIRED_TOPICS = ["/cam0/image_raw", "/cam1/image_raw", "/imu0", "/odometry"]


@dataclass
class _StereoMsg:
    timestamp_ns: int
    image: torch.Tensor


@dataclass
class _IMUMsg:
    timestamp_ns: int
    acc: torch.Tensor
    gyro: torch.Tensor


@dataclass
class _OdomMsg:
    timestamp_ns: int
    pose: pp.LieTensor


def _as_se3(mat4: np.ndarray, dtype: torch.dtype = torch.float32) -> pp.LieTensor:
    tensor = torch.tensor(mat4, dtype=dtype).unsqueeze(0)
    return pp.from_matrix(tensor, pp.SE3_type)


def _as_matrix4(raw: Any) -> np.ndarray:
    if isinstance(raw, np.ndarray):
        mat = raw
    elif isinstance(raw, list):
        mat = np.asarray(raw)
    elif isinstance(raw, dict) and "data" in raw:
        mat = np.asarray(raw["data"])
    else:
        raise ValueError(f"Unsupported matrix format: {type(raw)}")

    if mat.size == 16:
        return mat.reshape(4, 4)
    if mat.shape == (4, 4):
        return mat
    raise ValueError(f"Cannot convert {mat.shape} to 4x4 matrix")


def _extract_matrix(cfg: dict[str, Any], candidates: list[str]) -> np.ndarray | None:
    for key in candidates:
        if key in cfg:
            return _as_matrix4(cfg[key])
    return None


def _extract_intrinsics(cfg: dict[str, Any]) -> np.ndarray:
    if "intrinsics" in cfg:
        fx, fy, cx, cy = cfg["intrinsics"]
        return np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float32)
    if "K" in cfg:
        K = np.asarray(cfg["K"], dtype=np.float32)
        if K.size == 9:
            return K.reshape(3, 3)
        if K.shape == (3, 3):
            return K
    raise KeyError("Camera calibration must provide either intrinsics=[fx,fy,cx,cy] or K")


def compute_camera_relative_pose(T_wb_i: pp.LieTensor, T_wb_j: pp.LieTensor, T_bc: pp.LieTensor) -> pp.LieTensor:
    T_wc_i = T_wb_i @ T_bc
    T_wc_j = T_wb_j @ T_bc
    return T_wc_i.Inv() @ T_wc_j


class VIODE_Sequence(SequenceBase[StereoInertialFrame]):
    @classmethod
    def name(cls) -> str:
        return "VIODE"

    def __init__(self, config: SimpleNamespace | dict[str, Any]) -> None:
        cfg = self.config_dict2ns(config)
        self.root = Path(cfg.root)
        self.bag_path = self.root / cfg.bag
        if not self.bag_path.exists():
            raise FileNotFoundError(f"VIODE bag not found: {self.bag_path}")

        cam0_cfg = self._load_yaml(self.root / cfg.cam0_calib)
        cam1_cfg = self._load_yaml(self.root / cfg.cam1_calib)
        calib_cfg = self._load_yaml(self.root / cfg.calib)

        self.K0 = torch.tensor(_extract_intrinsics(cam0_cfg), dtype=torch.float32).unsqueeze(0)
        self.K1 = torch.tensor(_extract_intrinsics(cam1_cfg), dtype=torch.float32).unsqueeze(0)

        T_bc0 = _extract_matrix(calib_cfg, ["T_B_C0", "T_BS_cam0", "T_BS0"])
        T_bc1 = _extract_matrix(calib_cfg, ["T_B_C1", "T_BS_cam1", "T_BS1"])
        T_bi = _extract_matrix(calib_cfg, ["T_B_I", "T_BS_imu", "T_BS_imu0"])
        if T_bc0 is None:
            T_bc0 = _extract_matrix(cam0_cfg, ["T_BS"])
        if T_bc1 is None:
            T_bc1 = _extract_matrix(cam1_cfg, ["T_BS"])
        if T_bi is None:
            T_bi = np.eye(4, dtype=np.float32)
        if T_bc0 is None or T_bc1 is None:
            raise KeyError("Unable to resolve camera extrinsics T_BS for cam0/cam1 from calibration files")

        self.T_BC0 = _as_se3(T_bc0)
        self.T_BC1 = _as_se3(T_bc1)
        self.T_BI = _as_se3(T_bi)

        T_c0c1 = self.T_BC0.Inv() @ self.T_BC1
        self.baseline = T_c0c1.translation().norm().item()

        left, right, imu, odom = self._read_rosbag(self.bag_path)
        self.left_msgs = left
        self.right_msgs = right
        self.imu_msgs = imu
        self.odom_msgs = odom

        self.right_ts = [m.timestamp_ns for m in self.right_msgs]
        self.imu_ts = [m.timestamp_ns for m in self.imu_msgs]
        self.odom_ts = [m.timestamp_ns for m in self.odom_msgs]

        self.stereo_indices = self._build_stereo_index()
        super().__init__(len(self.stereo_indices))

    def __getitem__(self, local_index: int) -> StereoInertialFrame:
        idx = self.get_index(local_index)
        l_idx, r_idx = self.stereo_indices[idx]
        left = self.left_msgs[l_idx]
        right = self.right_msgs[r_idx]

        cur_ts = left.timestamp_ns
        prev_ts = self.left_msgs[self.stereo_indices[idx - 1][0]].timestamp_ns if idx > 0 else cur_ts
        imu_i0 = bisect_left(self.imu_ts, prev_ts)
        imu_i1 = bisect_left(self.imu_ts, cur_ts)
        if imu_i1 <= imu_i0:
            imu_i1 = min(imu_i0 + 1, len(self.imu_msgs))
            imu_i0 = max(0, imu_i1 - 1)

        imu_slice = self.imu_msgs[imu_i0:imu_i1]
        if len(imu_slice) == 0:
            imu_slice = [self.imu_msgs[min(imu_i0, len(self.imu_msgs) - 1)]]

        imu_acc = torch.stack([m.acc for m in imu_slice], dim=0).unsqueeze(0)
        imu_gyro = torch.stack([m.gyro for m in imu_slice], dim=0).unsqueeze(0)
        imu_time = torch.tensor([[m.timestamp_ns for m in imu_slice]], dtype=torch.long)

        odom_i = self._nearest_timestamp_index(self.odom_ts, cur_ts)
        gt_pose = self.odom_msgs[odom_i].pose

        h, w = left.image.size(-2), left.image.size(-1)
        return StereoInertialFrame(
            idx=[local_index],
            time_ns=[cur_ts],
            gt_pose=gt_pose,
            stereo=StereoData(
                T_BS=self.T_BC0,
                K=self.K0,
                baseline=torch.tensor([self.baseline], dtype=torch.float32),
                time_ns=[cur_ts],
                width=w,
                height=h,
                imageL=left.image,
                imageR=right.image,
            ),
            imu=IMUData(
                T_BS=self.T_BI,
                gravity=[9.81],
                time_ns=imu_time,
                acc=imu_acc,
                gyro=imu_gyro,
            ),
            gt_attitude=None,
        )

    @classmethod
    def is_valid_config(cls, config: SimpleNamespace | dict[str, Any] | None) -> None:
        if config is None:
            raise ValueError("VIODE config is required")
        if isinstance(config, dict):
            config = cls.config_dict2ns(config)
        cls._enforce_config_spec(
            config,
            {
                "root": lambda s: isinstance(s, str),
                "bag": lambda s: isinstance(s, str),
                "cam0_calib": lambda s: isinstance(s, str),
                "cam1_calib": lambda s: isinstance(s, str),
                "calib": lambda s: isinstance(s, str),
            },
            allow_excessive_cfg=True,
        )

    @staticmethod
    def _load_yaml(path: Path) -> dict[str, Any]:
        if not path.exists():
            raise FileNotFoundError(path)
        with path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        if not isinstance(data, dict):
            raise ValueError(f"YAML root must be a mapping: {path}")
        return data

    @staticmethod
    def _decode_ros_image(msg: Any) -> torch.Tensor:
        h = int(msg.height)
        w = int(msg.width)
        enc = str(msg.encoding).lower()
        raw = np.frombuffer(msg.data, dtype=np.uint8)

        if enc in {"rgb8", "bgr8"} and raw.size == h * w * 3:
            image = raw.reshape(h, w, 3)
            if enc == "bgr8":
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        elif enc == "mono8" and raw.size == h * w:
            gray = raw.reshape(h, w)
            image = np.repeat(gray[..., None], repeats=3, axis=-1)
        else:
            decoded = cv2.imdecode(raw, cv2.IMREAD_COLOR)
            if decoded is None:
                raise ValueError(f"Unsupported image encoding: {msg.encoding}")
            image = cv2.cvtColor(decoded, cv2.COLOR_BGR2RGB)

        return torch.tensor(image, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0) / 255.0

    @staticmethod
    def _msg_to_pose(msg: Any) -> pp.LieTensor:
        pos = msg.pose.pose.position
        ori = msg.pose.pose.orientation
        xyz_xyzw = torch.tensor(
            [[pos.x, pos.y, pos.z, ori.x, ori.y, ori.z, ori.w]],
            dtype=torch.float64,
        )
        return pp.SE3(xyz_xyzw).float()

    @staticmethod
    def _nearest_timestamp_index(sorted_ts: list[int], target: int) -> int:
        if len(sorted_ts) == 0:
            raise ValueError("Cannot query nearest timestamp from empty sequence")
        i = bisect_left(sorted_ts, target)
        if i == 0:
            return 0
        if i >= len(sorted_ts):
            return len(sorted_ts) - 1
        before = sorted_ts[i - 1]
        after = sorted_ts[i]
        return i if abs(after - target) < abs(target - before) else i - 1

    def _build_stereo_index(self) -> list[tuple[int, int]]:
        pairs: list[tuple[int, int]] = []
        for l_i, left in enumerate(self.left_msgs):
            r_i = self._nearest_timestamp_index(self.right_ts, left.timestamp_ns)
            if abs(self.right_ts[r_i] - left.timestamp_ns) > 5_000_000:
                continue
            pairs.append((l_i, r_i))
        if len(pairs) == 0:
            raise RuntimeError("No synchronized stereo pairs found in VIODE bag")
        return pairs

    def _read_rosbag(self, bag_path: Path) -> tuple[list[_StereoMsg], list[_StereoMsg], list[_IMUMsg], list[_OdomMsg]]:
        try:
            from rosbags.highlevel import AnyReader
        except ImportError as exc:
            raise ImportError("VIODE loader requires 'rosbags' package for rosbag parsing") from exc

        left_msgs: list[_StereoMsg] = []
        right_msgs: list[_StereoMsg] = []
        imu_msgs: list[_IMUMsg] = []
        odom_msgs: list[_OdomMsg] = []

        with AnyReader([bag_path]) as reader:
            conns = {c.topic: c for c in reader.connections if c.topic in REQUIRED_TOPICS}
            missing = [topic for topic in REQUIRED_TOPICS if topic not in conns]
            if missing:
                raise KeyError(f"Missing required VIODE topics in rosbag: {missing}")

            selected = [conns[topic] for topic in REQUIRED_TOPICS]
            for conn, timestamp, raw in reader.messages(connections=selected):
                msg = reader.deserialize(raw, conn.msgtype)
                ts = int(timestamp)
                if conn.topic == "/cam0/image_raw":
                    left_msgs.append(_StereoMsg(timestamp_ns=ts, image=self._decode_ros_image(msg)))
                elif conn.topic == "/cam1/image_raw":
                    right_msgs.append(_StereoMsg(timestamp_ns=ts, image=self._decode_ros_image(msg)))
                elif conn.topic == "/imu0":
                    imu_msgs.append(
                        _IMUMsg(
                            timestamp_ns=ts,
                            acc=torch.tensor(
                                [
                                    msg.linear_acceleration.x,
                                    msg.linear_acceleration.y,
                                    msg.linear_acceleration.z,
                                ],
                                dtype=torch.float32,
                            ),
                            gyro=torch.tensor(
                                [
                                    msg.angular_velocity.x,
                                    msg.angular_velocity.y,
                                    msg.angular_velocity.z,
                                ],
                                dtype=torch.float32,
                            ),
                        )
                    )
                elif conn.topic == "/odometry":
                    odom_msgs.append(_OdomMsg(timestamp_ns=ts, pose=self._msg_to_pose(msg)))

        if len(left_msgs) == 0 or len(right_msgs) == 0 or len(imu_msgs) == 0 or len(odom_msgs) == 0:
            raise RuntimeError("Incomplete VIODE rosbag content: one or more required streams are empty")

        left_msgs.sort(key=lambda m: m.timestamp_ns)
        right_msgs.sort(key=lambda m: m.timestamp_ns)
        imu_msgs.sort(key=lambda m: m.timestamp_ns)
        odom_msgs.sort(key=lambda m: m.timestamp_ns)
        return left_msgs, right_msgs, imu_msgs, odom_msgs
