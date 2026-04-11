"""
TartanAir V2 dataset loader.

Reads from the TartanAir V2 directory structure:
  <env>/<difficulty>/
    image_lcam_front/  — RGB images
    seg_lcam_front/    — semantic segmentation
    imu/               — simulated IMU
    flow/              — ground truth optical flow
    pose_lcam_front.txt — poses (NED frame)
"""

import os
import glob
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset

from .imu_utils import get_imu_window, compute_gt_preintegration
from .augmentations import VisualAugmentor, IMUAugmentor


class TartanAirDataset(Dataset):
    """PyTorch dataset for TartanAir V2.

    Provides consecutive frame pairs with IMU, GT masks, and optional GT flow.
    """

    def __init__(self, root: str, environments: list, difficulties: list,
                 cfg: dict, split: str = "train"):
        self.cfg = cfg
        self.root = root
        self.split = split
        self.max_imu = cfg.get("data", {}).get("imu_max_window_size", 15)

        seg_cfg = cfg.get("segmentation", {})
        self.dynamic_class_ids = set(seg_cfg.get("dynamic_class_ids",
                                                   [6, 7, 8, 9, 10, 11]))

        self.augment_visual = VisualAugmentor(cfg) if split == "train" else None
        self.augment_imu = IMUAugmentor(cfg) if split == "train" else None

        # Discover sequences
        self.pairs = []
        self.seq_cache = {}

        for env in environments:
            for diff in difficulties:
                seq_dir = os.path.join(root, env, diff)
                if not os.path.isdir(seq_dir):
                    continue

                img_dir = os.path.join(seq_dir, "image_lcam_front")
                if not os.path.isdir(img_dir):
                    continue

                img_files = sorted(glob.glob(os.path.join(img_dir, "*.png")))
                if len(img_files) < 2:
                    continue

                seq_key = f"{env}/{diff}"
                self.seq_cache[seq_key] = {
                    "dir": seq_dir,
                    "img_files": img_files,
                    "poses": self._load_poses(seq_dir),
                    "imu": self._load_imu(seq_dir),
                }

                for i in range(len(img_files) - 1):
                    self.pairs.append((seq_key, i))

        print(f"[TartanAir] {split}: {len(self.pairs)} frame pairs from "
              f"{len(self.seq_cache)} sequences")

    @staticmethod
    def _load_poses(seq_dir: str) -> np.ndarray:
        """Load poses from pose_lcam_front.txt.
        Format: each line is tx ty tz qx qy qz qw."""
        pose_file = os.path.join(seq_dir, "pose_lcam_front.txt")
        if os.path.exists(pose_file):
            return np.loadtxt(pose_file, dtype=np.float64)
        return None

    @staticmethod
    def _load_imu(seq_dir: str) -> dict:
        """Load IMU data from imu/ directory."""
        imu_dir = os.path.join(seq_dir, "imu")
        imu_file = os.path.join(imu_dir, "imu_data.txt")
        if os.path.exists(imu_file):
            raw = np.loadtxt(imu_file, dtype=np.float64)
            # Expect columns: timestamp, ax, ay, az, gx, gy, gz
            if raw.ndim == 2 and raw.shape[1] >= 7:
                return {
                    "timestamps": raw[:, 0],
                    "data": raw[:, 1:7],
                }
        # Fallback: try CSV
        imu_csv = os.path.join(imu_dir, "imu0.csv")
        if os.path.exists(imu_csv):
            raw = np.genfromtxt(imu_csv, delimiter=",", skip_header=1,
                                dtype=np.float64)
            if raw.ndim == 2 and raw.shape[1] >= 7:
                return {
                    "timestamps": raw[:, 0],
                    "data": raw[:, 1:7],
                }
        return None

    def __len__(self):
        return len(self.pairs)

    def _make_dynamic_mask(self, seg: np.ndarray) -> np.ndarray:
        mask = np.zeros(seg.shape[:2], dtype=np.float32)
        for cls_id in self.dynamic_class_ids:
            mask[seg == cls_id] = 1.0
        return mask

    def __getitem__(self, idx):
        seq_key, frame_idx = self.pairs[idx]
        seq = self.seq_cache[seq_key]
        seq_dir = seq["dir"]

        # Load images
        img_prev = cv2.imread(seq["img_files"][frame_idx])
        img_curr = cv2.imread(seq["img_files"][frame_idx + 1])
        img_prev = cv2.cvtColor(img_prev, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        img_curr = cv2.cvtColor(img_curr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

        # Load segmentation
        seg_dir = os.path.join(seq_dir, "seg_lcam_front")
        seg_files = sorted(glob.glob(os.path.join(seg_dir, "*.png")))
        if frame_idx + 1 < len(seg_files):
            seg = cv2.imread(seg_files[frame_idx + 1], cv2.IMREAD_UNCHANGED)
            if seg.ndim == 3:
                seg = seg[:, :, 0]
            proxy_mask = self._make_dynamic_mask(seg)
        else:
            proxy_mask = np.zeros(img_curr.shape[:2], dtype=np.float32)

        # Load flow (if available)
        flow_dir = os.path.join(seq_dir, "flow")
        gt_flow = None
        has_flow = False
        flow_files = sorted(glob.glob(os.path.join(flow_dir, "*.npy"))) \
            if os.path.isdir(flow_dir) else []
        if frame_idx < len(flow_files):
            gt_flow = np.load(flow_files[frame_idx]).astype(np.float32)
            has_flow = True

        # IMU window
        imu = seq["imu"]
        if imu is not None:
            # Assume camera timestamps are evenly spaced from IMU start
            cam_rate = self.cfg.get("data", {}).get("camera_rate_hz", 25)
            t_prev = imu["timestamps"][0] + frame_idx / cam_rate
            t_curr = t_prev + 1.0 / cam_rate
            imu_window, imu_valid = get_imu_window(
                imu["timestamps"], imu["data"],
                t_prev, t_curr, max_samples=self.max_imu,
            )
        else:
            imu_window = np.zeros((self.max_imu, 7), dtype=np.float32)
            imu_valid = np.zeros(self.max_imu, dtype=bool)
            imu_valid[0] = True  # at least one sample

        # GT preintegration from poses
        gt_preint = {
            "delta_R": np.eye(3, dtype=np.float32),
            "delta_v": np.zeros(3, dtype=np.float32),
            "delta_p": np.zeros(3, dtype=np.float32),
        }
        if seq["poses"] is not None and frame_idx + 1 < len(seq["poses"]):
            poses = seq["poses"]
            # Synthesise timestamps
            cam_rate = self.cfg.get("data", {}).get("camera_rate_hz", 25)
            timestamps = np.arange(len(poses)) / cam_rate
            gt_preint = compute_gt_preintegration(
                poses, timestamps, frame_idx, frame_idx + 1)

        # Augmentation
        flipped = [False]
        if self.augment_visual is not None:
            img_prev, img_curr, proxy_mask = self.augment_visual(
                img_prev, img_curr, proxy_mask, flipped)

        if self.augment_imu is not None:
            imu_window, imu_valid = self.augment_imu(
                imu_window, imu_valid, horizontally_flipped=flipped[0])

        # Convert to tensors
        img_prev = torch.from_numpy(img_prev).permute(2, 0, 1).float()
        img_curr = torch.from_numpy(img_curr).permute(2, 0, 1).float()
        imu_window = torch.from_numpy(imu_window).float()
        imu_valid = torch.from_numpy(imu_valid.astype(np.float32)).bool()

        gt_R = torch.from_numpy(gt_preint["delta_R"]).float()
        gt_v = torch.from_numpy(gt_preint["delta_v"]).float()
        gt_p = torch.from_numpy(gt_preint["delta_p"]).float()

        result = {
            "img_prev": img_prev,
            "img_curr": img_curr,
            "imu_window": imu_window,
            "imu_mask": imu_valid,
            "gt_R": gt_R,
            "gt_v": gt_v,
            "gt_p": gt_p,
            "has_flow": has_flow,
        }

        if gt_flow is not None:
            result["gt_flow"] = torch.from_numpy(gt_flow).permute(2, 0, 1).float()

        return result
