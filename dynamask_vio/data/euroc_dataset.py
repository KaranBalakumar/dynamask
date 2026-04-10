"""
EuRoC MAV dataset loader (ASL format).

Used for:
  1. IMU Head pretraining (Phase 1) — GT biases + GT trajectory
  2. Static baseline evaluation — no dynamic objects

Directory layout per sequence:
  <seq>/
    cam0/data/         — grayscale images (752×480)
    cam0/data.csv      — image timestamps
    imu0/data.csv      — IMU at 200Hz
    state_groundtruth_estimate0/data.csv — GT poses, velocities, biases
"""

import os
import glob
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset

from .imu_utils import get_imu_window, compute_gt_preintegration
from .augmentations import IMUAugmentor


class EuRoCDataset(Dataset):
    """EuRoC dataset for IMU head pretraining.

    Returns IMU windows + GT preintegration (no mask supervision).
    Images are loaded but used only for the visual backbone (with frozen weights
    in Phase 1).
    """

    def __init__(self, root: str, sequences: list, cfg: dict,
                 split: str = "train"):
        self.cfg = cfg
        self.root = root
        self.split = split
        self.max_imu = cfg.get("data", {}).get("imu_max_window_size", 15)

        # Camera intrinsics from config (EuRoC cam0 defaults)
        cam_cfg = cfg.get("camera", {})
        fx = cam_cfg.get("fx", 458.654)
        fy = cam_cfg.get("fy", 457.296)
        cx = cam_cfg.get("cx", 367.215)
        cy = cam_cfg.get("cy", 248.375)
        self.intrinsics = np.array([
            [fx, 0.0, cx],
            [0.0, fy, cy],
            [0.0, 0.0, 1.0],
        ], dtype=np.float32)

        self.augment_imu = IMUAugmentor(cfg) if split == "train" else None

        self.pairs = []
        self.sequences_data = []

        for seq_name in sequences:
            seq_dir = self._find_sequence_dir(root, seq_name)
            if seq_dir is None:
                print(f"[EuRoC] Warning: sequence not found: {seq_name}")
                continue

            seq_data = self._load_sequence(seq_dir)
            if seq_data is None:
                continue

            seq_idx = len(self.sequences_data)
            self.sequences_data.append(seq_data)

            n_frames = len(seq_data["image_timestamps"])
            for i in range(n_frames - 1):
                self.pairs.append((seq_idx, i))

        print(f"[EuRoC] {split}: {len(self.pairs)} frame pairs from "
              f"{len(self.sequences_data)} sequences")

    @staticmethod
    def _find_sequence_dir(root: str, seq_name: str):
        """Resolve a sequence name to its mav0 directory.

        Handles multiple EuRoC directory layouts:
          - root/seq_name/mav0/           (flat)
          - root/seq_name/                (no mav0 wrapper)
          - root/*/seq_name/mav0/         (e.g. machine_hall/MH_01_easy/mav0/)
          - root/*/seq_name/seq_name/mav0/ (double-nested from zip extraction)
        """
        candidates = [
            os.path.join(root, seq_name, "mav0"),
            os.path.join(root, seq_name),
        ]

        # Glob for nested layouts: root/*/seq_name/mav0 or root/*/seq_name/seq_name/mav0
        for parent in glob.glob(os.path.join(root, "*")):
            if os.path.isdir(parent):
                candidates.append(os.path.join(parent, seq_name, "mav0"))
                candidates.append(os.path.join(parent, seq_name, seq_name, "mav0"))
                candidates.append(os.path.join(parent, seq_name))
                candidates.append(os.path.join(parent, seq_name, seq_name))

        for path in candidates:
            if os.path.isdir(path):
                # Verify it has cam0 data
                cam_csv = os.path.join(path, "cam0", "data.csv")
                if os.path.exists(cam_csv):
                    return path
                # Maybe it's one level up from mav0
                cam_csv2 = os.path.join(path, "mav0", "cam0", "data.csv")
                if os.path.exists(cam_csv2):
                    return os.path.join(path, "mav0")

        return None

    @staticmethod
    def _load_sequence(seq_dir: str) -> dict:
        """Load one EuRoC sequence."""
        # Image timestamps
        cam_csv = os.path.join(seq_dir, "cam0", "data.csv")
        if not os.path.exists(cam_csv):
            print(f"[EuRoC] No cam0/data.csv in {seq_dir}")
            return None

        cam_data = np.genfromtxt(cam_csv, delimiter=",", skip_header=1,
                                  dtype=str)
        if cam_data.ndim == 1:
            cam_data = cam_data.reshape(1, -1)
        image_timestamps = cam_data[:, 0].astype(np.float64) / 1e9  # ns → s
        image_filenames = cam_data[:, 1] if cam_data.shape[1] > 1 else None

        # IMU data
        imu_csv = os.path.join(seq_dir, "imu0", "data.csv")
        if not os.path.exists(imu_csv):
            print(f"[EuRoC] No imu0/data.csv in {seq_dir}")
            return None

        imu_raw = np.genfromtxt(imu_csv, delimiter=",", skip_header=1,
                                 dtype=np.float64)
        imu_timestamps = imu_raw[:, 0] / 1e9
        # EuRoC IMU format: timestamp, wx, wy, wz, ax, ay, az
        # We want: ax, ay, az, gx, gy, gz
        imu_data = np.zeros((len(imu_raw), 6), dtype=np.float64)
        imu_data[:, 0:3] = imu_raw[:, 4:7]  # accel
        imu_data[:, 3:6] = imu_raw[:, 1:4]  # gyro

        # Ground truth
        gt_csv = os.path.join(seq_dir, "state_groundtruth_estimate0",
                               "data.csv")
        gt_data = None
        if os.path.exists(gt_csv):
            gt_raw = np.genfromtxt(gt_csv, delimiter=",", skip_header=1,
                                    dtype=np.float64)
            # Format: timestamp, px,py,pz, qw,qx,qy,qz, vx,vy,vz,
            #         bwx,bwy,bwz, bax,bay,baz
            gt_timestamps = gt_raw[:, 0] / 1e9
            # Convert quaternion from (qw,qx,qy,qz) to scipy convention (qx,qy,qz,qw)
            poses = np.zeros((len(gt_raw), 7), dtype=np.float64)
            poses[:, 0:3] = gt_raw[:, 1:4]    # position
            poses[:, 3] = gt_raw[:, 5]         # qx
            poses[:, 4] = gt_raw[:, 6]         # qy
            poses[:, 5] = gt_raw[:, 7]         # qz
            poses[:, 6] = gt_raw[:, 4]         # qw

            velocities = gt_raw[:, 8:11] if gt_raw.shape[1] > 10 else None
            biases_gyro = gt_raw[:, 11:14] if gt_raw.shape[1] > 13 else None
            biases_accel = gt_raw[:, 14:17] if gt_raw.shape[1] > 16 else None

            gt_data = {
                "timestamps": gt_timestamps,
                "poses": poses,
                "velocities": velocities,
                "biases_gyro": biases_gyro,
                "biases_accel": biases_accel,
            }

        return {
            "seq_dir": seq_dir,
            "image_timestamps": image_timestamps,
            "image_filenames": image_filenames,
            "imu_timestamps": imu_timestamps,
            "imu_data": imu_data,
            "gt": gt_data,
        }

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        seq_idx, frame_idx = self.pairs[idx]
        seq = self.sequences_data[seq_idx]

        t_prev = seq["image_timestamps"][frame_idx]
        t_curr = seq["image_timestamps"][frame_idx + 1]

        # Load images (grayscale → 3-channel)
        img_dir = os.path.join(seq["seq_dir"], "cam0", "data")
        if seq["image_filenames"] is not None:
            img_prev_path = os.path.join(img_dir,
                                          seq["image_filenames"][frame_idx].strip())
            img_curr_path = os.path.join(img_dir,
                                          seq["image_filenames"][frame_idx + 1].strip())
        else:
            # Try timestamp-based naming
            img_files = sorted(glob.glob(os.path.join(img_dir, "*.png")))
            img_prev_path = img_files[frame_idx] if frame_idx < len(img_files) else ""
            img_curr_path = img_files[frame_idx + 1] if frame_idx + 1 < len(img_files) else ""

        # Check file existence before imread to suppress cv2 warnings for
        # missing images (incomplete EuRoC extractions are common).
        img_prev = cv2.imread(img_prev_path, cv2.IMREAD_GRAYSCALE) \
            if os.path.isfile(img_prev_path) else None
        img_curr = cv2.imread(img_curr_path, cv2.IMREAD_GRAYSCALE) \
            if os.path.isfile(img_curr_path) else None

        if img_prev is None:
            img_prev = np.zeros((480, 752), dtype=np.uint8)
        if img_curr is None:
            img_curr = np.zeros((480, 752), dtype=np.uint8)

        # Convert grayscale to 3-channel, keep [0, 255] range
        # (model's forward() does its own normalization: 2*(x/255)-0.5)
        img_prev = np.stack([img_prev] * 3, axis=-1).astype(np.float32)
        img_curr = np.stack([img_curr] * 3, axis=-1).astype(np.float32)

        # IMU window
        imu_window, imu_valid = get_imu_window(
            seq["imu_timestamps"], seq["imu_data"],
            t_prev, t_curr, max_samples=self.max_imu,
        )

        # GT preintegration from ground truth trajectory
        gt_preint = {
            "delta_R": np.eye(3, dtype=np.float32),
            "delta_v": np.zeros(3, dtype=np.float32),
            "delta_p": np.zeros(3, dtype=np.float32),
        }

        if seq["gt"] is not None:
            gt = seq["gt"]
            # Find nearest GT pose indices to camera timestamps
            idx_i = np.searchsorted(gt["timestamps"], t_prev)
            idx_j = np.searchsorted(gt["timestamps"], t_curr)
            idx_i = np.clip(idx_i, 0, len(gt["timestamps"]) - 1)
            idx_j = np.clip(idx_j, 0, len(gt["timestamps"]) - 1)

            if idx_i != idx_j:
                gt_preint = compute_gt_preintegration(
                    gt["poses"], gt["timestamps"], idx_i, idx_j)

        # IMU augmentation (no visual augmentation for EuRoC — Phase 1 is IMU only)
        if self.augment_imu is not None:
            imu_window, imu_valid = self.augment_imu(imu_window, imu_valid)

        # EuRoC has no dynamic objects → mask is all zeros
        H, W = img_prev.shape[:2]
        gt_mask = np.zeros((H, W), dtype=np.float32)

        # Convert to tensors
        img_prev = torch.from_numpy(img_prev).permute(2, 0, 1).float()
        img_curr = torch.from_numpy(img_curr).permute(2, 0, 1).float()
        gt_mask = torch.from_numpy(gt_mask).unsqueeze(0).float()
        imu_window = torch.from_numpy(imu_window).float()
        imu_valid = torch.from_numpy(imu_valid.astype(np.float32)).bool()

        gt_R = torch.from_numpy(gt_preint["delta_R"]).float()
        gt_v = torch.from_numpy(gt_preint["delta_v"]).float()
        gt_p = torch.from_numpy(gt_preint["delta_p"]).float()

        intrinsics = torch.from_numpy(self.intrinsics).float()  # [3, 3]

        return {
            "img_prev": img_prev,
            "img_curr": img_curr,
            "gt_mask": gt_mask,
            "imu_window": imu_window,
            "imu_mask": imu_valid,
            "gt_R": gt_R,
            "gt_v": gt_v,
            "gt_p": gt_p,
            "intrinsics": intrinsics,
            "has_flow": False,
        }
