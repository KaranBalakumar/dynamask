"""
VIODE dataset loader — HDF5 edition.

Reads from pre-converted .h5 files (produced by convert_viode_bags.py) so that
RAM usage during training is minimal: only one frame pair is in memory at a
time, regardless of how many sequences or how large the bags were.

Convert bags first:
    micromamba run -n dynamask python -m dynamask_vio.data.convert_viode_bags \\
        --viode_root ./dataset/viode \\
        --out_root   ./dataset/viode_hdf5 \\
        --resize_h 480 --resize_w 640

Then point viode_hdf5_root in your config to ./dataset/viode_hdf5.
"""

import os
import threading

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

from .imu_utils import get_imu_window, compute_gt_preintegration
from .augmentations import VisualAugmentor, IMUAugmentor


# ──────────────────────────────────────────────────────────────────────────────
# Per-thread HDF5 file handle cache
# (h5py handles can't safely be shared across threads/processes; opening lazily
#  per worker avoids pickling issues with DataLoader num_workers > 0)
# ──────────────────────────────────────────────────────────────────────────────
_tls = threading.local()


def _open_h5(path: str) -> h5py.File:
    """Return a thread-local h5py file handle, opening it if necessary."""
    if not hasattr(_tls, "handles"):
        _tls.handles = {}
    if path not in _tls.handles:
        _tls.handles[path] = h5py.File(path, "r")
    return _tls.handles[path]


# ──────────────────────────────────────────────────────────────────────────────
# Metadata loader (loads ONLY small arrays: IMU, GT, timestamps)
# ──────────────────────────────────────────────────────────────────────────────

def _load_hdf5_meta(h5_path: str) -> dict:
    """Load everything except image pixels from an HDF5 sequence file.

    Returns a dict with:
        h5_path       – path to the file (images/seg read lazily in __getitem__)
        n_frames      – number of image frames
        n_seg         – number of seg frames (may differ from n_frames)
        img_timestamps – float64 [N]
        seg_timestamps – float64 [S]
        seg_encoding   – str
        imu_timestamps – float64 [M]
        imu_data       – float64 [M, 6]
        gt             – None | {"timestamps": [K], "poses": [K,7]}
    """
    with h5py.File(h5_path, "r") as hf:
        n_frames = hf["images"].shape[0]
        n_seg = hf["seg"].shape[0] if "seg" in hf else 0
        img_ts = hf["image_timestamps"][:] if "image_timestamps" in hf else np.arange(n_frames, dtype=np.float64)
        seg_ts = hf["seg_timestamps"][:] if "seg_timestamps" in hf else np.arange(n_seg, dtype=np.float64)
        seg_enc = str(hf.attrs.get("seg_encoding", "mono8"))

        imu_ts = hf["imu_timestamps"][:] if "imu_timestamps" in hf else np.empty(0, np.float64)
        imu_data = hf["imu_data"][:] if "imu_data" in hf else np.empty((0, 6), np.float64)

        gt = None
        if "gt_timestamps" in hf and "gt_poses" in hf:
            gt = {
                "timestamps": hf["gt_timestamps"][:],
                "poses":      hf["gt_poses"][:],
            }

    return {
        "h5_path":        h5_path,
        "n_frames":       n_frames,
        "n_seg":          n_seg,
        "img_timestamps": img_ts,
        "seg_timestamps": seg_ts,
        "seg_encoding":   seg_enc,
        "imu_timestamps": imu_ts,
        "imu_data":       imu_data,
        "gt":             gt,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Dataset
# ──────────────────────────────────────────────────────────────────────────────

class VIODEDataset(Dataset):
    """PyTorch dataset for VIODE.  Each sample is a consecutive frame pair.

    Requires bags to be pre-converted to HDF5 via convert_viode_bags.py.
    RAM footprint: O(IMU + GT) per sequence (small), images loaded on demand.
    """

    def __init__(self, root: str, sequences: list, cfg: dict,
                 split: str = "train"):
        """
        Args:
            root      : path to the HDF5 root (viode_hdf5_root in config)
            sequences : list of .h5 paths relative to root, or absolute paths
            cfg       : full config dict
            split     : "train", "val", or "test"
        """
        self.cfg = cfg
        self.split = split
        self.max_imu = cfg.get("data", {}).get("imu_max_window_size", 15)

        cam_cfg = cfg.get("camera", {})
        fx = cam_cfg.get("fx", 320.0)
        fy = cam_cfg.get("fy", 320.0)
        cx = cam_cfg.get("cx", 320.0)
        cy = cam_cfg.get("cy", 240.0)
        self.intrinsics = np.array([
            [fx, 0.0, cx],
            [0.0, fy, cy],
            [0.0, 0.0, 1.0],
        ], dtype=np.float32)

        static_ids = cfg.get("segmentation", {}).get("static_class_ids",
                                                      [0, 1, 2, 3, 4, 5, 6])
        self.static_class_ids = set(static_ids)
        self.seg_mode = cfg.get("segmentation", {}).get("mode", "auto")

        palette_cfg = cfg.get("segmentation", {}).get("palette_to_class_id", {})
        self.palette_to_class_id = {}
        for k, v in palette_cfg.items():
            if isinstance(k, str):
                parts = [p.strip() for p in k.split(",")]
                if len(parts) == 3:
                    try:
                        rgb = tuple(int(x) for x in parts)
                        self.palette_to_class_id[rgb] = int(v)
                    except ValueError:
                        pass

        self._auto_palette_map = {}

        self.augment_visual = VisualAugmentor(cfg) if split == "train" else None
        self.augment_imu = IMUAugmentor(cfg) if split == "train" else None

        # Build frame-pair index (stored as (seq_idx, frame_idx) tuples)
        self.pairs = []
        self.sequences = []  # list of metadata dicts

        for seq_path in sequences:
            if not os.path.isabs(seq_path):
                h5_path = os.path.join(root, seq_path)
                if not h5_path.endswith(".h5"):
                    # Allow callers to pass paths without extension
                    candidates = [
                        h5_path + ".h5",
                        os.path.join(root, seq_path.replace("/", "_") + ".h5"),
                    ]
                    h5_path = next((c for c in candidates if os.path.exists(c)), h5_path)
            else:
                h5_path = seq_path

            if not os.path.exists(h5_path):
                print(f"[VIODE] Warning: HDF5 not found: {h5_path}, skipping")
                print(f"[VIODE]   Run convert_viode_bags.py first.")
                continue

            print(f"[VIODE] Loading metadata {h5_path}...")
            meta = _load_hdf5_meta(h5_path)
            seq_idx = len(self.sequences)
            self.sequences.append(meta)

            n_pairs = min(meta["n_frames"], meta["n_seg"]) - 1
            for i in range(n_pairs):
                self.pairs.append((seq_idx, i))

        print(f"[VIODE] {split}: {len(self.pairs)} frame pairs from "
              f"{len(self.sequences)} sequences")

    def __len__(self):
        return len(self.pairs)

    # ── Segmentation helpers ────────────────────────────────────────────────

    def _seg_rgb_to_id(self, seg_rgb: np.ndarray) -> np.ndarray:
        H, W, _ = seg_rgb.shape
        unique_colors, inv = np.unique(seg_rgb.reshape(-1, 3), axis=0,
                                       return_inverse=True)

        lut = np.zeros(len(unique_colors), dtype=np.int32)
        if self.palette_to_class_id:
            for idx, c in enumerate(unique_colors):
                key = (int(c[0]), int(c[1]), int(c[2]))
                lut[idx] = int(self.palette_to_class_id.get(key, -1))
            for idx in np.where(lut < 0)[0]:
                key = tuple(int(x) for x in unique_colors[idx])
                if key not in self._auto_palette_map:
                    self._auto_palette_map[key] = len(self._auto_palette_map) + 1000
                lut[idx] = self._auto_palette_map[key]
        else:
            for idx, c in enumerate(unique_colors):
                key = (int(c[0]), int(c[1]), int(c[2]))
                if key not in self._auto_palette_map:
                    self._auto_palette_map[key] = len(self._auto_palette_map)
                lut[idx] = self._auto_palette_map[key]

        return lut[inv].reshape(H, W).astype(np.int32)

    def _decode_segmentation(self, seg: np.ndarray, encoding: str) -> np.ndarray:
        """Decode stored seg array to integer class-id map [H, W]."""
        mode = (self.seg_mode or "auto").lower()
        enc = (encoding or "").lower()

        if mode == "id":
            return seg[:, :, 0].astype(np.int32) if seg.ndim == 3 else seg.astype(np.int32)

        if mode == "rgb":
            return seg.astype(np.int32) if seg.ndim == 2 else self._seg_rgb_to_id(seg[:, :, :3])

        # auto
        if enc in ("rgb8", "bgr8") and seg.ndim == 3:
            return self._seg_rgb_to_id(seg[:, :, :3])
        if seg.ndim == 3:
            return seg[:, :, 0].astype(np.int32)
        return seg.astype(np.int32)

    def _make_dynamic_mask(self, seg: np.ndarray) -> np.ndarray:
        mask = np.ones_like(seg, dtype=np.float32)
        for cls_id in self.static_class_ids:
            mask[seg == cls_id] = 0.0
        return mask

    # ── Main item accessor ──────────────────────────────────────────────────

    def __getitem__(self, idx):
        seq_idx, frame_idx = self.pairs[idx]
        meta = self.sequences[seq_idx]

        # Open HDF5 lazily per thread (safe for DataLoader workers)
        hf = _open_h5(meta["h5_path"])

        # Images (read single frames from disk)
        img_prev = hf["images"][frame_idx].astype(np.float32)       # [H, W, 3]
        img_curr = hf["images"][frame_idx + 1].astype(np.float32)

        # Segmentation
        seg_raw = hf["seg"][frame_idx + 1]                           # [H,W] or [H,W,3]
        seg = self._decode_segmentation(seg_raw, meta["seg_encoding"])
        gt_mask = self._make_dynamic_mask(seg)

        # IMU window
        t_prev = meta["img_timestamps"][frame_idx]
        t_curr = meta["img_timestamps"][frame_idx + 1]
        imu_window, imu_valid = get_imu_window(
            meta["imu_timestamps"], meta["imu_data"],
            t_prev, t_curr, max_samples=self.max_imu,
        )

        # GT preintegration
        gt_preint = {
            "delta_R": np.eye(3, dtype=np.float32),
            "delta_v": np.zeros(3, dtype=np.float32),
            "delta_p": np.zeros(3, dtype=np.float32),
        }
        if meta["gt"] is not None:
            gt = meta["gt"]
            idx_i = int(np.clip(np.searchsorted(gt["timestamps"], t_prev),
                                0, len(gt["timestamps"]) - 1))
            idx_j = int(np.clip(np.searchsorted(gt["timestamps"], t_curr),
                                0, len(gt["timestamps"]) - 1))
            if idx_i != idx_j:
                gt_preint = compute_gt_preintegration(
                    gt["poses"], gt["timestamps"], idx_i, idx_j)

        # Augmentation
        flipped = [False]
        if self.augment_visual is not None:
            img_prev, img_curr, gt_mask = self.augment_visual(
                img_prev, img_curr, gt_mask, flipped)
        if self.augment_imu is not None:
            imu_window, imu_valid = self.augment_imu(
                imu_window, imu_valid, horizontally_flipped=flipped[0])

        # Tensors
        img_prev = torch.from_numpy(img_prev).permute(2, 0, 1).float()
        img_curr = torch.from_numpy(img_curr).permute(2, 0, 1).float()
        gt_mask  = torch.from_numpy(gt_mask).unsqueeze(0).float()
        imu_window = torch.from_numpy(imu_window).float()
        imu_valid  = torch.from_numpy(imu_valid.astype(np.float32)).bool()
        gt_R = torch.from_numpy(gt_preint["delta_R"]).float()
        gt_v = torch.from_numpy(gt_preint["delta_v"]).float()
        gt_p = torch.from_numpy(gt_preint["delta_p"]).float()
        intrinsics = torch.from_numpy(self.intrinsics).float()

        return {
            "img_prev":   img_prev,
            "img_curr":   img_curr,
            "gt_mask":    gt_mask,
            "imu_window": imu_window,
            "imu_mask":   imu_valid,
            "gt_R":       gt_R,
            "gt_v":       gt_v,
            "gt_p":       gt_p,
            "intrinsics": intrinsics,
            "has_flow":   False,
        }
