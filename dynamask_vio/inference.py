"""Single-sequence inference script for DynaMask V2.

Supports two input modes:
1) Images + IMU CSV
2) ROS bag (VIODE or EuRoC format)

Outputs:
  output_dir/
    masks/        — binary mask PNGs (white = dynamic)
    overlays/     — mask overlaid on input images (green = dynamic)
    probmaps/     — raw probability heatmaps
    results.json  — per-frame metadata (timestamps, dynamic ratio, IMU preint)

Usage:
    # Mode 1: images + IMU csv
    python -m dynamask_vio.inference \
        --checkpoint checkpoints/best.ckpt \
        --images /path/to/image_dir \
        --imu /path/to/imu.csv \
        --output /path/to/output_dir

    # Mode 2: ROS bag
    python -m dynamask_vio.inference \
        --checkpoint checkpoints/best.ckpt \
        --bag dataset/viode/parking_lot/3_high.bag \
        --output /path/to/output_dir
"""

import argparse
import glob
import os
import json

import cv2
import numpy as np
import torch
import yaml

from .models import DynaMaskVIO
from .data.imu_utils import get_imu_window
from .data.viode_dataset import _load_rosbag


def _load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def load_model(checkpoint_path: str, config_path: str,
               device: torch.device) -> DynaMaskVIO:
    cfg = _load_config(config_path)
    cfg.setdefault("model", {})["raft_checkpoint"] = None

    model = DynaMaskVIO(cfg)

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state = ckpt.get("state_dict", ckpt)
    state = {k.replace("model.", "", 1): v for k, v in state.items()
             if k.startswith("model.")} or state
    missing, unexpected = model.load_state_dict(state, strict=False)
    loaded = len(model.state_dict()) - len(missing)
    print(
        f"[Checkpoint] Loaded from {checkpoint_path} "
        f"(loaded={loaded}, missing={len(missing)}, unexpected={len(unexpected)})"
    )
    model = model.to(device).eval()
    return model


def _forward_pair(model: DynaMaskVIO,
                  img_prev: np.ndarray,
                  img_curr: np.ndarray,
                  imu_timestamps: np.ndarray,
                  imu_data: np.ndarray,
                  t_prev: float,
                  t_curr: float,
                  device: torch.device,
                  threshold: float,
                  max_imu: int):
    """Run inference on a single frame pair.

    Args:
        img_prev, img_curr: [H, W, 3] float32 in [0, 255]
    """
    imu_window, imu_valid = get_imu_window(
        imu_timestamps, imu_data,
        t_prev, t_curr,
        max_samples=max_imu,
    )

    img_p = torch.from_numpy(img_prev).permute(2, 0, 1).unsqueeze(0).to(device)
    img_c = torch.from_numpy(img_curr).permute(2, 0, 1).unsqueeze(0).to(device)
    imu_w = torch.from_numpy(imu_window).unsqueeze(0).to(device)
    imu_m = torch.from_numpy(imu_valid.astype(np.float32)).bool().unsqueeze(0).to(device)

    outputs = model(img_p, img_c, imu_w, imu_m)

    mask_prob = outputs["dynamic_mask"][0, 0].cpu().numpy()
    mask_bin = (mask_prob > threshold).astype(np.uint8) * 255

    summary = {
        "delta_R": outputs["delta_R"][0].cpu().numpy().tolist(),
        "delta_v": outputs["delta_v"][0].cpu().numpy().tolist(),
        "delta_p": outputs["delta_p"][0].cpu().numpy().tolist(),
        "mean_mask_prob": float(mask_prob.mean()),
        "dynamic_pixel_ratio": float((mask_prob > threshold).mean()),
    }
    return mask_bin, mask_prob, summary


def _make_overlay(image_uint8: np.ndarray, mask_prob: np.ndarray,
                  threshold: float = 0.5,
                  color: tuple = (0, 255, 0),
                  alpha: float = 0.4) -> np.ndarray:
    """Overlay dynamic mask on image."""
    overlay = image_uint8.copy()
    mask_bool = mask_prob > threshold
    for c in range(3):
        overlay[:, :, c] = np.where(
            mask_bool,
            np.clip(image_uint8[:, :, c] * (1 - alpha) + color[c] * alpha, 0, 255),
            image_uint8[:, :, c],
        )
    return overlay.astype(np.uint8)


def _prob_to_heatmap(mask_prob: np.ndarray) -> np.ndarray:
    """Convert probability map to a colored heatmap."""
    prob_uint8 = np.clip(mask_prob * 255, 0, 255).astype(np.uint8)
    return cv2.applyColorMap(prob_uint8, cv2.COLORMAP_TURBO)


def _setup_output_dirs(output_dir: str, save_overlays: bool, save_probmaps: bool):
    os.makedirs(os.path.join(output_dir, "masks"), exist_ok=True)
    if save_overlays:
        os.makedirs(os.path.join(output_dir, "overlays"), exist_ok=True)
    if save_probmaps:
        os.makedirs(os.path.join(output_dir, "probmaps"), exist_ok=True)


def run_inference(model: DynaMaskVIO, image_dir: str, imu_file: str,
                   output_dir: str, device: torch.device,
                   threshold: float = 0.5, max_imu: int = 15,
                   max_frames: int = 0,
                   save_overlays: bool = True,
                   save_probmaps: bool = True):
    """Run inference on a sequence of images + IMU CSV."""
    _setup_output_dirs(output_dir, save_overlays, save_probmaps)

    img_paths = sorted(glob.glob(os.path.join(image_dir, "*.png")) +
                        glob.glob(os.path.join(image_dir, "*.jpg")))
    if len(img_paths) < 2:
        print(f"Need at least 2 images, found {len(img_paths)}")
        return

    imu_raw = np.genfromtxt(imu_file, delimiter=",", skip_header=1,
                             dtype=np.float64)
    if imu_raw.ndim == 1:
        imu_raw = imu_raw.reshape(1, -1)
    imu_timestamps = imu_raw[:, 0]
    imu_data = imu_raw[:, 1:7]

    n_frames = len(img_paths)
    t_start = imu_timestamps[0]
    t_end = imu_timestamps[-1]
    cam_timestamps = np.linspace(t_start, t_end, n_frames)

    if max_frames > 0:
        n_frames = min(n_frames, max_frames)
        img_paths = img_paths[:n_frames]
        cam_timestamps = cam_timestamps[:n_frames]

    results = []

    with torch.no_grad():
        for i in range(n_frames - 1):
            img_prev_bgr = cv2.imread(img_paths[i])
            img_curr_bgr = cv2.imread(img_paths[i + 1])
            img_prev = cv2.cvtColor(img_prev_bgr, cv2.COLOR_BGR2RGB).astype(np.float32)
            img_curr = cv2.cvtColor(img_curr_bgr, cv2.COLOR_BGR2RGB).astype(np.float32)

            mask_bin, mask_prob, summary = _forward_pair(
                model=model,
                img_prev=img_prev,
                img_curr=img_curr,
                imu_timestamps=imu_timestamps,
                imu_data=imu_data,
                t_prev=cam_timestamps[i],
                t_curr=cam_timestamps[i + 1],
                device=device,
                threshold=threshold,
                max_imu=max_imu,
            )

            cv2.imwrite(
                os.path.join(output_dir, "masks", f"{i:06d}.png"), mask_bin)

            if save_overlays:
                img_curr_uint8 = np.clip(img_curr, 0, 255).astype(np.uint8)
                overlay = _make_overlay(img_curr_uint8, mask_prob, threshold)
                cv2.imwrite(
                    os.path.join(output_dir, "overlays", f"{i:06d}.png"),
                    cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

            if save_probmaps:
                heatmap = _prob_to_heatmap(mask_prob)
                cv2.imwrite(
                    os.path.join(output_dir, "probmaps", f"{i:06d}.png"), heatmap)

            results.append({"frame": i, **summary})

            if (i + 1) % 50 == 0 or i == n_frames - 2:
                print(f"  Frame {i + 1}/{n_frames - 1}, "
                      f"dynamic: {results[-1]['dynamic_pixel_ratio']:.2%}")

    summary_path = os.path.join(output_dir, "results.json")
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nSaved {len(results)} masks to {output_dir}/masks/")
    if save_overlays:
        print(f"Overlays: {output_dir}/overlays/")
    if save_probmaps:
        print(f"Probability maps: {output_dir}/probmaps/")
    print(f"Summary: {summary_path}")


def run_inference_rosbag(model: DynaMaskVIO, bag_path: str,
                         output_dir: str, device: torch.device,
                         threshold: float = 0.5, max_imu: int = 15,
                         max_frames: int = 0,
                         save_overlays: bool = True,
                         save_probmaps: bool = True):
    """Run inference directly from a ROS bag (VIODE or EuRoC)."""
    if not os.path.exists(bag_path):
        print(f"Bag not found: {bag_path}")
        return

    _setup_output_dirs(output_dir, save_overlays, save_probmaps)

    seq = _load_rosbag(bag_path)
    images = seq["images"]
    imu_timestamps = seq["imu_timestamps"]
    imu_data = seq["imu_data"]

    if len(images) < 2:
        print(f"Need at least 2 images in bag, found {len(images)}")
        return

    n_frames = len(images)
    if max_frames > 0:
        n_frames = min(n_frames, max_frames)
        images = images[:n_frames]

    results = []
    with torch.no_grad():
        for i in range(n_frames - 1):
            t_prev, img_prev_raw = images[i]
            t_curr, img_curr_raw = images[i + 1]

            # Keep images in [0, 255] float32 (model does its own normalization)
            img_prev = img_prev_raw.astype(np.float32)
            img_curr = img_curr_raw.astype(np.float32)

            mask_bin, mask_prob, summary = _forward_pair(
                model=model,
                img_prev=img_prev,
                img_curr=img_curr,
                imu_timestamps=imu_timestamps,
                imu_data=imu_data,
                t_prev=t_prev,
                t_curr=t_curr,
                device=device,
                threshold=threshold,
                max_imu=max_imu,
            )

            cv2.imwrite(
                os.path.join(output_dir, "masks", f"{i:06d}.png"), mask_bin)

            if save_overlays:
                img_curr_uint8 = np.clip(img_curr, 0, 255).astype(np.uint8)
                overlay = _make_overlay(img_curr_uint8, mask_prob, threshold)
                cv2.imwrite(
                    os.path.join(output_dir, "overlays", f"{i:06d}.png"),
                    cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

            if save_probmaps:
                heatmap = _prob_to_heatmap(mask_prob)
                cv2.imwrite(
                    os.path.join(output_dir, "probmaps", f"{i:06d}.png"), heatmap)

            results.append({
                "frame": i,
                "timestamp_prev": float(t_prev),
                "timestamp_curr": float(t_curr),
                **summary,
            })

            if (i + 1) % 50 == 0 or i == n_frames - 2:
                print(f"  Frame {i + 1}/{n_frames - 1}, "
                      f"dynamic: {results[-1]['dynamic_pixel_ratio']:.2%}")

    summary_path = os.path.join(output_dir, "results.json")
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nSaved {len(results)} masks to {output_dir}/masks/")
    if save_overlays:
        print(f"Overlays: {output_dir}/overlays/")
    if save_probmaps:
        print(f"Probability maps: {output_dir}/probmaps/")
    print(f"Summary: {summary_path}")


def main():
    parser = argparse.ArgumentParser(description="DynaMask V2 Inference")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--config", type=str,
                        default="dynamask_vio/configs/default.yaml")
    parser.add_argument("--images", type=str,
                        help="Directory of input images (sorted by name)")
    parser.add_argument("--imu", type=str,
                        help="IMU CSV file (timestamp, ax, ay, az, gx, gy, gz)")
    parser.add_argument("--bag", type=str,
                        help="ROS bag path for direct bag inference")
    parser.add_argument("--output", type=str, default="./output")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--max-frames", type=int, default=0,
                        help="Optional cap for number of frames processed")
    parser.add_argument("--no-overlays", action="store_true",
                        help="Skip saving overlay images")
    parser.add_argument("--no-probmaps", action="store_true",
                        help="Skip saving probability heatmaps")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(args.checkpoint, args.config, device)

    save_overlays = not args.no_overlays
    save_probmaps = not args.no_probmaps

    if args.bag:
        print(f"Running bag inference on {args.bag}")
        run_inference_rosbag(
            model=model,
            bag_path=args.bag,
            output_dir=args.output,
            device=device,
            threshold=args.threshold,
            max_frames=args.max_frames,
            save_overlays=save_overlays,
            save_probmaps=save_probmaps,
        )
        return

    if not args.images or not args.imu:
        raise ValueError("Provide either --bag, or both --images and --imu")

    print(f"Running inference on {args.images}")
    run_inference(
        model,
        args.images,
        args.imu,
        args.output,
        device,
        threshold=args.threshold,
        max_frames=args.max_frames,
        save_overlays=save_overlays,
        save_probmaps=save_probmaps,
    )


if __name__ == "__main__":
    main()
