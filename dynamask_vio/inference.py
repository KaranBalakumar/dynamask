"""Single-sequence inference script for DynaMask V2.5."""

from __future__ import annotations

import argparse
import glob
import json
import os

import cv2
import numpy as np
import torch
import yaml

from .data.imu_utils import get_imu_window
from .models import DynaMaskVIO

try:
    from .data.viode_dataset import _load_rosbag  # type: ignore[attr-defined]
except ImportError:
    _load_rosbag = None


def threshold_fixed(score: torch.Tensor, value: float = 0.5) -> torch.Tensor:
    """Fixed scalar threshold on calibrated score."""
    return score > value


def threshold_top_k_percent(score: torch.Tensor, pct: float = 0.10) -> torch.Tensor:
    """Mark top-k percent pixels as dynamic (resolution-independent)."""
    pct = float(max(0.0, min(1.0, pct)))
    if pct <= 0.0:
        return torch.zeros_like(score, dtype=torch.bool)
    k = max(1, int(round(pct * score.numel())))
    thresh = score.reshape(-1).topk(k).values[-1]
    return score >= thresh


def threshold_otsu(score: torch.Tensor) -> torch.Tensor:
    """Otsu thresholding on calibrated score in [0,1]."""
    score_01 = score.clamp(0.0, 1.0)
    hist = torch.histc(score_01.reshape(-1), bins=256, min=0.0, max=1.0)
    hist = hist / hist.sum().clamp(min=1.0)
    omega = torch.cumsum(hist, dim=0)
    bins = torch.linspace(0.0, 1.0, steps=256, device=score.device, dtype=score.dtype)
    mu = torch.cumsum(hist * bins, dim=0)
    mu_t = mu[-1]
    sigma_b2 = (mu_t * omega - mu).pow(2) / (omega * (1.0 - omega) + 1e-8)
    idx = torch.argmax(sigma_b2)
    otsu_thresh = bins[idx]
    return score_01 > otsu_thresh


def threshold_imu_adaptive(
    score: torch.Tensor,
    imu_trace_cov: torch.Tensor,
    imu_trace_cal: float = 1.0,
) -> torch.Tensor:
    """Adaptive threshold scaled by IMU confidence from trace(Sigma_preint)."""
    conf = torch.exp(-imu_trace_cov / max(float(imu_trace_cal), 1e-6)).clamp(0.0, 1.0)
    thresh = 0.70 - 0.40 * conf
    return score > thresh


def _load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def load_model(checkpoint_path: str, config_path: str, device: torch.device) -> DynaMaskVIO:
    cfg = _load_config(config_path)
    cfg.setdefault("model", {})["raft_checkpoint"] = None

    model = DynaMaskVIO(cfg)
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state = ckpt.get("state_dict", ckpt)
    state = {k.replace("model.", "", 1): v for k, v in state.items() if k.startswith("model.")} or state
    missing, unexpected = model.load_state_dict(state, strict=False)
    loaded = len(model.state_dict()) - len(missing)
    print(
        f"[Checkpoint] Loaded from {checkpoint_path} "
        f"(loaded={loaded}, missing={len(missing)}, unexpected={len(unexpected)})"
    )
    model = model.to(device).eval()
    return model


def _apply_threshold(
    score_cal: torch.Tensor,
    mode: str,
    threshold: float,
    topk_pct: float,
    sigma_preint: torch.Tensor,
    imu_trace_cal: float,
) -> torch.Tensor:
    if mode == "fixed":
        return threshold_fixed(score_cal, threshold)
    if mode == "topk":
        return threshold_top_k_percent(score_cal, topk_pct)
    if mode == "otsu":
        return threshold_otsu(score_cal)
    if mode == "imu-adaptive":
        trace_cov = torch.diagonal(sigma_preint, dim1=-2, dim2=-1).sum()
        return threshold_imu_adaptive(score_cal, trace_cov, imu_trace_cal=imu_trace_cal)
    raise ValueError(f"Unknown threshold mode: {mode}")


def _forward_pair(
    model: DynaMaskVIO,
    img_prev: np.ndarray,
    img_curr: np.ndarray,
    imu_timestamps: np.ndarray,
    imu_data: np.ndarray,
    t_prev: float,
    t_curr: float,
    device: torch.device,
    threshold: float,
    max_imu: int,
    threshold_mode: str,
    topk_pct: float,
    imu_trace_cal: float,
):
    imu_window, imu_valid = get_imu_window(
        imu_timestamps,
        imu_data,
        t_prev,
        t_curr,
        max_samples=max_imu,
    )

    img_p = torch.from_numpy(img_prev).permute(2, 0, 1).unsqueeze(0).to(device)
    img_c = torch.from_numpy(img_curr).permute(2, 0, 1).unsqueeze(0).to(device)
    imu_w = torch.from_numpy(imu_window).unsqueeze(0).to(device)
    imu_m = torch.from_numpy(imu_valid.astype(np.float32)).bool().unsqueeze(0).to(device)

    outputs = model(img_p, img_c, imu_w, imu_m)
    score_cal = outputs["score_cal"][0, 0]
    score_logit = outputs["score_logit"][0, 0]
    dynamic_mask = _apply_threshold(
        score_cal,
        threshold_mode,
        threshold,
        topk_pct,
        outputs["Sigma_preint"][0],
        imu_trace_cal,
    )

    mask_bin = dynamic_mask.cpu().numpy().astype(np.uint8) * 255
    score_np = score_cal.cpu().numpy()

    summary = {
        "delta_R": outputs["delta_R"][0].cpu().numpy().tolist(),
        "delta_v": outputs["delta_v"][0].cpu().numpy().tolist(),
        "delta_p": outputs["delta_p"][0].cpu().numpy().tolist(),
        "mean_score_cal": float(score_cal.mean().item()),
        "mean_score_logit": float(score_logit.mean().item()),
        "dynamic_pixel_ratio": float(dynamic_mask.float().mean().item()),
        "threshold_mode": threshold_mode,
    }
    return mask_bin, score_np, dynamic_mask.cpu().numpy().astype(bool), summary


def _make_overlay(
    image_uint8: np.ndarray,
    score_map: np.ndarray,
    *,
    threshold: float = 0.5,
    dynamic_mask: np.ndarray | None = None,
    color: tuple = (0, 255, 0),
    alpha: float = 0.4,
) -> np.ndarray:
    overlay = image_uint8.copy()
    mask_bool = dynamic_mask if dynamic_mask is not None else (score_map > threshold)
    for c in range(3):
        overlay[:, :, c] = np.where(
            mask_bool,
            np.clip(image_uint8[:, :, c] * (1 - alpha) + color[c] * alpha, 0, 255),
            image_uint8[:, :, c],
        )
    return overlay.astype(np.uint8)


def _prob_to_heatmap(score_map: np.ndarray) -> np.ndarray:
    score_uint8 = np.clip(score_map * 255, 0, 255).astype(np.uint8)
    return cv2.applyColorMap(score_uint8, cv2.COLORMAP_TURBO)


def _setup_output_dirs(output_dir: str, save_overlays: bool, save_probmaps: bool):
    os.makedirs(os.path.join(output_dir, "masks"), exist_ok=True)
    if save_overlays:
        os.makedirs(os.path.join(output_dir, "overlays"), exist_ok=True)
    if save_probmaps:
        os.makedirs(os.path.join(output_dir, "probmaps"), exist_ok=True)


def run_inference(
    model: DynaMaskVIO,
    image_dir: str,
    imu_file: str,
    output_dir: str,
    device: torch.device,
    threshold: float = 0.5,
    max_imu: int = 15,
    max_frames: int = 0,
    save_overlays: bool = True,
    save_probmaps: bool = True,
    threshold_mode: str = "fixed",
    topk_pct: float = 0.10,
    imu_trace_cal: float = 1.0,
):
    _setup_output_dirs(output_dir, save_overlays, save_probmaps)

    img_paths = sorted(glob.glob(os.path.join(image_dir, "*.png")) + glob.glob(os.path.join(image_dir, "*.jpg")))
    if len(img_paths) < 2:
        print(f"Need at least 2 images, found {len(img_paths)}")
        return

    imu_raw = np.genfromtxt(imu_file, delimiter=",", skip_header=1, dtype=np.float64)
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

            mask_bin, score_map, dynamic_mask, summary = _forward_pair(
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
                threshold_mode=threshold_mode,
                topk_pct=topk_pct,
                imu_trace_cal=imu_trace_cal,
            )

            cv2.imwrite(os.path.join(output_dir, "masks", f"{i:06d}.png"), mask_bin)

            if save_overlays:
                img_curr_uint8 = np.clip(img_curr, 0, 255).astype(np.uint8)
                overlay = _make_overlay(
                    img_curr_uint8,
                    score_map,
                    threshold=threshold,
                    dynamic_mask=dynamic_mask,
                )
                cv2.imwrite(
                    os.path.join(output_dir, "overlays", f"{i:06d}.png"),
                    cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR),
                )

            if save_probmaps:
                heatmap = _prob_to_heatmap(score_map)
                cv2.imwrite(os.path.join(output_dir, "probmaps", f"{i:06d}.png"), heatmap)

            results.append({"frame": i, **summary})
            if (i + 1) % 50 == 0 or i == n_frames - 2:
                print(f"  Frame {i + 1}/{n_frames - 1}, dynamic: {results[-1]['dynamic_pixel_ratio']:.2%}")

    summary_path = os.path.join(output_dir, "results.json")
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nSaved {len(results)} masks to {output_dir}/masks/")
    if save_overlays:
        print(f"Overlays: {output_dir}/overlays/")
    if save_probmaps:
        print(f"Score maps: {output_dir}/probmaps/")
    print(f"Summary: {summary_path}")


def run_inference_rosbag(
    model: DynaMaskVIO,
    bag_path: str,
    output_dir: str,
    device: torch.device,
    threshold: float = 0.5,
    max_imu: int = 15,
    max_frames: int = 0,
    save_overlays: bool = True,
    save_probmaps: bool = True,
    threshold_mode: str = "fixed",
    topk_pct: float = 0.10,
    imu_trace_cal: float = 1.0,
):
    if _load_rosbag is None:
        raise RuntimeError("ROS bag inference is unavailable: rosbags loader is not installed.")
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
            img_prev = img_prev_raw.astype(np.float32)
            img_curr = img_curr_raw.astype(np.float32)

            mask_bin, score_map, dynamic_mask, summary = _forward_pair(
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
                threshold_mode=threshold_mode,
                topk_pct=topk_pct,
                imu_trace_cal=imu_trace_cal,
            )

            cv2.imwrite(os.path.join(output_dir, "masks", f"{i:06d}.png"), mask_bin)
            if save_overlays:
                img_curr_uint8 = np.clip(img_curr, 0, 255).astype(np.uint8)
                overlay = _make_overlay(
                    img_curr_uint8,
                    score_map,
                    threshold=threshold,
                    dynamic_mask=dynamic_mask,
                )
                cv2.imwrite(
                    os.path.join(output_dir, "overlays", f"{i:06d}.png"),
                    cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR),
                )
            if save_probmaps:
                heatmap = _prob_to_heatmap(score_map)
                cv2.imwrite(os.path.join(output_dir, "probmaps", f"{i:06d}.png"), heatmap)

            results.append(
                {
                    "frame": i,
                    "timestamp_prev": float(t_prev),
                    "timestamp_curr": float(t_curr),
                    **summary,
                }
            )
            if (i + 1) % 50 == 0 or i == n_frames - 2:
                print(f"  Frame {i + 1}/{n_frames - 1}, dynamic: {results[-1]['dynamic_pixel_ratio']:.2%}")

    summary_path = os.path.join(output_dir, "results.json")
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved {len(results)} masks to {output_dir}/masks/")
    print(f"Summary: {summary_path}")


def main():
    parser = argparse.ArgumentParser(description="DynaMask V2.5 Inference")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--config", type=str, default="dynamask_vio/configs/default.yaml")
    parser.add_argument("--images", type=str, help="Directory of input images")
    parser.add_argument("--imu", type=str, help="IMU CSV file")
    parser.add_argument("--bag", type=str, help="ROS bag path")
    parser.add_argument("--output", type=str, default="./output")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument(
        "--threshold-mode",
        type=str,
        default="fixed",
        choices=["fixed", "topk", "otsu", "imu-adaptive"],
    )
    parser.add_argument("--topk-pct", type=float, default=0.10)
    parser.add_argument("--imu-trace-cal", type=float, default=1.0)
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--no-overlays", action="store_true")
    parser.add_argument("--no-probmaps", action="store_true")
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
            threshold_mode=args.threshold_mode,
            topk_pct=args.topk_pct,
            imu_trace_cal=args.imu_trace_cal,
        )
        return

    if not args.images or not args.imu:
        raise ValueError("Provide either --bag, or both --images and --imu")

    print(f"Running inference on {args.images}")
    run_inference(
        model=model,
        image_dir=args.images,
        imu_file=args.imu,
        output_dir=args.output,
        device=device,
        threshold=args.threshold,
        max_frames=args.max_frames,
        save_overlays=save_overlays,
        save_probmaps=save_probmaps,
        threshold_mode=args.threshold_mode,
        topk_pct=args.topk_pct,
        imu_trace_cal=args.imu_trace_cal,
    )


if __name__ == "__main__":
    main()
