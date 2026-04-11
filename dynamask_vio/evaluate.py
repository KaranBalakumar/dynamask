"""
Evaluation: optional mask metrics + VIO trajectory metrics.

Mask metrics (on VIODE test sequences):
  - IoU, Precision, Recall, F1 (per dynamic level)
  - Requires true motion-mask labels in the dataloader batch (`gt_mask`)

IMU metrics (on EuRoC test sequences):
  - RTE, ROE of 1-second preintegration

VIO trajectory metrics (requires OpenVINS):
  - ATE RMSE, RPE  (via evo package)

Usage:
    python -m dynamask_vio.evaluate --checkpoint <path> --config <path>
"""

import argparse
import os

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

from .models import DynaMaskVIO
from .data import VIODEDataset, EuRoCDataset


def _load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def _merge_configs(base: dict, override: dict) -> dict:
    merged = base.copy()
    for k, v in override.items():
        if isinstance(v, dict) and k in merged and isinstance(merged[k], dict):
            merged[k] = _merge_configs(merged[k], v)
        else:
            merged[k] = v
    return merged


# ── Mask Metrics ──

def compute_mask_metrics(pred_mask: torch.Tensor,
                          gt_mask: torch.Tensor,
                          threshold: float = 0.5) -> dict:
    """Compute IoU, Precision, Recall, F1 for a batch.

    Args:
        pred_mask: [B, 1, H, W] probabilities
        gt_mask:   [B, 1, H, W] binary

    Returns:
        dict with iou, precision, recall, f1 (scalars)
    """
    pred_bin = (pred_mask > threshold).float()
    gt = gt_mask.float()

    tp = (pred_bin * gt).sum()
    fp = (pred_bin * (1 - gt)).sum()
    fn = ((1 - pred_bin) * gt).sum()

    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    iou = tp / (tp + fp + fn + 1e-8)

    return {
        "iou": iou.item(),
        "precision": precision.item(),
        "recall": recall.item(),
        "f1": f1.item(),
    }


def evaluate_mask(model: DynaMaskVIO, dataloader: DataLoader,
                   device: torch.device, threshold: float = 0.5) -> dict:
    """Evaluate mask quality over a full dataloader."""
    model.eval()
    all_metrics = {"iou": [], "precision": [], "recall": [], "f1": []}

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Mask eval"):
            if "gt_mask" not in batch:
                raise RuntimeError(
                    "Mask evaluation requires batch['gt_mask'], but the current datasets "
                    "do not provide true motion-mask labels."
                )
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                     for k, v in batch.items()}

            outputs = model(batch["img_prev"], batch["img_curr"],
                            batch["imu_window"], batch["imu_mask"])

            metrics = compute_mask_metrics(outputs["dynamic_mask"],
                                            batch["gt_mask"], threshold)
            for k, v in metrics.items():
                all_metrics[k].append(v)

    return {k: np.mean(v) for k, v in all_metrics.items()}


# ── IMU Metrics ──

def evaluate_imu(model: DynaMaskVIO, dataloader: DataLoader,
                  device: torch.device) -> dict:
    """Evaluate IMU preintegration quality (RTE, ROE)."""
    from .models.preintegration import so3_log_map

    model.eval()
    rte_list = []
    roe_list = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="IMU eval"):
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                     for k, v in batch.items()}

            outputs = model(batch["img_prev"], batch["img_curr"],
                            batch["imu_window"], batch["imu_mask"])

            # Translation error
            pred_p = outputs["delta_p"]
            gt_p = batch["gt_p"]
            rte = (pred_p - gt_p).norm(dim=-1)  # [B]
            rte_list.extend(rte.cpu().tolist())

            # Rotation error
            pred_R = outputs["delta_R"]
            gt_R = batch["gt_R"]
            R_err = pred_R.transpose(-1, -2) @ gt_R
            angle_err = so3_log_map(R_err).norm(dim=-1)  # [B]
            roe_list.extend(angle_err.cpu().tolist())

    return {
        "rte_mean": np.mean(rte_list),
        "rte_median": np.median(rte_list),
        "roe_mean_deg": np.degrees(np.mean(roe_list)),
        "roe_median_deg": np.degrees(np.median(roe_list)),
    }


# ── VIO Trajectory Metrics (requires evo) ──

def evaluate_trajectory(est_traj_file: str, gt_traj_file: str) -> dict:
    """Compute ATE RMSE and RPE using the evo package.

    Args:
        est_traj_file: path to estimated trajectory (TUM format)
        gt_traj_file: path to ground truth trajectory (TUM format)

    Returns:
        dict with ate_rmse and rpe_rmse
    """
    from evo.core import metrics, sync
    from evo.tools import file_interface

    gt = file_interface.read_tum_trajectory_file(gt_traj_file)
    est = file_interface.read_tum_trajectory_file(est_traj_file)

    # Associate trajectories
    gt_sync, est_sync = sync.associate_trajectories(gt, est)

    # ATE
    ate_metric = metrics.APE(metrics.PoseRelation.translation_part)
    ate_metric.process_data((gt_sync, est_sync))
    ate_stats = ate_metric.get_all_statistics()

    # RPE (1-second intervals)
    rpe_metric = metrics.RPE(metrics.PoseRelation.translation_part,
                              delta=1.0, delta_unit=metrics.Unit.seconds)
    rpe_metric.process_data((gt_sync, est_sync))
    rpe_stats = rpe_metric.get_all_statistics()

    return {
        "ate_rmse": ate_stats["rmse"],
        "ate_mean": ate_stats["mean"],
        "rpe_rmse": rpe_stats["rmse"],
        "rpe_mean": rpe_stats["mean"],
    }


def main():
    parser = argparse.ArgumentParser(description="DynaMask-VIO Evaluation")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--config", type=str,
                        default="dynamask_vio/configs/default.yaml")
    parser.add_argument("--eval-mask", action="store_true",
                        help="Evaluate mask metrics on VIODE test set")
    parser.add_argument("--eval-imu", action="store_true",
                        help="Evaluate IMU metrics on EuRoC test set")
    parser.add_argument("--eval-traj", nargs=2, metavar=("EST", "GT"),
                        help="Evaluate trajectory from TUM files")
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()

    cfg = _load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # For checkpoint-based evaluation, avoid loading ImageNet backbone weights first.
    cfg.setdefault("model", {})["backbone_pretrained"] = False

    # Load model
    model = DynaMaskVIO(cfg)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    state = ckpt.get("state_dict", ckpt)
    # Strip "model." prefix from Lightning state dict
    state = {k.replace("model.", "", 1): v for k, v in state.items()
             if k.startswith("model.")} or state
    missing, unexpected = model.load_state_dict(state, strict=False)
    loaded = len(model.state_dict()) - len(missing)
    print(
        f"[Checkpoint] Loaded from {args.checkpoint} "
        f"(loaded={loaded}, missing={len(missing)}, unexpected={len(unexpected)})"
    )
    model = model.to(device)

    paths = cfg.get("paths", {})

    if args.eval_mask:
        viode_cfg_path = os.path.join(os.path.dirname(__file__),
                                       "configs", "viode.yaml")
        viode_cfg = _load_config(viode_cfg_path) if os.path.exists(viode_cfg_path) else {}
        merged = _merge_configs(cfg, viode_cfg)
        test_seqs = merged.get("splits", {}).get("test", [])

        viode_hdf5_root = paths.get("viode_hdf5_root",
                                     paths.get("viode_root", "./dataset/viode_hdf5"))
        dataset = VIODEDataset(
            viode_hdf5_root,
            test_seqs, merged, split="test",
        )
        loader = DataLoader(dataset, batch_size=cfg["training"]["batch_size"],
                            num_workers=4)
        metrics = evaluate_mask(model, loader, device, args.threshold)
        print("\n=== Mask Metrics (VIODE Test) ===")
        for k, v in metrics.items():
            print(f"  {k}: {v:.4f}")

    if args.eval_imu:
        euroc_cfg_path = os.path.join(os.path.dirname(__file__),
                                       "configs", "euroc.yaml")
        euroc_cfg = _load_config(euroc_cfg_path) if os.path.exists(euroc_cfg_path) else {}
        merged = _merge_configs(cfg, euroc_cfg)
        test_seqs = merged.get("splits", {}).get("test", [])

        dataset = EuRoCDataset(
            paths.get("euroc_root", "./dataset/euroc"),
            test_seqs, merged, split="test",
        )
        loader = DataLoader(dataset, batch_size=cfg["training"]["batch_size"],
                            num_workers=4)
        metrics = evaluate_imu(model, loader, device)
        print("\n=== IMU Metrics (EuRoC Test) ===")
        for k, v in metrics.items():
            print(f"  {k}: {v:.4f}")

    if args.eval_traj:
        metrics = evaluate_trajectory(args.eval_traj[0], args.eval_traj[1])
        print("\n=== Trajectory Metrics ===")
        for k, v in metrics.items():
            print(f"  {k}: {v:.4f}")


if __name__ == "__main__":
    main()
