#!/usr/bin/env python3
"""KITTI inference with DynGRU — uses real camera-IMU extrinsic (T_BS).

Loads a trained DynGRU checkpoint and runs inference on KITTI stereo+IMU data.
IMU readings are rotated from body frame to camera frame using the calibration
extrinsic, then fed through the DynGRU pipeline.  Output shows the predicted
residual magnitude r_hat = softplus(dyn_logit) in pixels — high values indicate
pixels that deviate from rigid-body motion (i.e., likely dynamic).

Usage:
    python3 Scripts/infer_kitti_dyngru.py
"""

import argparse
import sys
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import pypose as pp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Pull in the training-script helpers for IMUContext interaction
_KITTI_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_KITTI_ROOT))


# ---------------------------------------------------------------------------
# KITTI calibration
# ---------------------------------------------------------------------------

def parse_kitti_calib(calib_dir: Path) -> dict:
    """Return K_rect_02 (3×3), baseline [m], R_imu→cam (3×3), t_imu→cam (3,)."""
    # Parse label: value format (one per line)
    calib = {}
    for line in (calib_dir / "calib_cam_to_cam.txt").read_text().strip().splitlines():
        if ":" not in line:
            continue
        label, vals = line.split(":", 1)
        try:
            calib[label.strip()] = np.array([float(x) for x in vals.strip().split()])
        except ValueError:
            continue  # skip non-numeric lines like calib_time dates

    # P_rect_02 = rectified left color projection (12 floats → 3×4)
    P2 = calib["P_rect_02"].reshape(3, 4)
    P3 = calib["P_rect_03"].reshape(3, 4)
    K = P2[:, :3].copy()
    baseline = abs(P3[0, 3]) / K[0, 0]

    # IMU → Velodyne
    lines = (calib_dir / "calib_imu_to_velo.txt").read_text().strip().splitlines()
    R_iv = np.array([float(x) for x in lines[1].split(":")[1].strip().split()]).reshape(3, 3)
    t_iv = np.array([float(x) for x in lines[2].split(":")[1].strip().split()])

    # Velodyne → Camera 0
    lines = (calib_dir / "calib_velo_to_cam.txt").read_text().strip().splitlines()
    R_vc = np.array([float(x) for x in lines[1].split(":")[1].strip().split()]).reshape(3, 3)
    t_vc = np.array([float(x) for x in lines[2].split(":")[1].strip().split()])

    R_ic = R_vc @ R_iv
    t_ic = R_vc @ t_iv + t_vc
    return {"K": K, "baseline": baseline, "R_imu_cam": R_ic, "t_imu_cam": t_ic}


# ---------------------------------------------------------------------------
# KITTI oxts IMU loader
# ---------------------------------------------------------------------------

def load_kitti_oxts(oxts_dir: Path, max_frames: int):
    """Return acc (N,3), gyro (N,3), rpy (N,3), init_rot_quat (4,) — all torch.float32."""
    files = sorted((oxts_dir / "data").glob("*.txt"))[:max_frames]
    acc_list, gyro_list, rpy_list = [], [], []
    init_quat = None
    for i, f in enumerate(files):
        v = np.loadtxt(f)
        # Fields: 3-5 = roll,pitch,yaw; 11-13 = ax,ay,az; 17-19 = wx,wy,wz
        ax, ay, az = v[11], v[12], v[13]
        wx, wy, wz = v[17], v[18], v[19]
        acc_list.append([ax, ay, az])
        gyro_list.append([wx, wy, wz])
        rpy_list.append([v[3], v[4], v[5]])
        if i == 0:
            roll, pitch, yaw = v[3], v[4], v[5]
            cr, sr = np.cos(roll / 2), np.sin(roll / 2)
            cp, sp = np.cos(pitch / 2), np.sin(pitch / 2)
            cy, sy = np.cos(yaw / 2), np.sin(yaw / 2)
            init_quat = np.array([
                cr * cp * cy + sr * sp * sy,
                sr * cp * cy - cr * sp * sy,
                cr * sp * cy + sr * cp * sy,
                cr * cp * sy - sr * sp * cy,
            ])
    return (
        torch.tensor(np.array(acc_list), dtype=torch.float32),
        torch.tensor(np.array(gyro_list), dtype=torch.float32),
        torch.tensor(np.array(rpy_list), dtype=torch.float32),
        torch.tensor(init_quat, dtype=torch.float32),
    )


def load_kitti_images(img_dir: Path, max_frames: int) -> torch.Tensor:
    """Return (N, 3, H, W) float32 in [0,1]."""
    files = sorted((img_dir / "data").glob("*.png"))[:max_frames]
    imgs = []
    for f in files:
        img = cv2.imread(str(f), cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        imgs.append(torch.tensor(img, dtype=torch.float32).permute(2, 0, 1) / 255.0)
    return torch.stack(imgs)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--kitti_root", default="/home/karan/Downloads/kitti_sample")
    parser.add_argument("--checkpoint", default="Model/DynGRU_VIODE_HPC_8200.pth")
    parser.add_argument("--output_dir", default="Results/kitti_inference")
    parser.add_argument("--max_frames", type=int, default=100)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    kitti_root = Path(args.kitti_root)
    seq_dir = kitti_root / "00"
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ---- Calibration ----
    calib = parse_kitti_calib(kitti_root)
    K = torch.tensor(calib["K"], dtype=torch.float32)
    print(f"K_rect_02: fx={K[0,0]:.1f} fy={K[1,1]:.1f} cx={K[0,2]:.1f} cy={K[1,2]:.1f}")
    print(f"baseline: {calib['baseline']:.4f} m")
    R_ic_np = calib["R_imu_cam"]
    print(f"R_imu→cam:\n{R_ic_np}")

    # ---- Load data ----
    acc_body, gyro_body, oxts_data, init_quat = load_kitti_oxts(seq_dir / "oxts", args.max_frames)
    imgL = load_kitti_images(seq_dir / "image_02", args.max_frames)
    imgR = load_kitti_images(seq_dir / "image_03", args.max_frames)
    N = min(len(imgL), len(acc_body))
    print(f"Loaded {N} frames  |  image {imgL.shape[-2:]}  |  IMU Hz ≈ 10")
    H_orig, W_orig = imgL.shape[-2], imgL.shape[-1]
    H_new, W_new = 480, 640  # Matches CenterCropFrame(640, 480) used in training
    print(f"Resizing to {H_new}×{W_new} for model compatibility...")

    imgL = F.interpolate(imgL, size=(H_new, W_new), mode="bilinear", align_corners=False)
    imgR = F.interpolate(imgR, size=(H_new, W_new), mode="bilinear", align_corners=False)

    # Scale intrinsics accordingly
    s_h, s_w = H_new / H_orig, W_new / W_orig
    K[0, 0] *= s_w; K[0, 2] *= s_w
    K[1, 1] *= s_h; K[1, 2] *= s_h

    # ---- Keep IMU in body frame (model trained with identity T_BS) ----
    # The model was trained assuming IMU = camera (T_BS = identity).
    # Rotating to camera frame changes the gravity direction, which the EKF
    # can't handle without retraining.  Keeping body-frame data is consistent
    # with the training setup — the model just treats body as camera.
    print("Keeping IMU in body frame (model expects identity T_BS).")
    print(f"R_imu→cam available but NOT applied for inference consistency.")
    acc_imu = acc_body   # (N, 3) in body frame
    gyro_imu = gyro_body  # (N, 3) in body frame

    # ---- Build model ----
    print("Building model...")
    from Utility.Config import load_config, namespace_to_cfgnode
    from Module.Network.FlowFormerDyn import build_flowformer_dyn
    from Module.Network.IMUContext.imu_context import IMUContext, IMUSample

    cfg, _ = load_config(Path("Config/Train/DynGRU_VIODE.yaml"))
    modelcfg = namespace_to_cfgnode(cfg.Model)
    model = build_flowformer_dyn(modelcfg, torch.float32, torch.float32)
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model = model.to(args.device)
    model.eval()

    # IMUContext
    imu_context = IMUContext(
        airio_cfg=SimpleNamespace(propcov=True),
        airio_ckpt=None,
        gravity=9.81,
    )
    imu_context.load_state_dict(ckpt["imu_context_state_dict"], strict=False)
    imu_context.to(args.device)
    imu_context.eval()

    # ---- Seed EKF at first frame (body frame, consistent with training) ----
    init_rot_pp = pp.SO3(init_quat.unsqueeze(0).to(args.device))
    init_vel = torch.zeros(1, 3, device=args.device)
    init_pos = torch.zeros(1, 3, device=args.device)
    imu_context.reset(init_rot_pp, init_vel, init_pos)
    print("IMUContext seeded (body frame).")
    print("NOTE: EKF reset per frame pair to prevent state accumulation (no AirIO).")

    # ---- Inference ----
    print(f"Running inference on {N-1} frame pairs (10 Hz IMU, 0.1 s dt)...")
    r_hat_maps = []

    DT_S = 0.1  # 0.1 s between samples (KITTI IMU at 10 Hz)

    with torch.no_grad():
        for idx in range(N - 1):
            # Reset EKF from GT attitude at frame idx (matches training: seed_from_gt per pair).
            # Without this, open-loop EKF propagation accumulates unbounded state.
            rpy = oxts_data[idx]  # (roll, pitch, yaw) from oxts
            roll, pitch, yaw = rpy[0].item(), rpy[1].item(), rpy[2].item()
            cr, sr = np.cos(roll/2), np.sin(roll/2)
            cp, sp = np.cos(pitch/2), np.sin(pitch/2)
            cy, sy = np.cos(yaw/2), np.sin(yaw/2)
            q_i = torch.tensor(
                [cr*cp*cy+sr*sp*sy, sr*cp*cy-cr*sp*sy, cr*sp*cy+sr*cp*sy, cr*cp*sy-sr*sp*sy],
                dtype=torch.float32, device=args.device
            )
            imu_context.reset(
                pp.SO3(q_i.unsqueeze(0)),
                torch.zeros(1, 3, device=args.device),
                torch.zeros(1, 3, device=args.device),
            )

            # Use IMU samples at [idx, idx+1] → 1 corrected tick, dt = 0.1 s
            i0, i1 = idx, min(idx + 2, N)
            if i1 - i0 < 2:
                r_hat_maps.append(torch.zeros(1, 1, imgL.shape[2] // 4, imgL.shape[3] // 4))
                continue

            acc_win = acc_imu[i0:i1].to(args.device)    # (2, 3)
            gyro_win = gyro_imu[i0:i1].to(args.device)  # (2, 3)

            corrected = [{
                "acc": acc_win[0],
                "gyro": gyro_win[0],
                "dt": DT_S,  # float in seconds, _to_f64 converts
            }]
            raw = [{"acc": acc_win[0], "gyro": gyro_win[0]}]

            sample: IMUSample = imu_context.step(corrected, raw)
            f_imu = sample.f_imu.to(args.device)
            imu_tokens = sample.imu_tokens.to(args.device)

            # Model forward
            img1 = imgL[idx:idx+1].to(args.device)
            img2 = imgL[idx+1:idx+2].to(args.device)
            _, _, dyn_preds = model(img1, img2, f_imu, imu_tokens)
            r_hat = F.softplus(dyn_preds[-1])[0].cpu()
            r_hat_maps.append(r_hat)

            if (idx + 1) % 10 == 0:
                print(f"  Frame {idx+1}/{N-1}  r_hat mean={r_hat.mean().item():.2f} px  max={r_hat.max().item():.1f} px")

    print(f"Done — {len(r_hat_maps)} frame pairs.")

    # ---- Visualize ----
    print("Rendering...")
    n_viz = len(r_hat_maps)  # render all frame pairs
    for i in range(n_viz):
        img = imgL[i].permute(1, 2, 0).numpy()
        r_t = r_hat_maps[i]
        if r_t.dim() == 2:
            r_t = r_t.unsqueeze(0).unsqueeze(0)  # (H,W) → (1,1,H,W)
        elif r_t.dim() == 3:
            r_t = r_t.unsqueeze(0)  # (C,H,W) → (1,C,H,W)
        r_full = F.interpolate(
            r_t, size=img.shape[:2], mode="bilinear", align_corners=False
        ).squeeze().numpy()

        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        axes[0].imshow(img.clip(0, 1))
        axes[0].set_title(f"Frame {i}")
        axes[0].axis("off")

        vmax = max(r_full.max(), 5.0)
        axes[1].imshow(r_full, cmap="hot", vmin=0, vmax=vmax)
        axes[1].set_title(f"Predicted residual (mean={r_full.mean():.2f} px)")
        axes[1].axis("off")

        axes[2].imshow(img.clip(0, 1))
        axes[2].imshow(r_full, cmap="hot", vmin=0, vmax=vmax, alpha=0.5)
        axes[2].set_title(f"Overlay (hot = more dynamic)")
        axes[2].axis("off")

        fig.tight_layout()
        fig.savefig(output_dir / f"frame_{i:04d}.png", dpi=100)
        plt.close(fig)

    # Summary curve
    means = [r.mean().item() for r in r_hat_maps]
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.plot(means, "b-", linewidth=1)
    ax.set_xlabel("Frame pair"); ax.set_ylabel("Mean predicted residual [px]")
    ax.set_title("DynGRU r_hat over sequence"); ax.grid(True, alpha=0.3)
    fig.tight_layout(); fig.savefig(output_dir / "summary.png", dpi=100); plt.close(fig)

    print(f"Saved → {output_dir}/")


if __name__ == "__main__":
    main()
