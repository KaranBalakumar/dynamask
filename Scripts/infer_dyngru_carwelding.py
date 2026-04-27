"""Run dynGRU inference on the full CarWelding sequence and save outputs."""
import torch, os, pypose as pp
from pathlib import Path
from types import SimpleNamespace
from tqdm import tqdm

# --- Config ---
CKPT = "Model/FlowFormerDynDemo04-27-13-20-27/4000.pth"
DATA_ROOT = str(Path.home() / "Downloads/tartanair2/CarWelding/Data_hard/P001")
OUT_DIR = Path("Results/carwelding_inference")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# --- Load model ---
from Module.Network.FlowFormerDyn import build_flowformer_dyn
from Module.Network.FlowFormer.configs.submission import get_cfg
from Utility.Config import load_config, namespace_to_cfgnode

cfg = get_cfg()
cfg.latentcostformer.decoder_depth = 12
model = build_flowformer_dyn(cfg, torch.float32, torch.float32)
ckpt = torch.load(CKPT, map_location="cpu", weights_only=True)
model.load_ddp_state_dict(ckpt)
model = model.cuda().eval()
print(f"Loaded checkpoint: {CKPT}")

# --- Load data ---
from DataLoader.Dataset.TartanAir2 import TartanAirV2_Sequence
from DataLoader import CenterCropFrame, CastDataType

seq_cfg = SimpleNamespace(
    root=DATA_ROOT, compressed=True, use_real_imu=True, gravity=9.81,
    gtFlow=False, gtDepth=False, gtPose=True, imu_freq=100,
    imu_sim=SimpleNamespace(acc_bias=(0,0,0), acc_init_bias_noise=(0,0,0),
        acc_bias_instability=(0,0,0), acc_random_walk=(0,0,0),
        gyro_bias=(0,0,0), gyro_init_bias_noise=(0,0,0),
        gyro_bias_instability=(0,0,0), gyro_random_walk=(0,0,0)),
)
seq = TartanAirV2_Sequence(seq_cfg)
crop = CenterCropFrame(SimpleNamespace(height=480, width=640))
cast = CastDataType(SimpleNamespace(dtype="fp32"))
print(f"CarWelding: {len(seq)} frames, {len(seq)-1} pairs")

# --- IMUContext ---
from Module.Network.IMUContext.imu_context import IMUContext
from Train.MatchingNet.train_flowformer import _imu_data_to_ticks, _imu_data_unbatch, _attitude_unbatch

imu_ctx = IMUContext(airio_cfg=SimpleNamespace(propcov=True), airio_ckpt=None, gravity=9.81)
imu_ctx.cuda()

# --- Run inference ---
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

results = []
for i in tqdm(range(len(seq) - 1), desc="Inference"):
    f0 = cast.forward(crop.forward(seq[i]))
    f1 = cast.forward(crop.forward(seq[i + 1]))

    img1 = f0.stereo.imageL.cuda()
    img2 = f1.stereo.imageL.cuda()

    # IMU processing
    imu_b = _imu_data_unbatch(f1.imu, 0)
    att_b = _attitude_unbatch(f1.gt_attitude, 0)
    pose_b = f1.gt_pose[0] if f1.gt_pose is not None else None
    corrected, raw = _imu_data_to_ticks(imu_b)
    imu_ctx.seed_from_gt(att_b, pose_b)
    sample = imu_ctx.step(corrected, raw)

    # Model forward
    with torch.no_grad():
        flow_pre, cov_pre, dyn_pre = model.inference(
            img1, img2, sample.f_imu.cuda(), sample.imu_tokens.cuda()
        )

    c_map = dyn_pre[0].sigmoid().cpu().squeeze().numpy()  # (H, W)
    results.append({
        "frame": i,
        "c_mean": float(c_map.mean()),
        "c_std": float(c_map.std()),
        "c_below_0_3": float((c_map < 0.3).mean()),
        "c_above_0_7": float((c_map > 0.7).mean()),
    })

    # Save every 10th frame as image
    if i % 10 == 0:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        img_np = img1.squeeze().permute(1, 2, 0).cpu().clamp(0, 1).numpy()
        axes[0].imshow(img_np)
        axes[0].set_title(f"Frame {i}")
        axes[0].axis("off")
        heat = axes[1].imshow(c_map, cmap="RdYlBu_r", vmin=0, vmax=1)
        axes[1].set_title(f"Static confidence c (mean={c_map.mean():.3f})")
        axes[1].axis("off")
        plt.colorbar(heat, ax=axes[1])
        fig.tight_layout()
        fig.savefig(OUT_DIR / f"frame_{i:04d}.png", dpi=100)
        plt.close(fig)

# --- Summary ---
c_means = [r["c_mean"] for r in results]
c_below = [r["c_below_0_3"] for r in results]
c_above = [r["c_above_0_7"] for r in results]

print(f"\n=== CarWelding Inference Summary ({len(results)} pairs) ===")
print(f"c mean: {np.mean(c_means):.4f} ± {np.std(c_means):.4f}")
print(f"% pixels c<0.3 (dynamic): {np.mean(c_below)*100:.1f}%")
print(f"% pixels c>0.7 (static): {np.mean(c_above)*100:.1f}%")

# Find frames with most dynamic content
dynamic_frames = sorted(enumerate(c_below), key=lambda x: -x[1])[:10]
print(f"\nTop 10 most dynamic frames:")
for idx, pct in dynamic_frames:
    print(f"  Frame {idx}: {pct*100:.1f}% pixels with c<0.3 (dynamic)")

# Plot c over time
fig, ax = plt.subplots(figsize=(12, 4))
ax.plot(c_means, alpha=0.7)
ax.set_xlabel("Frame pair")
ax.set_ylabel("Mean static confidence c")
ax.set_title("DynGRU static confidence over CarWelding sequence")
ax.axhline(y=0.5, color="red", linestyle="--", alpha=0.5, label="c=0.5")
ax.legend()
fig.tight_layout()
fig.savefig(OUT_DIR / "c_over_time.png", dpi=100)
plt.close(fig)

print(f"\nOutputs saved to {OUT_DIR}")
