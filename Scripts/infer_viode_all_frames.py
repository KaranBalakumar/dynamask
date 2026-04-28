"""Run dynGRU inference on all VIODE high sequences, saving per-frame heatmaps."""
import torch, os, sys
from pathlib import Path
from types import SimpleNamespace
from tqdm import tqdm
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pypose as pp

CKPT = "Model/DynGRU_VIODE04-27-19-43-07/12000.pth"
SEQUENCES = [
    ("city_day", "3_high"),
    ("city_night", "3_high"),
    ("parking_lot", "3_high"),
]
DATA_BASE = Path.home() / "dynamask/dataset/viode_tartan"
OUT_BASE = Path("Results/viode_inference_12000")
OUT_BASE.mkdir(parents=True, exist_ok=True)

# --- Load model ---
from Module.Network.FlowFormerDyn import build_flowformer_dyn
from Module.Network.FlowFormer.configs.submission import get_cfg
from Utility.Config import load_config, namespace_to_cfgnode

cfg = get_cfg()
cfg.latentcostformer.decoder_depth = 12
model = build_flowformer_dyn(cfg, torch.float32, torch.float32)
ckpt = torch.load(CKPT, map_location="cpu", weights_only=True)
state = ckpt.get('model_state_dict', ckpt)
model.load_ddp_state_dict(state)
model = model.cuda().eval()
print(f"Loaded: {CKPT}")

# --- IMUContext ---
from Module.Network.IMUContext.imu_context import IMUContext
from Train.MatchingNet.train_flowformer import _imu_data_to_ticks
imu_ctx = IMUContext(airio_cfg=SimpleNamespace(propcov=True), airio_ckpt=None, gravity=9.81)
imu_ctx.cuda()

# --- Loop over sequences ---
for env, diff in SEQUENCES:
    root = DATA_BASE / env / diff
    print(f"\n{'='*60}")
    print(f"Inference: {env}/{diff}")
    print(f"{'='*60}")

    from DataLoader.Dataset.TartanAir2 import TartanAirV2_Sequence
    seq_cfg = SimpleNamespace(
        root=str(root), compressed=False, use_real_imu=True, gravity=9.81,
        gtFlow=False, gtDepth=False, gtPose=True, imu_freq=200,
        fixed_imu_samples=10,
        imu_sim=SimpleNamespace(acc_bias=(0,0,0), acc_init_bias_noise=(0,0,0),
            acc_bias_instability=(0,0,0), acc_random_walk=(0,0,0),
            gyro_bias=(0,0,0), gyro_init_bias_noise=(0,0,0),
            gyro_bias_instability=(0,0,0), gyro_random_walk=(0,0,0)),
    )
    seq = TartanAirV2_Sequence(seq_cfg)
    print(f"  {len(seq)} frames, {len(seq)-1} pairs")

    out_dir = OUT_BASE / f"{env}_{diff}"
    out_dir.mkdir(parents=True, exist_ok=True)

    c_values = []
    for i in tqdm(range(len(seq) - 1), desc=f"  {env}/{diff}"):
        f0 = seq[i]; f1 = seq[i+1]
        img1 = f0.stereo.imageL.cuda()
        img2 = f1.stereo.imageL.cuda()

        # IMU
        imu_b = f1.imu
        att_b = f1.gt_attitude
        pose_b = f1.gt_pose[0] if f1.gt_pose is not None else None
        corrected, raw = _imu_data_to_ticks(imu_b)
        imu_ctx.seed_from_gt(att_b, pose_b)
        sample = imu_ctx.step(corrected, raw)

        # Forward
        with torch.no_grad():
            _, _, dyn_pre = model.inference(
                img1, img2, sample.f_imu.cuda(), sample.imu_tokens.cuda()
            )
        c = dyn_pre[0].sigmoid().cpu().squeeze().numpy()  # (H, W)
        c_values.append(float(c.mean()))

        # Save image
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        img_np = img1.squeeze().permute(1,2,0).cpu().clamp(0,1).numpy()
        ax.imshow(img_np, alpha=0.5)
        heat = ax.imshow(c, cmap="RdYlBu_r", vmin=0, vmax=1, alpha=0.6)
        ax.set_title(f"Frame {i} | c mean={c.mean():.3f} | {env}/{diff}")
        ax.axis("off")
        plt.colorbar(heat, ax=ax, label="static confidence c", fraction=0.046)
        fig.tight_layout()
        fig.savefig(out_dir / f"frame_{i:05d}.png", dpi=100)
        plt.close(fig)

    print(f"  c mean={sum(c_values)/len(c_values):.4f}, "
          f"c<0.3: {sum(1 for x in c_values if x<0.3)/len(c_values)*100:.1f}%, "
          f"c>0.7: {sum(1 for x in c_values if x>0.7)/len(c_values)*100:.1f}%")

print(f"\nDone. Outputs: {OUT_BASE}")
