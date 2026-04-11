# DynaMask V2

Self-supervised dynamic masking for Visual-Inertial Odometry. A PyTorch system that learns to mask dynamic objects **without any ground truth mask labels**, using only images, IMU data, and camera intrinsics.

**Key idea**: A differentiable Bundle Adjustment (BA) module provides the self-supervised training signal. The mask tells BA which pixels are static; BA quality (via reprojection error and pose consistency with IMU) tells the mask whether it got it right.

---

## Table of Contents

- [Architecture](#architecture)
- [How It Works](#how-it-works)
- [Installation](#installation)
- [Datasets](#datasets)
- [Pretrained Weights](#pretrained-weights)
- [Training](#training)
- [Inference](#inference)
- [Evaluation](#evaluation)
- [Logging and Debugging](#logging-and-debugging)
- [ONNX Export](#onnx-export)
- [Project Structure](#project-structure)
- [Configuration Reference](#configuration-reference)
- [Environment Variables](#environment-variables)
- [Memory and Compute Requirements](#memory-and-compute-requirements)
- [Troubleshooting](#troubleshooting)
- [References](#references)

---

## Architecture

```
                                ┌──────────────┐
                                │  Raw IMU     │
                                │  [B, N, 7]   │
                                └──────┬───────┘
                                       │
                              ┌────────▼────────┐
                              │ Noise Corrector │ ── delta_bg, delta_ba, sigma2_g, sigma2_a
                              │ (dilated 1D CNN)│
                              └────────┬────────┘
                                       │ corrected IMU
                              ┌────────▼────────┐
                              │ Preintegrator   │ ── delta_R, delta_v, delta_p, Sigma [9x9]
                              │ (PyPose SO(3))  │
                              └────────┬────────┘
                                       │ f_imu [B, 128]
                        ┌──────────────┼──────────────┐
                        │              │              │
 ┌──────────┐    ┌──────▼───┐  ┌──────▼───┐  ┌──────▼───┐
 │ img_prev ├──► │ FiLM S1  │  │ FiLM S2  │  │ FiLM S3  │
 │ img_curr ├──► │ (RAFT    │  │          │  │          │
 └──────────┘    │ BasicEnc)│  └──────────┘  └──────────┘
                 └──────┬───┘        fmap_prev, fmap_curr [B, 128, H/8, W/8]
                        │
                 ┌──────▼──────────────────────────────┐
                 │ Context Encoder (RAFT BasicEncoder) │──► GRU init (tanh) + motion ctx (relu)
                 └──────┬──────────────────────────────┘
                        │
                 ┌──────▼────────────────────────────────┐
                 │ CorrBlock (all-pairs, 4-level pyramid)│
                 │ MotionEncoder + SepConvGRU (3 iters)  │
                 │ FlowHead (internal) + MaskHead        │
                 └──────┬───────────┬────────────────────┘
                        │           │
                   flow [B,2,H/8,W/8]  dynamic_mask [B,1,H,W]
                   (internal, for BA)    (exported output)
```

**Components**:

| Module | Description | Params |
|--------|-------------|--------|
| **Feature Encoder (fnet)** | RAFT BasicEncoder with InstanceNorm, FiLM-conditioned by IMU. Shared weights, called on both frames. Stages: 64 -> 96 -> 128 channels, 1/8 resolution. | ~1.1M |
| **Context Encoder (cnet)** | Same architecture as fnet but separate weights, no FiLM. Called on current frame only. Output split into GRU hidden init (tanh) + motion context (relu). | ~1.1M |
| **FiLM Layers** | Feature-wise Linear Modulation. 3 layers (one per encoder stage). Each: gamma_proj + beta_proj linear layers. Applies `gamma * features + beta` where gamma/beta are derived from f_imu. Initialized to identity (gamma=1, beta=0). | ~0.1M |
| **Flow Decoder** | RAFT Update Operator. CorrBlock builds all-pairs correlation volume with 4-level pyramid. MotionEncoder fuses correlation + flow. SepConvGRU refines flow iteratively (3 iterations). FlowHead predicts delta flow per iteration. MaskHead decodes mask from final GRU state with GradientClip (+-0.01). | ~1.4M |
| **IMU Encoder** | NoiseCorrector (dilated 1D CNN) predicts per-sample bias corrections + noise variances. FeatureMLP produces f_imu [128]. DifferentiablePreintegrator via PyPose SO3 computes delta_R/v/p with covariance Sigma [9x9]. | ~0.9M |
| **Differentiable BA** | Not a learned module. Gauss-Newton solver on SE(3) with SafeCholeskySolver (never throws), DPVO-style Jacobians, adaptive damping, hard outlier rejection. Provides L_pose and L_reproj gradients. | 0 |

**Total**: ~4.6M parameters.

---

## How It Works

### Self-supervised training signal

Traditional approaches require ground-truth dynamic masks (expensive to annotate). DynaMask V2 is **fully self-supervised**:

1. The model predicts a dynamic mask and optical flow from two frames + IMU
2. Pixels classified as **static** are selected as correspondences for Bundle Adjustment
3. The differentiable BA solves for camera pose using only these static points
4. Two signals flow backward through the mask:
   - **L_pose**: BA-estimated pose should agree with IMU-preintegrated pose (rotation geodesic + translation direction)
   - **L_reproj**: Reprojection error of static correspondences should be low
5. If the mask incorrectly labels a dynamic pixel as static, BA gets corrupted correspondences, the pose estimate degrades, and the loss increases -- teaching the mask to exclude that pixel next time

### Phased training

Training is split into phases to bootstrap each component safely:

- **Phase 1 (epochs 1-20)**: IMU encoder only, trained on EuRoC (static scenes with GT trajectories). All vision modules frozen. Learns bias correction and preintegration.
- **Phase 2a (epochs 21-30)**: Vision modules unfrozen. Photometric loss + flow smoothness train the flow decoder. BA runs but its gradients are **detached** -- no BA signal flows to the mask yet. This lets the flow settle before BA starts influencing mask learning.
- **Phase 2b (epochs 31-70)**: BA gradients enabled. L_pose ramps from 0 to 1 over 5 epochs to avoid initial instability. The full self-supervised loop is active.
- **Phase 3 (epochs 71-80)**: Fine-tuning on VIODE dynamic scenes. All losses at full weight. L_pose ramp = 1.

### FiLM conditioning

The IMU feature vector f_imu modulates visual features at each encoder stage via Feature-wise Linear Modulation:
```
output = gamma(f_imu) * visual_features + beta(f_imu)
```
This lets IMU context (e.g., "camera is rotating fast") influence what the visual encoder attends to. Initialized near identity so it doesn't disrupt pretrained RAFT features early in training.

### Differentiable BA details

The BA module follows DPVO's approach:
- **Jacobians**: Exact 2x6 reprojection Jacobians matching DPVO's `ba_cuda.cu`
- **Damping**: `H += (base_damping * diag(H) + ep) * I` matching DPVO's formula
- **SafeCholeskySolver**: Uses `torch.linalg.cholesky_ex` (returns info code instead of throwing). On failure, returns zero update. Custom backward: `dH = -xs @ dz^T`
- **Outlier rejection**: Hard threshold on reprojection error magnitude
- **Convergence**: Checked via mean reprojection error < threshold
- **Warm start**: Rotation initialized from IMU preintegrated rotation

---

## Installation

### Prerequisites

- NVIDIA GPU with CUDA 12.x (or CPU for testing)
- [micromamba](https://mamba.readthedocs.io/en/latest/installation/micromamba-installation.html) (recommended) or conda

### Setup

```bash
git clone <repo-url> dynamask && cd dynamask

# Create environment
micromamba env create -f environment.yml
micromamba activate dynamask

# Install PyTorch for your GPU:
# CUDA 12.1
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
# CUDA 12.4
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
# ROCm 6.2
pip install torch torchvision --index-url https://download.pytorch.org/whl/rocm6.2
# CPU only (dev/testing)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

### Verify installation

```bash
micromamba activate dynamask
python -c "
from dynamask_vio.models import DynaMaskVIO
import torch
cfg = {
    'model': {'imu_feature_dim': 128, 'encoder_output_dim': 128,
              'gru_hidden_dim': 128, 'gru_iterations': 3,
              'corr_levels': 4, 'corr_radius': 4},
    'imu': {'noise_corrector_hidden': 64, 'noise_corrector_dilations': [1,2,4],
            'initial_variance': 1e-4},
}
model = DynaMaskVIO(cfg)
out = model(torch.randn(1,3,480,640)*255, torch.randn(1,3,480,640)*255,
            torch.randn(1,10,7), torch.ones(1,10).bool())
print(f'OK - mask: {out[\"dynamic_mask\"].shape}, flow: {out[\"flow\"].shape}, params: {sum(p.numel() for p in model.parameters())/1e6:.2f}M')
"
```

Expected output:
```
OK - mask: torch.Size([1, 1, 480, 640]), flow: torch.Size([1, 2, 60, 80]), params: 4.64M
```

---

## Datasets

### VIODE (primary -- dynamic scenes, self-supervised training + evaluation)

Download ROS bags from [Zenodo](https://zenodo.org/record/4493401). Expected layout:

```
dataset/viode/
  city_day/
    0_none.bag    # no dynamic objects (static baseline)
    1_low.bag     # low density dynamic objects
    2_mid.bag     # medium density
    3_high.bag    # high density
  city_night/
    0_none.bag, 1_low.bag, 2_mid.bag, 3_high.bag
  parking_lot/
    0_none.bag, 1_low.bag, 2_mid.bag, 3_high.bag
```

**ROS bag topics extracted**:
| Topic | Type | Usage |
|-------|------|-------|
| `/cam0/image_raw` | sensor_msgs/Image | RGB/grayscale frames at 20Hz |
| `/imu0` | sensor_msgs/Imu | IMU at 200Hz (accel + gyro) |
| `/cam0/segmentation` | sensor_msgs/Image | GT semantic segmentation (for eval) |
| `/odometry` | nav_msgs/Odometry | GT odometry (for IMU loss supervision) |

**Camera intrinsics** (in `configs/viode.yaml`): fx=320, fy=320, cx=320, cy=240, image 640x480

**Segmentation**: Dynamic mask is derived from segmentation -- any pixel NOT in the configured static class IDs is treated as dynamic. Configure in `configs/viode.yaml`:
```yaml
segmentation:
  mode: "auto"           # "auto", "rgb", or "id"
  static_class_ids: [0, 1, 2, 3, 4, 5, 6]  # background, sky, ground, etc.
  palette_to_class_id: {}  # explicit "R,G,B" -> class_id mapping for rgb8 masks
```

**Train/val/test split** (from `configs/viode.yaml`):
- Train: `city_day/{0_none,1_low,2_mid,3_high}`, `city_night/{0_none,1_low,2_mid,3_high}`
- Val: `parking_lot/1_low`, `parking_lot/2_mid`
- Test: `parking_lot/3_high`, `city_night/3_high`

### EuRoC MAV (IMU pretraining -- Phase 1, static scenes)

Download ASL format from [ETH ASL](https://projects.asl.ethz.ch/datasets/doku.php?id=kmavvisualinertialdatasets). The loader auto-discovers sequences in nested directories:

```
dataset/euroc/
  machine_hall/
    MH_01_easy/MH_01_easy/mav0/
      cam0/data/          # grayscale images (752x480)
      cam0/data.csv       # image timestamps
      imu0/data.csv       # IMU at 200Hz (timestamp, wx,wy,wz, ax,ay,az)
      state_groundtruth_estimate0/data.csv  # GT poses, velocities, biases
  vicon_room1/
    V1_01_easy/V1_01_easy/mav0/...
  vicon_room2/
    V2_01_easy/V2_01_easy/mav0/...
```

The loader handles multiple directory layouts automatically (flat, nested, double-nested from zip extraction). You just specify sequence names like `MH_01_easy` in the config.

**Camera intrinsics** (in `configs/euroc.yaml`): fx=458.654, fy=457.296, cx=367.215, cy=248.375, image 752x480

**IMU noise parameters** (from EuRoC sensor.yaml, configured in `configs/euroc.yaml`):
```yaml
imu_params:
  gyro_noise_density: 1.6968e-04      # rad/s/sqrt(Hz)
  gyro_random_walk: 1.9393e-05        # rad/s^2/sqrt(Hz)
  accel_noise_density: 2.0000e-03     # m/s^2/sqrt(Hz)
  accel_random_walk: 3.0000e-03       # m/s^3/sqrt(Hz)
```

**Train/val/test split**:
- Train: MH_01_easy, MH_03_medium, MH_05_difficult, V1_02_medium, V2_01_easy, V2_03_difficult
- Val: MH_02_easy, V1_01_easy
- Test: MH_04_difficult, V1_03_difficult, V2_02_medium

### TartanAir V2 (supplementary diversity)

```python
import tartanair as ta
ta.init('./dataset/tartanair')
for env in ['Downtown', 'OldTown', 'Neighborhood', 'UrbanTree', 'AbandonedCable']:
    ta.download(env=env, difficulty=['easy', 'hard'],
                modality=['image', 'seg', 'imu', 'flow'],
                camera_name=['lcam_front'], unzip=True)
```

### Custom dataset paths

Override in `configs/default.yaml` or via environment:
```yaml
paths:
  viode_root: "./dataset/viode"
  euroc_root: "./dataset/euroc"
  tartanair_root: "./dataset/tartanair"
  checkpoint_dir: "./checkpoints"
  log_dir: "./logs"
```

### Data augmentation

Applied during training (configured in `configs/default.yaml` under `augmentation:`):

**Visual** (applied identically to both frames + mask):
- Color jitter (brightness, contrast, saturation, hue)
- Horizontal flip (with corresponding IMU axis negation)
- Random crop and resize
- Gaussian noise (sigma in [0, 0.02] * 255)

**IMU**:
- Additive Gaussian noise (accel: [0.01, 0.1] m/s^2, gyro: [0.001, 0.01] rad/s)
- Bias drift injection (random walk)
- Temporal offset perturbation ([-10, 10] ms)
- Sample dropout (10% rate)

---

## Pretrained Weights

### Download RAFT encoder weights

```bash
micromamba activate dynamask
python -m dynamask_vio.download_weights
```

This will:
1. Check if weights already exist at `dynamask_vio/weights/raft-things.pth`
2. Check local reference paths (e.g., `references/DPVO/thirdparty/RAFT/models/`)
3. If not found, download from the RAFT authors' release (~20MB)

Then update your config:
```yaml
model:
  raft_checkpoint: "dynamask_vio/weights/raft-things.pth"
```

**What gets loaded**: The RAFT checkpoint contains `module.fnet.*` and `module.cnet.*` keys. Our `load_raft_encoder_weights()` maps these to our BasicEncoder structure (conv1->stem.0, norm1->stem.1, layer1->stage1, etc.). FiLM layers are not in the RAFT checkpoint, so they initialize from scratch (near identity). Loading uses `strict=False`.

### Manual download

If automated download fails:
1. Clone `https://github.com/princeton-vl/RAFT`
2. Run `./download_models.sh`
3. Copy `models/raft-things.pth` to `dynamask_vio/weights/`

---

## Training

### Three-phase self-supervised schedule

| Phase | Epochs | Data | What trains | Losses | Notes |
|-------|--------|------|-------------|--------|-------|
| **1** | 1-20 | EuRoC | IMU encoder only | L_imu + L_cov | All vision modules frozen. Uses GT trajectory for supervision. |
| **2a** | 21-30 | VIODE + TartanAir | All (BA detached) | L_photo + L_smooth + L_reg + L_imu | BA runs but gradients don't flow to mask. Flow settles first. |
| **2b** | 31-70 | VIODE + TartanAir | All (BA gradients on) | ramp*L_pose + L_photo + L_reproj + L_reg + L_imu + L_smooth | L_pose ramps 0->1 over 5 epochs. Full self-supervised loop. |
| **3** | 71-80 | VIODE only | All (full loss) | L_pose + L_photo + L_reproj + L_reg + L_imu + L_smooth | Fine-tuning on dynamic scenes. All weights at 1.0. |

Phase transitions are automatic via `PhaseCallback`.

### All training commands

```bash
micromamba activate dynamask

# ── Basic training ──

# Default config (all phases, all datasets)
python -m dynamask_vio.train --config dynamask_vio/configs/default.yaml

# Resume from checkpoint
python -m dynamask_vio.train --config dynamask_vio/configs/default.yaml \
    --resume checkpoints/last.ckpt

# ── W&B options ──

# Disable W&B (TensorBoard only)
python -m dynamask_vio.train --no-wandb

# Custom W&B run name and tags
python -m dynamask_vio.train --wandb-name "exp-lr1e3-bs4" \
    --wandb-tags ablation high-lr small-batch

# ── Hardware options ──

# Multi-GPU (DDP)
python -m dynamask_vio.train --gpus 2

# Memory-constrained (12GB VRAM)
python -m dynamask_vio.train --config dynamask_vio/configs/viode_5ep_mem12gb.yaml

# ── Config overrides ──
# Edit configs/default.yaml directly, or create a new config file and pass it
```

### CLI arguments reference

| Argument | Default | Description |
|----------|---------|-------------|
| `--config` | `dynamask_vio/configs/default.yaml` | Path to YAML config |
| `--gpus` | 1 | Number of GPUs |
| `--resume` | None | Path to checkpoint to resume from |
| `--no-wandb` | false | Disable W&B, use TensorBoard only |
| `--wandb-name` | None | W&B run name |
| `--wandb-tags` | None | W&B tags (space-separated) |

### Key hyperparameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `training.lr` | 1e-4 | Base learning rate (OneCycleLR scheduler) |
| `training.batch_size` | 8 | Reduce to 4 or 2 for less VRAM |
| `training.grad_clip_norm` | 10.0 | Gradient clipping. Critical for BA stability. |
| `training.total_steps` | 240000 | Total OneCycleLR steps |
| `training.encoder_lr_multiplier` | 0.5 | Encoder LR = base * 0.5 (preserve RAFT features) |
| `training.imu_lr_multiplier` | 0.1 | IMU LR = base * 0.1 (preserve Phase 1 training) |
| `training.mixed_precision` | true | FP16 mixed precision. Disable if you see NaN. |
| `model.gru_iterations` | 3 | ConvGRU refinement iterations. Lower = faster but less accurate. |
| `model.corr_levels` | 4 | Correlation pyramid levels |
| `model.corr_radius` | 4 | Correlation lookup radius (window = 2r+1 = 9) |
| `ba.n_iters` | 3 | Gauss-Newton iterations per BA call |
| `ba.base_damping` | 1e-4 | Levenberg-Marquardt damping (matches DPVO) |
| `ba.ep_initial` | 10.0 | Extra diagonal damping at training start |
| `ba.ep_final` | 1.0 | Extra diagonal damping at convergence |
| `ba.num_correspondences` | 256 | Static points sampled for BA per frame pair |
| `ba.min_static_points` | 64 | Minimum static points required to run BA |
| `ba.outlier_threshold` | 100.0 | Reprojection error threshold for outlier rejection |
| `ba.convergence_threshold` | 50.0 | Mean reprojection error for BA convergence |
| `loss.lambda_pose` | 1.0 | BA-IMU pose consistency weight |
| `loss.lambda_photo` | 0.5 | Photometric consistency weight |
| `loss.lambda_reproj` | 0.3 | BA reprojection error weight |
| `loss.lambda_reg` | 0.1 | Mask regularisation weight |
| `loss.lambda_imu` | 1.0 | IMU preintegration error weight |
| `loss.lambda_cov` | 0.5 | Covariance NLL weight |
| `loss.lambda_smooth` | 0.2 | Edge-aware flow smoothness weight |
| `loss.pose_ramp_epochs` | 5 | L_pose ramp duration at Phase 2b start |

### Loss functions

| Loss | Active phases | Formula | Purpose |
|------|--------------|---------|---------|
| **L_imu** | All | `w_rot * ||Log(R_pred^T R_gt)||^2 + w_vel * ||v_pred - v_gt||^2 + w_pos * ||p_pred - p_gt||^2` | IMU encoder accuracy |
| **L_cov** | All | Negative log-likelihood with predicted Sigma [9x9] | Uncertainty calibration |
| **L_photo** | 2a, 2b, 3 | L1 on static regions (warped via flow) + relu(tau - error) on dynamic | Photometric consistency |
| **L_smooth** | 2a, 2b, 3 | `|grad(flow)| * exp(-|grad(image)|)` | Edge-aware flow regularization |
| **L_reg** | 2a, 2b, 3 | Ratio bounds [0.05, 0.6] + Total Variation | Prevent trivial all-static/all-dynamic masks |
| **L_pose** | 2b, 3 | `geodesic(R_ba, R_imu) + direction(t_ba, t_imu)` (direction-only for scale ambiguity) | BA-IMU consistency (main self-supervised signal) |
| **L_reproj** | 2b, 3 | Weighted MSE of BA reprojection residuals | BA correspondence quality |

### Checkpoints

Saved automatically to `checkpoints/` (configurable via `paths.checkpoint_dir`):
- `dynamask-v25-{epoch:02d}-{val/loss_total:.3f}.ckpt` -- top-3 by validation loss
- `last.ckpt` -- most recent epoch

Checkpoint format: PyTorch Lightning checkpoint with `state_dict` prefix `model.`. To load for downstream use:
```python
from dynamask_vio.inference import load_model
model = load_model("checkpoints/best.ckpt", "dynamask_vio/configs/default.yaml", device)
```

---

## Inference

### From images + IMU CSV

```bash
python -m dynamask_vio.inference \
    --checkpoint checkpoints/best.ckpt \
    --config dynamask_vio/configs/default.yaml \
    --images /path/to/image_dir/ \
    --imu /path/to/imu.csv \
    --output ./output/ \
    --threshold 0.5
```

**Image directory**: PNG or JPG files, sorted by filename.

**IMU CSV format** (with header row):
```csv
timestamp,ax,ay,az,gx,gy,gz
1597198367.700,0.12,-9.78,0.03,0.001,-0.002,0.001
1597198367.705,0.13,-9.79,0.02,0.001,-0.003,0.001
...
```

### From ROS bag (VIODE or EuRoC)

```bash
# VIODE bag
python -m dynamask_vio.inference \
    --checkpoint checkpoints/best.ckpt \
    --bag dataset/viode/parking_lot/3_high.bag \
    --output ./output/viode_parking_3high/

# Process first 100 frames only
python -m dynamask_vio.inference \
    --checkpoint checkpoints/best.ckpt \
    --bag dataset/viode/city_day/2_mid.bag \
    --output ./output/quick_test/ \
    --max-frames 100

# Skip overlay/heatmap generation (masks only, faster)
python -m dynamask_vio.inference \
    --checkpoint checkpoints/best.ckpt \
    --bag dataset/viode/parking_lot/3_high.bag \
    --output ./output/ \
    --no-overlays --no-probmaps
```

### Output structure

```
output/
  masks/           # binary mask PNGs (white=255 = dynamic, black=0 = static)
    000000.png
    000001.png
    ...
  overlays/        # mask overlaid on input image (green = dynamic, 40% alpha)
    000000.png
    ...
  probmaps/        # raw probability heatmaps (TURBO colormap, blue=0 -> red=1)
    000000.png
    ...
  results.json     # per-frame metadata
```

Each entry in `results.json`:
```json
{
  "frame": 0,
  "timestamp_prev": 1597198367.70,
  "timestamp_curr": 1597198367.75,
  "delta_R": [[0.999, -0.001, 0.002], [0.001, 0.999, -0.001], [-0.002, 0.001, 0.999]],
  "delta_v": [0.01, -0.02, 0.48],
  "delta_p": [0.0003, -0.0005, 0.012],
  "mean_mask_prob": 0.23,
  "dynamic_pixel_ratio": 0.15
}
```

### Inference CLI reference

| Flag | Default | Description |
|------|---------|-------------|
| `--checkpoint` | (required) | Path to trained checkpoint |
| `--config` | `dynamask_vio/configs/default.yaml` | Config YAML (must match training config) |
| `--images` | None | Directory of input images |
| `--imu` | None | IMU CSV file path |
| `--bag` | None | ROS bag path (alternative to --images/--imu) |
| `--output` | `./output` | Output directory |
| `--threshold` | 0.5 | Dynamic pixel probability threshold |
| `--max-frames` | 0 | Cap frame count (0 = process all) |
| `--no-overlays` | false | Skip overlay image generation |
| `--no-probmaps` | false | Skip probability heatmap generation |

---

## Evaluation

```bash
# Mask quality metrics require a dataset that provides true motion-mask labels.
# Current default datasets are self-supervised and do not provide gt_mask.
python -m dynamask_vio.evaluate \
    --checkpoint checkpoints/best.ckpt \
    --eval-mask

# IMU preintegration quality on EuRoC test set
python -m dynamask_vio.evaluate \
    --checkpoint checkpoints/best.ckpt \
    --eval-imu

# Trajectory comparison (requires TUM-format trajectory files)
python -m dynamask_vio.evaluate \
    --eval-traj estimated_traj.txt groundtruth_traj.txt
```

### Target metrics

| Metric | Target | Description |
|--------|--------|-------------|
| Mask IoU | > 0.50 (min), > 0.70 (stretch) | Intersection over Union on VIODE high-dynamic |
| Mask F1 | > 0.70 | Harmonic mean of precision and recall |
| IMU RTE | > 25% reduction | Relative Translation Error vs raw IMU |
| VIO ATE | > 20% improvement | Absolute Trajectory Error with masking vs without |
| EuRoC ATE | < 5% degradation | Must not hurt performance on static scenes |

---

## Logging and Debugging

### Weights & Biases

W&B is enabled by default. Set up:

```bash
# Option 1: .env file in project root (gitignored)
echo "WANDB_API_KEY=your_key_here" > .env
echo "WANDB_PROJECT=dynamask-vio" >> .env

# Option 2: global login
wandb login

# Option 3: environment variable
export WANDB_API_KEY=your_key_here
```

Disable with `--no-wandb`.

### TensorBoard (always active)

```bash
# Start TensorBoard
tensorboard --logdir logs/dynamask_v2/ --port 6006

# In a separate terminal or browser, open http://localhost:6006
```

### Offline logging

Runs in parallel with W&B (or standalone when `--no-wandb`). All debug data is saved locally to `logs/offline/<run_name>/`:

```
logs/offline/<run_name>/
  manifest.json       # config snapshot + run metadata (PID, timestamp)
  scalars.jsonl       # all scalar metrics (one JSON object per line)
  histograms.jsonl    # weight/gradient distributions
  alerts.jsonl        # anomaly detection events
  watch.jsonl         # periodic parameter snapshots (norms, means, stds)
  images/             # visualization PNGs organized by step
    step_00000100/
      000.png         # Frame t-1
      001.png         # Predicted mask overlay
      002.png         # Flow magnitude
      # If gt_mask is available, GT overlay + error map are also logged
  images.jsonl        # image index with captions
```

**Reading offline logs** (Python):
```python
import json

# Read scalars
with open("logs/offline/my_run/scalars.jsonl") as f:
    for line in f:
        row = json.loads(line)
        step = row["step"]
        metrics = row["metrics"]
        if "train/loss_total" in metrics:
            print(f"Step {step}: loss={metrics['train/loss_total']:.4f}")

# Read alerts
with open("logs/offline/my_run/alerts.jsonl") as f:
    for line in f:
        alert = json.loads(line)
        print(f"[{alert['level']}] Step {alert['step']}: {alert['title']} -- {alert['text']}")
```

### What gets logged

**Scalars** (every step by default):

| Category | Metrics | Description |
|----------|---------|-------------|
| **Losses** | `train/loss_imu`, `train/loss_cov`, `train/loss_photo`, `train/loss_smooth`, `train/loss_reg`, `train/loss_pose`, `train/loss_reproj`, `train/loss_total` | Per-component and total loss |
| **Phase** | `train/phase` (1, 2.0, 2.5, 3), `train/pose_ramp` (0.0-1.0) | Training phase and L_pose ramp value |
| **BA** | `train/ba_ran` (0/1), `train/ba_convergence_rate`, `train/ba_reproj_error_mean`, `train/ba_n_static_points`, `train/ba_static_weight_mean` | Whether BA ran, convergence, reprojection quality |
| **Gradients** | `gradients/norm_{feature_encoder,context_encoder,flow_decoder,imu_encoder,film_layers}`, `gradients/norm_total`, `gradients/ratio_*` | Per-submodule gradient norms and update ratios |
| **Mask** | `mask/mean_probability`, `mask/std_probability`, `mask/min_probability`, `mask/max_probability`, `mask/dynamic_pixel_ratio`, `mask/entropy` | Mask prediction statistics |
| **Flow** | `flow/magnitude_mean`, `flow/magnitude_max`, `flow/magnitude_std`, `flow/iter_0_magnitude_mean`, `flow/iter_1_magnitude_mean`, `flow/iter_2_magnitude_mean` | Flow magnitude and per-iteration convergence |
| **IMU** | `imu/mean_abs_delta_bg`, `imu/mean_abs_delta_ba`, `imu/max_abs_delta_bg`, `imu/max_abs_delta_ba`, `imu/mean_sigma2_g`, `imu/mean_sigma2_a` | Bias correction and noise variance magnitudes |
| **Preintegration** | `preint/delta_v_norm`, `preint/delta_p_norm`, `preint/delta_R_frob_from_I`, `preint/cov_diag_mean`, `preint/cov_diag_min`, `preint/cov_diag_max`, `preint/cov_min_eigenvalue` | Preintegration output and covariance health |
| **FiLM** | `film/stage{0,1,2}_gamma_weight_norm`, `film/stage{0,1,2}_beta_weight_norm` | FiLM modulation magnitude per encoder stage |
| **Performance** | `perf/batch_time_ms`, `perf/samples_per_sec`, `perf/gpu_memory_allocated_MB`, `perf/gpu_memory_reserved_MB` | Throughput and memory usage |
| **Validation** | `val/loss_total` | Validation loss (primary signal for self-supervised training) |

**Histograms** (every 100 steps): weight distributions, gradient distributions, mask probability distribution, flow magnitude distribution, bias corrections (delta_bg, delta_ba), noise variances (sigma2_g, sigma2_a), covariance diagonal

**Images** (every 200 steps): input frame t-1, predicted mask overlay (green), flow magnitude heatmap (TURBO colormap). If true mask GT exists in a dataset, GT overlay and error map are logged too.

**Alerts** (real-time anomaly detection):

| Alert | Trigger | Severity | Likely cause |
|-------|---------|----------|--------------|
| Loss spike | Loss > 5x running median | WARN | Unstable batch, high-dynamic scene |
| Gradient explosion | Any submodule norm > 100 | WARN | Learning rate too high, BA divergence |
| Gradient vanishing | Any submodule norm < 1e-8 for 10+ steps | WARN | Dead module, incorrect freezing |
| NaN/Inf in outputs | Any tensor contains NaN or Inf | ERROR | Numerical instability in BA or preintegration |
| Non-PD covariance | Min eigenvalue of Sigma < -1e-6 | WARN | Covariance propagation bug or overflow |
| Mask collapse | Mask std < 1e-4 | WARN | Model predicting constant value everywhere |

### Debugging guide

| Symptom | What to check | Likely cause | Fix |
|---------|--------------|--------------|-----|
| **Loss NaN** | `alerts.jsonl` for NaN/Inf | BA divergence or preint overflow | Increase `ba.base_damping`, reduce `training.lr`, disable `mixed_precision` |
| **Mask all ~0.5** | `mask/std_probability` stays < 0.01 | Mask collapse -- no learning signal | Check `train/ba_ran` (must be 1). Increase `loss.lambda_reg`. Verify intrinsics in batch. |
| **BA never runs** | `train/ba_ran` always 0 | No `intrinsics` key in batch | Check dataset config has `camera.fx/fy/cx/cy`. Verify dataset returns `intrinsics` tensor. |
| **BA never converges** | `train/ba_convergence_rate` stays 0 | Too few static points or poor flow | Reduce `ba.min_static_points` (e.g., 32). Increase `ba.convergence_threshold`. |
| **BA diverges** | `train/ba_reproj_error_mean` explodes | Damping too low or outlier threshold too high | Increase `ba.base_damping` (e.g., 1e-3). Reduce `ba.outlier_threshold` (e.g., 50). |
| **IMU not learning** | `imu/mean_abs_delta_bg` stays near 0 | Phase 1 not running or no EuRoC GT data | Verify EuRoC data paths. Check `train/phase` is 1 initially. |
| **FiLM not contributing** | `film/stage*_weight_norm` stays near initial | IMU features uninformative | Check `preint/delta_v_norm` is non-zero. May need longer Phase 1. |
| **Flow not converging** | `flow/iter_*_magnitude_mean` doesn't increase across iterations | Correlation volume issues | Verify image range is [0, 255] (not [0, 1]). Check `model.corr_levels`. |
| **Slow training** | `perf/batch_time_ms` > 1000 | Large batch or many GRU iters | Reduce `model.gru_iterations` to 2. Reduce `training.batch_size`. |
| **GPU OOM** | CUDA out of memory error | Batch too large for VRAM | Reduce `training.batch_size`. Reduce `model.corr_radius` to 3. |
| **Phase not switching** | `train/phase` stays at 1 | DataModule phase not updating | Check `PhaseCallback` is in callbacks list. Verify epoch count. |
| **No mask IoU shown** | Progress bar has no mask IoU metric | Current datasets do not provide true motion GT masks | Expected in current VIODE-only setup. Use `val/loss_total` as the primary validation signal. |

### Configure logging frequency

In `configs/default.yaml`:
```yaml
wandb:
  project: "dynamask-vio"
  entity: null                     # set via WANDB_ENTITY env var
  tags: []
  log_every_n_steps: 1             # scalars (set to 10 to reduce overhead)
  hist_every_n_steps: 100          # weight/gradient histograms
  image_every_n_steps: 200         # mask visualizations
  max_images: 4                    # images per visualization step
  watch_log_freq: 100              # wandb.watch gradient logging freq

offline_logging:
  enabled: true                    # always runs alongside W&B
  output_dir: "./logs/offline"
  mirror_when_wandb_enabled: true  # run both W&B and offline
  log_every_n_steps: 1
  hist_every_n_steps: 100
  image_every_n_steps: 200
  max_images: 4
  save_histograms: true
  save_images: true
  save_alerts: true
```

---

## ONNX Export

```bash
python -m dynamask_vio.export_onnx \
    --checkpoint checkpoints/best.ckpt \
    --output dynamask_v2.onnx \
    --opset 17
```

The exported model accepts dynamic batch size and variable IMU window length.

---

## Project Structure

```
dynamask/
  dynamask_vio/
    configs/
      default.yaml              # all hyperparameters (V2)
      viode.yaml                # VIODE intrinsics, splits, segmentation config
      euroc.yaml                # EuRoC intrinsics, splits, IMU noise params
      tartanair.yaml            # TartanAir V2 settings
      viode_5ep_mem12gb.yaml    # Low-memory quick training config
    models/
      __init__.py               # exports DynaMaskVIO
      dynamask.py               # full model wiring (fnet + cnet + FlowDecoder + IMU)
      backbone.py               # RAFT BasicEncoder with FiLM hooks + weight loader
      film.py                   # Feature-wise Linear Modulation (gamma/beta from IMU)
      flow_decoder.py           # CorrBlock + MotionEncoder + SepConvGRU + FlowHead + MaskHead
      differentiable_ba.py      # Gauss-Newton BA, SafeCholeskySolver, DPVO Jacobians
      imu_encoder.py            # NoiseCorrector + FeatureMLP + Preintegrator wrapper
      preintegration.py         # Differentiable IMU preintegration (PyPose SO3)
    losses/
      __init__.py               # re-exports all loss functions
      imu_loss.py               # imu_integration_loss + covariance_nll_loss
      self_supervised.py        # pose_consistency, photometric, reprojection, regularisation, smoothness
    data/
      __init__.py               # exports VIODEDataset, EuRoCDataset, TartanAirDataset
      viode_dataset.py          # VIODE ROS bag loader (images, IMU, segmentation, GT odom)
      euroc_dataset.py          # EuRoC ASL format loader (auto-discovers nested dirs)
      tartanair_dataset.py      # TartanAir V2 loader
      augmentations.py          # VisualAugmentor + IMUAugmentor (operates on [0,255] images)
      imu_utils.py              # get_imu_window, compute_gt_preintegration, compute_gt_bias
    train.py                    # PyTorch Lightning training (DynaMaskLitModule, PhaseDataModule, PhaseCallback)
    inference.py                # Single-sequence inference (images+IMU or ROS bag, saves masks/overlays/probmaps)
    evaluate.py                 # Evaluation metrics (IoU, F1, RTE)
    export_onnx.py              # ONNX export
    download_weights.py         # Download RAFT pretrained weights
    wandb_debug.py              # WandbDebugCallback + WandbModelWatchCallback
    offline_debug.py            # OfflineDebugCallback + OfflineModelWatchCallback + OfflineWriter
    weights/
      raft-things.pth           # RAFT pretrained (after running download_weights.py)
  dataset/
    viode/                      # VIODE ROS bags (city_day/, city_night/, parking_lot/)
    euroc/                      # EuRoC ASL sequences (machine_hall/, vicon_room1/, vicon_room2/)
  docs/
    DynaMask_V2_Method.md       # Detailed method design document
    DynaMask_VIO_FINAL_Implementation_Spec.md
    DynaMask_VIO_Project_Plan.md
  checkpoints/                  # saved during training
  logs/                         # TensorBoard logs + offline debug logs
```

---

## Configuration Reference

Complete `configs/default.yaml`:

```yaml
# ── Model (V2: RAFT-style encoder + iterative flow decoder) ──
model:
  imu_feature_dim: 128        # IMU feature vector dimension
  encoder_output_dim: 128     # RAFT encoder output channels
  gru_hidden_dim: 128         # SepConvGRU hidden dimension
  gru_iterations: 3           # flow refinement iterations (1-5, 3 recommended)
  corr_levels: 4              # correlation pyramid levels
  corr_radius: 4              # correlation lookup radius (window = 2r+1)
  raft_checkpoint: null       # path to RAFT pretrained weights (null = train from scratch)

# ── IMU Encoder ──
imu:
  noise_corrector_hidden: 64          # CNN hidden channels
  noise_corrector_dilations: [1, 2, 4] # dilated conv kernel sizes
  initial_variance: 1e-4              # initial noise variance estimate

# ── Training ──
training:
  batch_size: 8               # reduce to 4/2 for less VRAM
  num_workers: 8              # dataloader workers
  epochs: 80                  # total training epochs
  lr: 1e-4                    # base learning rate
  weight_decay: 1e-4          # AdamW weight decay
  grad_clip_norm: 10.0        # gradient clipping (critical for BA stability)
  mixed_precision: true       # FP16 mixed precision
  total_steps: 240000         # OneCycleLR total steps
  phase1_epochs: 20           # Phase 1: IMU pretraining on EuRoC
  phase2a_epochs: 10          # Phase 2a: photometric warmup, BA detached
  phase2b_epochs: 40          # Phase 2b: full BA gradients with L_pose ramp
  encoder_lr_multiplier: 0.5  # encoder LR = base * this
  imu_lr_multiplier: 0.1      # IMU LR = base * this

# ── Loss weights ──
loss:
  lambda_pose: 1.0            # BA pose vs IMU pose (main self-supervised signal)
  lambda_photo: 0.5           # photometric consistency
  lambda_reproj: 0.3          # BA reprojection error
  lambda_reg: 0.1             # mask regularisation (ratio + TV)
  lambda_imu: 1.0             # IMU preintegration error
  lambda_cov: 0.5             # covariance NLL
  lambda_smooth: 0.2          # edge-aware flow smoothness
  pose_rotation_weight: 5.0   # rotation vs translation in L_pose
  pose_direction_weight: 1.0  # translation direction weight in L_pose
  pose_ramp_epochs: 5         # L_pose ramp 0->1 duration at Phase 2b start
  imu_rotation_weight: 5.0    # rotation weight in L_imu
  imu_velocity_weight: 1.0    # velocity weight in L_imu
  imu_position_weight: 1.0    # position weight in L_imu

# ── Bundle Adjustment ──
ba:
  n_iters: 3                  # Gauss-Newton iterations
  base_damping: 1e-4          # LM damping (matches DPVO)
  ep_initial: 10.0            # extra diagonal damping at start
  ep_final: 1.0               # extra diagonal damping at convergence
  ep_decay_epochs: 10         # epochs over which ep decays
  outlier_threshold: 100.0    # reprojection error outlier threshold
  convergence_threshold: 50.0 # mean reproj error for convergence
  num_correspondences: 256    # static points sampled per frame
  min_static_points: 64       # minimum points to run BA

# ── Data ──
data:
  enabled_datasets: ["viode"]   # add "euroc"/"tartanair" when those datasets are available
  image_height: 480
  image_width: 640
  imu_max_window_size: 15     # max IMU samples per frame pair (padded)

# ── Camera (override per dataset in viode.yaml/euroc.yaml) ──
camera:
  fx: 320.0
  fy: 320.0
  cx: 320.0
  cy: 240.0

# ── Augmentation ──
augmentation:
  color_jitter_p: 0.8
  brightness: 0.2
  contrast: 0.2
  saturation: 0.2
  hue: 0.1
  horizontal_flip_p: 0.5
  random_crop_p: 0.3
  random_crop_scale: [0.8, 1.0]
  gaussian_noise_p: 0.3
  gaussian_noise_sigma: [0.0, 0.02]  # as fraction of 255
  imu_noise_accel: [0.01, 0.1]       # sigma range m/s^2
  imu_noise_gyro: [0.001, 0.01]      # sigma range rad/s
  bias_drift_p: 0.5
  bias_drift_b0_sigma: 0.01
  bias_drift_alpha_sigma: 0.001
  temporal_offset_p: 0.5
  temporal_offset_ms: [-10, 10]
  imu_dropout_p: 0.3
  imu_dropout_rate: 0.1

# ── Evaluation ──
eval:
  mask_threshold: 0.5
  min_features_fallback: 20

# ── Paths ──
paths:
  viode_root: "./dataset/viode"
  euroc_root: "./dataset/euroc"
  tartanair_root: "./dataset/tartanair"
  checkpoint_dir: "./checkpoints"
  log_dir: "./logs"

# ── W&B ──
wandb:
  project: "dynamask-vio"
  entity: null
  tags: []
  log_every_n_steps: 1
  hist_every_n_steps: 100
  image_every_n_steps: 200
  max_images: 4
  watch_log_freq: 100

# ── Offline Logging ──
offline_logging:
  enabled: true
  output_dir: "./logs/offline"
  mirror_when_wandb_enabled: true
  log_every_n_steps: 1
  hist_every_n_steps: 100
  image_every_n_steps: 200
  max_images: 4
  save_histograms: true
  save_images: true
  save_alerts: true

# ── Camera-to-IMU extrinsic (identity default, override per dataset) ──
extrinsics:
  T_CI: [[1, 0, 0, 0],
         [0, 1, 0, 0],
         [0, 0, 1, 0],
         [0, 0, 0, 1]]
```

---

## Environment Variables

| Variable | Description | Example |
|----------|-------------|---------|
| `WANDB_API_KEY` | W&B API key (alternative to `.env` or `wandb login`) | `abc123...` |
| `WANDB_PROJECT` | Override W&B project name | `dynamask-vio` |
| `WANDB_ENTITY` | Override W&B team/entity | `my-team` |
| `WANDB_MODE` | Set to `offline` for offline-only W&B | `offline` |
| `CUDA_VISIBLE_DEVICES` | Select GPUs | `0,1` |

---

## Memory and Compute Requirements

### GPU memory (approximate, single GPU)

| Batch size | Resolution | GRU iters | Corr radius | VRAM (FP16) | VRAM (FP32) |
|-----------|-----------|-----------|-------------|-------------|-------------|
| 8 | 480x640 | 3 | 4 | ~10 GB | ~18 GB |
| 4 | 480x640 | 3 | 4 | ~6 GB | ~10 GB |
| 2 | 480x640 | 3 | 4 | ~4 GB | ~6 GB |
| 8 | 480x640 | 2 | 3 | ~7 GB | ~12 GB |

**Reducing memory**:
- Lower `training.batch_size` (most effective)
- Lower `model.gru_iterations` (3 -> 2)
- Lower `model.corr_radius` (4 -> 3, reduces corr volume from 324 to 196 channels)
- Enable `training.mixed_precision: true`
- Reduce `ba.num_correspondences` (256 -> 128)

### Training time (approximate, single A100)

| Phase | Epochs | Time per epoch | Total |
|-------|--------|----------------|-------|
| Phase 1 (EuRoC) | 20 | ~5 min | ~1.5 hr |
| Phase 2a+2b (VIODE) | 50 | ~15 min | ~12.5 hr |
| Phase 3 (VIODE) | 10 | ~15 min | ~2.5 hr |
| **Total** | 80 | | **~16.5 hr** |

### Inference speed

| GPU | Resolution | FPS (single frame pair) |
|-----|-----------|------------------------|
| A100 | 480x640 | ~25 |
| RTX 3090 | 480x640 | ~18 |
| RTX 3060 | 480x640 | ~10 |
| CPU (i7) | 480x640 | ~0.5 |

---

## Troubleshooting

### Installation issues

**`ImportError: No module named 'pypose'`**
```bash
pip install pypose
```

**`ImportError: No module named 'rosbags'`**
```bash
pip install rosbags
```

**`torch.cuda.is_available()` returns False**
```bash
# Check CUDA installation
nvidia-smi
python -c "import torch; print(torch.version.cuda)"
# Reinstall PyTorch with correct CUDA version
```

### Dataset issues

**`[EuRoC] Warning: sequence not found: MH_01_easy`**
- Check `paths.euroc_root` points to the correct directory
- The loader searches recursively, so any of these layouts work:
  - `euroc_root/MH_01_easy/mav0/`
  - `euroc_root/machine_hall/MH_01_easy/mav0/`
  - `euroc_root/machine_hall/MH_01_easy/MH_01_easy/mav0/`

**`[VIODE] Warning: bag not found`**
- Verify the bag path relative to `paths.viode_root`
- Splits in `configs/viode.yaml` use format `city_day/1_low` (without `.bag` extension)
- The loader appends `.bag` automatically if needed

**`[VIODE] train: 0 frame pairs from 0 sequences`**
- No valid bags found at specified paths
- Check that bags are not corrupted: `python -c "from rosbags.rosbag1 import Reader; r = Reader('path/to/bag'); r.open()"`

### Training issues

**Loss immediately NaN**
1. Disable mixed precision: `training.mixed_precision: false`
2. Increase BA damping: `ba.base_damping: 1e-3`
3. Reduce learning rate: `training.lr: 5e-5`
4. Check that images are in [0, 255] range (not [0, 1])

**BA never runs (ba_ran = 0)**
- Verify `intrinsics` key exists in batch: add `camera.fx/fy/cx/cy` to your dataset config
- Check `ba.min_static_points` -- if mask predicts everything as dynamic, BA won't have enough points

**Mask collapse (all predictions ~0.5)**
- Increase `loss.lambda_reg` (e.g., 0.5)
- Verify BA is running and providing gradient signal
- Check gradient norms for `flow_decoder` and `film_layers` -- should be non-zero

**Training stuck in Phase 1**
- This is expected for first 20 epochs. Check `train/phase` metric.
- If EuRoC dataset is empty (0 pairs), training may appear stuck but is actually running on empty batches

**Out of memory**
- Reduce `training.batch_size` first
- Reduce `model.gru_iterations` (3 -> 2)
- Reduce `model.corr_radius` (4 -> 3)

**W&B not logging**
- Check `WANDB_API_KEY` is set
- Try `wandb login` first
- Check `logs/` for TensorBoard files (always created as fallback)

---

## References

- Teed & Deng, "RAFT: Recurrent All-Pairs Field Transforms for Optical Flow", ECCV 2020
- Teed & Deng, "Deep Patch Visual Odometry" (DPVO), NeurIPS 2023
- Forster et al., "On-Manifold Preintegration for Real-Time Visual-Inertial Odometry", TRO 2017
- Minoda et al., "VIODE: A Simulated Dataset to Address the Challenges of Visual-Inertial Odometry in Dynamic Environments", IEEE RA-L 2021

## License

TBD
