# DynGRU Training Guide

## Quick start

```bash
# 1. Convert VIODE ROS bags to TartanAir v2 format (one-time)
python3 Scripts/convert_viode_to_tartanair.py

# 2. Train on VIODE high-difficulty sequences, self-supervised
python3 -m Train.MatchingNet.train_flowformer \
    --config Config/Train/DynGRU_VIODE.yaml \
    --training_mode dyn_selfsup \
    --wandb

# 3. Train on TartanAir v2 all scenes (supervised, GT flow + depth)
python3 -m Train.MatchingNet.train_flowformer \
    --config Config/Train/DynGRU_TartanAir2_HPC_Supervised.yaml \
    --training_mode dyn \
    --wandb

# 4. Train on VIODE all sequences (self-supervised, no GT)
python3 -m Train.MatchingNet.train_flowformer \
    --config Config/Train/DynGRU_VIODE_HPC_SelfSup.yaml \
    --training_mode dyn_selfsup \
    --wandb
```

---

## Config file structure

Training configs live in `Config/Train/`. Each config has two sections:

### Section 1: `Model` — everything about the model and training hyperparameters

```yaml
Model:
  name: "DynGRU_VIODE"          # Used for wandb project name and save paths
  gamma: 0.85                    # Exponential decay weight for multi-iteration loss (γ^k)
  max_flow: 400                  # Max flow magnitude (px) for flow loss mask
  batch_size: 1                  # Batch size (VIODE=1, TartanAir can be 2-4)
  restore_ckpt: "Model/MACVO_FrontendCov.pth"  # Pretrained FlowFormerCov checkpoint
  training_mode: "dyn_selfsup"   # "dyn" (GT depth+flow), "dyn_selfsup" (estimated depth+flow)

  ### TRAINER
  num_steps: 37540               # Total training steps (3754 pairs × 10 epochs = 37540)
  mixed_precision: True           # FP16 training (saves VRAM, faster)
  lr: 2.0e-4                     # Learning rate (AdamW)
  clip: 1.0                      # Gradient clipping norm
  autosave_freq: 2000            # Save checkpoint every N steps
  log_freq: 100                  # Log metrics to wandb every N steps
  visual_freq: 200               # Save debug PNGs every N steps
  num_workers: 0                 # DataLoader workers (0 for IterableDataset)
  seed: 1234
  wandb: True                    # Enable/disable wandb logging

  ### latentcostformer — FlowFormer architecture (DO NOT CHANGE unless retraining backbone)
  latentcostformer:
    decoder_depth: 12            # K=12 GRU iterations
    ...
```

### Section 2: `Train` — which datasets to use

```yaml
Train:
  data: !flatten_seq
    - !include ../Sequence/Training_Dataset/VIODE_High.yaml
```

This points to a dataset config file. Each dataset config is a list of sequences:

```yaml
-   type: TartanAirv2                  # Maps to TartanAirV2_Sequence class
    name: city_day_3_high
    args:
        root: /home/karan/dynamask/dataset/viode_tartan/city_day/3_high
        compressed: false               # false = npy images, true = png images
        use_real_imu: true              # true = load real IMU npy files
        gravity: 9.81
        imu_freq: 200                   # IMU sample rate (Hz)
        imu_sim: { ... }                # Only used when use_real_imu=false
        gtDepth: false                  # Enable GT depth (TartanAir has it, VIODE doesn't)
        gtPose: true                    # Enable GT pose (required for rigid flow)
        gtFlow: false                   # Enable GT flow (TartanAir has it, VIODE doesn't)
```

---

## Training modes

| Mode | Flag | GT depth | GT flow | Supervision | When to use |
|------|------|----------|---------|-------------|-------------|
| `dyn` | `--training_mode dyn` | Required | Required | GT depth + GT flow → rigid flow pseudo-labels | TartanAir (has GT) |
| `dyn_selfsup` | `--training_mode dyn_selfsup` | Not needed | Not needed | FlowFormer stereo depth + FlowFormer flow + GT pose | VIODE (no GT depth/flow) |

Both modes use the **same residual regression loss**:

```
target = ||f_target − f_rigid||        # Residual magnitude [px] — from GT flow (dyn) or estimated flow (dyn_selfsup)
r_hat  = softplus(dyn_logit)           # Model predicts residual magnitude in pixels
loss   = Σ γ^k · smooth_L1(r_hat, target)    # γ-weighted across K=12 iterations
```

No cov head, no Mahalanobis normalization, no BCE, no confidence targets. The dyn head directly predicts how much each pixel deviates from rigid flow in pixels. At inference: `r_hat > 2px → flag as dynamic`.

---

## How to set up a new dataset

### Step 1: Get data into TartanAir v2 directory format

```
sequence_root/
  image_lcam_front/  000000.png 000001.png ...    # Left camera
  image_rcam_front/  000000.png 000001.png ...    # Right camera
  imu/
    acc.npy     (N,3) float64    # Accelerometer
    gyro.npy    (N,3) float64    # Gyroscope
    imu_time.npy (N,) float64    # IMU timestamps (seconds)
    cam_time.npy (M,) float32    # Camera timestamps (seconds)
  pose_lcam_front.txt  (M,7)     # [tx ty tz qx qy qz qw] per frame
```

For ROS bags, use: `python3 Scripts/convert_viode_to_tartanair.py`

### Step 2: Create dataset config

```yaml
# Config/Sequence/Training_Dataset/MyDataset.yaml
-   type: TartanAirv2
    name: my_sequence
    args:
        root: /path/to/sequence_root
        compressed: false
        use_real_imu: true
        gravity: 9.81
        imu_freq: 200
        imu_sim:
            acc_bias: [0.0, 0.0, 0.0]
            ...
        gtDepth: false
        gtPose: true
        gtFlow: false
```

### Step 3: Create training config

```yaml
# Config/Train/MyTraining.yaml
Model:
  name: "MyExperiment"
  ...
  num_steps: <total_pairs * epochs>
  ...

Train:
  data: !flatten_seq
    - !include ../Sequence/Training_Dataset/MyDataset.yaml
```

### Step 4: Launch

```bash
micromamba run -n dynamask python3 -m Train.MatchingNet.train_flowformer \
    --config Config/Train/MyTraining.yaml \
    --training_mode dyn_selfsup \
    --wandb
```

---

## How to monitor training

### Command line (real-time)

```bash
# Latest loss values
tail -100000 /tmp/dyngru_viode_high2.log | grep -oP 'Iter: \d+, Loss: [\d.]+' | tail -5

# Current epoch
echo "Epoch: $(( $(tail -100000 /tmp/dyngru_viode_high2.log | grep -oP 'Iter: (\d+)' | tail -1 | grep -oP '\d+') / 3754 )) / 10"

# Error count
grep -c 'Traceback\|CRASH' /tmp/dyngru_viode_high2.log
```

### Wandb dashboard

Training metrics appear under these groups:

| Group | Key metrics |
|-------|-------------|
| `train/` | `dyn_r_mean`, `dyn_r_max` (predicted residual [px]), `lr` |
| `dyngru/` | `alpha` (adapter gate), `token_weights_mean`, `grad_dyn_update`, `grad_frozen` (must be 0) |
| `imu/` | `f_imu_mean`, `f_imu_std` |
| `pseudo/` | `residual_mean` (target residual [px]) |
| `epe` | End-point error (flow accuracy) |

### Offline debug artifacts

Saved every `visual_freq` steps at `Results/<run_name>/debug/step_XXXXXX/`:

| File | What it shows |
|------|--------------|
| `dyn_overlay.png` | Input image + predicted residual r_hat heatmap + target residual (hot=high residual → likely dynamic) |
| `residual_map.png` | Multi-panel: `\\|f_est − f_rigid\\|`, GT residual (if available), flow covariance, predicted r_hat |
| `flow_comparison.png` | Estimated flow vs rigid flow vs difference |
| `imu_features.png` | IMU token heatmap (7×128) + f_imu line plot |
| `histograms.png` | r_hat histogram, residual distribution, token weights, gradient norms |
| `tensors.pt` | All raw tensors for custom analysis |
| `metadata.json` | Step number, timestamp, git hash |

### Model checkpoints

Saved every `autosave_freq` steps at `Model/<run_name>/<step>.pth`.

---

## Key parameters to change between runs

| Parameter | Where | When to change |
|-----------|-------|---------------|
| `num_steps` | `Config/Train/*.yaml` | Different dataset size or epoch count |
| `batch_size` | `Config/Train/*.yaml` | More/less GPU memory. VIODE must be 1 (IMU windows vary per sequence) |
| `visual_freq` | `Config/Train/*.yaml` | More/less frequent debug dumps |
| `log_freq` | `Config/Train/*.yaml` | More/less frequent wandb logging |
| `lr` | `Config/Train/*.yaml` | Different learning rate |
| `restore_ckpt` | `Config/Train/*.yaml` | Different pretrained checkpoint |
| `training_mode` | CLI `--training_mode` | Switch between `dyn` (has GT) and `dyn_selfsup` (no GT) |
| `gtDepth`, `gtPose`, `gtFlow` | Dataset config | Enable/disable GT data loading per sequence |
| `use_real_imu` | Dataset config | Real IMU files vs synthetic IMU from GT pose |
| Dataset list | Dataset config | Add/remove sequences |

---

## Current configs reference

### Training configs

| Config | Dataset | Mode | Epochs |
|--------|---------|------|--------|
| `DynGRU_VIODE.yaml` | VIODE (3 high-difficulty sequences) | `dyn_selfsup` | 10 |
| `DynGRU_VIODE_HPC_SelfSup.yaml` | VIODE (all 12 sequences) | `dyn_selfsup` | 10 |
| `DynGRU_TartanAir2_HPC_Supervised.yaml` | TartanAir2 (1,122 sequences across 74 scenes) | `dyn` | 10 |
| `DynGRU_tartanair2.yaml` | TartanAir2 (local, single sequence) | `dyn` | varies |

### Dataset configs

| Dataset config | Sequences | Type |
|---------------|-----------|------|
| `TartanAirV2_HPC_All.yaml` | 1,122 (all TartanAir2 scenes, Data_easy + Data_hard) | TartanAir v2, real IMU, GT depth+flow+pose |
| `VIODE_HPC_SelfSup.yaml` | 12 (all VIODE: city_day, city_night, parking_lot × 4 difficulties) | VIODE TartanAirv2 format, real IMU, GT pose only |
| `VIODE_High.yaml` | 3 (city_day, city_night, parking_lot × 3_high) | VIODE TartanAirv2 format, GT pose only |
| `TartanAirV2_Dyn_Local.yaml` | 1 (AbandonedFactory P001) | TartanAir v2, real IMU, GT depth+flow+pose (local) |
| `CarWelding_All.yaml` | 9 (CarWelding P000-P008) | TartanAir v2, real IMU, GT depth+flow+pose |
