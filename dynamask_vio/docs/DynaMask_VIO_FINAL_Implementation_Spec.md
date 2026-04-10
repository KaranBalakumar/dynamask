# DynaMask-VIO: Implementation Specification
## For Claude Code — Every Decision is Final, No Alternatives

---

## WHAT THIS PROJECT IS

A PyTorch deep learning system with two modules trained jointly:

1. **DynaMask**: Takes two consecutive monocular images + IMU data between them → outputs a per-pixel binary mask of dynamic regions
2. **IMU-Head**: Takes raw IMU window → outputs bias corrections, per-sample noise covariance, and preintegrated motion estimates

The masked image feeds into OpenVINS (an existing open-source VIO system). The IMU-Head outputs replace OpenVINS's constant noise parameters.

The camera intrinsics ARE used during training (available in all datasets) but the model architecture itself does NOT take intrinsics as input — it works in learned feature space. The camera-to-IMU extrinsic `T_CI` is a known constant loaded from a config file.

Synchronization between IMU and camera IS assumed at the dataset level (all training datasets are synchronized). Temporal robustness is achieved through data augmentation (random time offset injection during training). Online temporal offset estimation is OUT OF SCOPE for V1.

---

## PROJECT STRUCTURE

```
dynamask_vio/
├── configs/
│   ├── default.yaml              # All hyperparameters in ONE file
│   ├── viode.yaml                # VIODE dataset-specific overrides
│   ├── tartanair.yaml            # TartanAir dataset-specific overrides
│   └── euroc.yaml                # EuRoC dataset-specific overrides
├── data/
│   ├── __init__.py
│   ├── viode_dataset.py          # VIODE dataloader (primary training set)
│   ├── tartanair_dataset.py      # TartanAir V2 dataloader (supplementary)
│   ├── euroc_dataset.py          # EuRoC dataloader (IMU-head pretraining + static baseline)
│   ├── augmentations.py          # ALL augmentations (visual, IMU, temporal)
│   └── imu_utils.py              # IMU interpolation, windowing, preintegration GT
├── models/
│   ├── __init__.py
│   ├── dynamask.py               # Full model: backbone + FiLM + decoder + IMU head
│   ├── backbone.py               # DDRNet-23-slim (slimmed for binary segmentation)
│   ├── imu_encoder.py            # Noise corrector + differentiable preintegrator + feature MLP
│   ├── film.py                   # FiLM conditioning layers
│   ├── temporal_decoder.py       # Correlation volume + ConvGRU + output heads
│   └── preintegration.py         # Differentiable IMU preintegration (SO(3) + R^6)
├── losses/
│   ├── __init__.py
│   ├── mask_loss.py              # Focal loss for dynamic mask
│   ├── imu_loss.py               # Integration error loss + covariance NLL loss
│   └── flow_loss.py              # Flow residual loss (optional, for datasets with GT flow)
├── train.py                      # Training script (PyTorch Lightning)
├── evaluate.py                   # Evaluation: mask IoU + VIO trajectory metrics
├── export_onnx.py                # Export to ONNX for deployment
├── inference.py                  # Single-sequence inference script
└── requirements.txt
```

---

## DATASETS — EXACTLY WHAT TO DOWNLOAD AND HOW TO USE EACH

### Dataset 1: VIODE (PRIMARY — Training + Evaluation)

**What it is**: Simulated UAV in dynamic environments. 3 environments × 4 dynamic levels (none, low, medium, high). Synchronized stereo images + IMU + ground truth segmentation masks + ground truth trajectory.

**Download**: From Zenodo at https://zenodo.org/record/4493401
- Format: ROS bags
- Extract using: `rosbag` Python API or `ros_readbagfile`

**What we use from each bag**:
- `/cam0/image_raw` — 640×480 RGB images at 20Hz
- `/imu0` — 6-axis IMU at 200Hz (accel + gyro)
- `/cam0/segmentation` — Per-pixel semantic segmentation (GT)
- Ground truth trajectory from associated CSV files

**How to create binary dynamic masks from segmentation**:
```python
# VIODE segmentation labels — these are the DYNAMIC classes:
DYNAMIC_CLASSES = [
    # People/characters (the moving actors in VIODE scenes)
    # Check VIODE's seg_label_map for exact IDs — they use AirSim class IDs
    # People are typically class IDs that correspond to "Character" or "Person"
]
# For VIODE: ANY pixel that is NOT background/building/road/vegetation/sky is dynamic
# Simple heuristic: extract unique class IDs from static sequences, everything else is dynamic
# The VIODE paper's VINS-Mask does exactly this approach

dynamic_mask = np.zeros_like(seg_image[:,:,0], dtype=np.float32)
for cls_id in DYNAMIC_CLASSES:
    dynamic_mask[seg_image[:,:,0] == cls_id] = 1.0
```

**Train/Val/Test split**:
- Train: city_day (none, low, medium, high) + city_night (none, low, medium, high) 
- Val: parking_lot (low, medium)
- Test: parking_lot (high) + city_night (high)

**IMU ground truth for bias**: In simulation, the "true" IMU readings are noise-free. VIODE's IMU includes simulated noise. GT bias can be derived: `b_gt = imu_noisy - imu_ideal`. If ideal IMU is not available in bags, use ground truth trajectory + gravity to back-compute ideal readings:
```
a_ideal = R_wb^T @ (a_world - g)
w_ideal = R_wb^T @ w_world
b_a = a_raw - a_ideal
b_g = w_raw - w_ideal
```

### Dataset 2: TartanAir V2 (SUPPLEMENTARY — More diverse training data)

**What it is**: Large-scale simulated dataset with diverse environments. Has images, depth, segmentation, IMU, optical flow, poses.

**Download**: Using the `tartanair` Python package
```python
import tartanair as ta
ta.init('/path/to/tartanair')
# Download 5-10 environments with dynamic content:
for env in ['Downtown', 'OldTown', 'Neighborhood', 'UrbanTree', 'AbandonedCable']:
    ta.download(env=env, difficulty=['easy', 'hard'],
                modality=['image', 'seg', 'imu', 'flow'],
                camera_name=['lcam_front'], unzip=True)
```

**What we use**:
- `image_lcam_front/` — RGB images
- `seg_lcam_front/` — Semantic segmentation
- `imu/` — Simulated IMU data
- `flow/` — Ground truth optical flow (for flow residual loss)
- `pose_lcam_front.txt` — Ground truth poses (NED frame)

**Dynamic mask generation**: Same approach as VIODE — identify class IDs of people, vehicles, animals from the segmentation labels and create binary masks.

**Role in training**: Provides diverse visual environments that VIODE (only 3 environments) cannot cover. Used in all training phases alongside VIODE.

### Dataset 3: EuRoC MAV (IMU HEAD PRETRAINING + STATIC BASELINE)

**What it is**: Real-world MAV dataset. Indoor, mostly static scenes. High-quality synchronized camera + IMU + VICON ground truth.

**Download**: From https://projects.asl.ethz.ch/datasets/doku.php?id=kmavvisualinertialdatasets
- ASL format (CSV + images) — easier to work with than ROS bags

**What we use**:
- `cam0/data/` — 752×480 grayscale images at 20Hz
- `imu0/data.csv` — 6-axis IMU at 200Hz
- `state_groundtruth_estimate0/data.csv` — Positions, quaternions, velocities, biases

**Role**: 
1. **IMU Head pretraining (Phase 1)**: Train the noise corrector using EuRoC's GT biases and GT trajectory
2. **Static baseline**: Run OpenVINS with and without our masking to verify we don't degrade performance on static scenes
3. **NOT used for mask training** (no dynamic objects)

**Train/Val/Test split for IMU head**:
- Train: MH_01, MH_03, MH_05, V1_02, V2_01, V2_03
- Val: MH_02, V1_01
- Test: MH_04, V1_03, V2_02

---

## MODEL ARCHITECTURE — EXACT SPECIFICATIONS

### A. IMU Encoder (`imu_encoder.py`)

**CRITICAL DATA FLOW — DO NOT WIRE THIS WRONG:**
```
Raw IMU ──► Noise Corrector ──► CORRECTED IMU ──► Preintegrator ──► ΔR, Δv, Δp, Σ ──► Feature MLP ──► f_imu ──► FiLM conditioning
                │                                       │
                ├──► δb_g, δb_a  (bias corrections)     ├──► ΔR, Δv, Δp (to VIO backend)
                └──► σ²_g, σ²_a  (to VIO backend)       └──► Σ_preint   (to VIO backend)
```
The FiLM feature `f_imu` is derived from the CORRECTED preintegration, not raw. This is essential:
the mask loss backpropagates through FiLM → MLP → Preintegrator → Noise Corrector, so the
noise corrector learns from BOTH the direct IMU loss AND the mask loss. Better corrections →
better ego-motion prior → better dynamic/static classification → lower mask loss.

**Input**: `imu_window` tensor of shape `[B, N, 7]` where N is variable (typically 5-10 samples between frames at 200Hz/20-30Hz), 7 = (timestamp_delta, ax, ay, az, gx, gy, gz). `timestamp_delta` is relative to window start.

**Sub-module A1: Noise Corrector** (dilated 1D CNN)
```
Input: [B, N, 6] (accel + gyro, timestamps stripped)
Transpose to [B, 6, N] for Conv1d

Layer 1: Conv1d(in=6, out=64, kernel=3, padding='same', dilation=1) → BatchNorm1d → ReLU
Layer 2: Conv1d(in=64, out=64, kernel=3, padding='same', dilation=2) → BatchNorm1d → ReLU  
Layer 3: Conv1d(in=64, out=64, kernel=3, padding='same', dilation=4) → BatchNorm1d → ReLU

Bias Head: Conv1d(in=64, out=6, kernel=1) → output [B, 6, N] → transpose to [B, N, 6]
    Splits into: delta_bg [B, N, 3], delta_ba [B, N, 3]
    
Variance Head: Conv1d(in=64, out=6, kernel=1) → Softplus → output [B, 6, N] → transpose to [B, N, 6]  
    Splits into: sigma2_g [B, N, 3], sigma2_a [B, N, 3]
    Initialize Softplus bias so initial variance ≈ 1e-4 (typical IMU noise level)

Corrected measurements:
    gyro_corrected = gyro_raw - delta_bg
    accel_corrected = accel_raw - delta_ba
```

**Parameters**: ~50K

**Sub-module A2: Differentiable Preintegrator** (`preintegration.py`)
```
Input: corrected gyro [B,N,3], corrected accel [B,N,3], 
       sigma2_g [B,N,3], sigma2_a [B,N,3], dt [B,N,1]

Process: Standard Forster et al. preintegration, fully in PyTorch (no C++ needed)
    - Rotation: incremental Exp map on SO(3) using Rodrigues formula
    - Velocity: ΔR @ accel * dt accumulated  
    - Position: velocity * dt + 0.5 * ΔR @ accel * dt^2 accumulated
    - Covariance: First-order propagation using Jacobians A_k, B_k
    
Output:
    delta_R: [B, 3, 3]     — preintegrated rotation matrix
    delta_v: [B, 3]         — preintegrated velocity
    delta_p: [B, 3]         — preintegrated position  
    Sigma:   [B, 9, 9]      — preintegration covariance
```

**Parameters**: 0 (pure computation, no learnable params)

**Sub-module A3: Feature MLP**
```
Input: Concatenate [Log(delta_R) → [B,3], delta_v → [B,3], delta_p → [B,3], 
       diag(Sigma) → [B,9]] = [B, 18]

Linear(18, 64) → ReLU → Linear(64, 128) → f_imu [B, 128]
```

**Parameters**: ~10K

**Total IMU Encoder**: ~60K parameters, <0.5ms inference

### B. Visual Backbone (`backbone.py`)

**Architecture**: DDRNet-23-slim, slimmed for binary segmentation.

Use the open-source DDRNet implementation. Key modifications:
- Load pretrained ImageNet weights for the backbone
- Reduce final segmentation head channels: 64→32, 128→64 (since binary, not 19-class)
- Remove the 19-class classification head, replace with 1-channel output
- Input: RGB images [B, 3, H, W] where H=480, W=640

**The backbone is SHARED** between frame t-1 and frame t (same weights, called twice).

**Output**: Multi-scale features at stages 2, 3, 4:
- Stage 2: [B, C2, H/8, W/8]
- Stage 3: [B, C3, H/16, W/16]  
- Stage 4: [B, C4, H/16, W/16]

The primary feature map for correlation is the Stage 3 output at 1/16 resolution.

**Parameters**: ~2.0M (slimmed)

### C. FiLM Conditioning (`film.py`)

```python
class FiLM(nn.Module):
    def __init__(self, imu_dim=128, feature_channels=C):
        self.gamma_proj = nn.Linear(imu_dim, feature_channels)
        self.beta_proj = nn.Linear(imu_dim, feature_channels)
        # Initialize gamma close to 1, beta close to 0
        nn.init.ones_(self.gamma_proj.weight.data * 0.01)  
        nn.init.zeros_(self.gamma_proj.bias.data)
        nn.init.zeros_(self.beta_proj.weight.data)
        nn.init.zeros_(self.beta_proj.bias.data)
    
    def forward(self, features, f_imu):
        # features: [B, C, H, W]
        # f_imu: [B, 128]
        gamma = self.gamma_proj(f_imu).unsqueeze(-1).unsqueeze(-1) + 1.0  # [B, C, 1, 1], centered at 1
        beta = self.beta_proj(f_imu).unsqueeze(-1).unsqueeze(-1)           # [B, C, 1, 1], centered at 0
        return gamma * features + beta
```

**Applied at stages 2, 3, 4** of the backbone. Total added parameters: ~50K.

### D. Temporal Motion Decoder (`temporal_decoder.py`)

**Step 1: Correlation Volume**
```
Input: F_prev [B, C, H/16, W/16], F_curr [B, C, H/16, W/16]

For each pixel (x,y) in F_curr, correlate with a local window in F_prev:
    CV[b, x, y, dx, dy] = dot(F_curr[b, :, x, y], F_prev[b, :, x+dx, y+dy])
    
Search radius: r = 4 (so 9×9 = 81 displacement channels)
Output: [B, 81, H/16, W/16]

Implementation: Use torch grid_sample or unfold for efficiency.
```

**Step 2: ConvGRU Refinement (2 iterations)**
```
Input to GRU: correlation volume [B, 81, H/16, W/16] 
              + context features from backbone [B, C_ctx, H/16, W/16]

ConvGRU hidden state: [B, 128, H/16, W/16]

Each iteration:
    motion_features = Conv2d(cat(corr_slice, context, hidden), 128)
    z = sigmoid(Conv2d(cat(motion_features, hidden), 128))   # update gate
    r = sigmoid(Conv2d(cat(motion_features, hidden), 128))   # reset gate
    h_candidate = tanh(Conv2d(cat(motion_features, r * hidden), 128))
    hidden = (1 - z) * hidden + z * h_candidate

After 2 iterations, decode hidden state:
```

**Step 3: Output Heads**
```
Dynamic Mask Head:
    Conv2d(128, 64, 3, padding=1) → ReLU → Conv2d(64, 1, 1) → Upsample(16×) → Sigmoid
    Output: [B, 1, H, W] — per-pixel dynamic probability

Flow Residual Head (auxiliary):
    Conv2d(128, 64, 3, padding=1) → ReLU → Conv2d(64, 2, 1) → Upsample(16×)  
    Output: [B, 2, H, W] — residual flow after ego-motion subtraction
```

**Parameters**: ~300K (correlation is parameter-free, GRU + heads ~300K)

### E. Full Model Forward Pass (`dynamask.py`)

```python
class DynaMaskVIO(nn.Module):
    def forward(self, img_prev, img_curr, imu_window):
        """
        img_prev: [B, 3, 480, 640]
        img_curr: [B, 3, 480, 640]
        imu_window: [B, N, 7] (variable N, padded with mask)
        
        Returns dict with:
            'dynamic_mask': [B, 1, 480, 640]  — values in [0, 1]
            'flow_residual': [B, 2, 480, 640]  — auxiliary
            'delta_bg': [B, N, 3]  — gyro bias corrections
            'delta_ba': [B, N, 3]  — accel bias corrections  
            'sigma2_g': [B, N, 3]  — gyro noise variance
            'sigma2_a': [B, N, 3]  — accel noise variance
            'delta_R': [B, 3, 3]   — preintegrated rotation
            'delta_v': [B, 3]      — preintegrated velocity
            'delta_p': [B, 3]      — preintegrated position
            'Sigma_preint': [B, 9, 9] — preintegration covariance
        """
        # 1. IMU encoding
        imu_out = self.imu_encoder(imu_window)
        f_imu = imu_out['f_imu']  # [B, 128]
        
        # 2. Visual feature extraction (shared backbone, called twice)
        feats_prev = self.backbone(img_prev)  # dict of multi-scale features
        feats_curr = self.backbone(img_curr)
        
        # 3. FiLM conditioning
        for stage in ['stage2', 'stage3', 'stage4']:
            feats_prev[stage] = self.film_layers[stage](feats_prev[stage], f_imu)
            feats_curr[stage] = self.film_layers[stage](feats_curr[stage], f_imu)
        
        # 4. Temporal decoding
        mask, flow_res = self.temporal_decoder(feats_prev, feats_curr)
        
        # 5. Combine outputs
        return {
            'dynamic_mask': mask,
            'flow_residual': flow_res,
            **imu_out  # includes all IMU head outputs
        }
```

**Total model parameters**: ~2.5M
**Total inference time**: ~6-8ms at 640×480 on RTX 3060 (FP16)

---

## LOSS FUNCTIONS — EXACT FORMULAS

### Loss 1: Focal Loss for Dynamic Mask
```python
def focal_loss(pred, target, gamma=2.0, alpha=0.25):
    """
    pred: [B, 1, H, W] — sigmoid probabilities
    target: [B, 1, H, W] — binary ground truth (1=dynamic, 0=static)
    """
    bce = F.binary_cross_entropy(pred, target, reduction='none')
    pt = torch.where(target == 1, pred, 1 - pred)
    alpha_t = torch.where(target == 1, alpha, 1 - alpha)
    loss = alpha_t * (1 - pt) ** gamma * bce
    return loss.mean()
```
**Weight**: λ₁ = 1.0

### Loss 2: IMU Integration Error Loss
```python
def imu_integration_loss(pred_R, pred_v, pred_p, gt_R, gt_v, gt_p):
    """
    Supervises the preintegrated motion against ground truth.
    """
    # Rotation error: geodesic distance on SO(3)
    R_err = pred_R.transpose(-1, -2) @ gt_R
    angle_err = rotation_angle(R_err)  # ||Log(R_err)||
    loss_R = angle_err.pow(2).mean()
    
    # Velocity and position: L2
    loss_v = F.mse_loss(pred_v, gt_v)
    loss_p = F.mse_loss(pred_p, gt_p)
    
    return 5.0 * loss_R + 1.0 * loss_v + 1.0 * loss_p  # rotation weighted higher
```
**Weight**: λ₂ = 1.0

### Loss 3: Covariance Calibration Loss (NLL)
```python
def covariance_nll_loss(pred_R, pred_v, pred_p, gt_R, gt_v, gt_p, Sigma):
    """
    Negative log-likelihood: teaches the network to predict calibrated uncertainty.
    """
    # Compute 9-d error vector
    err_R = Log_map(gt_R.transpose(-1,-2) @ pred_R)  # [B, 3]
    err_v = gt_v - pred_v  # [B, 3]
    err_p = gt_p - pred_p  # [B, 3]
    err = torch.cat([err_R, err_v, err_p], dim=-1)  # [B, 9]
    
    # NLL = 0.5 * (e^T Σ^{-1} e + log|Σ|)
    Sigma_inv = torch.linalg.inv(Sigma + 1e-6 * torch.eye(9).to(Sigma.device))
    mahal = torch.einsum('bi,bij,bj->b', err, Sigma_inv, err)
    logdet = torch.logdet(Sigma + 1e-6 * torch.eye(9).to(Sigma.device))
    
    loss = 0.5 * (mahal + logdet).mean()
    return loss
```
**Weight**: λ₃ = 0.5

### Loss 4: Flow Residual Loss (only on datasets with GT optical flow)
```python
def flow_residual_loss(pred_flow_res, gt_flow, gt_mask):
    """
    On static pixels: flow residual should be 0 (all motion is ego-motion)
    On dynamic pixels: flow residual should match (gt_flow - ego_flow)
    
    For V1: simply supervise that static pixels have ~0 residual flow.
    """
    static_mask = (1 - gt_mask)  # [B, 1, H, W]
    loss = (pred_flow_res.pow(2) * static_mask).sum() / (static_mask.sum() + 1e-6)
    return loss
```
**Weight**: λ₄ = 0.3 (only when GT flow is available, else 0)

### Total Loss
```python
L_total = 1.0 * L_focal + 1.0 * L_imu_integration + 0.5 * L_covariance_nll + 0.3 * L_flow
```

---

## TRAINING PROCEDURE — EXACT STEPS

### Optimizer & Schedule
- **Optimizer**: AdamW, lr=1e-4, weight_decay=1e-4, betas=(0.9, 0.999)
- **Schedule**: CosineAnnealingWarmRestarts, T_0=30 epochs, T_mult=2, eta_min=1e-6
- **Warmup**: Linear warmup for first 5 epochs from lr=1e-6 to lr=1e-4
- **Batch size**: 8 (adjust for GPU memory)
- **Image size**: 480×640 (native VIODE resolution, no resizing)
- **Mixed precision**: FP16 via torch.cuda.amp

### Phase 1: IMU Head Pretraining (Epochs 1-20)

**Data**: EuRoC only
**Frozen**: Entire visual backbone + FiLM + temporal decoder
**Training**: Only IMU encoder (noise corrector + feature MLP)
**Losses**: L_imu_integration + L_covariance_nll (no mask loss, no flow loss)
**Purpose**: Give the IMU head a strong initialization before coupling with vision

**Ground truth for EuRoC**: 
- GT trajectory → compute GT preintegration between consecutive image timestamps
- GT biases from `state_groundtruth_estimate0/data.csv`

### Phase 2: Joint Training on Synthetic Data (Epochs 21-60)

**Data**: VIODE (all environments, all dynamic levels) + TartanAir (selected environments)
**Unfrozen**: Everything (backbone initialized from ImageNet pretrained DDRNet-23-slim)
**Losses**: All four losses (L_focal + L_imu + L_cov + L_flow where available)
**IMU head**: Learning rate 0.1× of backbone (to preserve Phase 1 training)

### Phase 3: Fine-tuning (Epochs 61-80)

**Data**: VIODE only (cleaner labels than TartanAir for dynamic masks)
**Unfrozen**: Everything
**Learning rate**: Reduced to 1e-5
**Losses**: L_focal + L_imu + L_cov
**Purpose**: Polish mask quality on the primary evaluation dataset

### Early Stopping
Monitor validation dynamic mask IoU on VIODE parking_lot (low + medium).
Stop if no improvement for 10 epochs. Save best checkpoint by val IoU.

---

## DATA AUGMENTATIONS — EXACT SPECIFICATIONS

### Visual Augmentations (applied to BOTH frames identically)
```python
# Color jitter (applied with p=0.8)
brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1

# Random horizontal flip (p=0.5)
# IMPORTANT: when flipping, also negate IMU gy and ax

# Random crop and resize (p=0.3)
# Scale factor: uniform(0.8, 1.0), then resize back to 480×640
# Update camera intrinsics in loss computation accordingly

# Gaussian noise on images (p=0.3)
# sigma ~ uniform(0, 0.02)
```

### IMU Augmentations
```python
# Additive Gaussian noise (always applied)
sigma_accel = uniform(0.01, 0.1)  # m/s^2
sigma_gyro = uniform(0.001, 0.01)  # rad/s

# Bias drift injection (p=0.5)  
# Add slow-varying bias: b(t) = b0 + alpha*t
# b0 ~ N(0, 0.01), alpha ~ N(0, 0.001)

# Temporal offset injection (p=0.5)
# Shift all IMU timestamps by dt ~ uniform(-10ms, +10ms)
# This simulates imperfect synchronization

# Sample dropout (p=0.3)
# Randomly drop 10% of IMU samples in the window
```

---

## EVALUATION METRICS — EXACTLY WHAT TO REPORT

### Mask Quality (on VIODE test sequences)
1. **IoU** (Intersection over Union) of dynamic mask vs GT
2. **Precision** (what fraction of predicted-dynamic pixels are truly dynamic)
3. **Recall** (what fraction of truly dynamic pixels are detected)
4. **F1 score**

Report separately for each dynamic level: none, low, medium, high.
Target: IoU > 0.65 on high-dynamic, F1 > 0.70.

### IMU Head Quality (on EuRoC test sequences)
1. **RTE** (Relative Translation Error) of 1-second IMU preintegration: compare with vs without learned corrections
2. **ROE** (Relative Orientation Error) of 1-second preintegration
3. Target: >25% reduction in RTE over raw IMU integration

### VIO Trajectory Accuracy (run OpenVINS with/without our mask)
1. **ATE RMSE** (Absolute Trajectory Error, root mean square)
2. **RPE** (Relative Pose Error at 1s intervals)
3. Use the `evo` Python package for computation
4. Run on VIODE test sequences (all 4 dynamic levels)
5. Run on EuRoC test sequences (verify no degradation on static scenes)

Comparison conditions:
- A: OpenVINS vanilla (no masking, constant IMU noise)
- B: OpenVINS + our binary mask (threshold dynamic_mask > 0.5)
- C: OpenVINS + our binary mask + learned IMU noise parameters
- D: OpenVINS + soft masking (inflate feature noise by 1/static_confidence) + learned IMU

---

## KEY IMPLEMENTATION DETAILS THAT MUST NOT BE WRONG

### 1. IMU Preintegration Must Use Variable dt
Phone and real IMUs have non-uniform sampling intervals. The preintegrator must compute `dt_i = timestamp[i+1] - timestamp[i]` per sample, NOT assume constant dt.

### 2. SO(3) Operations Must Be Numerically Stable
The Rodrigues formula for `Exp(ω)` has a singularity at `||ω|| = 0`. Use the small-angle approximation when `||ω|| < 1e-7`:
```python
def so3_exp(omega):
    theta = torch.norm(omega, dim=-1, keepdim=True)
    small = (theta < 1e-7).squeeze(-1)
    
    # Normal case: Rodrigues
    K = skew_symmetric(omega / (theta + 1e-10))
    R_normal = torch.eye(3) + torch.sin(theta)[...,None] * K + (1 - torch.cos(theta))[...,None] * K @ K
    
    # Small angle: first-order approximation
    R_small = torch.eye(3) + skew_symmetric(omega)
    
    R = torch.where(small[...,None,None], R_small, R_normal)
    return R
```

### 3. Correlation Volume Must Handle Boundary Pixels
When computing correlations at search radius r=4, pixels near image boundaries need zero-padding. Use `F.pad` on the feature maps before computing correlations.

### 4. IMU Window Padding for Batching
Different frame pairs may have different numbers of IMU samples between them. Pad to the maximum N in the batch and use a boolean mask to ignore padded samples in the noise corrector and preintegrator.

### 5. FiLM Initialization
Initialize FiLM so that at the start of training, it's approximately an identity transform (gamma≈1, beta≈0). This ensures the backbone works normally before the IMU head is trained.

### 6. Backbone Pretrained Weights
Use DDRNet-23-slim pretrained on ImageNet (available from the original DDRNet repo). The segmentation head is randomly initialized (since we're changing it to binary output).

### 7. Gradient Clipping
Clip gradients to max_norm=10.0 during training. The preintegration Jacobians can produce large gradients, especially early in training.

---

## WHAT SUCCESS LOOKS LIKE

**Minimum Viable Result** (V1 is a success if):
1. Dynamic mask IoU > 0.50 on VIODE high-dynamic test sequences
2. OpenVINS ATE on VIODE high-dynamic improves by >20% with masking vs without
3. OpenVINS ATE on EuRoC does NOT degrade (within 5% of vanilla) with masking
4. IMU preintegration RTE improves by >15% with learned corrections on EuRoC

**Stretch Goals** (V2):
1. Dynamic mask IoU > 0.70 on VIODE high-dynamic
2. Real-time inference demo (<10ms per frame)
3. Phone data collection and testing
4. Online temporal offset estimation
5. Soft masking integration with OpenVINS

---

## DEPENDENCIES

```
# requirements.txt
torch>=2.0.0
torchvision>=0.15.0
pytorch-lightning>=2.0.0
numpy>=1.24.0
scipy>=1.10.0
opencv-python>=4.7.0
pyyaml>=6.0
evo>=1.20.0           # trajectory evaluation
rosbags>=0.9.0        # reading VIODE ROS bags without ROS installed
matplotlib>=3.7.0
tqdm>=4.65.0
tensorboard>=2.13.0
einops>=0.6.0         # tensor reshaping utilities
timm>=0.9.0           # for pretrained backbone weights if needed
```

---

## CONFIG FILE (`configs/default.yaml`)

```yaml
# Model
model:
  backbone: "ddrnet_23_slim"
  backbone_pretrained: true
  imu_feature_dim: 128
  film_stages: ["stage2", "stage3", "stage4"]
  corr_search_radius: 4
  gru_iterations: 2
  gru_hidden_dim: 128

# IMU Encoder
imu:
  noise_corrector_hidden: 64
  noise_corrector_dilations: [1, 2, 4]
  initial_variance: 1.0e-4

# Training
training:
  batch_size: 8
  num_workers: 8
  epochs: 80
  lr: 1.0e-4
  weight_decay: 1.0e-4
  warmup_epochs: 5
  grad_clip_norm: 10.0
  mixed_precision: true
  
  # Phase schedule
  phase1_epochs: 20      # IMU-only on EuRoC
  phase2_epochs: 40      # Joint on VIODE + TartanAir (epochs 21-60)
  phase3_epochs: 20      # Fine-tune on VIODE (epochs 61-80)
  imu_lr_multiplier: 0.1 # IMU head lr = backbone_lr * this, in Phase 2+

# Loss weights
loss:
  focal_gamma: 2.0
  focal_alpha: 0.25
  lambda_mask: 1.0
  lambda_imu: 1.0
  lambda_cov: 0.5
  lambda_flow: 0.3

# Data
data:
  image_height: 480
  image_width: 640
  imu_max_window_size: 15  # max IMU samples per frame pair (for padding)

# Augmentation
augmentation:
  color_jitter_p: 0.8
  horizontal_flip_p: 0.5
  random_crop_p: 0.3
  random_crop_scale: [0.8, 1.0]
  imu_noise_accel: [0.01, 0.1]   # sigma range
  imu_noise_gyro: [0.001, 0.01]  # sigma range
  temporal_offset_p: 0.5
  temporal_offset_ms: [-10, 10]
  imu_dropout_p: 0.3
  imu_dropout_rate: 0.1

# Evaluation
eval:
  mask_threshold: 0.5
  min_features_fallback: 20  # if fewer features after masking, relax threshold
```
