# DynaMask-VIO: Intrinsic-Free, Sync-Free IMU-Guided Dynamic Masking for Visual-Inertial Odometry

## Project Philosophy & Core Constraints

This system must work with **just a phone camera and its IMU** — no camera intrinsics, no hardware synchronization, only the camera-to-IMU extrinsic (a rigid body transform `T_CI` that can be measured or roughly estimated). This fundamentally changes the entire architecture compared to classical VIO front-ends. The system has two responsibilities:

1. **Detect and mask dynamic pixels** (rigid + as-rigid-as-possible deformable objects) using learned visual-inertial features, and feed cleaned images to any VIO backend.
2. **Produce learned IMU quantities** (bias corrections, per-sample noise covariance, preintegration corrections) that the VIO backend can directly consume for improved state estimation.

---

## 1. Why "No Intrinsics" Changes Everything

Classical dynamic detection via geometric verification (epipolar check, reprojection error thresholding) requires camera intrinsics `K` to:
- Compute the Essential matrix `E = K^T F K` from the Fundamental matrix
- Reproject 3D landmarks into pixel space: `z = π(K, T_wc, X_w)`
- Warp frame t-1 to frame t via known camera model for photometric consistency

**Without `K`, none of these geometric operations are possible in their classical form.** This means:

- No geometric epipolar verification for dynamic point rejection
- No classical reprojection-based photometric loss during training
- No calibrated optical flow → ego-motion decomposition

**The solution**: operate entirely in **learned feature space** and **pixel-displacement space** (optical flow), where the fundamental matrix `F` (which doesn't need intrinsics) replaces the essential matrix, and learned representations replace calibrated geometric operations. The network must implicitly learn whatever camera model properties it needs from data.

### What we CAN still do without intrinsics:
- **Fundamental matrix estimation** from point correspondences (8-point algorithm, no `K` needed)
- **Optical flow computation** (purely pixel-space, no `K` needed)
- **Flow field decomposition**: Given IMU-predicted rotation `ΔR` and the extrinsic `T_CI`, we can predict the *rotational component* of optical flow without `K` — and learn a correction for the translational component
- **Relative feature motion analysis**: features that move inconsistently with the dominant flow pattern are dynamic — this is intrinsic-free

---

## 2. Model Architecture: DynaMask-VIO

### 2.1 System Overview

```
Phone Camera (20-30 Hz)                Phone IMU (100-200 Hz)
        │                                       │
        ▼                                       ▼
┌──────────────────┐                  ┌────────────────────────┐
│  Frame Buffer    │                  │  IMU Ring Buffer       │
│  (stores t-1, t) │                  │  (2-sec circular buf)  │
└────────┬─────────┘                  └──────────┬─────────────┘
         │                                       │
         │              ┌────────────────────┐   │
         │              │  Temporal Aligner   │◄──┘
         │              │  (Learned t_d est.) │
         │              └────────┬───────────┘
         │                       │
         │            ┌──────────▼───────────┐
         │            │   IMU Encoder Head   │
         │            │  ┌─────────────────┐ │
         │            │  │ Noise Corrector │ │──► δb_g, δb_a (bias corrections)
         │            │  │ (per-sample)     │ │──► Σ_g, Σ_a (per-sample covariance)
         │            │  └────────┬────────┘ │
         │            │  ┌────────▼────────┐ │
         │            │  │ Differentiable  │ │──► ΔR, Δv, Δp (preintegrated)
         │            │  │ Preintegrator   │ │──► Σ_preint (9×9 covariance)
         │            │  └────────┬────────┘ │
         │            │  ┌────────▼────────┐ │
         │            │  │ Feature MLP     │ │──► f_imu (128-d ego-motion feature)
         │            │  └─────────────────┘ │
         │            └──────────┬───────────┘
         │                       │
         ▼                       ▼
┌────────────────────────────────────────────┐
│           Visual Backbone (shared)          │
│           DDRNet-23-slim (slimmed)          │
│                                            │
│   Frame t-1 ──► Features F'(t-1)           │
│                    ▲ FiLM(f_imu)           │
│   Frame t   ──► Features F'(t)             │
│                    ▲ FiLM(f_imu)           │
└──────────────────────┬─────────────────────┘
                       │
         ┌─────────────▼─────────────┐
         │  Temporal Motion Decoder   │
         │                           │
         │  1. Correlation Volume    │
         │     (intrinsic-free,      │
         │      pixel-displacement)  │
         │                           │
         │  2. ConvGRU refinement    │
         │     (2-3 iterations)      │
         │                           │
         │  3. Multi-head output:    │
         │     ├─ Dynamic Mask [H×W] │
         │     ├─ Flow Residual [H×W×2]
         │     └─ Static Confidence  │
         │        per-feature [N×1]  │
         └───────────────────────────┘
                       │
         ┌─────────────▼─────────────┐
         │     OUTPUT TO VIO BACKEND  │
         │                           │
         │  • Masked image           │
         │  • Static confidence map  │
         │  • Bias corrections (δb)  │
         │  • Per-sample Σ_g, Σ_a    │
         │  • Preintegrated ΔR,Δv,Δp │
         │  • Preintegration Σ_preint│
         │  • Estimated time offset  │
         └───────────────────────────┘
```

### 2.2 Temporal Aligner: Handling Zero Synchronization

This is the most critical engineering module. When you just hook up a phone, the camera and IMU timestamps come from the same system clock (on Android/iOS), but with **different and unknown delays**: the camera has shutter + readout + USB/buffer delay (typically 10-50ms), while the IMU has minimal delay (<1ms). The effective time offset `t_d` is unknown and may drift.

**Architecture: A lightweight learned temporal offset estimator**

The core idea comes from VINS-Mono's online temporal calibration (Qin et al., 2018) and TON-VIO (2024), but adapted into the learned pipeline:

**Stage 1 — Coarse alignment via rotation correlation:**
- From IMU gyroscope data, integrate angular velocity over a sliding window to get rotation rate profile `ω(t)`
- From consecutive images, compute a global rotation estimate via feature tracking (intrinsic-free: use the rotation component of the fundamental matrix, or simply compute mean optical flow divergence which correlates with rotation magnitude)
- Cross-correlate these two signals with sub-sample interpolation to find the lag that maximizes correlation
- This gives a coarse `t_d` estimate (±5ms accuracy typically)
- **This runs once at startup and periodically (every 5s) as a sanity check**

**Stage 2 — Learned residual offset correction:**
- A small 1D temporal CNN (3 layers, dilated convolutions, ~20K params) takes:
  - Recent gyroscope window (200 samples = 1 second at 200Hz)
  - Recent image-derived rotation estimates (30 samples = 1 second at 30Hz, upsampled)
  - Current coarse `t_d` estimate
- Outputs: residual correction `Δt_d` (scalar, typically <5ms) and confidence `σ_td`
- The corrected offset `t_d + Δt_d` is used to slice the IMU buffer for the current image pair

**Stage 3 — Training-time robustness via temporal augmentation:**
- During training, inject random time offsets `t_d ~ Uniform(-30ms, +30ms)` between IMU and camera streams
- This forces the network to be robust to temporal misalignment even if Stage 1-2 aren't perfect
- Additionally, randomly drop 10-30% of IMU samples to simulate real phone jitter

**The IMU buffer slicing procedure at inference:**

```python
def get_imu_window(imu_buffer, t_img_prev, t_img_curr, t_d):
    """
    Given image timestamps and estimated offset, extract IMU window.
    t_d: estimated camera-to-IMU time offset (camera_time = imu_time + t_d)
    """
    # Convert image times to IMU clock
    t_start = t_img_prev - t_d
    t_end = t_img_curr - t_d

    # Binary search for IMU samples in [t_start, t_end]
    imu_slice = imu_buffer.query(t_start, t_end)

    # Interpolate boundary samples (SLERP for gyro, LERP for accel)
    imu_slice = interpolate_boundaries(imu_slice, t_start, t_end)

    return imu_slice  # Shape: [N, 7] (timestamp, ax, ay, az, gx, gy, gz)
```

### 2.3 IMU Encoder Head: The Learned IMU Module

This is the heart of what makes this system useful beyond just masking. The IMU head produces quantities that **any VIO backend can directly consume**. It follows the AirIMU philosophy but extends it.

**Input**: Raw IMU window `[N×6]` (3-axis accel + 3-axis gyro) between two image frames, plus timestamps `[N×1]`.

**Sub-module A: Per-Sample Noise Corrector**

A lightweight network that processes each IMU sample (with temporal context) and outputs:

```
Input:  [ω_x, ω_y, ω_z, a_x, a_y, a_z] × N samples (with timestamps)

Network: Dilated 1D-CNN (inspired by Calib-Net + AirIMU)
  - Layer 1: Conv1d(6, 64, kernel=3, dilation=1) + ReLU
  - Layer 2: Conv1d(64, 64, kernel=3, dilation=2) + ReLU
  - Layer 3: Conv1d(64, 64, kernel=3, dilation=4) + ReLU
  - Head A: Conv1d(64, 6, kernel=1) → δb = [δb_gx, δb_gy, δb_gz, δb_ax, δb_ay, δb_az]
  - Head B: Conv1d(64, 6, kernel=1) → Softplus → σ² = [σ²_gx, ..., σ²_az]

Output per sample i:
  - Bias correction: δb_i ∈ R^6  (additive correction to raw measurement)
  - Noise variance: σ²_i ∈ R^6_+ (per-axis, per-sample measurement variance)

Corrected measurement:
  ω̂_i = ω_raw_i - δb_g_i
  â_i = a_raw_i - δb_a_i
```

The **dilated convolutions** give each sample a receptive field of ~15 samples (~75ms at 200Hz), allowing the network to learn temporal patterns in the bias (e.g., bias changes during high-vibration periods, stationary periods, temperature effects as captured in the data).

The **per-sample variance** `σ²_i` is crucial: it tells the VIO backend "how much to trust this particular IMU reading." During high-vibration or rapid rotation, variances increase; during stationary periods, they decrease. This replaces the **constant noise parameters** (`σ_g`, `σ_a`, `σ_bg`, `σ_ba`) that VIO systems like VINS-Mono and OpenVINS typically require manual tuning for.

**Total parameters**: ~50K (very lightweight)

**Sub-module B: Differentiable Preintegrator**

Takes the corrected measurements `(ω̂_i, â_i)` and their variances, and performs physics-based preintegration following Forster et al. (TRO 2017):

```
Given corrected measurements ω̂_i, â_i and variances σ²_i for i=1...N:

Initialize: ΔR = I, Δv = 0, Δp = 0, Σ = 0 (9×9)

For each IMU sample i:
    dt_i = t_{i+1} - t_i  (variable dt handles irregular sampling!)

    # Rotation update (SO(3) exponential map)
    ΔR ← ΔR · Exp(ω̂_i · dt_i)

    # Velocity update
    Δv ← Δv + ΔR · â_i · dt_i

    # Position update
    Δp ← Δp + Δv · dt_i + 0.5 · ΔR · â_i · dt_i²

    # Covariance propagation (using LEARNED per-sample noise)
    Q_i = diag(σ²_g_i · dt_i, σ²_a_i · dt_i)  # 6×6 process noise
    A_i = [...] # State transition Jacobian (standard preintegration)
    B_i = [...] # Noise Jacobian
    Σ ← A_i · Σ · A_i^T + B_i · Q_i · B_i^T

Output:
  - ΔR ∈ SO(3), Δv ∈ R³, Δp ∈ R³  (preintegrated motion)
  - Σ ∈ R^{9×9}  (preintegration covariance with LEARNED noise)
```

**This is fully differentiable** — gradients flow back through the preintegration to train the noise corrector. The key insight from AirIMU is that **jointly training noise correction and covariance estimation produces better results than either alone** — the covariance loss acts as a regularizer preventing the noise corrector from learning spurious corrections.

**Variable `dt_i`** naturally handles irregular IMU sampling from phones (where the OS scheduler can cause jitter of ±1-5ms between samples).

**Sub-module C: Feature Projection MLP**

Compresses the preintegration outputs into the 128-d feature vector for FiLM conditioning:

```
Input: [ΔR (as 3-d axis-angle), Δv, Δp, diag(Σ)] = 3+3+3+9 = 18-d
MLP: Linear(18, 64) → ReLU → Linear(64, 128) → f_imu
```

The inclusion of `diag(Σ)` in the feature is important — it tells the visual backbone "how confident the IMU ego-motion estimate is." When IMU confidence is low (high Σ), the visual network should rely more on appearance-based dynamic detection; when IMU confidence is high, the network can leverage strong ego-motion priors.

### 2.4 Visual Backbone with FiLM Conditioning

**DDRNet-23-slim** (slimmed for binary segmentation):
- Original: 5.7M params, 19-class Cityscapes
- Slimmed: Reduce decoder channels by 3× (64→24, 128→48) → ~2.0M params
- Binary task (dynamic/static) needs far less capacity than 19-class segmentation

**FiLM injection at 3 stages** (same as before):
```
γ_l, β_l = Linear(f_imu, C_l), Linear(f_imu, C_l)
F'_l = γ_l ⊙ F_l + β_l
```

**Why FiLM works without intrinsics**: The FiLM conditioning doesn't encode geometric camera parameters — it encodes *ego-motion magnitude and uncertainty*. The visual backbone learns to associate "this pattern of feature changes across frames, given this ego-motion, indicates dynamic pixels" — all in learned feature space without explicit geometry.

### 2.5 Temporal Motion Decoder (Intrinsic-Free)

**Correlation cost volume in pixel space** — no camera model needed:

```
For FiLM-conditioned features F'(t-1) and F'(t) at 1/8 resolution:

CV(x, y, dx, dy) = <F'(t-1)[x,y], F'(t)[x+dx, y+dy]>

Search radius r = 4 pixels at 1/8 scale = 32 pixels at full resolution
This covers object displacements up to ~32px/frame (adequate for 30fps)
```

The correlation volume captures **all pixel displacements**, not just ego-motion-consistent ones. The ConvGRU then learns to classify: "given the IMU-predicted ego-motion (via FiLM), which displacement patterns are consistent with ego-motion (static) and which are not (dynamic)?"

**ConvGRU refinement** (2-3 iterations, RAFT-lite style):
- Input: correlation volume slice + context features + previous hidden state
- Output: refined dynamic probability map + flow residual
- Each iteration refines the mask, focusing on uncertain boundary regions

**Output heads**:
1. **Dynamic mask** `M ∈ [0,1]^{H×W}`: per-pixel probability of being dynamic (sigmoid)
2. **Flow residual** `Δf ∈ R^{H×W×2}`: optical flow after ego-motion subtraction (the "independent motion" flow)
3. **Per-feature static confidence** `c ∈ [0,1]^N`: for N tracked features, confidence that each is static (derived from mask by sampling at feature locations with soft bilinear interpolation)

### 2.6 Complete Parameter Budget

| Module | Parameters | Inference Time |
|--------|-----------|----------------|
| Temporal Aligner (1D CNN) | ~20K | ~0.1ms |
| IMU Noise Corrector (dilated CNN) | ~50K | ~0.2ms |
| Preintegrator (no learnable params) | 0 | ~0.1ms |
| Feature MLP | ~10K | <0.1ms |
| DDRNet-23-slim (slimmed) | ~2.0M | ~3ms |
| FiLM layers (×3 stages) | ~50K | <0.1ms |
| Correlation volume + ConvGRU | ~300K | ~2ms |
| Output heads | ~100K | ~0.3ms |
| **Total** | **~2.5M** | **~6ms** |

At 640×480 on RTX 3060 with TensorRT FP16. Comfortably within 33ms frame budget.

---

## 3. What the IMU Head Gives to the VIO Backend

This is where the system becomes more than just a masking network. The IMU head outputs are **directly pluggable** into standard VIO backends:

### 3.1 For Filter-Based Backends (MSCKF / OpenVINS)

**Standard MSCKF expects** (from its config file):
- `gyroscope_noise_density` (σ_g): constant, rad/s/√Hz
- `accelerometer_noise_density` (σ_a): constant, m/s²/√Hz
- `gyroscope_random_walk` (σ_bg): constant, rad/s²/√Hz
- `accelerometer_random_walk` (σ_ba): constant, m/s³/√Hz

**Our system provides instead**:
- `σ²_g_i, σ²_a_i`: **per-sample, time-varying** noise variance
- `δb_g_i, δb_a_i`: **per-sample bias corrections** (the filter no longer needs to estimate bias as part of its state!)

**Integration with OpenVINS:**

```cpp
// In the IMU propagation step, replace constant noise with learned noise:
// BEFORE (standard OpenVINS):
Eigen::Matrix<double,12,12> Q_d = Eigen::Matrix<double,12,12>::Zero();
Q_d.block(0,0,3,3) = sigma_g * sigma_g * dt * Eigen::Matrix3d::Identity();
Q_d.block(3,3,3,3) = sigma_bg * sigma_bg * dt * Eigen::Matrix3d::Identity();
Q_d.block(6,6,3,3) = sigma_a * sigma_a * dt * Eigen::Matrix3d::Identity();
Q_d.block(9,9,3,3) = sigma_ba * sigma_ba * dt * Eigen::Matrix3d::Identity();

// AFTER (with learned noise):
// For each IMU sample i in the propagation window:
Q_d.block(0,0,3,3) = learned_sigma_g_i.asDiagonal() * dt;  // per-axis, per-sample
Q_d.block(6,6,3,3) = learned_sigma_a_i.asDiagonal() * dt;  // per-axis, per-sample
// bias random walk terms can be kept constant or also learned

// For bias: instead of estimating b_g, b_a in the EKF state,
// subtract learned corrections from raw measurements BEFORE propagation:
w_corrected = w_raw - learned_delta_bg_i;
a_corrected = a_raw - learned_delta_ba_i;
```

**The massive advantage**: By predicting bias *outside* the filter state, the filter can use an **Invariant EKF** formulation (as shown by the "Learned IMU Bias for Invariant VIO" paper, 2025) where the state lives on a Lie group and covariance evolution is independent of state estimates — dramatically improving convergence and robustness.

Alternatively, in a more conservative integration: keep bias in the EKF state but use the learned bias as a strong **unary prior factor** that regularizes the bias estimate, preventing it from wandering into physically unrealistic values.

### 3.2 For Optimization-Based Backends (VINS-Mono, Ceres-based)

**IMU factor in the factor graph:**

The standard IMU preintegration factor in VINS-Mono computes residuals:
```
r_IMU = [r_ΔR, r_Δv, r_Δp, r_ba, r_bg]^T

Where:
r_ΔR = Log(ΔR_learned^T · R_i^T · R_j)
r_Δv = R_i^T(v_j - v_i - g·Δt) - Δv_learned
r_Δp = R_i^T(p_j - p_i - v_i·Δt - 0.5·g·Δt²) - Δp_learned

Information matrix: Σ_preint^{-1} (from learned covariance propagation)
```

The learned preintegration values `(ΔR, Δv, Δp)` and their covariance `Σ_preint` slot directly into the IMU factor — replacing the standard preintegration that uses constant noise parameters.

For bias: the learned bias corrections create **bias prior factors**:
```
r_bias = b_estimated - b_learned, weighted by Σ_bias_learned^{-1}
```

### 3.3 For Direct Methods (DSO)

DSO doesn't use IMU natively, but VI-DSO does. The mask output directly filters which pixels DSO selects for tracking (reject dynamic pixels), and the IMU preintegration provides motion priors between keyframes.

### 3.4 The Backend-Agnostic Interface

The cleanest integration path is a **ROS2 message interface**:

```
# Custom message: DynaMaskIMU.msg
std_msgs/Header header

# Visual outputs
sensor_msgs/Image masked_image          # Dynamic regions set to 0
sensor_msgs/Image dynamic_mask          # Float32 probability map [0,1]
sensor_msgs/Image static_confidence     # Float32 per-pixel static confidence

# Learned IMU outputs (for the window between this frame and previous)
float64 time_offset_estimate            # Estimated camera-IMU time offset t_d
float64 time_offset_confidence          # Confidence in t_d estimate

# Per-sample learned quantities (N = number of IMU samples in window)
float64[] timestamps                     # IMU sample timestamps (in IMU clock)
float64[] bias_correction_gyro          # [N×3] flattened: δb_gx, δb_gy, δb_gz per sample
float64[] bias_correction_accel         # [N×3] flattened: δb_ax, δb_ay, δb_az per sample
float64[] noise_variance_gyro           # [N×3] flattened: σ²_gx, σ²_gy, σ²_gz per sample
float64[] noise_variance_accel          # [N×3] flattened: σ²_ax, σ²_ay, σ²_az per sample

# Preintegrated quantities
geometry_msgs/Quaternion delta_rotation  # Preintegrated rotation ΔR as quaternion
geometry_msgs/Vector3 delta_velocity     # Preintegrated velocity Δv
geometry_msgs/Vector3 delta_position     # Preintegrated position Δp
float64[81] preint_covariance           # 9×9 preintegration covariance (row-major)
```

Any VIO backend can subscribe to this and use whichever fields it needs. A minimal integration just uses `masked_image`; a deep integration uses everything.

---

## 4. Training Strategy

### 4.1 Ground Truth Generation Without Intrinsics

Since we don't require intrinsics at inference, we can still USE intrinsics during training (they're available in all standard datasets). The training pipeline:

**For VIODE**: Ground-truth dynamic masks are provided directly. IMU ground truth (bias, noise-free measurements) available from simulation.

**For KITTI**: 
- Dynamic masks: SemanticKITTI labels → filter potentially-dynamic classes → verify actual motion via 3D bounding box tracklet displacement
- IMU GT bias: Recover from known GT trajectory: `b_g = ω_raw - R^T·ω_gt`, `b_a = a_raw - R^T·(a_gt + g)`

**For EuRoC**:
- Mostly static scenes (no dynamic objects) — use for IMU head training and static baseline verification
- IMU bias GT: provided in dataset (VICON-derived)

**For custom phone datasets**:
- Record with ARCore/ARKit which provides: camera poses, point clouds, IMU data, timestamps
- Generate pseudo-GT masks using offline heavyweight models (SAM2 + motion verification)
- IMU bias GT: derive from ARCore pose (which is itself VIO, so approximate but sufficient for training)

### 4.2 Loss Functions

**L_total = λ₁·L_mask + λ₂·L_imu_correction + λ₃·L_imu_covariance + λ₄·L_flow + λ₅·L_temporal + λ₆·L_consistency**

#### L_mask: Dynamic Mask Loss (λ₁ = 1.0)
Focal loss (γ=2.0, α=0.25) on per-pixel dynamic probability vs. ground truth mask. Handles 85-95% static pixel class imbalance.

#### L_imu_correction: IMU Bias/Noise Correction Loss (λ₂ = 1.0)
Following AirIMU, supervise through **integration error**:
```
L_imu = w_R · d_geodesic(ΔR_predicted, ΔR_gt) 
      + w_v · ||Δv_predicted - Δv_gt||² 
      + w_p · ||Δp_predicted - Δp_gt||²
```
Where `ΔR_gt, Δv_gt, Δp_gt` come from ground-truth poses. This implicitly trains the noise corrector — the network learns whatever bias/noise corrections minimize the integration drift.

#### L_imu_covariance: Learned Uncertainty Loss (λ₃ = 0.5)
**Negative log-likelihood** of the integration error under the predicted covariance:
```
e = [Log(ΔR_gt^T · ΔR_pred), Δv_gt - Δv_pred, Δp_gt - Δp_pred]  (9-d error)
L_cov = 0.5 · (e^T · Σ_preint^{-1} · e + log|Σ_preint|)
```
This trains the covariance to be **calibrated**: not too small (overconfident) and not too large (uninformative). The `log|Σ|` term prevents the trivial solution of infinite covariance.

#### L_flow: Flow Residual Loss (λ₄ = 0.3)
On datasets with optical flow GT (KITTI Scene Flow), or self-supervised via photometric consistency on static regions. The flow residual (motion after ego-motion subtraction) should be zero on static pixels and nonzero on dynamic pixels.

#### L_temporal: Temporal Offset Loss (λ₅ = 0.2)
When training with known synchronization (EuRoC, VIODE), after injecting artificial offsets:
```
L_temporal = ||t_d_predicted - t_d_injected||²
```

#### L_consistency: IMU-Visual Consistency (λ₆ = 0.3)
The ego-motion derived from the visual flow field on predicted-static regions should agree with IMU preintegration. This uses the **fundamental matrix** estimated from static-masked optical flow correspondences:
```
For static point pairs (p_i, p_j) where M(p_i) < 0.5:
  F = estimate_fundamental(static_points_prev, static_points_curr)  # No intrinsics needed!
  Rotation from F (using known T_CI extrinsic): R_visual
  L_consistency = d_geodesic(R_visual, ΔR_imu)
```

Note: The fundamental matrix gives rotation up to a projective ambiguity without `K`, but the **relative rotation** extracted from `F` via SVD is still meaningful for consistency checking — especially the rotation direction and relative magnitude.

### 4.3 Four-Phase Curriculum

**Phase 1 — IMU Head Pretraining (Epochs 1-20)**
- Train ONLY the IMU encoder head (noise corrector + preintegrator) on EuRoC + KITTI
- Use L_imu_correction + L_imu_covariance only
- This gives the IMU head a strong initialization before visual integration
- Validate: compare preintegration drift with and without corrections (target: >30% reduction)

**Phase 2 — Joint Mask + IMU on Synthetic (Epochs 20-50)**
- Unfreeze visual backbone, add VIODE + TartanAir
- Full L_mask + L_imu_correction + L_imu_covariance + L_flow
- Single-frame pair detection (t-1, t) with temporal reasoning
- Heavy temporal augmentation (±30ms offset injection, IMU sample dropout)

**Phase 3 — Real-World Adaptation (Epochs 50-80)**
- Fine-tune on KITTI with pseudo-GT masks
- Add L_consistency (IMU-visual agreement) and L_temporal
- Reduce temporal augmentation range to ±10ms (network should handle larger offsets from Phase 2)
- Add phone-like IMU noise augmentation (phone IMUs are noisier than KITTI's Oxford Technical Solutions RT3003)

**Phase 4 — Phone-Domain Transfer (Epochs 80-100)**
- Fine-tune on custom phone dataset
- Primary self-supervised losses (L_consistency + L_imu_correction) since phone data may lack GT masks
- L_mask with pseudo-labels from SAM2 teacher model (offline)
- Early stopping on held-out phone sequences

### 4.4 Data Augmentation Strategy

**Visual augmentations** (standard):
- Color jitter (brightness ±0.2, contrast ±0.2, saturation ±0.2)
- Random crop and resize (scale 0.8-1.2)
- **No horizontal flip** (or flip with IMU axis correction: negate gy, ax)

**IMU augmentations** (critical for phone generalization):
- Additive Gaussian noise: σ_accel = [0.01, 0.5] m/s² (phone range), σ_gyro = [0.001, 0.05] rad/s
- Random walk bias drift: inject slow-varying bias with τ ~ 100-500s time constant
- Sample rate jitter: randomly perturb IMU timestamps by ±2ms per sample
- Sample dropout: randomly remove 5-20% of IMU samples (simulating OS scheduling delays)
- Gravity direction perturbation: ±2° random rotation of gravity vector (simulates accelerometer misalignment)

**Temporal augmentations**:
- Random time offset injection: `t_d ~ Uniform(-30ms, +30ms)`
- Time-varying offset drift: `t_d(t) = t_d_0 + α·t` where `α ~ Uniform(-1ms/s, +1ms/s)`
- Camera frame drop: randomly skip 1-3 image frames (simulating phone thermal throttling)

**Dynamic object augmentations**:
- Copy-paste: cut dynamic objects from one frame pair and paste into static scenes
- Synthetic motion injection: apply random affine warps to static scene patches to create "fake dynamic" regions
- Scale variation: resize pasted dynamic objects (0.5×-2.0×) for scale robustness

---

## 5. Integration Depth: From Plug-and-Play to Deep Fusion

### Level 0: Plug-and-Play (Any Backend)
Just use the masked image. Set dynamic pixels to 0 or a sentinel value. The VIO backend extracts features only from non-masked regions.

```python
masked_img = img * (dynamic_mask < 0.5).float()
# Feed masked_img to ANY VIO system (VINS-Mono, OpenVINS, DSO, ORB-SLAM3)
```

### Level 1: Soft Feature Weighting
Instead of binary masking, weight each feature by its static confidence:

For **MSCKF/OpenVINS**: inflate measurement noise
```
R_feature_i = R_base / max(static_confidence_i, 0.01)
```

For **VINS-Mono/Ceres**: weight reprojection residuals
```
residual_weighted_i = sqrt(static_confidence_i) * residual_i
```

### Level 2: Learned IMU Integration
Replace the backend's constant IMU noise parameters with learned per-sample values. Replace (or augment) the backend's bias estimation with learned bias corrections.

This is the **recommended depth** for a research contribution — it shows that the learned IMU quantities actually improve VIO accuracy, which is a publishable result independent of the masking.

### Level 3: Full Deep Integration (Recommended for OpenVINS)
- Remove bias from the EKF state vector → Invariant EKF formulation
- Use learned per-sample noise in propagation
- Use learned bias as correction applied to raw IMU before propagation
- Soft-weight visual measurement residuals by static confidence
- Use the learned preintegration covariance in the EKF prediction step
- Online temporal offset as an estimated state (initialized from the learned estimate)

---

## 6. Implementation Plan

### 6.1 Technology Stack

| Component | Tool | Reason |
|-----------|------|--------|
| Training | PyTorch 2.x + Lightning | Standard, good data loading |
| IMU preintegration | Custom differentiable (PyTorch) | Must be differentiable for training |
| Model export | torch.onnx → TensorRT FP16 | 2-5× speedup over PyTorch |
| Deployment | ROS2 Humble/Jazzy | Standard robotics middleware |
| VIO Backend | OpenVINS (primary) | Modular, well-documented C++, supports online calibration, mask input |
| Evaluation | evo (Python) | ATE/RPE computation |
| Phone data collection | Android app (Camera2 API + SensorManager) | Raw sensor access with timestamps |

### 6.2 Phone Data Collection App

A minimal Android app that:
- Captures camera frames at 30fps via Camera2 API (records frame timestamps from `SENSOR_TIMESTAMP`)
- Logs IMU at 200Hz via SensorManager (records `event.timestamp`)
- Both timestamps use `SystemClock.elapsedRealtimeNanos()` on Android, so they share a clock basis — but with different pipeline delays
- Records camera-IMU extrinsic from ARCore's calibration API (`CameraConfig.getImageToCameraIntrinsics()` — wait, we said no intrinsics... but we CAN read the extrinsic from ARCore's `Frame.getAndroidSensorPose()` which gives IMU-to-camera transform)
- Saves as rosbag2 format for direct ROS2 playback

### 6.3 Sixteen-Week Timeline

**Weeks 1-3: Infrastructure**
- Set up ROS2 workspace, build OpenVINS from source, verify on EuRoC
- Implement differentiable preintegration module in PyTorch (test against GTSAM's preintegration for correctness)
- Build IMU ring buffer + temporal aligner in C++/Python
- Build Android data collection app

**Weeks 4-6: IMU Head Training**
- Implement IMU noise corrector (dilated 1D CNN)
- Train on EuRoC + KITTI with L_imu_correction + L_imu_covariance
- Validate preintegration accuracy improvement over raw IMU
- Target: >25% reduction in 1-second integration drift on EuRoC test sequences

**Weeks 7-10: Full Model Training**
- Implement DDRNet-23-slim backbone + FiLM layers + temporal decoder
- Phase 2 training on VIODE + TartanAir
- Phase 3 training on KITTI
- Export to TensorRT, benchmark inference latency
- Target: >70% dynamic mask IoU on VIODE high-dynamic, <8ms inference

**Weeks 11-13: VIO Integration + Ablation**
- Implement Level 0-3 integration with OpenVINS
- Run full benchmark suite:
  - VIODE (4 dynamic levels) × {no mask, binary mask, soft mask, full integration}
  - EuRoC (static baseline — verify no degradation)
  - KITTI dynamic sequences
- Run ablation studies (see below)

**Weeks 14-16: Phone Demo + Paper**
- Collect custom phone datasets (indoor office + outdoor street)
- Fine-tune model (Phase 4)
- Demo: walk around with phone, show real-time masking + VIO trajectory
- Write paper / technical report with all results

### 6.4 Ablation Studies

| ID | Ablation | What It Tests |
|----|----------|---------------|
| A1 | No masking (baseline VIO) | Establishes degradation in dynamic scenes |
| A2 | Binary mask only (Level 0) | Quantifies masking benefit alone |
| A3 | Soft mask (Level 1) vs binary | Tests soft vs hard decision boundary |
| A4 | With vs without IMU features (FiLM) | **Key ablation**: does IMU conditioning help masking? |
| A5 | With vs without temporal reasoning | Multi-frame vs single-frame detection |
| A6 | Learned IMU noise vs constant noise | Does learned per-sample variance help VIO? |
| A7 | Learned bias correction vs EKF bias | Does learned bias outperform online estimation? |
| A8 | With vs without temporal aligner | Impact of time offset estimation |
| A9 | Full integration (Level 3) vs Level 0 | End-to-end system benefit |
| A10 | Phone IMU vs research-grade IMU | Generalization across IMU quality |

**A4 is the most important ablation** — if IMU conditioning significantly improves mask quality, it validates the core thesis that inertial ego-motion should inform visual dynamic detection.

**A6 and A7 are independently publishable** — demonstrating that learned per-sample IMU noise and bias improve VIO accuracy, even without any masking.

---

## 7. Expected Challenges and Mitigation

### Challenge 1: Intrinsic-Free Ego-Motion Estimation During Training
**Problem**: Self-supervised losses like photometric consistency require image warping, which needs `K`.
**Mitigation**: Use intrinsics during training (available in all standard datasets) but ensure the model itself never receives `K` as input. The model's inference path is intrinsic-free; the training pipeline uses `K` only in loss computation. This is a clean separation — like training a depth network with GT depth supervision but deploying it without any depth sensor.

### Challenge 2: Phone IMU Quality
**Problem**: Phone MEMS IMUs (Bosch BMI260, InvenSense ICM-42688) are 10-100× noisier than research-grade IMUs (ADIS16448 in EuRoC, RT3003 in KITTI).
**Mitigation**: 
- Heavy IMU noise augmentation during training (inject phone-level noise into clean data)
- Phase 4 fine-tuning on actual phone data
- The learned noise corrector specifically targets this — it can learn phone-specific bias patterns
- AirIMU showed their approach works across IMUs from $0.5 automotive grade to $30K+ navigation grade

### Challenge 3: Feature Starvation in Highly Dynamic Scenes
**Problem**: If >80% of the image is dynamic (crowded intersection), aggressive masking leaves too few features.
**Mitigation**: 
- Adaptive threshold: if post-masking feature count < 20, raise threshold from 0.5 → 0.8 progressively
- Fall back to IMU-only propagation (the learned preintegration is accurate enough for 0.5-2 second gaps)
- The learned IMU head makes IMU-only periods much more survivable than with raw IMU

### Challenge 4: Deformable Objects (Trees, Cyclists)
**Problem**: Waving trees and pedaling cyclists have non-rigid motion that geometric methods miss.
**Mitigation**: 
- The learning-based approach handles this naturally — the network sees examples of deformable motion during training and learns to detect it
- FiLM conditioning helps: after accounting for ego-motion, residual motion in tree canopies is clearly non-zero, even though no semantic label captures "waving vegetation"
- VIODE includes various dynamic objects; TartanAir has environmental dynamics; copy-paste augmentation from COCO can inject diverse moving objects

### Challenge 5: Fundamental Matrix Degeneracy
**Problem**: F estimation fails under pure rotation (no translation) or planar scenes.
**Mitigation**: 
- IMU detects pure rotation (translation component of preintegration ≈ 0) → skip F-based consistency loss, rely purely on IMU ego-motion for that frame pair
- In practice, handheld/phone motion almost always has some translation component
- The mask network doesn't depend on F at inference — F is only used in training losses

### Challenge 6: Time Offset Drift on Phones
**Problem**: Android's sensor timestamps can drift relative to each other (despite sharing the same clock base), especially under thermal throttle.
**Mitigation**:
- The temporal aligner runs continuously, not just at startup
- Training with time-varying offset augmentation (`t_d(t) = t_d_0 + α·t`)
- The network's mask prediction is trained to be robust to ±30ms offset — typical phone drift is <10ms over minutes

---

## 8. Publishability and Contribution Assessment

### Novel Contributions:
1. **First intrinsic-free, sync-free learned dynamic masking system for VIO** — no existing system operates under these constraints
2. **IMU-conditioned visual dynamic detection via FiLM** — novel fusion of inertial ego-motion into per-pixel dynamic classification
3. **Jointly learned IMU noise correction, covariance estimation, AND visual dynamic masking** — a multi-task architecture that benefits both the masking and the VIO backend
4. **Learned temporal alignment** as part of the pipeline rather than assumed known
5. **Backend-agnostic interface with multiple integration depths** — practical contribution

### Target Venues:
- **Primary**: IEEE RA-L (with ICRA/IROS presentation) — good fit for systems-oriented VIO work
- **Reach**: CVPR/ECCV (if masking accuracy + ablation results are strong enough)
- **Alternative**: IROS full paper, ICRA full paper

### What Would Make This a Strong Submission:
- Clear improvement on VIODE (all 4 dynamic levels) with ATE/RPE numbers
- A4 ablation showing IMU conditioning matters for masking
- A6/A7 showing learned IMU quantities improve VIO independently
- Phone demo video showing real-time operation
- Open-source code + ROS2 package
