# DynaMask V2.5 — AirIMU-Bootstrapped Score Head with a Simplified Self-Supervised Loss Stack

## 0. Elevator Pitch

DynaMask V2.5 is a focused refinement of V2. It keeps V2's architecture (RAFT encoder + FiLM IMU conditioning + ConvGRU flow decoder + differentiable two-frame bundle adjustment) and changes three things:

1. **IMU encoder** — swap our from-scratch `NoiseCorrector` for a drop-in port of AirIMU's `CodeNet`, loaded verbatim from public pretrained weights and **frozen forever**. Kills the IMU pretraining phase entirely.
2. **Mask head → Score head** — remove the sigmoid, export a calibrated-or-raw continuous score, and train it with a pairwise ranking loss. The downstream VIO backend picks its own threshold (fixed, top-K%, Otsu, or IMU-adaptive).
3. **Loss stack** — cut from V2's five to **three** losses: `L_pose`, `L_rank`, `L_smooth`. Two-phase curriculum collapses into **one** single end-to-end training phase with a short warmup ramp on `L_pose`.

Net effect: **~1-2 days of compute saved, trivial ablation story (4 runs vs 2¹⁰), stronger IMU prior from epoch 0, and a more informative per-pixel output for downstream consumers.**

---

## 1. Motivation — What V2 Got Right and What V2 Is Missing

### 1.1 What V2 got right (and what we are deliberately preserving)

- **Intrinsic-free, mono-at-inference** — the shipped model takes left image + IMU window and produces a per-pixel dynamic signal. No camera intrinsics required at deployment. This is a genuine phone-deployable design and must not regress.
- **RAFT encoder + FiLM IMU conditioning** — RAFT features are purpose-built for dense matching, InstanceNorm survives batch=1 inference, and the FiLM channel is the right place to inject ego-motion priors. Nothing to change here.
- **Differentiable two-frame BA as the core self-supervision signal** — the geometric consistency oracle is sound. The safeguards (SafeCholeskySolver, GradientClip ±0.01, adaptive damping, depth-validity check, hard outlier rejection) are the right shape. They stay.

### 1.2 What V2 is missing

- **The IMU encoder is trained from scratch.** V2's `NoiseCorrector` is ~50K random-init parameters that need a whole Phase 0 on EuRoC before Phase 2 end-to-end training is even viable. This is ~1-2 days of compute that AirIMU has already spent, on far more data than we have.
- **The mask head is a binary classifier.** Sigmoid + BCE collapses the genuinely continuous concept of "dynamism" into a single threshold decision. Downstream VIO backends (OpenVINS, ORB-SLAM3, DPVO) each want different operating points, and a single threshold serves none of them optimally.
- **Five losses with overlapping signal.** V2's training uses `L_pose`, `L_photo`, `L_reproj`, `L_reg`, `L_imu`. Ablation across all five needs 2⁵ = 32 runs. In practice nobody will do that, and we won't know which losses actually pull weight.
- **Two non-trivial training phases.** Phase 0 (IMU pretrain) + Phase 2 (mask end-to-end) requires two configs, two optimizers, two checkpoints. Operational complexity without enough scientific return.

V2.5 addresses each of these without touching the architectural backbone.

---

## 2. IMU Encoder — AirIMU Drop-In

### 2.1 Why AirIMU

AirIMU's `CodeNet` is a published, peer-reviewed IMU denoiser with public pretrained weights trained on EuRoC, KITTI, TUM-VI, and SubT-MRS — orders of magnitude more IMU data than we have. It produces per-sample bias corrections and log-variance estimates that are directly consumable by a standard preintegrator. It is exactly the module we were re-inventing, except theirs already works.

### 2.2 Architecture mirror

We replace `dynamask_vio/models/imu_encoder.py::NoiseCorrector` with a new class `AirIMUCorrector` that mirrors AirIMU's `model/code.py::CodeNet` exactly, so `load_state_dict(strict=True)` succeeds against their public checkpoint without surgery.

```
class AirIMUCorrector(nn.Module):
    """
    Exact structural clone of AirIMU's CodeNet so their pretrained
    state_dict loads with strict=True. Zero deviation from upstream
    layer names, shapes, or tensor orderings.
    """
    cnn             : CNNEncoder(c=[6, 32, 64], k=[7, 7], s=[3, 3])   # downsamples ×9
    gru1            : nn.GRU(input_size=64,  hidden_size=128, num_layers=1, batch_first=True)
    gru2            : nn.GRU(input_size=128, hidden_size=256, num_layers=1, batch_first=True)
    accdecoder      : Sequential(Linear(256, 128), GELU(), Linear(128, 3))
    acccov_decoder  : Sequential(Linear(256, 128), GELU(), Linear(128, 3))
    gyrodecoder     : Sequential(Linear(256, 128), GELU(), Linear(128, 3))
    gyrocov_decoder : Sequential(Linear(256, 128), GELU(), Linear(128, 3))

    # Buffers (not parameters):
    gyro_std = tensor(pi/180)   # ≈ 0.01745
    acc_std  = tensor(0.1)
```

### 2.3 Input/output contract

**Input**: `imu_window: [B, N, 7]` = `(dt, ax, ay, az, gx, gy, gz)`.

Internally we split off `dt`, pass `[acc, gyro]` (shape `[B, N, 6]`) into the network, and consume the `dt` stream only in the preintegrator.

**Output (per call)**:
- `correction_acc : [B, N', 3]` — additive accel correction, scaled by `acc_std`
- `correction_gyro: [B, N', 3]` — additive gyro correction, scaled by `gyro_std`
- `acc_cov : [B, N', 3]` — per-axis accel variance, computed as `exp(raw - 5)`
- `gyro_cov: [B, N', 3]` — per-axis gyro variance, computed as `exp(raw - 5)`

where `N' = N − interval` with `interval = 9` (AirIMU's downsampling window). AirIMU's `_update` broadcast routine is ported verbatim to map the `N'`-length features back to `N`-length corrections.

### 2.4 Sign convention flip

AirIMU's convention is **additive**: `corrected = raw + correction`.

V2's convention is **subtractive**: `corrected = raw − delta_b`.

We adopt AirIMU's convention throughout `imu_encoder.py`. The `IMUEncoder.forward` call becomes:

```python
acc_correction, gyro_correction, acc_cov, gyro_cov = self.airimu(acc_raw, gyro_raw)
acc_corrected  = acc_raw  + acc_correction
gyro_corrected = gyro_raw + gyro_correction
```

This is a one-sign flip. All downstream preintegration logic is unchanged.

### 2.5 Freeze policy: frozen forever

**AirIMU's parameters stay frozen for the entire lifetime of training.** No Phase 2 thaw, no 0.01× LR discount, no periodic unfreezing. Reasons:

1. **No shortcut risk.** A frozen AirIMU cannot learn to "fake" IMU corrections that reduce mask loss at the expense of IMU correctness.
2. **No weight anchor regularizer needed.** Freezing is a harder, cleaner anchor than an L2 penalty.
3. **One less config flag, one less LR group, one less ablation axis.** Operational simplicity.
4. **We cannot beat their pretraining budget.** AirIMU saw more IMU data than we will ever assemble. Fine-tuning on a smaller dataset is likely to hurt, not help.

The only trainable IMU-side parameters are:
- `FeatureMLP` (18 → 64 → 128, ~10K params) — this trains jointly with the score head from epoch 0. It is small enough that a cold start is cheap.
- `DifferentiablePreintegrator` has no learnable parameters. It is pure physics.

### 2.6 State-dict loader

```python
def load_airimu_weights(encoder: AirIMUCorrector, ckpt_path: str):
    """Load AirIMU CodeNet weights with strict=True."""
    ckpt = torch.load(ckpt_path, map_location="cpu")
    state = ckpt.get("model_state_dict", ckpt)

    # AirIMU's CodeNet is wrapped in their ModelBase; strip the wrapping prefix
    # so the keys align with our AirIMUCorrector attribute names.
    cleaned = {}
    for k, v in state.items():
        if k.startswith("module."):
            k = k[len("module."):]
        cleaned[k] = v

    missing, unexpected = encoder.load_state_dict(cleaned, strict=False)
    # strict=False here ONLY because AirIMU's upstream checkpoint may contain
    # auxiliary buffers (gravity, integrator state) that aren't part of our
    # corrector. We log missing/unexpected and assert both are empty modulo
    # that known whitelist.
    _assert_airimu_load_clean(missing, unexpected)

    for p in encoder.parameters():
        p.requires_grad_(False)
    encoder.eval()
    return encoder
```

The whitelist is asserted in code so a silent weight-mismatch cannot regress into training.

### 2.7 What this buys us

| Metric | V2 | V2.5 |
|---|---|---|
| IMU encoder init | random (trained from scratch) | pretrained (EuRoC + KITTI + TUM-VI + SubT-MRS) |
| Phase 0 pretraining compute | ~1-2 GPU-days | 0 |
| Datasets required for training | TartanAir v2 + EuRoC (IMU pretrain) | TartanAir v2 only |
| Config files | 2 (phase 0, phase 2) | 1 |
| Optimizers / LR groups | 2 | 1 + layerwise decay on RAFT |

---

## 3. Score Head — Replacing the Binary Mask Head

### 3.1 Why score instead of binary mask

"Dynamic" is not a binary property. A parked car is ~10% dynamic (may start moving soon). A swinging arm is ~60% dynamic (as-rigid-as-possible deformation). A sprinting person is ~100% dynamic. A binary sigmoid + BCE at 0.5 collapses all of these onto the same decision.

A **continuous score** preserves this information and pushes the threshold decision to the downstream consumer, which matters because:

- **OpenVINS** is conservative — it wants very clean static features, so a high threshold is best.
- **ORB-SLAM3** is aggressive — it tolerates some dynamic contamination, so a lower threshold is best.
- **DPVO-style patch trackers** want a **top-K ranking**, not a threshold at all.
- **IMU-adaptive backends** want the threshold scaled by the IMU confidence — trust the mask more when the IMU is noisy.

A single binary mask serves none of these optimally. A score serves all of them.

Additionally, when a sigmoid saturates (logits outside ±5), you lose all ordering information near the boundary and gradients vanish. A raw score keeps the ordering intact and keeps gradients alive.

### 3.2 Architecture

```
Current V2 MaskHead:
    Conv2d(128, 64, 3×3) + ReLU
    Conv2d(64,  1, 1×1)
    GradientClip
    sigmoid(upsample(·))  →  mask ∈ [0, 1]^{H×W}

New V2.5 ScoreHead:
    Conv2d(128, 64, 3×3) + ReLU
    Conv2d(64,  1, 1×1)
    GradientClip
    upsample(·)           →  score_logit ∈ ℝ^{H×W}           (exported)
    clamp(score_logit, -10, 10) / T_calib  →  score_cal      (exported)
```

The network layers are identical. The only change is that the sigmoid is removed from the *exported* output. Internally, during training, we still apply a sigmoid where a loss term needs a probability (e.g., for temperature calibration fitting after training ends), but the BA-facing correspondence weighting uses the raw score, and downstream consumers get both.

### 3.3 Post-hoc temperature calibration

After training finishes, we fit a single scalar temperature `T_calib` on a held-out validation set (1000-5000 frames is plenty) by minimizing NLL between `sigmoid(score_logit / T)` and Sampson-distance-derived pseudo-labels. This is Guo et al. 2017 temperature scaling — one scalar, closed-form-ish, no network changes, no retraining. `T_calib` is baked into the exported ONNX as a constant scalar.

### 3.4 Deployment-time thresholding utilities

Shipped in `dynamask_vio/inference.py`:

```python
def threshold_fixed(score: Tensor, value: float = 0.5) -> Tensor:
    """Fixed scalar threshold on the calibrated score."""
    return score > value

def threshold_top_k_percent(score: Tensor, pct: float = 0.10) -> Tensor:
    """Mark the top `pct` fraction of pixels (by score) as dynamic.
    Resolution-independent — use when the downstream tracker wants a
    fixed fraction of usable static points regardless of scene content."""
    k = max(1, int(pct * score.numel()))
    thresh = score.view(-1).topk(k).values[-1]
    return score >= thresh

def threshold_otsu(score: Tensor) -> Tensor:
    """Otsu's method — find the bimodal split automatically.
    Use when the score distribution is approximately bimodal (common
    when the scene contains both very static and very dynamic regions)."""
    hist = torch.histc(score.flatten(), bins=256, min=0.0, max=1.0)
    ...  # standard Otsu implementation
    return score > otsu_thresh

def threshold_imu_adaptive(score: Tensor, imu_trace_cov: Tensor) -> Tensor:
    """Adaptive threshold scaled by IMU confidence.
    High IMU confidence (low Σ_preint) → trust mask aggressively (lower threshold).
    Low IMU confidence (high Σ_preint) → trust mask conservatively (higher threshold).
    The IMU confidence is trace(Σ_preint) normalized against a calibrated max."""
    conf = torch.exp(-imu_trace_cov / IMU_TRACE_CAL).clamp(0, 1)  # [0, 1]
    thresh = 0.70 - 0.40 * conf                                   # [0.30, 0.70]
    return score > thresh
```

The adaptive utility is the most interesting of the four — it closes the loop between the IMU encoder and the mask consumer, using the IMU's own reported confidence as the threshold controller.

### 3.5 ONNX export contract

The exported ONNX graph has these output tensors:

```
Output 0: score_logit       [B, 1, H, W]  float32  — unbounded raw logit
Output 1: score_cal         [B, 1, H, W]  float32  — sigmoid(logit / T_calib), in [0, 1]
Output 2: delta_bg          [B, N, 3]     float32  — gyro bias correction
Output 3: delta_ba          [B, N, 3]     float32  — accel bias correction
Output 4: sigma2_g          [B, N, 3]     float32  — gyro variance
Output 5: sigma2_a          [B, N, 3]     float32  — accel variance
Output 6: delta_R           [B, 3, 3]     float32  — preintegrated rotation
Output 7: delta_v           [B, 3]        float32  — preintegrated velocity
Output 8: delta_p           [B, 3]        float32  — preintegrated position
Output 9: Sigma_preint      [B, 9, 9]     float32  — preintegration covariance
```

V2's output was shape-compatible except `mask` replaced `score_logit` + `score_cal`. This is a **one-output-tensor addition**, which is a trivial downstream consumer change: existing code reading `mask` reads `score_cal` instead (semantically identical to V2's mask), and code that wants the unbounded score reads `score_logit`.

---

## 4. The 3-Loss Stack

### 4.1 Design philosophy

V2 had five losses (`L_pose`, `L_photo`, `L_reproj`, `L_reg`, `L_imu`) with overlapping signal. V2.5 cuts to **three**, chosen so that each loss has a distinct, non-overlapping role and can be ablated independently.

| Loss | Distinct role |
|---|---|
| `L_pose` | The geometric consistency oracle. Supplies the primary "is the mask correct?" signal through the differentiable BA. |
| `L_rank` | Direct score-ordering supervision from an intrinsic-free epipolar signal (Sampson distance). Ensures the score ranks pixels even before BA converges. |
| `L_smooth` | Prior on score spatial regularity. Prevents speckle, gives crisp boundaries at image edges. |

Anything that would be a fourth loss was either subsumed (`L_photo` → implicit in BA pose error), handled elsewhere (calibration → post-hoc temperature scaling), or deemed redundant (`L_fb_cycle` duplicates BA's own forward correspondence check).

### 4.2 `L_pose` — differentiable BA pose loss

```
L_pose = ‖log_SE3(T_BA · T_gt_inv)‖_rho
```

where:
- `T_BA` is the relative SE(3) pose returned by the differentiable two-frame BA (from V2, unchanged, with all its safeguards)
- `T_gt` is the ground-truth relative pose from the dataset (TartanAir v2 provides this)
- `log_SE3` is the matrix log on SE(3), returning a 6-vector (3 rotation + 3 translation)
- `‖·‖_rho` is a robust Huber norm (δ = 0.1 for rotation component, δ = 0.5 for translation)

**Correspondence weighting**: the BA consumes correspondences weighted by `(1 - σ(score_logit))`. Dynamic pixels get down-weighted → BA trusts them less → BA pose stays clean when the score is correct. The gradient flow is: pose loss → BA internals → correspondence weights → `sigmoid` → `score_logit` → ScoreHead → flow decoder → RAFT encoder → FiLM → FeatureMLP (→ frozen AirIMU, no-op).

**Rotation/translation split**: the rotation component is weighted 10× higher than the translation component in the final loss. Reason: the IMU preintegrates rotation accurately (gyro is reliable short-term), so rotation is the most trustworthy supervision target. Translation is scale-ambiguous and IMU-noisy.

**Warmup**: during the first 2 epochs, `L_pose`'s weight linearly ramps from 0.0 → 1.0. Reason: the BA is fragile at random initialization and can produce degenerate Hessians; letting `L_rank + L_smooth` do the early work gets the score head to a sensible regime before BA gradients come online.

Weight: `λ_pose = 1.0` (after warmup).

### 4.3 `L_rank` — pairwise margin ranking via Sampson distance

```
for each training pair:
    # Run current predictions
    score_logit = model(img_prev, img_curr, imu)
    flow        = model.flow  # internal, same batch

    # RANSAC a fundamental matrix from current "static-looking" correspondences
    static_mask = score_logit < score_logit.quantile(0.50)
    F, _ = ransac_fundamental(correspondences[static_mask])

    # Compute Sampson distance for ALL correspondences against F
    sampson = sampson_distance(correspondences, F)  # [K]

    # Sample K pairs (i, j) where i is low-Sampson, j is high-Sampson
    i_indices = torch.topk(-sampson, k=N_PAIRS).indices          # low Sampson
    j_indices = torch.topk( sampson, k=N_PAIRS).indices          # high Sampson

    # Pairwise margin loss: dynamic (high-Sampson) should score higher than
    # static (low-Sampson) by at least `margin`.
    score_i = score_logit[i_indices]
    score_j = score_logit[j_indices]
    L_rank = torch.clamp(MARGIN - (score_j - score_i), min=0).mean()
```

**Key properties**:
- **Intrinsic-free**: the fundamental matrix needs no camera intrinsics. Sampson distance is pure pixel-space geometry.
- **Self-bootstrapping**: the RANSAC F-matrix is estimated from the score's own current static estimate, but RANSAC is robust enough to tolerate moderate initial noise. Over a few hundred iterations, the score converges with the F-matrix.
- **Ordering, not probability**: `L_rank` does not care about absolute score values, only that dynamic > static. This is exactly what a downstream ranking consumer needs.
- **Class-balance free**: pairs are sampled 1:1 between low-Sampson and high-Sampson, so the static-majority class imbalance is handled automatically. No focal loss needed.

**Hyperparameters**:
- `N_PAIRS = 512` pairs per sample
- `MARGIN = 1.0` (in logit space)
- RANSAC: 200 iterations, 8-point algorithm, inlier threshold 1.0 pixel Sampson

Weight: `λ_rank = 0.3`.

### 4.4 `L_smooth` — edge-aware 2nd-order score smoothness

```
L_smooth = mean(‖∇²x score‖_1 · exp(-β‖∇x img‖) + ‖∇²y score‖_1 · exp(-β‖∇y img‖))
```

where `∇²` is the discrete 2nd-order spatial derivative and `β = 10`.

**Why 2nd-order**: 1st-order smoothness would penalize the score for having any boundary at all, which is wrong — we want sharp score boundaries at dynamic object edges. 2nd-order penalizes *curvature*, which allows crisp step-function edges but prevents speckle and jitter.

**Why edge-aware**: the `exp(-β‖∇img‖)` weight collapses the penalty to zero at image edges, so the score is free to change sharply where the image itself changes sharply (i.e., at object boundaries). This is the standard SfMLearner edge-aware smoothness prior.

Weight: `λ_smooth = 0.05`.

### 4.5 Total loss

```
L = λ_pose · L_pose  +  λ_rank · L_rank  +  λ_smooth · L_smooth
  = 1.0  · L_pose  +  0.3  · L_rank  +  0.05  · L_smooth
```

with `λ_pose` ramping from 0 → 1.0 over epochs 1-2.

### 4.6 Ablation matrix

Four runs give the full ablation story:

| Run | `L_pose` | `L_rank` | `L_smooth` | Hypothesis tested |
|---|---|---|---|---|
| **Full** | ✓ | ✓ | ✓ | Baseline |
| No `L_rank` | ✓ | ✗ | ✓ | Is the ranking signal necessary, or does BA alone suffice? |
| No `L_pose` | ✗ | ✓ | ✓ | Is BA necessary, or does pure Sampson-based ranking work? |
| No `L_smooth` | ✓ | ✓ | ✗ | Is the smoothness prior necessary, or do the other two give crisp enough scores? |

Four runs, four clear hypotheses, reportable in a single table.

### 4.7 What we dropped from V2 and why

| V2 loss | Fate in V2.5 | Reason |
|---|---|---|
| `L_photo` (photometric warp) | **Removed** | Subsumed by `L_pose`: if the BA pose is correct, the photometric warp is correct; no extra signal. Also noisy without intrinsics. |
| `L_reproj` (static-point reprojection) | **Removed** | Used only inside BA as an internal step; no longer a separate loss term. |
| `L_reg` (entropy balance) | **Removed** | `L_rank`'s pairwise sampling handles class imbalance natively. Entropy loss is a band-aid for binary classifiers. |
| `L_imu` (preintegration error) | **Removed** | AirIMU is frozen; nothing to regularize. |

---

## 5. Single-Phase Training Curriculum

### 5.1 Phases collapse to one

V2 had Phase 0 (IMU pretrain on EuRoC) and Phase 2 (mask end-to-end). V2.5 has **one phase**. Phase 0 is gone because AirIMU weights replace it. There is no Phase 2 thaw because AirIMU is frozen forever.

### 5.2 Training loop

```
Epoch 1-2   : L_pose weight ramps 0.0 → 1.0 (BA warmup)
              L_rank and L_smooth at full weight from epoch 1
              Score head learns from Sampson ordering alone during these epochs

Epoch 3-N   : All three losses at full weight
              Single cosine LR schedule, peak LR = 2e-4
              Batch size: 8 (gradient-checkpointed RAFT encoder lets us fit this on a 24GB GPU)
              Effective batch via accumulation: 32
              Total epochs: ~15-20 on TartanAir v2 (roughly 5-7 GPU-days on a single A100)
```

### 5.3 Optimizer

```python
optim = torch.optim.AdamW([
    {"params": score_head.parameters(),     "lr": 2e-4},
    {"params": flow_decoder.parameters(),   "lr": 2e-4},
    {"params": feature_mlp.parameters(),    "lr": 2e-4},
    {"params": film_layers.parameters(),    "lr": 2e-4},
    {"params": feature_encoder.parameters(),"lr": 2e-5},  # 0.1× LR decay on pretrained RAFT fnet
    {"params": context_encoder.parameters(),"lr": 2e-5},  # 0.1× LR decay on pretrained RAFT cnet
    # airimu_corrector NOT included — frozen
], weight_decay=1e-4)

scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optim, max_lr=2e-4, pct_start=0.1, anneal_strategy="cos",
    total_steps=len(train_loader) * NUM_EPOCHS,
)
```

### 5.4 Data

- **Primary**: TartanAir v2 (stereo, IMU, GT pose, GT dynamic masks for sanity check only — never used as supervision)
- **Held-out validation for temperature calibration**: 5000 frames randomly held out from TartanAir v2 at the start of training, untouched until the very end
- **Deployment sanity check**: TUM-VI walking_dynamic sequence (real phone-like data with dynamic humans)

EuRoC is no longer required for training (only consulted if/when temperature scaling wants a second calibration domain).

---

## 6. Essential Coding Nuances (5, was 10)

Each is a small, mechanical code change with a measurable impact. All are orthogonal and can be adopted independently.

### 6.1 Multi-iteration score supervision (RAFT-style)

V2's score head fires once, after all GRU iterations complete. RAFT and its descendants train the update operator by supervising every iteration with geometrically decreasing weight. We do the same for the score head.

```python
# In flow_decoder.forward:
score_logits_per_iter = []
h = initial_hidden
flow = zeros_like(init_flow)
for i in range(N_GRU_ITERS):
    h, flow_delta = gru_step(h, corr_lookup(flow), inp)
    flow = flow + flow_delta
    score_logit_i = score_head(h)                    # <-- apply at every iteration
    score_logits_per_iter.append(score_logit_i)

# In loss computation:
GAMMA = 0.8
total_score_loss = 0
for i, score_i in enumerate(score_logits_per_iter):
    weight = GAMMA ** (N_GRU_ITERS - 1 - i)
    total_score_loss += weight * compute_score_losses(score_i, ...)
```

Cost: ~10 lines. Benefit: noticeably faster convergence (RAFT's own ablations show 15-30% faster convergence with per-iter supervision).

### 6.2 Layerwise LR decay on RAFT backbone

RAFT's feature and context encoders are pretrained on FlyingThings3D/Sintel/KITTI and should be touched gently. Use 0.1× LR for them relative to the newly initialized decoder/score head. See the optimizer config in §5.3.

Cost: a parameter-group split in the optimizer. Benefit: prevents catastrophic forgetting of RAFT's matching features during the first few epochs when the score head is still garbage.

### 6.3 Gradient checkpointing on the RAFT encoder

RAFT encoder activation memory dominates the per-sample memory footprint. Wrapping the encoder's residual blocks in `torch.utils.checkpoint.checkpoint` cuts activation memory by ~40%, which doubles the max batch size on a 24GB GPU (from 4 to 8).

```python
from torch.utils.checkpoint import checkpoint_sequential

# In BasicEncoder.forward:
out = self.stem(x)
out = checkpoint_sequential(self.layer1, 2, out, use_reentrant=False)
out = checkpoint_sequential(self.layer2, 2, out, use_reentrant=False)
out = checkpoint_sequential(self.layer3, 2, out, use_reentrant=False)
out = self.out_proj(out)
return out
```

Cost: ~5 lines and a config flag. Benefit: larger effective batch, better BA gradient averaging, faster convergence.

### 6.4 IMU augmentation actually turned on

V2's project plan mentions training-time IMU augmentation (time-offset injection, random IMU sample dropout) but the implementation is disabled. Turn it on:

- **Time offset**: inject `t_d ~ Uniform(-30ms, +30ms)` between IMU and camera streams per batch. Trains the mask head to be robust to phone temporal misalignment.
- **IMU dropout**: randomly drop 10-30% of IMU samples per window. Trains the preintegrator + feature MLP to handle the scheduler jitter real phones exhibit.
- **Accel/gyro noise injection**: add `N(0, σ_aug²)` noise with `σ_aug` matching typical consumer-grade IMU noise.

Cost: ~30 lines in the data loader. Benefit: deployment robustness. Without this, the model only works on clean synthetic IMU.

### 6.5 Score logit clipping

Score logits can drift to very large magnitudes during training if a few outlier samples pull hard on the margin loss. Clamp the exported `score_logit` to `[-10, 10]` at the output boundary:

```python
# In ScoreHead.forward, final line before return:
score_logit = score_logit.clamp(min=-10.0, max=10.0)
```

Cost: one line. Benefit: prevents gradient saturation of the sigmoid used in temperature calibration, keeps the score distribution well-conditioned, makes FP16 export numerically safe.

Note: this is a *forward* clamp, not a gradient clamp (which is what `GradientClip` does). Both safeguards co-exist.

### 6.6 What we dropped from the full nuance list and why

| Nuance | Fate | Reason |
|---|---|---|
| EMA teacher for self-distillation | Dropped | Marginal gain, doubles the model memory footprint, adds a hyperparameter (EMA decay). |
| Stochastic Weight Averaging (SWA) | Dropped (easy to add later) | Free-ish improvement but not essential for the first results. Adding later is a no-op for architecture. |
| GroupNorm swap | Dropped | AirIMU uses BatchNorm, and we are freezing AirIMU. RAFT uses InstanceNorm. Nothing to swap. |
| SafeCholesky failure rate logging | Dropped (add as metric, not a "nuance") | Logged to W&B as a scalar metric; no architectural change. |
| Curriculum on motion magnitude | Dropped | Over-engineered; TartanAir v2's motion distribution is already diverse enough. |

---

## 7. File-by-File Implementation Plan

### 7.1 New files

| File | Purpose | LOC estimate |
|---|---|---|
| `dynamask_vio/models/airimu_corrector.py` | `AirIMUCorrector` class (exact `CodeNet` clone), `load_airimu_weights` helper, `_update` broadcast routine | ~150 |
| `dynamask_vio/models/score_head.py` | `ScoreHead` class (replaces mask head), exports raw logit | ~40 |
| `dynamask_vio/losses/pairwise_rank.py` | `L_rank` — pairwise margin ranking via Sampson distance, with a differentiable RANSAC-F helper | ~120 |
| `dynamask_vio/losses/smoothness.py` | `L_smooth` — edge-aware 2nd-order smoothness | ~30 |
| `dynamask_vio/inference.py` (append section) | Four threshold utilities: `threshold_fixed`, `threshold_top_k_percent`, `threshold_otsu`, `threshold_imu_adaptive` | ~80 |
| `dynamask_vio/calibration/temperature_scaling.py` | Post-training temperature calibration routine (fit one scalar `T_calib` on held-out set) | ~60 |

### 7.2 Modified files

| File | Change |
|---|---|
| `dynamask_vio/models/imu_encoder.py` | Remove `NoiseCorrector` class entirely. Refactor `IMUEncoder` to instantiate `AirIMUCorrector`, flip sign convention from subtractive to additive. |
| `dynamask_vio/models/dynamask.py` | Wire `ScoreHead` in place of the old mask head. Apply score head at every GRU iteration (multi-iteration supervision). |
| `dynamask_vio/losses/self_supervised.py` | Delete `L_photo`, `L_reproj`, `L_reg`, `L_imu`. Keep `L_pose`. Add calls to `L_rank` and `L_smooth`. Implement the 0→1 `λ_pose` warmup. |
| `dynamask_vio/losses/imu_loss.py` | Delete (no longer used; AirIMU is frozen). |
| `dynamask_vio/train.py` | Remove Phase 0 code path entirely. Keep only the single end-to-end loop. Update optimizer to use parameter groups with RAFT layerwise LR decay. Load AirIMU checkpoint at startup and freeze. |
| `dynamask_vio/configs/*.yaml` | Delete Phase 0 config. Single config file with one phase. Add `airimu_weights_path`, `score_head`, `thresholds` sections. |
| `dynamask_vio/data/augmentations.py` | Enable time-offset, IMU-dropout, and IMU-noise augmentation that were planned but disabled. |
| `dynamask_vio/export_onnx.py` | Export `score_logit` and `score_cal` instead of `mask`. Bake `T_calib` as an ONNX constant. |
| `dynamask_vio/download_weights.py` | Add AirIMU checkpoint download entry. |

### 7.3 Deleted files

| File | Reason |
|---|---|
| `dynamask_vio/losses/imu_loss.py` | AirIMU is frozen — no IMU loss needed. |
| Any Phase 0 config files | No Phase 0. |

---

## 8. Why This Is Better (and Where It Might Be Worse) Than Current V2

### 8.1 Better

**1. IMU encoder starts strong, not from random init.**
V2 needs 1-2 GPU-days of EuRoC pretraining before its noise corrector is even usable. V2.5 skips this entirely — AirIMU's weights are trained on more data than we will ever collect. From epoch 0, the preintegrated rotation is accurate enough that `f_imu` provides useful FiLM conditioning to the visual backbone. V2 took ~5 epochs to reach comparable IMU quality.

**2. Training curriculum is single-phase.**
V2 has two phases, two configs, two checkpoints, two optimizer states. V2.5 has one. Operationally this is much simpler: no checkpoint hand-offs, no per-phase hyperparameter tuning, no "did we forget to load Phase 0 weights?" bugs.

**3. Loss stack is trivially ablatable.**
V2's five losses need 2⁵ = 32 ablation runs to characterize fully; nobody will do this. V2.5's three losses need exactly four runs (full + three leave-one-outs) to produce the full ablation story, which can be a single table in a paper or report.

**4. Score output is strictly more informative than a binary mask.**
The downstream VIO backend can threshold the score however it wants — fixed value, top-K%, Otsu, IMU-adaptive. A binary mask is lossy and commits to one operating point. A score commits to none.

**5. No shortcut risk through the IMU encoder.**
V2's noise corrector is trainable end-to-end, which in principle means the mask loss can corrupt its weights via gradient shortcuts (e.g., "learn to emit small IMU corrections that happen to make the BA pose match GT even for bad masks"). V2.5 eliminates this class of failure mode by freezing AirIMU.

**6. Fewer datasets required.**
V2 needs TartanAir v2 + EuRoC (for IMU pretrain). V2.5 needs only TartanAir v2. EuRoC becomes optional (nice-to-have for cross-domain sanity check).

**7. Deployment-time flexibility.**
V2 ships a single binary mask. V2.5 ships a score + a library of four threshold strategies + optional IMU-adaptive thresholding. Deployment engineers can tune for their specific backend without retraining.

**8. Faster training convergence from RAFT-style multi-iteration supervision.**
V2 supervises the score head once at the end of all GRU iterations. V2.5 supervises it at every iteration with geometrically decreasing weight. RAFT's own ablations show 15-30% faster convergence with this pattern.

**9. Pairwise ranking loss handles class imbalance natively.**
V2's entropy + balance loss is a band-aid for sigmoid-BCE's sensitivity to the static-heavy class distribution (~95% of pixels are static). V2.5's margin ranking samples 1:1 from low-Sampson and high-Sampson pools — class balance is automatic.

### 8.2 Where it might be worse

**1. New dependency on AirIMU.**
We are now tied to AirIMU's published weight format and architectural conventions. If AirIMU ever breaks their checkpoint format, we need to re-port. Mitigation: vendor a copy of the specific checkpoint we use in `dynamask_vio/weights/airimu_codenet_{version}.pth` and freeze it in git-lfs.

**2. Score-based output requires downstream changes for any consumer expecting a binary mask.**
V2 consumers reading the `mask` output must switch to `score_cal` (or threshold `score_logit` themselves). This is a one-line change per consumer but it *is* a change. Mitigation: provide a compatibility wrapper `get_binary_mask(score_cal, threshold=0.5)` in `inference.py`.

**3. Temperature calibration adds a post-training step.**
V2 is "train, export, done." V2.5 is "train, fit temperature on held-out, export, done." One extra step in the pipeline, one extra dataset split to manage. Mitigation: automate temperature fitting inside `export_onnx.py` so it happens as part of export, not as a separate user action.

**4. Frozen AirIMU cannot adapt to domain shift.**
If our deployment IMU distribution is very different from AirIMU's training distribution (e.g., we deploy on a low-end phone whose IMU noise is far worse than AirIMU saw), a frozen encoder will not adapt. Mitigation: document this clearly; if we actually observe domain shift in practice, we can add an unfreeze escape hatch later (the code structure supports it). This is a future problem to handle when we have evidence it exists.

**5. Fewer training signals means less defense in depth.**
V2's five losses provided a kind of redundancy: if one loss was off, the others could compensate. V2.5's three losses are tighter — if one is broken, it shows up loudly. This is a feature for ablation clarity but a risk during initial development: a bug in any one loss will block training convergence. Mitigation: the three losses are each small and well-understood, so bugs should be rare and loud.

**6. RANSAC inside `L_rank` is a moving target during early training.**
The fundamental matrix used by `L_rank` is RANSAC'd from the score's own current estimate. Early in training this estimate is random, so the F-matrix is noisy, so the ranking signal is noisy. In the worst case this could produce a local minimum. Mitigation: the 2-epoch warmup period is specifically designed to let the score settle under `L_rank` alone before `L_pose` comes online and couples the system more tightly. If this turns out to be insufficient, we can bootstrap the F-matrix for the first epoch using a simple motion-magnitude heuristic (points with very low flow magnitude are likely static), then switch to score-based selection.

### 8.3 One-line summary

**V2.5 is V2 with its IMU encoder replaced by a frozen AirIMU, its mask head replaced by a continuous score head, its five losses cut to three, and its two training phases merged into one.**

Everything else — RAFT backbone, FiLM conditioning, ConvGRU flow decoder, differentiable BA with all its safeguards, intrinsic-free mono inference, ONNX export — is preserved.

---

## 9. Success Criteria

V2.5 ships when, on TartanAir v2 held-out and TUM-VI walking_dynamic:

| Metric | Target |
|---|---|
| Score AP against GT dynamic masks (TartanAir v2) | ≥ 0.65 (V2 achieved ~0.55 with its binary mask) |
| Downstream ATE improvement (OpenVINS on TUM-VI walking_dynamic, score-thresholded) | ≥ 15% reduction vs. no-mask baseline |
| IMU rotation error (5-frame window, TUM-VI) | ≤ AirIMU's own published number on the same sequence |
| Inference latency (score head only, 480×640 on mobile GPU) | ≤ 25 ms/frame |
| Training time to convergence (single A100) | ≤ 7 days |
| Ablation completeness | All 4 runs completed, results in a single table |

---

## 10. Risks and Mitigations

| Risk | Probability | Mitigation |
|---|---|---|
| `L_rank` RANSAC is noisy early in training, causing local minima | Medium | 2-epoch warmup with `L_pose` faded in; fallback bootstrap via motion-magnitude heuristic if needed. |
| AirIMU checkpoint format shifts upstream | Low | Vendor the exact checkpoint in git-lfs; freeze a known-good version. |
| BA Hessian becomes singular under margin-loss-induced extreme score distributions | Low | `SafeCholeskySolver` (carried over from V2) zeros the update and gradient on singular Hessians. Log singular-rate to W&B. |
| Temperature calibration fits an unstable scalar because held-out set is too small | Low | Held-out set is 5000 frames; `T_calib` is a single scalar, so sample efficiency is high. |
| FP16 training saturates the score logit (|logit| > 65504) | Very low | §6.5 clamp at ±10 at the head output makes this impossible. |
| Multi-iteration supervision causes the score head to overfit early-GRU noise | Low | Geometric weighting `0.8^(N-i)` heavily favors later iterations; this is RAFT's own tested scheme. |
| Score head ordering collapses when all pairs have similar Sampson (e.g., stationary camera) | Medium | `L_rank` mean is skipped when the Sampson spread is below a threshold; the step falls back to `L_pose + L_smooth` only. |
| Frozen AirIMU under-performs on TUM-VI due to domain shift | Medium | Evaluate early; if observed, the freeze is an implementation choice that can be reverted with a one-line config flag. Design documents the escape hatch. |

---

## 11. Open Questions (to resolve before implementation)

1. **Which specific AirIMU checkpoint to vendor?** CodeNet trained on EuRoC-only? On the combined-domain dataset? On SubT-MRS? The combined-domain checkpoint is likely safest for deployment robustness, but the EuRoC-only checkpoint may transfer more cleanly to our existing EuRoC evaluation. Resolve after a quick cross-domain sanity check.

2. **Margin value for `L_rank`.** The design specifies `MARGIN = 1.0` in logit space, but this is a guess. Sweep `{0.5, 1.0, 2.0}` on a small ablation run before fixing.

3. **Number of pairs per sample for `L_rank`.** The design specifies `N_PAIRS = 512`. This is a memory/signal trade-off; confirm with a memory profile on the target batch size.

4. **Should `L_smooth` be applied to the raw logit or the calibrated score?** The design applies it to the logit. Calibrated-score smoothness would be more semantically meaningful but would couple `L_smooth` to `T_calib`, which is fit post-training. Logit smoothness is cleaner; keep it unless there is evidence it produces visual artifacts.

5. **Does `L_rank` need a minimum-Sampson-spread guard?** If the scene is stationary (camera + scene both static), Sampson distance is near-zero everywhere and the ranking signal degenerates. The design mentions this in §10 but does not specify the threshold. Resolve empirically on the first few training runs.

These questions are all implementation-level and do not change the design. They are listed here so they are not forgotten when writing code.
