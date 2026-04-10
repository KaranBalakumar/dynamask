# DynaMask-V2: Self-Supervised Dynamic Masking via IMU-Conditioned Optical Flow and Differentiable Bundle Adjustment

---

## 1. Problem Statement

DynaMask-V1 trains a dynamic mask predictor supervised by ground truth segmentation masks from VIODE (simulation). This creates two fatal problems:

1. **No GT masks in the wild.** Real-world deployments (phones, drones, robots) never have per-pixel dynamic object annotations. The model can only train on synthetic data and must generalize across a sim-to-real gap.
2. **Semantic supervision is brittle.** GT masks are semantic ("this is a person") not geometric ("this pixel violates the rigid scene assumption"). A parked car is semantically dynamic but geometrically static. A swinging door is semantically static but geometrically dynamic. We want geometric dynamism, not semantic class membership.

**The fix:** replace GT mask supervision entirely with a self-supervised geometric signal. The camera-IMU system already contains enough information to determine what is moving — we just need to extract it differentiably during training.

---

## 2. Core Idea (One Paragraph)

Predict dense optical flow between consecutive frames using a lightweight RAFT-style decoder whose features come from RAFT's own pretrained feature encoder, FiLM-conditioned on learned IMU ego-motion embeddings. The IMU tells the network how the camera moved, so the network can implicitly separate ego-motion-induced flow from dynamic-object-induced flow. A mask head on top of the flow decoder's hidden state predicts per-pixel dynamic probability. During training only, static correspondences (selected by the mask) are fed into a differentiable two-frame bundle adjustment that uses dataset-provided camera intrinsics to estimate the camera pose. The BA pose is compared against IMU preintegration and (when available) ground truth — this geometric consistency signal backpropagates through the BA, through the mask, through the flow decoder, through FiLM, and into the IMU encoder. At inference, the BA is discarded and flow is not exported. The shipped model outputs only: dynamic mask + IMU quantities (bias corrections, per-sample noise, preintegration, covariance). No intrinsics needed. No BA needed. Identical output interface to V1 — zero backend changes.

---

## 3. Why This Works

The key insight is that the IMU provides a strong ego-motion prior (especially rotation — gyroscope is accurate short-term) that breaks an otherwise chicken-and-egg problem:

```
Without IMU:  need mask to estimate ego-motion, need ego-motion to determine mask
With IMU:     IMU provides ego-motion prior → network learns which flow is ego vs dynamic
              → mask emerges from flow decomposition → BA refines the training signal
```

The differentiable BA during training acts as a **geometric consistency oracle**: "given your predicted mask, do the remaining (static) correspondences produce a geometrically consistent reconstruction?" If the mask is wrong (misses dynamic objects), the BA will have high reprojection error and produce a pose that disagrees with the IMU — generating gradients that push the mask to improve.

The IMU warm-starts the BA and prevents degenerate solutions, since the BA always has a good initial pose estimate from preintegration regardless of mask quality. However, IMU warm-starting alone is not sufficient to survive the first epochs — additional safeguards are needed (see Section 7.3.2 for the full bootstrap safety analysis, informed by DPVO's training strategies).

---

## 4. Architecture

### 4.1 Why RAFT's Encoder Instead of DDRNet

DDRNet-23-slim was designed for semantic segmentation: BatchNorm (degrades at batch=1 inference), ImageNet pretraining (classification features, not matching features), multi-class pixel labelling. Our task is dense matching + motion decomposition — a fundamentally different feature regime.

RAFT's encoder is purpose-built for this:

| Property | DDRNet-23-slim | RAFT BasicEncoder |
|----------|---------------|-------------------|
| **Designed for** | Semantic segmentation | Dense feature matching |
| **Normalisation** | BatchNorm | InstanceNorm (works at batch=1, better domain transfer) |
| **Pretraining** | ImageNet (classification) | FlyingChairs → FlyingThings3D → Sintel → KITTI (flow on real + photorealistic synthetic) |
| **Domain gap** | Large (classification → matching) | None (flow → flow) |
| **Params** | ~1.8M | ~1.48M (feature + context encoder) |
| **Output res** | Multi-scale (1/8, 1/16) | Single-scale 1/8 |

The RAFT encoder has seen real-world images (KITTI) and photorealistic synthetic (Sintel, FlyingThings3D) during pretraining. Its features are already optimised for the correlation-based matching that our flow decoder performs. DDRNet features would need to learn matching from scratch during our training — a significant handicap.

**RAFT uses two encoders:**
- **Feature encoder** (shared weights, called on both frames): produces features for the correlation volume. This is the matching backbone.
- **Context encoder** (called on frame t only): produces the initial GRU hidden state and contextual features for the update operator. This encodes scene structure.

Both use the same architecture (BasicEncoder) but separate weights — RAFT found this works better than a single shared encoder because matching features and context features serve different purposes. We load both from the pretrained RAFT checkpoint.

### 4.2 Full System Diagram

```
═══════════════════════════════════════════════════════════════════════
                        INFERENCE PIPELINE (ships on device)
═══════════════════════════════════════════════════════════════════════

 Frame t-1 ─┐                                    IMU window [N×7]
 Frame t   ─┤                                         │
             │                                         ▼
             │                               ┌──────────────────┐
             │                               │   IMU Encoder    │
             │                               │  (NoiseCorrector │
             │                               │   + Preintegrator│
             │                               │   + FeatureMLP)  │
             │                               └────────┬─────────┘
             │                                        │
             │                              ┌─────────┴──────────┐
             │                              │                    │
             │                         f_imu [128]        IMU quantities
             │                              │            (δb, σ², ΔR,Δv,Δp,Σ)
             │                              │                    │
             │           ┌──────────────────┘                    │
             │           │                                       │
             ▼           ▼                                       │
    ┌─────────────────────────┐                                  │
    │  RAFT Feature Encoder   │                                  │
    │  (BasicEncoder, shared  │                                  │
    │   weights, called twice │                                  │
    │   on frame t-1 & t)     │                                  │
    │                         │◄── FiLM(f_imu) at each           │
    │  InstanceNorm, ResBlocks│    residual stage                │
    │  64 → 96 → 128 channels│                                  │
    │  Output: [B,128,H/8,W/8]                                  │
    └────────┬────────────────┘                                  │
             │                                                   │
             │        ┌──────────────────────┐                   │
             │        │ RAFT Context Encoder  │                  │
             │        │ (BasicEncoder, frame t│                  │
             │        │  only, separate wts)  │                  │
             │        │                       │                  │
             │        │ Output: [B,128,H/8,W/8]                 │
             │        │ → split into:         │                  │
             │        │   net: GRU hidden init│                  │
             │        │   inp: motion context │                  │
             │        └──────────┬────────────┘                  │
             │                   │                               │
             ▼                   ▼                               │
    ┌─────────────────────────────────────────┐                  │
    │     Flow Decoder (RAFT Update Operator) │                  │
    │                                         │                  │
    │  1. Correlation Volume                  │                  │
    │     (4-level pyramid, local lookup r=4) │                  │
    │                                         │                  │
    │  2. Motion Encoder                      │                  │
    │     (corr features + current flow)      │                  │
    │                                         │                  │
    │  3. ConvGRU (3 iterations, 128-ch)      │                  │
    │     Iteratively refines flow estimate   │                  │
    │                                         │                  │
    │  4. Heads from final GRU hidden state:  │                  │
    │     ├─ Flow [2, H/8, W/8] (INTERNAL)    │                  │
    │     └─ Mask [1, H, W]   ← EXPORTED      │                  │
    └────────┬────────────────────────────────┘                  │
             │                                                   │
             │  (flow stays internal — drives                    │
             │   GRU state, feeds BA at train                    │
             │   time, NOT exported)                             │
             │                                                   │
             ▼                                                   ▼
    ┌──────────────────────────────────────────────────────────────┐
    │                    OUTPUT TO ANY VIO BACKEND                 │
    │                                                              │
    │  • Dynamic mask [H×W]         (mask features in dynamic px) │
    │  • Bias corrections δb_g, δb_a   [N×3] each                 │
    │  • Per-sample noise σ²_g, σ²_a   [N×3] each                 │
    │  • Preintegrated ΔR [3×3], Δv [3], Δp [3]                  │
    │  • Preintegration covariance Σ [9×9]                        │
    │                                                              │
    │  ─── IDENTICAL TO V1 OUTPUT (minus flow_residual) ───       │
    │  ─── ZERO CHANGES NEEDED IN ANY VIO BACKEND       ───       │
    └──────────────────────────────────────────────────────────────┘


═══════════════════════════════════════════════════════════════════════
                  TRAINING-ONLY SCAFFOLD (discarded at inference)
═══════════════════════════════════════════════════════════════════════

    From inference pipeline:
    ├─ predicted flow (internal) ───┐
    ├─ predicted mask ──────────────┤
    └─ IMU preintegrated pose ──────┤
                                    ▼
                        ┌───────────────────────┐
                        │  Static Correspondence│
                        │  Selector             │
                        │                       │
                        │  mask < 0.3 → static  │
                        │  sample top-K points  │
                        └───────────┬───────────┘
                                    │
                            K correspondences
                            (u,v) ↔ (u',v')
                                    │
                                    ▼
                        ┌───────────────────────┐
                        │  Differentiable       │
                        │  Two-Frame BA         │  ← uses dataset K (intrinsics)
                        │                       │  ← warm-start R from IMU ΔR
                        │  Gauss-Newton, 3 iter │
                        │  Hard outlier reject  │
                        │  + SafeCholeskySolver │
                        │  + adaptive damping   │
                        └───────────┬───────────┘
                                    │
                              BA pose (R̂, t̂)
                                    │
                                    ▼
                        ┌───────────────────────┐
                        │  Self-Supervised       │
                        │  Loss Functions        │
                        │                       │
                        │  L_pose: BA vs IMU     │
                        │  L_photo: warp error   │
                        │  L_reproj: static pts  │
                        │  L_reg: mask balance   │
                        │  L_imu: preint error   │
                        └───────────────────────┘
                                    │
                              ∇ backprop
                                    │
                    ┌───────────────┼───────────────┐
                    ▼               ▼               ▼
              Mask Head      Flow Decoder      IMU Encoder
              (learn what    (learn accurate   (learn better
               is dynamic)    correspondences)  ego-motion)
```

### 4.3 What Stays From V1

| Module | Params | Change |
|--------|--------|--------|
| IMU Encoder (NoiseCorrector + Preintegrator + FeatureMLP) | ~55K | **No change.** Same architecture, same outputs. |
| FiLM conditioning | ~74K | **Adapted.** Applied to RAFT feature encoder's 3 residual stages (64, 96, 128 ch) instead of DDRNet stages. Same mechanism, different channel widths. |

### 4.4 What Changes From V1

| V1 Component | V2 Replacement | Why |
|-------------|----------------|-----|
| DDRNet-23-slim backbone (~1.8M) | RAFT Feature Encoder + Context Encoder (~1.48M) | Flow-native features with InstanceNorm; pretrained on real+synthetic flow datasets; lighter |
| TemporalDecoder (corr + GRU → mask + flow_res) | RAFT-style Update Operator (corr pyramid + GRU → flow + mask) | Proper iterative flow refinement; correlation pyramid instead of single-level; mask from GRU state |
| Correlation at 1/16 on stage3 (128-ch) | 4-level correlation pyramid at 1/8 on RAFT features (128-ch) | Higher resolution, multi-scale matching |
| Focal loss on GT masks | Self-supervised loss from differentiable BA | No GT masks needed |
| Flow residual + mask as dual outputs | Mask-only output (flow is internal) | Lighter export; backends don't consume flow; zero interface change |

### 4.5 What Is Added (Training Only)

| Module | Params | Purpose |
|--------|--------|---------|
| Differentiable Two-Frame BA | 0 (no learned params) | Provides geometric consistency signal for self-supervision |
| Static Correspondence Selector | 0 | Differentiable top-K selection using mask probabilities as soft weights |

---

## 5. Component Details

### 5.1 RAFT Feature Encoder (replaces DDRNet-23-slim)

The standard RAFT BasicEncoder. Loaded directly from pretrained RAFT checkpoints (e.g. `raft-things.pth`, `raft-sintel.pth`).

```
Architecture (BasicEncoder):
  Conv2d(3, 64, 7×7, stride=2, padding=3) + InstanceNorm + ReLU     → [B, 64,  H/2,  W/2]
  ResidualBlock(64,  64)  × 2                                        → [B, 64,  H/2,  W/2]
  ResidualBlock(64,  96,  stride=2) + ResidualBlock(96,  96)         → [B, 96,  H/4,  W/4]
  ResidualBlock(96,  128, stride=2) + ResidualBlock(128, 128)        → [B, 128, H/8,  W/8]
  Conv2d(128, 128, 1×1)                                              → [B, 128, H/8,  W/8]
```

Each `ResidualBlock` is two 3×3 convolutions with InstanceNorm and ReLU, plus a residual connection. When the channel count changes, a 1×1 projection handles the skip connection.

**FiLM conditioning points:**
FiLM is applied after each residual stage (3 scales), modulating visual features with IMU ego-motion information before they enter the next stage:

```
FiLM after stage 1:  FiLM(imu_dim=128, feature_channels=64)    → 16,512 params
FiLM after stage 2:  FiLM(imu_dim=128, feature_channels=96)    → 24,672 params
FiLM after stage 3:  FiLM(imu_dim=128, feature_channels=128)   → 32,896 params
                                                        Total:    74,080 params
```

This is less than V1's MultiscaleFiLM (~100K) because we apply to 3 stages of a narrower encoder.

**Feature encoder parameter count:**
```
Stem:     Conv(3, 64, 7×7)                       =   9,408
Stage 1:  2 × ResBlock(64, 64)                    = 147,456 + norms
Stage 2:  ResBlock(64, 96, stride=2) + ResBlock(96, 96) = ~200K + norms
Stage 3:  ResBlock(96, 128, stride=2) + ResBlock(128, 128) = ~350K + norms
Output:   Conv(128, 128, 1×1)                     =  16,384
──────────────────────────────────────────────────────────────
Feature encoder total:                              ~740K params
```

**Two encoders, separate weights:**
```
Feature encoder (fnet): ~740K params  — called on BOTH frames (shared weights)
Context encoder (cnet): ~740K params  — called on frame t ONLY (separate weights)
──────────────────────────────────────────────────────────────────────────
Total backbone:         ~1,480K params  (vs 1,800K for DDRNet-23-slim → 18% lighter)
```

**Pretrained weight loading:**
```python
# Load from RAFT checkpoint (e.g. raft-things.pth)
raft_ckpt = torch.load("raft-things.pth", map_location="cpu")

# RAFT checkpoint keys: "module.fnet.*", "module.cnet.*", "module.update_block.*"
fnet_state = {k.replace("module.fnet.", ""): v
              for k, v in raft_ckpt.items() if k.startswith("module.fnet.")}
cnet_state = {k.replace("module.cnet.", ""): v
              for k, v in raft_ckpt.items() if k.startswith("module.cnet.")}

model.feature_encoder.load_state_dict(fnet_state, strict=False)  # FiLM layers will be missing → OK
model.context_encoder.load_state_dict(cnet_state, strict=False)
```

The `strict=False` is needed because our feature encoder has FiLM layers that don't exist in the original RAFT checkpoint. These initialise as identity (gamma=1, beta=0) by design, so the encoder behaves identically to pretrained RAFT at the start of training.

**Available pretrained checkpoints (all public on the RAFT GitHub repo):**
| Checkpoint | Training data | Best for |
|-----------|--------------|----------|
| `raft-chairs.pth` | FlyingChairs (synthetic) | Not recommended — too simple |
| `raft-things.pth` | FlyingChairs → FlyingThings3D (photorealistic synthetic) | **Recommended starting point** |
| `raft-sintel.pth` | + Sintel (photorealistic synthetic, complex motion) | Best synthetic features |
| `raft-kitti.pth` | + KITTI (real outdoor driving) | Best if target domain is driving |
| `raft-small.pth` | FlyingThings3D, small architecture | If even lighter model needed |

Recommendation: **start with `raft-things.pth`** (broadest synthetic pretraining, no domain bias toward driving or cinematic motion).

### 5.2 RAFT Context Encoder

Same architecture as the feature encoder but with separate weights. Called on frame t only. Its output is split into two parts:

```
context_output = context_encoder(img_curr)  # [B, 128, H/8, W/8]

# Split via 1×1 convolutions (following RAFT's design)
net = tanh(conv_net(context_output))        # [B, 128, H/8, W/8]  — GRU hidden init
inp = relu(conv_inp(context_output))        # [B, 128, H/8, W/8]  — motion context
```

- `net`: initialises the ConvGRU hidden state (replaces V1's zero initialisation)
- `inp`: provides scene context to the motion encoder at each GRU iteration

### 5.3 Flow Decoder (RAFT Update Operator)

Follows RAFT's update operator architecture. Operates at 1/8 resolution.

**Correlation pyramid (replaces V1's single-level local correlation):**

RAFT computes an all-pairs correlation volume between the two feature maps, then builds a 4-level pyramid by average-pooling. At each GRU iteration, it looks up local correlation at each pyramid level around the current flow estimate.

```
# All-pairs correlation at 1/8 resolution
# For 480×640 input: features are 60×80 = 4,800 pixels
# Correlation volume: 4800 × 4800 × 4 bytes = 92 MB (FP32) or 46 MB (FP16)
# This is computed once and reused across GRU iterations

corr_volume = einsum('bchw,bcij->bhwij', feat_prev, feat_curr)  # [B, H/8, W/8, H/8, W/8]

# Build 4-level pyramid by pooling the last two dims
corr_pyramid = [corr_volume]
for _ in range(3):
    corr_pyramid.append(avg_pool2d(corr_pyramid[-1], kernel_size=2))

# At each GRU iteration, lookup radius-4 neighborhood at each level:
# Level 0: 9×9 = 81 values  (fine)
# Level 1: 9×9 = 81 values  (2× coarser)
# Level 2: 9×9 = 81 values  (4× coarser)
# Level 3: 9×9 = 81 values  (8× coarser)
# Total: 324 correlation features per pixel
```

**Note on memory:** The all-pairs correlation (46 MB at FP16 for 480×640) fits in budget. For higher resolutions or tighter memory, we can fall back to local-only correlation (same as V1 but at 1/8), which uses <1 MB.

**GRU update loop:**

```
flow = zeros([B, 2, H/8, W/8])
h = net  # from context encoder, [B, 128, H/8, W/8]

for i in range(3):                                       # 3 iterations (RAFT default: 12)
    corr_features = corr_pyramid_lookup(flow)            # [B, 324, H/8, W/8]
    motion_feat = MotionEncoder(corr_features, flow)     # [B, 128, H/8, W/8]
    gru_input = torch.cat([inp, motion_feat], dim=1)     # [B, 256, H/8, W/8]
    h = SepConvGRU(gru_input, h)                         # [B, 128, H/8, W/8]
    delta_flow = FlowHead(h)                             # [B, 2, H/8, W/8]
    flow = flow + delta_flow

# Output: flow at 1/8 resolution (internal), mask from final h
```

**Output heads:**

```
FlowHead (INTERNAL — not exported):
  Conv(128, 256, 3×3) + ReLU
  Conv(256, 2, 3×3)
  Output: [B, 2, H/8, W/8]  — used by BA at training, discarded at inference

MaskHead (EXPORTED):
  Conv(128, 64, 3×3) + ReLU
  Conv(64, 1, 1×1)
  GradientClip()         ← per-element ±0.01 backward clamp (from DPVO blocks.py:74-82)
  Output: sigmoid(upsample([B, 1, H/8, W/8])) → [B, 1, H, W]
```

**Why flow must still be computed internally even though it's not exported:**
The flow drives the GRU iterations. Each iteration uses the current flow estimate to look up correlation features at the right locations. Without flow, the GRU hidden state would be meaningless — the mask head would have nothing useful to decode. Flow is the internal mechanism; mask is the external product.

**Flow decoder parameter counts:**
```
MotionEncoder:
  Conv(324, 128, 3×3) + ReLU              = 373,248 + 128  = 373,376
  Conv(128, 128, 3×3) + ReLU              = 147,456 + 128  = 147,584
                                             subtotal:  ~521K

SepConvGRU (separable, 256 input, 128 hidden):
  Horizontal GRU:
    conv_z: Conv1d(256+128, 128, 1×5)     = 245,760
    conv_r: Conv1d(256+128, 128, 1×5)     = 245,760
    conv_h: Conv1d(256+128, 128, 1×5)     = 245,760
  Vertical GRU:
    conv_z: Conv1d(256+128, 128, 5×1)     = 245,760
    conv_r: Conv1d(256+128, 128, 5×1)     = 245,760
    conv_h: Conv1d(256+128, 128, 5×1)     = 245,760
                                             subtotal:  ~1,474K
                                             
  (This is larger than V1's ConvGRU because RAFT uses separable GRU
   for better spatial reasoning. Can reduce to standard ConvGRU ~500K 
   if budget is tight — see Section 6 budget analysis.)

FlowHead (internal):
  Conv(128, 256, 3×3) + ReLU              = 295,168
  Conv(256, 2, 3×3)                       = 4,610
                                             subtotal:  ~300K

MaskHead (exported):
  Conv(128, 64, 3×3) + ReLU               = 73,792
  Conv(64, 1, 1×1)                         = 65
  GradientClip (0 params, clamps grads)    = 0
                                             subtotal:  ~74K

Context split projections:
  conv_net: Conv(128, 128, 1×1)            = 16,512
  conv_inp: Conv(128, 128, 1×1)            = 16,512
                                             subtotal:  ~33K

Flow Decoder Total:                                    ~2,402K params
```

### 5.4 Why Flow Is Not an Output

Flow is computed internally to drive the iterative GRU refinement and to provide correspondences for the training-time BA. But it is not exported for inference because:

1. **Backends don't consume it.** OpenVINS, ORB-SLAM3, and VINS-Mono all run their own feature tracking (ORB, KLT, etc.). They need a mask to filter features, not a competing flow field. Feeding them external flow would require deep backend modification.

2. **Lighter ONNX export.** Flow at full resolution is [2, 480, 640] = 614K floats. Not huge, but unnecessary bandwidth on mobile.

3. **Interface stability.** V1 outputs {mask, δb, σ², ΔR, Δv, Δp, Σ}. V2 outputs the same (minus flow_residual, which was auxiliary and not consumed by any backend). Zero changes needed downstream.

4. **Clean separation of concerns.** The model's job is to tell the backend "which pixels to trust" and "what the IMU says." Flow is an implementation detail of how the model determines this, not a product.

The flow decoder still runs at inference — it's what makes the mask good. The flow tensor is simply not included in the ONNX export's output list.

### 5.5 Flow-Derived Dynamic Mask

The mask head operates on the final GRU hidden state `h`, which encodes:
- Correlation patterns (what moved where)
- Flow residual structure (after iterative refinement)
- IMU-conditioned context (via FiLM on feature encoder)

The GRU hidden state `h` implicitly represents "how well this pixel's motion is explained by a single rigid ego-motion." The mask head learns to decode this into a dynamic probability. No explicit flow decomposition formula is needed — the network learns the decomposition end-to-end through the BA training signal.

**Why not explicitly compute ego-flow and subtract?**
Computing ego-flow from IMU rotation requires camera intrinsics K (to map rotation to pixel displacements). At inference we don't have K. The FiLM conditioning gives the network the IMU information; the network learns the implicit decomposition itself.

**Why a separate mask head instead of thresholding flow magnitude?**
Flow magnitude alone is a poor mask signal. A camera translating forward produces large flow at image edges (static scene parallax). A slowly-moving person in the distance produces small flow. The mask head learns to account for these geometric effects through the BA training signal.

### 5.6 Differentiable Two-Frame Bundle Adjustment (Training Only)

This module has **zero learnable parameters**. It is a differentiable computation graph that:
1. Takes static correspondences + camera intrinsics
2. Optimises a relative camera pose
3. Returns the optimised pose, through which gradients flow back to the mask and flow predictors

The design is informed by DPVO's (Teed et al., NeurIPS 2023) differentiable BA, which demonstrated that tightly coupling a learned network with geometric optimisation is trainable end-to-end — provided specific safeguards are in place against early-training instability. Our BA is a simplified two-frame specialisation of their multi-frame system.

**Input:**
```
correspondences: [B, K, 4]  — (u, v, u', v') for K selected static points
weights:         [B, K]     — soft static confidence (1 - mask_prob)
K_matrix:        [B, 3, 3]  — camera intrinsics (from dataset, NOT a model input)
R_init:          [B, 3, 3]  — IMU preintegrated rotation (warm start)
epoch:           int        — current training epoch (controls adaptive damping)
```

**Algorithm: Robustified Weighted Gauss-Newton on SE(3)**

```python
class SafeCholeskySolver(torch.autograd.Function):
    """
    Differentiable Cholesky solve with failure recovery.
    Adopted from DPVO (dpvo/ba.py:12-37): if the Hessian is singular or
    near-singular (degenerate correspondence configuration), returns zero
    update and zero gradient. This prevents NaN propagation from a single
    bad sample from corrupting the entire batch.
    """
    @staticmethod
    def forward(ctx, H, b):
        U, info = torch.linalg.cholesky_ex(H)
        if torch.any(info):
            ctx.failed = True
            return torch.zeros_like(b)
        xs = torch.cholesky_solve(b.unsqueeze(-1), U).squeeze(-1)
        ctx.save_for_backward(U, xs)
        ctx.failed = False
        return xs

    @staticmethod
    def backward(ctx, grad_x):
        if ctx.failed:
            return None, None
        U, xs = ctx.saved_tensors
        dz = torch.cholesky_solve(grad_x.unsqueeze(-1), U).squeeze(-1)
        dH = -torch.einsum('bi,bj->bij', xs, dz)
        return dH, dz


class GradientClip(torch.autograd.Function):
    """
    Per-element gradient clamp (identity in forward, clamp in backward).
    Adopted from DPVO (dpvo/blocks.py:74-82) which clamps to [-0.01, 0.01].
    This is far more protective than global norm clipping: even if the BA
    produces a bad pose, the gradient signal reaching the network is bounded
    per-element, preventing any single outlier sample from causing a large
    parameter update.
    """
    @staticmethod
    def forward(ctx, x):
        return x
    @staticmethod
    def backward(ctx, grad_x):
        grad_x = torch.where(torch.isnan(grad_x), torch.zeros_like(grad_x), grad_x)
        return grad_x.clamp(min=-0.01, max=0.01)


def differentiable_ba(correspondences, weights, K, R_init, epoch, phase2_start,
                      n_iters=3, base_damping=1e-4, outlier_threshold=100.0):
    """
    Estimates relative pose (R, t) from weighted correspondences.
    Fully differentiable — gradients flow back through correspondences and weights.

    Key differences from naive Gauss-Newton (informed by DPVO):
      1. Hard outlier rejection (DPVO uses 250px threshold for multi-frame;
         we use 100px for two-frame short-baseline)
      2. SafeCholeskySolver with failure recovery (zero update + zero grad on singular H)
      3. Adaptive damping: ep=10.0 early in Phase 2, decaying to 1.0 over 10 epochs
         (DPVO uses ep=10.0 throughout training — extremely conservative)
      4. Per-element gradient clipping on outputs via GradientClip
      5. Depth validity check (Z > 0.2) before including a point in normal equations

    Note on iteration count: DPVO uses 2 GN iterations inside a larger 18-step
    network-BA loop. Our two-frame BA is standalone, so we use 3 iterations —
    sufficient because the IMU warm-start puts us near the solution.
    """
    B, K_pts, _ = correspondences.shape

    # Adaptive damping: high early in Phase 2 to prevent wild updates,
    # decaying to standard value as the mask head improves.
    # DPVO uses a fixed ep=10.0 throughout training (dpvo/net.py:258).
    epochs_into_phase2 = max(0, epoch - phase2_start)
    ep = max(1.0, 10.0 * (1.0 - epochs_into_phase2 / 10.0))

    # Initialise pose: rotation from IMU, translation as unit forward
    R = R_init.clone()
    t = torch.tensor([0., 0., 1.], device=R.device).expand(B, 3).clone()

    # Extract correspondences
    uv1 = correspondences[:, :, :2]   # [B, K, 2] — frame t-1
    uv2 = correspondences[:, :, 2:4]  # [B, K, 2] — frame t

    # Bearing vectors (normalised image coordinates)
    K_inv = torch.linalg.inv(K)
    ones = torch.ones(B, K_pts, 1, device=K.device)
    p1_h = torch.cat([uv1, ones], dim=-1)  # [B, K, 3]
    p2_h = torch.cat([uv2, ones], dim=-1)

    p1 = torch.bmm(p1_h, K_inv.transpose(-1, -2))  # [B, K, 3]
    p2 = torch.bmm(p2_h, K_inv.transpose(-1, -2))

    for i in range(n_iters):
        # Triangulate 3D points via DLT (midpoint method)
        X = triangulate_midpoint(p1, p2, R, t)  # [B, K, 3]

        # Reproject into frame 2
        X_2 = torch.bmm(X, R.transpose(-1, -2)) + t.unsqueeze(1)  # [B, K, 3]
        proj_2 = torch.bmm(X_2, K.transpose(-1, -2))               # [B, K, 3]
        Z = X_2[:, :, 2]                                            # [B, K]
        proj_2_px = proj_2[:, :, :2] / (proj_2[:, :, 2:3] + 1e-8)  # [B, K, 2]

        # Reprojection error
        err = proj_2_px - uv2  # [B, K, 2]

        # ── Hard outlier rejection (DPVO-style) ──
        # DPVO (dpvo/ba.py:98): v *= (r.norm(dim=-1) < 250).float()
        # DPVO (dpvo/fastba/ba_cuda.cu:305): sqrt(rx*rx+ry*ry) < 128
        # We use 100px for our short-baseline two-frame case.
        err_norm = err.norm(dim=-1)                            # [B, K]
        valid = (err_norm < outlier_threshold) & (Z > 0.2)     # [B, K] bool
        valid_f = valid.float()                                 # [B, K]

        # Effective weights: soft mask confidence × hard validity
        w_eff = weights * valid_f                               # [B, K]

        # Compute Jacobian of reprojection w.r.t. pose (6-DoF)
        J = compute_reprojection_jacobian(X_2, K)  # [B, K, 2, 6]

        # Weighted least squares: solve (J^T W J + λI) δξ = -J^T W e
        JtWJ = torch.einsum('bkij,bk,bkil->bjl', J, w_eff, J)    # [B, 6, 6]
        JtWe = torch.einsum('bkij,bk,bki->bj', J, w_eff, err)    # [B, 6]

        # Damped normal equations
        # DPVO (dpvo/ba.py:73): A + (ep + lm * A) * I  with ep=10
        # DPVO (dpvo/fastba/ba_cuda.cu:546): S += I * (1e-4 * S + 1.0)
        JtWJ += (ep + base_damping * JtWJ) * torch.eye(6, device=JtWJ.device)

        # SafeCholeskySolver: returns zero update on singular H
        delta_xi = SafeCholeskySolver.apply(JtWJ, -JtWe)  # [B, 6]

        # Per-element gradient clamp on the update
        delta_xi = GradientClip.apply(delta_xi)

        # Update pose via exponential map
        R, t = se3_exp_update(R, t, delta_xi)

    # ── Convergence sanity check ──
    # If mean reprojection error is still > 50px after all iterations,
    # the BA failed to converge (bad correspondences). Detach the pose
    # so this sample contributes zero gradient to L_pose.
    mean_err = err.norm(dim=-1).mean(dim=-1)       # [B]
    converged = (mean_err < 50.0).float()           # [B]
    # Detach pose for non-converged samples (gradient stops here)
    R = R * converged.view(B, 1, 1) + R.detach() * (1 - converged.view(B, 1, 1))
    t = t * converged.view(B, 1)    + t.detach() * (1 - converged.view(B, 1))

    return R, t, err, converged
```

**Why 3 GN iterations, not 5:**
DPVO uses only 2 GN iterations per network-BA step (inside an 18-step loop where the network re-predicts targets each time). Our BA is standalone, but with IMU warm-starting the rotation, 3 iterations are sufficient — the initial pose is already close to the solution. Using fewer iterations also reduces the risk of the solver wandering off during early training when correspondences are noisy.

**Backward pass — unrolled with SafeCholeskySolver (not implicit differentiation):**

The original design proposed implicit differentiation at convergence. After studying DPVO's approach, we use **unrolled differentiation through the GN iterations** via `SafeCholeskySolver`, for two reasons:

1. **DPVO demonstrates this works.** DPVO unrolls 2 GN iterations and differentiates through them via `CholeskySolver` (a custom autograd function with Cholesky forward + analytic backward). This is simpler to implement, debug, and does not require the BA to actually converge (early in training, 3 iterations may not reach the optimum, so the implicit function theorem's convergence assumption is violated).

2. **3 iterations is cheap to unroll.** The memory cost of storing activations for 3 GN iterations on a 6-DoF system is negligible. Implicit differentiation's memory advantage matters for DROID-SLAM's 100+ pose system — not for our single relative pose.

The `SafeCholeskySolver` autograd function handles the backward pass:
```
Forward:  H δξ = b  →  Cholesky decompose H = U U^T, solve via cholesky_solve
Backward: dH = -δξ @ (U^{-T} U^{-1} grad_x)^T,  db = U^{-T} U^{-1} grad_x
Failure:  if Cholesky fails → return zero update (forward) and zero gradient (backward)
```

This failure recovery is critical: it means a single degenerate sample (e.g., all correspondences are collinear) produces zero gradient rather than NaN, and training continues normally.

**Memory cost of BA during training:**
```
K = 256 selected points per frame pair
Jacobian: [B, 256, 2, 6]  = 3072 × B floats  ≈  negligible
Hessian:  [B, 6, 6]       = 36 × B floats    ≈  negligible
3D points: [B, 256, 3]    = 768 × B floats   ≈  negligible
Cholesky factor: [B, 6, 6] × 3 iters saved for backward ≈ negligible
```
The BA adds effectively zero memory overhead.

### 5.7 Self-Supervised Loss Functions

**No GT masks. No GT flow. Only: images, IMU, camera intrinsics (training datasets), camera-IMU extrinsic.**

```
L_total = λ_pose · L_pose  +  λ_photo · L_photo  +  λ_reproj · L_reproj
        + λ_reg  · L_reg   +  λ_imu   · L_imu    +  λ_smooth · L_smooth
```

#### L_pose — Pose Consistency (primary self-supervised signal)

The BA-estimated pose should agree with the IMU-preintegrated pose.

```
L_pose = w_R · ‖Log(R_BA^T · R_IMU)‖²  +  w_t · ‖t_BA/‖t_BA‖ - t_IMU/‖t_IMU‖‖²
```

Note: translation direction only (not magnitude), since monocular BA has scale ambiguity and IMU preintegrated translation over one frame pair is noisy. Rotation comparison is exact since both sources estimate absolute rotation.

**Why this works:** If the mask misses a dynamic object, the BA will fit the static+dynamic correspondences to a single rigid motion. This produces a pose that disagrees with the IMU (which accurately measures ego-rotation). The disagreement generates gradients that push the mask to exclude the dynamic object.

#### L_photo — Photometric Consistency on Static Regions

Warp frame t-1 to frame t using predicted flow (internal), and measure photometric error on predicted-static regions. Dynamic regions should have high photometric error after ego-motion warping.

```
warp = bilinear_sample(img_prev, flow)
photo_err = ‖warp - img_curr‖₁  (robust L1, per-pixel)

L_photo = mean(photo_err × (1 - mask))  # only penalise on static regions
        + mean(max(0, τ - photo_err) × mask)  # dynamic regions SHOULD have high error
```

The second term prevents the trivial solution of masking everything as dynamic.

#### L_reproj — BA Reprojection Error

The BA's own reprojection error on static correspondences should be low.

```
L_reproj = weighted_mean(‖reproj_error‖², weights=static_confidence)
```

This loss drives the flow to produce accurate correspondences on static regions and drives the mask to exclude points that create high reprojection error.

#### L_reg — Mask Regularisation

Prevent trivial solutions:
```
mask_ratio = mean(mask)  # fraction of image predicted as dynamic

# Should be between 5% and 60% (most scenes are mostly static)
L_reg = max(0, 0.05 - mask_ratio)² + max(0, mask_ratio - 0.6)²

# Spatial smoothness (connected regions, not salt-and-pepper)
L_reg += λ_tv · TV(mask)  # total variation
```

#### L_imu — IMU Preintegration Error (from V1, unchanged)

Supervise the IMU encoder's preintegration against ground truth pose (when available in dataset).

```
L_imu = imu_integration_loss(pred_R, pred_v, pred_p, gt_R, gt_v, gt_p)
      + λ_cov · covariance_nll_loss(...)
```

Identical to V1. Ensures the IMU encoder produces good ego-motion estimates → good FiLM conditioning → better flow and mask.

#### L_smooth — Flow Smoothness

Edge-aware flow smoothness (flow should be smooth except at image edges):
```
L_smooth = mean(|∇flow| × exp(-|∇img|))
```

Standard in self-supervised optical flow literature (UnFlow, DDFlow).

#### Loss Weights (defaults)

```yaml
loss:
  lambda_pose: 1.0       # primary self-supervised signal (ramped 0→1 over epochs 31-36)
  lambda_photo: 0.5      # photometric consistency (active from Phase 2a, epoch 21)
  lambda_reproj: 0.3     # BA reprojection quality (active from Phase 2b, epoch 31)
  lambda_reg: 0.1        # mask regularisation (active from Phase 2a, epoch 21)
  lambda_imu: 1.0        # IMU preintegration (from V1, active all phases)
  lambda_smooth: 0.2     # flow smoothness (active from Phase 2a, epoch 21)
  pose_rotation_weight: 5.0
  pose_direction_weight: 1.0
  pose_ramp_epochs: 5    # epochs to ramp L_pose from 0 to lambda_pose

ba:
  n_iters: 3             # GN iterations (DPVO uses 2 inside 18-step loop)
  base_damping: 1e-4     # multiplicative LM damping
  ep_initial: 10.0       # additive damping at Phase 2b start (DPVO uses 10.0 fixed)
  ep_final: 1.0          # additive damping at steady state
  ep_decay_epochs: 10    # epochs to decay ep from initial to final
  outlier_threshold: 100.0  # px; hard rejection (DPVO uses 250 for multi-frame)
  convergence_threshold: 50.0  # px; mean reproj error above this → detach BA pose
  min_static_points: 64  # skip BA loss if fewer than this many static correspondences
  num_correspondences: 256  # K: number of static correspondences fed to BA
```

**Phase-specific loss activation:**
```
Phase 1  (epochs 1-20):   L_imu only
Phase 2a (epochs 21-30):  L_photo + L_smooth + L_reg + L_imu
                          (BA runs but pose is detached — no L_pose, no L_reproj gradient)
Phase 2b (epochs 31-70):  ramp * L_pose + L_photo + L_reproj + L_reg + L_imu + L_smooth
Phase 3  (epochs 71-80):  L_pose + L_photo + L_reproj + L_reg + L_imu + L_smooth (ramp=1)
```

---

## 6. Memory Budget

### 6.1 Inference (target: < 1 GB)

**Model parameters (FP16):**
```
RAFT Feature Encoder (fnet):     740K params ×  2B  =   1.5 MB
RAFT Context Encoder (cnet):     740K params ×  2B  =   1.5 MB
IMU Encoder:                      55K params ×  2B  =   0.1 MB
FiLM (3 stages on fnet):          74K params ×  2B  =   0.1 MB
Flow Decoder (update operator): 2,402K params ×  2B  =   4.8 MB
────────────────────────────────────────────────────────────────
Total model:                    4,011K params          =   8.0 MB
```

**Peak activation memory (FP16, batch=1, 480×640 input):**
```
Input images (2 frames):          2 × 3 × 480 × 640 × 2B  =    3.7 MB
Feature maps fnet (2 frames):     2 × 128 × 60 × 80 × 2B  =    2.5 MB
Context maps cnet (1 frame):      1 × 128 × 60 × 80 × 2B  =    1.2 MB
Correlation volume (all-pairs):   60×80 × 60×80 × 2B       =   46.1 MB
Correlation pyramid (pooled):     ~25% overhead             =   11.5 MB
GRU hidden state:                 128 × 60 × 80 × 2B       =    1.2 MB
GRU iteration intermediates:      ~3 × 128 × 60 × 80 × 2B  =    3.7 MB
Motion encoder intermediates:     128 × 60 × 80 × 2B        =    1.2 MB
Flow (internal, 1/8 only):       2 × 60 × 80 × 2B          =    0.02 MB
Mask (1/8 + full res):           (1×60×80 + 1×480×640)×2B   =    0.6 MB
IMU intermediates:                                           =    1.0 MB
────────────────────────────────────────────────────────────────────
Peak activations:                                           ≈   72.7 MB
```

**Total inference memory:**
```
CUDA context overhead:           ~300 MB  (unavoidable, one-time)
Model parameters:                   8 MB
Peak activations:                  73 MB
PyTorch allocator overhead:       ~50 MB
────────────────────────────────────────
Total:                           ~431 MB  ✓  well under 1 GB
```

With ONNX Runtime (no CUDA context, no PyTorch allocator), inference drops to ~100 MB.

**If memory is tighter — fallback options:**
| Change | Memory saved | Quality impact |
|--------|-------------|----------------|
| Local correlation (radius=4) instead of all-pairs pyramid | ~55 MB activations | Moderate — loses multi-scale matching |
| Standard ConvGRU instead of SepConvGRU | ~1 MB params (974K fewer) | Minor — RAFT-Small uses standard GRU and works fine |
| RAFT-Small encoders instead of BasicEncoder | ~1 MB params (1M fewer) | Moderate — weaker features, fewer channels |
| 2 GRU iterations instead of 3 | ~4 MB activations | Minor — 2 iterations is sufficient for short-baseline pairs |

With all fallbacks applied: ~280 MB total. Even a mobile GPU can handle this.

### 6.2 Training

```
Forward activations:              73 MB   (same as inference)
Backward gradient storage:      ~220 MB   (activations saved for backprop)
Optimizer states (Adam, FP32):    32 MB   (2× model params in FP32)
Differentiable BA:                <1 MB   (sparse, 256 points)
Batch of 8:                      ~2.3 GB  (activations × 8 + gradients)
────────────────────────────────────────────
Training peak:                  ~2.6 GB   on a single GPU (FP16 mixed precision)
```

Fits on any modern GPU (even 4GB with batch=2).

---

## 7. Training Pipeline

### 7.1 Phased Training

**Phase 1 — IMU Encoder Pretraining (20 epochs)**
- Dataset: EuRoC (real IMU + GT poses, all static scenes)
- Loss: L_imu only
- Freeze: feature encoder, context encoder, flow decoder, mask head
- Purpose: learn accurate bias corrections and preintegration before coupling with vision
- Identical to V1.

**Phase 2 — Joint Self-Supervised Training (50 epochs)**

Phase 2 is internally divided into two sub-phases to handle the bootstrap problem — the period when the mask head is untrained and predicts near-random probabilities, making the BA's input correspondences unreliable. This design is informed by DPVO's training strategy, where the first 1000 steps freeze poses to ground truth and only optimise structure (`structure_only=True` in `train.py:80`), letting the network learn what good predictions look like before coupling them with geometric optimisation.

We cannot directly copy DPVO's approach (freeze poses to GT) because our mask head — not the BA — is the primary learned component. Instead, we phase in the BA signal gradually:

**Phase 2a — Flow + Photometric Warmup (epochs 21-30, 10 epochs)**
- Dataset: VIODE + TartanAir (80% static/low-dynamic scenes; see curriculum below)
- Loss: L_photo + L_smooth + L_reg + L_imu (NO L_pose, NO L_reproj)
- The BA runs but its pose output is **detached** — no gradient flows from BA to the network
- The BA pose is logged for monitoring (watch convergence rate and pose-IMU agreement) but does not influence learning
- Purpose: the flow decoder and mask head learn basic motion patterns from photometric signal alone. By epoch 30, the mask head has seen thousands of frames and produces spatially coherent (if imperfect) dynamic probabilities — no longer random noise.
- This mirrors DPVO's warmup: DPVO's `structure_only` phase lets the network produce meaningful correlation-based predictions before those predictions influence pose estimation. Our Phase 2a lets the mask head produce meaningful static/dynamic separation before that separation influences the BA training signal.

**Phase 2b — Full Self-Supervised Training (epochs 31-70, 40 epochs)**
- Dataset: VIODE + TartanAir (progressive curriculum, see below)
- Loss: `λ_pose_ramp * L_pose` + L_photo + L_reproj + L_reg + L_imu + L_smooth
- The BA is fully connected — gradients flow from L_pose through the BA, through the mask and flow decoder, into the encoders
- `λ_pose_ramp` warms up linearly from 0.0 to `λ_pose` over the first 5 epochs of Phase 2b (epochs 31-36):
  ```python
  ramp = min(1.0, (epoch - 30) / 5.0)
  loss += ramp * lambda_pose * L_pose
  ```
  This prevents a sudden gradient shock at the transition. DPVO achieves a similar effect by only enabling pose loss after step 2 of each sample's 18-step optimisation loop (`train.py:116: if not so and i >= 2`), giving depth estimates time to stabilise before they influence the pose loss.
- BA damping decays from ep=10.0 to ep=1.0 over the first 10 epochs of Phase 2b (see Section 5.6)
- Train: all parameters (IMU encoder at 0.1× learning rate)
- RAFT encoders fine-tune from pretrained weights (0.5× learning rate to preserve flow features)
- The BA uses dataset camera intrinsics K (available in all training datasets)
- GT poses used as additional supervision for L_imu, and as a sanity-check for L_pose
- Key: VIODE's GT segmentation masks are **NOT** used for supervision. They are only used for evaluation (computing mask IoU to track training progress).

**Phase 3 — Fine-Tuning (10 epochs)**
- Dataset: VIODE dynamic sequences only
- Loss: same as Phase 2b (full loss, ramp=1.0), lower learning rate (cosine decay)
- BA damping at steady-state value (ep=1.0)
- Focus: refine mask quality on challenging high-dynamic scenes

**Summary of bootstrap timeline:**

```
Phase 1 (epochs 1-20):    IMU only. No vision, no BA.
Phase 2a (epochs 21-30):  Vision + IMU, photometric loss. BA runs but is DETACHED.
                          Mask head learns basic motion patterns.
Phase 2b (epochs 31-70):  Full self-supervision. BA gradient enabled.
                          L_pose ramps 0→1 over epochs 31-36.
                          BA damping ep decays 10→1 over epochs 31-41.
Phase 3 (epochs 71-80):   Fine-tune on hard dynamic scenes.
```

### 7.2 Curriculum Strategy

Start with "easy" scenes and progress to harder ones:

```
Phase 2a (epochs 21-30):   80% static/low-dynamic,  20% medium-dynamic
Phase 2b early (31-45):    50% low-dynamic,          50% medium/high-dynamic
Phase 2b late  (45-70):    20% low-dynamic,          80% high-dynamic
Phase 3  (epochs 71-80):   100% high-dynamic (VIODE only)
```

This prevents the mask from collapsing to all-static early in training when the flow decoder hasn't converged yet. The curriculum is especially important during Phase 2a: in mostly-static scenes, the mask head's predictions are less critical (almost all correspondences are static regardless of what the mask says), so the photometric loss provides a clean learning signal even with an untrained mask.

**Why this curriculum matters for BA stability:**
In a scene with 5% dynamic pixels, even a random mask that selects 256 "static" correspondences will accidentally get ~95% correct. The BA can easily converge with 5% outliers, especially with damping ep=10.0 and hard outlier rejection at 100px. By the time the curriculum introduces 50%+ dynamic scenes (epoch 45+), the mask head has had 25 epochs of training and produces meaningful predictions.

### 7.3 Key Training Details

#### 7.3.1 Static Correspondence Selection

- From the predicted mask, select pixels with dynamic probability < 0.3
- From the predicted flow (internal), extract pixel correspondences: (u,v) → (u + flow_x, v + flow_y)
- Sample K=256 points uniformly across the image (avoid clustering)
- Use soft weights (1 - mask_prob) rather than hard selection for gradient flow
- Minimum K=64: if fewer than 64 static points, skip BA loss for this sample

#### 7.3.2 BA Bootstrap Safety Mechanisms

The following mechanisms work together to ensure stable training from the moment the BA is first connected (epoch 31). Each is informed by a specific mechanism in DPVO's training:

| Mechanism | Our Implementation | DPVO Reference | Why It Matters |
|-----------|-------------------|----------------|----------------|
| **Phase 2a warmup** | BA detached for 10 epochs; mask learns from L_photo first | `structure_only=True` for first 1000 steps (`train.py:80`) | Mask head sees thousands of frames before its predictions influence the BA. At epoch 31, mask probs are spatially coherent, not random. |
| **L_pose ramp** | 0 → λ_pose over 5 epochs | Pose loss disabled for first 2 steps per sample (`train.py:116`) | Prevents gradient shock when BA signal is first connected. |
| **SafeCholeskySolver** | Zero update + zero gradient on Cholesky failure | `CholeskySolver` returns `torch.zeros_like(b)` on failure (`ba.py:19-20`) | Singular or near-singular Hessians (degenerate correspondences) produce no training signal rather than NaN. |
| **Per-element gradient clamp** | `GradientClip` clamps backward gradients to ±0.01 | `GradClip.backward` clamps to ±0.01 (`blocks.py:82`) | Even if BA produces a bad pose on one sample, the gradient reaching the network is bounded per-element. Far more protective than global norm clipping. |
| **Hard outlier rejection** | Points with reproj error > 100px or depth Z < 0.2 zeroed out | `v *= (r.norm(dim=-1) < 250).float()` (`ba.py:98`) + bounds check (`ba.py:100-106`) | Dynamic points that slip through the mask produce large reprojection errors and are automatically excluded from the normal equations. |
| **Adaptive damping** | ep decays from 10.0 to 1.0 over 10 epochs | Fixed `ep=10` throughout training (`net.py:258`) | DPVO uses permanent high damping. We relax it as the mask improves, since our IMU warm-start already constrains the solution. |
| **Convergence check** | Mean reproj error > 50px → detach BA pose from graph | No direct analogue (DPVO's 2-iter BA within 18-step loop is self-correcting) | Safety valve: a BA that converged to garbage contributes zero gradient to L_pose. |
| **Curriculum** | 80% static scenes during Phase 2a | DPVO trains on TartanAir without dynamic objects | In mostly-static scenes, even a random mask produces ~95% correct static correspondences. |

**The fundamental difference from DPVO:**

DPVO's network predicts 2D corrections (`delta`) to reprojected coordinates. At random initialisation, `delta ≈ 0`, so the BA receives `target ≈ current_reprojection` — effectively no new information, and the BA makes small, safe updates. The system is **stable by default** because random network outputs cannot break the BA.

Our mask head at random initialisation predicts `p(dynamic) ≈ 0.5` everywhere (sigmoid of near-zero logits), meaning the static correspondence selector feeds the BA a mix of static and dynamic points with roughly equal weights. This **can** break the BA if left unprotected — hence the multi-layered safety mechanisms above.

The Phase 2a warmup is the single most important addition: it transforms the mask head from "random predictions" to "roughly correct spatial patterns" before the BA training signal is connected. The remaining mechanisms (damping, outlier rejection, gradient clipping, safe solver) handle the residual risk.

#### 7.3.3 BA Warm-Starting

- Always initialise BA rotation from IMU preintegrated ΔR (gyroscope short-term rotation is accurate to < 0.1 deg over a frame pair at 20 Hz)
- Initialise translation as unit vector in forward camera direction `[0, 0, 1]`
- With IMU warm-start, the BA starts near the true solution. Gauss-Newton is a local optimiser — 3 damped iterations from a good initialisation produce a reasonable pose even with imperfect correspondences.
- DPVO does not have an IMU warm-start (it is a pure visual system). Instead, it relies on its motion model (`DAMPED_LINEAR` extrapolation from previous poses). Our IMU prior is strictly stronger.

#### 7.3.4 Gradient Flow Through BA

- `SafeCholeskySolver` for the normal equations (differentiable Cholesky with failure recovery)
- Unrolled differentiation through 3 GN iterations (not implicit differentiation — see Section 5.6 for rationale)
- `GradientClip` on the SE(3) update `delta_xi`: per-element backward clamp to ±0.01
- Global gradient norm clip at 10.0 for the full model (same as DPVO: `train.py:123`)
- `SafeCholeskySolver` failure → zero gradient for that batch element (no NaN propagation)
- Convergence check → detach non-converged BA poses (no bad-gradient propagation)

#### 7.3.5 Per-Element Gradient Clipping on Mask Head Output

The mask head's output passes through a `GradientClip` layer before the sigmoid:

```python
self.mask_head = nn.Sequential(
    nn.Conv2d(128, 64, 3, padding=1),
    nn.ReLU(inplace=True),
    nn.Conv2d(64, 1, 1),
    GradientClip(),     # ← per-element ±0.01 backward clamp
)
# sigmoid applied after, outside the sequential
```

This is adopted directly from DPVO, where both the `delta` (flow correction) and `weight` (confidence) heads include `GradientClip` (`net.py:62-71`). The rationale: BA-derived gradients can be noisy and high-variance, especially early in training. Per-element clamping ensures no single pixel's gradient dominates the mask head's parameter update.

#### 7.3.6 RAFT Encoder Learning Rate

- Feature encoder: 0.5× base learning rate (preserve pretrained flow features)
- Context encoder: 0.5× base learning rate
- IMU encoder: 0.1× base learning rate (preserve Phase 1 pretraining)
- Flow decoder + mask head: 1.0× base learning rate

DPVO uses a single learning rate (8e-5) with OneCycleLR for all parameters. Our multi-group schedule is more conservative because we have pretrained components (RAFT encoders, IMU encoder from Phase 1) that should not be overwritten.

---

## 8. Datasets and Intrinsics

The differentiable BA requires camera intrinsics K during training. All target datasets provide these:

| Dataset | Intrinsics | Resolution | IMU Rate | Dynamic Content |
|---------|-----------|------------|----------|-----------------|
| VIODE | Known K per camera | 640×480 | 200 Hz | Simulated people, multi-level |
| TartanAir V2 | Known K | 640×480 | 200 Hz | Simulated vehicles, people |
| EuRoC | Known K per camera | 752×480 | 200 Hz | Static (for IMU pretraining) |
| Real phone (inference) | **Unknown** | Variable | 100-400 Hz | Arbitrary |

The critical insight: **K is a training-time dependency, not an inference-time dependency.** The trained model never sees K as input — it operates entirely in learned feature space. The BA (which needs K) is discarded after training.

---

## 9. Inference Pipeline (What Ships)

At inference, the pipeline is simple and fast:

```python
class DynaMaskV2Inference:
    """Deployed model. No BA, no intrinsics, no flow output."""

    def __init__(self, onnx_path):
        self.session = ort.InferenceSession(onnx_path)

    def __call__(self, img_prev, img_curr, imu_window):
        """
        Args:
            img_prev:    [1, 3, H, W]  uint8 or float
            img_curr:    [1, 3, H, W]
            imu_window:  [1, N, 7]     (timestamp, ax, ay, az, gx, gy, gz)

        Returns:
            mask:     [H, W]     float in [0,1], threshold at 0.5
            imu_out:  dict with:
                delta_bg:     [N, 3]   gyro bias corrections
                delta_ba:     [N, 3]   accel bias corrections
                sigma2_g:     [N, 3]   per-sample gyro noise variance
                sigma2_a:     [N, 3]   per-sample accel noise variance
                delta_R:      [3, 3]   preintegrated rotation
                delta_v:      [3]      preintegrated velocity
                delta_p:      [3]      preintegrated position
                Sigma_preint: [9, 9]   preintegration covariance
        """
        outputs = self.session.run(None, {
            "img_prev": img_prev,
            "img_curr": img_curr,
            "imu_window": imu_window,
        })
        return parse_outputs(outputs)
```

**Backend integration example (OpenVINS):**
```python
mask, imu_out = dynamask(img_prev, img_curr, imu_window)

# Mask out features in dynamic regions — this is the ONLY change to the backend
static_features = feature_detector.detect(img_curr)
static_features = [f for f in static_features if mask[f.y, f.x] < 0.5]

# Feed IMU quantities (same interface as V1)
openvins.process_frame(img_curr, static_features, imu_out)
```

**No backend changes required.** The output signature matches V1 exactly (mask + IMU quantities). The flow_residual from V1 was never consumed by any backend.

---

## 10. What Changes in the Current Codebase

### Files Modified

| File | Change |
|------|--------|
| `models/backbone.py` | **Rewrite** → RAFT BasicEncoder with FiLM insertion points. Load from `raft-things.pth`. |
| `models/temporal_decoder.py` | **Rewrite** → `models/flow_decoder.py`. RAFT update operator with correlation pyramid, SepConvGRU, flow head (internal), mask head (exported). |
| `models/dynamask.py` | Update wiring: RAFT fnet + cnet replace DDRNet. Forward returns mask + IMU (no flow). |
| `models/film.py` | Update channel dims: 64/96/128 for RAFT stages instead of DDRNet's 64/128/128. |
| `losses/mask_loss.py` | **Delete.** No GT masks. |
| `losses/flow_loss.py` | **Rewrite** → self-supervised photometric + smoothness losses. |
| `configs/default.yaml` | New backbone config, new loss weights, BA config, RAFT checkpoint path. |
| `train.py` | Add BA to training loop (Phase 2b+). Phase 2a/2b split with BA detach logic. Multi-group LR schedule for RAFT encoders. L_pose ramp. BA damping schedule. |
| `evaluate.py` | Keep mask IoU against GT (benchmarking). Add flow EPE if GT flow available. |
| `export_onnx.py` | Export mask + IMU outputs only. Flow is excluded from ONNX graph. |

### Files Added

| File | Purpose |
|------|---------|
| `models/flow_decoder.py` | RAFT update operator: correlation pyramid, SepConvGRU, flow + mask heads. |
| `models/differentiable_ba.py` | Differentiable two-frame BA with SafeCholeskySolver, GradientClip, hard outlier rejection, adaptive damping, convergence check. Training only. |
| `losses/self_supervised.py` | L_pose, L_photo, L_reproj, L_reg, L_smooth. |

### Files Removed

| File | Reason |
|------|--------|
| `losses/mask_loss.py` | No GT masks → no focal loss. |
| `models/temporal_decoder.py` | Replaced by `flow_decoder.py`. |

### New Dependencies

| Package | Purpose |
|---------|---------|
| None new | RAFT encoder is pure PyTorch (Conv2d + InstanceNorm + ReLU). No new dependencies beyond existing PyTorch + PyPose. |

### New Assets

| File | Source |
|------|--------|
| `weights/raft-things.pth` | Download from [RAFT GitHub](https://github.com/princeton-vl/RAFT). ~20 MB. Contains fnet, cnet, and update_block weights. |

---

## 11. Related Work and Novelty

### Direct Precedent

| Paper | Contribution | Gap We Fill |
|-------|-------------|-------------|
| **RAFT** (Teed & Deng, 2020) | Iterative optical flow with correlation pyramid + ConvGRU | No IMU conditioning, no dynamic masking, no VIO integration |
| **DROID-SLAM** (Teed & Deng, 2021) | Differentiable BA as training signal for learned SLAM | No IMU, no dynamic handling, no masking, keeps BA at inference |
| **RigidMask** (Yang & Ramanan, 2021) | Self-supervised rigid motion segmentation from flow | No IMU prior → struggles with ego-motion ambiguity |
| **DPVO/DPV-SLAM** (Teed et al., 2023/24) | Patch-based differentiable visual odometry with learned confidence weights + Schur-complement BA | No IMU, no explicit dynamic masking. **Key training techniques adopted:** SafeCholeskySolver with failure recovery, per-element gradient clipping (±0.01), structure-only warmup for first 1000 steps, high BA damping (ep=10), hard outlier rejection in BA normal equations. |
| **AirIMU** (Qiu et al., 2024) | Learned IMU denoising for VIO | No visual component, no dynamic masking |
| **BA-Net** (Tang & Tan, 2019) | Feature-metric differentiable BA | No IMU, no self-supervised masking |

### Our Novel Combination

1. **FiLM-conditioned RAFT features with IMU ego-motion embeddings.** Nobody has used IMU features to modulate a flow encoder via FiLM. The IMU prior makes ego/dynamic decomposition fundamentally easier than pure vision. Using RAFT's pretrained encoder (trained on real + synthetic flow data) as the backbone gives us matching-quality features from day one.

2. **Self-supervised dynamic masking via differentiable BA + IMU.** The combination of IMU-as-ego-prior + BA-as-geometric-verifier produces a mask training signal that requires no annotations.

3. **Training scaffold architecture.** The differentiable BA, camera intrinsics, and optical flow output are training-time scaffolding. The deployed model is a standard feedforward network that outputs only a mask + IMU quantities — no optimisation loops, no intrinsics, no flow, no backend assumptions. This clean separation doesn't exist in prior work (DROID-SLAM keeps the BA at inference; RAFT outputs flow).

4. **Intrinsic-free inference from intrinsic-dependent training.** The model uses K during training (for BA) but doesn't take K as input. It learns camera-agnostic features that generalise across devices.

5. **DPVO-informed BA training stability.** We adopt and adapt DPVO's battle-tested mechanisms for training through a differentiable BA: SafeCholeskySolver with failure recovery, per-element gradient clipping, high initial damping, hard outlier rejection, and phased warmup. The key adaptation is that DPVO's network outputs 2D corrections (safe at random init — delta≈0 produces near-zero BA updates), while our mask head at random init produces p≈0.5 (unsafe — feeds mixed static/dynamic correspondences to BA). Our Phase 2a warmup compensates for this structural difference by letting the mask head learn from photometric signal before connecting the BA gradient.

---

## 12. Real-World Use Cases

**AR on phones (ARCore / ARKit enhancement):**
Camera + IMU are standard on every phone. Dynamic people in AR scenes corrupt SLAM tracking. Current approach: heuristic feature rejection or pre-trained person segmentation. Our approach: learned geometric masking that handles arbitrary dynamic objects (not just people) without semantic priors. Deploy as an ONNX model (~100 MB inference) that runs before the VIO backend.

**Autonomous driving / ADAS:**
Vehicles have cameras + IMUs. Visual odometry is corrupted by other vehicles, pedestrians, debris, animals. Current state-of-art uses pre-trained object detectors (YOLO, etc.) for masking — but these miss unusual dynamic objects (rolling tires, falling cargo, animals). Our approach handles arbitrary dynamic objects through geometric self-supervision.

**Drone navigation in dynamic environments:**
UAVs with cheap cameras + MEMS IMUs operating in crowds, traffic, or wildlife. Weight and compute constraints make heavy segmentation models impractical. Our model is ~4M params, runs in ~431 MB GPU memory (or ~100 MB with ONNX Runtime), and produces a mask + IMU quantities — everything a lightweight VIO backend needs.

**Service robotics:**
Robots in hospitals, warehouses, sidewalks — environments with moving people, carts, doors. Camera + IMU is the cheapest sensor package. Our model provides dynamic masking without requiring a GPU-heavy segmentation model or domain-specific training data.

**Crowd-sourced mapping (Mapillary, Google Street View):**
Phone data with camera + IMU collected at scale. Dynamic objects (cars, people) must be identified for map building. Semantic segmentation misses novel dynamic objects. Our self-supervised approach generalises to arbitrary dynamic content because it uses geometry, not semantics.

---

## 13. Summary of Key Decisions

| Decision | Rationale |
|----------|-----------|
| RAFT BasicEncoder replaces DDRNet-23-slim | Flow-native features + InstanceNorm + pretrained on real+synthetic flow data; 18% lighter |
| Separate feature + context encoders (RAFT design) | Feature encoder learns matching; context encoder learns structure — different objectives benefit from separate weights |
| FiLM conditioning on RAFT feature encoder | IMU ego-motion prior makes dynamic/static decomposition tractable without intrinsics |
| Flow computed internally, NOT exported | Backends don't consume flow; lighter export; identical interface to V1 |
| Mask-only output (+ IMU quantities) | Zero backend changes required; clean separation of concerns |
| Differentiable BA for training only | BA provides geometric consistency signal; removing at inference eliminates intrinsics dependency |
| Unrolled differentiation (3 GN iters) via SafeCholeskySolver | Simpler than implicit diff; does not assume BA convergence (violated early in training); DPVO validates this approach with 2-iter unrolled BA |
| SafeCholeskySolver with failure recovery | Singular Hessians (degenerate configs) produce zero update + zero gradient instead of NaN — adopted from DPVO's `CholeskySolver` (`ba.py:12-37`) |
| Per-element gradient clamp (±0.01) on BA outputs and mask head | Far more protective than global norm clipping; adopted from DPVO's `GradClip` (`blocks.py:74-82`); bounds worst-case gradient from any single sample |
| Hard outlier rejection (100px) + depth check (Z>0.2) in BA | Dynamic points that slip through the mask produce large reprojection errors and are auto-excluded from normal equations — DPVO uses 250px threshold (`ba.py:98`) + bounds check (`ba_cuda.cu:305`) |
| Adaptive BA damping: ep=10→1 over 10 epochs | DPVO uses fixed ep=10 (extremely conservative); we relax as mask quality improves, since IMU warm-start already constrains the solution |
| Phase 2a: 10-epoch photometric warmup with BA detached | DPVO's `structure_only=True` for first 1000 steps; our analogue lets the mask head learn spatial motion patterns before its predictions influence the BA training signal |
| L_pose ramp: 0→1 over 5 epochs at Phase 2b start | DPVO enables pose loss only after step 2 of each 18-step loop; prevents gradient shock when BA signal is first connected |
| Convergence check: detach BA pose if mean reproj > 50px | Safety valve against non-converged BA producing misleading gradients; no direct DPVO analogue (its 18-step loop is self-correcting) |
| Correlation pyramid (4-level, all-pairs) | Multi-scale matching catches both small and large displacements |
| 3 GRU iterations (vs RAFT's 12) | Sufficient for short-baseline frame pairs; saves 4× compute |
| Warm-start BA from IMU rotation | Breaks bootstrap problem — BA converges even with poor initial mask; strictly stronger than DPVO's motion-model initialisation |
| ~4M total params, ~431 MB inference GPU memory | Fits the ~1 GB budget; deployable on mobile GPUs; ~100 MB with ONNX Runtime |
| Pretrained checkpoint: raft-things.pth | Broadest synthetic pretraining without domain bias toward driving or cinematic motion |
