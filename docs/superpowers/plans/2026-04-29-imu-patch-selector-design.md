# IMU-Conditioned Patch Selector for Deep Patch Visual-Inertial Odometry — Architecture Design

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace DPVO's random patch selection with a learned, IMU-conditioned patch selector that jointly predicts patch trackability, dynamic probability, and per-patch covariance, enabling robust visual-inertial odometry in dynamic scenes.

**Architecture:** A CNN image encoder conditioned by IMU context through complementary FiLM (global, uniform modulation) + CLIP-adapter-gated cross-attention (spatially selective modulation). Three lightweight heads operate on the motion-aware feature map: selectability, dynamic classifier, and covariance. Patches are selected via differentiable Gumbel-Softmax Top-K.

**Tech Stack:** PyTorch, DPVO (BasicEncoder4 backbone + iterative Update + BA), IMUContext (EKF + feature MLP), PyPose (differentiable BA), Cityscapes/KITTI-360 (object patches for semi-synthetic data)

---

## Table of Contents

1. [Motivation & Problem Statement](#1-motivation--problem-statement)
2. [Architecture Overview](#2-architecture-overview)
3. [IMU Conditioning: FiLM + CLIP-Adapter Cross-Attention](#3-imu-conditioning-film--clip-adapter-cross-attention)
4. [Gumbel-Softmax Differentiable Patch Selection](#4-gumbel-softmax-differentiable-patch-selection)
5. [Three Prediction Heads](#5-three-prediction-heads)
6. [Training Data Pipeline](#6-training-data-pipeline)
7. [Three-Stage Training Formulation](#7-three-stage-training-formulation)
8. [Complete PyTorch Module Interfaces](#8-complete-pytorch-module-interfaces)
9. [Implementation Plan](#9-implementation-plan)

---

## 1. Motivation & Problem Statement

### 1.1 What DPVO Gets Wrong

DPVO (Deep Patch Visual Odometry, Teed et al. 2023) is a SOTA learned VO system with three fundamental limitations:

1. **Random patch selection.** Patches are chosen via `torch.randint(1, w-1)` with an optional gradient-magnitude bias (net.py:131-133). There is no learned notion of which image regions are good for tracking. Approximately 30-50% of patches land on textureless regions (sky, road, uniform walls), producing noisy or degenerate correspondences.

2. **No dynamic awareness.** DPVO assumes a static world. Patches that land on moving objects (cars, pedestrians) inject corrupted residuals into the bundle adjustment. The BA solver weights all residuals equally, so a single dynamic patch can pull the optimization away from the correct solution.

3. **No metric uncertainty.** The Update network outputs a per-coordinate weight `w ∈ [0,1]` (net.py:67-71), but this is a heuristic confidence score, not a proper covariance. There is no probabilistic interpretation — the weight is learned implicitly via the pose loss without any uncertainty calibration.

### 1.2 What DynGRU Proved

DynGRU demonstrated that IMU-conditioned features can reliably distinguish static from dynamic pixels. The key architectural insight was using complementary FiLM (global IMU modulation) + CLIP-adapter cross-attention (spatially selective IMU modulation) to condition visual features on ego-motion.

### 1.3 The Core Insight

**IMU data carries information about WHERE to look.** The rigid flow field f_rigid predicted from IMU ego-motion tells you which image regions are consistent with static geometry and which are not. A pixel near the focus of expansion responds to forward translation very differently than a pixel at the image corner. This spatial non-uniformity means IMU conditioning must be spatially selective — which is exactly what the FiLM + cross-attention combination provides.

---

## 2. Architecture Overview

```
                          ┌───────────────────────────┐
                          │     IMU (accel + gyro)    │
                          └─────────────┬─────────────┘
                                        │
                          ┌─────────────┴─────────────┐
                          │        IMUContext         │
                          │    (EKF + Feature MLP)    │
                          └─────┬──────────┬──────────┘
                                │          │
                    f_imu [B,256]          imu_tokens [B,8,64]
                       (FiLM)                (Cross-Attn)
                                │          │
    ┌──────────┐                │          │
    │  Image   │                ▼          ▼
    │  Pair    │    ┌─────────────────────────────────────┐
    │ (t, t+1) │───►│     IMU-Conditioned Encoder         │
    └──────────┘    │                                     │
                    │  Conv7(s=2)                         │
                    │    → ResBlock1(32) → [FiLM₁]        │
                    │    → ResBlock2(64) → [FiLM₂]        │
                    │    → Conv1x1(128) → [FiLM₃]         │
                    │    → [CLIP-Adapter Cross-Attn]      │
                    │         ↑tanh(α) gate, α₀=0         │
                    └────────────────┬────────────────────┘
                                     │
                    ┌────────────────┴─────────────────┐
                    │   Motion-Aware Feature Map       │
                    │      [B, 128, H/4, W/4]          │
                    └──────┬──────────┬─────────┬──────┘
                           │          │         │
              ┌────────────▼──┐ ┌─────▼──────┐ ┌▼──────────────┐
              │ Selectability │ │  Dynamic   │ │  Covariance   │
              │     Head      │ │ Classifier │ │     Head      │
              │ score(x,y)    │ │ P(dyn|x,y) │ │  Σ(x,y) 2x2   │
              └──────┬────────┘ └─────┬──────┘ └───────┬───────┘
                     │                │                │
                     │    final_score = score · (1−P_dyn)
                     │               │                 │
                     └───────┬───────┘                 │
                             │                         │
                    ┌────────▼──────────┐              │
                    │  Gumbel-Softmax   │◄─────────────┘
                    │     Top-K         │   (Σ at selected coords)
                    └────────┬──────────┘
                             │
                    ┌────────▼──────────┐
                    │  K Selected       │
                    │  Patches + Σ      │
                    │  (x₁,y₁,Σ₁),...,  │
                    │  (x_K,y_K,Σ_K)    │
                    └────────┬──────────┘
                             │
                    ┌────────▼──────────┐
                    │  Patch Tracking   │
                    │  + Weighted BA    │
                    │  (Σ⁻¹ as weights) │
                    └───────────────────┘

         ┌─────────────────────────────────────────┐
         │  Training (dashed): Pose loss gradients │
         │  flow back through BA → Patches →       │
         │  Gumbel-ST → Heads → Encoder            │
         └─────────────────────────────────────────┘
```

The system processes a pair of images (t, t+1) along with inter-frame IMU data through four stages:

1. **IMUContext:** EKF propagation + feature MLP produces a global IMU feature `f_imu ∈ ℝ²⁵⁶` and learned semantic tokens `imu_tokens ∈ ℝ^(8×64)`.

2. **IMU-Conditioned Image Encoder:** A `BasicEncoder4` CNN with FiLM layers injected after each residual block, plus a CLIP-adapter-gated cross-attention block at the final feature map. Produces a motion-aware feature map at H/4 × W/4 resolution.

3. **Three Prediction Heads:** All lightweight (1×1 conv-based), operating on the same fused feature map:
   - **Selectability Head:** `score(x,y) ∈ [0,1]` — how trackable is this pixel?
   - **Dynamic Classifier:** `P(dyn|x,y) ∈ [0,1]` — is this pixel on a dynamic object?
   - **Covariance Head:** `Σ(x,y) ∈ ℝ²ˣ²` — predicted 2D correspondence uncertainty.

4. **Gumbel-Softmax Top-K Selection:** `final_score = score · (1 − P_dyn)`. K patches are selected via straight-through Gumbel-Softmax, and their associated Σ values are passed to the weighted BA solver.

---

## 3. IMU Conditioning: FiLM + CLIP-Adapter Cross-Attention

```
                          ┌─────────────────────────────────────────────┐
                          │            IMU Feature f_imu                │
                          │               [B, 256]                      │
                          └──────────────────┬──────────────────────────┘
                                             │
                    ┌────────────────────────┼──────────────────────────┐
                    │                        │                          │
                    ▼                        ▼                          ▼
          ┌──────────────────┐   ┌──────────────────────┐   ┌──────────────────┐
          │  FiLM Pathway    │   │   Slot Projector     │   │  Feature Map x   │
          │  (Global,Uniform)│   │   256 → [B,8,64]     │   │  [B, C, H, W]    │
          └───────┬──────────┘   └──────────┬───────────┘   └────────┬─────────┘
                  │                         │                        │
                  ▼                         ▼                        │
    ┌──────────────────────┐    ┌─────────────────────────────────┐  │
    │  MLP: 256 → 2C       │    │    Cross-Attn Pathway           │  │
    │  γ,β = chunk(params) │    │    (Spatially Selective)        │  │
    └─────────┬────────────┘    │                                 │  │
              │                 │ Q = Conv1x1(x)  → [B,HW,D]      │◄─┘
              ▼                 │ K = Linear(slots) → [B,N,D]     │
    ┌──────────────────────┐    │ V = Linear(slots) → [B,N,D]     │
    │  x_film =             │   │                                 │
    │  x ⊙ (1+γ) + β        │   │  attn = softmax(QKᵀ/√D)         │
    │                       │   │  attn_out = attn ⊗ V            │
    │  Same (γ,β) at every  │   │  → [B,HW,D] → reshape [B,C,H,W] │
    │  spatial location.    │   │                                 │
    │                       │   │  Per-pixel attention to each    │
    │  Encodes: "we are     │   │  of N=8 IMU slots.              │
    │  in operating mode X" │   │                                 │
    │                       │   │  Encodes: "what does mode X     │
    └───────────┬───────────┘   │  mean for pixel (u,v)?"         │
                │               └──────────────┬──────────────────┘
                │                              │
                │              ┌───────────────┴───────────────┐
                │              │     CLIP-Adapter Gate         │
                │              │                               │
                │              │   gate = tanh(α) · attn_out   │
                │              │   α = nn.Parameter(zeros(1))  │
                │              │                               │
                │              │   α₀=0 → gate=0 at init       │
                │              │   ∂L/∂α → α grows as CA       │
                │              │   proves useful               │
                │              └───────────────┬───────────────┘
                │                              │
                └──────────┬───────────────────┘
                           │
                           ▼
              ┌──────────────────────────────────────┐
              │                                      │
              │  x_fused = x_film + tanh(α)·attn_out │
              │                                      │
              │  FiLM:  global operating mode        │
              │  CA:    per-pixel spatial nuance     │
              │  Gate:  cold-start protection        │
              │                                      │
              │  → IMU-Conditioned Feature Map       │
              │    [B, C, H, W]                      │
              └──────────────────────────────────────┘
```

### 3.1 Why Two Complementary Streams

FiLM and cross-attention solve fundamentally different problems in IMU conditioning:

| | FiLM | Cross-Attention |
|---|---|---|
| **Operation** | `x' = x ⊙ (1+γ) + β` per channel | `Σ softmax(Q_img · K_slot) · V_slot` per position |
| **Spatial selectivity** | None — uniform per channel | Full — each pixel attends differently to each IMU slot |
| **What it encodes** | "We are in operating mode X" | "I am at position (u,v); what does mode X mean for ME?" |
| **Cost** | 2·C parameters | O(HW · N_slots · D) |
| **Cold start risk** | Low (small MLP) | High without gating |

**Example:** Forward translation Δp = [0.1, 0, 0]:
- FiLM modulates all channels equally: "we're translating forward, amplify depth-sensitive features."
- Cross-attention per pixel: a pixel at the FOE (where f_rigid ≈ 0) attends to the "stationarity" slot, while a pixel at the corner (where f_rigid is large) attends to the "forward motion" slot.

FiLM alone cannot encode this spatial variation because the same (γ, β) is broadcast to every (x,y) location. Cross-attention alone would need to relearn the global gating from scratch at cold start. Together: FiLM handles the global mode switch; cross-attention adds per-pixel nuance.

### 3.2 FiLM Implementation

Injected at three points in the encoder (after ResBlock1, ResBlock2, and the final Conv1x1):

```python
class FiLMGenerator(nn.Module):
    """Generate (gamma, beta) from IMU feature for FiLM modulation."""
    def __init__(self, imu_dim: int = 256, feat_dim: int = 32):
        super().__init__()
        self.proj = nn.Linear(imu_dim, feat_dim * 2)

    def forward(self, f_imu: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (gamma, beta), each [B, feat_dim, 1, 1]."""
        params = self.proj(f_imu)  # [B, 2*feat_dim]
        gamma, beta = params.chunk(2, dim=1)
        return gamma.unsqueeze(-1).unsqueeze(-1), beta.unsqueeze(-1).unsqueeze(-1)


# Usage in encoder forward():
x = self.layer1(x)
gamma1, beta1 = self.film1(f_imu)
x = x * (1.0 + gamma1) + beta1      # FiLM modulation

x = self.layer2(x)
gamma2, beta2 = self.film2(f_imu)
x = x * (1.0 + gamma2) + beta2

x = self.conv2(x)
gamma3, beta3 = self.film3(f_imu)
x = x * (1.0 + gamma3) + beta3
```

**Key:** The `(1 + gamma)` formulation (standard FiLM) ensures identity at initialization when gamma≈0. This is preferred over DynGRU's `x * gamma + beta` because FiLM generators are zero-initialized, and the encoder should produce meaningful features even before IMU conditioning is learned.

### 3.3 IMU Semantic Slots

The IMU feature `f_imu` is projected into N=8 learned semantic slots, each specializing in a different motion component:

```python
class IMUSlotProjector(nn.Module):
    """Project IMU features into N learned semantic slots."""
    def __init__(self, imu_dim: int = 256, n_slots: int = 8, slot_dim: int = 64):
        super().__init__()
        self.n_slots = n_slots
        self.slot_dim = slot_dim
        self.proj = nn.Sequential(
            nn.Linear(imu_dim, n_slots * slot_dim),
            nn.LayerNorm(n_slots * slot_dim),
        )

    def forward(self, f_imu: torch.Tensor) -> torch.Tensor:
        """Returns imu_tokens [B, n_slots, slot_dim]."""
        return self.proj(f_imu).view(-1, self.n_slots, self.slot_dim)


# Diversity regularizer (prevents slot collapse):
def slot_diversity_loss(tokens: torch.Tensor) -> torch.Tensor:
    """Penalize cosine similarity between different slots."""
    S = F.normalize(tokens, dim=-1)  # [B, N, D]
    sim = S @ S.transpose(-2, -1)    # [B, N, N]
    mask = ~torch.eye(sim.shape[-1], dtype=bool, device=sim.device)
    return sim[:, mask].abs().mean()  # Push apart
```

Slots are NOT explicitly supervised to match specific semantics (dR, dv, dp, etc. as in DynGRU). Instead, they self-organize through the downstream task gradients. This is a design choice: DynGRU's 7 explicit slots work well for a pixel-level classifier, but for patch selection the slots should learn whatever decomposition is most useful for the selection + tracking task.

### 3.4 CLIP-Adapter Cross-Attention

A lightweight single-head cross-attention where each spatial pixel attends to all N IMU slots:

```python
class IMUCrossAttn(nn.Module):
    """CLIP-adapter style cross-attention: spatial pixels attend to IMU slots."""
    def __init__(self, feat_dim: int = 128, n_slots: int = 8, slot_dim: int = 64):
        super().__init__()
        self.scale = slot_dim ** -0.5
        self.q_proj = nn.Conv2d(feat_dim, slot_dim, 1)   # 1x1 conv for spatial Q
        self.k_proj = nn.Linear(slot_dim, slot_dim)       # linear for slot K
        self.v_proj = nn.Linear(slot_dim, slot_dim)       # linear for slot V
        self.out_proj = nn.Conv2d(slot_dim, feat_dim, 1)  # project back to feat_dim
        self.per_slot_gain = nn.Parameter(torch.ones(n_slots))

    def forward(self, x: torch.Tensor, imu_tokens: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x:          [B, C, H, W]  spatial feature map
            imu_tokens: [B, N, D]     IMU semantic slots
        Returns:
            out:        [B, C, H, W]  attention-modulated features
        """
        B, C, H, W = x.shape
        N = imu_tokens.shape[1]

        # Per-slot learnable gain (allows muting weak slots)
        tokens = imu_tokens * self.per_slot_gain.view(1, -1, 1)  # [B, N, D]

        # Q from spatial features
        q = self.q_proj(x).view(B, -1, H*W).transpose(1, 2)      # [B, HW, D]

        # K, V from IMU slots
        k = self.k_proj(tokens)                                    # [B, N, D]
        v = self.v_proj(tokens)                                    # [B, N, D]

        # Scaled dot-product attention
        attn = torch.softmax(q @ k.transpose(-2, -1) * self.scale, dim=-1)  # [B, HW, N]
        out = attn @ v                                              # [B, HW, D]
        out = out.transpose(1, 2).view(B, -1, H, W)                # [B, D, H, W]
        return self.out_proj(out)                                   # [B, C, H, W]
```

### 3.5 The CLIP-Adapter Gate

The critical detail: a learned scalar `alpha` initialized to zero gates the cross-attention contribution.

```python
# In the encoder's __init__:
self.alpha = nn.Parameter(torch.zeros(1))

# In forward():
x_film = x * (1.0 + gamma) + beta             # FiLM pathway
x_attn = self.imu_attn(x_film, imu_tokens)    # Cross-attention pathway
x = x_film + torch.tanh(self.alpha) * x_attn  # Gated combination
```

**Why this matters:**

At initialization (`alpha = 0`): `tanh(0) = 0`, so `x = x_film`. The cross-attention pathway contributes NOTHING. The system behaves identically to a FiLM-only encoder.

During training: The gradient through alpha is `∂L/∂α = ∂L/∂x · x_attn · sech²(α)`. At initialization, `sech²(0) = 1`, so the cross-attention block receives full gradients. It learns what spatially selective corrections are useful. As those corrections prove beneficial, `alpha` grows in magnitude (positive or negative), gradually phasing in the cross-attention contribution.

**The system cannot regress below FiLM-only performance** because the gate can only ADD the attention contribution (or subtract it, if alpha goes negative). The FiLM baseline is preserved structurally, not just empirically.

This is the same pattern validated in DynGRU (dynhead.py:127,145), following the CLIP-adapter formulation from Gao et al. (2021).

### 3.6 Complete Fusion Block Pseudocode

```python
class IMUFusionEncoder(nn.Module):
    """
    Image encoder with IMU conditioning via FiLM + CLIP-adapter cross-attention.

    Architecture:
        Conv7(s=2) → ResBlock1(32) → [FiLM₁] → ResBlock2(64, s=2) → [FiLM₂]
        → Conv1x1(128) → [FiLM₃] → [CLIP-Adapter Cross-Attn]
    """
    def __init__(self, imu_dim=256, feat_dim=128, n_slots=8, slot_dim=64):
        super().__init__()
        # Encoder backbone (same as DPVO's BasicEncoder4)
        self.conv1 = nn.Conv2d(3, 32, 7, stride=2, padding=3)
        self.norm1 = nn.InstanceNorm2d(32)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(32, 32, stride=1)
        self.layer2 = self._make_layer(32, 64, stride=2)
        self.conv2 = nn.Conv2d(64, feat_dim, 1)

        # FiLM generators (one per injection point)
        self.film1 = FiLMGenerator(imu_dim, 32)
        self.film2 = FiLMGenerator(imu_dim, 64)
        self.film3 = FiLMGenerator(imu_dim, feat_dim)

        # CLIP-adapter cross-attention
        self.imu_attn = IMUCrossAttn(feat_dim, n_slots, slot_dim)
        self.alpha = nn.Parameter(torch.zeros(1))

    def forward(self, images, f_imu, imu_tokens):
        # Stage 0: Conv7 + Norm
        x = self.conv1(images)
        x = self.norm1(x)
        x = self.relu(x)

        # Stage 1: ResBlock1 → FiLM₁
        x = self.layer1(x)
        g1, b1 = self.film1(f_imu)
        x = x * (1.0 + g1) + b1

        # Stage 2: ResBlock2 (stride=2) → FiLM₂
        x = self.layer2(x)
        g2, b2 = self.film2(f_imu)
        x = x * (1.0 + g2) + b2

        # Stage 3: Conv1x1 → FiLM₃
        x = self.conv2(x)
        g3, b3 = self.film3(f_imu)
        x = x * (1.0 + g3) + b3

        # Stage 4: CLIP-adapter cross-attention
        x = x + torch.tanh(self.alpha) * self.imu_attn(x, imu_tokens)

        return x  # [B, 128, H/4, W/4]
```

---

## 4. Gumbel-Softmax Differentiable Patch Selection

```
  ┌──────────────────┐   ┌──────────────────┐   ┌──────────────────┐
  │  Selectability   │   │     Dynamic      │   │   Covariance     │
  │  score(x,y)      │   │   P(dyn|x,y)     │   │   Σ(x,y) 2×2     │
  └────────┬─────────┘   └────────┬─────────┘   └────────┬─────────┘
           │                      │                      │
           │   final_score =      │                      │
           │   score · (1−P_dyn)  │                      │
           │                      │                      │
           └──────────┬───────────┘                      │
                      │                                  │
                      ▼                                  │
        ┌──────────────────────────┐                     │
        │  logits = log(score + ε) │                     │
        │  [B, H·W]                │                     │
        └────────────┬─────────────┘                     │
                     │                                   │
                     ▼                                   │
        ┌──────────────────────────────────────┐         │
        │      Gumbel-Max Trick                │         │
        │                                      │         │
        │  U ~ Uniform(0,1)                    │         │
        │  g = logits − log(−log(U))           │         │
        │  (perturbed logits)                  │         │
        └────────┬─────────────────────────────┘         │
                 │                                       │
    ┌────────────┴────────────┐                          │
    │                         │                          │
    ▼                         ▼                          │
┌──────────────────┐  ┌──────────────────────────┐       │
│  FORWARD: Hard   │  │  BACKWARD: Soft          │       │
│                  │  │                          │       │
│  indices =       │  │  soft = softmax(logits   │       │
│    topk(g, K)    │  │         / τ)             │       │
│  hard = one_hot  │  │                          │       │
│                  │  │  ∂L/∂logits =            │       │
│  Exact (x,y)     │  │    ∂L/∂hard · soft       │       │
│  coordinates     │  │                          │       │
│  used for patch  │  │  Temperature τ:          │       │
│  extraction      │  │    τ=1.0 → soft,explore  │       │
│                  │  │    τ=0.1 → sharp,select  │       │
└────────┬─────────┘  └──────────────┬───────────┘       │
         │                           │                   │
         └───────────┬───────────────┘                   │
                     │                                   │
                     ▼                                   │
        ┌──────────────────────────────────────┐         │
        │  Straight-Through Gumbel-Softmax     │         │
        │                                      │         │
        │  out = hard − soft.detach() + soft   │         │
        │                                      │         │
        │  Forward:  hard one-hot → picks      │         │
        │            K exact positions         │◄────────┘
        │  Backward: soft grad → flows into   (Σ sampled
        │            score heads               at indices
        │                                      via bilinear
        │  Diversity: L_div = Σᵢⱼ 1/||pᵢ−pⱼ||  grid_sample)
        └───────────────────┬──────────────────┘
                            │
                            ▼
        ┌───────────────────────────────────────┐
        │  K Selected Patches + Covariances     │
        │                                       │
        │  (x₁,y₁, Σ₁)  (x₂,y₂, Σ₂)  ...        │
        │       ↓           ↓                   │
        │  Feature Extraction (bilinear sample) │
        │       ↓           ↓                   │
        │  DPVO Patch Tracking + Weighted BA    │
        │  (residuals weighted by Σ⁻¹)          │
        └───────────────────────────────────────┘
```

### 4.1 The Problem

We have a score map `S(x,y) = score(x,y) · (1 − P_dyn(x,y))` and we want to select K patches at the highest-scoring locations. The `argtopk` operation is non-differentiable, breaking gradient flow from the BA pose loss back to the selector heads.

### 4.2 Straight-Through Gumbel-Softmax Top-K

**Forward pass (hard, exact):** Standard top-K selection on Gumbel-perturbed scores.

```python
def gumbel_topk_forward(scores, K):
    """Hard top-K with Gumbel noise. Returns (indices, hard_weights)."""
    # Add Gumbel noise
    U = torch.rand_like(scores)
    g = scores.log() - (-U.log()).log()  # Gumbel-Max trick

    # Top-K indices
    _, indices = torch.topk(g.flatten(), K)

    # One-hot encoding (hard selection)
    hard = torch.zeros_like(scores.flatten())
    hard.scatter_(0, indices, 1.0)
    hard = hard.view_as(scores)

    return indices, hard
```

**Backward pass (soft, differentiable):** Use softmax with temperature as a continuous relaxation.

```python
def gumbel_softmax_backward(scores, tau=0.5):
    """Soft relaxation of top-K for gradient estimation."""
    # Temperature-controlled softmax
    soft = F.softmax(scores.flatten() / tau, dim=0)
    soft = soft.view_as(scores)
    return soft
```

**Straight-through estimator:** Use the hard selection in the forward pass but propagate gradients through the soft relaxation.

```python
class GumbelTopK(torch.autograd.Function):
    @staticmethod
    def forward(ctx, scores, K, tau):
        _, hard = gumbel_topk_forward(scores, K)
        soft = gumbel_softmax_backward(scores, tau)
        ctx.save_for_backward(soft)
        ctx.tau = tau
        return hard  # Hard selection used for actual patches

    @staticmethod
    def backward(ctx, grad_output):
        soft, = ctx.saved_tensors
        return grad_output * soft, None, None  # Gradients flow through soft
```

### 4.3 Temperature Annealing

Lower temperature → sharper selection (closer to true top-K). Higher temperature → softer, more uniform (better gradient flow).

```
τ_start = 1.0    (more exploratory, better gradient signal)
τ_end   = 0.1    (closer to hard top-K)

Schedule:  τ(t) = max(τ_end, τ_start * exp(-t / anneal_steps))
```

### 4.4 Spatial Diversity Regularizer

Top-K alone can produce clustered patches (all on the single best corner). A repulsion term encourages spatial coverage:

```python
def spatial_diversity_loss(coords, sigma=0.1):
    """
    Penalize nearby patches to encourage spatial coverage.

    Args:
        coords: [B, K, 2]  normalized coordinates of selected patches
        sigma:  bandwidth of repulsion kernel
    Returns:
        scalar loss
    """
    # Pairwise distances
    diff = coords.unsqueeze(2) - coords.unsqueeze(1)  # [B, K, K, 2]
    dist = (diff ** 2).sum(dim=-1)  # [B, K, K]

    # Repulsion: 1/distance (avoid collapse)
    return (1.0 / (dist + 1e-6)).mean()
```

### 4.5 Patch Extraction from Selected Coordinates

Selected coordinates are in H/4 × W/4 feature space. Feature patches and context vectors are extracted via bilinear sampling:

```python
def extract_patches(fmap, coords, patch_size=3):
    """
    Extract patches at selected coordinates.

    Args:
        fmap:   [B, C, H, W]   feature map (H/4 resolution)
        coords: [B, K, 2]      normalized coordinates [-1, 1]
        patch_size: int        spatial extent of each patch
    Returns:
        gmap:  [B, K, C, P, P]  feature patches
        imap:  [B, K, C, 1, 1]  context vectors (center pixel)
    """
    B, C, H, W = fmap.shape
    K = coords.shape[1]
    P = patch_size

    # Create sampling grid around each coordinate
    offset = (P - 1) / 2
    grid_y, grid_x = torch.meshgrid(
        torch.linspace(-offset, offset, P, device=fmap.device),
        torch.linspace(-offset, offset, P, device=fmap.device),
        indexing='ij'
    )
    grid = torch.stack([grid_x, grid_y], dim=-1).float()  # [P, P, 2]

    # Add per-patch offsets to coordinates
    grid = coords[:, :, None, None, :] + grid[None, None, :, :, :]  # [B, K, P, P, 2]
    grid = grid / torch.tensor([W/2, H/2], device=fmap.device) - 1.0  # normalize

    # Bilinear sample
    gmap = F.grid_sample(fmap, grid.view(B, K*P*P, 1, 2).contiguous(),
                         mode='bilinear', align_corners=True)
    gmap = gmap.view(B, C, K, P, P).permute(0, 2, 1, 3, 4)  # [B, K, C, P, P]

    # Context vector: center pixel only
    imap = F.grid_sample(fmap, coords.view(B, K, 1, 2).contiguous(),
                         mode='bilinear', align_corners=True)
    imap = imap.view(B, C, K, 1, 1).permute(0, 2, 1, 3, 4)  # [B, K, C, 1, 1]

    return gmap, imap
```

---

## 5. Three Prediction Heads

All three heads operate on the same IMU-conditioned feature map `[B, 128, H/4, W/4]` and are implemented as lightweight 1×1 convolutional networks.

### 5.1 Selectability Head

Predicts how trackable each pixel is for the DPVO iterative LK tracker.

```python
class SelectabilityHead(nn.Module):
    """Predict per-pixel trackability score."""
    def __init__(self, feat_dim=128, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(feat_dim, hidden_dim, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, 1, 1),
            nn.Sigmoid(),  # score ∈ [0, 1]
        )

    def forward(self, fmap):
        return self.net(fmap)  # [B, 1, H, W]
```

**Supervision:** Self-supervised. Tracking quality is measured as the normalized convergence of the iterative LK refinement:

```
q_track(patch_i) = 1.0 − min(1.0, ||delta_final|| / max_displacement)
```

where `delta_final` is the final flow update vector from DPVO's Update operator. A patch that converges quickly (small delta) gets a high quality score. A patch that oscillates or diverges gets a low score.

```python
def selectability_loss(scores, tracking_quality):
    """MSE between predicted score and empirical tracking quality."""
    return F.mse_loss(scores, tracking_quality)
```

### 5.2 Dynamic Classifier Head

Predicts the probability that a pixel lies on a dynamically moving object.

```python
class DynamicHead(nn.Module):
    """Predict per-pixel P(dynamic | image, IMU)."""
    def __init__(self, feat_dim=128, hidden_dim=64):
        super().__init__()
        # IMU disagreement feature: ||f_est - f_rigid|| per pixel
        self.net = nn.Sequential(
            nn.Conv2d(feat_dim + 3, hidden_dim, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, 1, 1),
            nn.Sigmoid(),
        )

    def forward(self, fmap, imu_disagreement):
        """
        Args:
            fmap:              [B, 128, H, W]  fused features
            imu_disagreement:  [B, 3, H, W]    (flow_est - flow_rigid) magnitude
        """
        inp = torch.cat([fmap, imu_disagreement], dim=1)
        return self.net(inp)  # [B, 1, H, W]
```

The `imu_disagreement` input provides an explicit geometric cue: pixels where the observed flow differs substantially from the IMU-predicted rigid flow are more likely to be dynamic. This is computed from the DPVO flow estimate (available from the Update operator) and the rigid flow from IMU ego-motion.

**Supervision:** Binary cross-entropy on synthetic data where GT dynamic masks are free (from insertion). On real data, self-supervised via photometric consistency as a proxy signal.

```python
def dynamic_loss(p_dyn, gt_mask=None, photometric_consistency=None, mode='synthetic'):
    if mode == 'synthetic' and gt_mask is not None:
        return F.binary_cross_entropy(p_dyn, gt_mask)
    elif mode == 'real' and photometric_consistency is not None:
        # Photometrically inconsistent → likely dynamic
        pseudo_label = (photometric_consistency < 0.95).float().detach()
        return F.binary_cross_entropy(p_dyn, pseudo_label)
```

### 5.3 Covariance Head

Predicts per-pixel 2×2 correspondence covariance matrix. Parameterized as (log_var_u, log_var_v, correlation) for unconstrained optimization.

```python
class CovarianceHead(nn.Module):
    """Predict per-pixel 2x2 covariance."""
    def __init__(self, feat_dim=128, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(feat_dim, hidden_dim, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, 3, 1),  # (log_var_u, log_var_v, rho)
        )

    def forward(self, fmap):
        raw = self.net(fmap)  # [B, 3, H, W]

        # Exponentiate variances (must be positive)
        var_u = torch.exp(raw[:, 0:1])       # σ_u²
        var_v = torch.exp(raw[:, 1:2])       # σ_v²
        rho = torch.tanh(raw[:, 2:3])        # correlation ∈ [-1, 1]
        cov_uv = rho * torch.sqrt(var_u * var_v)

        # Assemble 2x2 covariance
        # Σ = [[σ_u², ρσ_uσ_v], [ρσ_uσ_v, σ_v²]]
        return var_u, var_v, cov_uv  # [B, 1, H, W] each


def assemble_covariance(var_u, var_v, cov_uv):
    """Assemble per-pixel 2x2 covariance matrices."""
    B, _, H, W = var_u.shape
    sigma = torch.zeros(B, H, W, 2, 2, device=var_u.device)
    sigma[..., 0, 0] = var_u.squeeze(1)
    sigma[..., 0, 1] = cov_uv.squeeze(1)
    sigma[..., 1, 0] = cov_uv.squeeze(1)
    sigma[..., 1, 1] = var_v.squeeze(1)
    return sigma  # [B, H, W, 2, 2]
```

**Supervision:** Self-supervised MLE. The predicted covariance should match the empirical residual:

```python
def covariance_loss(sigma, empirical_residual):
    """
    Gaussian MLE: minimize ||Sigma - r*r^T||
    where r = flow_rigid - flow_est (the tracking residual).
    """
    r = empirical_residual.unsqueeze(-1)  # [B, H, W, 2, 1]
    rrt = r @ r.transpose(-2, -1)         # [B, H, W, 2, 2]
    diff = sigma - rrt.detach()           # Detach: covariance matches residual, not vice versa
    return (diff ** 2).mean()
```

### 5.4 Combined Score

```python
def compute_patch_scores(selectability, p_dyn):
    """Combine selectability and dynamic probability into final score."""
    return selectability * (1.0 - p_dyn)
```

A patch gets selected if it is BOTH trackable AND static. The multiplicative combination means: a highly trackable patch on a dynamic object (score=0.9, p_dyn=0.9) gets final_score=0.09 → unlikely to be selected. A moderately trackable patch on a static surface (score=0.6, p_dyn=0.0) gets final_score=0.6 → likely selected.

---

## 6. Training Data Pipeline

```
 ┌───────────────────────┐   ┌───────────────────────┐   ┌──────────────────────┐
 │ Real Static VI Dataset│   │  Object Patch Library │   │  3D Trajectory Gen   │
 │                       │   │                       │   │                      │
 │ EuRoC / KITTI-odom    │   │ Cityscapes / KITTI360 │   │ Smooth SE3 paths     │
 │                       │   │ BDD100K               │   │ Random velocity      │
 │ → Real IMU            │   │                       │   │ Random start pos     │
 │ → Real camera noise   │   │ → RGBA object patches │   │ → T frames of poses  │
 │ → Real lighting       │   │   (instance masks)    │   │                      │
 │ → GT pose ✓           │   │                       │   │                      │
 └───────────┬───────────┘   └───────────┬───────────┘   └──────────┬───────────┘
             │                           │                          │
             └───────────────────────────┼──────────────────────────┘
                                         │
                                         ▼
                    ┌─────────────────────────────────────────┐
                    │      Dynamic Object Insertion Engine    │
                    │                                         │
                    │  For each frame:                        │
                    │  1. Project 3D traj → 2D bbox at depth d│
                    │  2. Resize object patch to bbox         │
                    │  3. Alpha-blend (Poisson if available)  │
                    │  4. Handle occlusion by depth ordering  │
                    │  5. Record: binary mask + 2D flow       │
                    │     + depth at dynamic pixels           │
                    └──────────────────┬──────────────────────┘
                                       │
                    ┌──────────────────▼──────────────────────┐
                    │     Insertion Augmentation              │
                    │                                         │
                    │  Prevent network from detecting         │
                    │  "pasted artifact" instead of dynamics: │
                    │                                         │
                    │  • Color jitter (brightness, contrast,  │
                    │    saturation ±20%)                     │
                    │  • Random scale ±10%                    │
                    │  • Random rotation ±10°                 │
                    │  • Gaussian blur σ∈[0,1] (p=0.3)        │
                    │  • Motion blur along traj (p=0.3)       │
                    └──────────────────┬──────────────────────┘
                                       │
                                       ▼
                    ┌─────────────────────────────────────────┐
                    │          Training Sample                │
                    │                                         │
                    │  {                                      │
                    │    images:      [B, T, 3, H, W]         │
                    │    imu_data:    [B, T, 6]               │
                    │    poses:       [B, T, 7]  (GT from     │
                    │    dynamic_mask:[B, T, 1, H, W]  orig   │
                    │    K:           [4]        dataset)     │
                    │    inserted_flow: [B,T,2,H,W] (optional)│
                    │  }                                      │
                    └─────────────────────────────────────────┘
```

### 6.1 The Semi-Synthetic Approach

**Why not train on TartanAir?** TartanAir has <5% dynamic content and the IMU is simulated. **Why not use only real data?** No existing VI dataset has dynamic scenes with GT pose.

**Solution:** Take real static VI datasets (EuRoC, KITTI-odometry) with real IMU + real camera + GT pose, and synthetically insert dynamic objects with known trajectories.

### 6.2 Object Patch Library

Extract object instances from segmentation datasets:

| Dataset | Content | Objects |
|---------|---------|---------|
| Cityscapes | Urban street scenes | Cars, pedestrians, cyclists, motorcycles |
| KITTI-360 | 360° driving | Cars, trucks, pedestrians (more viewpoints) |
| BDD100K | Diverse driving | Cars, pedestrians, riders, buses |

Each object is extracted as an RGBA patch (with alpha mask from instance segmentation).

### 6.3 Trajectory Generation

For each inserted object, sample a smooth 3D trajectory in the camera frame:

```python
def sample_object_trajectory(T, velocity_range=(0.5, 8.0), cam_frame=True):
    """
    Sample a smooth 3D trajectory for an inserted object.

    Args:
        T: number of frames
        velocity_range: (min, max) velocity in m/s
    Returns:
        poses: [T, 7]  SE3 poses in camera frame (quat + tvec)
    """
    # Random initial position at a reasonable depth
    init_depth = torch.rand(1) * 20 + 3  # 3-23 meters
    init_x = torch.randn(1) * 2           # lateral offset
    init_y = torch.randn(1) * 1.5         # vertical offset

    # Random velocity direction (mostly lateral to simulate crossing traffic)
    velocity = torch.rand(1) * (velocity_range[1] - velocity_range[0]) + velocity_range[0]
    direction = F.normalize(torch.randn(3), dim=-1)
    direction[2] *= 0.3  # Reduce depth motion

    # Generate smooth trajectory with random walk perturbations
    positions = [torch.tensor([init_x, init_y, init_depth])]
    for t in range(1, T):
        perturbation = torch.randn(3) * 0.05  # small random walk
        positions.append(positions[-1] + direction * velocity * 0.1 + perturbation)

    # Convert to SE3
    poses = []
    for pos in positions:
        # Object faces camera (identity rotation with small random tilt)
        rot = SO3.exp(torch.randn(3) * 0.1)
        poses.append(torch.cat([rot.quaternion(), pos]))

    return torch.stack(poses)
```

### 6.4 Insertion Engine

For each frame, render inserted objects with proper depth ordering:

```python
def insert_dynamic_objects(frame, object_patches, trajectories, depth_map, K):
    """
    Insert dynamic objects into a frame.

    Args:
        frame:          [3, H, W]  original image
        object_patches: list of [C, H_obj, W_obj] with alpha
        trajectories:   list of [T, 7] SE3 poses per object
        depth_map:      [H, W]     approximate depth (from stereo or SfM)
        K:              [3, 3]     camera intrinsics

    Returns:
        augmented:      [3, H, W]  image with inserted objects
        dynamic_mask:   [H, W]     binary mask (1=dynamic)
        inserted_flow:  [2, H, W]  optical flow for dynamic pixels
    """
    dynamic_mask = torch.zeros(H, W)
    inserted_flow = torch.zeros(2, H, W)

    # Sort objects by depth (far to near) for correct occlusion
    obj_depths = [traj[frame_idx, 6] for traj in trajectories]  # z translation
    depth_order = sorted(range(len(obj_depths)), key=lambda i: obj_depths[i], reverse=True)

    for obj_idx in depth_order:
        # Project object center to image
        pose = trajectories[obj_idx][frame_idx]
        tvec = pose[4:7]
        uv = K @ tvec
        uv = uv[:2] / uv[2]

        # Compute bounding box at this depth
        obj_h, obj_w = object_patches[obj_idx].shape[1:]
        scale = K[0, 0] * 0.15 / tvec[2]  # 15cm object at depth z
        bbox_w = int(obj_w * scale)
        bbox_h = int(obj_h * scale)

        # Resize and blend object patch
        obj_resized = F.interpolate(object_patches[obj_idx].unsqueeze(0),
                                    size=(bbox_h, bbox_w), mode='bilinear')
        # ... composite onto frame using alpha blending

        # Record mask and flow for dynamic pixels
        # flow = (uv_current - uv_previous_frame) for dynamic region
        dynamic_mask[bbox_region] = 1.0
        inserted_flow[:, bbox_region] = flow_at_bbox

    return augmented, dynamic_mask, inserted_flow
```

### 6.5 Insertion Augmentation

To prevent the network from learning "pasted artifact" detection instead of actual dynamics:

```python
def augment_inserted_object(obj_patch):
    """Randomly augment inserted object to reduce sim-to-real gap."""
    # Color jitter
    brightness = 0.8 + 0.4 * torch.rand(1)
    contrast = 0.8 + 0.4 * torch.rand(1)
    saturation = 0.8 + 0.4 * torch.rand(1)
    obj_patch = adjust_brightness(obj_patch, brightness)
    obj_patch = adjust_contrast(obj_patch, contrast)
    obj_patch = adjust_saturation(obj_patch, saturation)

    # Random scale (±10%)
    scale = 0.9 + 0.2 * torch.rand(1)
    obj_patch = F.interpolate(obj_patch, scale_factor=scale)

    # Random rotation (±10°)
    angle = (torch.rand(1) - 0.5) * 20
    obj_patch = rotate(obj_patch, angle)

    # Random blur
    if torch.rand(1) < 0.3:
        sigma = torch.rand(1) * 1.0
        obj_patch = gaussian_blur(obj_patch, sigma)

    # Motion blur (along trajectory direction)
    if torch.rand(1) < 0.3:
        obj_patch = motion_blur(obj_patch, kernel_size=5)

    return obj_patch
```

### 6.6 Dataset Statistics (Target)

| Split | Source | Sequences | Frames | % Dynamic Pixels |
|---|---|---|---|---|
| Train | EuRoC MH01-05 + KITTI 00,02,05,06,08 | ~8 | ~60K | 5-25% (varied) |
| Val | EuRoC V101 + KITTI 07 | ~2 | ~5K | 5-25% |
| Test (static) | EuRoC V201-203 + KITTI 09 | ~4 | ~10K | 0% (clean) |
| Test (dynamic) | EuRoC MH01 + inserted objects | ~1 | ~3K | 5-25% |

---

## 7. Three-Stage Training Formulation

```
 ┌──────────────────────────────────────────────────────────────────────────┐
 │                         STAGE 1: Encoder Warmup                          │
 │                            (~2,000 steps)                                │
 │                                                                          │
 │  Frozen:  All 3 heads (selectability, dynamic, covariance)               │
 │  Active:  Encoder + FiLM + Cross-Attention                               │
 │  Loss:    L = L_flow = ||x_pred − x_gt|| · mask                          │
 │           (DPVO flow reprojection loss, weight=0.1)                      │
 │  Goal:    Encoder produces features that support patch correspondence    │
 │           before heads need to make fine-grained predictions             │
 └────────────────────────────────────┬─────────────────────────────────────┘
                                     │
                                     ▼
 ┌──────────────────────────────────────────────────────────────────────────┐
 │                       STAGE 2: Head Pretraining                          │
 │                            (~10,000 steps)                               │
 │                                                                          │
 │  Frozen:  Encoder backbone (low LR: 1e-5)                                │
 │  Active:  All 3 heads (LR: 1e-3)                                         │
 │                                                                          │
 │  Losses:                                                                 │
 │  ┌──────────────────┬───────────────────────────────┬──────────┐         │
 │  │ Head             │ Loss                          │ Weight   │         │
 │  ├──────────────────┼───────────────────────────────┼──────────┤         │
 │  │ Selectability    │ L_sel = MSE(score, track_q)   │   1.0    │         │
 │  │ Dynamic          │ L_dyn = BCE(P_dyn, synth_mask)│   2.0    │         │
 │  │ Covariance       │ L_cov = ||Σ − r·rᵀ||_F        │   0.1    │         │
 │  │ Flow (auxiliary) │ L_flow (same as Stage 1)      │   0.05   │         │
 │  │ Diversity        │ L_div = slot_div + spatial_cov│   0.01   │         │
 │  └──────────────────┴───────────────────────────────┴──────────┘         │
 └────────────────────────────────────┬─────────────────────────────────────┘
                                      │
                                      ▼
 ┌──────────────────────────────────────────────────────────────────────────┐
 │                   STAGE 3: End-to-End Differentiable BA                  │
 │                            (~50,000 steps)                               │
 │                                                                          │
 │  Frozen:  Nothing. Full system unfrozen (LR: 5e-5 all params)            │
 │                                                                          │
 │  Loss:    L = L_pose + 0.1·L_cov + 0.01·L_div                            │ 
 │                                                                          │
 │           L_pose = ||log(dP · dG⁻¹)||    (SE3 log-map error)             │
 │                    ├── translation norm                                  │
 │                    └── rotation norm                                     │
 │                                                                          │
 │  Key:  Gradients flow:                                                   │
 │        Pose Error → BA Solver → Patch Residuals → Gumbel-ST → Heads      │
 │                                                                          │
 │        → Cov head learns: "underestimating Σ on bad patch → pose error"  │
 │        → Dyn head learns: "selecting dynamic patches → pose error"       │
 │        → Sel head learns: "low-texture patches → poor convergence"       │
 └──────────────────────────────────────────────────────────────────────────┘
```

### 7.1 Stage 1: Encoder Warmup (2K steps)

**Goal:** Train the encoder + FiLM + cross-attention to produce reasonable features before the heads are expected to make fine-grained predictions.

**What's frozen:** All three heads (selectability, dynamic, covariance).

**Loss:** DPVO's flow reprojection loss (net.py:87-89). The encoder must produce features that support accurate patch correspondence, which is a prerequisite for the heads to make useful predictions.

```
L_stage1 = L_flow = ||x_pred - x_gt|| · mask     (DPVO flow loss, weight=0.1)
```

### 7.2 Stage 2: Head Pretraining (10K steps)

**Goal:** Train the three heads with direct supervision before end-to-end BA fine-tuning.

**What's frozen:** Encoder backbone (low LR: 1e-5 vs 2e-4 for heads).

**Losses:**

| Head | Loss | Weight |
|---|---|---|
| Selectability | `L_sel = MSE(score, tracking_quality)` | 1.0 |
| Dynamic | `L_dyn = BCE(P_dyn, synthetic_mask)` | 2.0 |
| Covariance | `L_cov = ||Σ − r·rᵀ||_F` | 0.1 |
| Flow (auxiliary) | `L_flow` as in Stage 1 | 0.05 |
| Diversity | `L_div = slot_diversity + spatial_coverage` | 0.01 |

```
L_stage2 = 1.0*L_sel + 2.0*L_dyn + 0.1*L_cov + 0.05*L_flow + 0.01*L_div
```

### 7.3 Stage 3: End-to-End Differentiable BA (50K steps)

**Goal:** Fine-tune the full system so that patch selection, covariance prediction, and tracking co-adapt through the BA pose loss.

**What's frozen:** Nothing. Full system unfrozen.

**Key:** Gradients flow from the BA pose error back through the pose update, through the patch correspondences, through the Gumbel-Softmax selector, into all three heads. The covariance head learns that underestimating uncertainty on a bad patch hurts pose accuracy. The dynamic head learns that selecting dynamic patches (even highly trackable ones) increases pose error. The selectability head learns which visual patterns produce reliable correspondences.

```
L_stage3 = L_pose + 0.1*L_cov + 0.01*L_div
```

where `L_pose` is DPVO's SE3 log-map error on relative poses:

```python
dP = P_pred[:, ii].inv() * P_pred[:, jj]       # Relative pose from prediction
dG = P_gt[:, ii].inv() * P_gt[:, jj]            # Relative pose from ground truth
e = (dP * dG.inv()).log()                       # SE3 error in Lie algebra
L_pose = e[..., 0:3].norm(dim=-1).mean() + \    # Translation error
         e[..., 3:6].norm(dim=-1).mean()        # Rotation error
```

### 7.4 Optimization Details

| Parameter | Value |
|---|---|
| Optimizer | AdamW |
| Learning rate (enc + FiLM + CA) | 2e-4 (warmup), 1e-5 (S2), 5e-5 (S3) |
| Learning rate (heads) | 1e-3 (S2), 5e-5 (S3) |
| Weight decay | 1e-6 |
| Scheduler | OneCycleLR, pct_start=0.05 |
| Batch size | 4 (2-frame pairs × 2 sequences) |
| Gradient clipping | 1.0 (max norm) |
| Mixed precision | AMP (torch.cuda.amp) |
| Hardware | 1× A100 or H100 (40GB+) |

---

## 8. Complete PyTorch Module Interfaces

### 8.1 IMUContext Integration

```python
# Module/Network/IMUContext/imu_context.py  (existing, extended with slot projector)

class IMUSample(NamedTuple):
    """Output of IMUContext.step() for one timestep."""
    f_imu: torch.Tensor          # [1, 256]      global feature for FiLM
    imu_tokens: torch.Tensor     # [1, N, D]     semantic slots for cross-attn
    z_raw: torch.Tensor          # [34]          raw IMU state vector
    state: torch.Tensor          # [15]          EKF state
    P_diag: torch.Tensor         # [15]          EKF covariance diagonal
    R_b2c: torch.Tensor          # [3, 3]        body-to-camera rotation
    T_rel: torch.Tensor          # [4, 4]        relative camera pose (for rigid flow)
```

### 8.2 Full Patch Selector Module

```python
# Module/Network/PatchSelector/imu_patch_selector.py  (NEW FILE)

class IMUPatchSelector(nn.Module):
    """
    IMU-conditioned patch selector for DPVO-like systems.

    Replaces DPVO's random Patchifier with learned, IMU-aware selection.

    Architecture:
        IMU → IMUContext → (f_imu, imu_tokens)
        Image → IMUFusionEncoder → Motion-Aware Feature Map
        Feature Map → [SelectabilityHead, DynamicHead, CovarianceHead]
        Scores → GumbelSoftmaxTopK → K Patches + Σ
    """

    def __init__(self,
                 imu_dim: int = 256,
                 feat_dim: int = 128,
                 n_slots: int = 8,
                 slot_dim: int = 64,
                 patches_per_image: int = 80,
                 patch_size: int = 3,
                 gumbel_tau: float = 0.5):
        super().__init__()

        # IMU context (shared with training)
        self.imu_context = IMUContext(airio_ckpt=None)

        # IMU feature projector (34-dim z → 256-dim f_imu)
        self.imu_projector = nn.Sequential(
            nn.Linear(34, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 256),
        )

        # IMU slot projector (for cross-attention)
        self.slot_projector = IMUSlotProjector(256, n_slots, slot_dim)

        # IMU-conditioned image encoder
        self.encoder = IMUFusionEncoder(256, feat_dim, n_slots, slot_dim)

        # Three prediction heads
        self.selectability_head = SelectabilityHead(feat_dim)
        self.dynamic_head = DynamicHead(feat_dim)
        self.covariance_head = CovarianceHead(feat_dim)

        # Selection parameters
        self.patches_per_image = patches_per_image
        self.patch_size = patch_size
        self.gumbel_tau = gumbel_tau

    def forward(self, images, imu_data, poses_gt=None):
        """
        Args:
            images:   [B, N, 3, H, W]     image sequence
            imu_data: [B, N, T_imu, 6]   inter-frame IMU measurements
            poses_gt: [B, N, 7]           (optional) GT poses for training

        Returns:
            fmap:       [B, N, feat_dim, H/4, W/4]  full feature maps
            gmap:       [B, N*K, feat_dim, P, P]     feature patches
            imap:       [B, N*K, DIM, 1, 1]           context vectors
            patches:    [B, N*K, 3, P, P]             initial (x,y,d) patches
            index:      [N*K]                          frame index per patch
            selected_coords: [B, N, K, 2]             selected patch coords
            covariances:     [B, N, K, 2, 2]          per-patch covariance
            p_dyn:           [B, N, 1, H/4, W/4]     dynamic probability map
        """
        B, N, C, H, W = images.shape

        # Step 1: Extract IMU features per frame pair
        f_imu_list, tokens_list, T_rel_list = [], [], []
        for b in range(B):
            self.imu_context.seed_from_gt(poses_gt[b, 0])  # if training
            for n in range(N - 1):
                sample = self.imu_context.step(imu_data[b, n])
                f_imu_list.append(self.imu_projector(sample.z_raw))
                tokens_list.append(self.slot_projector(f_imu_list[-1]))
                T_rel_list.append(sample.T_rel)

        f_imu = torch.stack(f_imu_list).view(B, -1, 256)       # [B, N-1, 256]
        imu_tokens = torch.stack(tokens_list).view(B, -1, 8, 64)  # [B, N-1, 8, 64]

        # Step 2: Encode images with IMU conditioning
        images_flat = images.view(B * N, C, H, W)
        f_imu_expanded = f_imu.repeat_interleave(N, dim=0)  # or per-frame
        imu_tokens_expanded = imu_tokens.repeat_interleave(N, dim=0)

        fmaps = self.encoder(images_flat, f_imu_expanded, imu_tokens_expanded)
        fmaps = fmaps.view(B, N, -1, H//4, W//4)

        # Step 3: Predict per-frame scores
        selectability = self.selectability_head(fmaps.view(B*N, -1, H//4, W//4))
        p_dyn = self.dynamic_head(fmaps.view(B*N, -1, H//4, W//4), imu_disagreement=None)
        var_u, var_v, cov_uv = self.covariance_head(fmaps.view(B*N, -1, H//4, W//4))

        selectability = selectability.view(B, N, 1, H//4, W//4)
        p_dyn = p_dyn.view(B, N, 1, H//4, W//4)

        # Step 4: Compute final scores
        scores = selectability * (1.0 - p_dyn)  # [B, N, 1, H/4, W/4]

        # Step 5: Gumbel-Softmax Top-K selection
        coords_list, cov_list = [], []
        for n in range(N):
            hard, coords = self._select_patches_gumbel(
                scores[:, n], self.patches_per_image, self.gumbel_tau
            )
            coords_list.append(coords)
            # Extract covariances at selected locations
            cov_list.append(self._extract_covariances(
                var_u[:, n], var_v[:, n], cov_uv[:, n], coords
            ))

        selected_coords = torch.stack(coords_list, dim=1)  # [B, N, K, 2]
        covariances = torch.stack(cov_list, dim=1)          # [B, N, K, 2, 2]

        # Step 6: Extract patches from feature maps
        gmap, imap, patches, index = self._extract_patches(
            fmaps, selected_coords
        )

        return (fmaps, gmap, imap, patches, index,
                selected_coords, covariances, p_dyn)

    def _select_patches_gumbel(self, scores, K, tau):
        """Straight-through Gumbel-Softmax Top-K selection."""
        B, _, H, W = scores.shape
        logits = (scores + 1e-8).log().flatten(1)  # [B, H*W]

        if self.training:
            hard, soft, indices = gumbel_topk_st(logits, K, tau)
        else:
            _, indices = torch.topk(logits, K, dim=1)
            hard = torch.zeros_like(logits).scatter_(1, indices, 1.0)

        # Convert indices to coordinates
        y = indices // W
        x = indices % W
        coords = torch.stack([x, y], dim=-1).float()  # [B, K, 2]

        # Normalize to [-1, 1] for grid_sample
        coords_norm = coords / torch.tensor([W/2, H/2], device=coords.device) - 1.0

        return hard, coords_norm

    def _extract_covariances(self, var_u, var_v, cov_uv, coords_norm):
        """Bilinear sample covariances at selected coords."""
        sigma_u = F.grid_sample(var_u, coords_norm.unsqueeze(1).unsqueeze(1),
                                mode='bilinear', align_corners=True)
        sigma_v = F.grid_sample(var_v, coords_norm.unsqueeze(1).unsqueeze(1),
                                mode='bilinear', align_corners=True)
        sigma_uv = F.grid_sample(cov_uv, coords_norm.unsqueeze(1).unsqueeze(1),
                                 mode='bilinear', align_corners=True)
        # Assemble 2x2 matrices
        K = coords_norm.shape[1]
        Sigma = torch.zeros(B, K, 2, 2, device=var_u.device)
        Sigma[..., 0, 0] = sigma_u.squeeze()
        Sigma[..., 1, 1] = sigma_v.squeeze()
        Sigma[..., 0, 1] = sigma_uv.squeeze()
        Sigma[..., 1, 0] = Sigma[..., 0, 1]
        return Sigma

    def _extract_patches(self, fmaps, coords_norm):
        """Extract feature patches at selected coordinates."""
        B, N, C, H, W = fmaps.shape
        K = coords_norm.shape[2]
        P = self.patch_size

        fmap_flat = fmaps.view(B * N, C, H, W)
        coords_flat = coords_norm.view(B * N, K, 2)

        # Feature patches
        gmap = extract_patches_grid(fmap_flat, coords_flat, P)  # [B*N, K, C, P, P]

        # Context vectors
        imap = F.grid_sample(fmap_flat, coords_flat.unsqueeze(2).unsqueeze(2),
                            mode='bilinear', align_corners=True)
        imap = imap.view(-1, K, C, 1, 1)

        # Initialize depth patches (random + median from previous frames)
        d_init = torch.rand(B*N, K, 1, P, P, device=fmaps.device)
        grid_y, grid_x = torch.meshgrid(
            torch.linspace(-1, 1, P, device=fmaps.device),
            torch.linspace(-1, 1, P, device=fmaps.device),
            indexing='ij'
        )
        patches = torch.cat([
            grid_x[None, None, :, :].expand(B*N, K, 1, P, P),
            grid_y[None, None, :, :].expand(B*N, K, 1, P, P),
            d_init,
        ], dim=2)  # [B*N, K, 3, P, P]

        # Frame index
        frames_per_image = torch.arange(N, device=fmaps.device).view(N, 1)
        index = frames_per_image.repeat(1, K).reshape(-1)  # [N*K]

        return gmap, imap, patches, index
```

### 8.3 Modified DPVO VONet

```python
# Baseline/DPVO/dpvo/net.py  (MODIFIED: replace Patchifier with IMUPatchSelector)

class IMUVONet(nn.Module):
    """DPVO VONet with IMU-conditioned patch selection."""
    def __init__(self, use_viewer=False):
        super().__init__()
        self.P = 3
        self.patchify = IMUPatchSelector(
            imu_dim=256, feat_dim=128, n_slots=8, slot_dim=64,
            patches_per_image=80, patch_size=3, gumbel_tau=0.5
        )
        self.update = Update(self.P)  # unchanged

        self.DIM = 384
        self.RES = 4

    @autocast(enabled=False)
    def forward(self, images, poses, disps, intrinsics, imu_data,
                M=1024, STEPS=12, P=1, structure_only=False):
        """Estimates SE3 or Sim3 between frame pairs with IMU conditioning."""
        images = 2 * (images / 255.0) - 0.5
        intrinsics = intrinsics / 4.0
        disps = disps[:, :, 1::4, 1::4].float()

        # IMU-conditioned patch selection
        (fmap, gmap, imap, patches, ix,
         selected_coords, covariances, p_dyn) = self.patchify(
            images, imu_data, poses_gt=(poses if structure_only else None)
        )

        # Remaining DPVO pipeline unchanged
        corr_fn = CorrBlock(fmap, gmap)
        # ... (same as original VONet.forward from line 196 onward)

        # KEY MODIFICATION: Pass covariances as per-patch weights in BA
        # Instead of equal weights, use Σ⁻¹ to weight each patch
        # In BA: weights = Σ⁻¹ (information matrix) rather than sigmoid confidence
```

### 8.4 Gumbel-Softmax Utility

```python
# Module/Network/PatchSelector/gumbel.py  (NEW FILE)

def gumbel_topk_st(logits, K, tau=0.5):
    """
    Straight-through Gumbel-Softmax Top-K.

    Forward:  hard top-K with Gumbel noise
    Backward: soft relaxation gradient

    Args:
        logits: [B, N]  log-scores for each candidate position
        K:      int     number to select
        tau:    float   temperature (lower = sharper)

    Returns:
        hard:    [B, N]  one-hot encoding of selected positions
        soft:    [B, N]  soft relaxation (for backward)
        indices: [B, K]  selected indices
    """
    # Gumbel noise
    U = torch.rand_like(logits)
    gumbel_noise = -(-U.log()).log()
    g = logits + gumbel_noise

    # Hard top-K
    _, indices = torch.topk(g, K, dim=1)
    hard = torch.zeros_like(logits)
    hard = hard.scatter(1, indices, 1.0)

    # Soft relaxation
    soft = F.softmax(logits / tau, dim=1)

    # Straight-through: forward uses hard, backward uses soft
    hard_with_grad = hard - soft.detach() + soft

    return hard_with_grad, soft, indices
```

---

## 9. Implementation Plan

### Task 1: IMU Slot Projector + IMUFusionEncoder
**Files:** `Module/Network/PatchSelector/__init__.py`, `Module/Network/PatchSelector/fusion.py`
**Test:** `Scripts/UnitTest/test_imu_fusion_encoder.py`

- [ ] Step 1: Write `FiLMGenerator`, verify output shapes with random inputs
- [ ] Step 2: Write `IMUSlotProjector`, verify shape `[B, N_slots, D_slot]`
- [ ] Step 3: Write `IMUFusionEncoder`, verify forward pass with synthetic IMU
- [ ] Step 4: Verify `alpha` init = 0 and `tanh(alpha)` gradient flows correctly
- [ ] Step 5: Write `IMUCrossAttn`, verify per-pixel attention maps

### Task 2: Three Prediction Heads
**Files:** `Module/Network/PatchSelector/heads.py`
**Test:** `Scripts/UnitTest/test_patch_selector_heads.py`

- [ ] Step 1: Write `SelectabilityHead`, verify output ∈ [0,1]
- [ ] Step 2: Write `DynamicHead`, verify accepts IMU disagreement input
- [ ] Step 3: Write `CovarianceHead`, verify Σ is PSD (check eigenvalues ≥ 0)
- [ ] Step 4: Test gradient flow through all heads independently

### Task 3: Gumbel-Softmax Top-K
**Files:** `Module/Network/PatchSelector/gumbel.py`
**Test:** `Scripts/UnitTest/test_gumbel_topk.py`

- [ ] Step 1: Write `gumbel_topk_st` function
- [ ] Step 2: Test forward returns correct K indices
- [ ] Step 3: Test backward gradients flow to input logits
- [ ] Step 4: Test temperature annealing schedule
- [ ] Step 5: Test spatial diversity loss

### Task 4: Full IMUPatchSelector
**Files:** `Module/Network/PatchSelector/imu_patch_selector.py`
**Test:** `Scripts/UnitTest/test_imu_patch_selector.py`

- [ ] Step 1: Integrate IMUFusionEncoder + Heads + GumbelTopK
- [ ] Step 2: Test patch extraction at selected coordinates
- [ ] Step 3: Test IMUContext wiring (seed + step loop)
- [ ] Step 4: Test end-to-end forward pass with synthetic data

### Task 5: Modified IMUVONet
**Files:** `Baseline/DPVO/dpvo/imunet.py`
**Test:** Integration test with a 2-frame TartanAir sample

- [ ] Step 1: Create `IMUVONet` subclassing from VONet
- [ ] Step 2: Wire IMUPatchSelector output into CorrBlock + Update
- [ ] Step 3: Modify BA to accept per-patch Σ⁻¹ weights
- [ ] Step 4: End-to-end smoke test with 2-frame forward pass

### Task 6: Training Script
**Files:** `Train/DPVOTrain/train_imu_selector.py`
**Test:** 100-step overfit test on a single sequence

- [ ] Step 1: Write Stage 1 warmup loop (encoder + flow loss)
- [ ] Step 2: Write Stage 2 head pretraining loop
- [ ] Step 3: Write Stage 3 differentiable BA loop
- [ ] Step 4: Overfit test: loss decreases on 1-sequence, 100 steps

### Task 7: Semi-Synthetic Data Pipeline
**Files:** `DataLoader/Synthetic/synthetic_dynamic.py`
**Test:** Visual inspection of generated frames

- [ ] Step 1: Write object patch extractor from Cityscapes
- [ ] Step 2: Write trajectory sampler
- [ ] Step 3: Write insertion engine with depth ordering
- [ ] Step 4: Write augmentation module
- [ ] Step 5: Generate and visually verify 100 training frames

---

## Appendix A: FiLM Formula Clarification

The standard FiLM paper (Perez et al. 2018) defines:

```
FiLM(x | γ, β) = γ ⊙ x + β
```

where γ and β are learned functions of the conditioning input. In practice, two variants exist:

| Variant | Formula | Initialization | Behavior at init |
|---------|---------|----------------|-----------------|
| **Standard** | `γ · x + β` | γ≈0 (from Linear), β≈0 | Near identity (γ≈0 → x≈0) |
| **Residual** | `(1+γ)·x + β` | γ≈0, β≈0 | Exact identity (γ=0 → x unchanged) |

DynGRU uses the standard variant. This design uses the **residual variant** because:

1. The FiLM generators are separate modules from the encoder backbone. At initialization, the encoder should produce meaningful features even without IMU conditioning.
2. The CLIP-adapter cross-attention is designed as an additive residual on top of the FiLM baseline. The baseline should be identity-preserving.
3. Empirically, residual FiLM produces more stable training in multi-stage pipelines.

## Appendix B: IMU Slot Count Rationale

DynGRU uses 7 explicit semantic slots (dR, dv, dp, g, bias, cov, dt) with separate MLP projections per slot. This design uses 8 learned slots with a shared projection. The trade-off:

- **Explicit slots (DynGRU):** Interpretable, guaranteed semantics, but potentially misses useful decompositions (e.g., slot for "turning + accelerating" might be useful).
- **Learned slots (this design):** Flexible, can discover useful decompositions from data, but risks slot collapse (mitigated by diversity regularizer).

8 slots are chosen as a reasonable starting point. Ablation: {4, 8, 12, 16} on the validation set.

## Appendix C: Comparison with Related Work

| System | Patch Selection | IMU Conditioning | Dynamic Handling | Metric Covariance |
|---|---|---|---|---|
| DPVO | Random ± gradient bias | None | None (static assumption) | None (sigmoid weight) |
| DROID-SLAM | Random + proximity | None | None | None |
| DynGRU | N/A (pixel-level) | FiLM + CLIP-adapter CA | Yes (per-pixel) | MAC-VO cov head |
| **Ours** | Learned (Gumbel Top-K) | FiLM + CLIP-adapter CA | Yes (per-pixel) | Yes (per-patch 2×2 Σ) |
