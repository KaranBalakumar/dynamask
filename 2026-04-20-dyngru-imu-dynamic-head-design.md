# DynGRU: IMU-Conditioned Dynamic-Pixel Head for MAC-VO

**Date:** 2026-04-20
**Status:** Design approved, ready for implementation planning
**Supersedes:** `2026-04-11-static-confidence-head-method-theory.md` (read-flowformer-context variant)
**Related:** `2026-04-11-static-confidence-head-macvo-spec.md` (infrastructure scaffolding — reused)

---

## 0. TL;DR

Replace the current post-hoc `StaticConfidenceHead` in MAC-VO with **DynGRU**: a ConvGRU-based decoder head that runs **co-iteratively with FlowFormer's K=12 decoder loop**, reads the same three tensors FlowFormer's own update GRU reads (`inp`, `motion_features`, `motion_features_global`), and is IMU-conditioned via a **two-stream scheme**: FiLM (cheap global γ,β modulation from a 34-D IMU global feature) plus a **CLIP-adapter-style single-head cross-attention** in which the IMU preintegration is tokenized into 7 semantic slots (ΔR, Δv, Δp, R^T·g, bias, Σ, dt) that the spatial feature map attends to. Both streams draw from the same **Velocity-EKF** (15-D state) that fuses AirIMU-corrected IMU with Air-IO body-frame velocity observations. The head emits per-pixel static confidence `c ∈ [0,1]` at H/4, which weights visual reprojection factors inside MAC-VO's existing **two-frame PGO**. A single 15-D IMU factor (Forster preintegration with bias Jacobians, EKF-derived covariance) is added to the same PGO graph. DRT-VIO-Init bootstraps gyro bias, gravity, extrinsic, and initial keyframe states.

Training is two-phase: **rigid-flow-residual pretrain** on TartanAir (using GT pose + GT depth to derive pseudo-static labels — no dynamic-mask GT assumed), then self-consistency finetune on EuRoC + KITTI-360 with a 3-frame static-consensus loss and a no-backprop-through-LM trick. Added inference cost over vanilla MAC-VO: ~9 ms/pair on RTX 4090. Added trainable params: ~2.7 M (FiLM + cross-attention adapter).

Staged escalation: if long-sequence drift is unacceptable post-training, promote to sliding-window VI-PGO (Stage B, out of scope).

---

## 1. Problem statement and scope

### 1.1 What's wrong with the current head

The current `StaticConfidenceHead` (at `Module/Network/DynamicHead/head.py`) reads FlowFormer's **final** flow + depth + a pooled context feature `phi8`, after the 12-iteration decoder loop has already burned its compute and committed to a flow field. Limitations:

1. **Information bottleneck:** the decoder's rich per-iter motion features are collapsed into a single flow field before the head sees them. Iteration-level dynamics (which pixels kept shifting their flow estimate vs. locked in early) is a strong dynamic-object cue, and it's gone by the time the head runs.
2. **No IMU.** The head has zero access to ego-motion priors. A rigid-world prediction of flow from IMU-integrated pose + depth is exactly the thing a dynamic-pixel detector wants to residual-compare against the estimated flow — but the current head has no way to form that comparison.
3. **Post-hoc weighting only.** The head weights factors in the backend but cannot influence the flow estimate itself. In dynamic scenes, FlowFormer's decoder is already being pulled toward moving-object flow; a downstream mask cannot undo that.

### 1.2 DynGRU design goals

- **Co-iterative** with FlowFormer's decoder. Reads the **same** three tensors FlowFormer's own update GRU reads, at each of the K=12 iterations, at H/8 resolution.
- **IMU-conditioned** via two complementary streams (§3.6): (i) FiLM using a 34-D global feature vector derived from a Velocity EKF (§2), (ii) a CLIP-adapter single-head cross-attention over 7 semantic IMU tokens (ΔR, Δv, Δp, R^T·g, bias, Σ, dt) with a gated residual initialized to zero. The feature encodes corrected angular velocity, linear acceleration, EKF-integrated pose delta, AirIO velocity prior, and their covariances — at the camera timestamp.
- **Temporal** — maintains a hidden state warped frame-to-frame via forward flow, in addition to intra-frame ConvGRU iteration.
- **Output** — per-pixel static confidence `c ∈ [0,1]` at H/4, upsampled from H/8 via a zero-init residual refine path.
- **Backend-consumable** — `c` directly weights MAC-VO's existing visual reprojection factors; no backend rewrite beyond adding one IMU factor.

### 1.3 Non-goals (explicitly out of scope)

- Sliding-window VI-PGO (Stage B).
- Online learning of IMU extrinsic / gravity (handled by DRT-init once).
- Semantic class labels for dynamic objects (binary static/dynamic only).
- Unfreezing FlowFormerCov backbone during main training (optional ablation only).
- Replacing the stereo depth network.

### 1.4 Success criteria

- **Phase A:** on TartanAir val, rigid-flow-residual agreement — pixels with rigid-flow residual < τ(D) should receive `c > 0.8` in ≥ 90% of cases; pixels with residual > 3·τ(D) should receive `c < 0.3` in ≥ 85% of cases. (Measured against derived pseudo-labels, not dynamic-mask GT.)
- **Phase B:** EuRoC MH_01–05 ATE ≤ baseline MAC-VO − 15%; KITTI-360 dynamic-scene ATE ≤ baseline MAC-VO − 25%.
- **Runtime:** ≤ 55 ms/pair on RTX 4090 (baseline MAC-VO: ~41 ms).

---

## 2. IMU pipeline: AirIMU → Velocity EKF → Air-IO (closed loop)

### 2.1 Why this shape, not parallel AirIMU + Air-IO

An earlier sketch of this design ran AirIMU and Air-IO as siblings. That's wrong. Air-IO's `CodeNetMotionwithRot.encoder(feature, ori)` **requires an orientation input** — it cannot run on raw IMU alone. The correct shape is a closed loop where:

1. **AirIMU** corrects raw (a, ω) → corrected IMU + per-sample covariance.
2. **Velocity EKF** propagates a 15-D state `(R, V, P, b_g, b_a)` using the corrected IMU. This gives orientation `R_t` at every IMU tick.
3. **Air-IO** consumes (raw IMU, `R_t` from the EKF) → body-frame velocity `v̂_B` + covariance.
4. **Velocity EKF** takes `v̂_B` as an observation and updates its state.
5. The EKF state at camera timestamps is sampled to build the 34-D `z_imu_global`.

This matches the reference Air-IO implementation (`references/Air-IO/EKF/IMUofflinerunner.py` `SingleIMU.propogate_update`).

### 2.2 EKF state, propagation, update

**State (15-D):** `x = [R, V, P, b_g, b_a]` where `R ∈ SO(3)` stored as quat internally, `V, P ∈ ℝ³` in world frame, `b_g, b_a ∈ ℝ³`.

**Propagation** (every IMU tick, 200 Hz): standard strapdown with corrected `(a, ω)` from AirIMU, gravity from DRT-init. Process-noise covariance comes from AirIMU's per-sample `(acc_cov, gyro_cov)` plus fixed bias random walk.

**Velocity update** (every AirIO output, typically 10 Hz — matches camera rate): innovation = `R_t^T · V − v̂_B`, H matrix has −1 on velocity block and a skew-symmetric block on R; R matrix = Air-IO's predicted covariance, clipped.

**Bias-ref sync:** after each two-frame PGO solve (§5), optimized `(b_g, b_a)` is pushed back to the EKF as the new bias point with covariance rows reset. This prevents EKF-PGO bias divergence (standard ROVIO trick).

### 2.3 Constructing `z_imu_global` (34-D)

At camera timestamp `t_{k+1}`, after the EKF has propagated+updated over `[t_k, t_{k+1}]`, sample:

| Field | Dim | Source |
|---|---|---|
| `ΔR` (log) | 3 | `Log(R_k^T · R_{k+1})` from EKF |
| `Δv` | 3 | EKF |
| `Δp` | 3 | EKF |
| `diag(Σ_{ΔR,Δv,Δp})` | 9 | EKF covariance |
| `v̂_B` (AirIO) | 3 | Air-IO output at `t_{k+1}` |
| `diag(Σ_{v̂_B})` | 3 | Air-IO cov |
| `b_g, b_a` | 6 | EKF |
| `dt` | 1 | `t_{k+1} − t_k` |
| `R_{k+1}^T · g_W` (body-frame gravity at cam time) | 3 | EKF rotation + DRT-init gravity |
| **Total** | **34** | |

The 34-D vector serves two downstream consumers:

1. **Global feature `f_imu ∈ ℝ^128`** — produced by a small MLP (34 → 128 → 128). Feeds the FiLM layer of every `DynUpdateBlock` (global γ,β modulation).
2. **Token tensor `imu_tokens ∈ ℝ^{B×7×128}`** — produced by 7 per-token projections, one per semantic slot. Slot layout and per-slot input widths:

| Token idx | Semantic slot | Input dim | Projection |
|---|---|---|---|
| 0 | ΔR (log) | 3 | `Linear(3, 128)` |
| 1 | Δv | 3 | `Linear(3, 128)` |
| 2 | Δp | 3 | `Linear(3, 128)` |
| 3 | R^T·g_W | 3 | `Linear(3, 128)` |
| 4 | (b_g, b_a) | 6 | `Linear(6, 128)` |
| 5 | diag(Σ) | 9 | `Linear(9, 128)` |
| 6 | dt | 1 | `Linear(1, 128)` |

Tokens feed the cross-attention adapter inside every `DynUpdateBlock` (§3.6). Note that `v̂_B` and `diag(Σ_{v̂_B})` from Air-IO remain part of the 34-D global feature that feeds FiLM, but they do not get their own dedicated attention token — their effect on the EKF state (primarily via the velocity update, which then propagates into `Δv, Δp, Σ`) is already captured by the 7 EKF-derived tokens, so adding them as separate slots would be redundant and would inflate the token count.

### 2.4 What DRT-init provides (one-time, at startup)

- `b_g*`: initial gyro bias (EKF seeds `b_g ← b_g*`).
- `R_BC, t_BC`: IMU↔cam extrinsic (treated as known from calib; DRT refines).
- `g_W`: gravity direction in world frame.
- Initial keyframe states `(R_0, v_0, p_0)` — seed EKF and first PGO anchor.

DRT-loose runs on the first ~1s of stereo + IMU, then we're in steady-state.

---

## 3. DynGRU architecture

### 3.1 Where it taps into FlowFormer

Inside `MemoryDecoder.forward()` loop (at `Module/Network/FlowFormer/core/decoder.py`), at every iteration `k ∈ [0, K)`, `GMAUpdateBlock` computes three tensors at H/8:

```python
motion_features_k        = self.encoder(flow, corr)           # 128 ch
motion_features_global_k = self.aggregator(attention, motion_features_k)  # 128 ch (GMA output)
inp_cat_k                = torch.cat([inp, motion_features_k, motion_features_global_k], dim=1)  # 384 ch
net = self.gru(net, inp_cat_k)
```

DynGRU reads the same `{inp, motion_features_k, motion_features_global_k}` with a **read-only hook** (no gradient flow back into FlowFormer unless that ablation is explicitly turned on). FlowFormer weights stay frozen in Phase A; optional unfreeze of the last decoder block in Phase B.

### 3.2 Local-vs-global motion deviation as implicit signal

The core dynamic-vs-static cue available to the decoder is the difference between per-pixel `motion_features_k` and globally-attended `motion_features_global_k`. For a rigid pixel the two agree; for a dynamic pixel, local motion deviates from globally-attended motion. We do **not** form this as an explicit `Δmotion` tensor — instead, by feeding both through the same `inp_cat = [flow_inp, motion_feat, motion_feat_global]` that CovGRU already consumes (§6.3), the `SepConvGRU` can learn a deviation-sensitive representation from the first 1×1 conv of its gates. Keeping the GRU input shape identical to CovGRU's is a deliberate design choice: it preserves the exact integration pattern MAC-VO uses for the parallel cov branch and avoids a second, structurally different GRU inside the same decoder loop.

### 3.3 Cell structure (H/8, per iteration — mirrors CovGRU)

Matching `CovUpdateBlock` exactly in shape, with two added IMU-conditioning sublayers (FiLM for global modulation, a gated cross-attention adapter for spatial selectivity — see §3.6) and a 1-logit classifier instead of a 2-channel cov head.

```
Inputs per iter k:
  flow_inp            : 128 ch H/8     (FlowFormer's context branch, already proj'd)
  motion_k            : 128 ch H/8     (from update_block.encoder(flow, corr))
  motion_g_k          : 128 ch H/8     (from update_block.aggregator(attention, motion_k))
  inp_cat             : 384 ch H/8     (cat of the three — IDENTICAL to flow/cov GRU input)
  f_imu               : 128-D          (from IMUPipeline, broadcast spatially for FiLM)
  imu_tokens          : 7 × 128        (from IMUPipeline, semantic IMU slots for cross-attn)
  dyn_net_{k-1}       : 128 ch H/8     (DynGRU hidden state, warped from prev frame pair at k=0)

1. dyn_net_k = SepConvGRU(dyn_net_{k-1}, inp_cat)                      # identical to cov_update.gru call
2. dyn_net_k = FiLMLayer(dyn_net_k, f_imu)                             # per-channel γ·h + β from MLP(f_imu)
3. dyn_net_k = dyn_net_k + tanh(α) · IMUCrossAttn(dyn_net_k, imu_tokens)  # CLIP-adapter gated residual; α init 0
4. delta_dyn_k = DynHead(dyn_net_k)                                    # 4-conv stack → 1 ch H/8 logits, clamped to [-10, 10]
5. dyn_mask_k  = 0.25 · mask_conv(dyn_net_k)                           # RAFT convex-combination upsample weights
6. dyn_up_k    = upsample_convex(delta_dyn_k, dyn_mask_k)              # H/8 → H/4 (2× convex, not the 8× used for flow)
7. Store dyn_up_k in dyn_predictions[k] for deep supervision.
```

At the end of the K-iteration loop, `dyn_predictions[-1]` is the final H/4 logit map. Sigmoid with a learned temperature buffer gives `c ∈ [0,1]`. No separate "finalize" path: the per-iter head is the only head, exactly as `CovHead` works in CovGRU.

### 3.4 Two-timescale temporal state

- **Intra-frame (fast):** `dyn_net` updates at each of K=12 decoder iterations.
- **Inter-frame (slow):** at the start of a new frame pair, `dyn_net_init` is the previous pair's `dyn_net_{K-1}` **warped forward by the current forward flow** (bilinear sample, zeroed where warping goes out of bounds). This propagates per-pixel dynamic-belief across frames without a separate recurrent network. First frame: `dyn_net_init = flow_net.clone()` (same init CovGRU uses for `fcov_net`).

### 3.5 Parameter budget

| Block | Params | Source |
|---|---|---|
| `DynUpdateBlock.gru` (SepConvGRU, hidden=128, input=384) | ~1.4 M | mirrors `CovUpdateBlock.gru` |
| `DynUpdateBlock.film` (FiLM: 128 → 256 split into γ,β) | ~0.03 M | new (conditioning) |
| `DynUpdateBlock.imu_attn` (IMUCrossAttn: Q/K/V linears, 128→128) | ~0.05 M | new (§3.6) |
| `DynUpdateBlock.alpha` (scalar gate) | ~0.000001 M | new (§3.6) |
| `DynHead` (4-conv stack, 128 → 256 → 128 → 64 → 1) | ~0.7 M | mirrors `CovHead` with 1-ch output |
| `DynUpdateBlock.mask` (128 → 256 → 576) | ~0.45 M | mirrors `CovUpdateBlock.mask` |
| `IMUPipeline.feature_mlp` (34 → 128 → 128) | ~0.02 M | new (§2.3) |
| `IMUPipeline.token_proj` (7 per-slot `Linear(·, 128)`) | ~0.004 M | new (§2.3 token table) |
| **Total trainable** | **~2.65 M** | |

Frozen upstream: FlowFormerCov (16 M), AirIMU corrector (1.1 M), Air-IO (0.9 M). The adapter adds only ~50k params but materially increases the expressive power of IMU conditioning, so in practice we quote ~2.7 M in §0.

### 3.6 IMU conditioning: FiLM + CLIP-adapter cross-attention

DynGRU uses two complementary IMU-conditioning streams. FiLM alone cannot encode that different pixels care about different components of the IMU signal (e.g., pixels near the epipole are more sensitive to Δp than to R^T·g), because FiLM applies the same (γ, β) at every spatial location. Cross-attention alone loses the cheap global gating FiLM provides and forces the attention block to relearn it. The two together are complementary: FiLM handles global modulation, the adapter adds spatially-selective nuance.

#### 3.6.1 Motivation

- **FiLM (global, cheap, uniform):** `x' = γ·x + β` with (γ, β) produced from `f_imu`. Same modulation everywhere — correct when the conditioner is uniform in effect (e.g., "we're rotating fast, trust IMU more").
- **Cross-attention (local, selective):** each spatial location of `dyn_net` chooses how to attend to each semantic IMU slot. Correct when the conditioner's effect is spatially varying (e.g., a pixel near the focus-of-expansion responds to Δp differently than a pixel at the image edge).
- **CLIP-adapter trick:** gate the attention contribution by `tanh(α)` with `α` initialized to 0. The block starts indistinguishable from FiLM-only and can only add spatial selectivity during training — no cold-start regression.

#### 3.6.2 Token construction

`IMUPipeline` emits `imu_tokens ∈ ℝ^{B×7×128}` via 7 independent `Linear` projections from the per-slot inputs listed in §2.3. Tokens are semantic, not positional: each slot has a fixed identity the network can learn. No positional encoding is added (7 tokens, distinct input provenance — positional encoding would only add noise).

#### 3.6.3 Cross-attention math (single head)

Let `H = dyn_net ∈ ℝ^{B×128×H/8×W/8}`. Define spatial query, token key/value:

```
Q = reshape(H, (B, HW, 128)) · W_Q      # (B, HW, 128)
K = imu_tokens · W_K                    # (B, 7, 128)
V = imu_tokens · W_V                    # (B, 7, 128)
A = softmax(Q · K^T / √128)             # (B, HW, 7)  — per-pixel attention over 7 slots
Δ = A · V                               # (B, HW, 128)
Δ = reshape(Δ, (B, 128, H/8, W/8))
H' = FiLM(H, f_imu) + tanh(α) · Δ       # α is a learned scalar, init 0
```

Single head (not multi-head): 7 tokens is too few to split meaningfully; the 7 learned input projections already give per-token linear capacity. Temperature is fixed at `√128` (standard). No dropout in the attention softmax — attention weights are only 7-wide so dropout gives a very noisy gradient.

#### 3.6.4 Compute cost

Per update block invocation: attention MACs = `B · HW · 7 · 128` for QK^T and `B · HW · 7 · 128` for AV, plus three linear projections (`W_Q, W_K, W_V`). At H/8 = 60×80 for a 480×640 input, HW = 4800, so ≈ 8.6M MACs per iter × 12 iters ≈ 100M MACs added total — roughly 0.5 ms on RTX 4090. Negligible compared to FlowFormer's ~32 ms.

#### 3.6.5 Initialization

- `W_Q, W_K, W_V`: standard `xavier_uniform`.
- `α`: `nn.Parameter(torch.zeros(1))` — **critical** for the gated-residual property.
- Token projections: standard linear init.
- FiLM is unchanged from the earlier design.

#### 3.6.6 Optional future extensions (not in Phase A/B)

- Replace single-head with small multi-head (e.g., 4 heads × 32 dim) once 7 → ≥16 tokens (e.g., by tokenizing per-axis instead of per-vector).
- Add a learnable `[CLS]` IMU token that absorbs all-slot context, then concatenate with the 7 semantic tokens.
- Per-layer `α` (12 scalars) instead of one shared scalar — enables the adapter to be strong in late iters, soft in early iters.

These are tracked in §7.4 ablations as "adapter variants"; defer until Phase B is stable.

---

## 4. Supervision and losses

### 4.1 Why hybrid derived-pseudo-GT + self-consistency, not one or the other

**No dynamic-mask GT is assumed to exist** for any dataset we use. Phase A instead derives pseudo-static labels from GT pose + GT depth via rigid-flow residual: a pixel is pseudo-static when its estimated flow agrees with the rigid-flow reprojection predicted by GT pose + GT depth, within a depth-adaptive threshold. This transfers the strong pose/depth supervision signal into a per-pixel rigidity cue without needing a direct dynamic mask. Pure self-supervision is too weak a signal to train a ~2.7M-param head from scratch; derived pseudo-GT bridges the gap.

Phase B then closes the synthetic-to-real gap without needing any per-pixel supervision.

### 4.2 Phase A: rigid-flow-residual pretrain (≈8 epochs)

**Data:** TartanAir (full, all sequences — uses GT pose + GT depth only, no mask GT). IMU synthesized from GT pose + injected bias/noise profiles from a standard AirIMU training pipeline.

**Pseudo-static label construction (per pixel, per frame pair):**

```
# Predicted rigid flow from GT pose + GT depth:
f_rigid(i) = project(K · T_GT · D_GT(i) · K^-1 · [u_i; v_i; 1]) − [u_i; v_i]
# Estimated flow from FlowFormer at final iter:
f_est(i)   = model output

# Depth-adaptive threshold:
τ(D_i) = τ_0 + α · (f/D_i) · ||t_GT||      # τ_0 = 0.5 px, α = 0.3

residual_i = ||f_est(i) − f_rigid(i)||
M_pseudo(i) = 1 if residual_i < τ(D_i)       # likely static
            = 0 if residual_i > 3·τ(D_i)     # likely dynamic
            = IGNORE otherwise               # ambiguous band, excluded from loss
```

Pixels failing forward-backward flow consistency (occlusion) are also set to IGNORE.

**Loss per frame pair:** sum over decoder iterations k ∈ [0, K), γ-weighted with `γ_k = 0.85^(K−1−k)`:

```
L_k = FocalBCE(sigmoid(ℓ_k), M_pseudo, ignore=IGNORE)   # focal α=0.25, γ=2
L_A = Σ_k γ_k · L_k
```

Rationale: the rigid-flow residual is a **per-pixel rigidity test** that requires only pose + depth GT, both available in TartanAir. The IGNORE band deliberately skips ambiguous pixels rather than forcing a hard label on them.

### 4.3 Phase B: self-consistency finetune (≈3 epochs)

**Data:** EuRoC + KITTI-360 (real IMU, real stereo). No dynamic GT.

**Losses:**
- `L_pose` (weight 1.0): GT-pose reprojection residual, `c`-weighted. Pixels with high `c` should reproject well under GT pose.
- `L_consistency` (weight 0.5): 3-frame static-consensus. If pixel `i` in frame t is labeled static by DynGRU, and pixel `i'` (warped via estimated flow) in frame t+1 is also labeled static, and again at t+2, the three should agree on the rigid-world reprojection within 2 px. Violations penalize inconsistency.
- `L_sparsity` (weight 0.01): `mean(1 − c)` prior against marking-everything-static.
- `L_entropy` (weight 0.05): `c·log c + (1−c)·log(1−c)` mean — encourages decisive (0 or 1) predictions, discourages hedging at 0.5.

**Critical trick:** gradients do **not** flow through the PGO LM solver. The PGO runs in-loop (so `c` affects reprojection residuals used in `L_pose`), but we detach at the LM output. Only the residual-reweighting path carries gradient. This is the standard RAFT-VO / DROID-SLAM approach and avoids implicit-differentiation costs.

### 4.4 Optimizer, schedules

- AdamW, lr 2e-4 cosine → 1e-5, weight decay 1e-4.
- Phase A: batch 8, 2× A6000, ~8 epochs, ~60 h.
- Phase B: batch 4 (longer sequences), same GPUs, ~3 epochs, ~25 h.
- Optional Phase-B ablation: unfreeze last FlowFormer decoder block, lr 1e-6 on those params.

---

## 5. Backend: two-frame PGO + one IMU factor (Option II)

### 5.1 What changes in the factor graph

Current MAC-VO two-frame PGO has reprojection + stereo disparity factors. Option II adds **exactly one** new factor per pair:

**15-D IMU factor** — `r_IMU = [r_ΔR, r_Δv, r_Δp, r_bg, r_ba]`:
- `r_ΔR = Log(ΔR̃ · Exp(J_R_bg · δb_g)^T · R_k^T · R_{k+1})`
- `r_Δv = R_k^T · (v_{k+1} − v_k − g·dt) − (Δṽ + J_v_bg·δb_g + J_v_ba·δb_a)`
- `r_Δp = R_k^T · (p_{k+1} − p_k − v_k·dt − 0.5·g·dt²) − (Δp̃ + J_p_bg·δb_g + J_p_ba·δb_a)`
- `r_bg = b_g,{k+1} − b_g,k`
- `r_ba = b_a,{k+1} − b_a,k`

Where `(ΔR̃, Δṽ, Δp̃, J_*)` come from the EKF's preintegration-equivalent over `[t_k, t_{k+1}]`, and info matrix = `(EKF Σ_{k+1})^{-1}` clipped to a max condition number.

### 5.2 Other changes

- **Visual reprojection factors** — each per-pixel residual is now weighted by `c_i` (in addition to the existing per-pixel σ from FlowFormerCov). Pixels with `c_i < 0.1` are dropped entirely (cheaper than full-precision downweighting).
- **Bias-ref sync loop** — post-solve, push optimized `(b_g, b_a)` back to EKF (§2.2).
- **Everything else unchanged** — keyframe management, pypose LM solver, stereo factors, optimization schedule.

### 5.3 Data flow (single frame pair, inference)

```
IMU buffer [t_k, t_{k+1}] (200 Hz)
  → AirIMU corrector          → corrected (a, ω) + per-sample cov
  → Velocity EKF propagate    → state (R,V,P,b_g,b_a) at each tick
  → Air-IO (raw IMU + EKF R)  → v̂_B + Σ_v @ 10 Hz
  → EKF velocity update       → corrected state
  → sample @ t_{k+1}          → 34-D z_imu_global + preintegration deltas

Stereo pair @ t_{k+1} + prev
  → FlowFormer encoder (frozen)
  → FlowFormer decoder loop (K=12 iters)
       ├→ per-iter: {inp, motion_k, motion_g_k}
       └→ final: flow + uncertainty
  → DynGRU co-iterates with decoder, conditioned on (f_imu, imu_tokens)
      - FiLM uses f_imu = feature_mlp(z_imu_global)
      - CLIP-adapter cross-attn uses imu_tokens = 7 semantic slots of 128-D
  → c_{k+1} at H/4

Two-frame PGO (pypose LM):
  - reprojection factors (c-weighted)
  - stereo factors
  - IMU factor (EKF-derived, Forster form)
  → optimized pose, velocity, biases
  → push biases back to EKF
```

### 5.4 Compute budget (inference, RTX 4090)

| Component | Time |
|---|---|
| FlowFormerCov fwd (K=12) | ~32 ms |
| AirIMU corrector | ~0.4 ms |
| EKF (200 Hz over 100 ms window) | ~0.3 ms |
| Air-IO inference | ~0.8 ms |
| DynGRU co-iter (12 iters) | ~4 ms |
| Two-frame PGO + IMU factor (LM ~6 iters) | ~12 ms |
| **Total** | **~50 ms (~20 Hz)** |

Baseline MAC-VO: ~41 ms. Added ~9 ms. On Jetson Orin AGX expect ~2× slower → ~10 Hz (original MAC-VO target).

### 5.5 Staged escalation

If post-training ATE on long sequences (KITTI-360 full runs, > 3 km) exceeds target, promote to sliding-window VI-PGO (Stage B). The `r_IMU` residual code ports verbatim — only marginalization (Schur complement to drop oldest keyframe) and keyframe management are new. Stage B is explicitly out of scope for this spec.

---

## 6. Module structure and interfaces

### 6.1 Follow the existing CovGRU pattern

MAC-VO already extends FlowFormer in exactly the way DynGRU needs to integrate. The existing `MemoryCovDecoder` (`Module/Network/FlowFormerCov/covhead.py`) subclasses FlowFormer's `MemoryDecoder`, adds a sibling `CovUpdateBlock` (with its own `SepConvGRU`, `CovHead`, and RAFT-style upsample mask), runs it co-iteratively in the decoder loop consuming the **same** `inp_cat = [flow_inp, motion_feat, motion_feat_global]` as the flow GRU, and returns a parallel `cov_predictions` sequence alongside `flow_predictions`.

DynGRU follows the identical pattern: a `DynUpdateBlock` (sibling to `CovUpdateBlock`), a `MemoryDynDecoder` subclassing `MemoryCovDecoder`, and a `FlowFormerDyn` subclassing `FlowFormerCov` that returns `(flow_predictions, cov_predictions, dyn_predictions)`. The only structural deviations from CovGRU: `DynUpdateBlock` additionally takes `f_imu` (for FiLM) and `imu_tokens` (for the cross-attention adapter, §3.6), both sourced from `IMUPipeline`. CovGRU has no conditioning input.

### 6.2 New / modified files

```
Module/Network/
  FlowFormerDyn/                      NEW   (mirrors FlowFormerCov/)
    __init__.py                       NEW   build_flowformer_dyn(cfg, ...)
    flownet.py                        NEW   class FlowFormerDyn(FlowFormerCov)
    dynhead.py                        NEW   class DynHead, DynUpdateBlock, MemoryDynDecoder

  AirIMU/
    corrector.py                      KEEP (frozen ckpt)
    preintegration.py                 KEEP (Forster-form bias Jacobians for IMU factor)
    encoder.py                        REWRITE   IMUEncoder now samples 34-D from EKF state
    ekf.py                            NEW   15-D velocity EKF (port from references/Air-IO/EKF)
    airio.py                          NEW   wrapper around CodeNetMotionwithRot

  DynamicHead/                        DEPRECATED   (post-hoc head kept only as ablation baseline)

Pipeline/
  vo_pipeline.py                      MODIFY   instantiate IMUPipeline + FlowFormerDyn; wire EKF bias-sync
  backend_pgo.py                      MODIFY   add 15-D IMU factor; c-weighted reprojection
```

### 6.3 Module shapes (mirror of CovGRU)

All four classes parallel (and extend) the CovGRU trio. `DynHead` matches `CovHead` structurally; `IMUCrossAttn` is the adapter (§3.6.3); `DynUpdateBlock` matches `CovUpdateBlock` with added FiLM + `IMUCrossAttn` + learnable gate `α`; `MemoryDynDecoder` extends `MemoryCovDecoder` with a third predictions list and threads `(f_imu, imu_tokens)` through the loop.

```python
# Module/Network/FlowFormerDyn/dynhead.py

from ..FlowFormer.core.gru import SepConvGRU
from ..FlowFormerCov.covhead import MemoryCovDecoder


class DynHead(nn.Module):
    """Per-iteration 1-logit head at H/8, mirrors CovHead structure."""
    def __init__(self, input_dim: int = 128, hidden_dim: int = 256):
        super().__init__()
        self.conv1 = nn.Conv2d(input_dim, hidden_dim, 3, padding=1)
        self.conv2 = nn.Conv2d(hidden_dim, hidden_dim // 2, 3, padding=1)
        self.conv3 = nn.Conv2d(hidden_dim // 2, hidden_dim // 4, 3, padding=1)
        self.conv4 = nn.Conv2d(hidden_dim // 4, 1, 3, padding=1)   # 1 logit (static/dynamic)
        self.relu  = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv2(self.relu(self.conv1(x)))
        x = self.conv4(self.relu(self.conv3(x)))
        return x


class IMUCrossAttn(nn.Module):
    """Single-head cross-attention: spatial feature map (Q) attends to 7 semantic IMU tokens (K,V)."""
    def __init__(self, feat_dim: int = 128, n_tokens: int = 7):
        super().__init__()
        self.n_tokens = n_tokens
        self.scale = feat_dim ** -0.5
        self.q_proj = nn.Linear(feat_dim, feat_dim)
        self.k_proj = nn.Linear(feat_dim, feat_dim)
        self.v_proj = nn.Linear(feat_dim, feat_dim)

    def forward(self, h: torch.Tensor, imu_tokens: torch.Tensor) -> torch.Tensor:
        B, C, H, W = h.shape
        q = self.q_proj(h.flatten(2).transpose(1, 2))              # (B, HW, C)
        k = self.k_proj(imu_tokens)                                # (B, N, C)
        v = self.v_proj(imu_tokens)                                # (B, N, C)
        attn = torch.softmax(q @ k.transpose(-2, -1) * self.scale, dim=-1)  # (B, HW, N)
        out = attn @ v                                             # (B, HW, C)
        return out.transpose(1, 2).reshape(B, C, H, W)


class DynUpdateBlock(nn.Module):
    """Sibling of CovUpdateBlock: own SepConvGRU + DynHead + upsample mask, IMU-conditioned via FiLM + CLIP-adapter."""
    def __init__(self, args, hidden_dim: int = 128, imu_dim: int = 128, n_imu_tokens: int = 7):
        super().__init__()
        self.args = args
        # Same inp_cat width as flow/cov GRUs: 128 (flow_inp) + 128 (motion) + 128 (motion_g) = 384
        self.gru  = SepConvGRU(hidden_dim=hidden_dim, input_dim=128 + hidden_dim + hidden_dim)
        # FiLM: cheap global modulation from f_imu (§3.6.1)
        self.film = FiLMLayer(cond_dim=imu_dim, feat_dim=hidden_dim)
        # CLIP-adapter: spatially selective attention over 7 semantic IMU tokens (§3.6.3)
        self.imu_attn = IMUCrossAttn(feat_dim=hidden_dim, n_tokens=n_imu_tokens)
        # Gated residual scalar — init 0 so block starts as pure FiLM (§3.6.5)
        self.alpha = nn.Parameter(torch.zeros(1))
        self.dyn_head = DynHead(hidden_dim, hidden_dim=256)
        self.mask = nn.Sequential(
            nn.Conv2d(hidden_dim, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 64 * 9, 1, padding=0),                  # RAFT 8× upsample mask
        )

    def forward(
        self,
        dyn_net: torch.Tensor,
        inp_cat: torch.Tensor,
        f_imu: torch.Tensor,
        imu_tokens: torch.Tensor,                                  # (B, 7, 128)
    ):
        dyn_net   = self.gru(dyn_net, inp_cat)                     # co-iterative update
        dyn_net   = self.film(dyn_net, f_imu) \
                  + torch.tanh(self.alpha) * self.imu_attn(dyn_net, imu_tokens)  # FiLM + gated adapter
        delta_dyn = self.dyn_head(dyn_net)                         # 1-ch logits @ H/8
        mask      = 0.25 * self.mask(dyn_net)                      # RAFT convex upsample
        return dyn_net, delta_dyn, mask


class MemoryDynDecoder(MemoryCovDecoder):
    """Extends MemoryCovDecoder with a third sibling update branch. Keeps flow + cov behavior unchanged."""
    def __init__(self, cfg, decoder_dtype: torch.dtype):
        super().__init__(cfg, decoder_dtype)
        self.dyn_update = DynUpdateBlock(self.cfg, hidden_dim=128, imu_dim=128)
        self.dyn_update = self.dyn_update.to(dtype=self.decoder_dtype)

    def forward(self, cost_memory, context, cost_maps, f_imu, imu_tokens):   # extra args: f_imu, imu_tokens
        # Body mirrors MemoryCovDecoder.forward up to inp_cat construction, then adds a
        # parallel `dyn_net` branch updated inside the same `for _ in range(self.depth)` loop
        # with the same `inp_cat`, plus `f_imu` for FiLM and `imu_tokens` for the adapter.
        ...  # identical setup: initialize_flow, proj, split(flow_net, flow_inp), attention, etc.
        fdyn_net = flow_net.clone().to(dtype=self.decoder_dtype)    # init from tanh'd context half, like fcov_net

        for _ in range(self.depth):
            # ... identical block for encoding cost token, GMA update, flow/cov GRU ...
            with torch.cuda.nvtx.range("Dyn Update Block"):
                fdyn_net, delta_dyn, dyn_mask = self.dyn_update(fdyn_net, inp_cat, f_imu, imu_tokens)
            # Upsample dyn at H/8 → H/4 using RAFT convex combination (not 8× — we stop at H/4 per §3.3)
            # Implementation detail: reuse a 2× variant of self.upsample_flow or a dedicated conv.
            dyn_predictions.append(upsample_dyn_logits(delta_dyn, dyn_mask))

        if self.training:
            return flow_predictions, cov_predictions, dyn_predictions
        return (flow_predictions[-1], ...), (cov_predictions[-1], ...), (dyn_predictions[-1],)
```

```python
# Module/Network/FlowFormerDyn/flownet.py

from ..FlowFormerCov.flownet import FlowFormerCov
from .dynhead import MemoryDynDecoder


class FlowFormerDyn(FlowFormerCov):
    def __init__(self, cfg, encoder_dtype=torch.float32, decoder_dtype=torch.float32):
        super().__init__(cfg, encoder_dtype, decoder_dtype)
        # Replace the cov-only decoder with the dyn-extended variant:
        self.memory_decoder = MemoryDynDecoder(self.cfg, decoder_dtype)

    def forward(self, image1, image2, f_imu, imu_tokens):
        # Identical encoder path to FlowFormerCov, then:
        flow_predictions, cov_predictions, dyn_predictions = self.memory_decoder(
            cost_memory, context, cost_maps, f_imu, imu_tokens
        )
        return flow_predictions, cov_predictions, dyn_predictions
```

### 6.4 IMU pipeline interface

The visual side stays Pythonically thin (mirrors FlowFormerCov). The IMU pipeline is a single class that composes AirIMU + EKF + Air-IO:

```python
# Module/Network/AirIMU/encoder.py  (rewrite)

class IMUPipeline(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.corrector = AirIMUCorrector.from_ckpt(cfg.airimu_ckpt)      # frozen
        self.airio     = AirIOWrapper.from_ckpt(cfg.airio_ckpt)          # frozen
        self.ekf       = VelocityEKF(cfg.ekf)                            # stateful, analytic

        # Global 34-D → 128 feature for FiLM.
        self.feature_mlp = nn.Sequential(
            nn.Linear(34, 128), nn.ReLU(True),
            nn.Linear(128, 128),
        )
        # Per-slot token projections for the CLIP-adapter cross-attention (§2.3, §3.6.2).
        # Order must match the cross-attn's expected token layout.
        self.token_projs = nn.ModuleDict({
            "dR":   nn.Linear(3, 128),   # so3_log(ΔR)
            "dv":   nn.Linear(3, 128),   # Δv
            "dp":   nn.Linear(3, 128),   # Δp
            "g":    nn.Linear(3, 128),   # R^T · g_W
            "bias": nn.Linear(6, 128),   # (b_g, b_a)
            "cov":  nn.Linear(9, 128),   # diag(Σ)
            "dt":   nn.Linear(1, 128),   # scalar dt, projected
        })

    def step(self, imu_window: dict, t_cam: float) -> IMUSample:
        """One camera-tick call. Propagates EKF, runs Air-IO, updates EKF, samples global feature + tokens."""
        corr = self.corrector.inference(imu_window)
        self.ekf.propagate(corr.acc_corrected, corr.gyro_corrected, corr.cov, imu_window["dt"])
        v_B, Sigma_v = self.airio(imu_window["raw"], self.ekf.R_at(t_cam))
        self.ekf.update_velocity(v_B, Sigma_v, t_cam)

        slots = self._build_slots(t_cam)                                 # dict of per-slot tensors
        z = torch.cat([slots["dR"], slots["dv"], slots["dp"],
                       slots["cov"], slots["v_B"], slots["sig_v"],
                       slots["bias"], slots["dt"], slots["g"]], dim=-1)  # 34-D
        f_imu = self.feature_mlp(z)                                      # (B, 128)  — FiLM input

        imu_tokens = torch.stack([                                       # (B, 7, 128)  — cross-attn input
            self.token_projs["dR"](slots["dR"]),
            self.token_projs["dv"](slots["dv"]),
            self.token_projs["dp"](slots["dp"]),
            self.token_projs["g"](slots["g"]),
            self.token_projs["bias"](slots["bias"]),
            self.token_projs["cov"](slots["cov"]),
            self.token_projs["dt"](slots["dt"]),
        ], dim=1)

        return IMUSample(f_imu=f_imu, imu_tokens=imu_tokens,
                         preint=self.ekf.preint_summary(), z=z)

    def push_biases(self, b_g: torch.Tensor, b_a: torch.Tensor) -> None:
        """Called post-PGO to sync optimized biases back into the EKF."""
        self.ekf.reset_biases(b_g, b_a)
```

### 6.5 Pipeline wiring (one frame pair)

```python
# Pipeline/vo_pipeline.py  (modify)

imu_sample = imu_pipeline.step(imu_window, t_cam)                        # §2, emits f_imu + imu_tokens
flow_pre, cov_pre, dyn_pre = flowformer_dyn(                             # §6.3
    image1, image2, imu_sample.f_imu, imu_sample.imu_tokens,
)
c = torch.sigmoid(dyn_pre[-1])                                           # H/4 static confidence

pose, v, b_g, b_a = backend_pgo.solve(
    pixels=pixels, flow=flow_pre[-1], cov=cov_pre[-1], c=c,              # c-weighted reprojection
    imu_factor=ImuFactor(imu_sample.preint),                             # §5
)
imu_pipeline.push_biases(b_g, b_a)                                       # bias-sync
```

This keeps the visual stack's extension style identical to how CovGRU was added on top of FlowFormer, and the IMU stack cleanly isolated behind a single `IMUPipeline` facade.

---

## 7. Testing and validation plan

### 7.1 Unit tests

- `test_ekf_propagation.py` — consistency with pypose preintegration under zero bias noise.
- `test_airimu_encoder.py` — 34-D feature vector shape + finite values across edge dt.
- `test_dyngru_cell.py` — gradient flows cleanly; with `α = 0`, `DynUpdateBlock` output ≡ FiLM-only output at step 0 (validates the CLIP-adapter gated-residual property).
- `test_imu_cross_attn.py` — `IMUCrossAttn` output shape matches input spatial dims; attention weights sum to 1 over the 7-token axis.
- `test_imu_factor_jacobians.py` — numerical Jacobian match vs analytic for `r_ΔR, r_Δv, r_Δp`.

### 7.2 Integration tests

- `test_pipeline_euroc_mh01.py` — 10-second slice of MH_01 runs end-to-end, produces finite poses.
- `test_bias_sync.py` — EKF and PGO bias agree within 1e-3 after 100 frames.
- `test_dyngru_pseudo_label_agreement.py` — on a TartanAir val sample, `c` agrees with the rigid-flow-residual pseudo-label on non-IGNORE pixels ≥ 80% after Phase A init.

### 7.3 Benchmarks

- EuRoC MH_01–05, V1_01–03, V2_01–03: ATE vs baseline MAC-VO and vs MAC-VO + current StaticConfidenceHead.
- KITTI-360: dynamic-scene subset, same metrics.
- TartanAir val: pseudo-label agreement rates as defined in §1.4 (not mask IoU).

### 7.4 Ablations (paper-ready)

- **IMU conditioning variants** (isolates the FiLM + adapter design):
  - No IMU conditioning (drops both FiLM and adapter).
  - FiLM-only (drops the cross-attn adapter, `α ≡ 0`).
  - Adapter-only (drops FiLM, only cross-attn).
  - FiLM + adapter (default).
  - FiLM + adapter with per-layer `α` (12 scalars).
- DynGRU without deviation trick (`Δmotion`) (shows that cue specifically).
- Post-hoc head (current StaticConfidenceHead) vs co-iterative DynGRU.
- No IMU factor in PGO (shows backend contribution).
- EKF-only vs sliding-window (Stage B, optional).

---

## 8. Risks and mitigations

| Risk | Mitigation |
|---|---|
| FlowFormer frozen-decoder bottlenecks dynamic performance | Phase-B ablation: unfreeze last decoder block, lr 1e-6 |
| EKF divergence on long sequences | Bias-sync loop + clipped covariance + periodic EKF reset if `cond(Σ) > 1e6` |
| Air-IO cov is over-confident (known failure mode from paper) | Inflate Σ_v by a fixed factor during Phase B; ablate factor |
| Synthetic-to-real domain gap in Phase A | Phase B is specifically designed to close this; monitor Phase-A pseudo-label agreement rate on TartanAir val throughout finetune to detect catastrophic forgetting |
| DynGRU learns to over-weight IMU in textureless regions and drops visual info | L_sparsity + L_entropy encourage decisive; monitor `c` histogram |
| Two-frame PGO accumulates drift | Stage-B sliding window is pre-designed; promotion is scoped |

---

## 9. Implementation plan reference

Implementation sequencing, step-by-step task breakdown, per-step test strategy, and rollout ordering go in the companion plan produced by the writing-plans skill, not in this spec.

---

## 10. Glossary

- **GMA** — Global Motion Aggregate (FlowFormer's attention-weighted motion feature).
- **DRT-VIO-Init (loose)** — stereo-adapted visual-inertial bootstrap producing gravity, extrinsic, biases.
- **AirIMU** — neural IMU corrector (raw → corrected + cov).
- **Air-IO** — neural body-frame velocity estimator (raw IMU + orientation → v_B + cov).
- **Velocity EKF** — 15-D `(R,V,P,b_g,b_a)` EKF fusing corrected IMU + Air-IO velocity observation.
- **Forster preintegration** — standard VI preintegration with first-order bias Jacobians.
- **Bias-ref sync** — post-solve push of PGO-optimized biases back to EKF.
- **Co-iterative** — DynGRU runs inside FlowFormer's decoder loop, reading per-iter tensors.
- **FiLM** — Feature-wise Linear Modulation: `x' = γ(c)·x + β(c)` (Perez et al., 2018). Uniform spatial modulation from a global conditioner `c`.
- **CLIP-adapter** — lightweight adapter pattern (Gao et al., 2021) that injects conditioning via a gated residual `out = base + tanh(α)·adapter(·)` with `α` init 0, so the block starts as identity and can only add information during training.
- **IMU token** — one of 7 semantic 128-D vectors derived from EKF preintegration components (ΔR, Δv, Δp, R^T·g, bias, Σ, dt); consumed as K,V by the adapter's cross-attention.
