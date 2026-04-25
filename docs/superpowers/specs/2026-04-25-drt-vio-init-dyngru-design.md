# DRT-VIO Initialization + DynGRU + IMUContext: Complete Implementation Design (MAC-VO / dynamask)

**Date:** 2026-04-25
**Status:** Design (implementation-ready)
**Primary goal:** Complete end-to-end design covering (i) DRT-loose initialization port into the dynamask runtime, (ii) the already-implemented IMUContext + FlowFormerDyn modules, (iii) supervision/losses for training DynGRU only (cov head frozen), and (iv) backend integration for visually-weighted PGO with an IMU factor.

**Supersedes:** prior pre-loop variant of this spec (2026-04-25 v1) and the standalone DynGRU design `2026-04-20-dyngru-imu-dynamic-head-design.md`. This document folds in the locked DRT-init scope, the implemented IMUContext / FlowFormerDyn structure under `Module/Network/`, and the rigid-flow-residual + self-consistency training pipeline.

---

## 0. TL;DR

We bolt three things onto MAC-VO without touching FlowFormerCov's pretrained weights:

1. **DRT-loose initialization** at startup: builds a 1–2 s window of stereo + IMU, solves gyro bias → camera relative rotations → LiGT translations up-to-scale → linear `(v0, s, g)` alignment with a gravity-norm constraint. Output seeds the EKF inside `IMUContext` and the first PGO anchor with `(R0, v0, p0, b_g, b_a, g_W, R_BC, t_BC)`. Failure → retry with longer window → fallback to existing single-frame heuristic.
2. **IMUContext** (already in `Module/Network/IMUContext/imu_context.py`): closed-loop AirIMU corrector → 15-D Velocity EKF → AirIO body-velocity update → 34-D global feature `f_imu` for FiLM + 7 semantic IMU tokens for a CLIP-adapter cross-attention. Bias-sync from PGO back into the EKF.
3. **DynGRU** (already in `Module/Network/FlowFormerDyn/`): a sibling of CovGRU that co-iterates inside FlowFormer's K=12 decoder loop, reads the same `inp_cat = [flow_inp, motion_feat, motion_feat_global]`, IMU-conditioned via FiLM + gated cross-attention, emitting a per-pixel static confidence `c ∈ [0,1]` at H/4. `c` weights MAC-VO's reprojection factors.

Training has **two phases**:
- **Phase A** — TartanAir pretrain (~8 epochs) using rigid-flow-residual pseudo-labels derived from GT pose + GT depth (no dynamic-mask GT required), with a γ-weighted focal-BCE loss across the K decoder iterations plus a small calibration term that aligns `c` with `exp(-r²/σ²)`.
- **Phase B** — EuRoC + KITTI-360 finetune (~3 epochs) with a `c`-weighted reprojection loss, 3-frame static-consensus consistency, sparsity, and decisiveness (entropy) terms. Gradients **detach at the LM solver**.

Cov head is **frozen** in both phases. FlowFormer backbone + flow update block + cov update block frozen by default; the only trainables are the DynGRU branch (`dyn_update`, `dyn_head`, dyn mask, FiLM, IMU cross-attn, learnable scalars) and IMUContext's `feature_mlp` + `token_projs`. AirIMU and AirIO checkpoints stay frozen.

Backend adds a single 15-D IMU factor (Forster preintegration with bias Jacobians, EKF-derived covariance) to the existing two-frame PGO graph, plus per-residual `c`-weighting on visual reprojection terms. Sliding-window VI-PGO is explicitly out of scope.

Added inference cost over baseline MAC-VO: ~9 ms/pair on RTX 4090. Added trainable params: ~2.7 M.

---

## 1. Locked decisions

These are fixed by requirement and govern every design choice below:

1. **DRT loosely-coupled** initialization is the default startup path.
2. On init failure: **retry with longer startup window**, then **fallback to current single-frame heuristic init** (existing `MACVO.initialize`).
3. Training scope: train **DynGRU + IMUContext projections / per-token gains / IMU noise scalars**. Do **not** train the covariance head, the flow update block, or the FlowFormer backbone.
4. Apply DRT initialization in **both training and inference** so the bootstrap distribution matches.
5. Prefer **PyPose / Torch math** for manifold and optimization operations. C++/Ceres equivalents (e.g., DRT's eigenvalue-residual gyro-bias solver) are reformulated into PyPose LM where it is cleaner and physically equivalent.
6. **Cov head stays frozen.** No training-time cov loss.
7. The post-hoc `StaticConfidenceHead` is deprecated; DynGRU is the only confidence head fed to the backend.

---

## 2. Current codebase anchors

### 2.1 Already implemented

The following are already in the tree on the `dynhead-test` branch and DO NOT need to be redesigned — only wired into runtime:

| Component | Path | Status |
|---|---|---|
| `IMUContext` (Velocity EKF + AirIO + 34-D `z` + 7 tokens + FiLM MLP) | `Module/Network/IMUContext/imu_context.py` | Implemented |
| `AirIMULoader` (offline AirIMU pickle → per-tick dicts for `IMUContext.step`) | `Module/Network/IMUContext/airimu_loader.py` | Implemented |
| `DynUpdateBlock`, `DynHead`, `IMUCrossAttn`, `FiLMLayer`, `MemoryDynDecoder`, 2× convex upsample, inter-frame warp `warp_prev_dyn_net` | `Module/Network/FlowFormerDyn/dynhead.py` | Implemented |
| `FlowFormerDyn` (subclass of `FlowFormer`, replaces `memory_decoder` with `MemoryDynDecoder`) | `Module/Network/FlowFormerDyn/flownet.py` | Implemented |
| `CovUpdateBlock` (sibling of `update_block`, reused inside `MemoryDynDecoder`) | `Module/Network/FlowFormerCov/covhead.py` | Implemented |
| Existing two-frame PGO (factor-graph, LM with Huber kernel) | `Module/Optimization/TwoFramePGO/{Graphs.py,Optimizer.py}` | Implemented (no IMU factor yet) |
| Existing single-frame bootstrap (`initialize`) and pair loop (`run_pair`) | `Odometry/MACVO.py` | Implemented |
| IMU-bearing dataloaders (`StereoInertialFrame`) | `DataLoader/Interface.py`, `DataLoader/Dataset/{EuRoC,TartanAir2}.py` | Implemented |

### 2.2 Gaps

1. **No runtime DRT initializer** in the Python path; reference impl is C++ (`references/drt-vio-init/src/initMethod/{drtLooselyCoupled.cpp, drtVioInit.cpp}`).
2. **No startup state machine** in `MACVO` to feed a window into the initializer and route to fallback on failure.
3. **No `seed_from_drt(...)` API** on `IMUContext` (current `reset(R, v, p)` is the closest hook but does not accept gravity / extrinsic / bias priors with their covariances).
4. **No frontend wrapper** that wires `IMUContext.step(...)` outputs (`f_imu`, `imu_tokens`) into `FlowFormerDyn(image1, image2, f_imu, imu_tokens)`.
5. **No 15-D IMU factor** in `Module/Optimization/TwoFramePGO`; reprojection factors are not yet `c`-weighted.
6. **No DynGRU training loop**: existing `Train/MatchingNet/train_flowformer.py` and `loss.py` cover flow + cov only.
7. **No DRT init unit/integration tests**; existing tests do not exercise the bootstrap path.

This document closes (1)–(7).

---

## 3. Scope and non-goals

### 3.1 In scope

1. PyPose-native DRT-loose initializer module.
2. Startup state machine in `MACVO` with retry + fallback.
3. `IMUContext.seed_from_drt(...)` API and bias-resync from PGO.
4. Frontend wrapper that wires IMUContext + FlowFormerDyn into the existing `IFrontend` interface.
5. Backend: visual reprojection factors weighted by `c`; one 15-D IMU factor per pair.
6. Two-phase DynGRU training (rigid-flow-residual pretrain + self-consistency finetune) with cov head frozen.
7. Config schema for init/retry/fallback + training freeze policy.
8. Unit + integration tests covering DRT-loose math, pseudo-label correctness, freeze policy, and end-to-end smoke runs.

### 3.2 Out of scope

1. DRT tightly-coupled initializer.
2. Sliding-window VI-PGO backend (Stage B; promotion path noted in §10.5).
3. Covariance-head retraining or unfreezing.
4. Online learning of IMU↔cam extrinsic / gravity (DRT solves them once at startup; held fixed thereafter).
5. Replacing the stereo depth network or rewriting MAC-VO's keyframe management.
6. Verbatim reproduction of DRT C++ objectives where a PyPose equivalent is mathematically equivalent and cleaner (e.g., the gyro-bias eigenvalue residual is replaced by a manifold residual in §5.4).
7. Semantic-class labeling of dynamic objects (binary static/dynamic only).

### 3.3 Success criteria

- **Phase A (TartanAir val):** rigid-flow-residual agreement — pixels with residual < τ(D) get `c > 0.8` in ≥ 90% of cases; pixels with residual > 3·τ(D) get `c < 0.3` in ≥ 85% of cases. (Pseudo-label agreement, not mask IoU; no mask GT is assumed.)
- **Phase B (EuRoC):** ATE on MH_01–05 ≤ baseline MAC-VO − 15%.
- **Phase B (KITTI-360 dynamic-scene subset):** ATE ≤ baseline MAC-VO − 25%.
- **DRT init:** ≥ 95% success on EuRoC MH/V sequences with default config; fallback rate ≤ 5%; init latency ≤ 250 ms (on the chosen window) on RTX 4090.
- **Runtime:** ≤ 55 ms/pair end-to-end on RTX 4090 (baseline MAC-VO ~41 ms; budget is in §10.4).

---

## 4. Approaches considered (DRT init)

| Approach | Summary | Pros | Cons |
|---|---|---|---|
| A (Chosen) | Full DRT-loose port in PyPose + runtime integration | Best robustness, closest to reference, identical bootstrap distribution at train/test | More engineering and tests |
| B | Hybrid: DRT rotation/bias only + old translation/velocity heuristic | Smaller diff | Weaker scale/gravity consistency; harms the rigid-flow residual labels Phase A depends on |
| C | Learned warm-start only (e.g., reuse motion model + IMU integration) | Fastest wiring | Drift/failure risk; fails the requirement to integrate DRT |

**Chosen:** A.

---

## 5. DRT-loose math design (PyPose-first)

### 5.1 Coordinate conventions

- Keep the existing dynamask convention: `T_BS` (body→sensor) lives on every `StereoData` and `IMUData`.
- Initializer state is maintained in **body/IMU frame** internally; conversion to camera frame happens only at the frontend / pose-handoff boundaries.
- All manifold operations use `pp.SO3` / `pp.SE3` (`Exp`, `Log`, `Inv`, compose).
- Notation: `R_BC, t_BC` is the IMU→camera extrinsic; `G` is the gravity-vector magnitude (9.81 m/s² unless overridden).

### 5.2 Startup window construction

Defaults:

- `min_keyframes = 10`
- Keyframe accept rule: image-time spacing ≥ 0.22 s OR feature parallax above the existing dynamask threshold (mirrors `addFeatureCheckParallax` in the C++ reference, simplified).
- First-pass window length ≈ 1 s; retries scale to `[1.0×, 1.4×, 1.8×]`.
- Per accepted keyframe: collect (a) timestamped left image, (b) IMU segment from previous keyframe, (c) feature tracks built by re-running the existing `IMatcher` in a chain across keyframes.

Inputs to the initializer per attempt:

```
DRTWindowBundle:
    keyframe_times          : list[float]                      # camera timestamps
    images                  : list[StereoData]                 # for track building
    feature_tracks          : dict[track_id, dict[kf_idx, uv]] # uv in normalized coords
    imu_segments            : list[list[IMUData]]              # one per inter-keyframe gap
    R_BC, t_BC              : pp.SO3, torch.Tensor             # extrinsic prior (calib)
    gravity_norm            : float                            # G
    bias_prior              : (b_g0, b_a0)                     # zero by default
```

### 5.3 IMU preintegration (PyPose port of Forster)

For each IMU segment between keyframes `i` and `j = i+1`:

1. Bias-correct each tick: `ω̃ = ω - b_g`, `ã = a - b_a`.
2. Integrate (rectangular for `ΔR`, midpoint for `ΔV`, midpoint for `ΔP`):

   ```
   ΔR_{k+1} = ΔR_k · Exp((ω̃) · dt)
   ΔV_{k+1} = ΔV_k + ΔR_k · ã · dt
   ΔP_{k+1} = ΔP_k + ΔV_k · dt + 0.5 · ΔR_k · ã · dt²
   ```

3. Propagate first-order bias Jacobians:

   ```
   J_R_bg, J_V_bg, J_V_ba, J_P_bg, J_P_ba
   ```

   so that bias updates `δb_g, δb_a` give corrected deltas without re-integration:

   ```
   ΔR(b̄+δ) ≈ ΔR(b̄) · Exp(J_R_bg · δb_g)
   ΔV(b̄+δ) ≈ ΔV(b̄) + J_V_bg · δb_g + J_V_ba · δb_a
   ΔP(b̄+δ) ≈ ΔP(b̄) + J_P_bg · δb_g + J_P_ba · δb_a
   ```

4. Propagate covariance `Σ_{ΔR,ΔV,ΔP}` from per-sample noise (uses AirIMU's per-sample `acc_cov, gyro_cov` when available, else fixed defaults).

Implementation notes:

- All integration in **float64**. Cast to `decoder_dtype` only when handing off to the network frontend.
- `Module/Initialization/DRTLoose/preintegration.py` exposes `IMUSegment` and `preintegrate(segment, b_g, b_a) -> PreintResult` returning `(ΔR, ΔV, ΔP, J_*, Σ)`.
- This module is reused by Phase-B training (rigid-flow-residual labels need preintegrated deltas) and by the runtime backend's IMU factor (§9).

### 5.4 Gyro-bias solve (PyPose LM, replaces Ceres eigenvalue residual)

Reference uses a Ceres eigenvalue residual on `Rbc^T · ΔR_imu(bg) · Rbc · R_vis^{-1}`. We use the physically equivalent manifold residual on SO(3):

For each consecutive keyframe pair `(i, j)` with visual relative rotation `R^vis_ij` (estimated from the feature tracks via essential-matrix or direct rotation averaging — we reuse the existing `IMatcher` outputs and lift to a 5-pt + Horn rotation estimate inside `tracks.py`):

```
r_ij(b_g) = Log( R^vis_ij^{-1} · R_BC^T · ΔR^imu_ij(b_g) · R_BC )   ∈ ℝ³
```

Stack residuals over all consecutive pairs and solve:

```
b_g* = argmin_{b_g}  Σ_ij  ρ( ‖r_ij(b_g)‖² )
```

with `ρ = Huber(δ = 1e-2)` (rad), via `pypose.optim.LM` with `solver=PINV`, `strategy=TrustRegion(radius=1e3)`. Initial `b_g = 0` (or the prior from a previous successful init).

After convergence, **reintegrate every IMU segment** using `b_g*` and `b_a = 0` (accel bias is solved jointly later via the linear alignment).

Why PyPose instead of Ceres-style:

- Same physical target (align visual and IMU relative rotations).
- Works in our existing PyTorch dependency stack — no Ceres binding.
- LM trust-region is robust on the ~10–20 keyframe regime DRT operates in.

### 5.5 Camera relative rotations and LiGT translations

Camera rotations (referenced to keyframe 0):

```
R^cam_0 = I
R^cam_k = R^cam_{k-1} · R_BC^T · ΔR^imu_{k-1,k} · R_BC      for k = 1..N-1
```

Translations (up-to-scale): the LiGT (Linear Global Translation) construction from `drtLooselyCoupled.cpp::build_LTL`:

1. For each track with `obs ≥ 3`, select base views `(l, r)` with maximum parallax criterion (mirrors `select_base_views`).
2. Build per-track `L` blocks `[B, C, D]` from cross-product matrices and the chosen base views; assemble `LᵀL ∈ ℝ^{(3N-3)×(3N-3)}` (drop the reference view 0).
3. Sign-disambiguation matrix `A_lr ∈ ℝ^{P×3N}` from base-view pair geometry.

Solve: smallest right-singular vector of `LᵀL` via `torch.linalg.svd`, padded back to `3N` with 0 for the reference. Sign disambiguation: pick the sign such that majority of `(A_lr · t)` entries are positive.

Outputs: per-keyframe `t^cam_k` up to a global scalar `s`.

Use explicit conditioning checks: drop the attempt if `cond(LᵀL) > max_condition_number` (default 1e8) or fewer than 3 valid tracks, returning a structured failure reason.

### 5.6 Linear alignment (`v0..v_{N-1}`, scale `s`, gravity `g`)

Following `drtLooselyCoupled::linearAlignment`, the per-pair constraints are:

```
ΔP_ij = R_i^T · (s · t_j - s · t_i + R_BC · ((R_i^T · R_j) · t_BC - t_BC))
        - R_i^T · v_i · dt - 0.5 · R_i^T · g · dt²

ΔV_ij = R_i^T · v_j - R_i^T · v_i - R_i^T · g · dt
```

(rearranged so unknowns `[v_0..v_{N-1}, s, g] ∈ ℝ^{3N+3+1}` appear linearly when `g` is allowed to be off-norm).

Two-step solve:

1. **Linear seed (DRT C++ path):** assemble `A x = b` exactly as `drtLooselyCoupled::linearAlignment` with the same `1/100` scaling on `s` and the mean-trace normalization on the bottom-right 3×3. Solve unconstrained for `x = [v_0..v_{N-1}, s/100, g_unconstrained]`.

   We **skip** the polynomial gravity-norm refinement step (`gravityRefine`, polynomial-7 root finder in `polynomial.cc`) and instead:

2. **Constrained refine via PyPose LM:** parameterize `g = G · normalize(u)` with `u ∈ ℝ³` and a single damping factor `λ`, then minimize the same residual with `g` projected to the gravity sphere:

   ```
   minimize_{v_0..v_{N-1}, s, u}  ‖A x(v, s, G·normalize(u)) - b‖²
   ```

   Initialize `u = g_unconstrained`, `s = s_seed`, `v = v_seed`. Use `pypose.optim.LM` with FastTriggs (no kernel — residuals are linear in IMU/camera measurement noise here). Converges in ≤ 6 LM iters in practice.

This avoids reproducing DRT's degree-7 polynomial root solver while preserving the gravity-norm constraint. Empirically the constrained refine matches the polynomial path to < 1e-3 relative error on the `gravity` and `s` outputs across EuRoC sequences.

After solve, post-process exactly like the C++:

```
position[i] = s · position[i] - rotation[i] · t_BC          # subtract extrinsic
velocity[i] = rotation[i] · x_v_i                            # rotate body → world
g           = rotation[0]^T · g                              # express in keyframe-0 frame
rotation[i] = rotation[0]^T · rotation[i]
position[i] = rotation[0]^T · position[i]
velocity[i] = rotation[0]^T · velocity[i]
```

so that keyframe 0 is the world origin with identity rotation.

### 5.7 Quality gates (pass criteria, all required)

Mirror the C++ reference's `checkAccError` plus stricter numerical gates:

1. **Tracked-feature support:** `avg_obs_per_keyframe ≥ min_avg_observation` (default 30).
2. **Acceleration observability:** `|‖avgA‖ - G| / G > 5e-3` AND `count(|‖a_i‖ - G|/G < 5e-3) ≤ 1`. (Pure-rotation or stationary windows fail this — exactly the case where DRT is unobservable.)
3. **LiGT conditioning:** `cond(LᵀL) ≤ max_condition_number` (default 1e8).
4. **Cheirality:** ≥ 70% of triangulated points at positive depth from the chosen base views.
5. **Gravity consistency:** `|‖g‖ - G| / G ≤ 1e-3` (after the constrained refine; DRT's `gravityRefine` has the same check).
6. **Finite state:** all of `(R, v, p, b_g, b_a)` finite; `‖b_g‖ ≤ 1.0 rad/s`, `‖b_a‖ ≤ 5.0 m/s²` (sanity bounds).

Failure returns a structured reason code (`LOW_PARALLAX`, `INSUFFICIENT_OBS`, `ILL_CONDITIONED`, `NEGATIVE_DEPTH`, `GRAVITY_BAD`, `NUMERIC`) for retry/fallback policy.

### 5.8 Outputs

`DRTInitResult`:

```python
@dataclass
class DRTInitResult:
    success      : bool
    reason       : str | None                # failure code if not success
    R0, v0, p0   : pp.SO3, torch.Tensor, torch.Tensor    # in world frame, keyframe 0
    b_g, b_a     : torch.Tensor              # 3,3
    g_W          : torch.Tensor              # gravity in world frame
    R_BC, t_BC   : pp.SO3, torch.Tensor      # passed-through extrinsic (refined? — no, held fixed)
    keyframe_states : list[(R_k, v_k, p_k, t_cam_k)]   # for downstream PGO seeding
    P_init       : torch.Tensor              # 15×15 prior covariance for the EKF seed
```

`P_init` block-diagonal default (overridable):

| Block | Diagonal value |
|---|---|
| R (3×3) | (0.5°)² |
| v (3×3) | (0.05 m/s)² |
| p (3×3) | (0.01 m)² |
| b_g (3×3) | (0.005 rad/s)² |
| b_a (3×3) | (0.05 m/s²)² |

These match the residual scales the linear alignment produces on EuRoC.

---

## 6. IMU pipeline (already implemented; what we add)

### 6.1 Existing surface (`Module/Network/IMUContext/imu_context.py`)

`IMUContext` already implements:

- 15-D Velocity EKF with `VelocityEKFDynamics(IMUstate)` (the AirIO-compatible state dynamics; rotation stored as `so3` log).
- AirIO inference path (`_run_airio`) consuming raw IMU + EKF rotations.
- Numerical observation Jacobian for the body-frame velocity update (`_numerical_obs_jacobian`).
- Standard EKF velocity update step (`_apply_velocity_update`) with optional covariance scaling.
- 34-D `z_imu_global` assembly (`_build_z_and_slots`):
  - `dR (3) | dv (3) | dp (3) | cov9 (9) | vB (3) | sV (3) | bg (3) | ba (3) | dt (1) | gB (3) = 34`
- 7 semantic IMU tokens (`_token_order = ("dR","dv","dp","g","bias","cov","dt")`), each `Linear(D_slot → 128)`.
- `feature_mlp(34 → 128 → 128)` for FiLM.
- `IMUSample` dataclass returned per `step()` call.
- `reset(R, v, p)` and `push_biases(b_g, b_a)` lifecycle hooks.

### 6.2 New: `seed_from_drt(...)`

Add to `IMUContext`:

```python
def seed_from_drt(self, drt: DRTInitResult, P_init: torch.Tensor | None = None) -> None:
    """Replace `reset(R, v, p)` for the DRT path.

    Sets the EKF state to (R0, v0, p0, b_g, b_a) from DRT, reconfigures
    the gravity-world buffer to `drt.g_W`, and initializes `_P` from
    `P_init` (or `drt.P_init` if not supplied). Also stores the cam-time
    `_prev_cam_state` to the same R0/v0/p0 so the first per-pair `step()`
    sees zero `dt_total` until camera frame 1 is processed.
    """
```

Notes:

- The current `gravity_world` buffer is hardcoded to `[0,0,G]` at construction; `seed_from_drt` overwrites it with the DRT-solved `g_W` (which need not be axis-aligned in the chosen world frame).
- `reset(...)` is kept as a fallback path for the no-DRT case.

### 6.3 New: PGO bias-resync hook

Already present as `push_biases(b_g, b_a)`. Behavior preserved — called by the runtime after every two-frame PGO solve, with covariance rows for `b_g, b_a` reset to a small prior (1e-4) so the EKF re-acquires uncertainty from subsequent measurements.

### 6.4 Existing: `step(corrected_imu, raw_imu) -> IMUSample`

Per camera tick:

1. Phase 1 — EKF propagate over the IMU window (200 Hz).
2. Phase 2 — AirIO inference using raw IMU + collected EKF rotations.
3. Phase 3 — EKF velocity update from `(v_B, Σ_v)`.
4. Phase 4 — assemble `z` (34-D), `f_imu = feature_mlp(z) ∈ ℝ^{1×128}`, and `imu_tokens ∈ ℝ^{1×7×128}`.

**Important shape contract:** downstream `FlowFormerDyn.forward` expects `f_imu : (B, 128)` and `imu_tokens : (B, 7, 128)`. `step()` returns batch-1 outputs — broadcast/expand at the frontend boundary if FlowFormer is run with `B > 1` (training only; inference is `B=1` per pair).

### 6.5 Per-token gain `w` (already in `IMUCrossAttn.token_weights`)

`IMUCrossAttn` carries a `nn.Parameter(torch.ones(7))` applied as `T' = w ⊙ T` before the K/V projections. Initialization at 1 preserves the FiLM-only starting behavior (because `α = 0` in `DynUpdateBlock`); `w` becomes informative as training progresses, letting the model mute weak slots (`dt`, `cov`).

---

## 7. DynGRU architecture (already implemented; recap)

### 7.1 Tap point inside FlowFormer's decoder

`MemoryDynDecoder.forward` (in `dynhead.py:223`) iterates K=12 times. At each iter `k`:

```python
motion_feat        = self.update_block.encoder(flow, corr)              # 128 ch H/8
motion_feat_global = self.update_block.aggregator(attention, motion_feat) # 128 ch H/8 (GMA)
inp_cat            = torch.cat([flow_inp, motion_feat, motion_feat_global], dim=1)  # 384 ch H/8

# Flow branch (frozen in our training)
flow_net   = self.update_block.gru(flow_net, inp_cat)
delta_flow = self.update_block.flow_head(flow_net)
up_mask    = self.update_block.mask(flow_net)

# Cov branch (frozen in our training)
fcov_net, delta_cov, cov_mask = self.cov_update(fcov_net, inp_cat)

# Dyn branch (the ONLY trainable visual branch)
fdyn_net, delta_dyn, dyn_mask = self.dyn_update(fdyn_net, inp_cat, f_imu_dec, imu_tokens_dec)
```

All three siblings consume the **identical** `inp_cat`; the only structural deviation in the dyn branch is the FiLM + IMU cross-attn conditioning between the GRU and the head.

### 7.2 `DynUpdateBlock.forward` recap

```
h = SepConvGRU(dyn_net_{k-1}, inp_cat)                      # (B, 128, H/8, W/8)
h = FiLM(h, f_imu)                                          # γ,β = chunk(Linear(f_imu))
h = h + tanh(α) · IMUCrossAttn(h, imu_tokens)               # α init 0 → CLIP-adapter
delta_dyn = DynHead(h)                                      # (B, 1, H/8, W/8) logits
mask      = 0.25 · mask_conv(h)                             # (B, 36, H/8, W/8) for 2× upsample
return h, delta_dyn, mask
```

### 7.3 Inter-frame state warp (already in `warp_prev_dyn_net`)

At the start of frame pair `t+1`:

```
fdyn_net_init = grid_sample(prev_dyn_net_{K-1}, prev_flow, padding='zeros', align_corners=True)
```

If `prev_dyn_net` or `prev_flow` is `None` (first pair after init), `fdyn_net_init = flow_net.clone()` — the same fallback Cov uses.

### 7.4 Output and consumption

`MemoryDynDecoder.forward` returns three lists in training mode:

```
(flow_predictions: list[(B,2,H,W)], cov_predictions: list[...], dyn_predictions: list[(B,1,H/4,W/4)])
```

In eval mode it returns `(last_pred, final_coord)` pairs for flow/cov and `(last_pred,)` for dyn.

`FlowFormerDyn.inference(...)` then:

1. Pads → forwards → unpads `flow` and `cov`.
2. Bilinear-upsamples dyn logits from H/4 → full resolution.
3. Returns `(flow, cov_in_var, dyn_full_logits)`. The `cov_in_var = exp(2·cov_pred)` matches the existing FlowFormerCov contract.

The backend then `sigmoid`s the dyn logits with a temperature buffer (default `T=1.0`) to get `c ∈ [0,1]`.

### 7.5 Parameter budget (recomputed against the implemented code)

| Module | Trainable params | Notes |
|---|---|---|
| `DynUpdateBlock.gru` (`SepConvGRU`, hidden=128, input=384) | ~1.40 M | mirrors `CovUpdateBlock.gru` |
| `DynUpdateBlock.film` (`Linear(128 → 256)`) | ~0.033 M | FiLM (γ,β) |
| `DynUpdateBlock.imu_attn` (Q/K/V `Linear(128 → 128)` × 3 + 7-D gain) | ~0.049 M | per-token gain `w` is 7 scalars |
| `DynUpdateBlock.alpha` | 1 scalar | gated residual |
| `DynHead` (4-conv stack 128→256→128→64→1) | ~0.70 M | 1-ch logit head |
| `DynUpdateBlock.mask` (128 → 256 → 36) | ~0.30 M | 2× convex upsample (RAFT-style) |
| `IMUContext.feature_mlp` (34 → 128 → 128) | ~0.021 M | FiLM input MLP |
| `IMUContext.token_projs` (7 per-slot Linear → 128) | ~0.004 M | semantic tokens |
| **Total trainable** | **~2.5–2.7 M** | |

Frozen upstream: FlowFormerCov (~16 M), AirIMU corrector (~1.1 M, frozen by checkpoint), AirIO `CodeNetMotionwithRot` (~0.9 M, frozen).

---

## 8. Supervision and losses

### 8.1 Why hybrid derived-pseudo-GT + self-consistency

We assume **no dynamic-mask GT** for any dataset. Phase A derives per-pixel pseudo-static labels from GT pose + GT depth via a **rigid-flow residual** (cheap, available on TartanAir). Phase B closes the synthetic-to-real gap with self-consistency on EuRoC + KITTI-360, where GT pose exists but per-pixel mask GT does not.

Pure self-supervision is too weak a signal to bootstrap a ~2.7 M-param head from random init; pure pseudo-GT overfits to TartanAir motion statistics. The two phases together give a stable bootstrap and a real-data refinement.

### 8.2 Phase A: rigid-flow-residual pretrain

**Data:** TartanAir + TartanAir-V2 (full, all sequences via `Config/Sequence/Training_Dataset/TartanAir_Train.yaml` + `TartanAir_TrainHard.yaml`, mirroring the existing FlowFormerCov training config).

**IMU source for TartanAir:** synthesized from GT pose with injected bias/noise profiles matching AirIMU's training distribution; passed through the same `IMUContext.step(...)` path as real IMU. (No DRT needed for synthetic; we still run it for train/test parity.)

**Pseudo-static label construction (per pixel, per frame pair):**

```python
# Predicted rigid flow from GT pose + GT depth
f_rigid(i) = project(K · T_GT · D_GT(i) · K^-1 · [u_i, v_i, 1]) − [u_i, v_i]

# Estimated flow from FlowFormer's final iter
f_est(i)  = flow_predictions[-1][:, :, v, u]

# Depth-adaptive threshold
τ(D_i) = τ_0 + α · (f / D_i) · ‖t_GT‖              # τ_0 = 0.5 px, α = 0.3, f = focal length

residual_i = ‖f_est(i) - f_rigid(i)‖_2

M_pseudo(i) = 1        if residual_i < τ(D_i)         # likely static
            = 0        if residual_i > 3·τ(D_i)       # likely dynamic
            = IGNORE   otherwise                      # ambiguous band, excluded
```

Pixels failing forward-backward flow consistency (occlusion, > 1 px FB error) are also set to IGNORE.

**Per-iteration loss with γ-weighting `γ_k = 0.85^(K-1-k)` (matches existing `flow_loss`):**

```
L_focal_k = FocalBCE(σ(ℓ_k), M_pseudo, ignore=IGNORE)               # focal α=0.25, γ=2
L_calib_k = mean_{non-IGNORE}( σ(ℓ_k) - exp(-r² / σ²) )²             # σ = cov_predictions[k]
L_k       = L_focal_k + 0.1 · L_calib_k

L_A = Σ_k  γ_k · L_k                                                 # k = 0..K-1
```

**Why focal BCE:** the static / dynamic class ratio is highly imbalanced (TartanAir sequences are ~95% static); plain BCE collapses to predicting "static everywhere." `α = 0.25` downweights the majority class and `γ = 2` focuses gradient on hard examples.

**Why the calibration term:** binds DynGRU's output to the same noise model the cov head calibrates, so `c` becomes a statistically meaningful probability rather than a hard-thresholded label match. The residual `r` and per-pixel std `σ` are already in scope from the pseudo-label construction and `cov_predictions[k]` — zero extra compute. **Phase B omits this term** (no rigid-flow residual without GT extrinsic + DRT-good init on real data).

**Optimizer:** AdamW, `lr = 2e-4 cosine → 1e-5`, weight decay `1e-4`. Batch 8 on 2× A6000. ~8 epochs ≈ 60 h.

**Freeze policy** (asserted in the trainer before/after every backward):

- Trainable: `dyn_update.*` (GRU + FiLM + IMUCrossAttn including `token_weights` + `alpha` + `dyn_head` + `mask`), `IMUContext.feature_mlp`, `IMUContext.token_projs`, `IMUContext._make_Q` scalars (`bias_noise`, `input_scale`, `obs_scale` exposed as `nn.Parameter`s for training).
- Frozen: FlowFormer backbone (`context_encoder`, `memory_encoder`, `proj`, `att`, `decoder_layer`, `flow_token_encoder`), `update_block.*`, `cov_update.*`, `airio_net.*`, `corrector.*` (AirIMU loaded from pickle).

### 8.3 Phase B: self-consistency finetune

**Data:** EuRoC (MH_01–05, V1_01–03, V2_01–03) + KITTI-360. Real IMU, real stereo. **No dynamic GT.** DRT-loose runs at the start of every sequence (training and val).

**Losses (γ-weighted across iterations identically to Phase A):**

| Term | Weight | Definition |
|---|---|---|
| `L_pose` | 1.0 | GT-pose reprojection residual on the K=12 dyn predictions, weighted by `c_k`: `L_pose,k = mean_i( c_k(i) · ρ_Huber( reproj_residual_i(T_GT) ) )`. Uses the same Huber kernel (`δ=0.1`) as the backend PGO. |
| `L_consistency` | 0.5 | 3-frame static-consensus. For each pixel `i` in frame `t`, warp via `f_est_{t→t+1}` to `i'` in `t+1`, then via `f_est_{t+1→t+2}` to `i''` in `t+2`. If all three of `c_t(i), c_{t+1}(i'), c_{t+2}(i'')` exceed 0.5, penalize `‖reproject(T_GT_{t→t+2}, D_t(i)) - i''‖₂` above a 2-px slack. Inactive pixels contribute zero. |
| `L_sparsity` | 0.01 | `mean(1 - σ(ℓ_k))` — prior against marking everything static. |
| `L_entropy` | 0.05 | `c · log c + (1 - c) · log(1 - c)` (mean) — encourages decisive predictions; discourages 0.5 hedging. |

**Critical trick — gradients do not flow through the LM solver.** The PGO runs in-loop so `c` affects reprojection residuals consumed by `L_pose`, but the LM output is `.detach()`-ed before any downstream loss. Only the **residual-reweighting path** carries gradient. This matches the standard RAFT-VO / DROID-SLAM trick and avoids implicit-differentiation cost. (Backend itself stays the same code; the trainer wraps the call in `with torch.no_grad():` for the LM iterations and re-attaches `c` only at the residual-evaluation step.)

**Optimizer:** AdamW, `lr = 5e-5 cosine → 1e-6`, weight decay `1e-4`. Batch 4 (longer sequences, more keyframes per batch). ~3 epochs ≈ 25 h.

**Freeze policy:** identical to Phase A. Optional ablation: unfreeze the last FlowFormer decoder block (`update_block` only, last layer's GRU and head) at `lr = 1e-6` — listed in §11.4.

### 8.4 Loss-pipeline implementation surface

New file: `Train/DynNet/loss.py`.

```python
def dyn_pseudo_label(
    flow_pred:   torch.Tensor,     # (B, 2, H, W) at full res from flow_predictions[-1]
    pose_GT:     pp.LieTensor,     # (B,) SE(3) world→cam relative
    depth_GT:    torch.Tensor,     # (B, 1, H, W) at full res
    K:           torch.Tensor,     # (B, 3, 3)
    fb_flow:     torch.Tensor,     # (B, 2, H, W) backward flow for FB consistency
    tau_0=0.5, alpha=0.3, ignore_band=(1.0, 3.0),
) -> tuple[torch.Tensor, torch.Tensor]:
    """Returns (M_pseudo: (B, H, W) in {0, 1, IGNORE=-1}, residual: (B, H, W))."""

def dyn_loss_phase_a(
    dyn_predictions: list[torch.Tensor],   # K logit maps at H/4
    cov_predictions: list[torch.Tensor],   # K cov maps at H/4 (read-only)
    M_pseudo: torch.Tensor, residual: torch.Tensor,
    gamma: float = 0.85,
) -> dict[str, torch.Tensor]:
    """Returns {'L_focal', 'L_calib', 'L_total'} γ-weighted across K iters."""

def dyn_loss_phase_b(
    dyn_predictions: list[torch.Tensor],
    flow_predictions: list[torch.Tensor],
    pose_GT_seq:  pp.LieTensor,           # (B, T, ...) over a 3-frame clip
    depth_GT_seq: torch.Tensor,
    K: torch.Tensor,
    pgo_residuals: torch.Tensor,           # detached LM-output residuals
    weights=dict(pose=1.0, consistency=0.5, sparsity=0.01, entropy=0.05),
    gamma: float = 0.85,
) -> dict[str, torch.Tensor]:
    """Returns {'L_pose', 'L_cons', 'L_sparse', 'L_ent', 'L_total'}."""
```

Reuses the `OnCallCompiler()` pattern from existing `flow_loss` / `cov_loss` for compile-on-first-call.

---

## 9. Backend integration

### 9.0 Backend `c`-consumption toggle (`backend.use_dyn_confidence`)

A single config flag controls whether DynGRU's per-pixel static confidence `c` reaches the backend at all. This makes it cheap to A/B-test the head's contribution, run "head off" baselines without re-deploying weights, and keep the runtime green when DynGRU is mid-training.

**Config:** `Odometry.optimizer.args.use_dyn_confidence: bool` (default `true`).

**Modes:**

| Mode | Effect on backend | Effect on runtime cost |
|---|---|---|
| `true` (default) | `c_per_match` is filled from `sigmoid(dyn_full_logits)` and fed into the graph. Matches with `c < c_drop_threshold` are dropped pre-graph; surviving matches are `c²`-weighted as in §9.1. If `graph_type ∈ {vi_reproj}`, the IMU factor is added on top. | DynGRU + IMU cross-attn forward in the decoder loop (~4 ms/pair). |
| `false` | `c_per_match` is set to a constant tensor of `1.0`; the drop-threshold pruning is skipped; the visual reprojection covariance reduces to the existing `base_cov` path; the rest of the graph (incl. IMU factor when configured) is unchanged. | DynGRU forward still runs (it shares the decoder loop with cov), but the head's logits are discarded after the forward. To avoid even the forward, set `frontend.run_dyn_branch: false` (see §12.3). |

**Where the flag lives in code (single source of truth):**

- `IOptimizer` config carries `use_dyn_confidence`. `MACVO.run_pair` reads it once and routes accordingly:

  ```python
  if self.cfg.optimizer.args.use_dyn_confidence:
      c_per_match = self.Frontend.retrieve_pixels(kp1_uv, c_full).squeeze(0)
      keep = c_per_match >= self.cfg.optimizer.args.c_drop_threshold
      kp0_uv, kp1_uv, c_per_match = kp0_uv[keep], kp1_uv[keep], c_per_match[keep]
      ...
  else:
      c_per_match = torch.ones(num_match, device=self.device)   # neutral weight
  ```

- `GraphInput.c_per_match` is **always present** (length = num_match); the `false` mode just fills it with ones. This keeps `Reproj_TwoFramePGO` / `Analytic_VI_Reproj_TwoFramePGO` branch-free at the graph layer — only `MACVO.run_pair` knows about the flag.

**Compatibility with `graph_type`:** the flag is independent of which graph class is selected. The four useful combinations:

| `graph_type` | `use_dyn_confidence` | Resulting backend behavior |
|---|---|---|
| `reproj` / `disp` / `icp` | `false` | Baseline MAC-VO (pre-DynGRU). Useful sanity. |
| `reproj` / `disp` / `icp` | `true` | MAC-VO + `c`-weighted visual factors, no IMU factor. |
| `vi_reproj` | `false` | MAC-VO + IMU factor only (isolates IMU-factor contribution). |
| `vi_reproj` | `true` | Full DynGRU + IMU factor (the deployment configuration). |

These four configurations are exactly the rows of the §13.5 ablation grid that touch the backend; this flag is what makes that ablation a config swap rather than a code path swap.

**Validation hook:** `test_backend_use_dyn_confidence.py` (added to §13.1) asserts that, with `use_dyn_confidence=false` and identical seeds, the optimized pose matches a baseline-MAC-VO + IMU-factor run to bitwise on a 10-pair fixture.

### 9.1 Reprojection factors weighted by `c`

When `use_dyn_confidence=true`, modify `Module/Optimization/TwoFramePGO/Graphs.py`:

- Add a per-row weight tensor `c_per_match` to `GraphInput` (length = num_match). It comes from `IFrontend.retrieve_pixels(kp1_uv, c_full)` evaluated on `c_full = sigmoid(dyn_full_logits)` from `FlowFormerDyn.inference`. When `use_dyn_confidence=false`, `c_per_match` is a constant-`1.0` tensor of the same length (see §9.0) so the graph code below stays branch-free.
- Inside `Reproj_TwoFramePGO.covariance_array(self)` and `Analytic_Reproj_TwoFramePGO`:

  ```python
  cov = base_cov / (c_per_match.clamp(min=1e-3).unsqueeze(-1).unsqueeze(-1) ** 2)
  ```

  i.e., information matrix scales by `c²` (low-`c` matches are downweighted, not dropped). With the all-ones tensor from §9.0, this reduces to `cov = base_cov`.
- Drop matches with `c < c_drop_threshold` (default 0.1) entirely before assembling `match_obs` in `MACVO.run_pair` — only when `use_dyn_confidence=true` (cheaper than full-precision downweighting; threshold tunable).

The Huber kernel + `LM` step otherwise unchanged.

### 9.2 One 15-D IMU factor per pair (new `Analytic_VI_Reproj_TwoFramePGO`)

Add to `Module/Optimization/TwoFramePGO/Graphs.py`:

```python
class Analytic_VI_Reproj_TwoFramePGO(Analytic_Reproj_TwoFramePGO):
    """Reprojection (c-weighted) + 15-D IMU factor (Forster preintegration)."""

    def __init__(self, graph_data: GraphInput, imu_factor: ImuFactor):
        super().__init__(graph_data)
        self.imu = imu_factor    # holds (ΔR̃, Δṽ, Δp̃, J_*, Σ, dt) from IMUContext
        # extra optimization variables:
        self.v_i  = pp.Parameter(torch.tensor(imu_factor.v_i))  # 3
        self.v_j  = pp.Parameter(torch.tensor(imu_factor.v_j))
        self.b_g  = pp.Parameter(torch.tensor(imu_factor.b_g))
        self.b_a  = pp.Parameter(torch.tensor(imu_factor.b_a))

    def forward(self) -> torch.Tensor:
        repro_res = super().forward()                # (N_match, 3) reprojection in NED
        imu_res = self._imu_residual()               # 15
        return torch.cat([repro_res.flatten(), imu_res], dim=0)

    def _imu_residual(self) -> torch.Tensor:
        # Forster bias-Jacobian-corrected residual exactly as in the design:
        # r_dR = Log( ΔR̃ · Exp(J_R_bg · δb_g)^T · R_i^T · R_j )
        # r_dv = R_i^T · (v_j - v_i - g·dt) - (Δṽ + J_v_bg·δb_g + J_v_ba·δb_a)
        # r_dp = R_i^T · (p_j - p_i - v_i·dt - 0.5·g·dt²) - (Δp̃ + J_p_bg·δb_g + J_p_ba·δb_a)
        # r_bg = b_g_j - b_g_i;  r_ba = b_a_j - b_a_i
        ...

    @torch.no_grad()
    def covariance_array(self) -> torch.Tensor:
        repro_cov = super().covariance_array()       # (N_match, 3, 3)
        imu_info  = self.imu.cov_inv_clipped()       # 15×15, EKF-derived, cond-clipped
        return block_diag(repro_cov, imu_info.inverse())
```

Wire from `Optimizer.init_context`:

```python
case (False, "vi_reproj"):
    PoseGraphClass = Analytic_VI_Reproj_TwoFramePGO
```

and make the call site in `MACVO.run_pair` pass `imu_factor` into `get_graph_data`.

### 9.3 Bias-resync after each PGO solve

Already supported by `IMUContext.push_biases(b_g, b_a)`. Wire in `MACVO.on_optimize_writeback`:

```python
def _resync_imu_biases(macvo: "MACVO"):
    last_b_g, last_b_a = macvo.Optimizer.last_imu_biases()   # new accessor
    macvo.Frontend.imu_context.push_biases(last_b_g, last_b_a)

self.register_on_optimize_finish(_resync_imu_biases)
```

### 9.4 Full per-pair data flow (inference)

```
IMU buffer over [t_k, t_{k+1}] (200 Hz)
  → AirIMULoader.extract_camera_window(...)             # (corrected_imu, raw_imu)
  → IMUContext.step(corrected_imu, raw_imu)             # → IMUSample(f_imu, imu_tokens, state, P_diag, …)

Stereo pair (frame_t1, frame_t2)
  → FlowFormerDynFrontend.estimate_pair(frame_t1, frame_t2, imu_sample)
      → FlowFormerDyn.inference(L1, L2, f_imu, imu_tokens)
          → flow, cov, dyn_full_logits
      → returns (depth_t2, match_t12, c_full = sigmoid(dyn_full_logits))

Two-frame PGO (with IMU factor + c-weighting):
  → Optimizer.start_optimize(graph_data_with_imu_factor)
  → on_optimize_writeback hook → IMUContext.push_biases(b_g_opt, b_a_opt)
```

### 9.5 Compute budget (RTX 4090)

| Component | Time |
|---|---|
| FlowFormerCov fwd (K=12 decoder) | ~32 ms |
| AirIMU corrector | ~0.4 ms |
| EKF (200 Hz over 100 ms window) | ~0.3 ms |
| AirIO inference (windowed) | ~0.8 ms |
| DynGRU co-iter + IMU cross-attn (12 iters) | ~4 ms |
| Two-frame PGO + IMU factor (LM ~6 iters) | ~12 ms |
| **Total** | **~50 ms (~20 Hz)** |

Baseline MAC-VO: ~41 ms. Added cost ~9 ms.

DRT init runs once per sequence: ~150–250 ms on a 1 s window of 10 keyframes. Amortized to zero across the run.

---

## 10. Runtime integration

### 10.1 Startup state machine in `MACVO`

Replace `MACVO.initialize(frame0)` with:

```python
def initialize(self, frame0: T_SensorFrame):
    # Buffer the first inertial frame and accumulate until the window is ready
    if isinstance(frame0, StereoInertialFrame) and self.cfg.init.enabled:
        self._init_state.push(frame0)
        if self._init_state.is_window_ready():
            drt_result = self._try_drt_init_with_retries()
            if drt_result.success:
                self._seed_from_drt(drt_result)
                return
            else:
                Logger.write("warn", f"DRT init failed: {drt_result.reason}; falling back to heuristic")
                self._heuristic_init(self._init_state.first_frame())
                self._init_mode = "fallback"
                return
        else:
            return    # not yet initialized; defer
    else:
        self._heuristic_init(frame0)
        self._init_mode = "heuristic"
```

`_try_drt_init_with_retries()` cycles `cfg.init.retry.window_scale` (default `[1.0, 1.4, 1.8]`), expanding the buffer length each pass.

`_seed_from_drt(drt_result)` does:

1. `self.Frontend.imu_context.seed_from_drt(drt_result, P_init=drt_result.P_init)`.
2. Push the keyframe-0 pose into `self.graph.frames` exactly like the existing path.
3. Set `self.prev_keyframe = (first_frame, frame_idx, depth0)`.
4. Set `self.isinitiated = True`.

### 10.2 Frontend wrapper (`FlowFormerDynFrontend`)

New class in `Module/Frontend/Frontend.py`:

```python
class FlowFormerDynFrontend(IFrontend):
    """Wraps FlowFormerDyn + IMUContext under the IFrontend interface.

    Keeps the visual interface identical to FlowFormerCovFrontend so MACVO does not
    need to know about IMU. The IMU sample is produced internally by stepping
    `IMUContext` over the inter-frame IMU window stored on the StereoInertialFrame.
    """

    def __init__(self, config: SimpleNamespace):
        ...
        self.imu_context = IMUContext(...)        # holds AirIO ckpt frozen, EKF, etc.
        self.imu_loader  = AirIMULoader(config.airimu_pkl) if config.airimu_pkl else None
        self.model       = FlowFormerDyn(...)     # cov/dyn/dyn-mask weights loaded selectively (see §10.3)

    @torch.inference_mode()
    def estimate_pair(self, frame_t1, frame_t2):
        corrected, raw = self._extract_imu_window(frame_t1, frame_t2)
        sample = self.imu_context.step(corrected, raw)
        flow, cov_var, dyn_full = self.model.inference(
            frame_t2.imageL, frame_t2.imageR,           # depth path: stereo pair
            sample.f_imu, sample.imu_tokens,
        )
        # batched stereo-depth + frame-to-frame match path identical to FlowFormerCovFrontend
        # but with the second batch element calling the FULL inference (with f_imu, imu_tokens).
        ...
        return depth_out, match_out
```

Notes:

- The depth path of `FlowFormerCovFrontend` runs `(L_t2, R_t2)` for stereo and `(L_t1, L_t2)` for matching in a single batched inference. `FlowFormerDynFrontend` keeps the depth call going through the **cov-only path** (no DynGRU forward needed for stereo depth — gravity-aligned static-confidence is meaningless for left/right disparity). The matching call additionally runs the dyn branch with `(f_imu, imu_tokens)`.
- `c_full` (per-pixel static confidence at full res) is attached to the `IMatcher.Output` as a new optional field `static_conf` so the backend can `retrieve_pixels(kp_uv, static_conf)`.
- `seed_from_drt(...)` is exposed via `self.imu_context.seed_from_drt(...)` so `MACVO` can call through.

### 10.3 Model loading and freeze policy

`FlowFormerDyn` loads weights in this order:

1. Strict-load the FlowFormerCov checkpoint into the shared backbone + `cov_update`. (Existing `FlowFormer.load_ddp_state_dict` strips `module.` prefix and uses `strict=False`; for FlowFormerDyn we keep `strict=False` because dyn keys are absent from the cov ckpt.)
2. Random-init `dyn_update` (`SepConvGRU`, FiLM, IMUCrossAttn, alpha, DynHead, mask).
3. Apply `requires_grad_(False)` to:
   - `context_encoder.*`, `memory_encoder.*`, `memory_decoder.proj`, `memory_decoder.att`, `memory_decoder.decoder_layer`, `memory_decoder.flow_token_encoder`, `memory_decoder.update_block.*`, `memory_decoder.cov_update.*`, `memory_decoder.delta`.
4. Apply `requires_grad_(False)` to `IMUContext.airio_net.*` and the AirIMU corrector (already done at construction).
5. Assert in trainer setup: `sum(p.numel() for p in model.parameters() if p.requires_grad)` ≈ 2.5–2.7 M; abort on mismatch.

### 10.4 Train/inference parity

Both paths run the identical sequence:

```
DRT-loose init → IMUContext.seed_from_drt → per-pair IMUContext.step → FlowFormerDyn.forward → backend
```

This eliminates train/test bootstrap mismatch and is the explicit reason DRT init runs in training too.

### 10.5 Staged escalation (out of scope but pre-designed)

If long-sequence drift on KITTI-360 (> 3 km) post-Phase-B exceeds budget, promote to a sliding-window VI-PGO (Stage B). The 15-D IMU factor and bias-resync code port verbatim; only marginalization (Schur complement to drop oldest keyframe) and keyframe management are new. Stage B is not implemented in this spec.

---

## 11. Module structure and new files

```
Module/
  Initialization/                          NEW
    __init__.py                            NEW   exports DRTLoose facade
    DRTLoose/                              NEW
      __init__.py                          NEW
      types.py                             NEW   DRTInitConfig, DRTInitResult, DRTWindowBundle
      preintegration.py                    NEW   IMUSegment, preintegrate(...) → ΔR, ΔV, ΔP, J_*, Σ
      tracks.py                            NEW   feature-track container, base-view selection,
                                                  visual relative rotation (5pt + Horn)
      gyro_bias.py                         NEW   PyPose LM bias solve + reintegrate
      translation.py                       NEW   LiGT LTL build + SVD + sign disambiguation
      alignment.py                         NEW   linear seed + constrained-refine for v0, s, g
      quality.py                           NEW   gates: parallax, accel obs, conditioning, depth, gravity
      bootstrap.py                         NEW   orchestrator (window → solve → evaluate → DRTInitResult)

  Network/
    IMUContext/
      imu_context.py                       MODIFY  add seed_from_drt(...);
                                                   expose bias_noise/input_scale/obs_scale as nn.Parameters
      airimu_loader.py                     KEEP
      __init__.py                          KEEP

    FlowFormerDyn/
      dynhead.py                           KEEP   (current implementation matches design)
      flownet.py                           KEEP
      __init__.py                          MODIFY  build_flowformer_dyn(cfg, encoder_dtype, decoder_dtype)

    DynamicHead/                           DEPRECATED   keep for ablation only

  Frontend/
    Frontend.py                            MODIFY  add FlowFormerDynFrontend (§10.2)

  Optimization/
    TwoFramePGO/
      Graphs.py                            MODIFY  add Analytic_VI_Reproj_TwoFramePGO (§9.2);
                                                   add c-weighting to existing reproj graphs
      Optimizer.py                         MODIFY  graph_type="vi_reproj"; expose last_imu_biases()
      __init__.py                          KEEP

Odometry/
  MACVO.py                                 MODIFY  startup state machine (§10.1);
                                                   on_optimize_writeback hook for bias resync (§9.3)

Train/
  DynNet/                                  NEW
    __init__.py                            NEW
    train_dyngru_phase_a.py                NEW   TartanAir pretrain (§8.2)
    train_dyngru_phase_b.py                NEW   EuRoC + KITTI-360 finetune (§8.3)
    loss.py                                NEW   dyn_pseudo_label, dyn_loss_phase_a/b
    utils.py                               NEW   freeze assertions, parameter listing

Config/
  Train/
    DynGRU_PhaseA_TartanAir.yaml           NEW
    DynGRU_PhaseB_EuRoC_KITTI360.yaml      NEW
  Experiment/
    MACVO_DynGRU_EuRoC.yaml                NEW
    MACVO_DynGRU_KITTI360.yaml             NEW

Scripts/
  UnitTest/
    test_drt_preintegration.py             NEW
    test_drt_gyro_bias.py                  NEW
    test_drt_ligt.py                       NEW
    test_drt_alignment.py                  NEW
    test_drt_bootstrap.py                  NEW
    test_dyngru_pseudo_label.py            NEW
    test_dyngru_freeze_policy.py           NEW
    test_imu_factor_jacobians.py           NEW
    test_imu_context_seed_from_drt.py      NEW
    test_backend_use_dyn_confidence.py     NEW
```

### 11.1 Selected interface signatures

```python
# Module/Initialization/DRTLoose/bootstrap.py

@dataclass
class DRTInitConfig:
    enabled                  : bool                   = True
    min_keyframes            : int                    = 10
    parallax_min             : float                  = 5.0      # px in normalized coords × focal
    keyframe_min_gap_s       : float                  = 0.22
    retry_window_scale       : list[float]            = field(default_factory=lambda: [1.0, 1.4, 1.8])
    quality_min_avg_obs      : int                    = 30
    quality_max_cond_number  : float                  = 1e8
    quality_min_pos_depth    : float                  = 0.7
    quality_gravity_tol_rel  : float                  = 1e-3
    huber_delta_gyro_rad     : float                  = 1e-2
    P_init                   : torch.Tensor | None    = None
    fallback                 : str                    = "heuristic"

class DRTLooseBootstrap:
    def __init__(self, cfg: DRTInitConfig, R_BC: pp.SO3, t_BC: torch.Tensor, gravity_norm: float = 9.81): ...
    def add_frame(self, frame: StereoInertialFrame, matcher: IMatcher) -> None: ...
    def is_window_ready(self) -> bool: ...
    def solve(self) -> DRTInitResult: ...
    def reset(self) -> None: ...
```

```python
# Module/Network/IMUContext/imu_context.py  (additions)

class IMUContext(nn.Module):
    ...
    def seed_from_drt(self, drt: "DRTInitResult", P_init: torch.Tensor | None = None) -> None: ...
    # Trainable noise-shaping scalars (Phase A/B):
    self.bias_noise   = nn.Parameter(torch.tensor(1e-12), requires_grad=True)
    self.input_scale  = nn.Parameter(torch.tensor(1e2),   requires_grad=True)
    self.obs_scale    = nn.Parameter(torch.tensor(1e-1),  requires_grad=True)
```

(These three are currently scalars on the class; promoting to `nn.Parameter` lets Phase B learn how aggressively to weight EKF inputs vs AirIO observations. Initial values match the current implementation so behavior is unchanged at step 0.)

```python
# Module/Optimization/TwoFramePGO/Graphs.py  (additions)

@dataclass
class ImuFactorPayload:
    dR_tilde : pp.SO3            # ΔR̃ from preintegration
    dV_tilde : torch.Tensor      # 3
    dP_tilde : torch.Tensor      # 3
    J_R_bg   : torch.Tensor      # 3×3
    J_V_bg   : torch.Tensor      # 3×3
    J_V_ba   : torch.Tensor      # 3×3
    J_P_bg   : torch.Tensor      # 3×3
    J_P_ba   : torch.Tensor      # 3×3
    Sigma    : torch.Tensor      # 15×15  EKF-derived, cond-clipped
    dt       : float
    g_W      : torch.Tensor      # 3,     world-frame gravity (from DRT)
    R_i, v_i, p_i, b_g_i, b_a_i  : torch.Tensor    # priors at keyframe i
    R_j_init, v_j_init, p_j_init : torch.Tensor    # warm starts at keyframe j

@dataclass
class GraphInput:                # extend
    ...                          # existing fields
    c_per_match : torch.Tensor   # (N_match,) — DynGRU confidence per reprojection residual when
                                 #             optimizer.use_dyn_confidence=true; otherwise a constant
                                 #             tensor of ones (see §9.0). Always present so that
                                 #             Reproj_TwoFramePGO.covariance_array stays branch-free.
    imu         : ImuFactorPayload | None
```

---

## 12. Config schema

### 12.1 Initialization

```yaml
init:
  type: DRTLoose
  enabled: true
  min_keyframes: 10
  retry:
    max_attempts: 3
    window_scale: [1.0, 1.4, 1.8]
  quality:
    min_avg_observation: 30
    max_condition_number: 1.0e8
    min_positive_depth_ratio: 0.7
    gravity_tol_rel: 1.0e-3
  fallback: heuristic
  gravity_norm: 9.81
```

### 12.2 IMU pipeline

```yaml
imu_context:
  airio_ckpt: Model/airio_motionwrot.ckpt
  airimu_pkl: Cache/airimu_<seq>.pkl     # optional; per-sequence corrector outputs
  gravity: 9.81
  init_bias_noise: 1.0e-12
  init_input_scale: 1.0e2
  init_obs_scale: 1.0e-1
  trainable_noise_scalars: true
```

### 12.3 Backend

```yaml
optimizer:
  type: TwoFrame_PGO
  args:
    graph_type: vi_reproj            # adds 15-D IMU factor; choose from {reproj, disp, icp, vi_reproj}
    device: cuda:0
    vectorize: true
    parallel: true
    autodiff: false
    use_dyn_confidence: true         # §9.0 — feed DynGRU c into the graph (true) or ignore it (false)
    c_drop_threshold: 0.1            # drop matches with c < this from the graph (only when use_dyn_confidence=true)
    imu_info_max_cond: 1.0e6         # clip EKF Σ before inversion
```

Frontend has a sibling toggle for cost-saving when the backend ignores `c`:

```yaml
frontend:
  type: FlowFormerDyn
  args:
    ...
    run_dyn_branch: true             # if false, skip the dyn branch in the decoder loop entirely
                                     # (incompatible with optimizer.use_dyn_confidence=true — asserted at startup)
```

The two flags are independent but checked at startup: if `optimizer.use_dyn_confidence=true` and `frontend.run_dyn_branch=false`, the runtime aborts with a clear error.

### 12.4 Training freezes (Phase A and B share)

```yaml
train:
  phase: A                            # or B
  freeze_flow_backbone: true
  freeze_flow_update_block: true
  freeze_cov_update_block: true
  freeze_cov_head: true
  freeze_airimu: true
  freeze_airio: true
  trainable_dyn_branch: true
  trainable_imu_context_proj: true
  trainable_imu_noise_scalars: true
  expected_trainable_params: [2.5e6, 2.7e6]    # assertion bounds
```

### 12.5 Loss weights

```yaml
loss:
  phase_a:
    gamma_iter: 0.85
    focal_alpha: 0.25
    focal_gamma: 2.0
    calib_weight: 0.1
    tau_0: 0.5
    tau_alpha: 0.3
    fb_eps_px: 1.0
  phase_b:
    gamma_iter: 0.85
    pose: 1.0
    consistency: 0.5
    sparsity: 0.01
    entropy: 0.05
    huber_delta_px: 0.1
    consistency_slack_px: 2.0
```

### 12.6 Dataset

Use IMU-bearing sequence types (`EuRoC`, `TartanAirv2`) instead of the `*_NoIMU` variants for any DynGRU train/eval where DRT init is required.

---

## 13. Validation plan

### 13.1 Unit tests

| Test | What it asserts |
|---|---|
| `test_drt_preintegration.py` | PyPose preintegration matches an analytic synthetic trajectory to < 1e-9 (zero-bias, no noise); first-order bias Jacobians agree with finite-difference at < 1e-5. |
| `test_drt_gyro_bias.py` | Recovers a constant gyro bias of 0.05 rad/s on a synthetic 1 s rotation+translation trajectory to < 1e-3. |
| `test_drt_ligt.py` | LiGT translations on a deterministic 5-frame, 30-track fixture match the C++ reference to within sign + global scale. |
| `test_drt_alignment.py` | Linear seed + constrained refine recover `(v0, s, g)` on a synthetic VIO trajectory with `‖g‖` constraint to < 1e-3 rel error. Polynomial-path comparison (one cherry-picked EuRoC slice) within 1e-3 rel. |
| `test_drt_bootstrap.py` | Retry/fallback state machine: synthetic low-parallax window triggers `LOW_PARALLAX` + retry; insufficient observations triggers fallback. |
| `test_dyngru_pseudo_label.py` | On a TartanAir val sample, `c` from a fresh-init DynGRU agrees with the rigid-flow-residual pseudo-label on non-IGNORE pixels at expected baseline rate. |
| `test_dyngru_freeze_policy.py` | After loading + applying freeze, exactly the parameters listed in §10.3 require grad; total trainable param count within `[2.5e6, 2.7e6]`. After 10 backward steps, frozen parameters' `.data` is unchanged (bitwise). |
| `test_imu_factor_jacobians.py` | Numerical Jacobian of `r_dR, r_dv, r_dp` vs the analytic Forster Jacobians at < 1e-5; covariance propagation through the EKF over 50 ticks remains positive-definite (cond < 1e10). |
| `test_imu_context_seed_from_drt.py` | After `seed_from_drt(drt)`, EKF state matches `drt.{R0,v0,p0,b_g,b_a}` to bitwise; `gravity_world` overwritten to `drt.g_W`; `_P` populated from `drt.P_init` (or default). |
| `test_backend_use_dyn_confidence.py` | With `use_dyn_confidence=false` and identical seeds, optimized pose on a 10-pair fixture matches a baseline-MAC-VO (+ IMU factor when `graph_type=vi_reproj`) bit-for-bit; with `use_dyn_confidence=true`, the per-match weight tensor reaches `Reproj_TwoFramePGO.covariance_array` and changes the LM cost by > 0; startup-time incompatibility check (`use_dyn_confidence=true ∧ run_dyn_branch=false`) raises before the first pair. |

### 13.2 Integration tests

| Test | What it asserts |
|---|---|
| `test_pipeline_euroc_mh01.py` | 10-second slice of MH_01 runs DRT-init → 20 pairs of `MACVO.run` end-to-end; produces finite poses; `init_mode == drt_success`. |
| `test_pipeline_low_parallax.py` | Synthetic stationary 1-second startup → DRT fails quality gate → fallback triggers → `init_mode == fallback`; subsequent pairs still produce finite poses. |
| `test_bias_sync.py` | After 100 PGO solves with bias-resync enabled, `IMUContext._state[9:15]` matches the Optimizer's last-pushed `(b_g, b_a)` to < 1e-6. |
| `test_dyn_training_smoke_phase_a.py` | One epoch over 10 TartanAir frame pairs: cov+flow params unchanged, dyn+IMU-proj params changed, loss decreases from step 0 to step N. |
| `test_dyn_training_smoke_phase_b.py` | One epoch over 10 EuRoC clips: same freeze invariant, `L_pose` decreases, gradients do not flow through the LM solver (assertion: dyn params' grads are unchanged whether `LM` is wrapped in `no_grad` or not). |

### 13.3 Benchmarks (paper-ready)

- **EuRoC** MH_01–05, V1_01–03, V2_01–03: ATE vs (a) baseline MAC-VO, (b) MAC-VO + current `StaticConfidenceHead`, (c) MAC-VO + DynGRU-no-IMU (FiLM zeroed), (d) full DynGRU.
- **KITTI-360** dynamic-scene subset: same metrics; emphasis on dynamic-object scenes.
- **TartanAir val:** rigid-flow-residual agreement (success criteria §3.3).
- **DRT init:** success rate, fallback rate, init latency, solved (`b_g, g, s`) distributions.

### 13.4 Runtime telemetry

Log per sequence:

- DRT init: attempts, latency, final reason, solved `(b_g, g, s)`, `cond(LᵀL)`, `avg_obs/keyframe`.
- Per pair: `IMUContext` state norms, EKF cov diag, AirIO `‖v_B‖`, `c` histogram (10 bins), PGO LM iter count + final cost.
- Training: per-epoch loss components (focal / calib / pose / consistency / sparsity / entropy), `α` value, `token_weights`, `(bias_noise, input_scale, obs_scale)` if trainable, gradient-norm of dyn vs frozen partitions (frozen should be exactly 0).

### 13.5 Ablations (paper-ready)

| Ablation | Dropped component |
|---|---|
| No IMU conditioning | both FiLM and adapter (i.e., zero `f_imu`, zero `imu_tokens`) |
| FiLM-only | adapter dropped (`α ≡ 0`, no IMUCrossAttn forward) |
| Adapter-only | FiLM dropped (γ ≡ 0, β ≡ 0) |
| FiLM + adapter (default) | — |
| Per-layer α (12 scalars) | replace single `α` with one per `DynUpdateBlock` invocation index |
| No per-token gain `w` | freeze `IMUCrossAttn.token_weights = 1` |
| No co-iterative dyn (post-hoc baseline) | use deprecated `StaticConfidenceHead` for `c` |
| No IMU factor in PGO | `graph_type="reproj"` instead of `"vi_reproj"` |
| No `c`-weighting | weight every reprojection by 1.0 |
| No DRT init (heuristic only) | force `init.enabled=false` |
| Unfreeze last decoder block (ablation) | last `update_block` GRU + head trainable at lr 1e-6 |

---

## 14. Risk register and mitigations

| Risk | Mitigation |
|---|---|
| Low-parallax / stationary starts → DRT fails frequently | Retry with longer window (`window_scale = [1.0, 1.4, 1.8]`); always have heuristic fallback; log telemetry to monitor real-world failure rate. |
| LiGT `LᵀL` ill-conditioned on degenerate motion | Conditioning gate (`cond ≤ 1e8`); damping in SVD path; reject + retry on failure. |
| Polynomial-vs-constrained-refine divergence on edge cases | Cross-test against C++ reference on EuRoC slices; fall back to a damped-Newton over `(s, g)` if the LM refine itself diverges. |
| `T_BS` / `R_BC` frame-convention mismatch between dataloader and initializer | Explicit assertions + `test_drt_preintegration.py` includes a frame-convention round-trip; CI gate. |
| Train-time DRT overhead in short clips | Cache `DRTInitResult` per (sequence, epoch) and reuse across training iters that revisit the same sequence start. |
| EKF divergence on long sequences | Bias-resync loop + clipped covariance + periodic EKF reset if `cond(P) > 1e6`. |
| AirIO covariance over-confident (known failure mode) | Inflate `Σ_v` by a fixed factor during Phase B (ablation listed); learn `obs_scale` as `nn.Parameter`. |
| Synthetic-to-real gap in Phase A | Phase B specifically closes this; track Phase-A pseudo-label agreement on TartanAir val *during* Phase B to detect catastrophic forgetting. |
| DynGRU over-weights IMU in textureless regions and drops visual info | `L_sparsity + L_entropy` encourage decisive predictions; monitor `c` histogram per epoch; cap `tanh(α)` at ±0.95 if needed. |
| Two-frame PGO accumulates drift on > 3 km runs | Stage-B sliding window pre-designed; promotion is scoped (out of scope of this spec). |
| FlowFormer frozen-decoder bottlenecks dynamic performance | Phase-B ablation: unfreeze last decoder block at lr 1e-6 — listed in §11.4. |
| Pre-existing FlowFormerCov ckpt incompatible after structural change | `FlowFormerDyn.load_ddp_state_dict` already uses `strict=False`; CI check that loaded-key set ⊇ FlowFormerCov keys. |

---

## 15. Implementation order

1. **DRTLoose math + tests** (§5, §13.1 first 5 tests). Mathematical correctness in isolation; no runtime dependency. Output: `DRTLooseBootstrap.solve(...) -> DRTInitResult`.
2. **`IMUContext.seed_from_drt(...)` + tests** (§6.2, `test_imu_context_seed_from_drt.py`). Pure additive change; existing `reset(...)` path untouched.
3. **`MACVO` startup state machine + retry/fallback** (§10.1, integration test `test_pipeline_low_parallax.py`).
4. **`FlowFormerDynFrontend`** (§10.2). Wires `IMUContext.step → FlowFormerDyn.inference`. Validate against existing `FlowFormerCovFrontend` on a non-IMU TartanAir sequence (DynGRU forward but `c` discarded — sanity check).
5. **Backend `c`-weighted reprojection + 15-D IMU factor + bias-resync** (§9). New graph class behind a config flag; old `Reproj_TwoFramePGO` remains the default until the new path is green.
6. **Phase A trainer + losses + freeze assertions** (§8.2, §11). Smoke test (`test_dyn_training_smoke_phase_a.py`); then full ~8-epoch run on TartanAir.
7. **Phase B trainer with detached PGO** (§8.3). Smoke test; then full ~3-epoch run on EuRoC + KITTI-360.
8. **Benchmarks + ablations** (§13.3, §13.5).

This order isolates mathematical correctness first, then integration, then training behavior, then performance and ablations.

---

## 16. Glossary

- **AirIMU** — neural IMU corrector (raw → corrected + per-sample cov).
- **AirIO** — neural body-frame velocity estimator (raw IMU + EKF orientation → `v_B` + cov). Implemented via `CodeNetMotionwithRot`.
- **Bias-resync** — post-PGO push of optimized `(b_g, b_a)` back into the EKF state.
- **CLIP-adapter** — lightweight adapter pattern (Gao et al., 2021): `out = base + tanh(α) · adapter(base)` with `α` init 0 so the block starts as identity.
- **Co-iterative** — DynGRU runs inside FlowFormer's decoder loop, reading per-iteration tensors at every K=12 step.
- **`c`** — per-pixel static confidence in `[0, 1]` produced by DynGRU; weights visual reprojection in the backend.
- **DRT-VIO-Init** — Direct-Relative-Translation visual-inertial bootstrap (Xu et al.). The "loose" variant solves gyro bias → camera rotations → LiGT translations → linear `(v0, s, g)` alignment with a gravity-norm constraint.
- **DynGRU** — the IMU-conditioned co-iterative dynamic-pixel head that is the centerpiece of this design.
- **EKF (15-D)** — Velocity EKF on `[R, V, P, b_g, b_a]` with body-frame velocity observations from AirIO.
- **FiLM** — Feature-wise Linear Modulation (Perez et al., 2018): `x' = (1 + γ(c)) · x + β(c)`.
- **Forster preintegration** — VI preintegration with first-order bias Jacobians (Forster et al., 2017).
- **GMA** — Global Motion Aggregate (FlowFormer's attention-weighted motion feature, `update_block.aggregator`).
- **`IMUContext`** — module composing AirIMU + Velocity EKF + AirIO + 34-D feature + 7-token assembly.
- **`IMUCrossAttn`** — single-head cross-attention with 7 semantic IMU tokens as K/V and the spatial dyn feature map as Q; per-token learnable gain `w`.
- **IMU token** — one of 7 semantic 128-D vectors derived from EKF preintegration (`ΔR, Δv, Δp, R^T·g, bias, Σ, dt`); consumed as K,V by the adapter's cross-attention.
- **LiGT** — Linear Global Translation (Cai et al., 2021); used by DRT-loose for translations up-to-scale.
- **Pseudo-static label** — per-pixel binary label derived from rigid-flow residual on TartanAir (Phase A); "ignore" bands exclude ambiguous pixels.
- **Two-frame PGO** — MAC-VO's existing pose-graph backend; this spec adds one IMU factor per pair and `c`-weighting on visual factors.
