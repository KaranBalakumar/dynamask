# DynGRU Implementation Progress

**Design spec:** `2026-04-20-dyngru-imu-dynamic-head-design.md`
**Last updated:** 2026-04-25

This document tracks what has been implemented and tested so far, and what remains. It is scoped to the DynGRU + IMU-context + backend work described in the design spec; unrelated MAC-VO modules are out of scope.

---

## 1. Implemented and tested

### 1.1 `FlowFormerDyn/` — visual head (co-iterative DynGRU inside FlowFormer decoder)

Location: `Module/Network/FlowFormerDyn/`

| File | Status | Notes |
|---|---|---|
| `__init__.py` | ✅ | Factory `build_flowformer_dyn(cfg, encoder_dtype, decoder_dtype)` |
| `flownet.py` | ✅ | `FlowFormerDyn(FlowFormer)` — mirrors the `FlowFormerCov` pattern (direct subclass of `FlowFormer`, not `FlowFormerCov`), forward `(image1, image2, f_imu, imu_tokens, prev_dyn_net?, prev_flow?) → (flow_preds, cov_preds, dyn_preds)`; `inference()` wraps with `InputPadder` and bilinear-upsamples dyn logits from H/4 to full resolution |
| `dynhead.py` | ✅ | All classes + helpers (see breakdown below) |

#### `dynhead.py` — per-class status

| Class / function | Status | Role |
|---|---|---|
| `FiLMLayer` | ✅ | `x' = (1+γ)·x + β`, (γ,β) from `Linear(cond_dim, 2·feat_dim)` of `f_imu`; broadcasts over spatial dims |
| `IMUCrossAttn` | ✅ | Single-head cross-attention, spatial Q attends to 7 IMU tokens; includes **per-token learnable weights** `nn.Parameter(torch.ones(7))` applied before K/V projections (§3.6.2 addition) |
| `DynHead` | ✅ | 4-conv stack 128→256→128→64→1 at H/8, mirrors `CovHead` with 1-logit output |
| `DynUpdateBlock` | ✅ | Sibling of `CovUpdateBlock`; pipeline: SepConvGRU → FiLM → `+ tanh(α)·IMUCrossAttn(film_output, tokens)` → (delta_dyn 1-ch, mask 36-ch); α init 0 per CLIP-adapter pattern |
| `upsample_dyn_logits` | ✅ | 2× RAFT convex upsample (H/8 → H/4), mask shape `(N, 1·2·2·9, H, W)` |
| `warp_prev_dyn_net` | ✅ | `grid_sample(padding='zeros', mode='bilinear', align_corners=True)` — OOB pixels zero out; no cov-gate (§3.4 decision) |
| `MemoryDynDecoder(MemoryDecoder)` | ✅ | Subclasses `MemoryDecoder` **directly** (not `MemoryCovDecoder`); owns both `cov_update` (`CovUpdateBlock` reused from FlowFormerCov) and `dyn_update` (`DynUpdateBlock`) as siblings to the built-in flow `update_block`; threads `(f_imu, imu_tokens, prev_dyn_net?, prev_flow?)` through the K=12 iteration loop; returns `(flow_predictions, cov_predictions, dyn_predictions)` |

#### Param budget (verified empirically)

```
DynUpdateBlock               = 3,018,413 params
├─ gru       (SepConvGRU)    = 1,969,920
├─ film      (FiLM)          =    33,024
├─ imu_attn  (IMUCrossAttn)  =    49,539   (incl. 7-D token_weights)
├─ alpha     (scalar)        =         1
├─ dyn_head  (4 convs)       =   664,193
└─ mask      (2× upsample)   =   304,420   (2 convs: 128→256→36)
```

Total matches spec §3.5 target of ~2.65M trainable (excluding `IMUPipeline.feature_mlp` + `token_projs`, which live in `IMUContext/`, ~25k more).

#### Verified numerically (tests run with `python3 -c ...` scripts)

- **FiLM reconstruction** — γ, β from forward proj recover exactly from forced conditioner.
- **IMUCrossAttn** — attention rows sum to 1; Q·K^T/√d·V matches closed-form; uniform-token case produces uniform-over-tokens attention.
- **Token weights** — shape `(7,)`, init `== 1.0`; gradient flows; zeroing `token_weights` produces measurably different output from all-ones (bias-only contribution remains, as expected).
- **DynUpdateBlock gating** — at `α=0` the block output equals `FiLM(h, f_imu)` (adapter contribution cancels); as `α→∞`, `tanh(α)→1` saturates.
- **upsample_dyn_logits** — shape ✓, RAFT convex-combination sum = 1 over 9 neighbors ✓, top-left and center neighbor placement correct ✓, row-major tile layout matches reshape order.
- **`warp_prev_dyn_net`** — identity case (zero flow → `warp(prev) == prev`); 1-px shift (`warp(prev, disp=(1,0))[y,x] == prev[y,x+1]`); huge flow → OOB → zeros (`max|out| < 1e-6`).
- **Optional prev-state kwargs** — `MemoryDynDecoder.forward` and `FlowFormerDyn.forward` accept `prev_dyn_net` / `prev_flow`; both default `None` → fallback init `flow_net.clone()` (first-pair behavior). `prev_cov_logvar` was removed after user-raised objection about co-trained signal being redundant.
- **Forster preintegration** — ΔR, Δv, Δp, g_B all match closed-form to 1e-9 (verified earlier during math-check pass).

### 1.2 `IMUContext/` — IMU pipeline (AirIMU + Velocity EKF + Air-IO, closed loop)

Location: `Module/Network/IMUContext/`

| File | Status | Notes |
|---|---|---|
| `__init__.py` | ✅ | Package exports |
| `airimu_loader.py` | ✅ | AirIMU checkpoint loader (frozen corrector) |
| `imu_context.py` | ✅ | `IMUContext(nn.Module)` — EKF + Air-IO fused; `VelocityEKFDynamics(IMUstate)` 15-D dynamics subclass with `state_transition` (strapdown) and `observation` (body-frame velocity from `R^T·V`); **`seed_from_drt(drt, P_init=None)`** added (see §1.3) |

**Outputs per `step()` (see `IMUSample` dataclass):**
- `f_imu` — `(1, 128)` global feature from `feature_mlp(z_raw)`
- `imu_tokens` — `(1, 7, 128)` via 7 per-slot `Linear` projections
- `z_raw` — `(34,)` composed as `cat(dR, dv, dp, diag(Σ_{9}), v_B, diag(Σ_v), b_g, b_a, dt, g_B)`
- `state` — `(15,)` EKF state `[R(3), V(3), P(3), b_g(3), b_a(3)]`
- `P_diag` — `(15,)` EKF covariance diag
- `airio_vel, airio_cov` — body-frame velocity + diag covariance from Air-IO

**Token layout (matches §2.3):** `dR(3)`, `dv(3)`, `dp(3)`, `g = R^T·g_W (3)`, `bias = (b_g, b_a) (6)`, `cov = diag(Σ_{ΔR,Δv,Δp}) (9)`, `dt(1)` — each projected to 128-D independently.

### 1.3 `Initialization/DRTLoose/` — DRT-loose VIO initializer

Location: `Module/Initialization/DRTLoose/`

PyPose-native port of DRT-VIO-Init (Zhao 2023) adapted for stereo+IMU bootstrap: gyro-bias refinement → visual rotations via preintegration → LiGT translations → linear (v₀, s, g) alignment with gravity-norm constraint. All arithmetic in float64.

| File | Status | Notes |
|---|---|---|
| `__init__.py` | ✅ | Re-exports all public symbols |
| `types.py` | ✅ | `DRTInitConfig` (quality gates, retry policy, fallback mode) + `DRTInitResult` (payload fields R0/v0/p0/b_g/b_a/g_W/scale/P_init, factory methods `.failure(reason, retry)` and `.success_result(...)` with auto-constructed 15×15 block-diagonal P_init) |
| `preintegration.py` | ✅ | `IMUPreintegrator` + `PreintResult`; Forster 2017 SO3 manifold ops via `pp.so3`; bias Jacobians J_R_bg, J_V_bg, J_V_ba, J_P_bg, J_P_ba with correct sign (−Jr·dt per eq. 45); 9×9 covariance propagation |
| `tracks.py` | ✅ | `FeatureTrack`, `KeyframeBundle`, `tracks_with_min_obs`, `select_base_views` (max cross-product parallax), `_bearing` (pixel → unit 3-vector) |
| `translation.py` | ✅ | `build_LTL` (epipolar-constraint LTL accumulation), `recover_translations` (smallest right-singular vector of LTL), `resolve_translation_sign` (majority-vote on A_lr@t), `check_ltl_conditioning` (cond ≤ max_cond) |
| `gyro_bias.py` | ✅ | `rotation_residual` (SO3 log residual), `solve_gyro_bias` Gauss-Newton with Huber weighting (δ=1e-2 rad) and first-order bias correction |
| `alignment.py` | ✅ | `linear_alignment` builds `(6*(N-1), 3N+4)` system for velocities/scale/gravity; solved via `torch.linalg.lstsq(rcond=1e-10)`; gravity normalized to `g_norm`; `AlignmentResult` dataclass |
| `quality.py` | ✅ | Six gate functions: `check_avg_observation` (≥30), `check_acceleration_observability` (mean deviation + static-sample count), `check_ltl_conditioning_gate` (≤1e8), `check_positive_depth` (≥0.7 ratio), `check_gravity_consistency` (\|‖g‖-G\|/G ≤ 1e-3), `check_state_finite` (finite + ‖b_g‖≤1.0, ‖b_a‖≤5.0); `run_all_quality_gates` |
| `bootstrap.py` | ✅ | `DRTLooseBootstrap` stateful accumulator (`add_frame`, `is_window_ready`, `solve`, `reset`); `run_drt_with_retry(cfg, solver_fn)` tries `min(max_attempts+1, len(window_scales))` scales then falls back to heuristic |

**`IMUContext.seed_from_drt(drt, P_init=None)`** — added to `imu_context.py`; sets EKF state from DRT result (R0→SO3.Log, v0, p0, b_g, b_a), overwrites `gravity_world`, initializes `_P` from `drt.P_init` (or 15×15 identity if absent).

**Unit tests (152 passing, 8 pre-existing local failures excluded):**

| Test file | Count | Coverage |
|---|---|---|
| `test_drt_preintegration.py` | 5 | constant-acc closed-form, b_a FD Jacobian, b_g FD Jacobian (right-tangent-space), nonzero-bias, reset |
| `test_drt_translation_alignment.py` | 32 | tracks utilities, LiGT LTL build + recovery + sign disambiguation + conditioning, gyro-bias residual + Gauss-Newton, linear alignment + gravity normalization |
| `test_drt_bootstrap_policy.py` | 10 | config/result schema defaults, retry policy (fail-all, succeed-on-2nd, succeed-on-1st, max-attempts limit, fallback heuristic), quality gate thresholds, P_init construction |
| `test_drt_quality.py` | 30 | all six gate functions, edge cases, `run_all_quality_gates` ordering |
| `test_imu_context_seed.py` | 6 | `seed_from_drt` via `MinimalCtx` duck-type fixture: state vector, gravity_world, covariance init, custom P_init override, failure assertion |

**Bug fixed during implementation:** `J_R_bg` sign error (was `+Jr·dt`, should be `−Jr·dt` per Forster eq. 45) and step-index bug (`J_V_bg` was reading updated J_R_bg at step k+1 instead of step k). Both caught by code reviewer and corrected before downstream tasks.

### 1.4 `Frontend/` — `FlowFormerDynFrontend` + `static_conf` field

Location: `Module/Frontend/`

| Change | Status | Notes |
|---|---|---|
| `Frontend.py` — `FlowFormerDynFrontend` | ✅ | Wraps `FlowFormerDyn` under `IFrontend` interface; exposes `self.imu_context = None` attribute; `estimate_pair` returns `IMatcher.Output` with `static_conf=dyn` populated |
| `Matching.py` — `IMatcher.Output.static_conf` | ✅ | New optional field `torch.Tensor \| None = None` for DynGRU static confidence map; pre-existing typeguard 4.5.1 incompatibility in `from_partial_cov` also fixed |

### 1.5 Dyn training mode

Location: `Train/MatchingNet/`

| Change | Status | Notes |
|---|---|---|
| `utils.py` — `T_TrainType` | ✅ | Added `"dyn"` literal |
| `train_flowformer.py` — model build | ✅ | `if train_mode == "dyn":` builds `FlowFormerDyn` instead of `FlowFormerCov` |
| `train_flowformer.py` — freeze schedule | ✅ | `case "dyn":` freezes all, unfreezes `dyn_update.parameters()` only, asserts `cov_update` frozen |
| `Config/Train/FlowFormerDyn_Demo.yaml` | ✅ | Demo training config: `training_mode: dyn`, `batch_size: 4`, `lr: 2e-4`, `weight_decay: 1e-4` |

### 1.6 MACVO startup state machine

Location: `Odometry/MACVO.py`

| Change | Status | Notes |
|---|---|---|
| `_drt_init_buffer: list` | ✅ | Added to `__init__`; accumulates frames until DRT window is ready |
| `_handle_init(frame)` | ✅ | Checks `cfg.init.enabled` + IMU presence, accumulates buffer, calls `run_drt_with_retry`, falls back to heuristic if DRT fails or disabled |
| `_seed_from_drt(drt_result, frame)` | ✅ | Calls `frontend.imu_context.seed_from_drt` if frontend exposes the attribute; pushes identity SE3 pose to graph as first anchor |

### 1.7 MACVO experiment configs — `init:` block added

All four MACVO configs now include the `init:` block wired to `DRTLoose`:

| Config file | Status |
|---|---|
| `Config/Experiment/MACVO/MACVO_Fast.yaml` | ✅ |
| `Config/Experiment/MACVO/MACVO_Performant.yaml` | ✅ |
| `Config/Experiment/MACVO/Paper_Reproduce.yaml` | ✅ |
| `Scripts/UnitTest/assets/test_config/MACVO/MACVO.yaml` | ✅ |

### 1.8 Design spec additions (all landed in the approved spec)

- §3.4 — inter-frame warp rewritten as pure `grid_sample(padding='zeros')`, cov-gate explicitly rejected with rationale (co-trained → correlated → no independent signal).
- §3.6.2 — per-token learnable weights documented; motivation (ΔR/Δp dominate, dt/Σ secondary), init=1, lives in `IMUCrossAttn` adapter branch only.
- §3.6.3 — attention math updated to show `T' = w ⊙ imu_tokens` before K/V projections.
- §3.5 — param budget table has a row for `token_weights` (7 params).
- §4.2 — `L_calib = (c − exp(−r²/σ²))²` added to Phase A loss (not Phase B, which lacks the rigid-flow residual); weight 0.1.

---

## 2. Pending

### 2.1 AirIMU / EKF / Air-IO — verification and any remaining glue

- **Air-IO checkpoints and config paths** — spec names `cfg.airio_ckpt`; confirm the loader in `IMUContext` can find the pretrained weights in `Module/Network/Air-IO/pre-trained/`.
- **EKF ↔ PGO bias-sync path** (`IMUContext.push_biases(b_g, b_a)`) — interface hook is referenced in spec §6.4 but not yet wired; needs to accept optimized biases post-PGO and reset EKF bias point + covariance rows. The EKF itself (`IMUEKF` from Air-IO) supports this but the wrapper method is not yet present on `IMUContext`. (`seed_from_drt` is implemented and seeds startup state; the post-PGO update hook is a separate remaining item.)
- **Unit test: EKF propagation vs pypose closed-form** (spec §7.1 `test_ekf_propagation.py`).
- **Unit test: IMU factor numerical-vs-analytic Jacobian** (spec §7.1 `test_imu_factor_jacobians.py`).

### 2.2 Initialization — DRT-VIO-Init (loose)

**✅ Implemented** — see §1.3 for full module breakdown and test coverage.

Remaining scaffolding gap: `DRTLooseBootstrap.solve()` currently uses identity visual rotations as placeholders for `R_cam`. To produce accurate translations and alignment, it must receive real visual rotation estimates from the frontend (either from `FlowFormerDynFrontend` or the stereo matcher). This wiring is the primary remaining item; the math pipeline itself is complete.

### 2.3 Backend — two-frame PGO + 15-D IMU factor

**Explicitly deferred by user.**

- **15-D IMU factor** (spec §5.1) — `r_IMU = [r_ΔR, r_Δv, r_Δp, r_bg, r_ba]`, Forster form with EKF-derived preintegration and first-order bias corrections; info matrix = `(EKF Σ_{k+1})^{-1}` with cond-number clip. Implemented as a pypose factor node.
- **`c`-weighted reprojection factors** — each per-pixel residual scales by the DynGRU confidence `c_i`; pixels with `c_i < 0.1` dropped.
- **Bias-ref sync loop** — post-LM, push optimized `(b_g, b_a)` back to EKF (hook in 2.1).
- **Files** (per spec §6.2):
  - `Pipeline/backend_pgo.py` — modify to add IMU factor + c-weighted reprojection
  - `Pipeline/vo_pipeline.py` — modify to instantiate `IMUContext` + `FlowFormerDyn`; wire the bias-sync loop

### 2.4 Losses

None implemented yet (spec §4 is complete on paper; no code).

- **Phase A (`L_A`):**
  - Rigid-flow residual computation from GT pose + GT depth (per-pixel)
  - Depth-adaptive threshold `τ(D_i) = τ_0 + α·(f/D_i)·||t_GT||` (τ_0=0.5 px, α=0.3)
  - Pseudo-label assignment with IGNORE band + forward-backward consistency mask
  - `FocalBCE` (α=0.25, γ=2) on per-iter logits with γ_k = 0.85^(K−1−k)
  - `L_calib = (sigmoid(ℓ_k) − exp(−r²/σ²))²` using `r` from the pseudo-label residual and `σ` from `cov_predictions[k]`; weight 0.1
- **Phase B:**
  - `L_pose` (weight 1.0) — `c`-weighted GT-pose reprojection residual
  - `L_consistency` (weight 0.5) — 3-frame static-consensus under estimated flow
  - `L_sparsity` (weight 0.01) — `mean(1 − c)`
  - `L_entropy` (weight 0.05) — binary entropy on `c`
- **LM detach trick** — gradients do not flow through the PGO solver; residual-reweighting gradient only. Implementation pattern: forward-through-LM, backward gated at LM output (RAFT-VO / DROID-SLAM style).

### 2.5 Training pipeline

- **DataLoader — Phase A (TartanAir):** pair sampling with (image1, image2, T_GT, D_GT, K_intrinsics); synthesized IMU from GT pose + bias/noise profiles. IMU synthesis routine is not yet present.
- **DataLoader — Phase B (EuRoC + KITTI-360):** stereo pair + real IMU window, synchronized at camera timestamp. Real IMU parsing code may partially exist inside Air-IO — needs a MAC-VO-facing adapter.
- **Training loop:** AdamW, cosine lr 2e-4 → 1e-5, wd 1e-4. Phase A: batch 8, 2× A6000, ~8 epochs (~60 h). Phase B: batch 4, ~3 epochs (~25 h).
- **Freezing schedule:** FlowFormerCov / AirIMU / Air-IO frozen in Phase A; optional unfreeze of FlowFormer's last decoder block in Phase B (lr 1e-6 on those params).

### 2.6 Tests (unit + integration, per spec §7)

Committed tests as of 2026-04-25:

| Test file | Status | Scope |
|---|---|---|
| `test_drt_preintegration.py` | ✅ 5 passing | IMU preintegration, Jacobians |
| `test_drt_translation_alignment.py` | ✅ 32 passing | LiGT, gyro-bias, linear alignment |
| `test_drt_bootstrap_policy.py` | ✅ 10 passing | DRTInitConfig/Result schema, retry/fallback policy |
| `test_drt_quality.py` | ✅ 30 passing | All 6 quality gate functions |
| `test_imu_context_seed.py` | ✅ 6 passing | `seed_from_drt` wiring |

Still pending:

| Test | Scope | Status |
|---|---|---|
| `test_dyngru_cell.py` | α=0 → FiLM-only output; gradient flow | pending (informally verified inline) |
| `test_imu_cross_attn.py` | Output shape; attention weights sum to 1 | pending (informally verified) |
| `test_ekf_propagation.py` | EKF vs pypose preintegration at zero bias noise | pending |
| `test_airimu_encoder.py` | 34-D feature vector shape + finite values | pending |
| `test_imu_factor_jacobians.py` | Numerical vs analytic Jacobian for `r_ΔR, r_Δv, r_Δp` | pending (blocked on 2.3) |
| `test_pipeline_euroc_mh01.py` | 10-s MH_01 slice end-to-end, finite poses | pending (blocked on 2.3, 2.5) |
| `test_bias_sync.py` | EKF ↔ PGO biases agree within 1e-3 after 100 frames | pending (blocked on 2.1, 2.3) |
| `test_dyngru_pseudo_label_agreement.py` | `c` agrees with rigid-flow pseudo-label ≥ 80% | pending (blocked on 2.4, 2.5) |

### 2.7 End-to-end forward smoke test on GPU

Attempted on CPU-only machine; hit `RuntimeError: NVTX functions not installed` from `torch.cuda.nvtx.range` calls inherited from MemoryDecoder / CovDecoder. Same pattern exists in MemoryCovDecoder and works in CUDA environments. Needs a single forward pass on a CUDA-capable box to confirm shape/dtype/device flow through the full decoder with real inputs.

### 2.8 Benchmarks and ablations (spec §7.3, §7.4)

All blocked on §2.3–2.5.

- **Benchmarks:** EuRoC MH_01–05 / V1_01–03 / V2_01–03 ATE; KITTI-360 dynamic-scene subset; TartanAir val pseudo-label agreement rates.
- **Ablations:**
  - No-IMU (drop FiLM + adapter)
  - FiLM-only (`α ≡ 0`, drop adapter)
  - Adapter-only (drop FiLM)
  - FiLM + adapter (default)
  - Per-layer α (12 scalars instead of 1)
  - DynGRU without implicit deviation cue
  - Post-hoc head vs co-iterative DynGRU
  - No IMU factor in PGO (backend-only contribution)

### 2.9 Staged escalation (Stage B, sliding-window VI-PGO)

**Non-goal for current scope** (spec §1.3, §5.5). Deferred until Stage A hits its drift targets.

---

## 3. Summary state

| Area | Status |
|---|---|
| DynGRU cell (head + update block + upsample + warp) | ✅ implemented, numerically verified in isolation |
| FlowFormer decoder subclass wiring | ✅ implemented, **end-to-end forward not yet confirmed on GPU** |
| IMU-conditioning streams (FiLM + gated adapter + token weights) | ✅ implemented, math verified |
| IMU pipeline (EKF + Air-IO fused) | ✅ core implemented; bias-sync hook (push_biases) pending |
| `IMUContext.seed_from_drt` | ✅ implemented, 6 tests passing |
| DRT-VIO-Init (Module/Initialization/DRTLoose/) | ✅ implemented, 83 tests passing; visual-rotation wiring gap remains |
| FlowFormerDynFrontend + `static_conf` field | ✅ implemented |
| Dyn training mode + FlowFormerDyn_Demo.yaml | ✅ implemented |
| MACVO startup state machine (_handle_init, _seed_from_drt) | ✅ implemented |
| MACVO experiment configs (init: block) | ✅ all 4 configs updated |
| Backend PGO + 15-D IMU factor | ⏸ deferred (user) |
| Losses (Phase A + Phase B) | ⏳ not started |
| Training pipeline + dataloaders | ⏳ not started |
| Unit + integration tests (DynGRU cell, EKF, pipeline) | ⏳ not started (inline math-checks done, not committed) |
| Benchmarks + ablations | ⏳ blocked |

**Immediate unblocked next steps:** wire real visual rotations from the frontend into `DRTLooseBootstrap.solve()` (§2.2 gap), Phase A loss + dataloader (§2.4, §2.5 TartanAir slice), GPU smoke test (§2.7), and the remaining DynGRU + EKF unit tests (§2.6).

**Blocked on user-deferred work:** anything touching PGO (§2.3), and by extension the post-PGO bias-sync hook (§2.1).
