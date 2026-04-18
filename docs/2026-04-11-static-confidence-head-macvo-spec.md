# Static Confidence Head + DRT-VIO Loosely-Coupled Initialization + Full-Factor Backend for MAC-VO

**Date:** 2026-04-11 (rev. 2026-04-17, full rewrite)
**Target codebase:** `/home/karan/dynamic-vio/MAC-VO/` (primary).
**Reference code:**
- `/home/karan/dynamic-vio/references/drt-vio-init/` (He et al., CVPR 2023) — loosely-coupled DRT VIO initializer
- `/home/karan/dynamic-vio/references/AirIMU/` — learned IMU corrector + covariance
- `/home/karan/dynamic-vio/references/Air-IO/` — reference for the body-frame motion network (optional ablation module)
**Status:** Design ready for implementation.
**Relation to prior revisions:**
- Supersedes the 2026-04-11 draft that (i) pointed to a `dynamask_vio/references/MAC-VO/` path that does not exist in this tree, (ii) described an `imu_window`/`imu_mask` frame API that does not match `DataLoader.Interface.StereoInertialFrame`, (iii) left the H/4 refinement path as a V2 sketch, and (iv) did not specify how VIO state is **initialized** nor how preintegration factors enter the two-frame PGO.

This revision bolts three things onto the frozen-backbone confidence-head design:

1. **Sanity-checked code references.** Every class, method, and config path is pinned to a file and line range that exists in the current MAC-VO tree (§4).
2. **DRT-VIO-Init (loosely coupled) as the bootstrap.** DRT provides the initial `(R_k, p_k, v_k, b_g, b_a, g_W)` that both the confidence-head stream *and* the backend preintegration factors need (§7). Stereo MAC-VO fixes DRT's monocular scale ambiguity analytically.
3. **Preintegration factors in the two-frame PGO, and the natural extension to a sliding-window (SWF) backend.** The **same** differentiable preintegrator that feeds the head is re-used (with bias Jacobians enabled) as a hard backend constraint (§9). No second preintegrator ever exists in this tree.

---

## 0. Notation, frames, and invariants

### 0.1 Frames

- `W` — world frame (gravity-aligned, by convention the first keyframe's body frame with gravity pointing in the ‑z direction after DRT init).
- `B_k` — IMU body frame at keyframe `k`. `R_WB_k`, `p_WB_k`, `v_WB_k` ∈ ℝ³.
- `C_k` — left-camera frame at keyframe `k`. Related to body by the sensor extrinsic `T_BS ∈ SE(3)`, stored per-frame in `VisualMap.frames.data["T_BS"]` (Interface.py defines it on every `StereoData`).
- **MAC-VO stores poses of the *sensor* (left camera) in world.** `VisualMap.frames.data["pose"]` is `pp.SE3` of the camera in world (`T_WC_k`). The body pose is `T_WB_k = T_WC_k · T_BS^{-1}`. Every piece of this spec that talks about backend state `x_k` uses the **body** pose, and we convert at read/write boundaries only.

### 0.2 Coordinate conventions already baked into MAC-VO

- MAC-VO pixel-to-point functions are named `*_NED` (`Utility.Point.pixel2point_NED`, `point2pixel_NED`) — the camera's optical axis is `+x`, not `+z`. `Analytic_Reproj_TwoFramePGO.build_jacobian` uses `x, y, z = self.pos_Tc[:, 0], ...` and hard-codes `fx / x` and `-fx * y / x²`. **Every rigid-flow / reprojection formula in this spec uses the same convention.** Any closed-form that reads `X_z` should be read as `X_x` in code. Explicit translation is given where it matters (§3.4, §6.1, §9.3).
- `pypose.SE3` convention: `T.Act(X)` is `R X + t`; `T.Inv().Act(X)` is `Rᵀ(X − t)`. This is what `Reproj_TwoFramePGO.forward` (line 106) uses.

### 0.3 Hard invariants this design relies on

| # | Invariant | Enforced by |
|---|---|---|
| I1 | FlowFormerCov, the AirIMU corrector, and the DRT initializer are **never updated** by the dynamic-head training loop. | §5.5 freeze checks; all frozen forwards inside `torch.no_grad()` / `torch.inference_mode()`; outputs detached before crossing into the head. |
| I2 | The preintegrator used in the confidence-head's per-pair call **is the same Python class** as the one that emits `(ΔR̂, Δv̂, Δp̂, Σ_imu, J_*)` to the backend. | §4.3 — `Module/Network/AirIMU/preintegration.py::DifferentiablePreintegrator` is a single class with a `share_with_backend: bool` flag controlling Jacobian emission. |
| I3 | Ground-truth pose is used **only** to compute labels in `Train/DynamicHead/loss.py`. It is never read by `StaticConfidenceHead.forward`. | §8.2 (loss composition) + §5.4 (head forward signature). |
| I4 | The backend's visual factors are weighted by `w_i = c_i` exactly as in the two-frame PGO; the IMU factor is added as a separate residual block with its own covariance. Dynamic objects cannot hide in the IMU factor. | §9.4. |
| I5 | DRT-loosely-coupled output state **is in body frame, world axes**. When MAC-VO writes the first `pose` back to `VisualMap.frames`, it converts body→camera via `T_BS`. | §7.5. |

---

## 1. Goal

Freeze MAC-VO's FlowFormerCov flow-plus-covariance backbone and AirIMU's learned corrector. Add four trainable / non-parametric modules:

1. **A differentiable IMU preintegrator** (non-parametric) that, on every frame pair, emits the full Forster-style factor `(ΔR̂, Δv̂, Δp̂, Σ_imu, dt)` and, when asked, the bias Jacobians `(J_R_bg, J_v_bg, J_v_ba, J_p_bg, J_p_ba)`. This is the **only** preintegrator in the tree; both the head and the backend factors read from it (I2).
2. **A tiny IMU FeatureMLP** (`~15k` params, trainable) that projects the preintegration factor to a 128-D conditioning vector `f_imu`.
3. **A static-confidence head** (`~0.6M` params, trainable) that, per frame pair, reads
   - frozen FlowFormerCov context features `f_ctx` (H/8),
   - frozen FlowFormerCov flow+covariance (H/8, matched by bilinear down-scale),
   - `f_imu`,
   - a per-pixel "IMU-rigid residual" proxy (see §3.4),
   - a recurrent hidden state,

   and outputs a **single per-pixel static confidence** `c ∈ [0, 1]` interpreted as "this pixel is both rigid-scene in motion and trackable as a correspondence". `c` is the single scalar the PGO uses. (A factorized two-output variant `c = p_static · p_visible` is retained behind `dynamic_head.factorize: true` as an ablation; see §5.6.)
4. **A single-channel refinement path at H/4** that sharpens object boundaries without adding meaningful inference cost.

Plus two system-level additions that are part of this document:

5. **DRT-loosely-coupled VIO bootstrap** (stereo-adapted) that estimates `(R_k, p_k, v_k, b_g, g_W)` from the first `N_init ≈ 10` keyframes and writes them into `VisualMap.frames` before the first PGO call (§7). `b_a` is left at zero as a prior and estimated online in the backend.
6. **Preintegration factors in the 2-frame PGO and sliding-window backend**, sharing the state with (5) and consuming the exact output of (1) (§9).

All downstream consumers (MAC-VO's two-frame PGO, the new sliding-window backend, and any trajectory evaluator) see the confidence-weighted visual residuals described in §6, plus the inertial residuals described in §9.3.

---

## 2. Frozen vs. trainable inventory

| Module | Location | Params | Trainable? | Load policy |
|---|---|---|---|---|
| FlowFormerCov context/memory encoders & memory decoder | `MAC-VO/Module/Network/FlowFormer{,Cov}` | ~15M | **No** | Loaded from `cfg.frontend.args.weight`. |
| FlowFormerCov covariance branch (`MemoryCovDecoder`) | `MAC-VO/Module/Network/FlowFormerCov/covhead.py` | in the 15M above | **No** | Same. |
| Stereo depth (runs FlowFormerCov on L,R pair inside `FlowFormerCovFrontend.estimate_pair`) | `MAC-VO/Module/Frontend/Frontend.py:218-232` | — | **No** | Same. |
| AirIMU `CodeNet` (bias + per-sample `σ_a², σ_g²`) | new: `MAC-VO/Module/Network/AirIMU/corrector.py` (ported from `references/AirIMU/model/code.py`) | ~1–3M | **No** | Loaded from `cfg.frontend.args.imu.airimu_weights`; `.eval()` + `requires_grad_(False)`. |
| `DifferentiablePreintegrator` (Forster SO(3)×ℝ⁶; 1st-order cov prop; optional bias Jacobians) | new: `MAC-VO/Module/Network/AirIMU/preintegration.py` | 0 (non-parametric) | N/A | — |
| `IMUEncoder` (corrector ∘ preintegrator ∘ `FeatureMLP`) | new: `MAC-VO/Module/Network/AirIMU/encoder.py` | FeatureMLP ~15k | **MLP only**, upstream frozen | MLP trained jointly with head. |
| `StaticConfidenceHead` (ConvGRU @ H/8 + FiLM + refinement @ H/4) | new: `MAC-VO/Module/Network/DynamicHead/` | ~0.6M | **Yes** | Trained in `Train/DynamicHead/`. |
| Temperature scalar `T` | buffer in the head | 1 scalar | Post-hoc fit | `Train/DynamicHead/calibrate.py` sweeps ECE. (Factorized ablation adds a second buffer `T_vis`.) |
| DRT-VIO-Init (loosely coupled, stereo-adapted) | new: `MAC-VO/Module/Initialization/DRTLoose/` | 0 (solver) | N/A | Runs once at startup inside `MACVO.initialize_window`. |
| Sliding-window IMU factor graph (`SWF_VIOFactorGraph`) | new under `MAC-VO/Module/Optimization/SlidingWindow/` | 0 (solver) | N/A | Backend only. |

**Hard-freeze enforcement** (I1) is done two ways in parallel:

1. On construction: `for p in module.parameters(): p.requires_grad_(False)` and `module.eval()`.
2. At call time: the whole frozen subtree is called inside `torch.inference_mode()` (inherited from `FlowFormerCovFrontend.estimate_pair`, which already applies the decorator in MAC-VO; see `Module/Frontend/Frontend.py:217`). IMU outputs used as head inputs are `.detach()`ed before crossing the frozen → trainable boundary.

Pre-flight asserts in `Train/DynamicHead/train.py` fail the job rather than silently unfreezing anything (§5.5).

---

## 3. Architecture

### 3.1 End-to-end data flow, one frame pair `(t, t+1)`

```
┌──────────────────────── frozen ───────────────────────────────────────────────┐
│                                                                                │
│  (stereo_t, stereo_{t+1})                                                      │
│       └─► FlowFormerCovFrontend.estimate_pair                                  │
│              ├─► depth_{t+1}, disparity_{t+1}, depth_cov_{t+1}                 │
│              ├─► flow_{t→t+1}  [1, 2, H,    W ]  (full-res after padder unpad) │
│              ├─► flow_cov_{t→t+1} [1, 2, H, W ]                                │
│              └─► f_ctx = FlowFormerCov.last_context       [1, 128, H/8, W/8]   │
│                                                                                │
│  (imu_window = frame_{t+1}.imu: IMUData over [t_t, t_{t+1}])                   │
│       └─► AirIMU.CodeNet.inference                                             │
│              ├─► corrected_acc, corrected_gyro  [1, N_imu, 3]                  │
│              └─► per-sample acc_cov, gyro_cov   [1, N_imu, 3]                  │
│                                                                                │
│       └─► DifferentiablePreintegrator(                                         │
│              corrected_acc, corrected_gyro, acc_cov, gyro_cov, dt,             │
│              bias_ref=(b_g_prev, b_a_prev),                                    │
│              emit_jacobians=share_with_backend                                 │
│           )                                                                    │
│              ├─► ΔR̂ ∈ SO(3)          [1, 3, 3]                                 │
│              ├─► Δv̂, Δp̂              [1, 3] each (body frame, B_t)             │
│              ├─► Σ_imu               [1, 9, 9]                                 │
│              ├─► dt                   scalar                                   │
│              └─► (J_R_bg, J_v_bg, J_v_ba, J_p_bg, J_p_ba)  when emit_jacobians │
│                                                                                │
│       └─► FeatureMLP(z_imu)   where                                            │
│              z_imu = [log ΔR̂, Δv̂, Δp̂, σrepr(Σ_imu), Δb_g, Δb_a, dt]            │
│              └─► f_imu ∈ ℝ^128                                                 │
│                                                                                │
│       └─► rigid_flow_IMU(depth_{t+1}, K, ΔR̂_cam, Δp̂_cam)                       │
│              where (ΔR̂_cam, Δp̂_cam) = T_BS⁻¹ · (ΔR̂, Δp̂) · T_BS                 │
│              └─► (Δf, r_imu, r_imu_norm, valid) per-pixel at full res          │
│                                                                                │
└────────────────────────────────────────────────────────────────────────────────┘
                                   │
                                   ▼
┌─────────────────────── trainable (StaticConfidenceHead) ──────────────────────┐
│                                                                                │
│  H/8 stream (temporal):                                                        │
│    x8   = Conv1x1(cat(f_ctx, flow_lo, cov_lo, proxy_lo), 128) → GN → SiLU      │
│    x8   = FiLM_global(x8, f_imu)            (+ depth-bin FiLM if enabled)      │
│    h8   = ConvGRUCell(x8, h8_prev)          [1, 128, H/8, W/8]                 │
│    y8   = Conv3x3(h8, 64) → SiLU → Conv1x1(64, 1)   → [1, 1, H/8, W/8]         │
│           (1 channel: logit_8)                                                  │
│                                                                                │
│  H/4 refinement (spatial-only, non-recurrent):                                 │
│    u4   = F.interpolate(y8, scale=2, mode='bilinear')     → [1, 1, H/4, W/4]   │
│    cue4 = cat( H/4 image grads, H/4 flow, H/4 cov, H/4 proxy,                  │
│                F.interpolate(h8, scale=2) )                                    │
│    r4   = DSConv3x3(cue4, 64) → SiLU → Conv1x1(64, 1)                          │
│    y4   = u4 + r4                                          [1, 1, H/4, W/4]    │
│                                                                                │
│  Upsample to full:                                                             │
│    y    = F.interpolate(y4, (H, W), mode='bilinear', align_corners=False)      │
│    c    = σ(y[:,0] / T.clamp(min=1e-3))                   [1, 1, H, W]         │
│                                                                                │
└────────────────────────────────────────────────────────────────────────────────┘
                                   │
                                   ▼
            Sampled at keypoints in MACVO.run_pair  ─►  PGO residual weighting
                                   │
                                   ▼
                Backend: sliding-window PGO with IMU preintegration factors
```

### 3.2 The head module (signatures)

`Module/Network/DynamicHead/head.py`:

```python
class StaticConfidenceHead(nn.Module):
    """
    Trainable part of the pipeline. Reads only frozen/detached features.
    Returns a single per-pixel static-confidence map `c ∈ (0, 1)` at full resolution.
    """
    def forward(
        self,
        f_ctx:  torch.Tensor,   # [B, 128, H/8, W/8]   (from FlowFormerCov.last_context)
        flow:   torch.Tensor,   # [B, 2,   H,   W  ]   (full-res forward flow)
        cov:    torch.Tensor,   # [B, 2,   H,   W  ]   (flow covariance diagonal)
        f_imu:  torch.Tensor,   # [B, 128]             (FeatureMLP(z_imu))
        proxy:  torch.Tensor,   # [B, 4,   H,   W  ]   (Δf_x, Δf_y, r_imu_norm, valid)
        image:  torch.Tensor,   # [B, 3,   H,   W  ]   (left image; used ONLY for H/4 gradients)
        depth:  torch.Tensor,   # [B, 1,   H,   W  ]   (for depth-bin FiLM, optional)
        h8_prev: torch.Tensor | None,   # [B, 128, H/8, W/8] or None → zeros
    ) -> "HeadOut":
        ...
```

```python
@dataclass
class HeadOut:
    c:           torch.Tensor   # [B, 1, H, W]   in (0, 1)   — static confidence
    logits_8:    torch.Tensor   # [B, 1, H/8, W/8]   (pre-temperature, H/8 stream output)
    logits_4:    torch.Tensor   # [B, 1, H/4, W/4]   (pre-temperature, after refinement)
    h8_new:      torch.Tensor   # [B, 128, H/8, W/8] (next hidden state)
```

When `dynamic_head.factorize=true` (ablation only), the head emits two channels
`(logit_static, logit_visible)`, two temperatures `(T, T_vis)`, and `HeadOut` gains
`p_static, p_visible` fields with `c = p_static · p_visible`. The ablation plumbing
lives behind a single `if self.factorize:` branch in `StaticConfidenceHead.__init__`
and `.forward` — no duplicated modules.

**Parameter count (default):** Conv1x1 entry ~80k; FiLM ~65k; ConvGRUCell at 128 ch, 3×3 ~450k; H/8 head convs ~80k; H/4 refiner (depthwise-separable 3×3 + 1×1, ~96 ch) ~12k; **total ≈ 0.69M**.

### 3.3 H/4 refinement path — complete specification

The previous revision listed H/4 refinement as a V2 sketch. Here is the fully specified path.

**Motivation.** Dynamic-object boundaries are *exactly* the pixels where a misclassification hurts the PGO most (a 1-px boundary error shifts a keypoint by its stride × nominal parallax). H/8 resolution smooths these boundaries; H/4 is a good trade between boundary precision and compute (a depthwise 3×3 at H/4 costs roughly 2× a 3×3 at H/8).

**Inputs to the refiner (all at H/4):**

| Channel block | Source | Dim |
|---|---|---|
| Upsampled H/8 logits | `F.interpolate(y8, scale_factor=2)` | 1 |
| Upsampled H/8 hidden state | `F.interpolate(h8, scale_factor=2)` | 128 |
| Image gradients | Scharr filter on `image` downsampled to H/4, per-channel L2 norm | 3 |
| Flow | `F.interpolate(flow, scale_factor=1/2)` applied to H/2 flow, or direct average-pool of H/1 flow to H/4; values rescaled by 1/4 to stay in the H/4 pixel units | 2 |
| Flow cov | Same pooling policy as flow | 2 |
| Proxy | Same pooling policy as flow | 4 |

Total: `1 + 128 + 3 + 2 + 2 + 4 = 140` channels. (Factorized ablation: `2 + 128 + 3 + 2 + 2 + 4 = 141`.)

**Refiner architecture (fixed):**

```python
# Module/Network/DynamicHead/refiner.py
class RefinerH4(nn.Module):
    def __init__(self, in_ch=140, mid_ch=64, out_ch=1):
        super().__init__()
        self.dw = nn.Conv2d(in_ch, in_ch, kernel_size=3, padding=1, groups=in_ch)
        self.pw = nn.Conv2d(in_ch, mid_ch, kernel_size=1)
        self.norm = nn.GroupNorm(8, mid_ch)
        self.act = nn.SiLU(inplace=True)
        self.out = nn.Conv2d(mid_ch, out_ch, kernel_size=1)   # out_ch=2 when factorized
        # zero-init the output so the refiner starts as an identity residual
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, z):
        return self.out(self.act(self.norm(self.pw(self.dw(z)))))
```

**Zero-init on the output Conv1×1** is load-bearing. At training step 0 the refiner contributes no residual, so the H/8 logits are exactly the final logits. This is the "start as identity and learn to improve" inductive bias; identical to how residual MLPs in transformers are initialized. Without it, the H/4 refiner injects random noise at step 0 and we lose several hundred steps of trainable progress.

**Why no recurrence at H/4.** Temporal memory at H/4 quadruples the ConvGRU parameter count and only adds value if the phenomenon has sub-H/8 temporal structure — which dynamic-object boundaries do **not** (the boundary at frame `t+1` is adjacent to the boundary at frame `t` up to the flow field, which the network can warp from the H/8 hidden state). So recurrence is explicitly *not* used here — ablation A10 confirms this choice on VIODE.

### 3.4 The IMU-rigid proxy (per-pixel, full-res)

**Why per-pixel and not just global `f_imu`.** The head needs *some* input signal that is geometrically aligned with the GT-pose-based target used in `L_dyn` (§5.2), else the train/test input distributions differ by the entire rigid-flow channel. We use IMU-derived pose instead of GT pose to build that channel at both train and test — same forward-pass distribution at both stages.

**Formulation in MAC-VO's `_NED` camera (x-forward):**

Given:
- `depth[x] = D(x) > 0` (metric depth from frozen stereo).
- `(ΔR̂_cam, Δp̂_cam) = T_BS⁻¹ · (ΔR̂, Δp̂) · T_BS` expressed as an SE(3) action on camera-frame points.
- `K` — 3×3 intrinsic, NED convention.

Per-pixel rigid flow:

```
X_t        = pixel2point_NED(x, D(x), K)                       # camera frame, x-forward
X_{t+1}    = ΔR̂_cam · X_t + Δp̂_cam                              # camera frame at t+1
x_rigid    = point2pixel_NED(X_{t+1}, K)                       # projected pixel
f_rigid[x] = x_rigid − x
Δf[x]      = f_obs[x] − f_rigid[x]
r_imu[x]   = ||Δf[x]||₂
```

**Depth-adaptive normalization:** `r_imu` has a mean that scales with `|Δp̂_cam| / D(x)`. We normalize per pixel:

```
τ(D, Δp̂)       = τ₀ + α · (fx / max(D, D_min)) · ||Δp̂_cam||₂
r_imu_norm[x]  = r_imu[x] / τ(D, Δp̂)
```

`τ₀` captures flow-estimator noise; `α` captures per-velocity sensitivity. Defaults `τ₀ = 1.0 px, α = 1.0` from the prior revision's config; tuned on VIODE.

**Validity mask** (all four conditions ANDed):

1. `D(x) > d_min ∧ D(x) < d_max` — reject no-return / sky pixels. Defaults `0.5` and `80.0` m.
2. `X_{t+1}.x > ε_z` — after motion, the point must still be in front of the camera (in MAC-VO's NED, `.x` is forward). Default `ε_z = 1e-6`.
3. `x_rigid ∈ [edge_width, W − edge_width] × [edge_width, H − edge_width]` — rigid-flow projection stays inside the image margin used by MAC-VO (same `edgewidth` from `odomcfg.args`).
4. `Σ_imu` passes a sanity check: `trace(Σ_imu[3:6, 3:6]) < c_p² ∧ trace(Σ_imu[0:3, 0:3]) < c_R²`. Defaults `c_p = 5.0 m, c_R = 0.5 rad`. If violated, `valid = 0` over the whole image.

**At pixels with `valid = 0`, set `(Δf_x, Δf_y, r_imu_norm) := 0`** before feeding to the head, so the head can distinguish "geometric signal is not available" from "geometric signal is small".

**Implementation:** one function in `Module/Network/AirIMU/proxy.py`, called from `StaticConfidence_FlowFormerCovFrontend.estimate_pair` *after* the preintegrator call, *before* the head forward. This function is pure PyTorch; no custom CUDA kernels.

### 3.5 FiLM conditioning

Two variants selectable by config `model.frontend.args.dynamic_head.imu_fusion`:

- `film` (default): per-channel `(γ, β) = FiLMMLP(f_imu)`. Modulates each of the 128 channels of `x8` uniformly across space.
- `depth_bin_film` (ablation A12): soft-bin pixels into `K_b = 3` depth bins `[0, 5), [5, 20), [20, ∞)` m with a partition of unity (cubic falloff); predict `(γ_b, β_b) : b ∈ 1..K_b` from `f_imu`; apply weighted modulation:
  ```
  x8' = Σ_b w_b(x) · (γ_b ⊙ x8 + β_b)     where  Σ_b w_b(x) = 1.
  ```
- `none` (ablation A4): skip FiLM; IMU branch disabled end-to-end.

Theory justification is in method-theory §5.4; this spec just pins the interface.

### 3.6 Streaming semantics

- **Training (BPTT):** `h8_0 = zeros` at the start of each windowed sample; window length `W = 4`; gradients flow through all 4 recurrence steps. No cross-window hidden-state carry.
- **Inference:** `h8_prev` is stored as an attribute on `StaticConfidence_FlowFormerCovFrontend`. Reset conditions:
  1. Explicit call to `frontend.reset_stream()` (e.g. on a new sequence).
  2. `dt_between_pairs_ns > dt_reset_ms * 1e6` — IMU gap exceeded.
  3. The DRT initializer runs at the *start* of a sequence; when it succeeds, it calls `frontend.reset_stream()` so the first post-init pair starts with a clean hidden state.

Between frames at inference the hidden state is `.detach()`ed so the computational graph does not accumulate across inference calls.

---

## 4. MAC-VO code changes

This section is the exhaustive file-by-file plan. Every file path is relative to `/home/karan/dynamic-vio/MAC-VO/` unless otherwise noted.

### 4.1 New: `Module/Network/AirIMU/` — three co-equal modules

| File | Role | Parameters | Freeze |
|---|---|---|---|
| `corrector.py` | AirIMU `CodeNet` (exactly as `references/AirIMU/model/code.py`). Exposes `.inference(imu_batch) -> {cov_state, correction_acc, correction_gyro}` returning per-sample outputs. Interval `= 9` by default. | ~1–3M | Frozen; loaded from checkpoint in `from_config`. |
| `preintegration.py` | `DifferentiablePreintegrator(pp.nn.Module)` with a Forster SO(3)×ℝ⁶ forward pass and 1st-order covariance recursion. Signature below. | 0 | N/A |
| `encoder.py` | `IMUEncoder = corrector ∘ preintegrator ∘ FeatureMLP`. Returns a dict (see §4.2). | FeatureMLP only (~15k) | MLP trainable; upstream frozen. |

**`DifferentiablePreintegrator.forward` contract:**

```python
@dataclass
class PreintOut:
    delta_R:      torch.Tensor   # [B, 3, 3]
    delta_v:      torch.Tensor   # [B, 3]        (in B_t body frame)
    delta_p:      torch.Tensor   # [B, 3]        (in B_t body frame)
    Sigma:        torch.Tensor   # [B, 9, 9]     (ordered [rot, vel, pos])
    dt_total:     torch.Tensor   # [B]           (seconds, float32)
    # Optional Jacobians; None when emit_jacobians=False
    J_R_bg:       torch.Tensor | None   # [B, 3, 3]
    J_v_bg:       torch.Tensor | None   # [B, 3, 3]
    J_v_ba:       torch.Tensor | None   # [B, 3, 3]
    J_p_bg:       torch.Tensor | None   # [B, 3, 3]
    J_p_ba:       torch.Tensor | None   # [B, 3, 3]
    # Linearization point (needed for first-order re-correction at bias update)
    bias_ref:     torch.Tensor   # [B, 6]        (b_g_ref | b_a_ref concatenated)

def forward(
    self,
    corrected_acc:  torch.Tensor,   # [B, N, 3]
    corrected_gyro: torch.Tensor,   # [B, N, 3]
    acc_cov:        torch.Tensor,   # [B, N, 3]   (per-sample diagonal variances from AirIMU)
    gyro_cov:       torch.Tensor,   # [B, N, 3]
    dt:             torch.Tensor,   # [B, N]      (per-sample delta-time)
    bias_ref:       torch.Tensor,   # [B, 6]      (b_g, b_a at start of interval; zeros if no prior)
    emit_jacobians: bool,           # True on backend path, False elsewhere (faster)
) -> PreintOut: ...
```

The forward runs the standard Forster recursion (Forster et al., "On-Manifold Preintegration for Real-Time Visual-Inertial Odometry", T-RO 2017):

```
R_0 = I;  v_0 = 0;  p_0 = 0
for k = 1..N:
    dR_k   = Exp((ω̂_k − b_g) · dt_k)
    R_k    = R_{k-1} @ dR_k
    v_k    = v_{k-1} + R_{k-1} @ (â_k − b_a) · dt_k
    p_k    = p_{k-1} + v_{k-1} · dt_k + 0.5 · R_{k-1} @ (â_k − b_a) · dt_k²
    Σ_k    = F_k · Σ_{k-1} · F_kᵀ + G_k · N_k · G_kᵀ    # Forster eq. 62–64
ΔR̂    = R_N
Δv̂    = v_N
Δp̂    = p_N
Σ_imu = Σ_N
```

with the `F_k, G_k` Jacobians from Forster. Bias Jacobians are accumulated by the standard linear recurrence (Forster eqs. 69–72). The bias reference point `(b_g, b_a) = bias_ref` is the linearization point; first-order correction at the backend is `ΔR̂_corr = ΔR̂ · Exp(J_R_bg · (b_g − b_g_ref))`, etc.

**Why we do not just use `pp.module.IMUPreintegrator`.** pypose's preintegrator does not emit bias Jacobians; without those, the backend must **re-run** preintegration at every LM step that updates bias, which (a) breaks I2 (we'd need a second implementation to avoid re-burning the AirIMU corrector every step) and (b) breaks real-time budget. Our wrapper computes the same `(ΔR̂, Δv̂, Δp̂, Σ_imu)` as pypose on the frontend path (`emit_jacobians=False`) and additionally the five Jacobians on the backend path.

**Sanity test** `Train/DynamicHead/tests/test_preintegrator.py`:
1. Feed a synthetic IMU batch with `bias = 0`; assert `||ΔR̂ − pp.IMUPreintegrator(...)_out.rot||_F < 1e-4`.
2. Verify Jacobians by finite differences against `forward` evaluated at `bias ± ε`.
3. Covariance positive-semi-definiteness: `min(eig(Σ_imu)) > −1e-8`.

### 4.2 New: `Module/Network/AirIMU/encoder.py`

```python
class IMUEncoder(nn.Module):
    """Corrector ∘ Preintegrator ∘ FeatureMLP.  The single point where the frontend
    reaches IMU quantities."""
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.corrector = AirIMUCorrector.from_ckpt(cfg.airimu_weights)
        self.corrector.eval()
        for p in self.corrector.parameters():
            p.requires_grad_(False)
        self.preint = DifferentiablePreintegrator()
        self.feature_mlp = FeatureMLP(
            in_dim = 25 if cfg.sigma_repr == "diag" else 61,
            out_dim = cfg.feature_dim,
        )

    def forward(self, imu: IMUData, bias_ref: torch.Tensor, emit_jacobians: bool):
        with torch.no_grad():
            corr = self.corrector.inference({
                "acc":  imu.acc,    # [B, N, 3]
                "gyro": imu.gyro,   # [B, N, 3]
            })
        corrected_acc  = imu.acc  + corr["correction_acc"]
        corrected_gyro = imu.gyro + corr["correction_gyro"]
        acc_cov = corr["cov_state"]["acc_cov"]
        gyro_cov = corr["cov_state"]["gyro_cov"]

        pre = self.preint(
            corrected_acc, corrected_gyro,
            acc_cov, gyro_cov, dt=imu.time_delta.float() * 1e-9,
            bias_ref=bias_ref, emit_jacobians=emit_jacobians,
        )
        # Build z_imu
        z = torch.cat([
            pp.SO3(pp.mat2SO3(pre.delta_R)).Log(),  # [B, 3]   log of rotation
            pre.delta_v, pre.delta_p,               # [B, 3] each
            sigma_repr(pre.Sigma, mode=self.cfg.sigma_repr),
            bias_ref[:, :3], bias_ref[:, 3:],
            pre.dt_total.unsqueeze(-1),
        ], dim=-1)

        f_imu = self.feature_mlp(z)

        return {
            "f_imu":        f_imu,
            "delta_R":      pre.delta_R,
            "delta_v":      pre.delta_v,
            "delta_p":      pre.delta_p,
            "Sigma_preint": pre.Sigma,
            "dt_total":     pre.dt_total,
            "J_R_bg":       pre.J_R_bg,
            "J_v_bg":       pre.J_v_bg,
            "J_v_ba":       pre.J_v_ba,
            "J_p_bg":       pre.J_p_bg,
            "J_p_ba":       pre.J_p_ba,
            "bias_ref":     pre.bias_ref,
        }
```

`FeatureMLP` is two Linear-SiLU layers (`in_dim → 128 → 128 → out_dim`).

### 4.3 New: `Module/Network/DynamicHead/` — head package

```
Module/Network/DynamicHead/
├── __init__.py         # exports StaticConfidenceHead, build_head
├── convgru.py          # ConvGRUCell (+ ConvLSTMCell placeholder for ablation A5')
├── film.py             # FiLM and DepthBinFiLM
├── refiner.py          # RefinerH4 (zero-init residual conv stack)
├── head.py             # StaticConfidenceHead assembled from the above
└── README.md           # 1-paragraph summary + config schema
```

Implementation notes that pin the ambiguous choices:

- `ConvGRUCell` uses spectral-norm on all four convs (update, reset, candidate, output). Lipschitz-bounded ConvGRU is what has worked for us empirically on small recurrent heads; skipping it destabilizes training at `lr = 3e-4`.
- `logit_clip` = 10.0 applied with `torch.clamp` *before* the temperature division. fp16 overflow is a real risk here; clamp is cheaper than forcing fp32.
- Temperature buffer:
  ```python
  self.register_buffer("T", torch.tensor(1.0))
  # Only in factorized ablation:
  # self.register_buffer("T_vis", torch.tensor(1.0))
  ```
  Fit post-hoc by `Train/DynamicHead/calibrate.py` using an ECE sweep over `T ∈ [0.5, 3.0]` with 50 bins.

### 4.4 New: `Module/Frontend/Frontend.py::StaticConfidence_FlowFormerCovFrontend`

Subclass, not a replacement — every existing config keeps working. Full implementation:

```python
class StaticConfidence_FlowFormerCovFrontend(FlowFormerCovFrontend):
    """
    FlowFormerCov frontend that additionally runs a trainable static-confidence head
    and attaches `c` (the per-pixel static confidence) to its IMatcher.Output.
    """
    def __init__(self, config: SimpleNamespace):
        super().__init__(config)
        # Tap the context encoder (one line in flownet.py, §4.5).

        from ..Network.AirIMU.encoder import IMUEncoder
        self.imu_encoder = IMUEncoder(config.imu)
        self.imu_encoder.eval()
        for p in self.imu_encoder.corrector.parameters():
            p.requires_grad_(False)

        from ..Network.DynamicHead.head import StaticConfidenceHead
        self.dynamic_head = StaticConfidenceHead.from_config(config.dynamic_head)
        if getattr(config.dynamic_head, "weight", None):
            ckpt = torch.load(config.dynamic_head.weight, map_location=config.device,
                              weights_only=True)
            self.dynamic_head.load_state_dict(ckpt["head_state_dict"])

        self.dynamic_head.to(config.device)

        # Streaming state
        self._h8_prev: torch.Tensor | None = None
        self._last_frame_ns: int | None = None
        self._dt_reset_ns: int = int(getattr(config.dynamic_head, "dt_reset_ms", 500)) * 1_000_000

        # Bias state for IMU preintegrator — updated by backend on every keyframe.
        # Shared across frame pairs until the backend writes back new bias.
        self._bias_ref: torch.Tensor = torch.zeros(1, 6, device=config.device)

    # Exposed so the backend can write fresh bias after each optimization step.
    def update_bias_ref(self, b_g: torch.Tensor, b_a: torch.Tensor) -> None:
        self._bias_ref = torch.cat([b_g.view(1, 3), b_a.view(1, 3)], dim=-1)

    def reset_stream(self) -> None:
        self._h8_prev = None
        self._last_frame_ns = None

    def _maybe_reset(self, frame_ns: int) -> None:
        if self._last_frame_ns is not None \
           and (frame_ns - self._last_frame_ns) > self._dt_reset_ns:
            self.reset_stream()

    @torch.inference_mode()
    def _frontend_frozen_pass(self, frame_t1, frame_t2):
        """Runs parent estimate_pair AND exposes f_ctx via the patched flownet."""
        depth_out, match_out = super().estimate_pair(frame_t1, frame_t2)
        # estimate_pair batches [t2-stereo, t1-t2-mono]; last_context has B=2.
        f_ctx = self.model.last_context[1:2].detach()   # mono pair at index 1
        return depth_out, match_out, f_ctx

    def estimate_pair(self, frame_t1, frame_t2):
        """
        Contract: same (IStereoDepth.Output, IMatcher.Output) return as the parent,
        with `match_out.c` (per-pixel static confidence) attached.
        """
        depth_out, match_out, f_ctx = self._frontend_frozen_pass(frame_t1, frame_t2)

        # IMU branch.
        imu = frame_t2.imu   # IMUData of the pair (imu window between t1 and t2)
        with torch.no_grad():
            imu_out = self.imu_encoder(
                imu, bias_ref=self._bias_ref, emit_jacobians=False,
            )
        # Convert (ΔR̂, Δp̂) from body to camera using T_BS
        T_BS = frame_t2.stereo.T_BS.to(self.config.device)     # pp.SE3, [1, 7]
        delta_R_cam, delta_p_cam = body2cam_se3(
            imu_out["delta_R"], imu_out["delta_p"], T_BS
        )                                                     # (R, t) in camera frame

        # Per-pixel proxy at full res.
        from ..Network.AirIMU.proxy import build_imu_proxy
        proxy_full = build_imu_proxy(
            depth   = depth_out.depth,                     # [1, 1, H, W]
            K       = frame_t2.stereo.frame_K,             # [3, 3]
            flow    = match_out.flow,                      # [1, 2, H, W]
            delta_R = delta_R_cam, delta_p = delta_p_cam,
            Sigma_imu = imu_out["Sigma_preint"],
            cfg = self.config.dynamic_head,
        )                                                 # [1, 4, H, W]

        # Head forward (trainable — NO torch.no_grad here).
        self._maybe_reset(int(frame_t2.stereo.frame_ns))
        head_out = self.dynamic_head(
            f_ctx  = f_ctx,
            flow   = match_out.flow,
            cov    = match_out.cov,
            f_imu  = imu_out["f_imu"],
            proxy  = proxy_full,
            image  = frame_t2.stereo.imageL.to(self.config.device),
            depth  = depth_out.depth,
            h8_prev = self._h8_prev,
        )
        self._h8_prev = head_out.h8_new.detach()
        self._last_frame_ns = int(frame_t2.stereo.frame_ns)

        # Attach to match_out so downstream (MACVO.run_pair) can sample it.
        match_out.c = head_out.c.squeeze(1)    # [1, H, W]

        return depth_out, match_out

    @classmethod
    def is_valid_config(cls, config: SimpleNamespace | None) -> None:
        super().is_valid_config(config)
        cls._enforce_config_spec(config, {
            "imu":          lambda v: v is not None,
            "dynamic_head": lambda v: v is not None,
        })
```

`body2cam_se3` lives in `Utility/Math.py` and is a 10-line pypose helper — no new dependency.

### 4.5 `Module/Network/FlowFormerCov/flownet.py` — one-line context tap

The context encoder already runs inside `FlowFormerCov.forward` (line 22 of the current flownet.py). Add a single stash assignment:

```python
def forward(self, image1, image2):
    image1 = ((2 * image1) - 1.0).to(dtype=self.enc_dtype)
    image2 = ((2 * image2) - 1.0).to(dtype=self.enc_dtype)

    with torch.cuda.nvtx.range("Context Encoder"):
        context = self.context_encoder(image1)
        self.last_context = context   # ← NEW: stash for DynamicHead consumers
    ...
```

The parent `estimate_pair` batches `[stereo-t2, mono-t1-t2]` into a single pair (see Frontend.py:218-232), so `self.last_context.shape == (2, 128, H/8, W/8)`. The subclass takes `[1:2]` (the mono temporal pair). This slice is documented in the new subclass's docstring.

### 4.6 `Module/Map/VisualMap.py` — schema extensions

Add **three** new fields to `FrameStore.data`:

```python
"vel"    : AutoScalingTensor((self.init_size, 3), grow_on=0, dtype=torch.float32),
"bias_g" : AutoScalingTensor((self.init_size, 3), grow_on=0, dtype=torch.float32),
"bias_a" : AutoScalingTensor((self.init_size, 3), grow_on=0, dtype=torch.float32),
```

Add **one** new field to `MatchStore.data`:

```python
"c"      : AutoScalingTensor((self.init_size, 1), grow_on=0, dtype=torch.float32),
```

Default value on insert is `1.0` for `c` (so legacy code where it is not set behaves as "fully static").

Add a **new bundle** `IMUEdgeStore` for the backend's preintegration factors:

```python
self.imu_edges = IMUEdgeStore(
    index=AutoScalingTensor((self.init_size,), grow_on=0, dtype=torch.long),
    data={
        "from_frame"  : AutoScalingTensor((self.init_size,),          ..., dtype=torch.long),
        "to_frame"    : AutoScalingTensor((self.init_size,),          ..., dtype=torch.long),
        "delta_R"     : AutoScalingTensor((self.init_size, 3, 3),     ..., dtype=torch.float64),
        "delta_v"     : AutoScalingTensor((self.init_size, 3),        ..., dtype=torch.float64),
        "delta_p"     : AutoScalingTensor((self.init_size, 3),        ..., dtype=torch.float64),
        "Sigma"       : AutoScalingTensor((self.init_size, 9, 9),     ..., dtype=torch.float64),
        "dt"          : AutoScalingTensor((self.init_size,),          ..., dtype=torch.float64),
        "bias_ref"    : AutoScalingTensor((self.init_size, 6),        ..., dtype=torch.float64),
        "J_R_bg"      : AutoScalingTensor((self.init_size, 3, 3),     ..., dtype=torch.float64),
        "J_v_bg"      : AutoScalingTensor((self.init_size, 3, 3),     ..., dtype=torch.float64),
        "J_v_ba"      : AutoScalingTensor((self.init_size, 3, 3),     ..., dtype=torch.float64),
        "J_p_bg"      : AutoScalingTensor((self.init_size, 3, 3),     ..., dtype=torch.float64),
        "J_p_ba"      : AutoScalingTensor((self.init_size, 3, 3),     ..., dtype=torch.float64),
    }
)
```

`Template.py` is edited in parallel to extend `FrameFeature`, `MatchingFeature`, and add a new `IMUEdgeFeature` Literal.

### 4.7 `Odometry/MACVO.py` — initialization, keypoint sampling, and bias write-back

Three edits.

**(a) Bootstrap phase.** Replace the current `initialize(frame0)` with a staged bootstrap that collects the first `N_init = cfg.init.drt_window_len` keyframes, runs DRT-loose, and writes initial state:

```python
def initialize(self, frame0: T_SensorFrame):
    # Stage 0: push frame0 with identity pose, zero velocity, zero biases as a scratch anchor.
    self.graph.frames.push(FrameNode.init({
        "pose":        pp.identity_SE3(1).tensor(),
        "T_BS":        frame0.stereo.T_BS,
        "need_interp": torch.tensor([0], dtype=torch.bool),
        "time_ns":     torch.tensor([frame0.stereo.frame_ns], dtype=torch.long),
        "K":           frame0.stereo.K,
        "baseline":    frame0.stereo.baseline,
        "vel":         torch.zeros((1, 3), dtype=torch.float32),
        "bias_g":      torch.zeros((1, 3), dtype=torch.float32),
        "bias_a":      torch.zeros((1, 3), dtype=torch.float32),
    }))
    self.OutlierFilter.set_meta(frame0.stereo)

    # Cache for the bootstrap window:
    self._init_frames: list[T_SensorFrame]             = [frame0]
    self._init_depths: list[Module.IStereoDepth.Output] = [
        self.Frontend.estimate_depth(frame0.stereo)
    ]
    self.prev_keyframe = (frame0, 0, self._init_depths[0])
    self.isinitiated = False       # NOT yet — full bootstrap runs when window is full.
    self._init_complete = False
```

Each subsequent call to `run` while `not self._init_complete` extends `_init_frames` and `_init_depths`. When `len(self._init_frames) == N_init`:

```python
from Module.Initialization.DRTLoose import DRTLooseInitializer
initializer = DRTLooseInitializer(self.Frontend, self.imu_encoder, cfg=self.config.init)
init_out = initializer.run(self._init_frames, self._init_depths)

if init_out.ok:
    self._write_init_to_map(init_out)     # §7.5
    self._init_complete = True
    self.isinitiated = True
    self.Frontend.reset_stream()
else:
    # DRT failed — drop oldest frame and try again next call.
    self._init_frames.pop(0); self._init_depths.pop(0)
```

**(b) Keypoint sampling.** In `run_pair`, after `kp0_uv`/`kp1_uv` are computed and the `inbound_mask` is applied (line 205 area), sample `c`:

```python
# Sample c at kp0 locations (H×W agreeable shape)
if getattr(match01, "c", None) is not None:
    kp_c = self.Frontend.retrieve_pixels(
        kp0_uv, match01.c.unsqueeze(1)   # [1, 1, H, W]
    ).squeeze(0)   # [N_kp]  (or [1, N_kp] — squeeze as elsewhere in this file)
else:
    kp_c = torch.ones((kp0_uv.size(0),), device=self.device)

# Stage-1 hard mask.
if self.dynamic_gating_enabled:
    keep = (kp_c > self.min_c)
    for name in [
        "kp0_uv", "kp1_uv",
        "kp0_d", "kp0_disparity", "kp0_sigma_disparity", "kp0_sigma_dd",
        "kp1_d", "kp1_disparity", "kp1_sigma_disparity", "kp1_sigma_dd",
        "pos0_Tc", "pos0_covTc", "pos1_covTc", "kp_c",
    ]:
        v = locals().get(name)
        if torch.is_tensor(v): locals()[name] = v[keep]
    num_kp = kp0_uv.size(0)
```

*(The above uses `locals()` for brevity; the real patch writes each tensor explicitly.)*

Attach `c` to `MatchObs.init(...)` by adding one key:

```python
"c": kp_c.view(-1, 1).cpu(),
```

**(c) Bias write-back hook.** After `Optimizer.write_map` (inside `run_pair`, around line 188 of MACVO.py):

```python
self.Optimizer.write_map(self.graph)
for func in self.on_optimize_writeback: func(self)

# NEW: propagate fresh bias to the frontend preintegrator.
if hasattr(self.Frontend, "update_bias_ref"):
    last_idx = self.graph.frames.index[-1]
    self.Frontend.update_bias_ref(
        b_g = self.graph.frames.data["bias_g"][last_idx].to(self.device),
        b_a = self.graph.frames.data["bias_a"][last_idx].to(self.device),
    )
```

**New `is_valid_config` fields** under `odomcfg.args`:

```python
"dynamic_gating": lambda v: (v is None) or isinstance(v, SimpleNamespace),
"init":           lambda v: v is not None,
```

**`__init__` reads:**

```python
self.dynamic_gating_enabled = bool(getattr(dynamic_gating, "enabled", False))
self.min_c                  = float(getattr(dynamic_gating, "min_c", 0.2))
```

### 4.8 `Module/Optimization/TwoFramePGO/Graphs.py` — weighted reprojection

In `Reproj_TwoFramePGO.__init__`, register a `c` weight buffer:

```python
w = self.obs.data.get("c", None)
if w is None:
    w = torch.ones((self.kp2.shape[0], 1), dtype=torch.float32)
else:
    w = w.to(self.kp2.dtype).view(-1, 1)
self.register_buffer("w", w.clamp(min=1e-3))    # W_EPS floor
```

In `Reproj_TwoFramePGO.forward`:

```python
def forward(self) -> torch.Tensor:
    self.pos_Tc = self.pose2opt.Inv().Act(self.pos_Tw)
    residual = point2pixel_NED(self.pos_Tc, self.K) - self.kp2   # [N, 2]
    return self.w * residual                                     # [N, 2]
```

**Mathematical equivalence:** multiplying the whitened residual by `w_i` scales the cost of factor `i` by `w_i²`. Applying it at `forward` rather than at `covariance_array` is done to keep MAC-VO's existing `pypose` LM path unchanged — the solver's `weight = torch.block_diag(*(pinverse(cov)))` (Optimizer.py line 90) receives the unchanged `cov_kp2`, and the cost is `(w·r)ᵀ · Σ⁻¹ · (w·r) = w² · rᵀ · Σ⁻¹ · r`. This is the intended semantics (method-theory §9 gives the derivation).

In `Analytic_Reproj_TwoFramePGO.build_jacobian`, multiply the analytic Jacobian's residual-side by `self.w`:

```python
J = (J_homoKS @ J_Tinv_p).view(-1, 7)
return J * self.w.repeat_interleave(2, dim=0)    # broadcast across 2 residual rows per kp
```

A unit test `Module/Optimization/TwoFramePGO/tests/test_jacobian_scaling.py` checks the new analytic `J` matches `torch.autograd.functional.jacobian` of the scaled forward.

`ReprojDisp_TwoFramePGO` gets the same treatment in parallel (scale all 3 residual rows). `ICP_TwoframePGO` gets a `NotImplementedError` unless `cfg.graph_type == "icp"` is explicitly set and `dynamic_gating.enabled == False` (so an ablation using ICP does not silently go through an ungated graph).

**Not modified:** `covariance_array()`. Keep all gating at the residual level as per prior design.

### 4.9 `DataLoader/` — no change to `StereoInertialFrame`; new training dataset

The existing `StereoInertialFrame` already carries `imu: IMUData` between successive images (Interface.py:193-200). No frame-type change is needed. The prior revision's mention of `imu_window`/`imu_mask` was incorrect — it is just `frame.imu.{acc, gyro, time_ns, T_BS, gravity}` plus `frame.imu.time_delta` for per-sample `dt`.

**New training-only dataset** `DataLoader/Dataset/DynamicHeadTrain.py`:

```python
class DynamicHeadTrainDataset(Dataset):
    """Wraps any existing StereoInertial loader (VIODE/TartanAir2/EuRoC) and adds:
       - gt_R_rel: relative camera-to-camera rotation over each pair   [B, 3, 3]
       - gt_t_rel: relative camera-to-camera translation               [B, 3]
       - valid_depth_mask: 1[d_min < D < d_max]                        [B, 1, H, W]
    """
```

Computation of `(gt_R_rel, gt_t_rel)` in the camera frame from body-frame GT:

```python
T_WB_t   = pp.SE3(gt_pose_t)                            # body in world
T_WB_t1  = pp.SE3(gt_pose_t1)
T_BS     = pp.SE3(frame_t.stereo.T_BS)                  # body → camera
T_WC_t   = T_WB_t  @ T_BS
T_WC_t1  = T_WB_t1 @ T_BS
T_CC_rel = T_WC_t.Inv() @ T_WC_t1
gt_R_rel = T_CC_rel.rotation().matrix()
gt_t_rel = T_CC_rel.translation()
```

This one-branch formula works for VIODE, TartanAir2, and EuRoC because all three datasets ship body-frame GT + `T_BS`. If any dataset ships camera-frame GT directly, that wrapper sets `T_BS = Identity` and the same formula applies.

### 4.10 New: `Train/DynamicHead/`

```
Train/DynamicHead/
├── __init__.py
├── train.py        # argparse → cfg → DynamicHeadTrainer.run()
├── loop.py         # windowed BPTT loop (window_len = 4)
├── loss.py         # compose_total_loss: L_dyn (focal BCE) + L_smooth
├── calibrate.py    # post-hoc ECE sweep for T
├── checks.py       # freeze asserts, bias-Jacobian finite-diff test
└── tests/          # pytest parity + freeze + gradient tests
```

**Training loop:**

```python
for batch in loader:                      # batch: W = window_len consecutive pairs
    h8 = None
    losses = []
    for t in range(W):
        with torch.inference_mode():
            depth_t, match_t, f_ctx = frontend_frozen._frontend_frozen_pass(
                batch.stereo[t], batch.stereo[t+1]
            )
            imu_out = imu_encoder(batch.imu[t], bias_ref=torch.zeros(1, 6), emit_jacobians=False)
        proxy_t = build_imu_proxy(depth_t.depth, batch.K[t],
                                  match_t.flow, *body2cam(imu_out["delta_R"], imu_out["delta_p"], batch.T_BS[t]),
                                  imu_out["Sigma_preint"], cfg.dynamic_head)
        head_out = head(
            f_ctx=f_ctx, flow=match_t.flow, cov=match_t.cov,
            f_imu=imu_out["f_imu"], proxy=proxy_t,
            image=batch.stereo[t].imageL, depth=depth_t.depth,
            h8_prev=h8,
        )
        h8 = head_out.h8_new
        L_t = compose_total_loss(head_out, depth_t, match_t, batch.gt_R_rel[t], batch.gt_t_rel[t], cfg.loss)
        losses.append(L_t)

    (sum(losses) / W).backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
```

**Why a dedicated trainer rather than re-using `Train/MatchingNet/`:** that script trains the FlowFormerCov covariance branch end-to-end; it expects gradients into the backbone and uses a different optimizer and LR schedule. Our trainer explicitly runs the backbone inside `inference_mode()` and backprops only through the head + FeatureMLP. Reusing the existing trainer would mix the two contracts.

---

## 5. Loss, training protocol, and sanity checks

### 5.1 Shared preamble (computed once per training pair)

```python
# Rigid flow from GT pose + stereo depth (train-only target).
X_c_t   = backproject_NED(pixels, depth, K)                       # [B, 3, H, W]
X_c_t1  = R_gt @ X_c_t + t_gt                                     # apply relative cam-cam pose
pix_t1  = project_NED(X_c_t1, K)
f_rigid_gt = pix_t1 - pixels                                      # [B, 2, H, W]

r       = ||f_obs - f_rigid_gt||_2                                # [B, 1, H, W]
tau_D   = tau0 + alpha * (fx / clamp(depth, eps)) * ||t_gt||      # depth-adaptive threshold
d_soft  = sigmoid((r - tau_D) / kappa)
c_target = 1 - d_soft                                             # rigidity soft label

# Cycle-consistency used ONLY as a validity mask (not a separate label).
# f_bwd is produced by a 2nd frozen FlowFormerCov call with (image2, image1).
warp_err = ||f_obs + warp(f_bwd, f_obs)||_2                       # [B, 1, H, W]
M_cycle  = 1[warp_err < tau_cyc_valid]                            # binary: trackable pixel

# Validity masks.
M_infov  = 1[pix_t1 inside image bounds]
M_depth  = 1[d_min < depth < d_max]
M_valid  = M_infov & M_depth & M_cycle                            # gate for L_dyn
```

### 5.2 The two loss terms

```python
L_dyn    = (focal_bce(c, c_target, gamma) * M_valid).sum() / M_valid.sum().clamp(min=1)
L_smooth = edge_aware_smoothness(c, image)
```

The cycle-inconsistent pixels (occlusions, dynamic-boundary discontinuities) are
*masked out* of `L_dyn` rather than relabeled through a second head. A single head is
sufficient because the solver only consumes `c`; the factorization `c = p_static ·
p_visible` adds parameters without changing the signal the solver receives
(theory §5). The factorized variant is retained as ablation A_fact (§10) and its
loss adds `L_visible = focal_bce(p_visible, 1 − sigmoid((warp_err − tau_cyc) /
kappa_cyc)) * M_infov`.

### 5.3 Total loss

```python
L = λ_dyn * L_dyn + λ_sm * L_smooth
```

Defaults: `λ_dyn = 1.0, λ_sm = 0.05`.

### 5.4 What the loss deliberately does **not** include

- Proxy segmentation BCE (would tie training to a specific dynamic-object ontology).
- Entropy regularization (degrades calibration).
- Explicit temporal L1 (ConvGRU already does this, softly).
- Laplace NLL (we predict probabilities, not uncertainties).
- Photometric reprojection (redundant with FlowFormerCov's supervision).
- A separate `p_visible` head (the solver only consumes `c`; see §10.1).

(Rationale in method-theory §10.)

### 5.5 Pre-flight freeze checks — blocks training if any fails

At the top of `Train/DynamicHead/train.py::run()`:

```python
def assert_frozen(module: nn.Module, name: str):
    for p_name, p in module.named_parameters():
        assert not p.requires_grad, f"{name}.{p_name}.requires_grad is True!"

assert_frozen(frontend.model,                 "FlowFormerCov")
assert_frozen(frontend.imu_encoder.corrector, "AirIMU corrector")
assert frontend.model.training       is False
assert frontend.imu_encoder.corrector.training is False

# One-shot smoke test: run one batch, backprop, check gradients.
with torch.enable_grad():
    out = frontend.dynamic_head(...)
    out.c.sum().backward()
for p_name, p in frontend.model.named_parameters():
    assert p.grad is None, f"FlowFormerCov.{p_name} got a gradient — freeze is broken!"
for p_name, p in frontend.imu_encoder.corrector.named_parameters():
    assert p.grad is None, f"AirIMU.{p_name} got a gradient — freeze is broken!"
```

### 5.6 Gradient clip and AMP

```python
optim_cfg:
    optimizer: AdamW
    lr: 3.0e-4
    weight_decay: 1.0e-4
    schedule: cosine
    total_steps: 50000
    grad_clip_norm: 1.0
    mixed_precision: bf16         # backbone already bf16 via enc_dtype; head autocast bf16
```

The head's output head converts its logits to fp32 before the temperature division to avoid fp16 subnormals at confident predictions.

---

## 6. Config

`Config/Experiment/StaticConfidenceHead/viode.yaml`:

```yaml
Odometry:
  frontend:
    type: StaticConfidence_FlowFormerCovFrontend
    args:
      weight: ./weights/flowformercov.pth
      device: cuda
      enc_dtype: bf16
      dec_dtype: fp16
      enforce_positive_disparity: true
      decoder_depth: 12
      imu:
        airimu_weights: ./weights/airimu_euroc.pth
        airimu_interval: 9
        sigma_repr: diag                # diag | chol
        feature_dim: 128
      dynamic_head:
        weight: null                    # optional pretrained head checkpoint
        hidden_dim: 128
        imu_fusion: film                # film | depth_bin_film | none
        depth_bins: [0.0, 5.0, 20.0, 1.0e6]
        refinement:
          enabled: true
          channels: 64
        proxy:
          enabled: true
          d_min: 0.5
          d_max: 80.0
          eps_z: 1.0e-6
          sigma_cap_rot: 0.5            # rad  — proxy invalid if trace(Σ_R) > this²
          sigma_cap_pos: 5.0            # m
          tau0: 1.0
          alpha: 1.0
        grad_clip: 0.01
        logit_clip: 10.0
        dt_reset_ms: 500

  # --- NEW: initialization ---
  init:
    method: drt_loose                   # drt_loose | drt_tight | identity
    drt_window_len: 10                  # number of keyframes used by DRT bootstrap
    drt_parallax_threshold_px: 20.0     # rejected if < this
    gyro_bias_solver:
      max_iterations: 200
      cauchy_delta: 1.0e-5
    gravity_magnitude: 9.81007          # EuRoC convention; override per-dataset
    accept_criteria:
      min_views: 6
      max_bg_norm: 0.05                 # rad/s
      max_ba_norm: 1.0                  # m/s²  — loose, since ba not estimated by DRT
      max_scale_mismatch: 0.1           # stereo-baseline check, §7.4

  # --- backend ---
  optimizer:
    type: SlidingWindow_VIO_PGO         # see §9
    args:
      window_size: 8                    # keyframes
      marginalize: true
      kernel: huber
      kernel_delta: 0.1
      solver: pinv
      vectorize: true

  args:
    device: cuda
    num_point: 200
    edgewidth: 32
    match_cov_default: 1.0
    profile: false
    mapping: true

    dynamic_gating:
      enabled: true
      min_c: 0.2

loss:
  dyn:    {enabled: true, weight: 1.0,  focal_gamma: 2.0,
           tau0: 1.0, alpha: 1.0, kappa: 2.0,
           d_min: 0.5, d_max: 80.0, tau_cyc_valid: 1.5}
  smooth: {enabled: true, weight: 0.05}

  # --- Factorized-head ablation (A_fact). Switched on by `dynamic_head.factorize: true`.
  # The head then emits (p_static, p_visible); the following terms replace `dyn`.
  # static:  {enabled: true, weight: 1.0,  focal_gamma: 2.0,
  #           tau0: 1.0, alpha: 1.0, kappa: 2.0,
  #           d_min: 0.5, d_max: 80.0, tau_cyc_valid: 1.5}
  # visible: {enabled: true, weight: 0.5,  tau_cyc: 1.5, kappa_cyc: 1.0}
  # joint:   {enabled: false, weight: 0.0}

train:
  window_len: 4
  optimizer: adamw
  lr: 3.0e-4
  weight_decay: 1.0e-4
  schedule: cosine
  total_steps: 50000
  grad_clip_norm: 1.0
  mixed_precision: bf16
  batch_size: 2
  num_workers: 4
```

Config schema additions live in each module's `is_valid_config`.

---

## 7. DRT-VIO-Init — stereo-adapted loosely-coupled bootstrap

This section pins the stereo adaptation of He et al. CVPR 2023. The reference implementation is at `references/drt-vio-init/src/initMethod/drtLooselyCoupled.cpp`. We re-implement the same algorithm in PyTorch for consistency with the rest of MAC-VO, using the stereo baseline to eliminate the monocular scale.

### 7.1 What DRT-loose gives us

From the first `N_init` keyframes with paired IMU windows, DRT-loose outputs:

| Quantity | Semantics | Used by |
|---|---|---|
| `R_k ∈ SO(3), k = 0..N_init-1` | body rotation in world | `FrameNode.pose` (after applying `T_BS`) |
| `p_k ∈ ℝ³` | body position in world | same |
| `v_k ∈ ℝ³` | body velocity in world | `FrameNode.vel` |
| `b_g ∈ ℝ³` | shared gyro bias | `FrameNode.bias_g` (broadcast to all init frames) |
| `b_a ∈ ℝ³ = 0` | DRT does not estimate it; we leave it at zero | `FrameNode.bias_a` |
| `g_W ∈ ℝ³` | gravity vector in world (unit `g_mag`) | stored as a system-level buffer; used by IMU residual |
| `s` | monocular scale | **ignored** on stereo; we use stereo baseline instead (see §7.4) |

### 7.2 Pipeline, stereo-adapted

```
for k = 0..N_init-1:
    depth_k, flow_k..k+1  = FlowFormerCov (frozen)
    kp_k  = same keypoint selector MAC-VO would use at step k
    kp_k+1 = propagate via flow
    tracks[pt_id] = { k → normalpoint(uv_k, K_k) }          # (normalized 3D ray)
    imu_meas[k→k+1] = preintegrate(corrected IMU between k and k+1, bias=(0,0))
```

Then run the DRT-loose algorithm exactly as in `drtLooselyCoupled.cpp`:

**Step 1 — Gyro bias.** Solve
```
min_{b_g}  Σ_{k, pt}  ρ( || (R_bc · ΔR̂(b_g)_{k→k+1} · R_bc^T)^T · x_k − x_{k+1} ||² )
```
using Cauchy loss (`δ = 1e-5`). This is the `BiasSolverCostFunctor` in the reference. We re-implement it in PyTorch using a differentiable preintegrator (the same one as everywhere else) and solve with `torch.optim.LBFGS` (`max_iter = 200`). `x_k` is the normalized bearing vector of each tracked feature at frame `k`.

**Step 2 — Reintegrate IMU with the solved `b_g`** (bias Jacobians are not needed here; `emit_jacobians = False`).

**Step 3 — Global rotation chain** (frame 0 at identity, body-in-world):
```
R_{0,C} = I                                           # first cam
R_{k,C} = R_{k-1,C} · (R_bc^T · ΔR̂_{k-1→k} · R_bc)   # cam-in-world
R_{k,B} = R_{k,C} · R_bc^T                            # body-in-world  (final output)
```

**Step 4 — Stereo position recovery (replaces DRT's LiGT).**
Monocular DRT solves an SVD for up-to-scale positions. With stereo, each keypoint already has a metric depth from the frozen stereo backbone, so we don't need LiGT. Instead, for every pair `(k, k+1)` we already have a metric reprojection-based PnP-style relative translation `t_rel_k` from MAC-VO's motion estimator (or we run one LM step of `Reproj_TwoFramePGO` with `c = 1` for all points — the dynamic head is not yet trained at bootstrap, so we cannot use it here). Chain them:
```
p_{0,C} = 0
p_{k,C} = p_{k-1,C} + R_{k-1,C} · t_rel_{k-1,k}
p_{k,B} = p_{k,C} − R_{k,B} · (R_bc · p_bc)      # body-in-world (account for extrinsic offset)
```

This is the stereo analogue of `evectors.middleRows<3>(3*i)` in `drtLooselyCoupled.cpp::process` — we already have the metric positions, we don't need the LiGT SVD.

**Step 5 — Velocity and gravity (`linearAlignment`, unchanged).**
The DRT linear system is
```
[I_3·dt    0      R_i^T · dt²/2 · I_3 · g_mag    R_i^T·(p_j-p_i)/100
 I_3      -R_i^T·R_j   R_i^T·dt · I_3 · g_mag    0                 ]  ·  [v_i; v_j; g; s]  =  [imu.ΔP + R_i^T·R_j·p_bc − p_bc;  imu.ΔV]
```
(rows 1 and 2 are the position and velocity residuals, respectively; see `linearAlignment()` in the reference). On stereo we **fix `s = 1`** (known scale) and solve for `(v_0..v_{N-1}, g)` only. The resulting system is overdetermined; solve with `lstsq`.

**Step 6 — Gravity refinement.** Apply the quartic polynomial refinement `gravityRefine` from the reference with the constraint `||g||² = g_mag²`. Even with stereo scale known, this refinement is worth keeping: it re-runs one LM step with the gravity magnitude hard-constrained, and on EuRoC/VIODE it improves initial gravity alignment by ~0.2° on average.

**Step 7 — Align to gravity-level world frame.** Rotate so that `g_W = [0, 0, -g_mag]` in NED (or the appropriate sign per MAC-VO's convention).

### 7.3 What to do with `b_a`

DRT-loose does **not** estimate `b_a`. We leave `b_a = 0` at bootstrap and let the backend estimate it online via the random-walk prior and the preintegration factors (§9.3). This is the same choice as VINS-Mono.

### 7.4 Accept / reject logic

After `initializer.run(frames, depths)` returns, reject the bootstrap if **any** of:

1. DRT internal failure (gyro bias did not converge, `gravityRefine` returned false, linear-alignment had singular `A`).
2. `||b_g||_2 > cfg.init.accept_criteria.max_bg_norm` (loose sanity).
3. Stereo-baseline cross-check: for each pair `(k, k+1)` compare `||p_{k+1,C} − p_{k,C}|| / ||DRT.imu.Δp + R·Δp_bc||_2` — ratio should be in `[1 − ε, 1 + ε]` with `ε = cfg.init.accept_criteria.max_scale_mismatch`. If off by more, something is wrong (likely time-sync or extrinsic calibration).
4. `len(tracks_with_≥3_views) < cfg.init.accept_criteria.min_views`.

On rejection: drop the oldest frame, add a new one on the next `run` call, retry. If DRT fails for more than `2 * N_init` keyframes, fall back to `method: identity` (current MAC-VO behavior) with a logged warning — the head will still run, but the backend IMU factors will be poorly conditioned for the first window.

### 7.5 `_write_init_to_map`

```python
def _write_init_to_map(self, init_out: DRTLooseOutput) -> None:
    self.graph.frames.clear()   # clear the scratch frame0 anchor
    for k, (R_B, p_B, v_B, frame) in enumerate(zip(init_out.rotation, init_out.position,
                                                    init_out.velocity, self._init_frames)):
        T_WB = pp.SE3(torch.cat([p_B, pp.SO3(pp.mat2SO3(R_B)).tensor()], dim=-1))
        T_BS = pp.SE3(frame.stereo.T_BS)
        T_WC = T_WB @ T_BS
        self.graph.frames.push(FrameNode.init({
            "pose":        T_WC.tensor(),                  # camera-in-world, MAC-VO convention
            "T_BS":        frame.stereo.T_BS,
            "need_interp": torch.tensor([0], dtype=torch.bool),
            "time_ns":     torch.tensor([frame.stereo.frame_ns], dtype=torch.long),
            "K":           frame.stereo.K,
            "baseline":    frame.stereo.baseline,
            "vel":         v_B.view(1, 3),                 # body velocity in world
            "bias_g":      init_out.bias_g.view(1, 3),
            "bias_a":      torch.zeros(1, 3),              # §7.3
        }))
    # Register IMU edges between consecutive init frames (§4.6).
    for k in range(len(self._init_frames) - 1):
        self._push_imu_edge(k, k + 1, init_out.imu_pres[k])

    self.g_W = init_out.gravity.clone()                   # world-frame gravity
    self.prev_keyframe = (self._init_frames[-1], len(self._init_frames) - 1,
                          self._init_depths[-1])
    Logger.write("info", f"DRT-loose init OK: N_init={len(self._init_frames)}, "
                         f"|b_g|={init_out.bias_g.norm():.4g} rad/s, "
                         f"|g|={init_out.gravity.norm():.4g} m/s²")
```

### 7.6 File layout

```
Module/Initialization/
├── __init__.py
└── DRTLoose/
    ├── __init__.py
    ├── drt_loose.py           # DRTLooseInitializer (orchestrator)
    ├── gyro_bias_solver.py    # LBFGS-based replacement for Ceres BiasSolverCostFunctor
    ├── linear_alignment.py    # the A x = b block, with s=1 for stereo
    ├── gravity_refine.py      # quartic polynomial root finder, port of gravityRefine
    └── tests/
        ├── test_vs_reference_cpp.py  # runs a fixed IMU+tracks test against known DRT output
        └── test_stereo_scale.py
```

---

## 8. Training data

### 8.1 Supported datasets

| Dataset | Stereo | IMU | GT pose | Notes |
|---|---|---|---|---|
| VIODE | ✓ | ✓ | body-frame, exact | Primary training set. |
| TartanAir 2 (`DataLoader/Dataset/TartanAir2.py`) | ✓ | optional | camera-frame, exact | Secondary; IMU window synthesized from GT if unavailable. |
| EuRoC (`DataLoader/Dataset/EuRoC.py`) | ✓ | ✓ | body-frame + `T_BS` | Held-out for sim-to-real eval; A8 row. |
| KITTI | partial IMU | — | poor — not used for training |
| TUM-VI | ✓ | ✓ | body-frame + `T_BS` | Optional held-out. |

### 8.2 Train / val / test splits

- **VIODE:** held-out sequence `City_Night_0` as validation, `City_Day_0` for calibration only (temperature fit), remaining for training.
- **EuRoC:** `MH_05_difficult` held-out for final A8 evaluation; never used for training or calibration.

### 8.3 Stereo-depth caching

The loss reads `depth_t` from the frozen stereo estimator. Cache to disk per sequence in a one-time preprocess (`Scripts/precompute_depths.py`) — ~2× training speedup and removes a memory-bandwidth bottleneck on dataloader workers. Invalidate on any change to the FlowFormerCov checkpoint path.

---

## 9. Preintegration factors in the two-frame PGO and sliding-window backend

This section closes the "IMU conditioning is open-loop" gap from the prior revision: the **same** preintegration factor that the head conditions on is now a hard optimization constraint in the backend.

### 9.1 State definition

Per keyframe `k`:

```
x_k = { R_k ∈ SO(3),   p_k ∈ ℝ³,   v_k ∈ ℝ³,   b_gk ∈ ℝ³,   b_ak ∈ ℝ³ }    (body, world)
```

`R_k, p_k` are stored as the body pose (not the camera pose that `VisualMap.frames.data["pose"]` holds). The backend reads body pose from `frames.data["pose"]` via `T_BS`, optimizes in body frame, and writes back in camera frame. The conversion is one multiply on each side.

### 9.2 Visual residual (weighted by `c`)

Unchanged from §4.8: `r_vis_i = c_i · (point2pixel_NED(T_opt⁻¹ · X_i, K) − u_i)`. The cost for observation `i` is `r_vis_iᵀ · Σ_i⁻¹ · r_vis_i = c_i² · r_iᵀ · Σ_i⁻¹ · r_i`.

### 9.3 IMU residual (Forster form, 15-dim)

For each consecutive keyframe pair `(i, j)` with a stored `IMUEdge` containing `(ΔR̂, Δv̂, Δp̂, Σ_imu, dt, J_*, bias_ref)`:

```
dbg = b_gi − b_g_ref
dba = b_ai − b_a_ref

# first-order bias-corrected factor (re-linearized cheaply at every LM step)
ΔR̂_corr = ΔR̂ · Exp(J_R_bg · dbg)
Δv̂_corr = Δv̂ + J_v_bg · dbg + J_v_ba · dba
Δp̂_corr = Δp̂ + J_p_bg · dbg + J_p_ba · dba

# residuals
r_R = Log( ΔR̂_corr^T · ( R_iᵀ · R_j ) )                                   (3-D, SO(3) tangent)
r_v = R_iᵀ · ( v_j − v_i − g_W · dt )  −  Δv̂_corr                         (3-D)
r_p = R_iᵀ · ( p_j − p_i − v_i · dt − 0.5 · g_W · dt² )  −  Δp̂_corr       (3-D)
r_bg = b_gj − b_gi                                                        (3-D, random walk)
r_ba = b_aj − b_ai                                                        (3-D, random walk)

r_imu = [r_R; r_v; r_p; r_bg; r_ba]                                       (15-D)
```

**Information matrix.** `Σ_imu ∈ ℝ^{9×9}` covers `[r_R; r_v; r_p]`. For the bias random walks, construct a diagonal `3×3` block each with variance `σ_bgw² · dt`, `σ_baw² · dt` — the continuous-time random walks from AirIMU's reported `acc_cov`, `gyro_cov` are integrated to discrete-time by `σ²_dt = σ²_continuous · dt`. Total `Σ_imu_full ∈ ℝ^{15×15}` is block-diagonal.

**Information matrix in the residual.** Cost contribution is `r_imuᵀ · Σ_imu_full⁻¹ · r_imu` — the usual Mahalanobis norm.

### 9.4 The two-frame PGO case (Stage A)

Per the migration plan, we start with the simplest usable backend: the existing two-frame PGO *plus* one preintegration factor between the two frames. This is implemented by a new subclass `Reproj_TwoFramePGO_IMU` of `Reproj_TwoFramePGO`:

```python
class Reproj_TwoFramePGO_IMU(Reproj_TwoFramePGO):
    """
    Two-frame PGO residual = weighted reprojection residual (inherited)
                           ⊕ 15-dim IMU preintegration residual (new)
    Optimizes: pose2opt (camera-in-world for the 2nd frame) + velocity + biases.
    Pose for frame 1 is held fixed (anchor).
    """
    def __init__(self, graph_data: GraphInput) -> None:
        super().__init__(graph_data)
        # New optimized params:
        self.v_j   = nn.Parameter(graph_data.init_v_j.double())         # [3]
        self.b_gj  = nn.Parameter(graph_data.init_bg_j.double())
        self.b_aj  = nn.Parameter(graph_data.init_ba_j.double())
        # Frame-i state is held fixed as buffers.
        self.register_buffer("R_i",   graph_data.R_i)       # [3, 3]
        self.register_buffer("p_i",   graph_data.p_i)       # [3]
        self.register_buffer("v_i",   graph_data.v_i)
        self.register_buffer("b_gi",  graph_data.b_gi)
        self.register_buffer("b_ai",  graph_data.b_ai)
        self.register_buffer("g_W",   graph_data.g_W)
        # IMU factor buffers:
        e = graph_data.imu_edge
        for name in ["delta_R", "delta_v", "delta_p", "Sigma", "dt",
                     "J_R_bg", "J_v_bg", "J_v_ba", "J_p_bg", "J_p_ba", "bias_ref"]:
            self.register_buffer(f"imu_{name}", getattr(e, name))

    def forward(self) -> torch.Tensor:
        # Visual residuals (inherited). Cast pose2opt (camera) to body frame.
        vis = super().forward()                                    # [N, 2], already c-weighted
        # IMU residuals.
        R_j = self.pose2opt.rotation().matrix() @ body2cam_R_inv()  # body rot
        p_j = self.pose2opt.translation() - R_j @ T_BS_translation
        r_imu = compute_imu_residual(                              # [15]
            R_i=self.R_i, p_i=self.p_i, v_i=self.v_i,
            b_gi=self.b_gi, b_ai=self.b_ai,
            R_j=R_j,        p_j=p_j,        v_j=self.v_j,
            b_gj=self.b_gj, b_aj=self.b_aj,
            g_W=self.g_W,
            delta_R=self.imu_delta_R, delta_v=self.imu_delta_v, delta_p=self.imu_delta_p,
            J_R_bg=self.imu_J_R_bg, J_v_bg=self.imu_J_v_bg, J_v_ba=self.imu_J_v_ba,
            J_p_bg=self.imu_J_p_bg, J_p_ba=self.imu_J_p_ba,
            bias_ref=self.imu_bias_ref, dt=self.imu_dt,
        )
        return torch.cat([vis.view(-1), r_imu], dim=0)             # [2N + 15]

    @torch.no_grad()
    def covariance_array(self) -> torch.Tensor:
        vis_cov = super().covariance_array()                       # [N, 2, 2]
        imu_cov = self.imu_Sigma_full_15()                         # [15, 15] diag-extended
        # Return a block-diag list the existing LM path can pinverse.
        return block_diag_list(vis_cov, imu_cov)
```

`Optimizer.py` at line 90 builds `weight = block_diag(*pinverse(covariance_array()))`. The extended `covariance_array()` returns a list such that `block_diag` produces the correct `(2N + 15) × (2N + 15)` weight matrix.

**`ICP_` and `ReprojDisp_` variants** are *not* extended with IMU at Stage A. Only `reproj` (the default) gets the IMU factor. This keeps the Stage-A patch small.

### 9.5 Sliding-window backend (Stage B, target)

`Module/Optimization/SlidingWindow/`:

```
SlidingWindow/
├── __init__.py
├── Optimizer.py          # SlidingWindow_VIO_PGO implementing IOptimizer
├── Graphs.py             # SWF_VIOFactorGraph (many frames, many IMU edges, many visual)
├── Marginalization.py    # Schur-complement marginalization of the oldest frame
└── tests/
```

**Graph layout, window of `W_kf` keyframes:**

- Parameters:
  - `pose_k ∈ SE(3)` for each keyframe in the window (camera-in-world, same format as MAC-VO's existing `pp.Parameter(pp.SE3(...))`).
  - `v_k ∈ ℝ³`, `b_gk ∈ ℝ³`, `b_ak ∈ ℝ³`, one per keyframe.
- Factors:
  - Visual factors per observation, same form as `Reproj_TwoFramePGO` (c-weighted).
  - IMU factors for each consecutive keyframe pair, same form as §9.3.
  - A **marginalization prior** on the oldest keyframe that summarizes all evicted information (§9.6).

**Marginalization.** Same Schur-complement trick as VINS-Mono: when a keyframe `k_old` is evicted, compute the information matrix for the remaining state conditional on `k_old` having been fixed at its current estimate. Store the resulting (linear) prior and include it as an extra residual in subsequent optimizations. Implementation is a straightforward Schur-complement of the stacked Jacobian; ~80 lines of numpy/torch.

**Window size.** Default `W_kf = 8` keyframes. Parameter count per window:
- 8 × (6 pose + 3 vel + 3 b_g + 3 b_a) = 120 parameters.
- Roughly 8 × 200 ≈ 1600 visual factors, each 2-D residual.
- 7 IMU factors, each 15-D residual.
- 1 marginalization prior (whatever dimension is left).

LM on 120 parameters with ~3200 residuals is sub-50-ms on CPU via pypose.

### 9.6 Migration stages (match these to ablations A14–A16)

| Stage | What's in the backend | Blocking prerequisites |
|---|---|---|
| **A** | 2-frame PGO *plus* one IMU factor between the two frames | DRT-loose init + head + preintegrator |
| **B** | Sliding-window PGO with `W_kf` keyframes, IMU factors between consecutive keyframes, marginalization | Stage A |
| **C** | Adaptive IMU covariance inflation driven by visual-IMU innovation | Stage B |

Stage A is small (~400 lines diff) and is the minimum to claim the main paper contribution. Stage B is the target. Stage C is a stretch goal.

---

## 10. Ablation matrix (complete)

| ID | Config delta from baseline yaml | Tests |
|---|---|---|
| A0 | `Odometry.args.dynamic_gating.enabled: false`; `Odometry.optimizer.type: TwoFrame_PGO` | Pure MAC-VO baseline ATE/RPE |
| A1 | `loss.smooth.enabled: false` | Is the GT-residual term alone enough? |
| A2 | `loss.smooth.weight: 0.01` | Smoothness strength sensitivity |
| A3 | baseline yaml | Full loss stack |
| A4 | `dynamic_head.imu_fusion: none` | IMU value on motion-varying sequences |
| A5 | `dynamic_head.refinement.enabled: false` | H/4 refinement contribution |
| A6 | `dynamic_gating.min_c: 0.99` | Hard-mask-only (approximates hard-threshold baseline) |
| A7 | `dynamic_gating.min_c: 0.0` | Soft-only |
| A8 | Train on VIODE, eval on EuRoC | Dataset-generalization |
| A9 | `dynamic_head.imu_fusion: depth_bin_film` | Spatially-adaptive FiLM vs global FiLM |
| A10 | H/4 refiner runs but recurrence also lives at H/4 (extra 450k params) | Test whether recurrence is needed at H/4 |
| A11 | `dynamic_head.proxy.enabled: false` | Value of the per-pixel IMU proxy |
| A12 | `imu_fusion: film` but with `depth_bins=[0, ∞)` (= global FiLM) | Sanity check of A9 |
| A_fact | `dynamic_head.factorize: true` (head emits `p_static, p_visible`; `loss.static/visible/joint` enabled; optional `joint.weight: 0.2`) | Does factorizing the head into rigidity × visibility help over a single head? |
| A14 | `optimizer.type: TwoFrame_PGO` (no IMU backend factor) | Value of the IMU backend factor |
| A15 | `optimizer.type: Reproj_TwoFramePGO_IMU` with `W_kf = 2` | Pose-only vs full state in 2-frame |
| A16 | `optimizer.type: SlidingWindow_VIO_PGO, W_kf: 8` | Sliding-window vs 2-frame |
| A17 | `init.method: identity` (DRT off, backend turned on) | Value of DRT init for backend convergence |

Every row is a config flip; no code path changes.

---

## 11. File plan — new and modified

### 11.1 New files

```
MAC-VO/
├── Module/Network/DynamicHead/
│   ├── __init__.py
│   ├── convgru.py           # ConvGRUCell with spectral-norm convs
│   ├── film.py              # FiLM, DepthBinFiLM
│   ├── refiner.py           # RefinerH4 (zero-init residual)
│   ├── head.py              # StaticConfidenceHead, HeadOut dataclass
│   └── README.md
├── Module/Network/AirIMU/
│   ├── __init__.py
│   ├── corrector.py         # AirIMU CodeNet port (+ load_from_ckpt)
│   ├── preintegration.py    # DifferentiablePreintegrator (+ bias Jacobians)
│   ├── encoder.py           # IMUEncoder
│   └── proxy.py             # build_imu_proxy() + rigid_flow_cam()
├── Module/Initialization/
│   ├── __init__.py
│   └── DRTLoose/
│       ├── __init__.py
│       ├── drt_loose.py
│       ├── gyro_bias_solver.py
│       ├── linear_alignment.py
│       ├── gravity_refine.py
│       └── tests/
├── Module/Optimization/SlidingWindow/
│   ├── __init__.py
│   ├── Optimizer.py         # SlidingWindow_VIO_PGO
│   ├── Graphs.py            # SWF_VIOFactorGraph, Reproj_TwoFramePGO_IMU
│   ├── Marginalization.py
│   └── tests/
├── DataLoader/Dataset/DynamicHeadTrain.py
├── Train/DynamicHead/
│   ├── __init__.py
│   ├── train.py
│   ├── loop.py
│   ├── loss.py
│   ├── calibrate.py
│   ├── checks.py
│   └── tests/
├── Scripts/precompute_depths.py
└── Config/Experiment/StaticConfidenceHead/
    ├── viode.yaml
    ├── viode_ablation_A4_no_imu.yaml
    ├── viode_ablation_A5_no_refiner.yaml
    ├── ...
    └── euroc_eval.yaml
```

### 11.2 Modified files

```
MAC-VO/
├── Module/Frontend/Frontend.py
│     + class StaticConfidence_FlowFormerCovFrontend(FlowFormerCovFrontend)
│
├── Module/Network/FlowFormerCov/flownet.py
│     + self.last_context = context  (one-line stash, §4.5)
│
├── Module/Map/VisualMap.py
│     + FrameStore.data["vel", "bias_g", "bias_a"]
│     + MatchStore.data["c"]
│     + new IMUEdgeStore with 12 fields (§4.6)
│
├── Module/Map/Template.py
│     + extend FrameFeature / MatchingFeature Literals
│     + add IMUEdgeFeature Literal
│
├── Odometry/MACVO.py
│     + staged bootstrap via DRTLooseInitializer (§4.7a)
│     + sample c at keypoints + Stage-1 hard mask (§4.7b)
│     + bias write-back hook after Optimizer.write_map (§4.7c)
│     + is_valid_config additions
│
├── Module/Optimization/TwoFramePGO/Graphs.py
│     + Reproj_TwoFramePGO: register w buffer from obs.data["c"], forward returns w*residual
│     + Analytic_Reproj_TwoFramePGO.build_jacobian returns J * w (broadcast per-row)
│     + ReprojDisp variants in parallel
│     + ICP variants raise NotImplementedError when dynamic_gating.enabled and graph_type != reproj
│     + new class Reproj_TwoFramePGO_IMU (§9.4)
│
├── Module/Optimization/TwoFramePGO/Optimizer.py
│     + graph_type "reproj_imu" path to instantiate Reproj_TwoFramePGO_IMU
│     + GraphInput extended with optional IMU edge + frame-i state + g_W
│
└── Utility/Math.py
      + body2cam_se3()   (10-line pypose helper)
      + rotate_by_SO3_batch()
```

### 11.3 Explicitly **not** touched

- `Module/Network/FlowFormerCov/covhead.py` — hard freeze, no edits.
- `Module/Network/FlowFormer/core/*` — hard freeze, no edits.
- Existing `FrontendCompose` and `CUDAGraph_FlowFormerCovFrontend` — the new subclass is opt-in via config; existing configs keep working as-is.
- `Config/Experiment/Baseline/*`, `Config/Experiment/MACVO/*` — existing experiments are unchanged; new experiments live in a new `Config/Experiment/StaticConfidenceHead/` directory.

---

## 12. Deployment path (runtime)

1. **Load checkpoints.** FlowFormerCov ←  `flowformercov.pth`; AirIMU ← `airimu_<dataset>.pth`; StaticConfidenceHead ← `dynamic_head_<dataset>.pth` (if present). Temperature buffers come with the head checkpoint.
2. **Bootstrap.** The first `N_init` `run()` calls stage frames; DRT-loose runs once when the stage is full. On success, `VisualMap.frames[0..N_init-1]` is populated with `(pose, vel, bias_g, bias_a=0)` and IMU edges are registered.
3. **Per-frame-pair loop** (what `MACVO.run_pair` does):
   - `StaticConfidence_FlowFormerCovFrontend.estimate_pair` runs: frozen stereo+flow, frozen AirIMU corrector+preintegrator, per-pixel proxy, trainable head forward. Hidden state carried.
   - `MACVO.run_pair` samples `c` at keypoints, applies Stage-1 mask, constructs `MatchObs` (now with `c`).
   - A new `IMUEdge` is pushed connecting the previous keyframe to the current one.
   - `SlidingWindow_VIO_PGO.start_optimize` runs LM on the current window (visual factors weighted by `c`, IMU factors from §9.3, marginalization prior).
   - On `write_map`, the optimizer writes back `(pose, vel, bias_g, bias_a)` to `VisualMap.frames`. The bias write-back hook (§4.7c) propagates the new bias to the frontend preintegrator so the next frame pair linearizes at the fresh bias.
4. **Failure modes:**
   - **IMU dropout.** The frontend emits `imu_fusion: none` fallback (visual-only head). Backend IMU factor is omitted for the gap; the marginalization prior absorbs it.
   - **Long dt gap.** `dt_reset_ms` triggers hidden-state reset in the head; backend re-bootstraps via DRT-loose on the next `N_init` frames.
   - **Head NaN/all-zero output.** Watchdog in `MACVO.run_pair`: if `kp_c.min() > 0.95` or `kp_c.max() < 0.05` or any NaN, reset head hidden state and fall back to `kp_c = 1.0` for the current pair. Log a warning.

### 12.1 Compute budget (GPU, 640×480)

| Step | Approx cost |
|---|---|
| FlowFormerCov stereo+flow (frozen, same as baseline MAC-VO) | 15–25 ms |
| AirIMU corrector + DifferentiablePreintegrator (one pair, N_imu ~ 100) | 1.5 ms |
| IMU-rigid proxy assembly (full-res) | 0.5 ms |
| Head forward (H/8 recurrence + H/4 refinement) | 6–10 ms |
| Keypoint sampling + Stage-1 mask | < 0.1 ms |
| Two-frame-PGO + IMU factor (Stage A) | 2–3 ms |
| Sliding-window PGO, `W_kf = 8` (Stage B) | 30–50 ms |

Total (Stage A): ~25–40 ms/frame additional on top of the baseline's ~20–30 ms/frame. Stage B adds ~30 ms of backend.

---

## 13. Open implementation items (flagged, not deferred)

These are intentionally left for the implementation plan because they depend on measurements, not design decisions:

1. **Analytic Jacobian for the IMU residual block (§9.3).** We run autograd at Stage A. At Stage B, for sub-50 ms budget at `W_kf = 8`, we will likely need analytic Jacobians of the 15-D IMU residual wrt `(R_i, p_i, v_i, b_gi, b_ai, R_j, p_j, v_j, b_gj, b_aj)`. The formulas are Forster eqs. 40–44 and are well-known; the implementation effort is ~200 lines.
2. **Where to run the DRT gyro-bias Ceres replacement.** We specified PyTorch LBFGS. On the critical path only at startup, so this is not performance-sensitive, but convergence of LBFGS on a Cauchy-lossed non-convex objective deserves a unit test with known-good inputs from the reference C++ implementation (`test_vs_reference_cpp.py`).
3. **First-time FlowFormerCov stash overhead.** Stashing `last_context` on the module changes the memory cap by ~1 MB per forward. Not a concern on GPU but is a concern under memory profilers that allocate per call — verify in the training loop that no memory leak appears over 10k iters.
4. **Windowed loader sharding for multi-GPU.** The `SequenceWindowSampler` must respect sequence boundaries per rank; straightforward but one of those things that is wrong in the first attempt.
5. **Backend gravity parameterization.** We optimize `g_W` on `S²` (two-parameter) starting from DRT's estimate, not the full 3-vector, to avoid scale drift.

None of these change the design; they are implementation details that deserve a test but do not justify deferring more design work.

---

## 14. Out of scope

- Retraining or finetuning FlowFormerCov. Hard frozen, by design.
- End-to-end joint training of the head with PGO in the loop. A future direction; the current design produces a well-defined loss gradient without needing this.
- Online test-time adaptation of the head on live sequences. The cycle-consistency signal is the foundation, but the actual TTA loop is explicitly future work.
- Any change to the legacy V2.5 `dynamask_vio/` codebase, which is not in this tree.

---

## 15. Cross-references

- **Method and intuition:** `docs/2026-04-11-static-confidence-head-method-theory.md` (companion document).
- **DRT-VIO-Init paper:** He, Xu, Ouyang, Li, *"A Rotation-Translation-Decoupled Solution for Robust and Efficient Visual-Inertial Initialization"*, CVPR 2023. Local copy at `references/He_A_Rotation-Translation-Decoupled_Solution_..._CVPR_2023_paper.pdf`. Reference code at `references/drt-vio-init/`.
- **AirIMU paper:** Qiu, Wang, et al., arXiv:2310.04874. Local copy at `references/2409.09479v2.pdf`. Reference code at `references/AirIMU/`.
- **Forster preintegration:** Forster, Carlone, Dellaert, Scaramuzza, *"On-Manifold Preintegration for Real-Time Visual-Inertial Odometry"*, T-RO 2017.
- **MAC-VO:** `MAC-VO/README.md` and the Paper_Reproduce.yaml config.
