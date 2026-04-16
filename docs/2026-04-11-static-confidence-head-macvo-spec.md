# Static Confidence Head for MAC-VO (frozen backbone + AirIMU)

**Date:** 2026-04-11
**Target codebase:** `dynamask_vio/references/MAC-VO/` (treated as primary for this spec)
**Status:** Design approved, ready for implementation planning
**Relation to prior docs:**
- Supersedes `docs/superpowers/specs/2026-04-11-flowformer-macvo-frozen-dynamic-head-design.md`
  (which proposed the same idea under a "dynamicness" naming convention; this doc pins
  the Framing A "static confidence" convention, the exact PGO residual-gating rule,
  and the config-driven three-term loss).
- Supersedes the high-level notes in `MACVO_DYNAMICNESS_COVARIANCE_SUPERVISION.md`.

---

## 1. Goal

Freeze MAC-VO's FlowFormerCov flow+covariance backbone and add one trainable module — a
per-pixel **static confidence head** `c : image → [0,1]^{H×W}` with `c=1` meaning
"definitely static, use this observation fully" and `c=0` meaning "definitely dynamic,
drop this observation". The head is temporally recurrent, conditioned on AirIMU
features, and its output gates MAC-VO's two-frame PGO by multiplying the whitened
residual of each reprojection factor.

All changes in this spec live under `dynamask_vio/references/MAC-VO/` unless a path is
prefixed with `dynamask_vio/` explicitly.

---

## 2. Frozen vs trainable

| Module | Source | Trainable |
|---|---|---|
| FlowFormer context encoder / memory encoder / memory decoder | `Module/Network/FlowFormerCov/` | No |
| Flow covariance branch (`MemoryCovDecoder`) | `Module/Network/FlowFormerCov/covhead.py` | No |
| Stereo depth (`FlowFormerCovFrontend.estimate_depth`) | `Module/Frontend/Frontend.py` | No |
| **AirIMU learned corrector** (bias + per-sample noise model) | new `Module/Network/AirIMU/corrector.py` | No (frozen) |
| **Differentiable IMU preintegrator** (Forster-style, SO(3)×R⁶, with covariance propagation and bias Jacobians) | new `Module/Network/AirIMU/preintegration.py` | No — non-parametric |
| **IMU FeatureMLP** (projects the preintegration factor to `f_imu`) | new `Module/Network/AirIMU/encoder.py` | **Yes** (small, trained with the head) |
| **StaticConfidenceHead** (ConvGRU + FiLM + Conv output) | new `Module/Network/DynamicHead/` | **Yes** |
| Temperature calibration scalar | stored as buffer in the head module | Calibrated post-training |

**The IMU branch is three stages, not one.** Splitting them matters for freeze
policy, for training semantics, and for sharing with the backend (§14.10):

1. *Corrector* is a learned network — frozen, loaded from an AirIMU checkpoint.
2. *Preintegrator* has no parameters — it's a differentiable numerical integrator
   that runs a Forster-style recursion with first-order covariance propagation. It
   produces the **preintegration factor** `(ΔR̂, Δv̂, Δp̂, Σ_imu)` and (for backend
   sharing) the bias Jacobians.
3. *FeatureMLP* is a small trainable projector from the factor to the head's
   conditioning vector `f_imu`. It is trained jointly with the head, *not* frozen.

No gradients flow from the head's loss back into the corrector. The preintegrator
has no parameters to train. Only the FeatureMLP and head are updated.

**Hard freeze is enforced two ways:** `requires_grad_(False)` on construction, and all
frozen forwards run inside `torch.no_grad()`/`torch.inference_mode()` blocks with their
outputs detached before entering the head. No gradient path exists from the head's loss
back into FlowFormerCov or AirIMU.

---

## 3. Architecture

### 3.1 Data flow per frame pair `(t, t+1)`

```
┌──────── frozen ──────────────────────────────────────────────────────┐
│                                                                       │
│  (stereo_t, stereo_{t+1}) ─▶ FlowFormerCovFrontend.estimate_pair      │
│                               ├─▶ depth map at t+1                    │
│                               └─▶ flow_{t→t+1}, cov_{t→t+1}           │
│                                                                       │
│  Also exposed (new): context features f_ctx from context_encoder      │
│                      at decoder resolution (H/8, W/8), 128 channels   │
│                                                                       │
│  (imu_window_t)                                                       │
│     └─► AirIMUCorrector ─► (â, ω̂, σ²_a, σ²_g, Δb_a, Δb_g)              │
│                          └─► DifferentiablePreintegrator               │
│                              ├─► ΔR̂ ∈ SO(3), Δv̂, Δp̂ ∈ R³              │
│                              ├─► Σ_imu ∈ R^{9×9}                       │
│                              └─► dt (window length)                    │
│                                 │                                     │
│                                 ├─► FeatureMLP([log ΔR̂, Δv̂, Δp̂,       │
│                                 │               diag Σ_imu, Δb_*, dt]) │
│                                 │   └─► f_imu ∈ R^{128}                │
│                                 │                                     │
│                                 └─► rigid_flow(D, K, ΔR̂, Δp̂)           │
│                                     └─► Δf, r_imu, r_imu_norm, valid  │
│                                         (per-pixel proxy channels)    │
│                                                                       │
└───────────────────────────────────────────────────────────────────────┘
             │
             ▼
┌──────── trainable ───────────────────────────────────────────────────┐
│                                                                       │
│  StaticConfidenceHead(                                                │
│    f_ctx    : [B, 128, H', W'],                                       │
│    flow     : [B, 2,   H', W'],  (bilinear-downsampled if needed)     │
│    cov      : [B, 3,   H', W'],  (matches flow resolution)            │
│    f_imu    : [B, D_imu],        (projection of preintegration factor)│
│    proxy    : [B, 4,   H', W'],  (Δf_x, Δf_y, r_imu_norm, valid)      │
│    h_prev   : [B, 128, H', W']   (ConvGRU hidden state, None = zeros) │
│  ) ─▶ logit_lowres, h_new                                             │
│                                                                       │
│  logit_full = F.interpolate(logit_lowres, (H, W))                     │
│  c_full     = sigmoid(logit_full / T_calib)                           │
│                                                                       │
└───────────────────────────────────────────────────────────────────────┘
             │
             ▼
   Sampled at keypoint locations in MACVO.run_pair  ─▶  PGO gating
```

### 3.2 StaticConfidenceHead internals

```
input stack (per frame):
    # Visual channels + per-pixel IMU-rigid proxy (§3.1 proxy block):
    #   Δf_x, Δf_y     : f_obs − f_rigid_imu  (2 ch)
    #   r_imu_norm     : ||Δf|| / τ(D)        (1 ch)
    #   valid          : inside-image ∧ valid-depth mask (1 ch)
    x = concat(f_ctx, flow, cov, proxy)         # [B, 128+2+3+4, H', W']
    x = Conv1x1(x, out=128) → GroupNorm → SiLU  # [B, 128, H', W']
    x = FiLM(x, f_imu)                          # per-channel (γ, β) from f_imu
                                                # f_imu encodes the *global* preint
                                                # factor (log ΔR̂, Δv̂, Δp̂, diag Σ_imu,
                                                # Δb_g, Δb_a, dt) — see §4.2.

temporal recurrence (ConvGRU cell):
    z = σ(Conv3x3(cat(h_prev, x)))              # update gate
    r = σ(Conv3x3(cat(h_prev, x)))              # reset gate
    q = tanh(Conv3x3(cat(r * h_prev, x)))
    h = (1 - z) * h_prev + z * q                # [B, 128, H', W']

output head:
    y = Conv3x3(h, 64)  → SiLU
    y = Conv1x1(y, 1)                           # [B, 1, H', W']
    y = GradientClip(clip=0.01)                 # reused pattern from score_head.py
    y = clamp(y, -10, 10)                       # fp16-safe
    logit_lowres = y

calibration:
    self.register_buffer("T_calib", torch.tensor(1.0))  # set by Train/DynamicHead/calibrate.py
    c_lowres  = sigmoid(logit_lowres / T_calib.clamp(min=1e-3))
    c_full    = F.interpolate(c_lowres, (H, W), mode='bilinear', align_corners=False)

`T_calib` is a buffer (not a `Parameter`) — it is fit post-hoc on a held-out split after
the head has finished training, using the standard ECE-minimizing sweep in
`Train/DynamicHead/calibrate.py`. This matches the existing
`temperature_calib` buffer convention in `dynamask_vio/models/dynamask.py`.
```

**Parameter count:** ~0.6M (dominated by the three ConvGRU 3×3 convs).

**Alternatives selectable via config (`model.dynamic_head.recurrence`):**
- `convgru` (default) — as above.
- `none` — drop the ConvGRU, pass `x` straight to the output head (ablation baseline).
- `convlstm` — drop-in replacement for longer-horizon memory.

**IMU fusion selectable via config (`model.dynamic_head.imu_fusion`):**
- `film` (default) — per-channel modulation.
- `concat` — tile `f_imu` to `[B, D_imu, H', W']` and concat to `x` before the Conv1x1.
- `none` — disable the IMU branch entirely; visual-only ablation.

### 3.3 Streaming semantics

- **Training:** BPTT over a short window (default `window_len = 4`). `h_0` initialized
  to zeros at the start of each window; no cross-window hidden-state carry.
- **Inference:** `h_t` is carried across frames as instance state on the new
  `StaticConfidence_FlowFormerCovFrontend` (see §4.1). Reset on explicit sequence
  boundary or when `dt > dt_reset_threshold` (detected from IMU timestamps).

---

## 4. MAC-VO code changes

### 4.1 `Module/Network/DynamicHead/` (NEW)

```
Module/Network/DynamicHead/
├── __init__.py               # exports StaticConfidenceHead, ConvGRUCell, build_head
├── convgru.py                # ConvGRUCell (plus ConvLSTMCell as alternate)
├── film.py                   # FiLM layer ported from dynamask_vio/models/film.py
├── head.py                   # StaticConfidenceHead module
└── README.md                 # 1-paragraph description + config contract
```

`StaticConfidenceHead.forward` signature:

```python
def forward(
    self,
    f_ctx: torch.Tensor,     # [B, 128, H', W'] — frozen FlowFormer context features
    flow:  torch.Tensor,     # [B, 2,   H', W']
    cov:   torch.Tensor,     # [B, 3,   H', W']  or [B, 2, H', W'] if diagonal-only
    f_imu: torch.Tensor | None,    # [B, D_imu] — global preint-factor projection
                                   # (None iff imu_fusion == "none")
    proxy: torch.Tensor | None,    # [B, 4, H', W'] — (Δf_x, Δf_y, r_imu_norm, valid)
                                   # (None iff imu_proxy == "off")
    h_prev: torch.Tensor | None,   # [B, 128, H', W'] or None
) -> tuple[torch.Tensor, torch.Tensor]:
    """Returns (logit_lowres, h_new). No sigmoid applied here."""
```

`f_imu` encodes the *global* IMU context (one vector per frame pair);
`proxy` encodes the *per-pixel* IMU context (geometric discrepancy at each pixel).
Both are derived from the same preintegration factor `(ΔR̂, Δv̂, Δp̂, Σ_imu)`.

### 4.2 `Module/Network/AirIMU/` (NEW — three co-equal modules)

The IMU branch is implemented as three modules with different freeze semantics.
Mis-grouping them (e.g. "freeze all of AirIMU") is a known source of bugs.

| File | Role | Parameters | Freeze |
|---|---|---|---|
| `corrector.py`      | AirIMU learned bias + noise-variance predictor | Yes (~few M) | **Frozen** (loaded from AirIMU checkpoint) |
| `preintegration.py` | Differentiable Forster preintegrator on SO(3)×R⁶ with 1st-order covariance propagation. Emits `(ΔR̂, Δv̂, Δp̂, Σ_imu)` and bias Jacobians `(J_R_bg, J_v_bg, J_v_ba, J_p_bg, J_p_ba)`. | None (non-parametric) | N/A |
| `encoder.py`        | `IMUEncoder` = corrector ∘ preintegrator ∘ `FeatureMLP`. Wraps the two above and adds a small trainable MLP that projects `[log ΔR̂, Δv̂, Δp̂, diag Σ_imu, Δb_g, Δb_a, dt]` → `f_imu ∈ R^{128}`. | FeatureMLP only (~10–20k) | MLP trainable, upstream frozen |

`IMUEncoder.forward` returns the whole preintegration dict, not just `f_imu`:

```python
{
    "f_imu":        [B, 128],         # FeatureMLP output (for FiLM / concat)
    "delta_R":      [B, 3, 3],        # ΔR̂  — used for f_rigid_imu proxy and backend
    "delta_v":      [B, 3],           # Δv̂
    "delta_p":      [B, 3],           # Δp̂  — used for f_rigid_imu proxy and backend
    "Sigma_preint": [B, 9, 9],        # Σ_imu
    "delta_bg":     [B, 3],           # AirIMU bias corrections
    "delta_ba":     [B, 3],
    # Backend-only (added when share_with_backend=True):
    "J_R_bg":       [B, 3, 3],
    "J_v_bg":       [B, 3, 3],
    "J_v_ba":       [B, 3, 3],
    "J_p_bg":       [B, 3, 3],
    "J_p_ba":       [B, 3, 3],
}
```

Consumers:

- The head's **FiLM/concat fusion** reads `f_imu`.
- The head's **per-pixel proxy input** reads `(delta_R, delta_p)` and computes
  `f_rigid_imu`, `Δf`, `r_imu`, `r_imu_norm`, `valid` (see §3.1 trainable block).
- The **backend IMU factor** (§14.10) reads
  `(delta_R, delta_v, delta_p, Sigma_preint, J_*)` from the *same* call — never
  re-implement preintegration backend-side.

The AirIMU checkpoint path is read from config. The corrector is set to
`eval()` and `requires_grad_(False)` on construction. The preintegrator is
non-parametric. The FeatureMLP has its own initialization and is trained jointly
with the static-confidence head.

### 4.3 `Module/Frontend/Frontend.py`

**Add a new subclass alongside `FlowFormerCovFrontend`:**

```python
class StaticConfidence_FlowFormerCovFrontend(FlowFormerCovFrontend):
    """
    FlowFormerCov frontend that also runs a trainable static-confidence head
    reading frozen decoder features and an AirIMU feature vector.
    """

    def __init__(self, config: SimpleNamespace):
        super().__init__(config)

        # Load + freeze AirIMU
        from ..Network.AirIMU.encoder import AirIMUEncoder
        self.imu_encoder = AirIMUEncoder.from_config(config.imu)
        self.imu_encoder.eval()
        for p in self.imu_encoder.parameters():
            p.requires_grad_(False)

        # Build + (optionally) load trainable dynamic head
        from ..Network.DynamicHead.head import StaticConfidenceHead
        self.dynamic_head = StaticConfidenceHead.from_config(config.dynamic_head)
        if config.dynamic_head.weight:  # optional pretrained checkpoint
            ckpt = torch.load(config.dynamic_head.weight, map_location=config.device)
            self.dynamic_head.load_state_dict(ckpt["head_state_dict"])

        self.dynamic_head.to(config.device)

        # Streaming hidden state (inference) and last-t timestamp for reset logic
        self._h_prev: torch.Tensor | None = None
        self._last_frame_ns: int | None = None
        self._dt_reset_ns: int = int(config.dynamic_head.get("dt_reset_ms", 500)) * 1_000_000

    def reset_stream(self) -> None:
        self._h_prev = None
        self._last_frame_ns = None

    def _maybe_reset(self, frame_ns: int) -> None:
        if self._last_frame_ns is None:
            return
        if frame_ns - self._last_frame_ns > self._dt_reset_ns:
            self.reset_stream()

    def estimate_pair(self, frame_t1, frame_t2):
        depth_out, match_out = super().estimate_pair(frame_t1, frame_t2)

        # Tap frozen decoder features.
        # NOTE: context_encoder is called inside FlowFormerCov.forward and not cached,
        # so we either (a) expose f_ctx via a forward hook on self.model.context_encoder,
        # or (b) patch FlowFormerCov to stash the last context output on itself.
        # Choosing (b): see §4.4 for the one-line patch.
        f_ctx = self.model.last_context.detach()   # [B, 128, H', W']

        # Resolve resolution of flow/cov to match f_ctx (downsample by factor 8).
        H, W = match_out.flow.shape[-2:]
        H8, W8 = f_ctx.shape[-2:]
        flow_lo = F.interpolate(match_out.flow, (H8, W8), mode="bilinear") / 8.0
        cov_lo  = F.interpolate(match_out.cov,  (H8, W8), mode="bilinear")

        # IMU branch: corrector → preintegrator → FeatureMLP.
        # IMPORTANT: keep the full preint dict. f_imu is only one of its consumers;
        # (ΔR̂, Δp̂) are re-used below to build per-pixel proxy channels, and Σ_imu
        # is what `f_imu` carries as a global trust signal.
        imu_window = frame_t2.imu_window.to(self.config.device)
        imu_mask   = frame_t2.imu_mask.to(self.config.device)
        with torch.no_grad():
            imu_out = self.imu_encoder(imu_window, imu_mask)
            # imu_out = {"f_imu", "delta_R", "delta_v", "delta_p",
            #            "Sigma_preint", "delta_bg", "delta_ba"}
        f_imu   = imu_out["f_imu"]
        delta_R = imu_out["delta_R"]     # [B, 3, 3]
        delta_p = imu_out["delta_p"]     # [B, 3]

        # Per-pixel IMU-rigid proxy channels (train/test symmetric geometric cue).
        # See §3.3 / theory §5.5 for why this is load-bearing.
        f_rigid_imu = rigid_flow(depth_out.depth, self.K, delta_R, delta_p)  # [B,2,H,W]
        delta_f     = match_out.flow - f_rigid_imu                            # [B,2,H,W]
        r_imu       = torch.linalg.vector_norm(delta_f, dim=1, keepdim=True)  # [B,1,H,W]
        r_imu_norm  = r_imu / tau_depth(depth_out.depth, delta_p)              # [B,1,H,W]
        valid       = compute_proxy_validity(depth_out.depth, match_out.flow) # [B,1,H,W]
        proxy_full  = torch.cat([delta_f, r_imu_norm, valid], dim=1)          # [B,4,H,W]
        proxy_lo    = F.interpolate(proxy_full, (H8, W8), mode="bilinear")
        proxy_lo[:, :2] = proxy_lo[:, :2] / 8.0  # flow-like channels scale w/ res

        # Inference reset logic.
        self._maybe_reset(int(frame_t2.stereo.frame_ns))

        # Head forward (trainable — no torch.no_grad here).
        logit_lo, h_new = self.dynamic_head(
            f_ctx, flow_lo, cov_lo, f_imu, proxy_lo, self._h_prev
        )
        self._h_prev = h_new.detach()   # detach between frames at inference
        self._last_frame_ns = int(frame_t2.stereo.frame_ns)

        c_lo  = torch.sigmoid(logit_lo / self.dynamic_head.T_calib)
        c_full = F.interpolate(c_lo, (H, W), mode="bilinear", align_corners=False)

        # Attach to match_out so downstream (MACVO.run_pair) can retrieve it.
        match_out.static_conf = c_full.squeeze(1)   # [B, H, W]

        return depth_out, match_out

    @classmethod
    def is_valid_config(cls, config: SimpleNamespace | None) -> None:
        super().is_valid_config(config)
        # additional required fields
        cls._enforce_config_spec(config, {
            "imu":          lambda v: v is not None,
            "dynamic_head": lambda v: v is not None,
        })
```

**Why a subclass, not a new class:** every existing config path, CUDAGraph variant,
and test exercises `FlowFormerCovFrontend`'s `estimate_pair` contract. Subclassing
keeps all of that working unchanged — the new class is opt-in by config.

### 4.4 `Module/Network/FlowFormerCov/flownet.py` — one-line context tap

The cleanest way to expose the context feature map without touching the decoder is
a one-line stash inside `FlowFormerCov.forward`:

```python
def forward(self, image1, image2):
    image1 = ((2 * image1) - 1.0).to(dtype=self.enc_dtype)
    image2 = ((2 * image2) - 1.0).to(dtype=self.enc_dtype)

    with torch.cuda.nvtx.range("Context Encoder"):
        context = self.context_encoder(image1)
        self.last_context = context  # ← NEW: stash for DynamicHead consumers
    ...
```

The existing `estimate_pair` batches two pairs into one forward (`[t2-stereo, t1-t2-mono]`),
so `self.last_context` ends up with batch-dim 2. `estimate_pair` in
`StaticConfidence_FlowFormerCovFrontend` picks slice `[1:2]` (the temporal pair). That
slicing rule is documented in the new class's docstring.

### 4.5 `Module/Map/VisualMap.py` — extend `MatchStore` schema

Add one field to `self.match`'s data dict (line 54 area):

```python
"static_conf": AutoScalingTensor((self.init_size, 1), grow_on=0, dtype=torch.float32),
```

Default value on insert is `1.0` so legacy code that doesn't populate it behaves as
"all-static" (gate becomes a no-op).

### 4.6 `Odometry/MACVO.py` — sample confidence at keypoints

In `run_pair`, after `kp0_uv`/`kp1_uv` are computed and before `MatchObs.init` (line 246
area), sample `static_conf` at the keypoint locations and attach it to the observation:

```python
# NEW: sample static confidence at keypoint locations
if hasattr(match01, "static_conf") and match01.static_conf is not None:
    # match01.static_conf: [B, H, W]  (B=1 in this path)
    kp_static = self.Frontend.retrieve_pixels(
        kp0_uv, match01.static_conf.unsqueeze(1)
    ).squeeze(0)   # [N, 1]
else:
    kp_static = torch.ones((num_kp, 1), device=self.device)

# Stage-1 hard mask: drop observations below threshold BEFORE building MatchObs.
if self.dynamic_gating_enabled:
    keep = (kp_static.squeeze(-1) > self.min_static_conf)
    kp0_uv, kp1_uv = kp0_uv[keep], kp1_uv[keep]
    # also slice all kp*_d, kp*_disparity, kp*_sigma_*, pos0_Tc, pos0_covTc, pos1_covTc,
    # kp_static, num_kp accordingly
    ...
```

`MatchObs.init(...)` gains one more key:

```python
"static_conf": kp_static.cpu(),
```

**New config fields read from `MACVO.__init__`:**
```python
self.dynamic_gating_enabled = bool(dynamic_gating.get("enabled", False))
self.min_static_conf        = float(dynamic_gating.get("min_static_conf", 0.2))
```

Added to the `is_valid_config` schema in `args`.

### 4.7 `Module/Optimization/TwoFramePGO/Graphs.py` — weighted residual

**`Reproj_TwoFramePGO`:** register the per-observation weight and multiply the
residual by it.

```python
class Reproj_TwoFramePGO(FactorGraph):
    def __init__(self, graph_data: GraphInput) -> None:
        super().__init__()
        ...
        # NEW: static confidence weight per observation
        w = self.obs.data.get("static_conf", None)
        if w is None:
            w = torch.ones((self.kp2.shape[0], 1), dtype=torch.float32)
        self.register_buffer("w", w.reshape(-1, 1))

    def forward(self) -> torch.Tensor:
        self.pos_Tc = self.pose2opt.Inv().Act(self.pos_Tw)
        residual = point2pixel_NED(self.pos_Tc, self.K) - self.kp2     # [N, 2]
        return self.w * residual                                        # ← NEW
```

This is mathematically equivalent to scaling the information matrix of factor `i` by
`w_i²` — exactly the "multiply the dynamicness probability directly to the whitened
residuals" semantics requested.

**`Analytic_Reproj_TwoFramePGO`:** same change to its `forward()` and an update to its
analytic Jacobian (multiply the Jacobian's residual-side by `self.w` too).

**`ICP_TwoframePGO`, `ReprojDisp_TwoFramePGO`** (+ their analytic variants):
the default config uses `graph_type: reproj`, so the other two are patched in the same
pattern **only when used**. Put an explicit `NotImplementedError` in them otherwise, to
avoid silently running ungated ICP factors during an ablation.

**`covariance_array()` is NOT modified** — all gating happens at the residual level per
the explicit design decision.

**Zero-weight factors:** Stage-1 hard-mask filter in `MACVO.run_pair` (§4.6) normally
prevents `w_i = 0` observations from reaching the graph. However, ablation A7 runs
with `min_static_conf = 0.0`, and a sufficiently confident head could still emit a
very-near-zero `c` at a kept observation. To keep pypose's `pinverse(cov)` path well-
conditioned regardless of gating config, the factor graph applies a safety floor:

```python
self.register_buffer("w", w.reshape(-1, 1).clamp(min=self.W_EPS))   # W_EPS = 1e-3
```

`W_EPS` is a class constant, not config-exposed — it exists only to prevent numerical
failure and should never be the dominant effect on any factor.

### 4.8 `DataLoader/` — carry IMU + GT pose

Two changes — a dataset (training only) and a frame type (shared with inference).

**Frame type:** make sure `StereoInertialFrame` (or whatever frame type the
`StaticConfidence_FlowFormerCovFrontend` consumes) carries `imu_window` and `imu_mask`
fields. On MAC-VO's inference path, `StereoFrame` may need to be bumped to
`StereoInertialFrame` in the config. At inference the dataset just serves the IMU
window between successive image timestamps — there's no GT requirement.

**Training dataset (new, under `DataLoader/Dataset/`):** `DynamicHeadTrainDataset` wraps
an existing MAC-VO stereo-inertial sequence source and additionally serves:

- `gt_R_rel`, `gt_t_rel` — relative SE(3) pose from cam at t to cam at t+1 (camera
  frame, not IMU body frame).
- `valid_depth_mask` — computed from the frozen stereo depth on the fly (can be cached
  to disk per sequence in a one-time preprocess).

For VIODE and TartanAir this is straightforward since both ship GT body-frame poses
plus the `T_BS` extrinsic — `gt_R_rel`, `gt_t_rel` are computed as:

```
T_cam_w_t  = T_body_w_t · T_BS
T_cam_w_t1 = T_body_w_t1 · T_BS
T_rel_cam  = T_cam_w_t.inv() · T_cam_w_t1
```

**No segmentation-based proxy mask is read.** This dataset works identically on any
sequence with stereo + IMU + GT poses.

### 4.9 `Train/DynamicHead/` (NEW) — training entry point

```
Train/DynamicHead/
├── __init__.py
├── train.py                  # main entry, argparse → config → training loop
├── loop.py                   # BPTT windowed loop
├── loss.py                   # dyn + smooth + cyc (see §5)
└── calibrate.py              # post-hoc temperature fit on held-out split
```

**Why not reuse `Train/MatchingNet/train_flowformer.py`:** that script trains the
FlowFormerCov covariance branch end-to-end; it expects gradients into the backbone. Our
training loop explicitly runs the backbone in `torch.no_grad()` and only backpropagates
through the head + temperature. Keeping it in its own file avoids cross-contamination.

**Training loop shape:**

```python
for batch in loader:     # batch is a window of window_len frames
    h = None
    losses = []
    for t in range(window_len):
        with torch.no_grad():
            depth_t, match_t = frontend_frozen.estimate_pair(batch.stereo[t-1], batch.stereo[t])
            f_ctx = frontend_frozen.model.last_context[1:2].detach()
            f_imu = imu_encoder(batch.imu_window[t], batch.imu_mask[t])["f_imu"]

        logit_lo, h = head(f_ctx, flow_lo_t, cov_lo_t, f_imu, h)
        c_full = sigmoid(F.interpolate(logit_lo, (H, W)) / head.T_calib)

        L_t = compose_total_loss(
            c_pred=c_full,
            flow=match_t.flow,
            depth=depth_t.depth,
            R_gt=batch.gt_R_rel[t],
            t_gt=batch.gt_t_rel[t],
            image=batch.stereo[t].imageL,
            cfg=cfg.loss,
        )
        losses.append(L_t)

    (sum(losses) / window_len).backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
```

---

## 5. Loss

Three terms, each independently toggled by `cfg.loss.<name>.enabled`.

### 5.1 Shared preamble

```python
# Rigid flow from GT pose + stereo depth.
X_c    = backproject(pixels, depth, K)            # [B, 3, H, W]
X_c_t  = R_gt @ X_c + t_gt
pix_t  = project(X_c_t, K)
f_rigid_gt = pix_t - pixels                       # [B, 2, H, W]

# Residual magnitude.
r = ||f_obs - f_rigid_gt||_2                      # [B, 1, H, W]

# Depth-adaptive threshold (effectively a 3D-velocity threshold).
tau_D = tau0 + alpha * (f / clamp(depth, eps)) * ||t_gt||

# Validity masks.
M_infov = 1[pix_t inside image bounds]
M_depth = 1[depth > d_min ∧ depth < d_max]
M_cycle = 1[||f_fwd + warp(f_bwd, f_fwd)||_2 < tau_cyc_valid]
M_valid = M_infov ∧ M_depth ∧ M_cycle

# Soft target: static confidence (Framing A).
d_soft  = sigmoid((r - tau_D) / kappa)
c_target = 1 - d_soft
```

Backward flow `f_bwd` is computed via a second frozen FlowFormerCov call with
`(image2, image1)` order. This is a cached `torch.no_grad()` pass; cost is ~1.5× a
single frame step.

### 5.2 Term 1 — GT-residual focal BCE (primary)

```python
if cfg.loss.dyn.enabled:
    L_dyn = focal_bce(c_pred, c_target, gamma=cfg.loss.dyn.focal_gamma)
    L_dyn = (L_dyn * M_valid).sum() / M_valid.sum().clamp(min=1)
```

Focal BCE (default `gamma=2.0`) because VIODE/TartanAir scenes are static-heavy; plain
BCE collapses to "predict static everywhere".

### 5.3 Term 2 — Edge-aware smoothness

```python
if cfg.loss.smooth.enabled:
    grad_c_x = |c_pred[..., :, 1:] - c_pred[..., :, :-1]|
    grad_I_x = |I[..., :, 1:] - I[..., :, :-1]|.mean(dim=1, keepdim=True)
    L_smooth_x = grad_c_x * exp(-grad_I_x)
    L_smooth_y = ... (same on the y axis)
    L_smooth = (L_smooth_x.mean() + L_smooth_y.mean()) / 2
```

### 5.4 Term 3 — Forward-backward cycle-consistency pseudo-label

Dataset-agnostic — no GT pose, no segmentation. Extra cost: second FlowFormerCov
forward pass for backward flow.

```python
if cfg.loss.cyc.enabled:
    warp_err = ||f_fwd + warp(f_bwd, f_fwd)||_2            # [B, 1, H, W]
    c_cyc_target = 1 - sigmoid((warp_err - tau_cyc_bce) / kappa_cyc)
    M_cyc_valid  = M_infov ∧ M_depth    # NOT M_cycle — that's what we're labeling
    L_cyc = bce(c_pred, c_cyc_target)
    L_cyc = (L_cyc * M_cyc_valid).sum() / M_cyc_valid.sum().clamp(min=1)
```

### 5.5 Total loss

```python
L = 0.
if cfg.loss.dyn.enabled:    L += cfg.loss.dyn.weight    * L_dyn
if cfg.loss.smooth.enabled: L += cfg.loss.smooth.weight * L_smooth
if cfg.loss.cyc.enabled:    L += cfg.loss.cyc.weight    * L_cyc
```

### 5.6 Deliberately not added

- Proxy segmentation BCE — no dataset outside synthetic VIODE/TartanAir ships per-pixel
  dynamic/static labels; any loss depending on them silently breaks on real data.
- Entropy regularization on `c_pred` — ConvGRU hidden state + focal BCE already prevent
  the degenerate `c = 0.5` collapse in practice.
- Explicit temporal-consistency L1 — the ConvGRU hidden state enforces this implicitly
  and more softly than a hard penalty.
- Laplace NLL over residual — `c` is a probability, not an uncertainty; BCE is correct.
- Photometric reprojection loss — redundant with FlowFormerCov's frozen flow signal.

---

## 6. Config

New experiment config at `Config/Experiment/StaticConfidenceHead/viode.yaml`:

```yaml
model:
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
        airimu_weights: ./weights/airimu.pth
        airimu_interval: 9
      dynamic_head:
        weight: null               # optional pretrained head checkpoint
        hidden_dim: 128
        recurrence: convgru        # convgru | none | convlstm
        imu_fusion: film           # film | concat | none
        imu_feature_dim: 128
        grad_clip: 0.01
        logit_clip: 10.0
        dt_reset_ms: 500

loss:
  dyn:
    enabled: true
    weight: 1.0
    focal_gamma: 2.0
    tau0: 1.0                      # px
    alpha: 1.0                     # depth-adaptive scaling factor
    kappa: 2.0                     # sigmoid sharpness
    d_min: 0.5
    d_max: 80.0
    tau_cyc_valid: 1.5             # px — for M_valid masking
  smooth:
    enabled: true
    weight: 0.05
  cyc:
    enabled: true                  # approved (B) — on by default
    weight: 0.3
    tau_cyc_bce: 1.5               # px — for L_cyc soft label
    kappa_cyc: 1.0

train:
  window_len: 4
  optimizer: adamw
  lr: 3.0e-4
  weight_decay: 1.0e-4
  schedule: cosine
  total_steps: 50000
  grad_clip_norm: 1.0
  mixed_precision: bf16            # backbone is bf16; head in fp32

odometry:
  dynamic_gating:
    enabled: true
    min_static_conf: 0.2           # Stage-1 hard mask threshold
```

### 6.1 Config schema additions

- `Module/Frontend/Frontend.py` — `StaticConfidence_FlowFormerCovFrontend.is_valid_config`
  adds `imu`, `dynamic_head` to the required spec dict.
- `Odometry/MACVO.py` — `is_valid_config` adds optional `dynamic_gating` sub-object
  under `args`.
- `Module/Optimization/TwoFramePGO/Optimizer.py` — no change. PGO gating is a dataset /
  map-level concern; the factor graph reads `static_conf` from `MatchObs` without any
  new config of its own.

---

## 7. Training data

### 7.1 Datasets

- **Primary:** VIODE (synthetic stereo + IMU + GT poses).
- **Secondary:** TartanAir 2 (the MAC-VO-native one under
  `DataLoader/Dataset/TartanAir2.py`), which already has stereo + GT poses; IMU support
  either already exists or is a thin extension.
- **Held-out eval:** EuRoC (via `DataLoader/Dataset/EuRoC.py`) — real data, stereo +
  IMU + GT poses, used for the dataset-generalization ablation (A8 below).

### 7.2 Windowed sampler

`Train/DynamicHead/loop.py` implements a `SequenceWindowSampler` that draws
non-overlapping (or stride-configured) windows of length `window_len` from each
sequence. The hidden state is reset at window boundaries during training.

---

## 8. Deployment path

At inference, nothing about the head's input changes between train and deploy, and GT
pose never enters the forward pass:

```
per frame ──▶  (imu window)  ─▶  frozen AirIMU  ─▶  f_imu
           \
            ▶  (stereo pair) ─▶  frozen FlowFormerCov  ─▶  f_ctx, flow, cov, depth
                                            │
                                            ▼
                          StaticConfidenceHead (ConvGRU streaming)
                                            │
                                            ▼
                                      c_full (H×W)
                                            │
                                            ▼
                      MACVO.run_pair samples at keypoints
                                            │
                                            ▼
                   Stage-1: drop observations with c < min_static_conf
                                            │
                                            ▼
                   Stage-2: multiply reprojection residual by c (Graphs.py)
                                            │
                                            ▼
                                   pose estimate
```

**What substitutes for GT pose at deploy:** AirIMU's per-frame-pair feature — that's
the whole reason the IMU branch is load-bearing.

**Streaming failure modes:**
- **Lost IMU** → `imu_fusion: none` fallback config runs a visual-only head as a
  degraded backup.
- **Long dt gap** → `dt_reset_ms` triggers a hidden-state reset.
- **Cold start** → first ~5 frames the head is underconfident; Stage-2's soft gating
  handles it naturally (confidence values near `min_static_conf` contribute linearly
  rather than being dropped entirely).

---

## 9. Ablation matrix

| ID | Config delta from the baseline yaml | Purpose |
|----|--------------------------------------|---------|
| A0 | `odometry.dynamic_gating.enabled: false` | MAC-VO baseline ATE/RPE |
| A1 | `loss.smooth.enabled: false; loss.cyc.enabled: false` | Is GT-residual alone enough? |
| A2 | `loss.cyc.enabled: false` | Does smoothness help? |
| A3 | (baseline yaml as-is) | Full loss stack |
| A4 | `model.frontend.args.dynamic_head.imu_fusion: none` | Is IMU load-bearing? |
| A5 | `model.frontend.args.dynamic_head.recurrence: none` | Is recurrence load-bearing? |
| A6 | `odometry.dynamic_gating.min_static_conf: 0.99` | Hard-only (approximates hard-mask baseline) |
| A7 | `odometry.dynamic_gating.min_static_conf: 0.0` | Soft-only (no hard mask) |
| A8 | Train on VIODE, eval on EuRoC | Dataset generalization |

Every one is a config flip; no code changes between rows.

---

## 10. File plan

### 10.1 New files

```
dynamask_vio/references/MAC-VO/
├── Module/Network/DynamicHead/
│   ├── __init__.py
│   ├── convgru.py
│   ├── film.py
│   ├── head.py
│   └── README.md
├── Module/Network/AirIMU/
│   ├── __init__.py
│   ├── corrector.py         # ported from dynamask_vio/models/airimu_corrector.py
│   ├── preintegration.py    # ported from dynamask_vio/models/preintegration.py
│   └── encoder.py           # ported from dynamask_vio/models/imu_encoder.py
├── DataLoader/Dataset/DynamicHeadTrain.py     # training dataset wrapper
├── Train/DynamicHead/
│   ├── __init__.py
│   ├── train.py
│   ├── loop.py
│   ├── loss.py
│   └── calibrate.py
└── Config/Experiment/StaticConfidenceHead/
    └── viode.yaml
```

### 10.2 Modified files

```
dynamask_vio/references/MAC-VO/
├── Module/Frontend/Frontend.py
│     + class StaticConfidence_FlowFormerCovFrontend(FlowFormerCovFrontend)
│
├── Module/Network/FlowFormerCov/flownet.py
│     + one-line context stash (self.last_context = context)
│
├── Module/Map/VisualMap.py
│     + "static_conf" field in MatchStore (AutoScalingTensor)
│
├── Odometry/MACVO.py
│     + sample static_conf at keypoints in run_pair
│     + Stage-1 hard mask filter
│     + self.dynamic_gating_enabled, self.min_static_conf (from cfg.args)
│     + is_valid_config schema update
│
└── Module/Optimization/TwoFramePGO/Graphs.py
      + register w buffer from graph_data.observations.data["static_conf"]
      + forward() returns self.w * residual
      + same in Analytic_Reproj_TwoFramePGO (with Jacobian scaling)
      + NotImplementedError-or-scaled in ICP/Disp variants
```

### 10.3 NOT touched

- `Module/Network/FlowFormerCov/covhead.py` — hard freeze, zero edits.
- `Module/Network/FlowFormer/core/*` — hard freeze, zero edits.
- `Module/Optimization/TwoFramePGO/Optimizer.py` — uses factor graphs through the
  `FactorGraph` interface; weighted residuals propagate naturally through
  `covariance_array()` pinverse.
- `dynamask_vio/models/dynamask.py` and the existing DynaMask V2.5 training path —
  retained for historical comparison against the new approach, never edited by this
  spec.

---

## 11. Hard-freeze verification checklist

Before starting training, the training entry point asserts every guarantee the design
relies on:

1. `for p in frontend.model.parameters(): assert not p.requires_grad`
2. `for p in frontend.imu_encoder.parameters(): assert not p.requires_grad`
3. A smoke-test training step asserts `frontend.model.parameters()` have `None`
   gradients after `backward()`.
4. `frontend.model.training` is `False` throughout training.
5. Every forward through `frontend.model` is inside `torch.no_grad()` (enforced by
   `FlowFormerCovFrontend.estimate_pair`'s `@torch.inference_mode()` decorator, which
   the new subclass inherits).

If any assertion fails, the training script exits before consuming GPU time.

---

## 12. Open items

These are intentionally deferred from this spec and will be decided during
implementation planning:

1. **Where exactly to cache stereo depth for training.** The loss uses `depth_t` at
   training time, which costs a FlowFormerCov forward pass. Options: (a) recompute on
   the fly inside the training loop (simple, costs ~1× extra forward per window step),
   (b) cache to disk per sequence in a one-time preprocess (complex, but ~2× training
   speedup). Decision deferred to the implementation plan.
2. **GT pose frame alignment.** VIODE ships IMU body-frame poses plus `T_BS`; TartanAir
   ships camera-frame poses directly. The training dataset layer needs one branch per
   dataset source to produce `gt_R_rel`, `gt_t_rel` in the camera frame. Implementation
   plan will audit each dataset.
3. **`Analytic_Reproj_TwoFramePGO` Jacobian scaling.** The analytic Jacobian is hand-
   derived; multiplying residual by `w` requires updating the analytic formula, not
   just the forward. Plan step will verify the gradient against autodiff on a small
   test case.
4. **ICP / Disp variants of the gating.** Default experiment uses `graph_type: reproj`.
   The ICP and Disp variants get gated only if an ablation requires them; until then
   they raise `NotImplementedError` if `dynamic_gating_enabled and graph_type != 'reproj'`
   so the error is loud.

---

## 13. Out of scope for this spec

- Retraining or finetuning FlowFormerCov (hard freeze is a hard constraint per user).
- Replacing MAC-VO's motion model, keyframe selector, or outlier filter.
- End-to-end joint training of the head with PGO in the loop.
- Online / test-time adaptation (the cycle-consistency loss is the foundation, but TTA
  loop is future work).
- Any change to `dynamask_vio/models/dynamask.py` or the V2.5 training stack.

---

## 14. V2 architectural refinements (added)

This section extends the approved design with five targeted upgrades. They preserve the
hard-freeze constraint and keep train/deploy symmetry.

### 14.1 Factorized confidence head: `p_static` and `p_visible`

The head now predicts two probabilities:

- `p_static` — probability that the pixel belongs to rigid static world geometry.
- `p_visible` — probability that the correspondence is reliable/trackable
  (not occluded/disoccluded, not cycle-inconsistent).

The effective confidence passed downstream is:

```python
c_eff = p_static * p_visible
```

Interpretation:

- dynamic but visible pixel: low `p_static`, high `p_visible` -> low `c_eff`
- static but untrackable pixel: high `p_static`, low `p_visible` -> low `c_eff`
- static and trackable pixel: high/high -> high `c_eff`

This separates "moving" from "unreliable correspondence" instead of forcing one scalar
to represent both.

### 14.2 H/4 refinement branch for small/thin movers

Keep ConvGRU temporal modeling at `H/8` (unchanged), then add a lightweight spatial
refiner at `H/4`:

```text
H/8 branch: ConvGRU -> logits_static_h8, logits_visible_h8
upsample x2 to H/4
concat with H/4 visual cues (image gradients / downsampled flow-cov / optional skip)
depthwise-separable 3x3 + 1x1 -> delta logits at H/4
H/4 logits = coarse logits + delta logits
upsample to full res
```

Goal: improve boundary precision and thin-structure handling while keeping most compute
at low resolution.

### 14.3 IMU-rigid residual proxy input — (now core; see §3.1 / §3.2)

Originally described here as a V2 refinement. The per-pixel proxy
(`Δf_x, Δf_y, r_imu_norm, valid`) computed from the preintegrator's `(ΔR̂, Δp̂)` and
frozen depth is now a **core input** of the head, not an addendum. It is listed in
the primary data-flow diagram (§3.1) and in the head's concat stack (§3.2), and
is assembled in `estimate_pair` (§4.1).

The rationale for promoting it is in theory doc §5.5: without this proxy, the head's
forward-pass input distribution is not actually symmetric across train and test,
because the geometric discrepancy signal exists only in the training loss. With it,
the train and test forward passes see the same types of geometric cues (produced by
the same operator on IMU-derived pose at test, GT pose at training).

### 14.4 Spatially adaptive IMU fusion (replacing global-only FiLM)

Default fusion becomes depth-bin FiLM (configurable), not only global FiLM:

```text
1. Partition pixels into soft near/mid/far bins from depth.
2. Predict (gamma, beta) per bin from f_imu.
3. Apply bin-weighted FiLM to feature map:
   x' = sum_b mask_b * (gamma_b * x + beta_b)
```

Alternative (`imu_fusion: cross_attention`): tiny cross-attention from spatial visual
tokens to an IMU token bank for ablation.

Rationale: ego-motion effect is spatially depth-dependent; a single global modulation is
often too coarse.

### 14.5 Solver-aware confidence mapper `w = g(c_eff)`

Instead of hard-coding `w = c_eff`, learn a monotonic mapper:

```python
w = g(c_eff)                 # monotonic increasing
w = w.clamp(min=W_EPS, max=1.0)
```

Constraints:

- monotonicity enforced by positive increments / isotonic parameterization
- endpoint anchors: `g(0) ~= 0`, `g(1) ~= 1`
- initialized as identity so training starts from previous behavior

`w` (not raw `c_eff`) is what scales reprojection residuals in PGO.

### 14.6 Loss update for factorized outputs

Replace single-output losses with:

```python
L_static  = focal_bce(p_static,  c_static_target)     # GT-rigid residual supervision
L_visible = bce(p_visible, c_visible_target)          # cycle/occlusion supervision
L_joint   = bce(c_eff, c_joint_target)                # optional weak coupling term
L_smooth  = edge_aware_smoothness(c_eff)
```

Where:

- `c_static_target` is from GT rigid residual (existing `L_dyn` target logic)
- `c_visible_target` is from cycle consistency / occlusion validity
- `c_joint_target` defaults to `c_static_target * c_visible_target` when enabled

Total:

```python
L = λs*L_static + λv*L_visible + λj*L_joint + λsm*L_smooth
```

### 14.7 Config additions

```yaml
model:
  frontend:
    args:
      dynamic_head:
        outputs: [static, visible]
        refinement:
          enabled: true
          res: h4
          width: 64
        proxy_input:
          enabled: true
          channels: [delta_f, r_imu, r_imu_norm, valid]
        imu_fusion: depth_bin_film      # depth_bin_film | film | cross_attention | none
        depth_bins: [0.0, 5.0, 20.0, 1e6]
        weight_mapper:
          enabled: true
          type: monotonic_piecewise
          num_knots: 8
          w_eps: 1.0e-3

loss:
  static:   {enabled: true, weight: 1.0, focal_gamma: 2.0}
  visible:  {enabled: true, weight: 0.5}
  joint:    {enabled: false, weight: 0.2}
  smooth:   {enabled: true, weight: 0.05}
```

### 14.8 Integration rule in PGO (unchanged location, new value)

`MACVO.run_pair` stores `c_eff` and mapped `w` at keypoints. Stage-1 hard mask uses
`c_eff` threshold (`min_static_conf`), Stage-2 factor weighting uses `w = g(c_eff)`.

### 14.9 Ablation extensions

Add the following rows to the matrix:

- A9: dual-head off (`p_visible := 1`) -> measures value of factorization
- A10: H/4 refinement off -> boundary/thin-object contribution
- A11: proxy input off -> value of IMU-rigid discrepancy cue
- A12: global FiLM vs depth-bin FiLM -> spatially adaptive fusion gain
- A13: identity mapper (`g(c)=c`) vs learned monotonic mapper -> solver-aware weighting gain

### 14.10 Backend upgrade: full IMU preintegration factors (VINS-style)

To close the loop on IMU reliability, extend backend optimization from pose-only to full
VIO state and include preintegration factors directly in the graph.

Per-keyframe state:

```text
x_k = {R_k, p_k, v_k, b_gk, b_ak}
```

where:

- `R_k, p_k`: orientation and position of keyframe `k`
- `v_k`: velocity in world frame
- `b_gk, b_ak`: gyroscope and accelerometer biases

For each consecutive keyframe pair `(i, j)`, preintegrate IMU samples over `[t_i, t_j]`
using the **same** `DifferentiablePreintegrator` instance that the frontend uses
(§4.2). Do not ship a second preintegrator for the backend — numerical drift between
two implementations is a known silent failure mode. The outputs are:

```text
delta_R_hat, delta_v_hat, delta_p_hat, dt        # factor mean
J_R_bg, J_v_bg, J_v_ba, J_p_bg, J_p_ba           # bias Jacobians
Sigma_imu                                         # 9×9 (extend to 15×15 when
                                                  #  bias random-walk blocks are
                                                  #  included)
```

When `share_with_backend=True`, the frontend's call already returns the Jacobians,
so the backend just consumes them. When used in frontend-only mode, the Jacobians
can be skipped for a small compute saving.

Residual block (plain-text form for implementation):

```text
dbg = b_gi - b_g_ref
dba = b_ai - b_a_ref

r_R = Log( (delta_R_hat * Exp(J_R_bg * dbg))^-1 * (R_i^-1 * R_j) )
r_v = R_i^-1 * (v_j - v_i - g*dt)
      - (delta_v_hat + J_v_bg*dbg + J_v_ba*dba)
r_p = R_i^-1 * (p_j - p_i - v_i*dt - 0.5*g*dt^2)
      - (delta_p_hat + J_p_bg*dbg + J_p_ba*dba)
r_bg = b_gj - b_gi
r_ba = b_aj - b_ai

r_imu = [r_R, r_v, r_p, r_bg, r_ba]   # 15D
```

Joint objective:

```text
L = sum visual factors (weighted by w = g(c_eff))
  + sum imu preintegration factors
  + priors (first pose/vel/bias, optional gravity prior)
```

This turns IMU from open-loop conditioning into a closed-loop backend constraint.

### 14.11 Concrete integration path in MAC-VO

1. **Map state extension (`Module/Map/VisualMap.py`)**
   - Add frame fields: `vel_w` `[N,3]`, `bias_g` `[N,3]`, `bias_a` `[N,3]`.
2. **IMU factor storage**
   - Add an IMU edge store linking `(frame_i, frame_j)` with preintegration payload
     (`delta_*`, Jacobians, covariance, `dt`).
3. **Graph input/output**
   - Extend optimization `GraphInput` to carry windowed frame states + IMU factors.
4. **Optimization graph**
   - Keep current visual factors (`Reproj_*`) and add `IMUPreintFactor`.
5. **Optimizer**
   - Add sliding-window optimizer variant (5-10 keyframes), not only two-frame pose.
6. **Odometry wiring (`Odometry/MACVO.py`)**
   - On keyframe creation, preintegrate IMU between previous and current keyframe and
     push one IMU factor.
7. **Robustness**
   - Inflate IMU covariance when visual-IMU innovation spikes or time-sync quality drops.

### 14.12 Migration stages

Stage plan to reduce risk:

1. **Stage A (minimal):** pose-only IMU relative factor in backend (no bias/velocity state).
2. **Stage B (target):** full state `{R,p,v,bg,ba}` + full 15D preintegration factor.
3. **Stage C (production):** adaptive IMU covariance inflation + failure monitoring.

### 14.13 Additional ablations for IMU-backend integration

- A14: visual-only backend vs visual + pose-only IMU factor
- A15: visual + pose-only IMU vs visual + full preintegration factor
- A16: full preintegration with fixed IMU covariance vs adaptive covariance inflation
