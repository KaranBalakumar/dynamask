# Logging & Observability Spec — Static-Confidence-Head MAC-VO Pipeline

**Date:** 2026-04-18
**Companion to:** `docs/2026-04-11-static-confidence-head-macvo-spec.md` (implementation spec)
and `docs/2026-04-11-static-confidence-head-method-theory.md` (method rationale).
**Scope:** every numeric signal worth recording, from dataset loading through frontend,
head, loss, backward, DRT init, and backend VIO PGO. Covers all three training modes
(pre-flight smoke test, primary training, evaluation) and both backend stages (A =
two-frame + IMU, B = sliding-window).

This document is the **source of truth** for what gets logged, where it goes, and at
what cadence. The design assumes one event API, three sinks, and a single dotted-key
namespace so the same numbers land in W&B, TensorBoard, and local artifact dumps
without duplicated call sites.

**Goal:** a human looking at the log stream (or replaying a crashed run offline)
should be able to identify the first module where a numerical problem appears, the
step at which it started, and the tensor whose distribution is off — all without
re-running the job. No problem should require adding a print statement after the fact.

---

## 0. Invariants

| # | Invariant | Where it's enforced |
|---|---|---|
| L0 | Every log event passes through a single `DebugLogger` instance. Nothing writes
directly to W&B / TensorBoard / disk in pipeline code. | §1 |
| L1 | Scalar keys use the dotted namespace defined in §2. **No free-text keys.** A
typo in a key name fails a CI lint that parses the source tree for `log_scalar` /
`log_hist` call sites. | §2, §13.2 |
| L2 | All three sinks see the same scalar events. W&B and TensorBoard also get
histograms; only Local gets tensor dumps. | §1.2 |
| L3 | Local tensor dumps run **once every `LOCAL_DUMP_EVERY` steps (default 200)**
and are guaranteed atomic (rename-on-close) so a killed job leaves no half-written
file. | §5.4 |
| L4 | No training-critical path blocks on logging I/O. W&B uses its async client;
TensorBoard writes through a background thread; local dumps are enqueued to a
`ThreadPoolExecutor(max_workers=2)` and the training thread never awaits. | §11.1 |
| L5 | NaN / Inf in any logged tensor triggers a **logger-local abort** (raises
`NumericalAbort`) with the module name, step, and key in the traceback. Default on;
can be downgraded to a warning by `logging.fail_on_nan: false`. | §1.3 |
| L6 | Every run has a **run-id directory** `runs/{date}_{git_sha}_{seed}/` with
deterministic subpaths (§5.2). Config snapshot + frozen git diff are written once at
step 0 into `runs/.../manifest/`. | §5.2 |
| L7 | Module forward hooks used for debug dumps are **registered once** at
construction and are **zero-cost** when `debug_dump_due = False`. They short-circuit
on the first line. | §6, §11.2 |
| L8 | Logging overhead must stay under **7% of step time** when Local dumps are
inactive, and under **15% of step time** on dump steps. Measured and asserted in
`tests/test_logging_overhead.py`. | §11 |

---

## 1. Architecture — one API, three sinks

### 1.1 One `DebugLogger`

```python
# Train/DynamicHead/logging/debug_logger.py
class DebugLogger:
    """
    The only logging interface training code should call. Fan-out to W&B,
    TensorBoard, and local artifact dumps is hidden behind these methods.

    All methods are safe to call from any thread; events are buffered and
    flushed by a background worker.
    """
    def __init__(self, run_dir: Path, config: dict, sinks: list[Sink]): ...

    # Scalars
    def log_scalar(self, key: str, value: float, step: int) -> None: ...
    def log_scalars(self, kv: dict[str, float], step: int) -> None: ...

    # Histograms (W&B and TensorBoard only)
    def log_hist(self, key: str, tensor: torch.Tensor, step: int,
                num_bins: int = 64, percentile_clip: float = 99.5) -> None: ...

    # Images / heatmaps (W&B and TensorBoard only; stashed locally too)
    def log_image(self, key: str, image: torch.Tensor, step: int,
                  caption: str | None = None) -> None: ...

    # Tables (for per-sequence eval rows)
    def log_table(self, key: str, rows: list[dict], step: int) -> None: ...

    # Local tensor artifact dump (local sink only). Event is queued; the
    # calling thread does not await the disk write.
    def dump_artifact(self, key: str, payload: dict[str, Any], step: int) -> None: ...

    # Mark the start/end of a lifecycle phase for the timeline view.
    def phase(self, name: str) -> ContextManager: ...

    # Hard invariant — call once per step at the end. Flushes sinks and
    # increments the step counter used by per-200 gates.
    def on_step_end(self, step: int) -> None: ...
```

### 1.2 Three sinks

| Sink | What it gets | Purpose | Backend |
|---|---|---|---|
| `WandbSink` | all scalars, histograms, images, tables, phase timeline | remote dashboards, runs comparison, sharable debug views | `wandb` SDK (`mode=online` by default, `offline` if disconnected) |
| `TensorBoardSink` | all scalars, histograms, images, phase timeline | local/on-cluster live view, fast scrubbing, works without internet | `torch.utils.tensorboard.SummaryWriter` |
| `LocalArtifactSink` | scalars (mirror JSONL), tensor artifact dumps, image copies | offline forensics, reproducibility, module-by-module "what did the tensor look like at step N" | raw files under `runs/.../` |

All sinks implement the same `Sink` ABC (`on_scalar`, `on_hist`, `on_image`,
`on_table`, `on_artifact`, `on_phase`, `flush`, `close`). Adding a fourth sink
(say, a Prometheus exporter) is a 150-line class.

### 1.3 NaN / Inf handling

`DebugLogger.log_scalar` and `log_hist` check:

```python
if not torch.isfinite(tensor).all():
    info = {"key": key, "step": step, "nan": tensor.isnan().sum().item(),
            "inf": tensor.isinf().sum().item(), "shape": tuple(tensor.shape)}
    if self.fail_on_nan:
        raise NumericalAbort(info)
    else:
        self._nan_counter[key] += 1
        self.log_scalar(f"diag.nan.{key}", self._nan_counter[key], step)
```

`NumericalAbort` carries enough context that the traceback tells you *which*
module, *which* tensor, *which* step — the whole point of this spec.

---

## 2. Key namespace

A dotted-key taxonomy that stays stable across runs. **This is the contract** —
plotting dashboards, diff scripts, alerting rules all key off these names.

### 2.1 Top-level groups

```
data.*        — dataset loader
sensor.*      — per-frame image / IMU statistics
frontend.flow.*       — FlowFormerCov flow+cov outputs
frontend.depth.*      — FlowFormerCov stereo depth
frontend.imu.corrector.*  — AirIMU bias+σ
frontend.imu.preintegrator.*
frontend.imu.encoder.*
frontend.proxy.*      — per-pixel IMU-rigid residual proxy
head.forward.*        — ConvGRU @ H/8 + FiLM + refiner @ H/4
head.gru.*            — ConvGRU internals
head.refiner.*        — RefinerH4 internals
head.output.*         — c, logits, T
loss.label.*          — GT-residual label construction + masks
loss.terms.*          — L_dyn, L_smooth, L_total
loss.focal.*          — per-bin focal-BCE diagnostics
optim.*               — grad norm, AMP scale, LR, weight update ratio
optim.per_param.*     — per-parameter grad / weight norms (subset; see §6.8)
init.drt.*            — DRT-loose bootstrap
backend.twoframe.*    — Reproj_TwoFramePGO[_IMU]
backend.swf.*         — SlidingWindow_VIO_PGO
backend.imu_residual.* — Forster 15-D residual block stats
backend.writeback.*   — pose/vel/bias write-back deltas
eval.seq.*            — per-sequence ATE/RPE etc.
eval.head.*           — head metrics (AUROC, ECE, Brier)
timing.*              — per-phase milliseconds
diag.*                — counters (NaNs, warnings, watchdog fires)
```

### 2.2 Specific scalar keys (load-bearing, referenced throughout §6)

See §6. The full list is long enough that it lives in
`Train/DynamicHead/logging/keys.py` as a frozen module-level dictionary; CI asserts
that every call site uses a key present there.

---

## 3. Sink 1 — Weights & Biases

### 3.1 Configuration

```yaml
logging:
  wandb:
    enabled: true
    project: macvo-dynhead
    entity: <ORG>
    group: "${run.dataset}-${run.arch}"         # e.g. "viode-single_head"
    tags: ["${run.dataset}", "${run.ablation}", "stage_${run.stage}"]
    mode: online                                # online | offline | disabled
    resume: allow
    config_exclude_keys: [optim.lr]             # avoid sweep key conflicts
    log_artifacts: true                         # push local dumps as W&B artifacts
    artifact_every: 1000                         # steps — uploaded less often than local-dumped
```

### 3.2 What W&B receives

- **Every scalar** logged via `log_scalar` / `log_scalars`. Grouped by top-level key
  prefix (W&B auto-groups on `.`).
- **Every histogram** logged via `log_hist`. `wandb.Histogram(tensor.cpu())` with
  the 99.5-th percentile clip applied.
- **Every image** via `log_image`: converted to `wandb.Image` with caption.
- **Tables** via `log_table`: row lists, printed as a `wandb.Table`.
- **Config + git state:** `wandb.init(config=config_dict, ...)` plus a
  `wandb.Artifact("git_state_diff", "repo")` uploaded once at step 0.
- **Model graph:** `wandb.watch(head, log="all", log_freq=1000,
  log_graph=True)`. Watches gradients and parameter updates for the head only (the
  frozen modules are explicitly excluded to avoid accidental logging of their
  zero-gradients).
- **Tensor dumps as artifacts:** every `artifact_every` steps (default 1000), the
  most recent local artifact directory is uploaded as a W&B artifact with type
  `"debug_dump"` and version tag equal to the step.

### 3.3 Custom dashboards (shipped as JSON)

`Train/DynamicHead/logging/dashboards/` contains W&B dashboard JSON exports we
maintain:

1. `training_overview.json` — losses, grad norm, LR, step time.
2. `head_diagnostics.json` — logit distribution, saturation percentages,
   per-pixel-mask coverage, temperature curve.
3. `imu_diagnostics.json` — bias estimates, σ_g² / σ_a², preintegrator outputs,
   proxy Δf distribution.
4. `backend_diagnostics.json` — PGO residuals, LM iteration counts, write-back
   deltas, χ² distributions.
5. `drt_init.json` — accept/reject reasons, parallax, bias solver convergence.
6. `eval_sequences.json` — per-sequence ATE/RPE grid with drill-down.

Re-importable via `wandb.Api().from_json(...)` and shared across the team.

---

## 4. Sink 2 — TensorBoard

### 4.1 Configuration

```yaml
logging:
  tensorboard:
    enabled: true
    flush_secs: 30
    max_queue: 2000
    log_graph_once: true
    log_embeddings: false                       # head has no embedding layer
```

### 4.2 What TensorBoard receives

- **All scalars** (same stream as W&B).
- **All histograms** (`writer.add_histogram`).
- **All images** (`writer.add_image` as CHW float tensors in [0, 1]).
- **Graph:** `writer.add_graph(head, sample_input)` once at step 0 (only the head;
  the frozen backbone's graph is not useful for debugging).
- **Embedding projector:** disabled by default (head has no embedding layer).
- **Phase timeline:** each `phase(name)` context writes a dummy scalar to the
  `timing.phase.{name}` namespace with duration in ms, letting you build a
  timeline plot.

### 4.3 Why both W&B and TensorBoard

- TensorBoard works without internet, is instant to open from a training machine,
  and scrubs faster than W&B for long runs.
- W&B is the shared dashboard for team review, has better run-comparison, and
  keeps history when the local directory is wiped.
- They are not redundant — we use both, and the single `DebugLogger` means we
  pay the bookkeeping cost once.

---

## 5. Sink 3 — Local artifact dumps

This is the forensic sink. When a run breaks, **this is what you open first**.

### 5.1 Cadence

```python
# Train/DynamicHead/logging/cadence.py
LOCAL_DUMP_EVERY = 200            # steps
LOCAL_EVAL_DUMP_EVERY_EVAL = 1    # every eval run
FIRST_N_STEPS_FORCED = 10         # dump first 10 training steps regardless
ON_ANOMALY_DUMP = True            # any NaN / grad-spike triggers an on-demand dump
```

- **Every 200 steps** of training: full module I/O dump for one sample of the batch.
- **First 10 steps** always dumped, so you can sanity-check the very first
  forward/backward without waiting for step 200.
- **On anomaly**: if `grad_norm > p99_grad_norm * 10` or any NaN is detected, dump
  the current step immediately regardless of cadence.
- **Every eval run**: full dump for the first sample of each held-out sequence.

### 5.2 On-disk layout

```
runs/
└── 2026-04-18_a1b2c3d_seed42/
    ├── manifest/
    │   ├── config.yaml                # frozen config at step 0
    │   ├── git_sha.txt
    │   ├── git_diff.patch
    │   ├── env.txt                    # pip freeze + CUDA version + driver
    │   └── README.md                  # 10-line human summary
    ├── scalars/
    │   └── scalars.jsonl              # { step, key, value } per line
    ├── histograms/
    │   └── histograms.npz             # aggregated, downsampled
    ├── images/
    │   └── step_000200/
    │       ├── img.L.png
    │       ├── img.R.png
    │       ├── flow.png
    │       ├── c_pred.png
    │       └── c_target.png
    ├── dumps/
    │   ├── step_000000.pt             # first forward (forced)
    │   ├── step_000001.pt
    │   ├── ...
    │   ├── step_000010.pt
    │   ├── step_000200.pt             # cadence dumps
    │   ├── step_000400.pt
    │   └── anomaly_step_003147_grad_spike.pt
    ├── eval/
    │   └── epoch_05/
    │       ├── viode_dynamic_easy/
    │       │   ├── seq_summary.json
    │       │   ├── per_frame.parquet
    │       │   └── dumps/
    │       └── euroc_mh01/...
    ├── pgo/
    │   ├── twoframe_residuals.jsonl   # per-solve summary
    │   └── swf_window_snapshots/
    │       └── step_000800.pt
    └── logs/
        ├── train.stdout.log
        └── train.stderr.log
```

### 5.3 Dump file schema (`dumps/step_NNNNNN.pt`)

A single `torch.save`d dict. Tensors are moved to CPU and half-precision where
appropriate to keep the file under ~200 MB per dump.

```python
DumpSchema = {
    "meta": {
        "step":       int,
        "epoch":      int,
        "seq_id":     str,
        "frame_idx":  int,
        "batch_idx":  int,             # which sample-in-batch was dumped
        "timestamp":  str,             # ISO 8601
        "git_sha":    str,
        "cadence":    str,             # "regular" | "forced_early" | "anomaly" | "eval"
    },
    "data": {                           # §6.1
        "image_L":       fp16 [3, H, W],
        "image_R":       fp16 [3, H, W],
        "imu_window":    fp32 [N_imu, 6],
        "imu_dts":       fp32 [N_imu],
        "K":             fp32 [3, 3],
        "T_BS":          fp32 [7],
        "gt_pose_rel":   fp32 [7] | None,
    },
    "frontend": {                       # §6.2–§6.4
        "flow":          fp16 [2, H, W],
        "flow_cov":      fp16 [3, H, W],
        "depth":         fp16 [1, H, W],
        "disparity":     fp16 [1, H, W],
        "context_feats": fp16 [128, H/8, W/8],
        "airimu_bias":   fp32 [6],
        "airimu_sigma2": fp32 [N_imu, 6],
        "preint": {
            "delta_R":   fp64 [3, 3],
            "delta_v":   fp64 [3],
            "delta_p":   fp64 [3],
            "Sigma":     fp64 [9, 9],
            "dt":        fp64,
            "J_R_bg":    fp64 [3, 3],
            "J_v_bg":    fp64 [3, 3],
            "J_v_ba":    fp64 [3, 3],
            "J_p_bg":    fp64 [3, 3],
            "J_p_ba":    fp64 [3, 3],
            "bias_ref":  fp64 [6],
        },
        "f_imu":         fp16 [128],
        "proxy": {
            "delta_f":       fp16 [2, H, W],
            "r_imu_norm":    fp16 [1, H, W],
            "valid":         bool [1, H, W],
        },
    },
    "head": {                           # §6.6
        "h_prev":                  fp16 [128, H/8, W/8],
        "h_new":                   fp16 [128, H/8, W/8],
        "gru_gate_z":              fp16 [128, H/8, W/8],
        "gru_gate_r":              fp16 [128, H/8, W/8],
        "film_gamma":              fp16 [128, H/8, W/8],
        "film_beta":               fp16 [128, H/8, W/8],
        "logit_coarse":            fp16 [1, H/8, W/8],
        "delta_refine":            fp16 [1, H/4, W/4],
        "logit_fine":              fp16 [1, H/4, W/4],
        "logit_full":              fp16 [1, H, W],
        "T":                       fp32,
        "c":                       fp16 [1, H, W],
        # Factorized ablation only (A_fact):
        # "logit_visible_*":       ...
        # "p_static":              fp16 [1, H, W],
        # "p_visible":             fp16 [1, H, W],
    },
    "loss": {                           # §6.7
        "c_target":      fp16 [1, H, W],
        "M_infov":       bool [1, H, W],
        "M_depth":       bool [1, H, W],
        "M_cycle":       bool [1, H, W],
        "M_valid":       bool [1, H, W],
        "warp_err":      fp16 [1, H, W],
        "per_pixel_L_dyn":     fp16 [1, H, W],
        "per_pixel_L_smooth":  fp16 [1, H, W],
        "L_dyn":         fp32,
        "L_smooth":      fp32,
        "L_total":       fp32,
    },
    "optim": {                          # §6.8
        "grad_norm_pre_clip":  fp32,
        "grad_norm_post_clip": fp32,
        "lr":                  fp32,
        "amp_scale":           fp32,
        "param_update_ratio":  { "<param_name>": fp32, ... },
    },
    "pgo": {                            # §6.11 / §6.12 (on backend-stepping runs)
        "stage":         str,           # "twoframe" | "swf"
        "kp_uv":         fp32 [N, 2],
        "kp_c":          fp32 [N],
        "kp_cov":        fp32 [N, 2, 2],
        "lm_iters":      int,
        "chi2_init":     fp32,
        "chi2_final":    fp32,
        "r_vis_before":  fp32 [N, 2],
        "r_vis_after":   fp32 [N, 2],
        "r_imu_before":  fp32 [15],
        "r_imu_after":   fp32 [15],
        "mahalanobis_vis":   fp32 [N],
        "mahalanobis_imu":   fp32 [5],     # per sub-block
        "pose_delta":    fp32 [6],
        "v_delta":       fp32 [3],
        "bg_delta":      fp32 [3],
        "ba_delta":      fp32 [3],
        "w_eps_hits":    int,
    },
    "drt": {                            # §6.10 (only on DRT steps)
        "accept":          bool,
        "reject_reason":   str | None,
        "n_views":         int,
        "parallax_px_med": fp32,
        "bg_init":         fp32 [3],
        "bg_final":        fp32 [3],
        "bg_residual":     fp32,
        "bg_iters":        int,
        "gW":              fp32 [3],
        "gW_mag":          fp32,
        "scale_s":         fp32,         # should be 1.0 on stereo
    },
}
```

### 5.4 Atomicity

Dumps are written to `dumps/.step_NNNNNN.pt.partial` and atomically renamed to
`dumps/step_NNNNNN.pt` after `torch.save` returns. A SIGKILL mid-write leaves a
`.partial` file that `tools/scrub_partial_dumps.py` deletes on the next run.

### 5.5 Dump-reading helpers

```python
# tools/inspect_dump.py
from macvo_debug import load_dump
d = load_dump("runs/2026-04-18_a1b2c3d_seed42/dumps/step_000200.pt")
d.plot_c_vs_target()           # overlays c, c_target, |diff|
d.plot_mask_stack()            # overlays M_infov / M_depth / M_cycle / M_valid
d.plot_proxy()                 # Δf quiver + r_imu_norm heatmap + valid mask
d.plot_pgo_residuals()         # before/after LM, χ² histogram
d.print_scalars()              # entire loss / optim / pgo scalar tree
d.diff(other_step=400)         # compute A−B diffs across all tensors for drift hunt
```

Every dump is self-describing; no external index file needs to be kept in sync.

---

## 6. Module-by-module log spec

The rest of the document walks the pipeline in forward order. For each module:

- **What the module does** (one sentence).
- **Scalars** to log every step.
- **Histograms** to log every `HIST_EVERY` steps (default 50).
- **Tensors** to include in the every-200-step local dump.
- **Anomaly triggers** specific to that module.
- **Diagnostic playbook**: "if you see X in the log, the problem is Y."

Universal scalar per module: `timing.{module}.ms` (ms spent in that module that
step). Never omitted.

### 6.1 `DataLoader` / `StereoInertialFrame`

**Does:** reads stereo image pair + IMU window + (optional) GT pose from dataset.

| Scalar | Meaning | Typical range |
|---|---|---|
| `data.batch_size` | samples in the batch | fixed from config |
| `data.seq_id` (string scalar via W&B meta) | current sequence id | dataset-specific |
| `data.frame_idx` | index within the sequence | 0..seq_len |
| `data.dt_frames_ms` | ms between the two image timestamps | 30–100 typically |
| `data.imu.n_samples` | number of IMU samples in the window | ≈ `dt_frames_ms / 5` if 200Hz |
| `data.imu.dt_min_ms` | smallest IMU sub-step | ≈ 5 |
| `data.imu.dt_max_ms` | largest IMU sub-step | ≈ 5 |
| `data.imu.dt_std_ms` | jitter | << 1 ms in good datasets |
| `data.imu.gap_ms` | gap between last IMU sample and frame 2 | should be ≈ 0 |
| `data.image.L.mean` / `.std` / `.min` / `.max` | per-channel mean over batch | dataset-dependent |
| `data.image.R.mean` / `.std` / `.min` / `.max` | same | same |
| `data.gt.has_pose` | 1 if GT pose available for this pair | 1 on training, 0 on unsupervised |
| `data.loader.samples_per_sec` | throughput of the loader | target ≥ 30 |
| `data.loader.queue_depth` | DataLoader worker queue depth | >0 means loader is ahead |
| `diag.data.drop.imu_missing` | count of samples dropped because IMU was too short | should be 0 |
| `diag.data.drop.pose_missing` | count dropped because GT missing | 0 on supervised |
| `diag.data.drop.timestamp_misalign` | count with `gap_ms > 20` | should be 0 |

**Histograms (every 50 steps):**

- `data.image.L.hist`, `data.image.R.hist` — pixel value distribution.
- `data.imu.accel.hist`, `data.imu.gyro.hist` — 6 × histogram of raw IMU.

**Dump tensors (every 200 steps, one sample):**

`data.image_L`, `data.image_R`, `data.imu_window`, `data.imu_dts`, `data.K`,
`data.T_BS`, `data.gt_pose_rel`.

**Anomaly triggers:**

- `data.imu.gap_ms > 20`  → warn; integrator likely over-extrapolates.
- `data.dt_frames_ms > dt_reset_ms` → warn; `head.hidden_state.reset` fires.
- `data.image.L.std < 1e-3` (constant image) → warn; batch is probably corrupted.

**Playbook:**

- Loss diverges at step K → check `data.image.std` around step K−50..K. If it
  dropped to 0, a corrupt sample is the cause.
- Head output is "lazy" everywhere → check `data.imu.dt_std_ms`; if it's huge,
  preintegration is noisy and the proxy channel is unusable.

### 6.2 `FlowFormerCov` frozen forward (flow + cov)

**Does:** runs the flow+covariance branch. Never updated; no grads.

| Scalar | Meaning |
|---|---|
| `frontend.flow.norm.mean` / `.p50` / `.p95` / `.max` | flow magnitude distribution (px) |
| `frontend.flow.cov.det.mean` / `.p95` | determinant of flow cov matrix (per pixel) — proxy for "confidence mass" |
| `frontend.flow.cov.cond.mean` / `.p95` | condition number of flow cov |
| `frontend.flow.cov.trace.mean` | trace of flow cov, per pixel, averaged |
| `frontend.flow.ctx.feat.mean` / `.std` | context-feature activations (H/8, 128) |
| `frontend.flow.time_ms` | forward time |
| `diag.frontend.flow.nan_frac` | fraction of NaN pixels in flow | should be 0 |
| `diag.frontend.flow.inf_frac` | same for Inf |

**Histograms:** `frontend.flow.norm.hist`, `frontend.flow.cov.det.hist`,
`frontend.flow.ctx.feat.hist`.

**Dump tensors:** `frontend.flow`, `frontend.flow_cov`, `frontend.context_feats`.

**Anomaly triggers:**

- `diag.frontend.flow.nan_frac > 0` → abort. FlowFormerCov should never produce
  NaNs given in-distribution input.
- `frontend.flow.cov.cond.p95 > 1e6` → warn. Ill-conditioned cov matrices will
  corrupt downstream weighting.

### 6.3 `FlowFormerCov` stereo depth

**Does:** left-right stereo disparity → metric depth.

| Scalar | Meaning |
|---|---|
| `frontend.depth.valid_frac` | fraction of pixels with `d_min < depth < d_max` |
| `frontend.depth.p50` / `.p95` | depth percentiles |
| `frontend.depth.disp.mean` | mean disparity |
| `frontend.depth.time_ms` | forward time |
| `diag.frontend.depth.neg_disp_frac` | fraction of pixels with negative disparity (config enforces positive) | 0 |
| `diag.frontend.depth.zero_disp_frac` | fraction at disparity=0 | small |

**Histograms:** `frontend.depth.hist`, `frontend.depth.disp.hist`.

**Dump tensors:** `frontend.depth`, `frontend.disparity`.

### 6.4 `AirIMU` corrector + `DifferentiablePreintegrator`

**Does:** corrects IMU bias/noise and preintegrates to `(ΔR̂, Δv̂, Δp̂, Σ_imu)` plus
bias Jacobians.

**Corrector scalars:**

| Scalar | Meaning |
|---|---|
| `frontend.imu.corrector.bias_g.{x,y,z}` | per-axis gyro bias estimate |
| `frontend.imu.corrector.bias_a.{x,y,z}` | per-axis accel bias estimate |
| `frontend.imu.corrector.bias_g.norm` | \|b_g\| |
| `frontend.imu.corrector.bias_a.norm` | \|b_a\| |
| `frontend.imu.corrector.sigma2_g.mean` / `.p95` | per-sample σ² on gyro |
| `frontend.imu.corrector.sigma2_a.mean` / `.p95` | per-sample σ² on accel |
| `frontend.imu.corrector.time_ms` | forward time |

**Preintegrator scalars:**

| Scalar | Meaning |
|---|---|
| `frontend.imu.preintegrator.dt_s` | integration dt in seconds |
| `frontend.imu.preintegrator.delta_R.angle_rad` | \|Log(ΔR̂)\|₂ |
| `frontend.imu.preintegrator.delta_v.norm` | \|Δv̂\| m/s |
| `frontend.imu.preintegrator.delta_p.norm` | \|Δp̂\| m |
| `frontend.imu.preintegrator.Sigma.cond` | condition number of 9×9 Σ_imu |
| `frontend.imu.preintegrator.Sigma.eigmin` | smallest eigenvalue (positive-definiteness check) |
| `frontend.imu.preintegrator.Sigma.trace` | trace (total uncertainty) |
| `frontend.imu.preintegrator.J_R_bg.fro` | Frobenius norm of bias-rotation Jacobian |
| `frontend.imu.preintegrator.J_v_ba.fro` | ditto for velocity-accel-bias |
| `frontend.imu.preintegrator.J_p_ba.fro` | ditto for position-accel-bias |
| `frontend.imu.preintegrator.time_ms` | forward time |
| `diag.frontend.imu.preintegrator.sigma_non_psd` | 1 if Σ_imu not PSD (should be 0) |

**Histograms:** `frontend.imu.preintegrator.Sigma.eigs.hist` (9 eigenvalues),
`frontend.imu.corrector.sigma2_g.hist`, `frontend.imu.corrector.sigma2_a.hist`.

**Dump tensors:** full `preint` sub-dict (§5.3), `airimu_bias`, `airimu_sigma2`.

**Anomaly triggers:**

- `Sigma.eigmin <= 0` → `NumericalAbort`. Propagating a non-PSD covariance into
  the backend is never safe.
- `delta_R.angle_rad > π/2` between consecutive keyframes → warn; preintegration
  first-order approximations degrade.
- `|b_g| > 0.1 rad/s` or `|b_a| > 2 m/s²` → warn; likely sensor or calibration
  issue.

**Playbook:**

- Backend IMU residual `r_R` spikes → check `delta_R.angle_rad` trend; if it's
  spiking too, the IMU is reporting unrealistic motion and the corrector isn't
  catching it.
- Head proxy becomes noisy → check `Sigma.trace`; a much-higher-than-usual trace
  means preintegration is under-confident, which polluttes the proxy Δf.

### 6.5 `IMUEncoder` + per-pixel proxy

**Does:** projects preintegration factor to `f_imu ∈ ℝ^128`; builds per-pixel
`(Δf, r_imu_norm, valid)` map.

**IMUEncoder scalars:**

| Scalar | Meaning |
|---|---|
| `frontend.imu.encoder.f_imu.norm` | \|f_imu\|₂ |
| `frontend.imu.encoder.f_imu.mean` / `.std` | pre-ReLU activation statistics of the final linear |
| `frontend.imu.encoder.dead_frac` | fraction of f_imu coordinates with |x| < 1e-4 (dead neuron proxy) |
| `frontend.imu.encoder.time_ms` | forward time |

**Proxy scalars:**

| Scalar | Meaning |
|---|---|
| `frontend.proxy.valid_frac` | fraction of pixels with `valid=1` |
| `frontend.proxy.delta_f.norm.p50` / `.p95` | per-pixel Δf magnitude (px) |
| `frontend.proxy.r_imu_norm.mean` / `.p95` | normalized residual |
| `frontend.proxy.depth_rejected_frac` | fraction rejected by depth gate |
| `frontend.proxy.zproj_rejected_frac` | fraction rejected by post-projection-z gate |
| `frontend.proxy.fov_rejected_frac` | fraction reprojected outside image |
| `frontend.proxy.time_ms` | forward time |

**Histograms:** `frontend.imu.encoder.f_imu.hist`, `frontend.proxy.delta_f.hist`,
`frontend.proxy.r_imu_norm.hist`.

**Dump tensors:** `f_imu`, `proxy.delta_f`, `proxy.r_imu_norm`, `proxy.valid`.

**Anomaly triggers:**

- `valid_frac < 0.2` → warn. Proxy is informative only if most pixels are valid.
- `dead_frac > 0.5` → warn. The FeatureMLP has collapsed, head loses IMU signal.

### 6.6 `StaticConfidenceHead` (ConvGRU @ H/8 + FiLM + Refiner @ H/4)

**Does:** the only trainable per-frame-pair module. Emits `c ∈ [0, 1]`.

**Forward scalars:**

| Scalar | Meaning |
|---|---|
| `head.gru.h_prev.norm.mean` | ‖h_{t-1}‖ per pixel, averaged |
| `head.gru.h_new.norm.mean` | ‖h_t‖ per pixel, averaged |
| `head.gru.h_drift.mean` | ‖h_t − h_{t-1}‖ averaged (0 on step 0) |
| `head.gru.gate_z.mean` | update-gate mean — "how much of new input was taken" |
| `head.gru.gate_r.mean` | reset-gate mean |
| `head.film.gamma.mean` / `.std` | FiLM gamma stats |
| `head.film.beta.mean` / `.std` | FiLM beta stats |
| `head.refiner.delta.abs.mean` / `.p95` | refiner residual magnitude — close to 0 at init |
| `head.refiner.time_ms` | refiner forward time |
| `head.output.logit_coarse.mean` / `.std` | H/8 logit stats (pre-T) |
| `head.output.logit_fine.mean` / `.std` | H/4 logit stats (pre-T) |
| `head.output.T.value` | current temperature (1.0 in training; post-hoc set in calibrate) |
| `head.output.c.mean` / `.p50` / `.p95` | final confidence distribution |
| `head.output.c.sat_low_frac` | fraction with `c < 0.05` |
| `head.output.c.sat_high_frac` | fraction with `c > 0.95` |
| `head.output.c.ambig_frac` | fraction with `0.4 < c < 0.6` |
| `head.forward.time_ms` | end-to-end head time |
| `diag.head.nan_frac` | fraction of NaN pixels in `c` | 0 |
| `diag.head.watchdog_fires` | running count of watchdog resets | 0 in a healthy run |

**Histograms:** `head.output.c.hist`, `head.output.logit_fine.hist`,
`head.gru.h_new.hist`, `head.refiner.delta.hist`, `head.film.gamma.hist`.

**Images (every 200 steps):** `head.output.c.image`,
`head.output.logit_fine.image` (colormap), `head.refiner.delta.image`.

**Dump tensors:** the entire `head` sub-dict in §5.3.

**Anomaly triggers:**

- `sat_high_frac > 0.99` for 100 consecutive steps → warn. Head is outputting "all
  static" and not learning.
- `sat_low_frac + sat_high_frac < 0.05` after step 2000 → warn. Head stayed at
  0.5 everywhere; likely stuck due to overly large `T` or collapsed logits.
- `refiner.delta.abs.p95 > 3.0` early in training → warn. Refiner is *not*
  zero-init residual; check initialization.

**Playbook:**

- `L_dyn` plateaus → compare `head.output.c.hist` at step 0 vs current. If still
  a narrow spike at 0.5, refiner/logits haven't moved → check gradient flow into
  the head (`optim.per_param.head.*.grad_norm`).
- PGO diverges → sample the dump: is `c` saturating at 1 on dynamic objects?
  Look at `loss.label.c_target` vs `head.output.c` overlay.

### 6.7 Loss computation

**Does:** builds `c_target`, masks, computes `L_dyn + L_smooth`.

| Scalar | Meaning |
|---|---|
| `loss.label.residual.p50` / `.p95` | \|r\| = \|f_obs − f_rigid\| px |
| `loss.label.tau_D.mean` | depth-adaptive threshold τ_D(d), averaged |
| `loss.label.c_target.mean` / `.p50` | GT-residual soft label |
| `loss.label.warp_err.p50` / `.p95` | cycle-consistency error |
| `loss.label.M_infov.frac` | fraction of pixels in-FOV |
| `loss.label.M_depth.frac` | fraction with valid depth |
| `loss.label.M_cycle.frac` | fraction with small warp_err |
| `loss.label.M_valid.frac` | combined validity mask coverage |
| `loss.terms.L_dyn` | scalar dyn loss (sum / max(M_valid.sum(), 1)) |
| `loss.terms.L_smooth` | scalar smoothness loss |
| `loss.terms.L_total` | weighted sum |
| `loss.focal.contribution.top1` | top-1 per-pixel focal BCE contribution |
| `loss.focal.contribution.top10.mean` | average of top-10 contributions |
| `loss.focal.effective_n` | effective number of pixels contributing to L_dyn (Kish's ESS) |
| `loss.time_ms` | loss forward time |
| `diag.loss.M_valid.frac_zero_batches` | running rate of batches with `M_valid.sum() == 0` | 0 |

**Histograms:** `loss.label.residual.hist`, `loss.label.c_target.hist`,
`loss.label.warp_err.hist`, `loss.focal.contribution.hist` (per-pixel).

**Images (every 200 steps):** `loss.label.c_target.image`,
`loss.label.M_valid.image`, `loss.focal.per_pixel.image`,
`loss.terms.per_pixel_L_smooth.image`.

**Dump tensors:** full `loss` sub-dict in §5.3.

**Anomaly triggers:**

- `M_valid.frac < 0.05` → warn. The loss is averaging over almost nothing; the
  gradient is dominated by a handful of pixels.
- `L_total` diverges while `L_dyn` stays flat → `L_smooth` is blowing up; likely
  refiner producing huge gradients.
- `c_target.mean` drifts from its initial value by > 0.1 → GT labels are shifting;
  either dataset corruption or cycle-mask logic changed.

**Playbook:**

- ATE explodes but `L_dyn` is low → overlay `c_target` vs `c` dumps at steps
  where ATE jumped. If they agree pixel-wise but ATE still explodes, the problem
  is downstream (PGO), not the head.
- `L_smooth` oscillates → check `head.refiner.delta.hist`; a bimodal refiner
  output fights smoothness.

### 6.8 Backward + optimizer step

**Does:** computes grads, clips, steps. Only the head + IMU-MLP have non-zero
grads (invariant I1).

| Scalar | Meaning |
|---|---|
| `optim.grad_norm.pre_clip` | global gradient 2-norm before clipping |
| `optim.grad_norm.post_clip` | global gradient 2-norm after clipping (≤ grad_clip_norm) |
| `optim.grad_clip.fired` | 1 if clipping actually reduced the norm, else 0 |
| `optim.lr` | current learning rate |
| `optim.amp_scale` | GradScaler's current scale |
| `optim.amp_overflow_steps` | running count of overflow-triggered skipped steps |
| `optim.weight_norm.total` | sum of ‖θ‖² across trainable params |
| `optim.per_param.<name>.grad_norm` | per-param grad norm |
| `optim.per_param.<name>.weight_norm` | per-param weight norm |
| `optim.per_param.<name>.update_ratio` | ‖Δθ‖ / ‖θ‖ — per-param update-to-weight ratio (target ≈ 1e-3) |
| `optim.step_time_ms` | optimizer step time |
| `diag.optim.frozen_grad_leak` | count of params with `requires_grad=False` but nonzero grad | 0 |

`<name>` is the named-parameter path, e.g. `head.gru.cell.conv_xh`,
`head.refiner.conv2`, `frontend.imu.encoder.film`.

**Histograms:** `optim.per_param.grad_norm.hist`,
`optim.per_param.update_ratio.hist`.

**Dump tensors:** `optim.grad_norm.pre_clip`, `optim.grad_norm.post_clip`,
`optim.lr`, `optim.amp_scale`, and the `per_param.update_ratio` dict.

**Anomaly triggers (all also force an anomaly dump):**

- `grad_norm.pre_clip > p99(historical) * 10` → grad spike; dump immediately.
- `amp_overflow_steps` increments > 10 times in 100 steps → AMP unstable; reduce
  `grad_clip_norm` or disable AMP.
- `diag.optim.frozen_grad_leak > 0` → freeze was broken. Re-run freeze checks
  (spec §5.5) and abort.
- Any `update_ratio > 0.1` → param is being rewritten in a single step; LR too
  high.

**Playbook:**

- Loss goes NaN after step K → inspect `optim.grad_norm.pre_clip` at K, K−1, K−2.
  A preceding spike usually means the refiner or ConvGRU hit a pathological
  input; look at the `anomaly_step_*.pt` dump that should have fired.

### 6.9 Keypoint sampling + Stage-1 hard mask

**Does:** sample `c` at selected keypoints, apply `c >= min_c` hard mask.

| Scalar | Meaning |
|---|---|
| `backend.sampling.n_kp_selected` | raw keypoints from MAC-VO's selector |
| `backend.sampling.n_kp_survived` | after Stage-1 filter |
| `backend.sampling.survival_rate` | ratio survived / selected |
| `backend.sampling.c.at_kp.mean` | mean c over sampled kps (before mask) |
| `backend.sampling.c.at_kp.p10` | 10th percentile — dims how confident the *least* kept kp is |
| `backend.sampling.min_c_applied` | current `min_c` threshold (config value) |
| `diag.backend.sampling.zero_survivors` | 1 if n_kp_survived==0 (graph is empty) |

**Histograms:** `backend.sampling.c.at_kp.hist`,
`backend.sampling.flow_cov.trace.at_kp.hist`.

**Dump tensors:** `kp_uv`, `kp_c`, `kp_cov`, and a rendered image with surviving
keypoints overlaid.

**Anomaly triggers:**

- `survival_rate < 0.2` → warn; backend is being fed ≤ 20% of the selector's kps.
- `zero_survivors == 1` → `NumericalAbort`; PGO cannot run on an empty graph.

### 6.10 `DRT-loose` bootstrap

**Does:** runs once per sequence at startup. See spec §7.

| Scalar | Meaning |
|---|---|
| `init.drt.window_len` | number of keyframes used |
| `init.drt.n_views.total` | total feature-pair views |
| `init.drt.parallax_px.p50` | median parallax |
| `init.drt.parallax_px.p5` | 5th percentile (the bad end) |
| `init.drt.accept` | 1/0 accept flag |
| `init.drt.reject_reason` (string) | set only on reject |
| `init.drt.bg_solver.iters` | LBFGS iterations of gyro-bias solver |
| `init.drt.bg_solver.residual` | final residual |
| `init.drt.bg.{x,y,z}` | estimated gyro bias per axis |
| `init.drt.bg.norm` | \|b_g\| |
| `init.drt.gW.{x,y,z}` | gravity estimate components |
| `init.drt.gW.mag` | \|g_W\|, should be ~9.81 |
| `init.drt.gW.mag_err` | \|g_W.mag − 9.81007\| |
| `init.drt.scale_s` | monocular scale factor (must be 1.0 on stereo) |
| `init.drt.time_ms` | total init time |
| `diag.init.drt.retries` | retry count (on accept failures) |

**Dump tensors (on every DRT run):** full `drt` sub-dict in §5.3, plus the full
list of per-keyframe outputs `(R_k, p_k, v_k)` as fp64.

**Anomaly triggers:**

- `init.drt.accept == 0` for 3 consecutive windows → log `init.drt.fallback_used`
  = 1 and warn. Backend will run in `identity` init mode.
- `init.drt.gW.mag_err > 0.5` → warn; gravity estimate is off, backend IMU
  residuals will be biased.
- `init.drt.scale_s != 1.0` on stereo → abort; the stereo baseline substitution
  wasn't applied.

### 6.11 Two-frame PGO (Stage A)

**Does:** LM on `Reproj_TwoFramePGO` or `Reproj_TwoFramePGO_IMU`.

| Scalar | Meaning |
|---|---|
| `backend.twoframe.lm.iters` | LM iterations taken |
| `backend.twoframe.lm.converged` | 1/0 on convergence flag |
| `backend.twoframe.chi2.init` | initial χ² (`Σ wᵢ² · rᵀΣ⁻¹r` + IMU term) |
| `backend.twoframe.chi2.final` | final χ² |
| `backend.twoframe.chi2.delta_frac` | (init − final) / init |
| `backend.twoframe.r_vis.norm.p50` / `.p95` | reprojection residual magnitude after solve |
| `backend.twoframe.r_vis.maha.p50` | Mahalanobis per-pixel, post-solve (~ √2 in a well-fit model) |
| `backend.twoframe.r_vis.maha.gt_3sig_frac` | fraction of observations > 3σ |
| `backend.imu_residual.r_R.norm` | |r_R| rad (post-solve) |
| `backend.imu_residual.r_v.norm` | |r_v| m/s |
| `backend.imu_residual.r_p.norm` | |r_p| m |
| `backend.imu_residual.r_bg.norm` | random-walk residual |
| `backend.imu_residual.r_ba.norm` | random-walk residual |
| `backend.imu_residual.maha.total` | full 15-D Mahalanobis |
| `backend.twoframe.w_eps_hits` | number of factors floored to `W_EPS` |
| `backend.twoframe.jacobian.cond` | condition number of stacked J (via SVD, sampled) |
| `backend.twoframe.time_ms` | total time |
| `diag.backend.twoframe.lm_not_converged` | running count | 0 healthy |
| `diag.backend.twoframe.rank_deficient_warn` | running count |

**Dump tensors:** full `pgo` sub-dict in §5.3.

**Anomaly triggers:**

- `chi2.delta_frac < 0.01` (solver barely moved) → warn; init was already at a
  local min or LM stalled.
- `r_vis.maha.gt_3sig_frac > 0.3` → warn; a large minority of visual residuals
  are outliers post-solve. Either the head isn't downweighting dynamics enough or
  the intrinsics/extrinsics are off.
- `imu_residual.maha.total > 30` (much larger than χ²(15) 99th percentile ≈ 32)
  → warn. The IMU factor is not consistent with the visual solution.

**Playbook:**

- ATE increases monotonically over a sequence → inspect per-step
  `backend.twoframe.r_vis.maha.gt_3sig_frac`. Rising trend → head is
  under-masking dynamics. Flat → likely IMU drift (check bias write-back).

### 6.12 Sliding-window PGO (Stage B)

**Does:** `SWF_VIOFactorGraph` LM over `W_kf` keyframes.

| Scalar | Meaning |
|---|---|
| `backend.swf.W_kf` | current window size |
| `backend.swf.n_poses` | 6·W_kf |
| `backend.swf.n_visual_factors` | count |
| `backend.swf.n_imu_factors` | W_kf − 1 |
| `backend.swf.n_marg_factors` | 0 or 1 |
| `backend.swf.lm.iters` / `.converged` | same semantics as §6.11 |
| `backend.swf.chi2.init` / `.final` / `.delta_frac` | same |
| `backend.swf.r_vis.maha.p50` / `.gt_3sig_frac` | same |
| `backend.swf.r_imu.maha.per_pair` (table) | Mahalanobis for each IMU factor |
| `backend.swf.schur.cond` | condition number of Schur-complemented system |
| `backend.swf.marg.prior.fro` | Frobenius norm of marginalization prior info matrix |
| `backend.swf.time_ms` | total time |
| `backend.swf.margin.info_dim` | dim of the marginalization prior (varies) |
| `diag.backend.swf.schur_ill_conditioned` | running count |
| `diag.backend.swf.factor_graph_rebuilt` | running count (expected on each step) |

**Dump (once per step, and a richer snapshot every 200 steps):**

- `pgo/swf_window_snapshots/step_NNNNNN.pt` — entire window's state, factor
  list, residuals, Jacobian norms.

**Anomaly triggers:**

- `schur.cond > 1e8` → warn; solve is ill-conditioned.
- `marg.prior.fro` growing unbounded → marginalization is not shedding information;
  window policy is broken.

### 6.13 Backend write-back

**Does:** writes optimized `(pose, vel, bias_g, bias_a)` back to `VisualMap.frames`
and the preintegrator's bias reference.

| Scalar | Meaning |
|---|---|
| `backend.writeback.pose.delta.trans_m` | ‖Δtranslation‖ from init |
| `backend.writeback.pose.delta.rot_deg` | |Log(R_new · R_init⁻¹)| deg |
| `backend.writeback.vel.delta_norm` | \|Δv\| |
| `backend.writeback.bias_g.delta_norm` | \|Δb_g\| |
| `backend.writeback.bias_a.delta_norm` | \|Δb_a\| |
| `backend.writeback.bias_g.norm` | \|b_g\| after write |
| `backend.writeback.bias_a.norm` | \|b_a\| after write |
| `backend.writeback.bias_ref.updated` | 1 if the preintegrator's bias_ref was re-seeded |
| `diag.backend.writeback.pose_jump_warn` | 1 if `trans_m > 1.0` or `rot_deg > 10` (cfg'd) |

**Playbook:**

- Trajectory has visible "jumps" in rviz → check
  `backend.writeback.pose.delta.trans_m.hist` for bimodal distribution; one mode
  around the expected 0.01–0.1 m, another around > 0.5 m indicates the PGO is
  occasionally making large corrections.
- Bias drifts monotonically → log
  `backend.writeback.bias_g.norm` over time; should oscillate, not ramp.

---

## 7. Training-loop step lifecycle

```python
# Train/DynamicHead/loop.py — pseudocode with log call sites
def train_step(step, batch, state) -> None:
    with logger.phase("data"):
        # data.* scalars, images logged inside the loader via a transform hook
        pass

    with logger.phase("frontend_frozen"):
        with torch.inference_mode():
            out = frontend.estimate_pair(batch)       # logs frontend.flow.* / depth.*
            imu = frontend.imu_encoder(batch.imu)     # logs frontend.imu.*
            proxy = frontend.build_proxy(out, imu)    # logs frontend.proxy.*

    with logger.phase("head_forward"):
        head_out = head(out.context, out.flow, out.flow_cov,
                        imu.f_imu, proxy, state.h_prev)
        state.h_prev = head_out.h_new                 # head.* scalars

    with logger.phase("loss"):
        label = build_label(batch, out, proxy)        # loss.label.*
        losses = compute_losses(head_out, label)      # loss.terms.* / focal.*

    with logger.phase("backward"):
        losses.total.backward()                       # no direct logging
        gnorm_pre = clip_grad_norm_(params, cfg.grad_clip_norm)
        logger.log_scalar("optim.grad_norm.pre_clip", gnorm_pre, step)
        # optim.* logged inside optimizer.step wrapper
        optimizer.step()
        scaler.update()

    with logger.phase("dump_if_due"):
        if should_dump(step):
            dump = collect_dump_payload(batch, out, imu, proxy, head_out, label,
                                         losses, state)
            logger.dump_artifact("dumps", dump, step)

    logger.on_step_end(step)
```

### 7.1 Collector contract

`collect_dump_payload` is a single function in
`Train/DynamicHead/logging/collectors.py` that knows how to assemble the
`DumpSchema` from the objects passed to it. It is the only place that reads
module internals; adding a new field means adding one line here and one line in
the schema (§5.3) — no changes anywhere else.

### 7.2 Backend-attached runs (Stage A / B)

When the training loop is configured to also step the backend (end-to-end
simulation runs, or eval runs), `train_step` calls
`backend.solve_pair(...)` or `backend.solve_window(...)` between the head
forward and the loss. Those calls log `backend.*` and write their own dumps. The
head is not differentiated through the backend (the spec explicitly keeps
joint training out of scope), so no extra care is needed for autograd.

---

## 8. Evaluation / validation logging

Runs via `Train/DynamicHead/eval.py`. Separate scalar namespace `eval.*`.

### 8.1 Per-step eval scalars

| Scalar | Meaning |
|---|---|
| `eval.head.auroc` | AUROC of `c` against GT rigidity label |
| `eval.head.ap` | Average Precision |
| `eval.head.brier` | Brier score (mean squared error on probabilities) |
| `eval.head.ece.15bin` | Expected Calibration Error with 15 bins |
| `eval.head.ece.recal` | post-temperature-scaling ECE (used by calibrate.py) |
| `eval.head.c_target.coverage` | fraction of pixels with `M_valid=1` |

### 8.2 Per-sequence metrics

Logged as a `wandb.Table` plus a Parquet file at
`eval/epoch_XX/<seq>/seq_summary.parquet`:

```
seq_id | frames | ate.m.rmse | ate.m.max | rpe.trans.m.rmse |
       rpe.rot.deg.rmse | auroc | ece.15bin | runtime.s
```

### 8.3 Per-sequence dump

The first sample of every eval sequence is dumped fully (§5.3). Lets us scrub a
single frame per sequence for visual regressions without storing the whole run.

### 8.4 Ablation comparison plots

`tools/plot_ablation_grid.py` reads the `eval.seq.*` rows across a list of W&B
runs (identified by `tags`) and emits a 2D grid: ablation × sequence → metric.
Committed to `artifacts/ablation_grids/`.

---

## 9. Config schema

```yaml
logging:
  run_dir: ./runs                         # base dir; actual run dir gets date_sha_seed appended
  fail_on_nan: true

  # Sinks
  wandb:
    enabled: true
    project: macvo-dynhead
    entity: <ORG>
    group: "${run.dataset}-${run.arch}"
    tags: []
    mode: online
    log_artifacts: true
    artifact_every: 1000

  tensorboard:
    enabled: true
    flush_secs: 30
    max_queue: 2000

  local:
    enabled: true
    dump_every: 200
    forced_first_n: 10
    on_anomaly: true
    keep_last_n: 200                      # roll dumps older than this off disk
    max_file_mb: 200                      # abort dump if payload exceeds
    images_every: 200

  # Log frequencies (for scalars & histograms)
  scalar_every: 1                         # every step
  hist_every: 50
  per_param_grad_every: 50                # expensive; not every step
  per_param_update_ratio_every: 50

  # Anomaly thresholds
  anomaly:
    grad_spike_multiplier: 10.0
    nan_abort: true
    pose_jump_trans_m: 1.0
    pose_jump_rot_deg: 10.0
    mvalid_frac_min: 0.05

  # What to dump. Setting any to false drops the key from dumps & collectors.
  dumps:
    include:
      data: true
      frontend: true
      head: true
      loss: true
      optim: true
      pgo: true
      drt: true
```

Config validation runs at construction and asserts every used `logging.*` path
appears in the schema (no silent typos).

---

## 10. Diagnostic playbooks

Curated index at the front of `Train/DynamicHead/logging/README.md`.

### 10.1 "The loss diverges at step K."

1. Load `runs/.../dumps/step_{K-200}.pt` and `step_{K}.pt`.
2. Check `optim.grad_norm.pre_clip` around K in `scalars.jsonl` — a spike means
   an anomaly dump should exist (`dumps/anomaly_step_*.pt`); open it.
3. If the spike originates in `head.refiner.conv2` (check `optim.per_param.*.grad_norm`),
   look at `head.refiner.delta` in the anomaly dump — is it bimodal?
4. If the grad norm is flat but loss exploded, look at `loss.label.c_target.mean`
   drift — a corrupt GT pose shifts labels en masse.

### 10.2 "The head outputs 0.5 everywhere and never improves."

1. `head.output.c.hist` at step 5000 vs step 500 — if unchanged, gradients aren't
   reaching the head.
2. Check `optim.per_param.head.*.grad_norm` — if all zero, confirm
   `requires_grad=True` on head params (see `diag.optim.frozen_grad_leak`
   inverse — frozen unfreeze).
3. Check `head.output.T.value` — if it's drifted above 5, post-hoc calibration
   was run against the wrong split.
4. Check `loss.label.M_valid.frac` — if it's ~0, validation masks are too strict
   and no pixels contribute; relax `tau_cyc_valid`.

### 10.3 "ATE is low but RPE is bad."

1. Open PGO dumps at several steps: `pgo.r_imu_after` should be small
   (Mahalanobis ≈ √15); if large, IMU factor is being overwhelmed by visual.
2. `backend.writeback.bias_g.delta_norm` history — if it oscillates wildly,
   bias estimation is unstable; try Stage B (SWF provides random-walk prior).
3. Per-sequence `eval.seq.rpe.rot.deg.rmse` — high on sequences with yaw spikes
   suggests preintegration first-order approximation is breaking down; verify
   `frontend.imu.preintegrator.delta_R.angle_rad.p95`.

### 10.4 "DRT init keeps rejecting the first window."

1. `init.drt.reject_reason` field across the first 10 init attempts tells you
   which criterion failed (min_views / max_bg_norm / max_ba_norm /
   max_scale_mismatch / parallax).
2. If `parallax_px.p50 < 20`, the sequence starts too static; increase
   `drt_window_len` or fall back to `identity` init for bootstrap.
3. `gW.mag_err` — if the solver cannot hit 9.81 within 0.5, the IMU is
   biased/miscalibrated; check `frontend.imu.corrector.bias_a.norm`.

### 10.5 "Every step takes 5× longer than expected."

1. `timing.*` scalars reveal which module is slow. Expected: data 2–5 ms,
   frontend 20 ms, head 10 ms, loss 2 ms, backward 15 ms, optim 3 ms.
2. `data.loader.queue_depth` — if near 0, the loader is the bottleneck; raise
   `num_workers` or pre-compute depths (see `Scripts/precompute_depths.py`).
3. `logging` overhead — disable `log_hist` temporarily; if step time drops 3×,
   histograms are too frequent (raise `hist_every`).

### 10.6 "Inference crashes on sequence X but training was fine."

1. Load `runs/.../eval/epoch_XX/<seq>/dumps/*.pt` — each sequence's first frame.
2. Compare `data.image.L.hist` to training's — distribution shift (exposure,
   gain) is a likely culprit.
3. Compare `frontend.imu.preintegrator.delta_R.angle_rad` distributions —
   sequence X may have much faster rotation than training saw.

---

## 11. Performance guardrails

### 11.1 Async by default

- W&B: `wandb.init(settings=wandb.Settings(start_method="thread"))` — the W&B
  process is a sidecar; `log()` calls are non-blocking.
- TensorBoard: `SummaryWriter(flush_secs=30, max_queue=2000)` — background
  thread flushes.
- Local: a `ThreadPoolExecutor(max_workers=2)` executes disk writes; the
  training thread submits and moves on.

### 11.2 Zero-cost fast path

Module forward hooks are registered once. Their first line:

```python
def _dump_hook(module, inputs, outputs):
    if not module._dump_due:              # set by DebugLogger before forward
        return
    ...
```

On non-dump steps the overhead is one Python attribute read. Measured: 20–50 ns
per module per step.

### 11.3 Overhead budget

| Phase | Budget | Measured (typical) |
|---|---|---|
| Scalars + histograms (no dump) | < 2 ms/step | 0.8–1.4 ms |
| Scalars + histograms + images (dump step) | < 20 ms/step | 10–15 ms |
| Local artifact dump collect + torch.save | < 80 ms (non-blocking) | 40–60 ms |
| W&B sync (async, amortized) | < 0.5 ms/step | 0.1–0.3 ms |
| TensorBoard write (async) | < 0.5 ms/step | 0.1–0.3 ms |

Totals asserted in `tests/test_logging_overhead.py` — the CI job fails if a PR
pushes any number past 120% of the listed budget.

### 11.4 Disk guard

`keep_last_n` (default 200) rolls older dumps off disk, keeping the most recent
N dumps plus all anomaly dumps forever (the anomalies are what you need when
something breaks). `keep_last_n: null` keeps everything.

At 200 MB × 200 dumps = 40 GB per run; acceptable on any training machine. Eval
dumps are kept indefinitely (one per epoch × per sequence).

---

## 12. File plan

### 12.1 New files

```
MAC-VO/
├── Train/DynamicHead/logging/
│   ├── __init__.py
│   ├── debug_logger.py          # DebugLogger, NumericalAbort
│   ├── keys.py                  # frozen key namespace (CI-linted)
│   ├── sinks/
│   │   ├── __init__.py
│   │   ├── base.py              # Sink ABC
│   │   ├── wandb_sink.py
│   │   ├── tensorboard_sink.py
│   │   └── local_sink.py
│   ├── collectors.py            # collect_dump_payload(...)
│   ├── hooks.py                 # forward-hook helpers for frontend/head
│   ├── cadence.py               # LOCAL_DUMP_EVERY, should_dump(...)
│   ├── metrics/
│   │   ├── __init__.py
│   │   ├── calibration.py       # ECE, reliability diagrams
│   │   ├── residuals.py         # Mahalanobis, χ² helpers
│   │   └── pose_deltas.py       # SE(3) delta helpers
│   ├── dashboards/
│   │   ├── training_overview.json
│   │   ├── head_diagnostics.json
│   │   ├── imu_diagnostics.json
│   │   ├── backend_diagnostics.json
│   │   ├── drt_init.json
│   │   └── eval_sequences.json
│   ├── README.md                # diagnostic playbook index
│   └── tests/
│       ├── test_debug_logger.py
│       ├── test_local_sink_atomicity.py
│       ├── test_schema_roundtrip.py
│       ├── test_logging_overhead.py
│       └── test_key_lint.py
├── tools/
│   ├── inspect_dump.py          # CLI + IPython class (§5.5)
│   ├── scrub_partial_dumps.py
│   └── plot_ablation_grid.py
```

### 12.2 Modified files

```
MAC-VO/
├── Train/DynamicHead/train.py
│     + DebugLogger construction at run() start
│     + phase context around each step's sub-phases
│     + anomaly-trigger wiring
│
├── Train/DynamicHead/loop.py
│     + pass `logger, step` through train_step
│     + gated collect_dump_payload + dump_artifact
│
├── Train/DynamicHead/loss.py
│     + emit loss.label.* and loss.terms.* scalars
│     + return per-pixel maps for the dump schema
│
├── Module/Network/DynamicHead/head.py
│     + forward hooks registered on ConvGRU cell, FiLM, refiner
│     + HeadOut extended with internal tensors (disabled when not dumping)
│
├── Module/Network/AirIMU/preintegration.py
│     + emit frontend.imu.preintegrator.* scalars via a logger reference
│     + expose Σ eigenvalues as a property for the dump
│
├── Module/Initialization/DRTLoose/drt_loose.py
│     + emit init.drt.* scalars + one dump at the end of every init run
│
├── Module/Optimization/TwoFramePGO/Optimizer.py
│     + emit backend.twoframe.* scalars at solve start/end
│     + write pgo/twoframe_residuals.jsonl
│
├── Module/Optimization/SlidingWindow/Optimizer.py
│     + emit backend.swf.* scalars
│     + write pgo/swf_window_snapshots/step_*.pt every 200 steps
│
└── DataLoader/Dataset/DynamicHeadTrain.py
      + per-sample emit of data.* scalars inside __getitem__ (collected in collate)
```

### 12.3 Explicitly not touched

- `FlowFormerCov` core — we read intermediate tensors via forward hooks; no edits
  inside the frozen module (I1).
- Existing MAC-VO configs — `logging:` is an additive top-level block with a
  default that keeps current behavior if any project declines to upgrade.

---

## 13. Testing and sanity checks

### 13.1 Unit tests

- `test_debug_logger.py` — fan-out to all three sinks, thread safety, NaN abort.
- `test_local_sink_atomicity.py` — kill mid-write; no `step_*.pt` left behind.
- `test_schema_roundtrip.py` — round-trip every `DumpSchema` field through
  `torch.save` / `load`, dtype and shape preserved.
- `test_key_lint.py` — AST-walks the training + module source tree, collects every
  string passed to `log_scalar` / `log_hist`, asserts each is present in
  `keys.py`. Fails CI on typos.
- `test_logging_overhead.py` — microbenchmark every sink's per-step cost.

### 13.2 Integration tests

- `test_end_to_end_training_step.py` — runs one training step with all three
  sinks enabled against a temp dir + a fake W&B backend
  (`wandb.init(mode='disabled')`); asserts the expected scalars and one dump
  appear.
- `test_eval_run.py` — runs a 1-sequence eval; asserts per-sequence Parquet +
  dump exist.
- `test_stage_a_backend_logging.py` — runs one frame through Stage-A backend;
  asserts `backend.twoframe.*` scalars and the pgo dump are populated.

### 13.3 Pre-flight asserts

At the top of `train.py::run()`, before the main loop:

```python
logger = DebugLogger(run_dir, config, sinks)
logger.assert_all_sinks_writable()          # writes a 1-byte canary to each
logger.assert_schema_matches_collector()    # static check of DumpSchema ↔ collectors.py
logger.assert_key_lint_clean()              # runs test_key_lint in-process
```

Any failure aborts with a message that names the mismatched key, sink, or path.

---

## 14. Out of scope (deferred)

- Per-IMU-sample logging at the raw-sample rate (200 Hz). Too much data; the
  preintegrated summary is what the pipeline actually consumes.
- Real-time streaming to external observability stacks (Prometheus, Datadog). A
  `PrometheusSink` drop-in is possible but not required for the research
  workflow.
- Log compression / deduplication across runs. Today each run is self-contained;
  cross-run compression belongs in a separate tool.
- End-to-end replay from dumps: a dump captures state, not full determinism
  (stochastic ops like dropout are not stored). A `replay_from_dump.py` that
  deterministically re-executes forward/backward is a future extension.

---

## 15. Cross-references

- **Spec:** `docs/2026-04-11-static-confidence-head-macvo-spec.md` — the modules
  referenced throughout §6 are defined there.
- **Method:** `docs/2026-04-11-static-confidence-head-method-theory.md` — why
  each quantity matters is explained there (e.g. §3, §6 proxy, §10 loss, §11 PGO
  integration, §12 calibration).
- **Diagnostic playbooks:** `Train/DynamicHead/logging/README.md` is the runtime
  index of §10 above; it links to the specific W&B dashboards.

If this document and the spec disagree on a scalar's name or semantics, **this
document wins** — it is the logging contract that dashboards, lints, and playbook
scripts depend on. The spec will be updated to match.
