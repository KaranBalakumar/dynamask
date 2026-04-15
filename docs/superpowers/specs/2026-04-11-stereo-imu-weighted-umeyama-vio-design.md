# Stereo+IMU VIO with Unrolled Gauss-Newton and Learned Dynamic Gating

**Target dataset:** VIODE (training + validation only).
**Tuning requirement:** zero per-sequence hyperparameters at inference. Same weights, same constants, any sequence. Matches MAC-VO's deployment story.
**Date:** 2026-04-11.
**Revision note:** this spec is GN-only: soft-gated, matrix-weighted unrolled Gauss-Newton for pair-wise pose, unified with the windowed factor-graph backend. The filename retains the original slug for git-history continuity.

---

## 1. Why this design exists

Monocular VIO systems need bundle adjustment because depth is triangulated from motion — pose and structure co-depend, and the only way to recover both is a joint nonlinear optimization. **Stereo changes this fundamentally.** Per-frame stereo gives metric depth directly, which means the pair-wise ego-motion problem collapses from "joint BA over depth and motion" to **rigid alignment of two 3D point clouds with known correspondences**.

Rigid alignment with known correspondences is a solved problem. In this design we solve it with a **fixed-iteration, matrix-weighted Gauss-Newton** optimizer on $SE(3)$, initialized by AirIMU. The solver is differentiable, deterministic (3 iterations), and numerically bounded by LM damping.

The only remaining job for the neural network is to produce:

1. Good correspondences,
2. A reliable dynamic/static gate (dynamic points violate rigidity and must be excluded), and
3. Per-point 3D uncertainty so matrix weights are calibrated.

Everything in this design follows from that separation.

This is not a tweak of the current DynaMask codebase. It is a ground-up replacement that deletes the differentiable BA pipeline, deletes the pairwise-rank and smoothness losses, deletes the iterative RAFT flow decoder, and replaces all of it with a leaner front-end plus a fixed-depth unrolled GN solver.

---

## 2. Core insight, stated precisely

Given two stereo frames at times $t$ and $t+1$, left-camera intrinsics $K$, and per-frame stereo depth $D_t, D_{t+1}$, any pixel $i$ in frame $t$ with depth $D_t^i$ lifts to a 3D point

$$X_t^i = D_t^i \cdot K^{-1} [u_i, v_i, 1]^\top.$$

If the same physical point appears at pixel $j = i + \delta f_i$ in frame $t+1$, it lifts to

$$X_{t+1}^i = D_{t+1}^{j} \cdot K^{-1} [u_j, v_j, 1]^\top.$$

For a **static** point, the rigid-body constraint says there exists a single $(R, t) \in SE(3)$ such that

$$X_{t+1}^i = R X_t^i + t.$$

Given many such correspondences, with per-point weights $w_i$, we minimize

$$\sum_i w_i \lVert (R X_t^i + t) - X_{t+1}^i \rVert_2^2$$

with a fixed 3-step matrix-weighted Gauss-Newton solver on $SE(3)$ initialized from IMU preintegration. All operations are differentiable in PyTorch, so gradients from a pose loss flow through the solver into the weight-producing heads.

The entire learning problem is therefore "produce correspondences, dynamic gates, and covariances such that the fixed GN solver recovers the true pose." That is a much smaller, tractable problem compared to jointly learning a large iterative BA stack.

---

## 3. Notation

| Symbol | Meaning | Shape |
|---|---|---|
| $I_t^L, I_t^R$ | Left/right stereo images at time $t$ | $[3, H, W]$ |
| $K$ | Left-camera intrinsics | $[3, 3]$ |
| $b$ | Stereo baseline (meters) | scalar |
| $D_t$ | Left-frame depth (meters) | $[1, H, W]$ |
| $\sigma_D$ | Depth std-dev (meters) | $[1, H, W]$ |
| $F_t$ | Feature map at $1/8$ resolution | $[C, H', W']$, $H'=H/8$ |
| $\Delta R_\text{imu}, \Delta v_\text{imu}$ | AirIMU-preintegrated rotation and velocity change, body frame | $[3,3], [3]$ |
| $\Sigma_{\Delta R}$ | Covariance of $\Delta R_\text{imu}$ in tangent space | $[3,3]$ |
| $\delta f_i$ | Residual translational flow at pixel $i$ (post-rotation-warp) | $[2]$ |
| $s_i$ | Dynamic score, $s_i \in [0,1]$ | scalar per pixel |
| $\Sigma_{2D}^i$ | Per-pixel image-plane matching covariance | $[2,2]$ (diagonal) |
| $\Sigma_{3D}^i$ | Per-pixel 3D point covariance | $[3,3]$ (full PSD) |
| $X_t^i, X_{t+1}^i$ | 3D points in left-camera frame | $[3]$ each |
| $R, t$ | Pair-wise relative pose from frame $t$ to $t+1$ | $[3,3], [3]$ |
| $\Sigma_\text{pose}$ | 6-DoF pose covariance on tangent space | $[6,6]$ |
| $R_\text{gt}, t_\text{gt}$ | Ground-truth relative pose from VIODE | $[3,3], [3]$ |

All 3D quantities are in the **left camera frame at time $t$** unless otherwise noted. The left image is the reference throughout.

Resolutions used in this spec: $H = 480, W = 752$ (matches VIODE's native resolution). $H' = 60, W' = 94$ at the $1/8$ feature stride.

---

## 4. System block diagram

```
                ┌───────────────────────────────────────┐
                │ FROZEN: pretrained components          │
                │                                        │
 I_t^L, I_t^R ──► Stereo depth (IGEV-Stereo)  ──► D_t, σ_D^t
I_{t+1}^L,R ──►                                 ──► D_{t+1}, σ_D^{t+1}
                │                                        │
 I_t^L      ──► Feature encoder (RAFT fnet) ──► F_t    │
 I_{t+1}^L  ──►                             ──► F_{t+1} │
                │                                        │
 IMU window ──► AirIMU preintegration       ──► ΔR_imu,│
                │                                ΔV_imu,│
                │                                Σ_ΔR   │
                └───────────────────────────────────────┘
                                    │
                                    ▼
                ┌───────────────────────────────────────┐
                │ LEARNED front-end heads                │
                │                                        │
                │  F_{t+1}_warp = homography_warp(        │
                │      F_{t+1}, H_rot = K·ΔR_imu·K^-1)   │
                │                                        │
                │  CorrVol = correlate(F_t, F_{t+1}_warp)│
                │                                        │
                │  ┌─────────────┐  ┌───────────────┐     │
                │  │ FlowResHead │  │  DynamicHead  │     │
                │  │   → δf      │  │   → s ∈[0,1]  │     │
                │  └─────────────┘  └───────────────┘     │
                │  ┌──────────────────┐                   │
                │  │ UncertaintyHead  │                   │
                │  │   → Σ_2D         │                   │
                │  └──────────────────┘                   │
                └───────────────────────────────────────┘
                                    │
                                    ▼
                ┌───────────────────────────────────────┐
                │ GEOMETRIC lift + matrix precision     │
                │                                        │
                │  1. Lift pixels → X_t, X_{t+1} in 3D  │
                │  2. Propagate Σ_2D + σ_D → Σ_3D (3×3) │
                │  3. Σ_res = Σ_3D^t + Σ_3D^{t+1}       │
                │  4. W_i = (1 - s_i)·(Σ_res + λI)^-1   │
                └───────────────────────────────────────┘
                                    │
                                    ▼
                ┌───────────────────────────────────────┐
                │ PER-PAIR GAUSS-NEWTON (3 unrolled its)│
                │                                        │
                │  Init (R,t) ← (ΔR_imu, Δp_imu)         │
                │  for k in 0..2:                        │
                │    e_i  = (R X_t + t) - X_{t+1}        │
                │    J_i  = [-[R X_t]×, I_3]             │
                │    H    = Σ J_i^T W_i J_i              │
                │    g    = Σ J_i^T W_i e_i              │
                │    δξ   = -(H + μI)^-1 g               │
                │    (R,t)← Exp_SE3(δξ)·(R,t)            │
                │  Σ_pose ← (H + μI)^-1                  │
                └───────────────────────────────────────┘
                                    │
                                    ▼  (train: pose loss via unrolled GN)
                                    ▼  (infer: keyframe edges → backend)
                ┌───────────────────────────────────────┐
                │ WINDOWED FACTOR GRAPH (same solver)    │
                │                                        │
                │  Variables: 10 keyframe SE(3) poses    │
                │  Factors:                              │
                │    • Visual 3D-3D (from per-pair H, g) │
                │    • IMU preintegration (AirIMU)       │
                │    • Prior on oldest-in-window         │
                │  Solver: iSAM2 LM over the window      │
                └───────────────────────────────────────┘
```

---

## 5. Frozen components

All three components below are frozen during training. Their weights come from public checkpoints, and they are never updated. This is the "generalist, known-good" backbone of the sure-shot claim.

### 5.1 Stereo depth: IGEV-Stereo

**Choice:** IGEV-Stereo with SceneFlow+KITTI pretrained weights.

**Alternative (acceptable fallback):** RAFT-Stereo with the same pretraining. Swappable with a one-line change.

**Output:** $D_t \in \mathbb{R}^{1 \times H \times W}$, metric depth in meters, computed from disparity $d_t$ via

$$D_t = \frac{f_x \cdot b}{d_t + \epsilon},$$

where $f_x$ is the left-camera focal length and $b$ is the stereo baseline.

**Disparity uncertainty:** IGEV does not output per-pixel confidence in its standard form, so we use a closed-form analytical model:

$$\sigma_d = \max(0.5\,\text{px},\ 0.05 \cdot d),$$

a fixed-coefficient model that is linear in disparity. The $0.5$ pixel floor is the theoretical Cramer-Rao lower bound for a well-calibrated stereo system; the $5\%$ multiplicative term captures subpixel refinement degradation at large disparities. These two constants are fixed across all sequences — they are not per-dataset knobs.

**Depth uncertainty** follows from the depth–disparity relation:

$$\sigma_D = \left\lvert \frac{\partial D}{\partial d} \right\rvert \sigma_d = \frac{f_x b}{d^2} \sigma_d.$$

This is the standard stereo depth error model and is a hard physical constraint, not a learned quantity.

**Why frozen:** SceneFlow and KITTI pretraining generalize well to synthetic stereo (VIODE is AirSim-based). Training stereo depth from scratch on VIODE alone would be a regression — SceneFlow has vastly more diverse stereo content than VIODE.

### 5.2 Feature encoder: RAFT fnet

**Choice:** The `BasicEncoder` from RAFT (Teed & Deng, ECCV 2020), loaded from the `raft-things.pth` checkpoint.

**Rationale:** This is the same encoder the existing DynaMask codebase already loads (`dynamask_vio/models/backbone.py` has the loader). It produces stable, discriminative features at $1/8$ resolution and is known to generalize to stereo data.

**Configuration:** `output_dim=256`, `norm_fn="instance"`. Shared weights between the two time frames. No FiLM conditioning (see §6.1 for why IMU conditioning happens via the warp, not via FiLM).

**Frozen during all training.** The encoder is not fine-tuned.

### 5.3 IMU preintegration: AirIMU

**Choice:** The existing `IMUEncoder` in `dynamask_vio/models/imu_encoder.py`, which wraps an AirIMU checkpoint.

**Inputs:** Gyroscope and accelerometer samples between timestamps $t$ and $t+1$, plus an availability mask.

**Outputs:**

- $\Delta R_\text{imu} \in SO(3)$: rotation in the body frame from $t$ to $t+1$.
- $\Delta v_\text{imu} \in \mathbb{R}^3$: velocity change.
- $\Sigma_{\Delta R} \in \mathbb{R}^{3 \times 3}$: covariance of $\Delta R$ in tangent space.
- Bias estimates (if available): $b_g, b_a$.

**Frame convention:** AirIMU outputs are in the IMU body frame. We pre-multiply by the IMU→camera extrinsic $T_\text{IC}$ to express $\Delta R_\text{imu}$ in the camera frame:

$$\Delta R_\text{imu}^\text{cam} = R_\text{IC} \cdot \Delta R_\text{imu}^\text{body} \cdot R_\text{IC}^\top.$$

From here on, $\Delta R_\text{imu}$ refers to the camera-frame rotation.

**Frozen during training.**

---

## 6. Learned front-end heads

The learned portion of the system consists of exactly three small heads sitting on top of a single correlation volume. All learned parameters live here. No learned components exist anywhere else in the pipeline.

### 6.1 IMU rotation pre-warp

**Problem being solved:** A pure rotation of the camera induces a global image warp regardless of depth. Removing this component from the feature map before correlation leaves only the translational parallax (depth-dependent) and dynamic-object motion (scene-dependent), dramatically shrinking the correspondence search space.

**Standard trick:** The infinite homography. For a pure rotation $R$, corresponding image points are related by

$$\tilde{x}_{t+1} \sim K R K^{-1} \tilde{x}_t,$$

a $3 \times 3$ homography, independent of depth. This is an exact relation when there is zero translation, and a very good approximation in the short-interval limit because translation-induced parallax is tiny compared to rotation-induced shift at typical frame rates.

**Implementation:**

1. Compute $H_\text{rot} = K \cdot \Delta R_\text{imu} \cdot K^{-1} \in \mathbb{R}^{3 \times 3}$.
2. Build the warped sampling grid: for each output pixel $(u,v)$ in the frame-$t$ coordinate system, compute the corresponding location in $F_{t+1}$ by applying $H_\text{rot}$ (scaled to feature-map resolution).
3. `grid_sample` $F_{t+1}$ with the warped grid to produce $F_{t+1}^\text{warp}$.

**No learned parameters.** This is a pure geometric operation.

**Effect on correspondence search:** the residual flow $\delta f$ after warping has magnitude proportional to translation parallax, which at VIODE's frame rate (typically 20–30 Hz) is $\leq 2$ pixels for most static points at typical depths. The correlation radius can therefore be set to 3 at $1/8$ stride, covering $\pm 24$ pixels in full resolution — plenty for both typical parallax and fast-moving dynamic objects.

**Why this replaces FiLM fusion:** FiLM conditioning asks the network to implicitly learn to invert camera rotation from IMU features. The homography pre-warp does this explicitly, exactly, and with zero parameters. It is strictly better for anything that an IMU rotation actually tells you.

### 6.2 Correlation volume

**Type:** All-pairs correlation, RAFT-style, 4 pyramid levels, local lookup radius 3.

**Input:** $F_t$ and $F_{t+1}^\text{warp}$, both $[B, C, H', W']$ with $C=256$.

**Output per pixel:** a length-$4 \cdot 7^2 = 196$ correlation feature vector (4 levels × $(2r+1)^2$ samples).

**Why radius 3 instead of 4:** Rotation is pre-compensated, so the residual search range is smaller. Shrinking the radius saves compute and tightens the statistics the flow head sees.

**Construction and lookup cost:** $O(BH'W' \cdot H'W')$ for all-pairs build; $O(BH'W' \cdot L \cdot (2r+1)^2)$ for lookup. At $B=8, H'=60, W'=94$ this is ~28M ops — trivial on GPU.

**This is not a learned layer.** Reused from RAFT's `CorrBlock`.

### 6.3 Residual-flow head (`FlowResHead`)

**Purpose:** Predict $\delta f \in \mathbb{R}^{2 \times H' \times W'}$, the residual translational flow in frame-$t$ pixel coordinates (at $1/8$ stride).

**Architecture:**

```
Input:  correlation features [B, 196, H', W']
Conv2d(196, 128, kernel=1)                 # level mixer
ReLU
Conv2d(128, 128, kernel=3, padding=1)      # local context
ReLU
Conv2d(128,  64, kernel=3, padding=1)
ReLU
Conv2d( 64,   2, kernel=1)                 # flow output
```

**Output:** $\delta f \in \mathbb{R}^{2 \times H' \times W'}$, unbounded.

**No iterative refinement.** Single feed-forward pass. No GRU. Rotation-warp + correlation gives enough signal that iterative refinement is not load-bearing for accuracy, and removing it halves the front-end latency and removes a failure surface.

**Approximately 330k parameters.**

### 6.4 Dynamic head (`DynHead`)

**Purpose:** Predict per-pixel dynamic probability $s \in [0,1]^{1 \times H' \times W'}$.

**Architecture:**

```
Input:  concat(correlation features, δf, |δf|)  [B, 196+2+1, H', W']
Conv2d(199, 128, kernel=1)
ReLU
Conv2d(128, 128, kernel=3, padding=1)
ReLU
Conv2d(128,  64, kernel=3, padding=1)
ReLU
Conv2d( 64,   1, kernel=1)
Sigmoid
```

**Input rationale:** Correlation features (matching quality), residual flow (magnitude and direction), and flow norm $|\delta f|$ (the single strongest dynamicness indicator after rotation compensation).

**Output:** $s \in [0,1]^{1 \times H' \times W'}$. Upsampled to full $H \times W$ via bilinear interpolation when used by losses that operate at full resolution.

**Approximately 350k parameters.**

### 6.5 Uncertainty head (`UncertaintyHead`)

**Purpose:** Predict per-pixel image-plane matching covariance $\Sigma_{2D}^i = \text{diag}(\sigma_u^2, \sigma_v^2) \in \mathbb{R}^{2 \times 2}$.

**Architecture:**

```
Input:  concat(correlation features, δf)  [B, 196+2, H', W']
Conv2d(198, 128, kernel=1)
ReLU
Conv2d(128,  64, kernel=3, padding=1)
ReLU
Conv2d( 64,   2, kernel=1)
                            # outputs: (log σ_u², log σ_v²)
Exp
Clamp to [1e-4, 100.0]      # numeric stability
```

**Output:** per-pixel $(\sigma_u^2, \sigma_v^2)$ in units of pixels². Diagonal only — no off-diagonal.

**Why diagonal:** A full $2 \times 2$ Cholesky adds one extra channel and risks numerical instability early in training without a large calibration benefit at the feature resolution. Diagonal is the smallest form that still captures the fact that matching along epipolar lines is often more certain than cross-line.

**Approximately 180k parameters.**

### 6.6 Head parameter budget

| Component | Parameters |
|---|---|
| FlowResHead | ~330k |
| DynHead | ~350k |
| UncertaintyHead | ~180k |
| **Total learned** | **~860k** |

Under one million parameters. The frozen backbone (RAFT fnet, IGEV-Stereo, AirIMU) adds roughly 11M, 13M, and 1.5M parameters respectively, but none of those are updated.

---

## 7. Pair-wise pose estimator (matrix-weighted unrolled Gauss-Newton)

This is the geometric core of the system. It contains no learned parameters. All of its inputs come from the front-end heads and the frozen components above.

### 7.1 Lifting to 3D

For each pixel $i = (u_i, v_i)$ at the feature-map stride:

$$X_t^i = D_t^i \cdot K^{-1} [u_i, v_i, 1]^\top,$$

where $D_t^i$ is the depth at $(u_i, v_i)$ from IGEV-Stereo, upsampled/downsampled to the feature resolution as needed.

For the corresponding point in frame $t+1$:

$$X_{t+1}^i = D_{t+1}^{j_i} \cdot K^{-1} [u_i + \delta u_i, v_i + \delta v_i, 1]^\top,$$

where $j_i = (u_i + \delta u_i, v_i + \delta v_i)$ comes from the predicted residual flow $\delta f_i$ **plus the pixel shift induced by the rotation warp**. Concretely, let $\tilde{u}_i = H_\text{rot} [u_i, v_i, 1]^\top / (H_\text{rot}[u_i,v_i,1]^\top)_3$ be the rotation-warped pixel. The full correspondence is

$$j_i = \tilde{u}_i + \delta f_i.$$

Depth $D_{t+1}$ is sampled at $j_i$ via bilinear interpolation. Points whose $j_i$ lands outside the image are masked out for the rest of this frame pair.

### 7.2 Uncertainty propagation into 3D

The 3D point $X = D \cdot K^{-1}[u,v,1]^\top$ is a function of three inputs: $(u, v, D)$. Its Jacobian is

$$J_\text{lift} = \frac{\partial X}{\partial (u, v, D)} = \begin{bmatrix} D \cdot K^{-1}_{:,0} & D \cdot K^{-1}_{:,1} & K^{-1}[u,v,1]^\top \end{bmatrix} \in \mathbb{R}^{3 \times 3}.$$

The input uncertainty is the block-diagonal combination of the learned matching covariance and the stereo depth covariance:

$$\Sigma_\text{in} = \begin{bmatrix} \Sigma_{2D} & 0 \\ 0 & \sigma_D^2 \end{bmatrix} \in \mathbb{R}^{3 \times 3}.$$

First-order propagation gives

$$\Sigma_{3D} = J_\text{lift} \cdot \Sigma_\text{in} \cdot J_\text{lift}^\top \in \mathbb{R}^{3 \times 3}.$$

**Accuracy note:** $X$ is linear in each input when the others are fixed, so the first-order propagation is exact for small cross-terms. With $\sigma_u, \sigma_v \leq 5$ px and $\sigma_D \leq 10\%$ of $D$, the first-order approximation is tight.

Both endpoints have uncertainty: $X_t^i$ has covariance $\Sigma_{3D}^{t,i}$, and $X_{t+1}^i$ has covariance $\Sigma_{3D}^{t+1,i}$. The total 3D residual covariance for the pair is

$$\Sigma_{\text{res}}^i = \Sigma_{3D}^{t,i} + \Sigma_{3D}^{t+1,i}.$$

This is the "total least squares" form — it does not pretend one endpoint is exact.

### 7.3 Soft weight construction

Each pixel contributes a **matrix-valued** information $W_i \in \mathbb{R}^{3 \times 3}$ that combines a soft static gate with the 3D residual covariance:

$$W_i = (1 - s_i) \cdot (\Sigma_\text{res}^i + \lambda I_3)^{-1}.$$

- $(1 - s_i)$ is the soft static gate. No dead zone. No threshold. A pixel with $s_i = 0$ contributes full precision; a pixel with $s_i = 1$ contributes zero. Everything in between scales linearly. The dynamic head therefore has continuous gradient everywhere, not a flat region past a hard cutoff.
- $(\Sigma_\text{res}^i + \lambda I_3)^{-1}$ is the full matrix precision from §7.2, with $\lambda = 10^{-4}\,\text{m}^2$ as a minimum-eigenvalue floor so the inverse stays well-conditioned even when a pixel's covariance has collapsed during training.

**No $\tau_s$.** Removing the hard gate removes one knob from the spec and eliminates the gradient discontinuity the dynamic head would otherwise see.

**Why matrix, not scalar:** the uncertainty head produces a full $3 \times 3$ $\Sigma_\text{res}^i$ — one direction in 3D can be more certain than another (e.g., depth vs. lateral for a far-away pixel). Collapsing that to a scalar via $\text{trace}$ throws away exactly the information the head is learning to produce. The Gauss-Newton solver in §7.4 consumes $W_i$ natively, so nothing is lost.

**All constants fixed across all sequences.** $\lambda$ is the only constant in this subsection and is a numerical stability floor, not a tuning knob.

### 7.4 Iterative weighted Gauss-Newton pose estimator

Given point pairs $\{(X_t^i, X_{t+1}^i)\}$ and per-pixel precisions $\{W_i\}$, the pair-wise pose $(R, t)$ is found by minimizing the matrix-weighted 3D residual

$$\mathcal{E}(R, t) = \sum_i e_i^\top W_i e_i, \qquad e_i = (R X_t^i + t) - X_{t+1}^i,$$

via a fixed number of unrolled Gauss-Newton iterations. No convergence check, no iteration-count knob, no LM damping schedule — just three iterations.

**Initialization** comes directly from AirIMU preintegration:

$$R^{(0)} = \Delta R_\text{imu}^\text{cam}, \qquad t^{(0)} = \Delta p_\text{imu}^\text{cam},$$

where $\Delta p_\text{imu}$ is obtained by integrating $\Delta v_\text{imu}$ over the inter-frame interval $\Delta t$ (first-order: $\Delta p = \Delta v \cdot \Delta t$, sufficient for the sub-100 ms gap between consecutive VIODE frames). Rotation init is essentially perfect ($<0.1°$ error); translation init is coarse but close enough that GN converges in 2–3 steps.

**Iteration $k \to k+1$:**

1. **Residuals** at current iterate:
   $$e_i^{(k)} = (R^{(k)} X_t^i + t^{(k)}) - X_{t+1}^i.$$

2. **Jacobians** in the $\mathfrak{se}(3)$ tangent space at the current iterate:
   $$J_i^{(k)} = \frac{\partial e_i}{\partial \xi}\bigg|_{(R^{(k)}, t^{(k)})} = \begin{bmatrix} -\left[R^{(k)} X_t^i\right]_\times & I_3 \end{bmatrix} \in \mathbb{R}^{3 \times 6}.$$

3. **Hessian and gradient** of the weighted cost:
   $$H^{(k)} = \sum_i (J_i^{(k)})^\top W_i J_i^{(k)} \in \mathbb{R}^{6 \times 6},$$
   $$g^{(k)} = \sum_i (J_i^{(k)})^\top W_i e_i^{(k)} \in \mathbb{R}^{6}.$$

4. **Levenberg-damped step** (tiny fixed damping):
   $$\delta \xi^{(k)} = -\bigl(H^{(k)} + \mu I_6\bigr)^{-1} g^{(k)}, \qquad \mu = 10^{-4}.$$
   The $\mu$ is a numerical cushion against near-singular $H$ in the early iterations; it is not adaptive.

5. **Retract onto $SE(3)$:**
   $$\begin{bmatrix} R^{(k+1)} \\ t^{(k+1)} \end{bmatrix} = \text{Exp}_{SE(3)}(\delta \xi^{(k)}) \cdot \begin{bmatrix} R^{(k)} \\ t^{(k)} \end{bmatrix},$$
   where $\text{Exp}_{SE(3)}$ is the standard matrix exponential on $\mathfrak{se}(3)$ (a left-multiplication update in the minimal-perturbation convention).

Three iterations. The final $(R^{(3)}, t^{(3)})$ is the output. All steps are PyTorch-native and differentiable — the full unrolled GN is a composition of matrix multiplies, one $6 \times 6$ inverse per iteration, and an $SE(3)$ exponential per iteration. Gradients flow from the pose loss through every iterate, through $W_i$, through the three heads.

**Numerics:** Hessian assembly and the $6 \times 6$ inverse run in float32 even under a bfloat16 training policy. The $SE(3)$ exponential uses the standard closed-form Rodrigues expansion with a Taylor-series fallback for small angles (thresholded at $\|\omega\| < 10^{-6}$).

**Why fixed three iterations (not convergence-based):** a fixed schedule is simpler to unroll for backprop, avoids a hyperparameter (convergence tolerance), and is more than enough for an over-determined 3D residual problem with $\sim$5600 point pairs (all valid pixels at $1/8$ stride) and a tight IMU init. Adaptive iteration counts are strictly worse for differentiable training because the unroll depth becomes data-dependent.

**No rank-deficiency fallback needed.** The $\mu I_6$ damping already keeps $H + \mu I$ well-conditioned. If the pair is genuinely degenerate (e.g., all pixels lost because of severe occlusion), the GN will just return something close to $(R^{(0)}, t^{(0)}) = (\Delta R_\text{imu}, \Delta p_\text{imu})$, which is exactly the graceful-degradation behavior we want.

### 7.5 Pose covariance

The 6-DoF pose covariance falls out of the final GN iterate for free:

$$\Sigma_\text{pose} = \bigl(H^{(3)} + \mu I_6\bigr)^{-1}.$$

No separate computation, no inconsistency with the solver's weighting, no extra inverse. $H^{(3)}$ is already the matrix-weighted information at the solution, which is the correct definition of pose uncertainty in the Gauss-Newton setting. This $\Sigma_\text{pose}$ feeds directly into the backend's visual factor in §8.

---

## 8. Windowed factor-graph backend (inference only)

The backend is a direct scaling-up of the per-pair Gauss-Newton machinery in §7.4 to multiple keyframes at once. **Same residuals, same Jacobians, same matrix-weighted cost, same LM-damped GN solver — just more variables.** This is the MAC-VO architectural pattern: one factor graph, one solver, one set of per-point precisions, applied consistently from the pair level up to the window level.

**Library:** GTSAM with its Python bindings (`gtsam` PyPI package). iSAM2 is the incremental smoother. Fixed for v1.

**Keyframe selection.** Insert a new keyframe whenever any of the following holds since the last keyframe:

- Translation norm $> 0.15$ m
- Rotation angle $> 5°$
- Time elapsed $> 0.5$ s

All three are global constants, not per-sequence. On non-keyframe frames we still run the §7.4 per-pair GN to get a relative pose for visualization, but we do not add it to the factor graph.

**Variables in the graph:**

- $\mathbf{T}_k \in SE(3)$ — pose of keyframe $k$ in world frame, one per keyframe in the active window (size 10).
- (Optional, v2) $\mathbf{v}_k, \mathbf{b}_k$ — body velocity and IMU biases, added when we extend to full VINS-Mono-style factor graphs. Not included in v1.

**Factors:**

1. **Visual 3D point-to-point factor** between each consecutive keyframe pair $(k, k+1)$. This is **not** a single $(R, t, \Sigma)$ summary — it is the full per-pixel residual from §7.4 preserved as a multi-point factor with precisions $\{W_i\}$. When linearized at the current $\mathbf{T}_k, \mathbf{T}_{k+1}$ iterate, its Hessian contribution is
   $$\sum_i J_{k,k+1}^i{}^\top W_i J_{k,k+1}^i,$$
   where $J_{k,k+1}^i$ is the relative-pose Jacobian from §7.4. In practice we either:
   - **(a)** attach the full point cloud to the factor (exact, higher memory), or
   - **(b)** pre-summarize into a $12 \times 12$ block Hessian + $12$-vector gradient (the relative-pose normal equations at the point of factor insertion) and attach that. This is a first-order approximation but eliminates per-pixel storage in the backend.
  
   v1 uses **(b)** for memory efficiency. The approximation is negligible because the solver re-linearizes so rarely that the per-pixel residual barely changes between updates.

2. **IMU preintegration factor** between each consecutive keyframe pair, from AirIMU: a full SO(3) rotation constraint $\Delta R_\text{imu}$ with covariance $\Sigma_{\Delta R}$, and a translation constraint $\Delta p_\text{imu}$ with $\Sigma_{\Delta p}$ (derived from $\Sigma_{\Delta v} \cdot \Delta t^2$). In GTSAM this is a `BetweenFactor<Pose3>` with a matrix noise model on the 6-DoF tangent space.

3. **Prior factor** on the oldest keyframe in the window to anchor the gauge. A tight Gaussian prior fixes the absolute pose of the oldest kept keyframe; the rest of the window is optimized relative to it. As the window slides, the gauge-anchor migrates forward.

**Solver:** iSAM2 with relinearization threshold $0.1$ rad / $0.1$ m (fixed, not tuned). On every keyframe insertion, iSAM2 performs a bounded number of LM iterations over the affected cliques. Runtime per update is bounded by the window size.

**Window management:** A strict sliding window of 10 keyframes. Once a keyframe falls out of the window, its factors are marginalized (GTSAM handles this natively via iSAM2) and it is removed from the active variable set. The marginalized information becomes a prior on the new "oldest" keyframe.

**No loop closure in v1.** VIODE sequences are short enough that loop closure is not required to get competitive ATE. v2 extension point.

**Output:** the optimized pose of the newest keyframe on every backend tick; intra-keyframe poses are composed from per-pair GN results against the newest keyframe pose.

**Unifying observation:** the per-pair GN of §7.4 is literally a factor graph with two nodes and one visual factor. The backend of §8 is the same thing with $N$ nodes and $2(N-1)$ factors (visual + IMU). Nothing in the front-end needs to know which regime it is running in. This is the "MAC-VO consistency" the original design was missing.

---

## 9. Tuning-free design (the MAC-VO-style deployment property)

The claim "no per-sequence tuning" requires every hyperparameter to be either:

(a) **Learned from training**, so it is fixed in the checkpoint weights, or
(b) **A global constant** set once and never touched per-sequence, or
(c) **Derived from the current frame** using a rule that is domain-invariant.

Here is the exhaustive list:

| Parameter | Type | Value |
|---|---|---|
| Feature resolution stride | (b) | 8 |
| Correlation pyramid levels | (b) | 4 |
| Correlation radius | (b) | 3 |
| $\lambda$ (matrix-precision floor, m²) | (b) | $10^{-4}$ |
| $\mu$ (LM damping) | (b) | $10^{-4}$ |
| Per-pair GN iterations | (b) | 3 |
| Keyframe translation threshold (m) | (b) | 0.15 |
| Keyframe rotation threshold (rad) | (b) | $5\pi/180$ |
| Keyframe time threshold (s) | (b) | 0.5 |
| Stereo disparity floor (px) | (b) | 0.5 |
| Stereo disparity slope | (b) | 0.05 |
| Backend window size (keyframes) | (b) | 10 |
| iSAM2 relinearization threshold | (b) | 0.1 rad / 0.1 m |
| $\sigma_u^2, \sigma_v^2$ (image covariance) | (a) | learned |
| $s_i$ (dynamic score) | (a) | learned |
| $\delta f_i$ (residual flow) | (a) | learned |
| $\tau_r$ (dynamic label threshold, m) | (c) | per-batch median of 3D residuals, floored at 0.05 m |
| $\kappa_r$ (dynamic label softness, m) | (c) | $0.5 \cdot \tau_r$ |

**Dropped from earlier drafts** (and not replaced): $\tau_s$ hard-gate threshold and scalar trace-based weight floor $\epsilon$. Both are removed because the soft-gated matrix-weighted GN does not need them.

Nothing in the inference pipeline is computed from per-sequence statistics. Nothing depends on knowing the scene type ("indoor" vs "outdoor", "slow" vs "fast"). The same checkpoint, the same constants, the same solver apply to every sequence.

**The reason this is sufficient:** Σ_3D is learned in **absolute metric units** (meters²), because the L_cov loss (§10.2) supervises the Mahalanobis distance against the actual 3D residual in meters. That means a trained Σ_3D is directly consumable by the solver without rescaling. This is the MAC-VO property.

---

## 10. Loss functions

All losses are computed at the $1/8$ feature-map resolution unless otherwise noted. This matches where the learned heads produce their outputs.

### 10.1 L_dyn — dynamic-mask BCE against 3D-residual labels

**Target construction.** For each pixel $i$ and each frame pair in the batch:

1. Use $(R_\text{gt}, t_\text{gt})$ from VIODE to compute the 3D residual assuming the pixel is static:
   $$r_i = \lVert (R_\text{gt} X_t^i + t_\text{gt}) - X_{t+1}^i \rVert_2.$$
   Note: $X_{t+1}^i$ here uses the **predicted correspondence** (via $\delta f_i$ and the rotation warp), not a GT correspondence. This is the residual that the solver will actually see. In early training this residual is dominated by bad correspondences; as $\delta f$ converges, it becomes dominated by true dynamic motion.

2. Compute the per-batch threshold self-calibratingly:
   $$\tau_r = \max(0.05\,\text{m},\ \text{median}_i(r_i)).$$
   The 0.05 m floor prevents collapse on all-static batches. Median is used instead of mean for outlier robustness.

3. Soft label:
   $$d_i = \sigma\left(\frac{r_i - \tau_r}{0.5 \tau_r}\right) \in (0, 1).$$

**Loss:**

$$\mathcal{L}_\text{dyn} = \frac{1}{N} \sum_i \text{BCE}(s_i, d_i),$$

where $N$ is the number of valid pixels (those whose $j_i$ lands inside the image and whose depth is valid).

**Critical property:** the label is derived from the network's own predicted correspondences. This creates a self-consistency loop: as $\delta f$ improves, $r_i$ better reflects true rigid-body violation, and $d_i$ becomes a cleaner target. Early training is noisier, but this is stabilized by the L_pose loss (§10.3) which shapes $\delta f$ directly.

**No proxy-mask warmup:** this spec assumes no ground-truth dynamic labels are available, so `L_dyn` is trained purely from 3D residual-derived soft targets.

### 10.2 L_cov — MAC-VO-style NLL on 3D residual (static pixels only)

**Static set:** $S = \{i : r_i < \tau_r\}$, where $r_i$ and $\tau_r$ are defined as in §10.1.

**Residual:** same $e_i = (R_\text{gt} X_t^i + t_\text{gt}) - X_{t+1}^i$.

**Loss:**

$$\mathcal{L}_\text{cov} = \frac{1}{|S|} \sum_{i \in S} \left[ e_i^\top (\Sigma_\text{res}^i)^{-1} e_i + \log \det \Sigma_\text{res}^i \right].$$

This is the standard Gaussian NLL. The Mahalanobis term forces $\Sigma$ to be large enough to explain the observed residual. The log-determinant term prevents $\Sigma$ from collapsing to zero. Together they calibrate $\Sigma_\text{res}$ to the true residual statistics at convergence, and because the residual is in meters, $\Sigma_\text{res}$ is in meter² — **absolute units**, deployment-ready.

**Important:** this loss is applied **only** to static pixels as identified by the GT-motion residual itself. Dynamic pixels are excluded, so the uncertainty head is never asked to explain object motion — it is only asked to calibrate the stereo matching + depth noise for rigid-body-valid correspondences.

**Clamping:** Both $\Sigma$ and the log-determinant are numerically stabilized. A minimum eigenvalue of $10^{-4}\,\text{m}^2$ is enforced on $\Sigma_\text{res}$ before inversion. The $\log\det$ is clamped to a range of $[-30, 30]$ to prevent gradient blow-up at initialization.

### 10.3 L_pose — direct SE(3) loss on the unrolled GN output

This is the gradient source that actually teaches every learned head to produce pose-useful outputs. It is the primary signal.

**Inputs:** $(R, t)$ from the unrolled GN solver, $(R_\text{gt}, t_\text{gt})$ from VIODE.

**Rotation error:**

$$\ell_R = \text{huber}\left( \lVert \log(R^\top R_\text{gt}) \rVert_2,\ \delta_R = 0.05\,\text{rad} \right),$$

where $\log$ is the $SO(3)$ logarithm (returns the rotation tangent vector).

**Translation error:**

$$\ell_t = \text{huber}\left( \lVert t - t_\text{gt} \rVert_2,\ \delta_t = 0.1\,\text{m} \right).$$

**Combined:**

$$\mathcal{L}_\text{pose} = 10 \cdot \ell_R + \ell_t.$$

The 10× rotation weight matches the rotation/translation error scale at VIODE's typical motion magnitudes (rad vs m). It is fixed.

**Gradient flow:** via differentiable unrolled GN, this loss shapes every upstream head:

- $\delta f$ → changes $X_{t+1}$ → changes $H$ → changes $(R, t)$.
- $s$ → changes weights → changes $H$.
- $\Sigma_{2D}$ → changes weights → changes $H$.

All three heads receive gradient from this single term. It is the *only* term that directly ties pose quality to head outputs; the other two losses are auxiliary shapers.

### 10.4 Total loss

$$\mathcal{L} = \lambda_\text{pose} \mathcal{L}_\text{pose} + \lambda_\text{dyn} \mathcal{L}_\text{dyn} + \lambda_\text{cov} \mathcal{L}_\text{cov}$$

with fixed weights:

- $\lambda_\text{pose} = 1.0$
- $\lambda_\text{dyn} = 1.0$
- $\lambda_\text{cov} = 0.5$

**Ablatability.** Setting any of $\lambda$ to zero disables the corresponding loss while leaving the other two functional. No cross-dependency makes a loss required for another to converge. This is the sure-shot property: if anything diverges in training, you can identify the culprit by disabling one term at a time.

---

## 11. Training setup (VIODE only)

### 11.1 Dataset: VIODE only

**No TartanAir. No EuRoC. No mixed-dataset training.** Only VIODE's train and validation splits are used. This is an explicit constraint.

**Why VIODE alone is sufficient:**

- VIODE has aligned stereo, IMU, GT poses, and segmentation masks for dynamic objects — everything the losses need.
- Multi-environment coverage (city day, city night, parking lot) provides enough scene diversity for the small learned head budget (~860k params).
- All generalist backbones (stereo, features, IMU) are pretrained on broader data, so the learned heads specialize to VIODE while the frozen parts carry cross-dataset generalization.

### 11.2 Splits

Use VIODE's official train and validation sequences as specified in the existing `dynamask_vio/configs/viode.yaml`. Do not re-split. Train on train, report on validation. No test leakage.

Specifically:

- **Train sequences:** whatever `splits.train` contains in the current `viode.yaml`. These are used for gradient updates.
- **Val sequences:** whatever `splits.val` contains. These are used for ATE/RPE and calibration reporting, but never for training or early-stopping criteria that would leak into model selection.

### 11.3 Data pipeline (what `VIODEDataset` must return)

The existing `VIODEDataset.__getitem__` returns most of what we need but must be extended:

**Already returned:**

- `img_prev`, `img_curr` — stereo left images at $t$ and $t+1$.
- `gt_R`, `gt_p` — relative pose from $t$ to $t+1$.
- `intrinsics` — $K$.
- `imu_window`, `imu_mask` — IMU samples and validity.

**To be added:**

- `img_prev_right`, `img_curr_right` — right stereo images for on-the-fly depth computation. (Alternative: precomputed depth in HDF5.)
- `depth_prev`, `depth_curr` — metric depth $[1, H, W]$, precomputed at HDF5 build time with a frozen IGEV-Stereo forward pass, then stored as float16 alongside the existing image fields. On-the-fly computation is explicitly **not** supported for v1 — the 15 ms per pair would dominate training throughput.
- `disparity_prev`, `disparity_curr` — disparity in pixels (for the analytic $\sigma_d$ model).
- `baseline` — stereo baseline $b$ in meters.

**Image size:** VIODE native resolution, no cropping. Cropping would require intrinsics updates and is not worth the code complexity for a dataset with fixed resolution.

### 11.4 Optimizer and schedule

- **Optimizer:** AdamW, $\beta = (0.9, 0.999)$, weight_decay $=10^{-4}$.
- **Base learning rate:** $2 \times 10^{-4}$ for the learned heads.
- **Encoder LR:** $0$ (frozen). Depth and IMU: $0$ (frozen).
- **Schedule:** `OneCycleLR`, max_lr $=2 \times 10^{-4}$, `pct_start=0.1`, `anneal_strategy="cos"`, `cycle_momentum=False`.
- **Total steps:** 80,000. VIODE is small enough that this is roughly 40 epochs at batch size 8.
- **Gradient clipping:** 1.0 on the global gradient norm.
- **Mixed precision:** bfloat16 everywhere except GN Hessian assembly/solve and $SE(3)$ update, which run in float32.

### 11.5 Batch composition

- **Batch size:** 8 frame pairs.
- **Workers:** 8 (matches existing codebase).
- **Shuffle:** enabled on train, disabled on val.

### 11.6 Augmentation

- **Horizontal flip:** 50% probability. When flipped, update `gt_p[0]`, `K[0,2]`, and the sign of gyro $\omega_y, \omega_z$ and accel $a_x$ as appropriate for the VIODE body frame. (The existing codebase has flip logic — reuse it.)
- **Photometric jitter:** brightness $\pm 0.2$, contrast $\pm 0.2$, gamma $\pm 0.1$. Applied identically to both stereo pair members.
- **No cropping, no rescaling, no depth perturbation, no random intrinsic modification.**
- **IMU noise injection:** add zero-mean Gaussian with $\sigma_\omega = 10^{-3}$ rad/s, $\sigma_a = 10^{-2}$ m/s² to the raw IMU samples during training. This matches the deployment-time noise profile and prevents the model from overfitting to VIODE's synthetic clean IMU.

### 11.7 Validation

- Validate every epoch on the full VIODE val split.
- Primary metric: ATE (per-sequence RMSE, then averaged).
- Secondary: RPE @ 1 frame, RPE @ 10 frames, and dynamicness self-consistency (residual-separation score using $r_i$ vs predicted $s_i$).
- Covariance calibration diagnostic: mean Mahalanobis² per static pixel; target is ~3 at convergence (chi² with 3 DoF).
- Checkpoint selection: min validation ATE.

### 11.8 Loggable diagnostics

Per training step:

- `loss_pose`, `loss_dyn`, `loss_cov`, `loss_total`
- $\tau_r$ (the adaptive threshold — should stabilize)
- Fraction of pixels classified as dynamic under the current threshold
- GN Hessian condition number $\kappa(H+\mu I)$
- GN step norm $\lVert \delta\xi \rVert_2$ and valid-pixel count
- Mean $|\delta f|$
- Per-pixel mean $\text{trace}(\Sigma_{3D})$

---

## 12. Inference pipeline

Inference is strictly feed-forward through the front-end, then unrolled GN, then backend. No learning. No per-sequence tuning.

**Per frame pair (online):**

1. Forward the frozen stereo depth: $D_t$, $\sigma_d$.
2. Forward AirIMU: $\Delta R_\text{imu}, \Delta v_\text{imu}$.
3. Forward the feature encoder: $F_t, F_{t+1}$.
4. Compute $H_\text{rot} = K \Delta R_\text{imu} K^{-1}$, warp $F_{t+1}$.
5. Build correlation volume.
6. Forward the three heads: $\delta f, s, \Sigma_{2D}$.
7. Lift to 3D, propagate uncertainty.
8. Build weights.
9. Unrolled GN (3 fixed iterations): $(R, t, \Sigma_\text{pose})$.
10. Send to backend as a new visual edge.

**Keyframe backend (GTSAM):**

11. If keyframe criterion met, append a new node and the above edge.
12. Add the IMU rotation edge with $\Sigma_{\Delta R}$.
13. Run iSAM2 update over the last 10 keyframes.
14. Emit optimized pose for the newest keyframe.

**Non-keyframe poses** are produced by composing the newest keyframe pose with the intra-keyframe visual+IMU deltas.

**Runtime target:** ≤ 30 ms front-end + ≤ 3 ms pair-wise GN + ≤ 10 ms backend per frame on an RTX 3090. The dominant cost is stereo depth (~15 ms) and IGEV-Stereo is compilable to TensorRT for deployment if latency becomes critical.

---

## 13. Evaluation protocol

### 13.1 Metrics

**Trajectory accuracy:**

- **ATE** (RMSE of translation after rigid alignment, per sequence, averaged).
- **RPE translation** at 1-frame lag and 10-frame lag.
- **RPE rotation** at 1-frame lag and 10-frame lag.

**Dynamicness quality (self-consistency, no label dependency):**

- **Residual-separation score:** $\mathbb{E}[r_i \mid s_i \in \text{top }20\%] - \mathbb{E}[r_i \mid s_i \in \text{bottom }20\%]$ (higher is better).
- **Pseudo-label AP:** AP of $s_i$ against residual-derived pseudo-labels $\mathbf{1}[r_i > \tau_r]$ (reported as self-consistency, not GT segmentation quality).

**Covariance calibration:**

- **Mean squared Mahalanobis distance** $\bar{m}^2 = \text{mean}_i (e_i^\top \Sigma_{\text{res},i}^{-1} e_i)$ over val-set static pixels. Target is ~3 (the expected value of a chi² with 3 DoF).
- **Reliability diagram:** plot empirical residual quantiles against $\Sigma$-predicted quantiles.

**Runtime:**

- Mean front-end latency (ms).
- Mean solver latency (ms).
- Mean backend latency (ms).

### 13.2 Baselines to compare against

- **Stereo-only:** turn off IMU (set $\Delta R_\text{imu} = I$, skip rotation pre-warp, drop IMU edges from backend).
- **VINS-Fusion** stereo+IMU (classical baseline).
- **Current DynaMask V2.5** (the pipeline we are replacing) on the same VIODE val split.
- **MAC-VO** if a VIODE run is available.

---

## 14. Ablation plan

All ablations train from scratch on the same VIODE train split with the same 80k steps:

| ID | Configuration | Purpose |
|---|---|---|
| A0 | Full system | Reference |
| A1 | $\lambda_\text{dyn} = 0$ | Measure value of dynamic supervision |
| A2 | $\lambda_\text{cov} = 0$ | Measure value of uncertainty calibration |
| A3 | $\lambda_\text{pose} = 0$ | Confirm pose supervision is load-bearing |
| A4 | No IMU rotation pre-warp | Measure warp vs FiLM / vs nothing |
| A5 | $\Sigma_{3D} = I$ (no uncertainty propagation) | Measure contribution of 3D cov formulation |
| A6 | Replace soft gate $(1-s_i)$ with hard binary gate $\mathbf{1}[s_i < 0.5]$ | Measure soft-gating benefit |
| A7 | No right-stereo depth (use monocular self-supervised depth) | Confirm stereo is essential |
| A8 | Train on half of VIODE | Data-efficiency check |

A0 is the primary headline number. A1–A3 prove each loss is load-bearing. A4–A6 prove the novel mechanisms pay off.

---

## 15. Failure modes and mitigations

**1. GN divergence from bad initialization.** Mitigated by AirIMU rotation init (typically <0.1° error over <50 ms gap) and translation init from $\Delta v_\text{imu} \cdot \Delta t$. LM damping $\mu = 10^{-4}$ keeps $H + \mu I$ well-conditioned even in the first iteration.

**2. Ill-conditioned Hessian (e.g. all points nearly coplanar after soft gating).** The $\mu I_6$ damping plus the $\lambda I_3$ floor on per-pixel precision together guarantee $H + \mu I$ is strictly positive definite. In the worst case the GN step degenerates toward the IMU init, which is exactly the graceful fallback behavior we want.

**3. Dynamic-head collapse to all-static.** Early training risk before $L_\text{dyn}$ signal is clean. Mitigated by robust residual labels ($\tau_r$ floor) and a linear ramp of $\lambda_\text{dyn}$ from 0 to 1 over the first 5k steps.

**4. Dynamic-head collapse to all-dynamic.** Would zero out all weights and produce near-singular $H$. Mitigated by $L_\text{pose}$ — all-dynamic means $W_i \to 0$, $H \to \mu I$, GN can barely move, pose stays at $(\Delta R_\text{imu}, \Delta p_\text{imu})$, which is wrong but not catastrophically so; the resulting pose loss backpropagates into reducing $s$.

**5. $\Sigma$ collapses to zero.** Penalized by the $\log \det$ term in $L_\text{cov}$. The per-pixel $\lambda I_3 = 10^{-4}\,\text{m}^2$ floor prevents numerical blow-up when computing $(\Sigma_\text{res} + \lambda I_3)^{-1}$.

**6. $\Sigma$ inflates to infinity.** Penalized by the Mahalanobis term in $L_\text{cov}$ (infinite $\Sigma$ cannot explain finite residuals). Equilibrium is the calibrated state.

**7. Stereo depth errors on textureless regions.** Handled by the disparity-dependent $\sigma_D$ model — low-texture regions get high $\sigma_D$, high $\Sigma_\text{res}$, low $W_i$, no biasing. Automatic.

**8. Dynamic object with the exact same motion as the camera.** Unavoidable from a single stereo pair without temporal context. Mitigated at the backend by the IMU preintegration factor, which disagrees with a dynamic-biased visual factor and pulls the optimization back toward the correct pose.

**9. Rolling shutter.** VIODE is synthetic global-shutter, so this does not apply. Flag as a known limitation for real-data deployment.

**10. Long-duration drift.** Addressed by the backend's IMU factors. Loop closure is a v2 extension.

**11. Timestamp/extrinsic calibration error between camera and IMU.** VIODE is perfectly calibrated. For deployment, this would become a real issue; out of scope for v1.

---

## 16. Implementation order

Phase 1 — **Data pipeline** (no model code yet):

1. Extend `VIODEDataset` to expose `depth_prev`, `depth_curr`, `disparity_prev`, `disparity_curr`, `baseline`, and the right-stereo images.
2. Precompute depth with a frozen IGEV-Stereo and store in HDF5 to avoid per-step compute.
3. Write a standalone test that loads a batch and asserts all new fields have the right shapes and finite values.

Phase 2 — **Geometric utilities** (no learning yet):

4. `rigid_flow_utils.py`: 3D lifting, rotation-homography warp, uncertainty propagation via the lifting Jacobian.
5. `se3_gn.py`: differentiable three-iteration weighted Gauss-Newton on $SE(3)$, with matrix-valued per-point precisions, LM damping, and the closed-form $SE(3)$ exponential.
6. Unit-test with synthetic $(R, t, \text{point clouds}, W_i)$: verify recovery is within $10^{-3}$ rad / $10^{-3}$ m of ground truth when init is the identity plus noise, and exact within float tolerance when init is already the solution.

Phase 3 — **Model heads**:

7. `flow_res_head.py`, `dynamic_head.py`, `uncertainty_head.py`.
8. `stereo_imu_vio_model.py`: wires up frozen backbones, rotation pre-warp, correlation, heads, 3D lifting with matrix uncertainty, and the unrolled GN pose estimator.
9. Forward-pass test on a real VIODE batch: shapes, finite values, non-trivial outputs, finite loss.

Phase 4 — **Losses**:

10. `losses/dyn_bce.py`, `losses/cov3d_nll.py`, `losses/pose_se3.py`.
11. Loss unit tests on controlled synthetic batches.

Phase 5 — **Training loop**:

12. New Lightning module that replaces `DynaMaskLitModule`. No BA. No rank/smoothness losses.
13. New config `viode_gn.yaml`.
14. 1k-step smoke run: confirm losses decrease, no NaNs, no GN linear-solve failures.
15. Full 80k training run.

Phase 6 — **Backend and evaluation**:

16. GTSAM-based pose-graph backend in `backends/gtsam_backend.py`.
17. Evaluation script that consumes a trained checkpoint, runs inference+backend on val sequences, and computes ATE/RPE/dynamicness self-consistency/calibration.

Each phase has clear unit tests and a clear gate before moving on. Phases 1–5 are pure training; phase 6 is inference-only and can be done in parallel with the final training run.

---

## 17. Deliverables

**Code:**

- `dynamask_vio/data/viode_dataset.py` — extended with the new fields.
- `dynamask_vio/models/stereo_imu_vio.py` — new unified model class.
- `dynamask_vio/models/heads/` — three head modules.
- `dynamask_vio/solvers/se3_gn.py` — unrolled matrix-weighted GN solver.
- `dynamask_vio/utils/rigid_flow.py` — lifting, warping, uncertainty propagation.
- `dynamask_vio/losses/dyn_bce.py`, `cov3d_nll.py`, `pose_se3.py`.
- `dynamask_vio/train_gn.py` — new training entry point (separate from the existing `train.py` so the two pipelines coexist until the new one is validated).
- `dynamask_vio/configs/viode_gn.yaml`.
- `dynamask_vio/backends/gtsam_backend.py`.

**Artifacts:**

- A trained checkpoint on VIODE train.
- A validation report with ATE/RPE/dynamicness self-consistency/calibration numbers on VIODE val.
- Ablation table (A0–A8).
- Runtime breakdown on the target GPU.

**Documentation:**

- This spec.
- A short README in `dynamask_vio/models/stereo_imu_vio.py` pointing back to this spec.
- A migration note explaining that the BA pipeline (`differentiable_ba.py`) and the pairwise-rank / smoothness losses are not used by this new system and can be deprecated once validated.

---

## 18. Scope boundary (what this spec does not cover)

The following are explicitly out of scope for v1 and should not creep in during implementation:

- **Any iterative flow refinement** (RAFT-style GRU updates). Single feed-forward only.
- **Any learned backend.** GTSAM is classical and fixed.
- **Any learned stereo depth.** IGEV-Stereo is frozen and pretrained.
- **Any learned IMU bias correction beyond AirIMU.** AirIMU is frozen.
- **Multi-dataset training.** VIODE only.
- **Loop closure.** v2.
- **Rolling-shutter correction.** VIODE is global-shutter.
- **Sub-pixel flow upsampling.** Residual flow lives at $1/8$; the solver operates at $1/8$. Upsampling to full resolution is only for visualization.
- **Online uncertainty calibration at inference.** The training-time calibration is sufficient by design (see §9).
- **Direct image photometric loss.** We do not use any photometric term. All supervision comes from $(R_\text{gt}, t_\text{gt})$ and the derived 3D residuals.

---

## 19. Why this is "sure-shot working"

A concrete summary of the claim.

**Reason 1: every component has one job.** The stereo backbone gives depth. The feature encoder gives features. AirIMU gives rotation and a translation prior. The three learned heads produce flow, dynamic score, and covariance. The GN solver produces the pose. The backend produces the trajectory. Nothing does two things at once. Nothing depends on another component being "approximately right" for it to converge.

**Reason 2: every learned parameter has direct supervision.**

- Residual flow receives gradient from $L_\text{pose}$ (through the unrolled GN) and from $L_\text{cov}$ (through $e_i$).
- Dynamic score receives gradient from $L_\text{dyn}$ (direct BCE) and from $L_\text{pose}$ (through the soft weights $W_i$).
- Uncertainty receives gradient from $L_\text{cov}$ (direct NLL) and from $L_\text{pose}$ (through the matrix precisions $W_i$).

No learned parameter depends on another parameter's outputs to get a gradient.

**Reason 3: pose estimation is tightly bounded.** Three unrolled GN iterations, fixed count, well-conditioned Hessian (via $\mu I_6$), massively over-determined ($\sim$5600 pairs vs. 6 unknowns), and a near-perfect rotation init from AirIMU. There is no adaptive iteration count, no convergence tolerance, no early-termination knob. The forward pass is deterministic in compute.

**Reason 4: the per-pair solver and the backend share code.** The GN machinery of §7.4 is literally a factor graph with two nodes and one visual factor. The backend is the same factor graph with $N$ nodes. Bugs in one are automatically bugs in the other, which means fixing one fixes both. No divergence between "tracking math" and "mapping math."

**Reason 5: every loss is ablatable to zero.** If training diverges, disable one term at a time to find the culprit. No "this loss is required for this other loss to have a meaningful signal" entanglement.

**Reason 6: zero per-sequence hyperparameters.** The deployment checkpoint works on any VIODE sequence without modification, and — if the frozen backbones generalize as expected — on sequences outside VIODE too. The soft-gated matrix-weighted formulation avoids hard dynamic thresholds at inference.

**Reason 7: the search space is small.** ~860k learned parameters. The heads are tiny CNNs. They are being asked to produce three interpretable quantities with strong direct supervision. Small model, big signal.

---

## 20. Architecture deep-dive (flowcharts and ASCII diagrams)

This section is intended to let a reader build the model in their head before building it in code. Every diagram is paired with the exact tensor shapes at that stage, using the concrete VIODE resolution $H \times W = 480 \times 752$ and feature stride 8 ($H' \times W' = 60 \times 94$), with batch size $B = 8$.

### 20.1 Top-level tensor-shape pipeline

End-to-end shapes, one frame pair at a time. $C = 256$ throughout.

```
INPUTS
======
I_t^L       : [B, 3,    H,  W ]  = [8, 3,    480, 752]   left image, t
I_t^R       : [B, 3,    H,  W ]                          right image, t
I_{t+1}^L   : [B, 3,    H,  W ]                          left image, t+1
I_{t+1}^R   : [B, 3,    H,  W ]                          right image, t+1
K           : [B, 3, 3]                                  camera intrinsics
imu_window  : [B, N_imu, 6]      ≈ [8, 10, 6]            acc+gyro samples
imu_mask    : [B, N_imu]                                 validity mask
gt_R, gt_t  : [B, 3, 3], [B, 3]                          training only

            │
            │  ┌─ FROZEN STEREO (IGEV-Stereo) ────────────────────┐
            ├─►│  D_t, D_{t+1} : [B, 1, H, W] each  (meters)      │
            │  │  σ_d^t, σ_d^{t+1}: [B, 1, H, W] each (pixels)    │
            │  └───────────────────────────────────────────────────┘
            │
            │  ┌─ FROZEN FEATURES (RAFT fnet) ────────────────────┐
            ├─►│  F_t, F_{t+1} : [B, C, H', W'] = [8, 256, 60, 94]│
            │  └───────────────────────────────────────────────────┘
            │
            │  ┌─ FROZEN IMU (AirIMU preintegration) ─────────────┐
            └─►│  ΔR_imu : [B, 3, 3]                               │
               │  Δv_imu : [B, 3]                                  │
               │  Σ_ΔR   : [B, 3, 3]                               │
               └───────────────────────────────────────────────────┘

IMU ROTATION PRE-WARP (no params)
=================================
    H_rot = K · ΔR_imu · K^{-1}       : [B, 3, 3]
    grid  = build_grid(H_rot, H', W') : [B, H', W', 2]
    F_{t+1}^warp = grid_sample(F_{t+1}, grid)
                                      : [B, C, H', W']

CORRELATION VOLUME (no params)
==============================
    CorrVol  = all_pairs_corr(F_t, F_{t+1}^warp)      [B, 196, H', W']
               (4 pyramid levels × (2·3+1)^2 = 196 channels)

LEARNED HEADS (three tiny CNNs)
===============================
    δf        = FlowResHead(CorrVol)                  [B, 2, H', W']
    s         = DynHead(CorrVol, δf, |δf|)            [B, 1, H', W']
    (σ_u²,
     σ_v²)    = UncertaintyHead(CorrVol, δf)          [B, 2, H', W']

GEOMETRIC LIFT + UNCERTAINTY PROPAGATION (no params)
====================================================
    # 3D points at feature stride
    X_t       : [B, 3, H', W']   = D_t ⊙ K^{-1}·[u,v,1]
    j         : [B, 2, H', W']   = H_rot applied to (u,v) + δf
    X_{t+1}   : [B, 3, H', W']   = D_{t+1}(j) ⊙ K^{-1}·[j_u,j_v,1]

    # Per-pixel 3×3 covariances
    Σ_3D^t    : [B, 3, 3, H', W']   via J_lift·diag(σ_u², σ_v², σ_D²)·J_liftᵀ
    Σ_3D^{t+1}: [B, 3, 3, H', W']   same at the matched location
    Σ_res     : [B, 3, 3, H', W']   = Σ_3D^t + Σ_3D^{t+1}

    # Matrix precision with soft gate
    W         : [B, 3, 3, H', W']   = (1 - s) · (Σ_res + λI)^{-1}

UNROLLED GAUSS-NEWTON (3 iters, no params)
==========================================
    (R, t)^{(0)} ← (ΔR_imu, Δv_imu · Δt)     [B, 3, 3], [B, 3]

    for k in 0, 1, 2:
        e^{(k)}  : [B, 3, H', W']             = (R^{(k)} X_t + t^{(k)}) - X_{t+1}
        J^{(k)}  : [B, 3, 6, H', W']          per-pixel Jacobian
        H^{(k)}  : [B, 6, 6]                  = Σ Jᵀ W J
        g^{(k)}  : [B, 6]                     = Σ Jᵀ W e
        δξ^{(k)} : [B, 6]                     = -(H + μI)^{-1} g
        (R, t)^{(k+1)} = Exp_SE3(δξ^{(k)}) · (R, t)^{(k)}

    Σ_pose   : [B, 6, 6]                      = (H^{(3)} + μI)^{-1}

OUTPUTS (per frame pair)
========================
    R, t     : [B, 3, 3], [B, 3]              pair-wise pose
    Σ_pose   : [B, 6, 6]                      pair-wise pose covariance
    s, Σ_3D  : passed to L_dyn and L_cov during training
    CorrVol, H^{(3)}, g^{(3)}: passed to the factor-graph backend at inference
```

### 20.2 IMU rotation pre-warp — how a single pixel gets warped

```
Pixel in frame t                    Corresponding pixel in F_{t+1}
(u, v)          ─ H_rot = K·ΔR·K^{-1} ─►          (u', v')
                                                      │
                                                      ▼
F_{t+1}  ────────────────────────────────►    bilinear sample
                                                      │
                                                      ▼
                                            F_{t+1}^warp(u, v)

After the warp, if the scene is perfectly rigid AND motion is pure rotation,
F_t(u, v) and F_{t+1}^warp(u, v) depict the same world point.
The residual flow δf captures:
  - translation parallax (depth-dependent shift)
  - dynamic-object motion (scene-dependent shift)
```

The homography is depth-independent because a pure rotation induces the infinite homography $K R K^{-1}$, which acts on the projective image plane without referencing depth. This is the textbook result and is one reason AirIMU's short-interval rotation prior is so useful: it *exactly* removes rotation from the feature map.

### 20.3 Correlation volume lookup (RAFT-style)

```
F_t (H', W', C)      F_{t+1}^warp (H', W', C)
      │                       │
      │                       │
      ▼                       ▼
           inner product
           per (i, j) pair
                  │
                  ▼
    corr : [B, H', W', H', W']     (all-pairs 4D)
                  │
                  ▼
    reshape, average-pool twice
                  │
                  ▼
    4-level pyramid:
      level 0: corr at full feature resolution
      level 1: 2× pooled
      level 2: 4× pooled
      level 3: 8× pooled

LOOKUP at pixel (u, v):
    for each level ℓ:
        sample (2r+1)² = 49 values in a
        radius-3 neighborhood around (u, v)
    stack → 4 · 49 = 196 values per pixel

Result: CorrVol : [B, 196, H', W']
```

The lookup is radius 3 (not 4 as in vanilla RAFT) because the rotation pre-warp removes most of the inter-frame pixel motion. After warping, residual flow magnitudes at typical stereo depths are within $\pm 3$ feature pixels, which is $\pm 24$ full-resolution pixels — plenty for both typical parallax and fast dynamic objects.

### 20.4 Per-head micro-architectures (exact layers and shapes)

```
FlowResHead
-----------
CorrVol [B, 196, H', W']
      │
      ▼ Conv2d(196, 128, k=1)       ReLU
      ▼ Conv2d(128, 128, k=3, p=1)  ReLU
      ▼ Conv2d(128,  64, k=3, p=1)  ReLU
      ▼ Conv2d( 64,   2, k=1)
      ▼
δf [B, 2, H', W']
Parameter count: ~330k


DynHead
-------
CorrVol  [B, 196, H', W']  ┐
δf       [B,   2, H', W']  ├─ concat ─► [B, 199, H', W']
|δf|     [B,   1, H', W']  ┘
      │
      ▼ Conv2d(199, 128, k=1)       ReLU
      ▼ Conv2d(128, 128, k=3, p=1)  ReLU
      ▼ Conv2d(128,  64, k=3, p=1)  ReLU
      ▼ Conv2d( 64,   1, k=1)
      ▼ Sigmoid
s [B, 1, H', W']
Parameter count: ~350k


UncertaintyHead
---------------
CorrVol  [B, 196, H', W']  ┐
δf       [B,   2, H', W']  ┴─ concat ─► [B, 198, H', W']
      │
      ▼ Conv2d(198, 128, k=1)       ReLU
      ▼ Conv2d(128,  64, k=3, p=1)  ReLU
      ▼ Conv2d( 64,   2, k=1)   # outputs (log σ_u², log σ_v²)
      ▼ exp, clamp to [1e-4, 1e2]
Σ_2D [B, 2, H', W']  (diagonal of 2×2 per-pixel image-plane covariance)
Parameter count: ~180k

Total learned params: ~860k
```

All three heads consume the **same** `CorrVol` tensor. The encoder forward is done **once** per frame pair; the heads are tiny and cheap. There is no shared intermediate representation between heads beyond `CorrVol` and (for `DynHead` and `UncertaintyHead`) the `δf` produced by `FlowResHead`, so the heads execute sequentially:

```
     CorrVol
        │
        ▼
   FlowResHead ──► δf ──────────┐
                                │
                                ▼
                           DynHead ──► s
                                │
                                ▼
                       UncertaintyHead ──► Σ_2D
```

`DynHead` and `UncertaintyHead` are independent once `δf` exists, so in practice they run in parallel on GPU.

### 20.5 Geometric lift with uncertainty (one pixel at a time)

```
Pixel i = (u, v) at feature stride
           │
           ▼
   D_t^i  ← sample from depth map at (u, v)
   σ_D^i  ← f_x · b · σ_d / d²   (analytic stereo error)
   σ_u²,
   σ_v²   ← from UncertaintyHead at (u, v)
           │
           ▼
   Σ_in  = diag(σ_u², σ_v², σ_D²)          [3, 3]
           │
           ▼
   J_lift = ∂X/∂(u, v, D)                   [3, 3]
          = [D·K⁻¹_{:,0}, D·K⁻¹_{:,1}, K⁻¹·[u,v,1]ᵀ]
           │
           ▼
   X_t^i    = D_t^i · K⁻¹ · [u, v, 1]ᵀ     [3]
   Σ_3D^{t,i} = J_lift · Σ_in · J_liftᵀ    [3, 3]
           │
           ▼
   # Repeat for X_{t+1}^i at (u + δu, v + δv)
   # with D_{t+1} sampled at that location
           │
           ▼
   Σ_res^i = Σ_3D^{t,i} + Σ_3D^{t+1,i}      [3, 3]
           │
           ▼
   W^i = (1 - s^i) · (Σ_res^i + λI)⁻¹        [3, 3]  ← soft gate
```

The per-pixel "weight" is a $3 \times 3$ matrix, not a scalar — this is the key structural change from the closed-form variant. The Gauss-Newton solver below consumes these matrices natively.

### 20.6 Unrolled Gauss-Newton — iteration-level ASCII trace

```
INIT
    R^{(0)} ← ΔR_imu              (near-perfect, <0.1° err)
    t^{(0)} ← Δv_imu · Δt          (coarse, ~10% of true translation)

─────────── ITERATION 0 ────────────
    for each pixel i:
        e_i  = (R^{(0)} X_t^i + t^{(0)}) - X_{t+1}^i     [3]
        J_i  = [-(R^{(0)} X_t^i)_×, I_3]                  [3, 6]
        W_i  = (from §20.5)                               [3, 3]

    H = Σ_i J_iᵀ W_i J_i                                  [6, 6]
    g = Σ_i J_iᵀ W_i e_i                                  [6]
    δξ = -(H + μI)⁻¹ g                                    [6]

    ξ_rot = δξ[0:3]   (rotation tangent)
    ξ_tr  = δξ[3:6]   (translation tangent)
    ΔR    = Exp_SO3(ξ_rot)
    R^{(1)} = ΔR · R^{(0)}
    t^{(1)} = ΔR · t^{(0)} + V(ξ_rot) · ξ_tr
               └──── left-update on SE(3) ────┘

─────────── ITERATION 1 ────────────
    (same computation, with (R, t)^{(1)})
    → (R, t)^{(2)}

─────────── ITERATION 2 ────────────
    (same computation, with (R, t)^{(2)})
    → (R, t)^{(3)}   ← FINAL OUTPUT

    Σ_pose = (H^{(3)} + μI)⁻¹                              [6, 6]
```

**Matrix sizes per iteration:** $H$ is $6\times6$, $g$ is a 6-vector, the linear solve is a $6\times6$ inverse. All cheap. The dominant cost is assembling $H$ and $g$, which is a batched einsum over all valid pixels — parallelizes trivially on GPU.

**Differentiation:** PyTorch's autograd traces the full unrolled loop. Gradients flow backward through three linear solves ($(H+\mu I)^{-1}$ — PyTorch has native gradients for matrix inverse and `solve`), three $SE(3)$ exponentials (Rodrigues formula, fully differentiable), and three Jacobian assemblies. No custom backward is required for any step.

### 20.7 Hessian assembly — how pixels contribute to $H$

```
For a single pixel i with W_i ∈ ℝ^{3×3} and J_i ∈ ℝ^{3×6}:

     J_iᵀ            W_i          J_i
  ┌────┐       ┌────────┐      ┌────┐
  │    │       │        │      │    │
  │ 6  │       │  3×3   │      │  6 │
  │  × │   ·   │  (full │   ·  │ ×  │
  │ 3  │       │ matrix │      │ 3  │
  │    │       │ precision│     │    │
  └────┘       └────────┘      └────┘
                │
                ▼
         6×6 rank-≤3 block

         │
         ▼   sum over all N ≈ 5640 valid pixels per frame pair
         │
         ▼
       H_total : [6, 6]   (sum of ~5640 rank-3 contributions → full rank)
```

At $B=8$ frames × $60 \times 94$ feature locations × three iterations, the einsums total ~2.7 M element-wise multiplications. On a modern GPU this is <1 ms. The GN solver is **not** the expensive part of the forward pass — the RAFT feature encoder is.

### 20.8 Gradient flow during training (who shapes whom)

This is the most important diagram for understanding whether the training will converge. It shows which loss term produces gradients for which head.

```
                                   L_pose                  L_dyn                 L_cov
                                     │                       │                     │
                                     │                       │                     │
                      ┌──────────────┴──────────┐             │                     │
                      ▼                         ▼             │                     │
             unrolled GN (R, t)        soft gate (1 - s)      │                     │
                      │                         │             │                     │
                      ▼                         ▼             │                     │
              weights W_i     ◄─────────────────┘             │                     │
                   ▲    ▲                                     │                     │
                   │    │                                     │                     │
                   │    └──── matrix precision                │                     │
                   │          (Σ_res + λI)^{-1}               │                     │
                   │                  ▲                       │                     │
                   │                  │                       │                     │
                   │                  └─── Σ_res  ◄────── propagate Σ_2D ◄──── UncertaintyHead
                   │                                                                │
                   │                                                                │
                   │                                             direct BCE         │
                   │                                                  │             │
                   │                                                  ▼             │
                   │                                              DynHead           │
                   │                                                                │
                   │                                                 direct NLL ────┘
                   │                                                  │
                   │                                                  ▼
                   │                                         UncertaintyHead (again)
                   │
                   │
                   │  (also flows through correspondences e_i)
                   │
                   └── FlowResHead (via 3D residual e_i through unrolled GN)

Legend:
   L_pose → shapes FlowResHead, DynHead, UncertaintyHead through the GN unroll
   L_dyn  → shapes DynHead directly (BCE)
   L_cov  → shapes UncertaintyHead directly (NLL)
            plus FlowResHead via the 3D residual e_i
```

**Key property:** every learned head has **at least two** gradient paths — one direct supervision (for `DynHead` and `UncertaintyHead`), plus one indirect path through `L_pose`. `FlowResHead` has no direct supervision but gets very strong gradient from `L_pose` because $\delta f$ enters the 3D residual $e_i$ directly.

**Why this is convergence-friendly:** if `L_pose` is working (i.e. the GN is returning something sensible), all three heads receive useful gradient. If `L_pose` is not yet working (early training), `L_dyn` and `L_cov` still shape `DynHead` and `UncertaintyHead` independently, so the heads move toward a reasonable state before `L_pose` becomes informative. There is **no chicken-and-egg regime** where all three losses need to be working simultaneously for any of them to produce useful gradient.

### 20.9 Training-step flowchart

```
                  ┌──────────────────────────┐
                  │  sample batch from VIODE │
                  │  (stereo, IMU, gt_R, gt_t)│
                  └────────────┬─────────────┘
                               │
                               ▼
                  ┌──────────────────────────┐
                  │  frozen stereo depth      │
                  │  frozen feature encoder   │
                  │  frozen AirIMU            │
                  └────────────┬─────────────┘
                               │
                               ▼
                  ┌──────────────────────────┐
                  │  homography pre-warp      │
                  │  correlation volume       │
                  └────────────┬─────────────┘
                               │
                               ▼
                  ┌──────────────────────────┐
                  │  FlowResHead → δf         │
                  │  DynHead     → s          │
                  │  UncertaintyHead → Σ_2D   │
                  └────────────┬─────────────┘
                               │
                               ▼
                  ┌──────────────────────────┐
                  │  lift to 3D, matrix W_i   │
                  └────────────┬─────────────┘
                               │
                               ▼
                  ┌──────────────────────────┐
                  │  unrolled GN × 3 iters    │
                  │  → (R, t)                 │
                  └────────────┬─────────────┘
                               │
                    ┌──────────┼──────────┐
                    ▼          ▼          ▼
              ┌─────────┐ ┌────────┐ ┌────────┐
              │ L_pose  │ │ L_dyn  │ │ L_cov  │
              │  (SE3)  │ │ (BCE)  │ │ (NLL)  │
              └────┬────┘ └───┬────┘ └───┬────┘
                   └──────────┼──────────┘
                              │ sum
                              ▼
                     ┌────────────────┐
                     │  L_total       │
                     │  backward      │
                     │  optimizer step│
                     └────────────────┘
```

### 20.10 Inference-time pipeline (with factor-graph backend)

```
TIME t                                              TIME t+1
======                                              ========
stereo pair ─►┐                                    stereo pair ─►┐
imu buffer ──►│                                    imu buffer ──►│
              │                                                  │
              ▼                                                  ▼
        front-end                                           front-end
     (shared with train)                                  (shared with train)
              │                                                  │
              ▼                                                  ▼
        per-pair GN                                         per-pair GN
              │                                                  │
              ▼                                                  ▼
      (R, t, Σ_pose)                                    (R, t, Σ_pose)
              │                                                  │
              │   ┌──── KEYFRAME SELECTION ─────┐                 │
              │   │ |t| > 0.15m OR              │                 │
              └──►│ angle > 5° OR               │◄────────────────┘
                  │ dt > 0.5s since last KF?    │
                  └──────────────┬──────────────┘
                                 │ yes
                                 ▼
                  ┌──────────────────────────────┐
                  │  NEW KEYFRAME INSERTED       │
                  │                              │
                  │  add variable T_{k+1}        │
                  │  add visual factor(T_k,T_{k+1})│
                  │  add IMU factor(T_k, T_{k+1})  │
                  │  run iSAM2 update            │
                  └──────────────┬───────────────┘
                                 │
                                 ▼
                  ┌──────────────────────────────┐
                  │  optimized trajectory        │
                  │  over last 10 keyframes      │
                  │                              │
                  │  marginalize oldest KF       │
                  │  if window > 10              │
                  └──────────────┬───────────────┘
                                 │
                                 ▼
                       publish latest pose
```

Non-keyframes are published by composing the latest keyframe pose with the per-pair GN delta (no backend update for those).

### 20.11 Factor graph structure at steady state

```
Variables (10 keyframes in the sliding window):
    T_{k-9}  T_{k-8}  T_{k-7}  ...  T_{k-1}  T_k

Factors:
                                     ┌─── prior on oldest KF ───┐
                                     ▼                          ▼
    ●━━━Vis━━━●━━━Vis━━━●  ...  ●━━━Vis━━━●━━━Vis━━━●
    │         │         │       │         │         │
   IMU       IMU       IMU     IMU       IMU       IMU
    │         │         │       │         │         │
    ●━━━━━━━━●━━━━━━━━━●  ...  ●━━━━━━━━━●━━━━━━━━━●
    T_{k-9}  T_{k-8}  T_{k-7}  T_{k-2}   T_{k-1}   T_k

Legend:
    ●                  variable node (SE(3) pose)
    ━━Vis━━            visual 3D-3D factor from per-pair GN
    ━━IMU━━            IMU preintegration factor (AirIMU)
    prior              anchor on the oldest kept keyframe
```

Each visual factor carries the $6\times6$ Hessian $H^{(3)}$ and gradient $g^{(3)}$ from §7.4 (per the "pre-summarized" option in §8). Each IMU factor carries $\Delta R_\text{imu}, \Delta p_\text{imu}$ and their noise model. The prior anchors the gauge. iSAM2 handles variable elimination and incremental updates.

### 20.12 Latency breakdown (budget, RTX 3090 target)

```
Component                             Target (ms)    Notes
──────────────────────────────────────────────────────────────
Stereo depth (IGEV-Stereo, both)          12         dominant
RAFT feature encoder (both frames)         4
AirIMU preintegration                     <1
IMU rotation pre-warp (grid_sample)       <1
Correlation volume build                   2
FlowResHead forward                        1
DynHead forward                            1
UncertaintyHead forward                    1
3D lifting + matrix W construction         1
Unrolled GN (3 iterations)                 2         cheap; ~5600 pixels
──────────────────────────────────────────────────────────────
FRONT-END TOTAL                           ~25 ms    (40 FPS budget)

Factor-graph backend (iSAM2 update)       ~10 ms   (on keyframes only)
```

Stereo depth is the dominant cost. IGEV-Stereo compiles cleanly to TensorRT, which brings it down to ~5 ms and puts the whole front-end in the 15 ms range for deployment. v1 does not require TensorRT compilation.

### 20.13 What a single debugging session looks like

A worked example of how the ablatability property helps in practice:

```
Symptom: training diverges at step ~5000 with NaN L_pose.

Step 1: disable L_pose (λ_pose = 0).
    → does training still diverge?
    → if no, the GN block is the culprit; check Hessian conditioning,
      check μ damping, check Jacobian signs.
    → if yes, divergence is elsewhere.

Step 2: disable L_cov (λ_cov = 0).
    → Σ_3D still supervised only by L_pose through the weights.
    → does Σ inflate to infinity? (log trace in diagnostics)
    → does L_pose explode because the weights vanish?

Step 3: check τ_r trajectory (the adaptive per-batch median).
    → is it collapsing toward zero on early batches?
    → if yes, the 0.05m floor is doing its job and the divergence
      is elsewhere; if no, investigate why residuals are huge.

Step 4: extend the λ_dyn ramp (e.g., 15k steps instead of 5k).
    → does DynHead recover?
    → if yes, early residual labels were too noisy.
```

This kind of linear-bisection debugging works because every loss is independently ablatable and every head has multiple gradient paths. Without that property, a diverging training run is a blind search.

---

*End of spec.*
