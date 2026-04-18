# Method, Theory, and Intuition — Static Confidence Head for MAC-VO

**Companion document to:** `2026-04-11-static-confidence-head-macvo-spec.md` (this revision, 2026-04-17)

This is the long-form, from-first-principles walkthrough of the method. The spec tells
you *what* to build and *how* to wire it. This document tells you *why* every piece is
shaped the way it is — which identity each choice rests on, which failure mode each
choice is trying to prevent, and what would break if you removed it. Read this when you
want to understand the idea, argue about it, or extend it. Read the spec when you want
to write the code.

The design has three pillars that are mutually reinforcing:

1. **A static-confidence head** (§2–§7) trained on a geometric residual identity, fed
   into the solver as a per-observation weight `c ∈ [0, 1]`. The head emits a single
   scalar per pixel; the cycle-consistency signal enters training as a *validity mask*
   on the loss, not as a second output. A factorized variant
   `c = p_static · p_visible` is preserved as an ablation (§5, spec §10).
2. **A DRT-loosely-coupled stereo-adapted bootstrap** (§8) that hands the backend a
   metrically-consistent initial state `(R_k, p_k, v_k, b_g, g_W)` over the first
   `N_init` keyframes.
3. **A shared Forster-style preintegrator** (§9) that emits one factor consumed by
   *both* the head (as a conditioning prior) and the backend (as a hard constraint).
   There is exactly one preintegrator in the tree; frontend conditioning and backend
   IMU factors never see mismatched numerics.

Every decision below can be traced back to one of these three pillars.

---

## 1. The problem, precisely

Visual-inertial odometry (VIO) is a nonlinear least-squares problem: find the camera
trajectory that best explains a set of pixel observations under a **rigid-scene**
assumption, optionally regularized by inertial measurements. "Rigid scene" means the
world does not move between frames — only the camera does. Every classical VO/VIO
solver (MAC-VO, DPVO, TartanVO, ORB-SLAM, VINS-Fusion, OpenVINS, …) bakes this
assumption into the cost function, because without it the 2-view epipolar geometry
and 3-view Structure-from-Motion constraints are ill-posed.

When a car drives past the camera, or a person walks in front of it, the assumption is
locally false. The pixels on that object move independently of the camera, and any
solver that treats them as rigid-scene observations will corrupt its pose estimate in
proportion to how many such observations there are and how far their motion disagrees
with the rigid-scene prediction.

The standard mitigations fall into three categories:

1. **Robust loss functions** — Huber, Cauchy, truncated L2. These downweight large
   residuals regardless of *cause*. A large residual from dynamic motion and a large
   residual from matching noise get the same treatment. Cheap but indiscriminate.
2. **Semantic masking** — run a segmentation network and drop observations on "car",
   "person", "bicycle". Discriminative but brittle: trained on a specific ontology,
   fires on parked cars, misses objects out-of-distribution.
3. **Motion-aware masking** — predict, per pixel, whether the pixel's motion is
   consistent with a single rigid-scene hypothesis. The right granularity: ignores
   parked cars (they are static), ignores static humans (they are static), doesn't
   care about object category, but requires the network to understand geometry
   rather than semantics.

This method is category (3). The head outputs, for every pixel, a probability
`c ∈ [0, 1]` interpreted as "how confident am I that this pixel is both (i)
rigid-scene in motion and (ii) trackable enough to be a usable correspondence". `c =
1` means "trust this pixel fully"; `c = 0` means "do not use it".

A careful read of that interpretation shows two distinct sub-claims — *rigidity* and
*visibility* — but in this design the head exposes only the product to the solver: a
single scalar `c`. The visibility sub-claim is still handled, but via a **validity
mask** on the training loss (§10, §3.5) rather than via a second head output. That
keeps the solver interface as simple as possible — one scalar per pixel — while still
letting us ignore pixels where the cycle signal is untrustworthy. §5 explains why the
factorization is not required in practice, and §10 shows how the mask absorbs the
work it would otherwise do.

---

## 2. Rigid flow: the identity the method rests on

### 2.1 The pinhole projection equation

For a pinhole camera with intrinsics `K`, a 3D point `X` in the camera frame projects
to pixel `x`:

```
x = π(X) = K · [X_x / X_z, X_y / X_z, 1]ᵀ
```

Given a depth map `D`, we back-project a pixel `x = (u, v)` to its 3D point in the
camera frame:

```
X(x) = D(x) · K⁻¹ · [u, v, 1]ᵀ
```

**Convention note specific to MAC-VO.** MAC-VO's geometry module uses a NED-style
convention with the optical axis along `+x`, not `+z`. The helpers
`Utility.Point.pixel2point_NED` / `point2pixel_NED` bake that in, and the analytic
Jacobian in `Analytic_Reproj_TwoFramePGO.build_jacobian` reads
`x, y, z = pos_Tc[:, 0], pos_Tc[:, 1], pos_Tc[:, 2]` with the division by `x²`. Every
rigid-flow formula in this document should be read against that axis; the closed form
is the same, only the symbol `z` is renamed to `x`. The spec (§0.2) pins this down for
implementers.

### 2.2 What flow should look like for a rigid scene

For two frames at times `t` and `t+1`, related by a rigid-body transform
`(R, t)` carrying camera-frame-`t` to camera-frame-`t+1`, a **rigid** scene point
visible in both produces:

```
X_{t+1} = R · X_t + t
x_{t+1} = π(X_{t+1})
f_rigid(x_t) = x_{t+1} − x_t
```

`f_rigid` is the **rigid flow**: the 2D optical flow that would be observed if every
pixel corresponded to a point static in the world and only appeared to move because
the camera moved. It is a deterministic function of the camera's motion, the scene
depth, and the intrinsics. Nothing else.

### 2.3 The residual identity

With a *measured* optical flow `f_obs(x_t)` produced by a correspondence network like
FlowFormerCov, the **flow residual**

```
r(x_t) = || f_obs(x_t) − f_rigid(x_t) ||₂
```

has a clean interpretation:

| Pixel's physical status | `r` should be… |
|---|---|
| Truly static, perfect depth, perfect flow | 0 |
| Truly static, noisy depth / noisy flow | small (proportional to sensor noise) |
| On a moving object | large (proportional to the object's own 3D velocity, projected) |
| Occlusion / disocclusion | large and meaningless (no true correspondence) |

Notice what is and isn't in this identity:

- **No assumption about object category** — the test is purely kinematic.
- **No semantic labels required** — the input is just depth, poses, and flow.
- **It needs the relative pose**, so it only works as a label if we have `(R, t)` from
  somewhere. At training time we have ground truth. At inference time we don't, and
  this asymmetry is the central design constraint of the method, addressed in §6.

---

## 3. The residual as a supervision signal

The residual isn't a label; it's a continuous quantity that needs to be turned into
one. The straightforward map is:

```
d*(x) = σ( (r(x) − τ) / κ )
c*(x) = 1 − d*(x)
```

where `σ` is the sigmoid, `τ` is a threshold, and `κ` sets softness. Small residuals
map to `c* ≈ 1` (static), large residuals map to `c* ≈ 0` (dynamic), intermediate
values are soft.

### 3.1 Why you can't use a hard threshold

The tempting alternative is a hard binarization `c* = 1[r < τ]`. This breaks for four
separate reasons:

1. **Decision-boundary instability during training.** A pixel on the edge of the
   threshold flips labels under tiny perturbations of depth, flow, or pose. The head
   sees its own label flicker and can't converge.
2. **BCE gradient vanishes on confident labels.** With binary `{0, 1}` labels and a
   sigmoid head, the gradient from a confident correct prediction drops to zero — most
   of the training signal then comes from the (small) set of pixels near the boundary.
   Soft labels spread the gradient over a larger fraction of the image.
3. **The residual is a noisy measurement of a latent variable.** What we care about is
   whether the pixel's *ground-truth dynamic velocity* is zero. The residual is a
   sample from a noise distribution centered at a function of that velocity. Treating a
   single noisy sample as a hard label discards the information in the noise
   distribution.
4. **Useful ordinal structure is lost.** A residual of 20 px is "more dynamic" than a
   residual of 5 px. Hard binarization collapses that ordering; the solver benefits
   from keeping it.

Soft labels preserve all four.

### 3.2 Why a fixed `τ` fails too

A constant threshold across the whole image is almost always wrong, for a reason
specific to VO and not present in most dynamic-mask papers: **rigid flow magnitude
depends on depth**.

A camera translating forward at 1 m/s at 30 Hz with focal length `f = 500`:

- A static point at 2 m depth: rigid flow ≈ `f · (1/2) / 30 ≈ 8 px`.
- A static point at 100 m depth: rigid flow ≈ `f / 3000 ≈ 0.17 px`.

Both are static. The observed flow at each equals the rigid flow plus roughly constant
noise (say, 0.3 px from FlowFormerCov). The residual at the *close* point is dominated
by flow noise (0.3 / 8 ≈ 4%); at the *far* point it is larger than the expected rigid
flow itself (0.3 / 0.17 > 1). A threshold that works for the close point flags half
the distant scene as dynamic; a threshold that works for the distant point misses all
dynamic objects in the near field.

### 3.3 The depth-adaptive threshold

Scale `τ` with the expected rigid-flow magnitude, which is itself predictable from
depth, translation magnitude, and focal length:

```
|f_rigid| ≈ (f / D) · ||t||              (order-of-magnitude)
τ(D)     = τ₀ + α · (f / D) · ||t||
```

`τ₀` is a floor (captures pure flow noise); the additive term grows as rigid flow
gets larger. Equivalently, the threshold is "how much 2D flow corresponds to a 3D
velocity above some cutoff" — a 3D-velocity threshold expressed in pixel units.

```
d*(x) = σ( (r(x) − τ(D(x))) / κ )
c*(x) = 1 − d*(x)
```

This is why the spec's config has both `tau0` and `alpha`. A single constant threshold
isn't a subtle tuning choice — it is a bug.

### 3.4 The focus-of-expansion blind spot

There's one regime the residual-based approach cannot fix on its own: the **focus of
expansion** (FOE), the direction the camera is translating. For pixels at the FOE, the
rigid flow vanishes for *all* depths — so a dynamic object there produces a residual
equal only to its own projected motion, which can be small if the object is also
moving in that direction.

The method cannot solve this purely from residuals. What it *can* do is exploit the
network's access to richer features: the 128-channel FlowFormerCov context feature map
encodes what the scene looks like, not just how fast things moved. A car directly ahead
still looks like a car, and those context features carry that information into the
head's decision. This is part of why the architecture reads `f_ctx` in addition to
`flow`, `cov`, and the IMU-rigid proxy.

### 3.5 Occlusion, the other major failure mode

Even for truly static pixels there is a regime where the residual is large and
meaningless: **occlusion and disocclusion**. Pixels visible in frame `t` but hidden in
`t+1` (or vice versa) have no true correspondence. FlowFormerCov will produce *some*
flow vector for them, but that vector is defined by whatever texture happens to be in
the target frame — it has no physical meaning. The rigid flow derived from GT pose is
the geometrically correct answer computed from depth even when the pixel disappears
in the target image.

So at occluded pixels, `f_obs` is visual nonsense and `f_rigid` is geometrically
correct but unrealized. Their difference is large not because the pixel is dynamic,
but because there is nothing to compare to.

Feeding this residual into a rigidity BCE as a dynamic label would poison training.
The fix is **cycle-consistency masking**: compute forward flow `f_fwd` and backward
flow `f_bwd`, warp `f_bwd` by `f_fwd`, and check that the composition returns close
to the origin:

```
warp_err(x_t) = || f_fwd(x_t) + f_bwd(x_t + f_fwd(x_t)) ||₂
```

Pixels where `warp_err` is large are occluded or at dynamic-boundary discontinuities.
They should not drive a rigidity label.

In this design the cycle check plays **one** role: a **validity mask** `M_cycle` on
the GT-residual loss. Pixels with large `warp_err` contribute zero gradient to
`L_dyn`, so occlusions and dynamic-boundary discontinuities simply do not drive the
rigidity label. The check is binary with a loose threshold `tau_cyc_valid` — we don't
need a calibrated soft label here, we need a gate.

The reason we chose a mask rather than a second supervised head — even though the
cycle signal *could* supply a real pseudo-label — is covered in §5. Short version: the
solver only consumes a single scalar `c`, so training a second `p_visible` head whose
only job is to re-combine with `p_static` multiplicatively adds parameters without
adding signal. The mask is the minimum-effort way to let occluded pixels abstain from
supervision, which is the only work the second head would do.

The factorized variant (`c = p_static · p_visible`, with the cycle signal promoted
back to a pseudo-label) is preserved as an ablation; §5 lays out the trade.

---

## 4. What inputs the head actually needs

Before the recurrence, the refinement, or the IMU branch, the question is: **what
should the head read**? There are three plausible inputs and each answers a different
question:

1. **The residual map `r` itself.** Feeding `r` as input produces a near-trivial
   network: a learned thresholder. Worse, it *requires* access to GT pose (or ego
   pose) to compute at inference. Not used.
2. **The flow and its covariance.** This tells the network "how fast is each pixel
   moving, and how confident is the flow estimator". A fast pixel with low flow
   covariance is suspicious regardless of the rest of the scene.
3. **The context feature map `f_ctx`.** FlowFormerCov's 128-channel context encoder
   output encodes what the scene *looks like* — textures, shapes, regions of coherent
   appearance. It gives the network access to "what kind of thing is this pixel part
   of" independent of local motion.

The head reads (2) and (3) — not (1). The residual enters *only* as a training label.
This is the train/test symmetry constraint: if the head depended on `r` as input it
would need GT pose at inference, which it doesn't have. The head never reads anything
it couldn't read in the wild.

### 4.1 Why frozen features are rich enough

FlowFormerCov was trained on large-scale flow data with covariance supervision. Its
context features have already learned to represent object extents, texture coherence,
depth discontinuities. The static-confidence head is effectively a *small decoder*
learning to map those features to a different output space (rigidity and visibility
instead of flow). ~0.6M head parameters are enough for that decoding, and the small
parameter count is a feature: fast to train, less data-hungry, no catastrophic
forgetting.

### 4.2 Why frozen — the deeper reason

The choice to hard-freeze MAC-VO is not convenience. Training through FlowFormerCov
would mean:

- ~15M additional trainable parameters.
- Large, diverse flow datasets needed to avoid corrupting the pretrained features.
- Risk of catastrophic forgetting of FlowFormerCov's covariance calibration.
- A 10–50× compute overhead per training step.

Freezing makes the training a *light supervised probe* — we're asking "can you read
static-or-not off of these features", not "relearn the whole flow problem". A much
easier problem, dramatically less data, and no risk to the foundation model. This is
the same pattern as CLIP-probing, SAM prompt tuning, and every modern
foundation-model-adapter method. It works for the same reason here.

---

## 5. Why a single head suffices (factorization as ablation)

There's a natural impulse in methods like this to split the confidence into two
factors — one for rigidity, one for observability — because the interpretation of the
label looks "two-part". This section argues that in our setting the split is not
needed by default, and shows where the factorized variant (kept as ablation `A_fact`
in spec §10) would actually pay off.

### 5.1 The two latent variables

A pixel's usability for the rigid-scene solver rests on two latent conditions:

1. **Rigidity.** Is the world point static (stationary in the world frame)?
2. **Observability.** Is the correspondence reliable (FlowFormerCov produced a
   trustworthy match at this pixel)?

And four canonical regimes, written out in full:

| Pixel state | Rigidity | Observability | What the solver should do |
|---|---|---|---|
| Stationary textured wall | high | high | use it fully |
| Stationary, inside an occlusion | high | low | ignore (correspondence meaningless) |
| Moving car, clearly tracked | low | high | ignore (violates rigid-scene) |
| Moving car, edge at occlusion | low | low | ignore (both reasons) |

In *all three* of the "ignore" rows the solver behavior is the same: downweight. It
doesn't matter whether the pixel is unobservable or dynamic — it matters only that
the pixel is unusable. What the solver reads off the head is a single scalar `c`, and
its action on that scalar (multiply the residual by `c`, floor by `W_EPS = 1e-3`) is
invariant under *why* `c` is low.

### 5.2 Why this means a single head is enough

A factorization `c = p_static · p_visible` only earns its keep if it feeds the solver
some signal that a single `c` cannot. In our pipeline it does not:

- **The solver interface is 1-D.** The only thing that propagates into the backend
  cost is `w = c`. Whether `c` came from a product or directly from a single head,
  the downstream numbers are identical.
- **Occluded pixels need to abstain from training, not feed a second label.** The
  legitimate engineering concern with a monolithic head is that the rigidity label
  computed from the GT residual is *meaningless* at occluded pixels. We fix that
  with a binary **validity mask** `M_cycle` that zeros out `L_dyn` on those pixels
  (§3.5, §10). That is exactly the work a second `p_visible` head would do in the
  factorized variant — with the mask we achieve the same exclusion using zero
  trainable parameters.
- **Noise decomposition is a property of labels, not of heads.** The concern that
  rigidity-label noise (depth noise) should not contaminate visibility-label noise
  (cycle-check noise) is real. But in the single-head design visibility noise is
  never used as a label at all — it only gates rigidity supervision. So the noise
  channels are still decoupled; they're just decoupled by *masking* rather than by
  *separating heads*.
- **Calibration is simpler.** One temperature `T` versus two (`T, T_vis`), one ECE
  sweep versus two, one post-hoc optimum versus a 2-D grid.

The upshot: every concern the factorization was designed to address has a
one-scalar, single-head answer that uses the cycle signal as a mask rather than as a
second supervised output.

### 5.3 When the factorization would pay off

There are two hypothetical situations in which a factorized head would beat a single
one:

1. **A downstream consumer that reads the two factors separately.** E.g., a
   solver that scales visual covariance by `p_visible⁻²` (observability) but hard-
   masks by `p_static` (rigidity). We do not have such a consumer; Stage-2 soft
   weighting uses `w = c` only.
2. **A much larger, more diverse training corpus** where the two label noise
   structures are strongly heterogeneous and partial labels are common. In that
   regime, dataset `A` might have pose GT but no cycle (train `p_static` only) and
   dataset `B` might have cycle-only (train `p_visible` only). Our training corpus
   (TartanAir + VIODE + optional EuRoC) has both signals available everywhere, so
   the scenario doesn't apply.

Neither matches the current system. We therefore default to the single-head design
and preserve the factorized variant as `A_fact`. If the ablation wins by a
meaningful margin on ATE/RPE we reconsider; the spec's `dynamic_head.factorize:
true` flag makes that a config flip, not a rewrite.

### 5.4 Why product, not sum, in the factorized variant

For completeness: in the factorized variant the combination is a product `c =
p_static · p_visible`. Both sub-probabilities must hold for the pixel to be usable;
if either collapses, the factor contribution should collapse too. A sum would say
"static **or** usable", which is wrong semantically and breaks the whitened-residual
interpretation (§10). The product is the correct combination under the "both
conditions must hold" semantics; we keep it exactly when running `A_fact`.

---

## 6. The IMU branch and closing the train/test loop

The elephant in the room: at training we have GT pose, at inference we do not. How
can we train a head to do at inference what it only saw at training with information
it won't have later?

The first-order answer: **we never feed GT pose into the head**. GT pose enters the
loss only. The head's *input* at training is identical to its input at inference;
only the label differs.

That answer alone is necessary but not sufficient. The head still needs some
motion prior at inference — otherwise it collapses to "does this look like a dynamic
object" (semantic), which is exactly what we're trying to avoid. Without a motion
prior there is no way to distinguish "flow due to ego-motion" from "flow due to object
motion" except by appearance.

The answer is the full **IMU preintegration factor**, fed to the head in two ways
(global conditioning + per-pixel proxy), and — critically — *the same factor* also
supplies the backend's inertial constraint (§9). Three stages, all of them matter:

```
raw IMU samples ─► AirIMU learned corrector ─► (â, ω̂, σ²_a, σ²_g, Δb_a, Δb_g)
                                              │
                                              ▼
                    Differentiable Forster-style Preintegrator
                       (SO(3)×ℝ⁶, 1st-order cov prop, bias Jacobians)
                                              │
                                              ▼
              (ΔR̂, Δv̂, Δp̂, Σ_imu ∈ ℝ^{9×9}, dt, J_R_bg, J_v_bg, J_v_ba, J_p_bg, J_p_ba)
                                              │
                                              ▼
                 small FeatureMLP ─► f_imu ∈ ℝ^{128}   (global conditioning)
                 geometric projector ─► (Δf, r_imu_norm, valid) per pixel (§6.5)
```

What the head consumes — "IMU context" — is therefore **the same quantity a VIO
backend places in a preintegration factor**: a predicted relative rotation,
velocity, and translation on SE(3)×ℝ³, propagated uncertainty, and bias health. This
is deliberate: the same tensor serves (i) head conditioning, (ii) per-pixel proxy
input, and (iii) backend inertial constraint, and the three consumers never see
inconsistent values (invariant I2 in the spec).

### 6.1 Why the preintegration factor, not AirIMU's encoder alone

Feeding "AirIMU's raw encoder state" into the head would be subtly wrong. The head's
job is to disambiguate flow-due-to-ego-motion from flow-due-to-scene-motion. The
operator that maps ego-motion to expected image flow is:

```
f_rigid_imu(x) = π( R_imu · K⁻¹ · D(x) · [u, v, 1]ᵀ + t_imu ) − x
```

This operator consumes a **relative pose** `(R_imu, t_imu) = (ΔR̂, Δp̂)`, not a
black-box context vector. Raw encoder features cannot be projected through this
operator; a preintegrated SE(3) increment can. Using the preintegrator is what makes
the IMU branch geometrically compatible with the rest of the method rather than a
hand-wave.

Equally important, preintegration produces a **calibrated uncertainty** Σ_imu that the
raw encoder cannot. That matters because the head's correct response to a
poorly-preintegrated window ("short dt but large jerk; biases drifting") should be to
distrust the motion prior and fall back on visual cues. Without Σ_imu as input, the
head has no way to learn that fallback behavior — it has to treat every IMU output
as equally trustworthy. With Σ_imu, it can.

### 6.2 What IMU context the head actually receives (global)

Per frame pair:

| Quantity | Dim | What it means to the head |
|---|---|---|
| `log(ΔR̂)` | 3 | relative rotation as a tangent vector (well-behaved at identity) |
| `Δv̂` | 3 | body-frame velocity change across the pair |
| `Δp̂` | 3 | body-frame translation across the pair |
| `diag(Σ_imu)` or `chol(Σ_imu)` | 9 or 45 | self-reported trust in `(ΔR̂, Δv̂, Δp̂)` |
| `Δb_g`, `Δb_a` | 3 + 3 | AirIMU-predicted bias corrections (large ⇒ IMU noisy) |
| `dt` | 1 | window length, so every other quantity is scale-normalizable |

Flattened to `25-D` (`diag(Σ_imu)`) or `61-D` (`chol(Σ_imu)`) and projected by
`FeatureMLP` to `f_imu ∈ ℝ^{128}`. The
MLP is trainable; everything upstream is frozen.

### 6.3 FiLM vs. concatenation, depth-bin FiLM vs. global FiLM

Global FiLM (Feature-wise Linear Modulation) takes the conditioning vector `f_imu` and
produces per-channel scale and shift `(γ, β)` that modulate an activation tensor:
`x' = γ · x + β`. Two reasons it beats concatenation here:

1. **Multiplicative routing.** Concatenation injects information additively; FiLM lets
   the network gate entire feature channels on and off based on the conditioning
   signal, which matches "trust motion cues vs. trust appearance cues".
2. **Spatial uniformity matches semantics.** `f_imu` is a scene-level quantity; FiLM
   applies it uniformly in space, which is the correct inductive bias for a whole-
   camera motion summary. Concatenation would force the network to learn the
   uniformity.

**Depth-bin FiLM** is a refinement: bucket pixels by their depth into `N_bins` bins
and produce per-bin `(γ, β)` pairs. This captures that the manifestation of a given
ego-motion on image flow depends on depth — a 1 m/s translation with a 2 m point is
dramatically different from a 1 m/s translation with a 100 m point (§3.2 again). A
global FiLM can't express that, because it has to apply the same modulation to both.
Depth-bin FiLM is a synthesis: global motion summary from IMU, local modulation
informed by depth.

### 6.4 Why IMU and not a learned visual pose estimate

A second network predicting pose from the stereo pair, then feeding *that* pose's
rigid flow as input to the head, has been tried (TartanVO-style architectures). It
fails for a specific reason: the visually-estimated pose is itself corrupted by
dynamic objects — the very thing we're trying to flag — so the head ends up reading a
pose already biased by the phenomenon it's supposed to detect. IMU doesn't have this
feedback loop: accel and gyro measurements are indifferent to visual dynamics.

### 6.5 The IMU-rigid residual proxy: per-pixel closed-loop cue

The conditioning vector `f_imu` gives the head a *global* motion prior. The
preintegrated pose `(ΔR̂, Δp̂)` also lets us build a *per-pixel* signal that directly
mirrors the GT-based supervision target — turning a portion of the training label
into something the head sees at inference time too.

For every pixel `x` with depth `D(x)` from the frozen stereo backbone:

```
f_obs(x)       = x_obs_next − x
f_rigid_imu(x) = π( ΔR̂ · K⁻¹ · D(x) · [u, v, 1]ᵀ + Δp̂ ) − x
Δf(x)          = f_obs(x) − f_rigid_imu(x)
r_imu(x)       = || Δf(x) ||₂
r_imu_norm(x)  = r_imu(x) / τ(D(x))      # depth-adaptive normalization
valid(x)       = 1[D(x) valid] · 1[X_t1.x > ε_z] · 1[x_rigid_next inside image]
```

The head reads four extra input channels: `Δf_x, Δf_y, r_imu_norm, valid`. These are
the **IMU-analog of the GT residual** used as a label in `L_dyn`. At training time
the GT residual supervises the output; at inference time the IMU residual feeds the
input. Both are computed by the same geometric operator, so the head sees features it
already knows how to interpret.

### 6.6 Why this closes the train/test asymmetry

The train/test symmetry argument (§6 opener) requires the head's *input distribution*
to be the same at train and test. If the head is trained with only `f_imu` and frozen
features, it has no explicit geometric channel to key off — the only signal carrying
"this is where the rigid prediction disagrees with observation" lives in the loss. A
head trained that way can still do reasonable work on in-distribution data, but has
no robust way to respond to out-of-distribution appearance because it has to invent
the rigid-vs-observed comparison internally from context features.

The proxy moves that signal into the forward pass, turning the training asymmetry into
a *marginal* asymmetry: at training, the GT residual also supervises the output,
providing an extra label beyond the proxy input; at inference, only the proxy input
exists. Both directions of the forward pass are the same.

This is the final piece that closes the loop on "how can you train with GT pose and
deploy without it". The factorized answer is:

- the head's *inputs* do not depend on GT pose (frozen features + IMU proxy),
- the head's *label* at training is the GT residual, used *only* to compute the loss,
- the proxy input is a noisy but faithful estimate of what the GT residual would say.

### 6.7 Why Σ_imu still matters even with the proxy

The proxy's reliability is a function of Σ_imu. When Σ_imu is large (short dt with
jerky motion, bias drift, time-sync issues), `f_rigid_imu` is untrustworthy and the
head should ignore the proxy channels. Feeding Σ_imu through `f_imu` gives the head
the global gating signal it needs to decide how much weight to put on the proxy
pixel-wise. Without Σ_imu the head has no way to express "I should trust the proxy
less on this window" and will end up treating every preintegration equally.

### 6.8 Why the IMU branch is "optional but load-bearing"

The config exposes `imu_fusion: none` as a switch. That ablation is useful because it
tells you how much of the head's performance comes from visual features alone vs the
IMU prior. Empirically the expectation is:

- **Smooth, mostly translational motion**: IMU adds a modest bump; visual features do
  most of the work.
- **Fast rotations, sharp jerks, motion blur**: IMU becomes dominant. Visual features
  are ambiguous in those regimes, and IMU is the only reliable pose signal.

So the ablation isn't pro-forma — it is expected to show a regime-dependent gap, and
that gap is part of the scientific story.

---

## 7. The H/8 recurrence and the H/4 refinement

### 7.1 Why temporal recurrence

A feed-forward head run independently per frame has to re-derive its belief about
every pixel each frame, burning capacity to recompute the same answer. A recurrent
head lets the belief propagate:

1. **Dynamic objects are temporally coherent.** A car is a car for more than one
   frame. A walking person keeps walking. The head's belief about a pixel is highly
   correlated with its belief about the warped pixel from the previous frame. The
   ConvGRU hidden state encodes that belief — a strictly smaller learning problem.
2. **Dynamicness is sometimes only visible over time.** A person standing next to a
   stopped car — both look static in one frame. The car starts moving at `t+1`. A
   feed-forward head flags the car exactly at `t+1`. A recurrent head that's been
   tracking the region can, in principle, anticipate the transition from earlier
   subtle cues (engine vibration? wheel turning?) and can be slower to flag the
   person when they finally step.
3. **Temporal smoothing without a hard constraint.** A non-recurrent head produces
   jittery frame-to-frame masks, which break downstream PGO gating (IRLS expects
   stable weights). A ConvGRU smooths temporally *for free* as a byproduct of
   architecture, not loss. This is why the spec deliberately omits `L_temporal` — the
   recurrence already does the job.

### 7.2 Why ConvGRU, not ConvLSTM

ConvGRU has one hidden state; ConvLSTM has hidden + cell state. LSTM's extra cell is
useful when the recurrence needs a *long-horizon memory* separable from its
*short-horizon output*. Dynamic-object tracking over a ~4-frame BPTT window has no
meaningful long-horizon memory to separate. ConvGRU is cheaper, has fewer
hyperparameters, and empirically matches ConvLSTM on shorter horizons. The config
exposes ConvLSTM as a fallback if hidden-state dynamics drift over longer sequences.

### 7.3 Why `window_len = 4`

BPTT over `W` frames flows gradients through `W` recurrence steps. Longer windows let
the head learn longer-horizon dependencies but cost memory and compute linearly.
Shorter windows underfit the temporal aspect. `W = 4` is pragmatic:

- Relevant temporal structure (object persistence across frames) is captured in 2–4
  frames.
- Memory stays well within GPU limits at full resolution.
- Dynamic-object behavior beyond 4 frames is dominated by scene-level drift that the
  head shouldn't try to model — that's ODE integration territory.

The hidden state is explicitly reset at the start of each training window. Small
cost: the first frame of each window runs with zero hidden state and underperforms
slightly. Large benefit: the training loop cannot learn to rely on infinitely long
hidden state that won't exist at inference.

At inference there is no such reset — the hidden state runs indefinitely across the
sequence, resetting only at explicit sequence boundaries or long IMU-dt gaps. This
train-test mismatch (windowed training, streaming inference) is standard for
recurrent models and generally doesn't cause problems.

### 7.4 Why H/4 refinement on top of H/8 recurrence

H/8 recurrence is computationally efficient and good for temporal memory. But its
spatial resolution is coarse: an H/8 map at 480 × 640 input is 60 × 80, which means
every output pixel covers an 8 × 8 image patch. Object boundaries and thin structures
(a pedestrian leg, a bicycle frame, the edge of a moving car) are smaller than that
8-pixel support, and those are precisely the pixels that cause the most damage to
pose estimation — small, high-residual, and right at a dynamic-static transition.

The refinement branch is a coarse-to-fine compromise:

```
temporal reasoning at H/8 (cheap, recurrent) + spatial sharpening at H/4 (targeted, feed-forward)
```

It does three things:

1. Bilinearly upsamples the H/8 `logit_coarse` (1 channel, pre-sigmoid) to H/4.
2. Concatenates with H/4 FlowFormer features (which have sharper spatial detail than
   the H/8 context features) plus the H/4 IMU proxy channels.
3. Passes through a *small residual refiner* that outputs `Δ` (1 channel). The
   final output is `c = sigmoid((logit_coarse_upsampled + Δ) / T)`.

In the factorized ablation the H/4 refiner produces two channels `Δ_static,
Δ_visible` with the same zero-init residual structure, and the final output is
`c = p_static · p_visible`. The spec refiner's `out_ch` is the only thing that
changes.

### 7.5 Why zero-init residual

The refiner is initialized so that `Δ = 0` at init. This matters:

- **Identity inductive bias.** At init the refinement is the identity (exact upsample
  of H/8). Training gradually introduces the sharpening where it helps, rather than
  fighting the coarse path for every pixel.
- **No bootstrapping penalty.** If we initialized `Δ` with a random mean-0 distribution
  the refiner would disrupt an already-sensible H/8 output before it learned to
  refine, introducing a spike in the loss early in training and a much longer
  convergence time.
- **Easier ablation.** Disabling the refiner at inference (set `Δ = 0`) recovers the
  coarse output exactly. The ablation A5 `refinement.enabled: false` reads exactly
  this setting at runtime.

### 7.6 Why no recurrence at H/4

Running the ConvGRU at H/4 would quadruple memory and compute for a feature the coarse
recurrence already captures at H/8. Temporal structure does not gain resolution by
refining spatially; it gains resolution by sampling more frames. Keeping H/4
feed-forward isolates the refinement to *spatial* sharpening and leaves the temporal
reasoning undisturbed.

### 7.7 Why the H/4 path adds negligible inference cost

The refiner is shallow (two conv + GELU + conv, projecting H/4 features to 1
channel in the default single-head design; 2 channels under `A_fact`). On a 640 ×
480 input the H/4 map is 160 × 120 — about 4× the pixels of the
H/8 map but a much smaller network. Benchmarks put the refiner at under 2 ms per
frame on a modern GPU. The rest of the system budget dominates.

---

## 8. DRT-VIO-Init: bootstrap theory (stereo-adapted)

### 8.1 Why initialization matters for VIO

A VIO backend that includes preintegration factors has a state space per keyframe `x_k
= (R_k, p_k, v_k, b_gk, b_ak)`. The IMU residual couples the pose, velocity, biases,
and gravity across keyframes nonlinearly — specifically, through the lift/log of
SO(3), a quadratic term in dt, and the bias-corrected expansion in §9.

Nonlinear least squares with bad initialization does not merely converge slowly; it
converges to *wrong local minima*. In particular:

- A bad initial gyro bias produces a drifting pre-integrated rotation and a pose
  estimate that rotates in the wrong direction, which then biases the visual
  residuals away from the correct Jacobians.
- A bad initial gravity direction injects systematic acceleration error into `Δv̂` and
  `Δp̂`. Over even a few frames this creates a position drift of order
  `0.5 · g_err · dt²` per frame, which on a 10 Hz keyframe sequence is ~5 cm per frame
  if `g_err ≈ 1 m/s²`.
- A bad initial velocity makes the first `r_v` residual large, which pulls `R_i`
  through the inter-frame coupling and destroys the visual residual's linearization
  point.

The point of DRT-VIO-Init (He et al., CVPR 2023) is to give the backend a *single
consistent set of initial states* that match the visual geometry to the inertial
pre-integration on the first `N_init` keyframes. Without it, the first sliding window
converges to something that looks plausible but is silently wrong by meters.

### 8.2 What DRT-loose provides

DRT-VIO-Init has two variants: tightly coupled (DRT-tight) and loosely coupled
(DRT-loose). The loose variant is what we use because:

- It doesn't require a separate Structure-from-Motion pipeline (which we don't have
  and don't want to add).
- Its output is what the backend actually needs: `(R_k, p_k, v_k, b_g, g_W, s)`.
- Stereo eliminates the monocular scale `s`, so the remaining quantities are directly
  metric.

The full DRT-loose algorithm, as implemented in
`references/drt-vio-init/src/initMethod/drtLooselyCoupled.cpp`, has six steps:

1. **Gyro bias estimation** via epipolar-rotation residuals.
2. **Reintegration** of IMU measurements with the solved gyro bias.
3. **Rotation chain** linking frame rotations through the rebiased IMU.
4. **LiGT SVD** for monocular translations (replaced on stereo).
5. **Linear alignment** of `[v_0..v_{N-1}, g, s]`.
6. **Gravity refinement** via a quartic polynomial under `||g|| = g_mag`.

Each step has a reason for being where it is.

### 8.3 Step 1: gyro bias via epipolar rotation residuals

Between two frames `(k, k+1)` we have:

- A set of corresponding bearing vectors `(x_k, x_{k+1})` for tracked features.
- An IMU-predicted relative rotation `ΔR̂(b_g)` between the two body frames, composed
  with the sensor extrinsic `R_bc`: `R_pred = R_bc · ΔR̂(b_g) · R_bcᵀ`.

If the camera only rotated between the two frames (no translation), the bearing
vectors would be related by `x_{k+1} = R_pred · x_k`. In general the camera also
translates, but under a small-baseline / small-translation approximation the
bearing-vector rotation is the dominant source of disparity for distant points.

The bias-solver cost function is:

```
J(b_g) = Σ_{k, pt}  ρ( || R_pred(b_g)ᵀ · x_k − x_{k+1} ||² )
```

where `ρ` is the Cauchy loss (`δ = 1e-5`). Minimizing `J` over `b_g` is a ~3-D
nonlinear least squares — small, well-conditioned, converges in a few LBFGS steps.

**Why this works without pose.** The residual only involves rotations and bearings.
Any translational component projects onto a direction orthogonal to the rotation we're
trying to solve for, and the Cauchy loss downweights the distant points where
translation matters least. So the estimator effectively ignores translations and
isolates the rotation-only information.

**Why Cauchy and not Huber.** Cauchy has a faster asymptotic falloff, which at
`δ = 1e-5` essentially zeroes out any pair whose residual is larger than a few times
the noise floor. That matters because this is *bootstrap* — we have no dynamic-
object rejection yet, so some tracked points are on moving objects. Cauchy's
aggressive falloff lets us absorb those pairs without explicit outlier rejection.

### 8.4 Step 2: reintegration

With `b_g` solved, we reintegrate the IMU between each pair `(k, k+1)` with the new
bias. This is a deterministic operation — no iteration, no solver — and produces the
`imu_meas[k→k+1] = (ΔR̂, Δv̂, Δp̂, Σ_imu)` with the corrected gyro bias. The accel bias
is still zero at this point; DRT doesn't estimate it.

### 8.5 Step 3: rotation chain

We now know every `ΔR̂_{k,k+1}` in body frame. Converting to camera frame and
chaining from a common origin:

```
R_{0,C} = I
R_{k,C} = R_{k-1,C} · (R_bcᵀ · ΔR̂_{k-1,k} · R_bc)         (cam-in-world)
R_{k,B} = R_{k,C} · R_bcᵀ                                 (body-in-world)
```

**Why chain and not independent estimate.** A naive per-pair rotation estimate would
exploit only local information. Chaining via preintegration exploits the *entire
history* of IMU measurements for each keyframe — `R_{k,B}` is the product of all
preintegrations from 0 to k — which is almost always more accurate than any single
2-frame estimate, particularly at high rotation rates.

### 8.6 Step 4: stereo-metric translation (replaces LiGT SVD)

In the monocular case, DRT-loose solves a large SVD (see `evectors.middleRows<3>(3*i)`
in the reference) to recover the translations up to a global scale. With stereo, we
already have metric depths for every tracked feature — so for every pair `(k, k+1)`
we can recover `t_rel_k` directly with a 1-step `Reproj_TwoFramePGO` (or equivalent
PnP estimator) using the known depths. The monocular SVD and its scale ambiguity
disappear entirely.

**Why we can still skip the LiGT.** LiGT is solving for positions *up to scale* from
bearings. Stereo gives us absolute scale from the baseline. There is no information
LiGT provides that the stereo backbone doesn't already have. Running LiGT anyway would
just add an unnecessary SVD and mix in information we don't need.

### 8.7 Step 5: linear alignment of velocity and gravity

With rotations and positions in hand, the remaining unknowns are the per-keyframe
velocities `v_k`, the gravity `g_W`, and (monocular only) the global scale `s`. The
linear alignment reads:

```
[I_3 dt    0       R_iᵀ · dt²/2 · g_mag    (R_iᵀ (p_j − p_i))/100]   [v_i ]   [ imu.Δp + R_iᵀ R_j · p_bc − p_bc ]
[I_3      -R_iᵀR_j R_iᵀ · dt   · g_mag    0                       ] · [v_j ] = [ imu.Δv                          ]
                                                                      [g   ]
                                                                      [s   ]
```

(rows 1, 2 are position- and velocity-consistency residuals; p_bc is the body-to-cam
translation of the extrinsic.) The derivation follows from the two IMU kinematic
identities:

```
p_j = p_i + v_i · dt + 0.5 · g · dt² + R_i · Δp̂
v_j = v_i + g · dt + R_i · Δv̂
```

substituting `p_j − p_i = (R_jᵀ R_i)ᵀ (...)` and stacking across all consecutive
pairs. On stereo we fix `s = 1` and drop the last column, which makes the system
overdetermined. `lstsq` solves for `(v_0..v_{N-1}, g)` simultaneously.

**Why solve simultaneously, not iteratively.** Velocity and gravity enter the
identities linearly; an iterative alternating solve would work but is slower and has
weaker convergence guarantees. The joint `lstsq` is closed-form and unbeatably fast
at `N_init = 10` keyframes.

### 8.8 Step 6: gravity refinement via quartic polynomial

`lstsq` gives an unconstrained `g`, but we know `||g|| = g_mag` (~9.8 m/s² for Earth).
The refinement enforces this constraint. The derivation parameterizes `g` in a local
tangent to the unit sphere around `g_0 = g_lstsq / ||g_lstsq||` and re-solves with the
unit-norm constraint, which after elimination reduces to a quartic polynomial in a
scalar. Newton or direct root-finding on a quartic is trivial and always has real
roots in the relevant regime.

Why refine even on stereo: the linear alignment is unconstrained in `||g||`, and
`lstsq` noise can push the gravity magnitude off by a few percent. The quartic
refinement brings it back exactly to `g_mag`, which is what the backend factor's
kinematic formulas assume. A 3% error in `||g||` over a 10 Hz, 5 m/s² maneuver
accumulates into a ~15 cm/s velocity error per keyframe — non-negligible.

### 8.9 Step 7: align to gravity-level world

Once `g_W` is known, rotate the whole state so that `g_W = [0, 0, -g_mag]` in the
output world frame. This is a gauge choice, but it's the gauge every downstream
consumer expects.

### 8.10 Why leave accel bias at zero

DRT-loose does not estimate `b_a`. Three reasons:

1. **Accel bias and gravity are nearly unobservable from a short initialization
   window.** Both produce constant-ish apparent accelerations over short windows; the
   gravity magnitude prior distinguishes them only at higher rotations. With `N_init
   = 10` keyframes and mostly straight motion, the observability is weak.
2. **The backend handles it.** Once the backend is running with preintegration
   factors and a random-walk bias prior, `b_a` becomes observable as the motion
   diversifies (turns, jerks). Letting the backend estimate it online under a tight
   bias-walk prior is more robust than trying to get it from a short bootstrap.
3. **VINS-Mono does exactly this.** The choice is not novel; it is a well-validated
   design decision.

Consequence: the first few backend windows will have mildly mis-estimated `Δp̂` due to
the bias-zero assumption. This manifests as a small `r_p` IMU residual that the
backend corrects by pulling `b_a` away from zero. The random-walk prior then allows
`b_a` to be non-zero on all subsequent keyframes without penalty.

### 8.11 Accept/reject: why this matters

The bootstrap can fail silently: the linear alignment can be singular if all `R_i`
are near-identity (no motion), gyro bias estimation can diverge on heavy occlusion,
gravity refinement can lock into a wrong local minimum at high rotation rates. The
spec's accept-criteria (max `|b_g|`, stereo-baseline cross-check, minimum view count)
are there to catch these *before* the state is written into the map.

If bootstrap fails after `2 · N_init` keyframes, fall back to `method: identity` (no
initialization; backend starts cold). The head still works without init, because its
forward pass doesn't depend on the initialization state; only the backend IMU factor
is poorly conditioned, and it gradually corrects itself with more keyframes.

### 8.12 Why DRT-loose and not DRT-tight

DRT-tight couples the bootstrap with a SfM-style visual solve for positions, which
is more accurate on monocular. On stereo, there is no scale ambiguity to resolve, so
the tight coupling buys very little — and it adds a fragile SfM dependency. DRT-loose
is the right choice for stereo.

### 8.13 Why initialization runs once and never repeats

DRT-loose is a *global* rotation-chain solver with closed-form linear alignment. It
does not smoothly extend to "incremental initialization" after the backend is already
running; you can't update `b_g` with DRT's formulas once the backend has absorbed
visual residuals into the bias estimate.

After bootstrap, the sliding-window backend is the sole estimator. The initialization
state becomes the first LM-iteration initial guess; the first backend iteration
adjusts everything simultaneously from that point. There is no "re-initialization"
step in production.

---

## 9. Preintegration factors: the math behind the backend constraint

### 9.1 The shared preintegrator

The single most important architectural invariant in the whole method is **one
preintegrator instance, two consumers**. The frontend reads its output as a
conditioning tensor; the backend reads its output as an optimization constraint. If
the two consumers saw slightly different preintegrations, the system would be
internally inconsistent: the head would condition on "the IMU says the camera moved X"
while the backend solved "subject to the IMU says the camera moved Y".

Invariant I2 in the spec enforces this: a single `DifferentiablePreintegrator` class
exists at `Module/Network/AirIMU/preintegration.py` with a `share_with_backend: bool`
flag. When true, it additionally emits the bias Jacobians `J_R_bg, J_v_bg, J_v_ba,
J_p_bg, J_p_ba`; when false, it skips them to save compute.

### 9.2 Forster preintegration refresher

Forster et al. (T-RO 2017) derived the preintegration factor as a concise summary of
an inertial window: given a sequence of IMU samples `(â_m, ω̂_m)` with corrections
`Δb_a, Δb_g` between keyframes `i` and `j`, the exact kinematic integration gives:

```
R_j = R_i · Π_m Exp( (ω̂_m − b_g − η_g_m) · Δt_m )
v_j = v_i + g · dt_ij
        + R_i · Σ_m Exp(...) · (â_m − b_a − η_a_m) · Δt_m
p_j = p_i + v_i · dt_ij + 0.5 · g · dt_ij²
        + R_i · Σ_m [Exp(...) · (â_m − b_a − η_a_m) · 0.5 · Δt_m²
                       + Σ_{n<m} Exp(...) · (â_n − b_a − η_a_n) · Δt_n · Δt_m]
```

The key observation is that the summands depend on `R_i`, `v_i`, `p_i`, and the
biases only through the *preintegrated* quantities:

```
ΔR̂_{ij} = Π_m Exp( (ω̂_m − b̂_g) · Δt_m )
Δv̂_{ij} = Σ_m ΔR̂_{im} · (â_m − b̂_a) · Δt_m
Δp̂_{ij} = Σ_m [ΔR̂_{im} · (â_m − b̂_a) · 0.5 · Δt_m² + Δv̂_{im} · Δt_m]
```

where `(b̂_g, b̂_a)` are the bias estimates at linearization time — the
"bias_ref" in our implementation.

These three tensors are **independent of the keyframe state `(R_i, v_i, p_i)`** and
depend on biases only through the reference values. They can be computed once when
the IMU window is received, stored with the factor, and reused at every LM iteration
without re-integrating the raw samples. That's the entire reason preintegration
exists: it reduces a per-LM-iteration O(N_imu) integration to a per-window
computation plus an O(1) evaluation per iteration.

### 9.3 First-order bias correction

If the backend updates the bias estimate from the reference, we don't want to
re-integrate. Instead, we use a first-order Taylor expansion around `(b̂_g, b̂_a)`:

```
ΔR̂(b_g) ≈ ΔR̂(b̂_g) · Exp( J_R_bg · (b_g − b̂_g) )
Δv̂(b_g, b_a) ≈ Δv̂(b̂_g, b̂_a) + J_v_bg · (b_g − b̂_g) + J_v_ba · (b_a − b̂_a)
Δp̂(b_g, b_a) ≈ Δp̂(b̂_g, b̂_a) + J_p_bg · (b_g − b̂_g) + J_p_ba · (b_a − b̂_a)
```

The Jacobians `J_R_bg, J_v_bg, J_v_ba, J_p_bg, J_p_ba` are computed alongside the
preintegrated quantities and stored with the factor. When the backend evaluates the
factor at iteration `k`, it reads `(b_gk, b_ak)`, computes `(dbg, dba) = (b_gk −
b̂_g, b_ak − b̂_a)`, and corrects the factor cheaply.

**Why this works.** The exact relationship between the preintegrated quantities and
the bias is nonlinear (the bias shows up inside `Exp` and `Σ_m ΔR̂_{im}`), but it is
smooth in a neighborhood of the reference bias. As long as the updated bias stays
within a few standard deviations of `b̂_g, b̂_a`, the first-order expansion is
accurate to machine precision for our purposes.

**When to re-linearize.** If the bias estimate drifts too far from the reference —
empirically, more than `3 · σ_bgw` — the first-order correction becomes inaccurate
and the factor should be re-preintegrated with the new reference. In practice this
happens rarely (biases change slowly under the random-walk prior) and is handled by
periodically replacing the bias reference when a new keyframe is marginalized in.

**Why the head-only preintegrator doesn't emit Jacobians.** The head consumes the
factor as a conditioning vector. It never updates the bias reference; it always sees
the preintegrator's current output, bias-reference and all. There's no re-
linearization step for the head, so computing the Jacobians would be wasted work.
`share_with_backend: false` skips them.

### 9.4 The 15-D inertial residual

For each consecutive keyframe pair `(i, j)` with a stored `IMUEdge` containing
`(ΔR̂, Δv̂, Δp̂, Σ_imu, dt, J_R_bg, J_v_bg, J_v_ba, J_p_bg, J_p_ba, bias_ref)`:

```
dbg = b_gi − b_g_ref                                        # bias updates
dba = b_ai − b_a_ref

ΔR̂_corr = ΔR̂ · Exp(J_R_bg · dbg)                           # bias-corrected expectations
Δv̂_corr = Δv̂ + J_v_bg · dbg + J_v_ba · dba
Δp̂_corr = Δp̂ + J_p_bg · dbg + J_p_ba · dba

r_R  = Log( ΔR̂_corrᵀ · (R_iᵀ · R_j) )                      (3-D, SO(3) tangent)
r_v  = R_iᵀ · ( v_j − v_i − g_W · dt )  −  Δv̂_corr          (3-D)
r_p  = R_iᵀ · ( p_j − p_i − v_i · dt − 0.5 · g_W · dt² )    (3-D)
         −  Δp̂_corr
r_bg = b_gj − b_gi                                          (3-D, random walk)
r_ba = b_aj − b_ai                                          (3-D, random walk)

r_imu = [r_R ; r_v ; r_p ; r_bg ; r_ba]                     (15-D)
```

Each residual has a clean interpretation:

- `r_R` — "in the body frame of frame `i`, the rotation predicted by IMU minus the
  rotation observed between the backend's `R_i` and `R_j` states". Lives in the
  tangent space at identity; behaves linearly for small rotation errors.
- `r_v` — "the IMU's predicted velocity change, corrected for gravity, minus the
  backend's velocity change". Scales linearly in `dt`.
- `r_p` — "the IMU's predicted position change, corrected for gravity and a velocity-
  times-dt term, minus the backend's position change". Scales linearly in `dt` (via
  `v_i · dt`) and quadratically in `dt` (via `0.5 · g · dt²`).
- `r_bg`, `r_ba` — the random-walk priors. The continuous-time random walk has
  variance `σ²_bw · dt` per time step; this discrete-time residual has unit variance
  under the covariance `σ²_bw · dt · I_3`.

### 9.5 The 15×15 information matrix

`Σ_imu` is a 9×9 covariance over `(r_R, r_v, r_p)`, propagated forward by the
Forster recursion. For the bias random walks, build a diagonal 6×6 block with variance
`σ²_bgw · dt` and `σ²_baw · dt` — the continuous-time random walks integrated to
discrete time. The total information matrix is block-diagonal:

```
Σ_imu_full = diag(Σ_imu_9×9, σ²_bgw · dt · I_3, σ²_baw · dt · I_3)   ∈ ℝ^{15×15}
```

The backend's cost contribution from one IMU factor is:

```
c_imu = r_imuᵀ · Σ_imu_full⁻¹ · r_imu          (scalar)
```

which is the Mahalanobis norm in 15-D.

**Why block-diagonal, not fully coupled.** The bias random walks are, by
construction, independent of the preintegrated rotation/velocity/position. Cross-
terms between `(r_R, r_v, r_p)` and `(r_bg, r_ba)` would only appear if we modeled a
correlated random walk, which we don't. Keeping them independent gives `Σ_imu_full` a
clean block structure and makes the backend's `pinverse` step numerically trivial.

### 9.6 Why preintegration is 1st-order in bias, not zero-order

One could imagine a simpler factor: compute the preintegration with `b̂_g = 0, b̂_a =
0` and just set the bias residual to `r_bg = b_gi`, `r_ba = b_ai` (i.e. a prior on
the bias itself being small). This would avoid the need for `J_R_bg, J_v_bg, ...`.

The problem: gyro biases of a consumer-grade MEMS IMU are typically 0.01–0.1 rad/s,
which over a 1 s window is a `ΔR̂` error of ~0.05 rad (~3°). Not small. A zero-order
approximation would have the head and backend fighting persistent systematic errors
in `ΔR̂, Δv̂, Δp̂`. The first-order correction absorbs these errors into the
optimization by letting the backend update `b_gi` and the factor self-corrects.

### 9.7 Why include the bias random-walk terms

Without `r_bg = b_gj − b_gi`, the backend has no mechanism to couple the bias across
keyframes. Each keyframe's bias would float independently, and the bias estimate
would be pinned by the inertial residual alone — which is a weak constraint for bias
(it's coupled only through the Jacobians, which are O(dt)).

With the random-walk term, the bias at keyframe `j` is softly tied to the bias at
keyframe `i` with variance `σ²_bgw · dt`. This regularizes the bias trajectory
across the window and prevents it from chasing noise. It's also physically correct:
IMU biases drift according to a random walk with known variance from the IMU
datasheet.

### 9.8 The two-frame PGO case (Stage A)

The migration plan starts with the simplest usable backend: the existing
`Reproj_TwoFramePGO` plus one preintegration factor between the two frames.

The existing factor graph has one optimized parameter (`pose2opt: pp.SE3`) and
visual residuals. The new factor graph `Reproj_TwoFramePGO_IMU` adds three more
optimized parameters (`v_j, b_gj, b_aj` — the state at the newer frame) and the
15-D IMU residual as described above. The state at the older frame is held fixed
(buffered, not optimized) as the anchor; both frames' states come from the current
`FrameNode` and `FrameNode`'s previous-frame neighbor in the map.

**Why two-frame is a useful stage.** The algorithmic complexity of the sliding-window
backend (marginalization, window management, multi-factor evaluation) is entirely
orthogonal to the mathematics of the IMU factor itself. Stage A isolates "does the
IMU factor work at all" from "does the sliding-window backend work". Every bug fix
and tuning done at Stage A transfers to Stage B directly.

**Why the anchor frame is held fixed.** Two-frame PGO has a natural gauge: the world
frame is pinned at frame `i`'s pose. Making `(R_i, p_i, v_i, b_gi, b_ai)` buffers
(not parameters) enforces that gauge and halves the parameter count. This is not a
modeling compromise; it's the correct thing to do.

**What about the ICP and ReprojDisp variants?** Stage A only augments `Reproj`, not
`ICP_TwoframePGO` or `ReprojDisp_TwoFramePGO`. Adding IMU to the other variants is
straightforward (the IMU factor is orthogonal to the choice of visual factor), but
it triples the Stage-A patch size for no scientific gain. If the final system uses
one of the other variants at inference (§10 defaults to `Reproj`), extending is a
one-class edit.

### 9.9 The sliding-window backend (Stage B)

Stage B generalizes Stage A to a window of `W_kf` keyframes. The per-keyframe state
expands:

```
x_k = {R_k ∈ SO(3), p_k ∈ ℝ³, v_k ∈ ℝ³, b_gk ∈ ℝ³, b_ak ∈ ℝ³}
```

Parameters per window: `W_kf × (6_pose + 3_v + 3_bg + 3_ba) = 15 · W_kf`. At the
default `W_kf = 8`, that's 120 parameters. Residuals:

- Visual: ~8 × 200 × 2 ≈ 3200 scalar residuals (c-weighted reprojection).
- IMU: 7 factors × 15 scalars = 105 scalar residuals.
- Marginalization prior: dimension depends on the evicted state; typically 60–90 scalars.

Levenberg-Marquardt on 120 parameters with ~3400 residuals runs in tens of milliseconds
on CPU via pypose.

### 9.10 Marginalization: Schur complement theory

When the window slides and the oldest keyframe is evicted, we don't want to simply
drop its state — that would throw away information that still affects the remaining
frames through shared factors (visual observations of points that are seen by both
old and new frames, IMU factors that couple them). Marginalization keeps the
information as a linear prior on the remaining states.

Given the stacked residual Jacobian `J` and covariance `Σ`, the information matrix
around the current linearization is:

```
H = Jᵀ · Σ⁻¹ · J
b = Jᵀ · Σ⁻¹ · r
```

Partition into evicted state `e` and remaining state `r`:

```
H = [H_ee  H_er]      b = [b_e]
    [H_re  H_rr]          [b_r]
```

The Schur complement of the evicted block is:

```
H_marg = H_rr − H_re · H_ee⁻¹ · H_er
b_marg = b_r  − H_re · H_ee⁻¹ · b_e
```

`(H_marg, b_marg)` define a quadratic prior on the remaining state that exactly
reproduces the information the evicted state contributed, linearized around the
current estimate. The backend adds this prior as an extra "residual" block:

```
r_marg(x_r) = H_marg · (x_r − x_r_lin) − b_marg
Σ_marg⁻¹   = H_marg     (identity information)
```

which, when multiplied into the Mahalanobis norm, gives exactly the Schur-complement
quadratic.

**Why this matters for correctness.** Without marginalization, the backend would
silently lose the information from every evicted keyframe. On long trajectories this
manifests as drift that grows faster than the square root of time — because the
system keeps "forgetting" constraints that would otherwise tie the current state to
early anchor points. Marginalization (and its linearization point) is a first-order
surrogate for the full information, and with the right implementation it's
indistinguishable from the full non-marginalized solve on the scale of one window.

**Why this is not a permanent prior.** The marginalization point is the state
estimate at the moment of eviction. If subsequent iterations move the remaining
states far from that point, the Schur approximation becomes inaccurate (because it's
linearized around the old estimate). The common fix is to periodically re-linearize
the marginalization by re-solving from scratch on the current window. In practice
this is rarely needed — the LM's own linearization re-solves within the window, and
the marginalization prior contributes only the evicted information.

### 9.11 Why close the IMU loop in the backend, not just the frontend

A frontend that conditions on IMU without a backend IMU factor is *informed* by IMU
but the optimized pose is still decided entirely by visual residuals. In a
dynamic-heavy scene where the head correctly downweights most visual observations,
the residual signal to the solver is weak and the pose estimate drifts. An IMU
factor in the backend provides a second, independent constraint that doesn't
depend on visual observations at all — so even in the pathological case where the
head masks out 90% of pixels, the IMU factor still constrains the pose.

This is why the backend factor is not optional. Without it, "static confidence head
+ visual-only PGO" is a fragile system that works when the head is well-trained and
fails badly otherwise. "Static confidence head + visual PGO + inertial factor"
degrades gracefully — the IMU factor never fails silently; at worst it has a large
residual that the backend absorbs into bias updates.

### 9.12 Adaptive IMU covariance (Stage C)

Stage B uses a fixed `Σ_imu_full` per factor (determined by the preintegrator's
propagation). Stage C adds an adaptive inflator: compute the innovation between the
IMU's prediction and the visual solve's consensus, and inflate `Σ_imu_full` by a
factor proportional to the innovation's Mahalanobis norm.

This is not covered in depth here because Stage C is a stretch goal. The principle
is the same as innovation-based covariance scaling in Kalman filters: if the IMU
persistently disagrees with the visual evidence, the IMU is probably noisier than
its self-reported `Σ_imu` claims (bad time sync, calibration drift), and the backend
should trust it less. The adaptive inflator catches this without a hand-tuned
override.

---

## 10. The loss: why these terms and why these weights

### 10.1 Why more than one term

A single-term loss (GT-residual focal BCE on `c`) is theoretically sufficient.
Additional terms are for **calibration and regularization**: the primary signal is
informative but noisy, and secondary signals reduce training variance without changing
the minimum.

Each additional term must earn its keep. The spec includes exactly these:

```
L = λ_dyn     · L_dyn      (focal BCE on c against GT-residual label, gated by M_valid)
  + λ_smooth  · L_smooth   (edge-aware smoothness on c)
```

The cycle-consistency signal does not appear as a loss term. It enters only through
`M_valid = M_infov & M_depth & M_cycle`, the validity mask that gates `L_dyn`. The
rest of this section explains each term, why no second supervised term is needed,
and why each of the several plausible additional terms was deliberately excluded.

### 10.2 `L_dyn` — GT-residual focal BCE (primary)

The load-bearing signal. Encodes the rigid-flow identity from §2 directly.

```
L_dyn = (focal_bce(c, c_target, γ) * M_valid).sum() / M_valid.sum().clamp(min=1)

with   c_target = 1 − sigmoid((r − τ_D(d))/κ)     (§3.3 depth-adaptive rigidity label)
       M_valid  = M_infov & M_depth & M_cycle     (§3.5 validity mask)
```

**Why focal BCE instead of plain BCE.** Dynamic pixels are a small minority — typically
1–10% in a driving scene, sometimes 0% indoors. Plain BCE averages loss over all
pixels, so the tiny dynamic minority is overwhelmed by the static majority and the
network can get 95%+ accuracy by predicting "static everywhere". Focal BCE multiplies
each pixel's BCE contribution by `(1 − p_correct)^γ`, downweighting easy (confident,
correct) pixels and letting the hard (dynamic minority) pixels dominate.

With `γ = 2`, a pixel correctly classified with probability 0.9 contributes 0.01× as
much as a pixel correctly classified with probability 0. Enough to rebalance toward
the minority class without completely discarding the majority.

**Why mask by `M_cycle`.** Occluded pixels have meaningless residuals. Including them
in `L_dyn` would train the head to predict "dynamic" wherever occlusion happens,
which is the wrong semantics for rigidity. The mask lets occluded pixels *abstain*
from supervision — the head learns nothing about them in either direction, and the
ConvGRU's temporal prior is free to carry whatever prediction it had from prior
frames.

### 10.3 Why no second supervised term

A natural impulse is to add a cycle-consistency BCE that supervises a separate
visibility head `p_visible`. We deliberately do not, for reasons covered in §5 and
summarized here:

- The solver consumes one scalar `c = w`. Whether `c` came from a product
  `p_static · p_visible` or directly from a single head, the downstream numerical
  behavior is identical.
- The job of a `p_visible` supervisory term is to abstain on unreliable pixels. The
  `M_cycle` mask accomplishes the same abstention with zero added parameters and
  one fewer temperature to calibrate.
- The noise decomposition concern (depth-label noise vs. cycle-label noise) is
  still satisfied: cycle noise only gates rigidity supervision; it never acts as a
  label.

The factorized variant is preserved as ablation `A_fact` (spec §10): with
`dynamic_head.factorize: true`, the head emits `(p_static, p_visible)` and
`L = λ_s L_static + λ_v L_visible + λ_sm L_smooth` (optionally with a joint BCE).
If that ablation wins on ATE/RPE by a meaningful margin on dynamic sequences, the
factorization becomes the default. The single-head design is the null hypothesis.

### 10.4 `L_smooth` — edge-aware smoothness on `c`

Smoothness regularizes the output to be locally coherent. Naive smoothness
`||∇c||` would discourage sharp transitions between dynamic objects and static
backgrounds — exactly where we want sharp transitions. Edge-aware smoothness weights
the penalty by `exp(−||∇I||)`, which is near 0 on image edges and near 1 in smooth
regions.

The effect is: "be smooth where the image is smooth, be sharp where it has edges".
Same trick as depth estimation, normal estimation, semantic segmentation. Image edges
tend to align with object boundaries, which is where `c` should transition.

**Why regularize `c` directly.** `c` is what the solver consumes, so the smoothness
regularizer targets that quantity. In the factorized variant the same argument says
we regularize the product `c = p_static · p_visible`, not the factors separately —
smoothness on `p_static` alone could produce a `c` with spurious transitions where
`p_visible` changes sharply.

### 10.5 What the loss does *not* include

- **Proxy segmentation BCE.** Requires per-pixel dynamic/static class labels, which
  only synthetic datasets have. Would make the method unable to train on real data
  and would silently train the head to match a synthetic-dataset ontology — the
  exact failure mode we're avoiding by not using category-based masking.
- **Entropy regularization `−H(c)`.** Encourages confident outputs (near 0 or 1).
  Sounds sophisticated but typically harms calibration. A well-calibrated output has
  some pixels near 0.5 (the ambiguous ones), and that ambiguity is useful information
  for downstream gating. Encouraging confidence for confidence's sake degrades this.
- **Temporal consistency L1 `||c_t − warp(c_{t-1}, f_obs)||₁`.** Same role as
  the ConvGRU hidden state (§7.1.3). Redundant and introduces a weight that fights the
  BCE term.
- **Laplace NLL over the residual.** The CoProU-VO paper uses this because it
  predicts an uncertainty, not a probability. The right loss for an uncertainty is
  NLL; for a probability is BCE. `c` is a probability. NLL here would just be a
  worse BCE.
- **Photometric reprojection loss.** FlowFormerCov is already trained with flow
  supervision; its flow output is cleaner than a photometric residual. Adding a
  photometric term re-litigates a problem already solved upstream.
- **A separate `p_visible` supervised head by default.** Covered in §10.3 and §5.

---

## 11. PGO integration: why the math works

### 11.1 The factor graph setup

MAC-VO's two-frame PGO solves, for a batch of 2D-3D observations, the pose `T` that
minimizes:

```
L(T) = Σ_i  r_i(T)ᵀ · Σ_i⁻¹ · r_i(T)
```

where `r_i(T)` is the i-th observation's reprojection residual under `T` and `Σ_i` is
its 2×2 pixel-space covariance. In the code (`Module/Optimization/TwoFramePGO/Graphs.py`),
`r_i` is returned by `forward()` and `Σ_i` by `covariance_array()`. pypose's LM
handles the rest.

### 11.2 What weighting a residual actually does

Suppose we multiply each factor's residual by `w_i = c_i ∈ [0, 1]`:

```
r_i'(T) = w_i · r_i(T)
L'(T)   = Σ_i (w_i · r_i)ᵀ · Σ_i⁻¹ · (w_i · r_i)
        = Σ_i w_i² · r_iᵀ · Σ_i⁻¹ · r_i
```

Equivalently, leave the residual alone and replace `Σ_i` with `Σ_i / w_i²`:

```
L''(T) = Σ_i r_iᵀ · (Σ_i / w_i²)⁻¹ · r_i  =  Σ_i w_i² · r_iᵀ · Σ_i⁻¹ · r_i  =  L'(T)
```

So **"multiply the residual by `w`"** is mathematically identical to **"inflate the
covariance by `1/w²`"**. Two views of the same whitening transform. The spec uses the
residual-multiplication path because that's what pypose's factor graph naturally
expresses.

### 11.3 Why the `w²` scaling is the right behavior

A factor with `w = 1.0` contributes its full log-likelihood. With `w = 0.5`, 0.25× —
equivalent to its covariance being 4× larger in every direction, i.e. "I trust this
observation half as much, so its variance is 4× wider". With `w = 0.1`, 0.01× —
essentially dead. Quadratic falloff is exactly what you want: near-static pixels count
fully, suspicious ones fade quickly, dynamic ones vanish.

A *linear* scaling (`w_i` instead of `w_i²`) falls off too slowly. A pixel with 50%
dynamic confidence would still contribute half its weight, which is too generous
given how damaging dynamic observations are to pose estimation.

### 11.4 Why the factor cannot be truly zeroed

If `w_i = 0`, the scaled covariance is `Σ / 0 = ∞`, which pypose can't represent.
Three options:

1. **Drop the factor before it enters the graph.** Hard threshold at selection.
2. **Soft floor `w_i = max(w_i, ε)` with `ε = 1e-3`.** Pypose sees a finite (huge)
   covariance instead of infinity. The factor is near-dead but technically present.
3. Trust-region tricks in LM. Not supported by pypose.

The spec uses **both (1) and (2)**:

- **Stage 1 (hard mask at `MACVO.run_pair`):** drop any observation with
  `c < min_c`. Observations the head is highly confident are dynamic
  never enter the graph.
- **Stage 2 (soft weight inside the factor graph):** for observations that survived,
  multiply residual by `c`. Safety floor `W_EPS = 1e-3` protects pypose from
  infinite covariance in ablation configs with `min_c = 0`.

### 11.5 Why both stages

- **Stage 1 alone** wastes optimization signal at boundary pixels ("probably static
  but slightly suspicious") by forcing a binary decision. Some pixels are genuinely
  ambiguous — a parked motorcycle that might drive off any second, a pedestrian about
  to walk — and the head's calibrated probability is the right way to express that.
- **Stage 2 alone** is also wasteful: a pixel with `c = 0.02` contributes a factor
  with weight 0.0004, which is numerically indistinguishable from zero but still
  costs CPU time in LM iterations. Worse, a dynamic factor with tiny weight but huge
  residual can still push the LM step in a bad direction numerically.

Both: a fast path that drops obviously-dynamic pixels, and a smooth path that soft-
weights everything else. The threshold between them is `min_c`; the
ablation matrix tests hard-only and soft-only as endpoints.

### 11.6 Why not learn `w = g(c)` with a learned map

One could imagine a monotonic learned map from `c` to `w` (an idea kept as a
design note in the earlier spec revision). Pros: allows the head to adapt the
calibration-to-optimization mapping based on LM dynamics. Cons: adds parameters,
complicates calibration, and requires backpropagating through LM iterations — which
the spec explicitly avoids (freeze the backbone, train the head).

The current spec pins `w = c` directly (after temperature scaling). A learned
`g(c)` is left as a future extension.

---

## 12. Calibration: what temperature scaling actually does

The head outputs one raw logit per pixel, passed through `c = sigmoid(logit / T)` to
produce the static confidence. `T` is a single learned scalar stored as a PyTorch
buffer (not a Parameter).

At `T = 1` you get the standard sigmoid. `T > 1` flattens the output toward 0.5
(less confident); `T < 1` sharpens toward 0 or 1 (more confident).

### 12.1 Why calibration is a separate step

During primary training, the logit is optimized to minimize focal BCE, which
incentivizes the logit distribution that minimizes training loss — not necessarily
the one whose sigmoid is a calibrated probability. Deep networks trained with BCE
notoriously end up *over-confident*: the raw sigmoid is too close to 0 or 1 relative
to the empirical correctness rate.

Temperature scaling (Guo et al., 2017) is the standard post-hoc fix: freeze the head,
sweep `T` on a held-out split, pick the `T` that minimizes expected calibration error
(ECE) of `sigmoid(logit / T)` against the empirical label distribution.

### 12.2 Why calibration matters *here* specifically

In classification, over-confident outputs are a nuisance but not a functional bug —
you'll argmax anyway. Here `c` is a continuous weight on a factor's residual
(§11), and the math only works if it is an actual probability. An over-confident
head over-weights good factors and over-down-weights bad factors, producing a bimodal
mask that behaves like a hard threshold with extra steps. Calibration keeps the
soft-weighting regime meaningful.

### 12.3 Why `T` is a buffer, not a Parameter

During primary training, `T = 1` and the logit trains against BCE. During post-hoc
calibration, everything else is frozen and only `T` is swept — but the sweep is done
offline, not via gradient descent. So `T` never needs to be a `Parameter`; it's a
learned scalar set once at the end of training and never updated. That matches the
semantics of a PyTorch buffer. The spec pins this to avoid a common footgun where
someone wires `T` into the optimizer and it drifts.

### 12.4 Why one temperature suffices

With a single output head there is one logit distribution to calibrate, so one `T`
is both necessary and sufficient. The factorized ablation `A_fact` (§5) adds a
second temperature `T_vis` because `p_static` and `p_visible` are supervised by
different labels with different noise structures (GT-residual for static, cycle-
error for visible), and their optimal temperatures are generally different. A
single temperature there would force them to share a calibration and under-
calibrate both. In the default single-head design none of that applies — only `T`
exists.

---

## 13. Training data and its constraints

The head needs three things per training sample:

1. A stereo image pair (for FlowFormerCov).
2. An IMU window between the two frames' timestamps (for AirIMU + preintegration).
3. A GT relative pose in the *camera* frame (for `L_dyn`'s rigid-flow label).

Nothing else. No segmentation, no instance labels, no manual dynamic annotations.

### 13.1 Supported datasets

| Dataset | Stereo | IMU | GT pose | Use |
|---|---|---|---|---|
| **VIODE** | ✓ | ✓ | body, exact | Primary training |
| **TartanAir 2** | ✓ | optional | camera, exact | Secondary |
| **EuRoC** | ✓ | ✓ | body + `T_BS` | Held-out sim-to-real eval (A8) |
| **KITTI** | partial IMU | — | poor | Not used for training |
| **TUM-VI** | ✓ | ✓ | body + `T_BS` | Optional held-out |

### 13.2 Why synthetic primary, real eval

Synthetic data is easier to validate:

- GT poses are exact (not GNSS-noisy).
- Depth is exact from the renderer (not stereo-estimation-noisy).
- Dynamic objects are scripted, so we know where they are and can sanity-check
  outputs against GT even though we don't use those labels for training.

Training on synthetic and evaluating on real (EuRoC) is the canonical test for
whether the method has learned something transferable vs something that exploits
synthetic-data artifacts. A8 in the ablation matrix is exactly this test.

### 13.3 The pose-frame pitfall

`gt_R`, `gt_t` can mean several things depending on dataset:

- Body-frame pose (IMU body frame) — VIODE, EuRoC, TUM-VI.
- Camera-frame pose — TartanAir.
- World-frame pose of body — EuRoC, KITTI.

The rigid-flow formula requires `(R, t)` to be the *camera-to-camera* relative pose.
Using the wrong frame produces systematically wrong rigid flow, which produces
systematically wrong soft labels, which trains the head to flag the wrong pixels as
dynamic. Silent failure: training will "converge", the loss will look fine, but the
head will be useless at inference.

The spec's §8.3 and the deferred-items list call this out explicitly. Every training
dataset wrapper needs to audit its pose-frame convention and convert to camera-frame
relative pose before returning.

### 13.4 IMU gap handling

Real data has IMU dropouts, missing samples, clock drift between camera and IMU.
The preintegrator must reject windows with:

- `dt` greater than a threshold (default 200 ms) — otherwise the first-order
  bias correction breaks down.
- Fewer than `N_imu_min` samples in the window — insufficient data for covariance
  propagation.
- Timestamp discontinuities greater than `5 × (1/imu_rate)` — signals a dropout.

On rejection, the factor is dropped and the head's hidden state is reset. Better to
have a gap than to feed corrupted data to a frozen pretrained head that can't
recover.

---

## 14. Deployment: what the full inference path looks like

At deploy time, for each new frame pair:

1. **Stereo pair + IMU window come in** (from the dataloader or live sensor stream).
2. **FlowFormerCov (frozen)** produces flow, flow covariance, depth, and the context
   feature map.
3. **AirIMU (frozen) + DifferentiablePreintegrator** produce
   `(ΔR̂, Δv̂, Δp̂, Σ_imu, dt, J_*, bias_ref)`.
4. **IMUEncoder** projects to `f_imu ∈ ℝ^{128}` and builds per-pixel
   `(Δf, r_imu_norm, valid)` maps.
5. **StaticConfidenceHead** reads `(f_ctx, flow, cov, f_imu, proxy, h_prev)`, advances
   its hidden state, outputs `c`. No GT pose, no GT anything.
6. **MAC-VO's keypoint selector** picks keypoints from flow as usual.
7. At each keypoint, **sample `c`** → per-keypoint static confidence `w_i`.
8. **Stage 1 filter**: drop keypoints with `w_i < min_c`.
9. **Build the factor graph** with surviving keypoints (visual factors) plus the
   **IMU edge** stored from step 3.
10. **Factor graph `forward()`** returns `[w_i · r_vis_i ; r_imu_15]`; pypose LM runs
    as normal and returns the optimized pose, velocity, and biases.
11. **Backend writes back**: updated `FrameNode` pose (converted body→camera via
    `T_BS`), updated `vel`, updated `bias_g`, `bias_a`.
12. **Preintegrator bias reference updated** so the *next* frame's preintegration is
    linearized around the current bias estimate.

The hidden state `h_t` is instance state on the frontend module, carried across calls
to `estimate_pair`. It resets on explicit sequence boundaries or long IMU-dt gaps.

### 14.1 What happens when things go wrong

- **IMU dropout**: runtime config flip to `imu_fusion: none` (visual-only fallback),
  or graceful degradation — the head without IMU is weaker but functional. A
  well-designed deployment monitors IMU health and switches modes automatically.
- **Hidden state corruption**: if the head produces NaN or all-zero outputs, reset
  the hidden state. Shouldn't happen after GradientClip and logit clamping during
  training, but defensive code helps.
- **Extreme motion**: at very high angular rates, preintegration becomes unreliable.
  `dt_reset_ms` triggers a hidden-state reset, preventing corrupted state from
  propagating.
- **Cold start**: first few frames run with zero hidden state and are slightly
  under-confident. Stage 2's soft-gating handles gracefully: ambiguous factors
  contribute reduced-but-nonzero weight, so the PGO still gets usable observations.

### 14.2 Compute budget

At inference, added work over vanilla MAC-VO:

- One context-encoder output stash: free (retains an existing tensor).
- One AirIMU forward pass per frame pair: ~2 ms on CPU, less on GPU.
- One `DifferentiablePreintegrator` call with `share_with_backend=True`: ~1 ms.
- One `StaticConfidenceHead` forward: ~5–10 ms on GPU at H/8 (temporal) + H/4 (refine).
- Keypoint sampling + Stage 1: sub-ms.
- Per-factor weight multiplication: no measurable overhead.
- Backend IMU factor evaluation: 15-D residual + 15×15 information, ~0.5 ms per LM
  iteration; over ~10 iterations, 5 ms total.

Total added: ~15–20 ms per frame on a typical GPU. MAC-VO's target frame budget is
30–50 ms; this is a 30–50% overhead. Acceptable, probably reducible via kernel fusion
and JIT compilation if the frame budget gets tight.

---

## 15. Why this will probably work — and where it might not

### 15.1 Reasons for optimism

- **The rigid-flow identity is tight.** A geometric fact, not a learned heuristic. The
  only slack comes from sensor noise and occlusion, both addressable with validity
  masks.
- **The features are rich.** FlowFormerCov is a modern, strong flow estimator, and its
  context features encode scene-level appearance. A small head reading those features
  has plenty of signal.
- **The training target is smooth and physically meaningful.** Depth-adaptive
  thresholds and focal BCE turn a naturally-imbalanced problem into a tractable one.
- **The gating mechanism is mathematically clean.** Multiplying whitened residuals by
  `c` is equivalent to covariance scaling and integrates naturally into pypose's
  IRLS loop without hacks.
- **Every architectural choice has an ablation.** If any piece turns out unnecessary,
  the ablation matrix will catch it. This isn't a kitchen-sink design; each component
  earns its keep.
- **The bootstrap is provably correct on stereo.** DRT-loose with `s = 1` has been
  validated by the original authors on monocular; removing the scale SVD can only
  help (no more SVD conditioning issues).
- **The backend IMU factor is industry-standard.** Forster preintegration with bias
  random walk is the same formulation used by VINS-Mono, OpenVINS, and ORB-SLAM3.
  The math is not novel; the novelty is the *co-design* with the static-confidence
  head.

### 15.2 Reasons to be cautious

- **Sim-to-real gap.** Training on VIODE and expecting EuRoC performance is a tall
  order. The head might exploit VIODE rendering artifacts instead of genuine motion
  cues. The A8 ablation is designed to catch this — if A8 fails, the method needs
  rethinking, not just tuning.
- **FOE blind spot.** Pixels directly at the focus of expansion are invisible to
  residual-based supervision. Long straight-ahead driving sequences will be weaker
  there. Acceptable, but worth knowing.
- **Hidden-state drift over long sequences.** ConvGRU trained on 4-frame windows,
  run on thousands-of-frame sequences at inference. If it accumulates state the
  training distribution never saw, output quality might degrade over long runs.
  `dt_reset_ms` and sequence-boundary resets mitigate but don't eliminate.
- **Interaction with constant-velocity motion model.** MAC-VO initializes LM with a
  CV prediction. If CV is badly biased by a dynamic-heavy previous frame, the first
  LM step moves pose the wrong way, then struggles to recover. Gating mitigates but
  this is a feedback loop worth watching.
- **Calibration drift between datasets.** `T` is fit on a held-out
  split of the training distribution. On a new dataset with different noise
  statistics, calibration might be off. Fixable by re-running calibration on a small
  labeled subset of each deployment target, but adds operational overhead.
- **DRT accept-reject false negatives.** The bootstrap can reject legitimately good
  initialization windows if the accept criteria are too strict. Too many rejections
  fall back to `identity` init, which is okay but costs us the Stage-A IMU
  constraint for the first window. Worth logging the accept/reject rate per
  deployment.
- **Preintegration first-order accuracy limit.** The bias-Jacobian correction is
  valid in a neighborhood of `bias_ref`. If the backend pushes the bias far from the
  reference, the factor becomes inaccurate and the optimization may drift. The
  random-walk prior keeps biases close to `bias_ref`, but a rare extreme maneuver
  (hard brake + sharp turn simultaneously) can push them out. The backend
  periodically updates `bias_ref` to re-center the factor.

### 15.3 The scientific story

The paper this method would support:

*"We show that a small (~0.6M parameter) temporally-recurrent head with coarse-to-fine
refinement, trained with GT-pose-based rigid-flow residuals and conditioned on IMU
preintegration factors, learns to predict a single per-pixel static confidence `c`
that meaningfully improves MAC-VO's ATE and RPE on both synthetic and real dynamic-
scene benchmarks. The head's confidence output weights visual reprojection residuals
in a two-stage gating scheme that is mathematically equivalent to covariance scaling
in the LM optimizer. A stereo-adapted DRT-VIO-Init bootstrap provides a metrically-
consistent initial state, and the same preintegration factor used to condition the
head is also added as a hard constraint to the backend — closing the IMU loop on
both sides. Crucially, the head operates on a hard-frozen MAC-VO backbone, so
training is data-efficient and preserves MAC-VO's calibrated covariance. Ablations
show (i) the IMU conditioning is most valuable during fast camera motion, (ii) a
cycle-consistency *validity mask* on the training loss is sufficient to handle
occlusion without a second supervised visibility head (the factorized
`c = p_static · p_visible` variant is retained as an ablation), (iii) the H/4
refinement sharpens object boundaries measurably, and (iv) backend IMU factors
reduce drift on long sequences where visual-only PGO diverges."*

Clean and defensible if it works. If it doesn't, the ablation matrix tells you which
piece failed and where to look next.

---

## 16. Common misconceptions

These points are easy to misread when implementing quickly. This section is blunt on
purpose.

### 16.1 "If GT pose is used in training, inference is impossible without GT."

False. GT pose is used *only* to build supervision targets (`c_target` in
`L_dyn`). The forward path at inference is:

1. frozen FlowFormerCov features,
2. frozen AirIMU + preintegration factor,
3. per-pixel IMU-rigid proxy (§6.5),
4. trainable head,
5. confidence-weighted PGO + IMU factor.

No GT tensor is read at runtime. The proxy input (§6.5) carries the geometric
comparison signal that GT-residual carries in training.

### 16.2 "`c` is the same thing as flow covariance."

False. Covariance models **measurement uncertainty** (how noisy a match is). Static
confidence models **rigidity × visibility** (how likely a pixel belongs to the static
world and how trackable the correspondence is). They correlate but are not
equivalent:

- High covariance + high `c`: textureless but static wall.
- Low covariance + low `c`: cleanly-textured moving object.

The head reads both `flow/cov` and higher-level context features precisely because
the two signals are complementary.

### 16.3 "Hard masking alone is enough."

Usually false in dynamic scenes with boundary ambiguity. Pure hard masking is brittle
to threshold choice and throws away useful but uncertain observations. Two-stage
design (hard + soft) is not redundant.

### 16.4 "Soft weighting alone is enough."

Also usually false. Very low-confidence factors still cost solver time and can create
numerical pathologies if they survive in large numbers. Stage-1 filtering reduces
this burden before optimization.

### 16.5 "This is equivalent to semantic masking."

No. Semantic masks ask "what class is this object?"; this method asks "is this
pixel's motion consistent with rigid ego-motion, and is the correspondence
trackable?" A parked car can be highly static; a moving object with an unseen
category can still be rejected.

### 16.6 "Why not train the whole backbone end-to-end?"

Because the design objective is *controlled intervention*: preserve MAC-VO's
pretrained flow/covariance behavior and learn only an additional reliability field.
Limits compute, limits overfitting, keeps ablations interpretable. The frozen
backbone is not a compute optimization; it's an epistemological choice about what
the method is claiming to learn.

### 16.7 "The recurrence is optional, so it's probably unimportant."

Not necessarily. The recurrence is optional for ablation hygiene, but it is expected
to be load-bearing when dynamic evidence is temporally delayed (e.g., an object
starts moving after a short static period) or briefly ambiguous in a single frame
pair. Disabling it is a valid ablation; assuming it is vestigial is a misread.

### 16.8 "The backend IMU factor is a luxury, not a necessity."

False. Without the backend factor, `c` weighting has to carry all the information
for dynamic-scene robustness — a fragile single point of failure. With the backend
factor, even when the head under-performs, the IMU factor provides a second
constraint that prevents drift. The backend factor is *structural*, not cosmetic.

### 16.9 "DRT-loose is a monocular algorithm; it doesn't apply here."

Partially true, mostly false. DRT-loose has two components that *are* monocular-
specific (LiGT SVD for translation, scale estimation in linear alignment) and four
that are sensor-agnostic (gyro bias estimator, reintegration, rotation chain,
gravity refinement). The spec keeps the four sensor-agnostic steps and replaces the
two monocular steps with stereo-metric equivalents. The end result is a
stereo-specialized DRT that is arguably *more robust* than the monocular version
because it removes the scale SVD's conditioning issues.

### 16.10 "`b_a = 0` at bootstrap is a bug."

It's a deliberate choice, validated by VINS-Mono and consistent with the
observability theory in §8.10. The random-walk prior on `b_a` in the backend lets it
drift from zero as the motion becomes diverse enough to make accel bias observable.

### 16.11 "The preintegrator needs bias Jacobians everywhere."

False. The frontend uses the preintegrator as a conditioning signal; it never
re-linearizes the bias. So `share_with_backend: false` skips the Jacobians. Only the
backend factor evaluation needs them, because the backend is actively optimizing the
bias and must evaluate the factor at updated biases cheaply. Computing Jacobians for
the frontend-only path is wasted work.

### 16.12 "Marginalization is a bookkeeping detail."

Dangerously wrong. Marginalization is the *only* mechanism that prevents the
sliding-window backend from leaking information out as the window slides. Without it,
the backend's drift grows super-linearly in time. Getting marginalization right
(correct Schur complement, correct linearization point, correct numerical pivoting)
is as important as any other part of the backend.

---

## 17. Reading guide

If you want to understand the method as quickly as possible:

1. **Read §2** (rigid-flow identity) — everything else builds on this.
2. **Read §6** (IMU branch and train/test closure) — the single most important design
   decision in the whole method.
3. **Read §9** (preintegration factor math) — the backend's constraint structure and
   why the same preintegrator serves frontend and backend.
4. **Read §11** (PGO math) — the gating mechanism is the simplest part but the one
   most often misunderstood.
5. Skim the rest as needed.

If you want to argue with the method:

- §3.4 (FOE blind spot) and §3.5 (occlusion) are the physics-level failure modes.
- §10.5 (what the loss does *not* include) is where the design most differs from
  standard practice.
- §15.2 (reasons to be cautious) lists the empirical risks.
- §16 (misconceptions) catches the most common implementer mistakes.

If you want to extend it:

- **Test-time adaptation** on deploy sequences using the cycle-consistency signal
  alone. The cycle signal is dataset-agnostic and doesn't need GT pose; on deploy
  it can either tighten the validity mask (re-shrinking `τ_cyc_valid`) or — under
  the factorized `A_fact` variant — drive an explicit unsupervised visibility
  loss `L_visible`.
- **Learned solver-aware confidence map `w = g(c)`**. A monotonic learned map
  lets the head adapt the calibration-to-optimization mapping to LM dynamics.
- **Adaptive IMU covariance** (Stage C in §9.12). Innovation-based inflation closes
  the loop on IMU reliability.
- **Joint training with PGO in the loop**. Freeze FlowFormerCov but differentiate
  through the LM iterations to get a pose-level loss on top of the pixel-level loss.
  Natural next step, explicitly out of scope for this spec.
- **Richer IMU fusion**. AirIMU gives a compact feature; cross-attention between
  IMU tokens and visual tokens would let the network use IMU in a spatially-varying
  way. Depth-bin FiLM is the current middle-ground; full cross-attention is the
  next step.
- **Full sliding-window marginalization**. Stage B is correct but uses a simple
  Schur-complement marginalization. More sophisticated approaches (re-linearization
  policies, partial marginalization with factor recovery) would be worth exploring
  for long-horizon deployments.

---

## 18. Cross-references to the spec

| Theory section | Spec section(s) |
|---|---|
| §2 Rigid flow identity | §0.2 (coordinate conventions), §3.1 (end-to-end data flow) |
| §3 Residual → supervision (incl. cycle as validity mask) | §5.1 (label construction), §5.2 (loss terms) |
| §5 Single-head default; factorization as ablation | §1 Goal (single-output head), §3.2 (HeadOut), §10 (ablation `A_fact`) |
| §6 IMU branch & train/test closure | §3.4 (IMU proxy), §4.3 (encoder), §5.4 (head forward) |
| §7 H/8 recurrence + H/4 refinement | §3.3 (architecture), §3.5 (refinement branch) |
| §8 DRT-loose bootstrap | §7 (complete initialization pipeline) |
| §9 Preintegration factor math | §9 (backend factor graph), §4.3 (shared preintegrator) |
| §10 Loss composition (`L_dyn + L_smooth`) | §5.2 (two loss terms), §5.3 (total loss) |
| §11 PGO integration math | §4.8 (Reproj_TwoFramePGO weighting), §9.2 (visual residual weighting) |
| §12 Calibration (single `T`) | §5.7 (post-hoc temperature fit), §6 (config buffers) |
| §13 Training data | §8 (datasets and splits) |
| §14 Deployment | §12 (runtime path) |

If the spec and this document disagree on any numerical detail, the spec wins —
that's the implementation contract. If they disagree on *reasoning*, this document
wins — the spec sometimes omits "why" for brevity.
