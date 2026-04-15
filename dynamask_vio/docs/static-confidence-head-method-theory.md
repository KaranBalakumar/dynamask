# Method, Theory, and Intuition — Static Confidence Head for MAC-VO

**Companion document to:** `2026-04-11-static-confidence-head-macvo-spec.md`

This is a long-form, from-first-principles walkthrough of the method. The spec tells
you *what* to build; this document tells you *why* every piece is shaped the way it is.
Read this when you want to understand the idea, argue about it, or extend it. Read the
spec when you want to write the code.

---

## 1. The problem, precisely

Visual-inertial odometry (VIO) is a nonlinear least-squares problem: we want the camera
pose that best explains a set of pixel observations under a rigid-scene assumption.
"Rigid scene" means the world does not move between frames — only the camera does. All
classical VO solvers (MAC-VO, DPVO, TartanVO, ORB-SLAM, VINS-Fusion…) bake this
assumption into the cost function.

When a car drives past the camera, or a person walks in front of it, this assumption
is locally false. The pixels on that object *do* move independently of the camera, and
any solver that treats them as rigid-scene observations will corrupt its pose estimate
in proportion to how many such observations there are and how strong their disagreement
with the rigid-scene prediction is.

The standard mitigations fall into three categories:

1. **Robust loss functions** — Huber, Cauchy, truncated L2. These downweight large
   residuals regardless of *cause*. A large residual caused by dynamic motion and a
   large residual caused by matching noise get the same treatment. This is cheap but
   indiscriminate.
2. **Semantic masking** — run a segmentation network, drop observations on "car",
   "person", "bicycle". This is discriminative but brittle: it relies on a model
   trained on a specific ontology, fails on categories not in the training set, and
   fires on *parked* cars just as enthusiastically as on moving ones.
3. **Motion-aware masking** — predict, per pixel, whether the pixel's motion is
   consistent with a single rigid-scene hypothesis. This is the right granularity:
   it ignores parked cars (they're static), it ignores static humans (they're static),
   and it doesn't care about object category. It also requires the network to
   understand geometry, not semantics.

This method is category (3). The spec builds a small network that outputs, for every
pixel of every frame, a number `c ∈ [0,1]` interpreted as "how confident am I that
this pixel's motion is rigid-scene motion". `c = 1` means "fully trust this pixel as a
static observation"; `c = 0` means "do not use this pixel as a static observation".

The rest of this document derives how such a network can be trained and deployed, step
by step, from first principles.

---

## 2. Rigid flow: the identity the method rests on

### 2.1 The pinhole projection equation

For a single pinhole camera with intrinsics matrix `K`, a 3D point `X` in the camera's
coordinate frame projects to the pixel `x`:

```
x = π(X) = K · [X_x / X_z, X_y / X_z, 1]ᵀ
```

Given a depth map `D`, we can back-project a pixel `x = (u, v)` to its 3D point in the
camera frame:

```
X(x) = D(x) · K⁻¹ · [u, v, 1]ᵀ
```

### 2.2 What flow should look like for a rigid scene

Now consider two frames taken at times `t` and `t+1`, with a relative rigid-body
transformation `(R, t)` carrying camera-frame-t to camera-frame-(t+1). For a **rigid**
scene point visible in both frames:

```
X_{t+1} = R · X_t + t
x_{t+1} = π(X_{t+1})
f_rigid(x_t) = x_{t+1} - x_t
```

`f_rigid` is the **rigid flow** — the 2D optical flow that would be observed if every
pixel corresponded to a point that was static in the world and only appeared to move
because the camera moved. It is a deterministic function of the camera's motion, the
scene depth, and the camera intrinsics. Nothing else.

### 2.3 The residual identity

Suppose we also have a *measured* optical flow `f_obs(x_t)`, produced by a
correspondence network like FlowFormerCov. The **flow residual**:

```
r(x_t) = || f_obs(x_t) - f_rigid(x_t) ||₂
```

has a clean interpretation:

| Pixel's physical status | `r` should be… |
|---|---|
| Truly static, perfect depth, perfect flow | 0 |
| Truly static, noisy depth / noisy flow | small (proportional to sensor noise) |
| On a moving object | large (proportional to the object's own 3D velocity, projected) |
| In an occlusion / disocclusion region | large and meaningless (there is no true correspondence to measure) |

Notice what is and isn't in this identity:

- **No assumption about object category** — the test is purely kinematic.
- **No semantic labels are needed** — the input is just depth, poses, and flow.
- **It needs the pose**, so it only works if we have `(R, t)` from somewhere. At
  training time we have ground truth. At inference time we don't — and this asymmetry
  is the central design constraint of the method, addressed in §5.

---

## 3. The residual as a supervision signal

The residual isn't a label — it's a continuous quantity that needs to be turned into
one. The straightforward thing is:

```
d*(x) = σ((r(x) - τ) / κ)
c*(x) = 1 - d*(x)
```

where `σ` is the sigmoid function, `τ` is a threshold, and `κ` controls softness.
Small residuals map to `c* ≈ 1` (static), large residuals map to `c* ≈ 0` (dynamic),
and values in between are soft.

### 3.1 Why you can't just use a hard threshold

The tempting alternative is a hard binarization: `c* = 1 if r < τ else 0`. This breaks
for four separate reasons:

1. **Decision boundary instability during training** — a pixel on the edge of the
   threshold flips labels with tiny changes in depth, flow, or pose. The head sees
   its own label flicker and can't converge.

2. **BCE gradient vanishes on confident labels** — with binary labels of `{0, 1}` and
   a sigmoid head, the gradient from a confident correct prediction drops to zero,
   meaning most of the training signal comes from the (small) set of pixels near the
   boundary. Soft labels spread the gradient over a larger fraction of the image.

3. **The residual is a noisy measurement of a latent variable.** What we actually care
   about is whether the pixel's *ground-truth dynamic velocity* is zero. The residual
   is a sample from a noise distribution centered at a function of that velocity.
   Treating a single noisy sample as a hard label discards the information in the
   noise distribution.

4. **Useful ordinal structure is lost.** A residual of 20 px is "more dynamic" than a
   residual of 5 px, and the solver should be told as much. Hard binarization collapses
   that ordering.

Soft labels preserve all four.

### 3.2 Why a simple fixed `τ` fails too

A constant threshold over the whole image is almost always wrong, for a reason that is
specific to VO and not present in most dynamic-mask papers: **rigid flow magnitude
depends on depth**.

Consider a camera translating forward at 1 m/s, frame rate 30 Hz, focal length `f`:
- A static point at 2 m depth produces rigid flow of roughly `f · (1 / 2) / 30 = f / 60`
  pixels per frame. At `f = 500` that's ~8 px.
- A static point at 100 m depth produces rigid flow of roughly `f / 3000` ≈ 0.17 px.

Both are static. The observed flow at each will be the rigid flow plus some constant
noise (say, 0.3 px from FlowFormerCov). So the residual at the *close* point is
dominated by flow noise (ratio 0.3 / 8 ≈ 4%), while the residual at the *far* point is
larger than the expected rigid flow itself (ratio 0.3 / 0.17 > 1). A single threshold
that works for the close point will flag half of the distant scene as "dynamic"; a
threshold that works for the distant point will miss all dynamic objects in the near
field.

### 3.3 The depth-adaptive threshold

The fix is to scale `τ` with the expected rigid-flow magnitude, which is itself
predictable from depth, translation magnitude, and focal length:

```
|f_rigid| ≈ (f / D) · ||t||              (order-of-magnitude approximation)
τ(D)     = τ₀ + α · (f / D) · ||t||
```

`τ₀` is a floor (captures pure flow noise) and the additive term grows as rigid flow
gets larger. Equivalently, the threshold is roughly "how much 2D flow corresponds to
a 3D velocity above some threshold" — it's a 3D-velocity cutoff expressed in pixel
units.

The soft label becomes:

```
d*(x) = σ((r(x) - τ(D(x))) / κ)
c*(x) = 1 - d*(x)
```

This is why the spec's config has both `tau0` and `alpha`. A single constant threshold
would be a bug, not just a suboptimal choice.

### 3.4 The focus-of-expansion blind spot

There's one regime this residual-based approach cannot fix on its own: the **focus of
expansion** (FOE), the direction in which the camera is translating. For pixels
directly at the FOE, the rigid flow vanishes for *all* depths — so a dynamic object
there produces a residual equal only to its own projected motion, which can be small
if the object is also moving in that direction.

The method can't solve this purely from residuals. What it *can* do is exploit the
network's access to richer features: the 128-channel FlowFormer context feature map
encodes what the scene looks like, not just how fast things moved. A car directly ahead
of the camera still looks like a car, and the context features carry that information
into the head's decision. This is part of why the architecture reads `f_ctx` in
addition to `flow` and `cov`, and not just the residual or the flow alone.

### 3.5 Occlusion is the other major failure mode

Even for truly static pixels, there is a regime where the residual is large and
meaningless: **occlusion and disocclusion**. Pixels that are visible in frame `t` but
hidden in frame `t+1` (or vice versa) have no true correspondence. FlowFormerCov will
produce *some* flow vector for them, but that vector is defined by whatever texture
happens to be there in the target frame — it has no physical meaning. The rigid flow
derived from GT pose, however, is the geometrically correct answer, computed from depth
even if the pixel disappears in the target image.

So at occluded pixels, `f_obs` is "visual nonsense" and `f_rigid` is "geometrically
correct but unrealized". Their difference is large not because the pixel is dynamic,
but because there's nothing to compare to.

Feeding this residual into the BCE as a dynamic label would poison the training. The
fix is **cycle-consistency masking**: compute forward flow `f_fwd` and backward flow
`f_bwd`, warp `f_bwd` by `f_fwd`, and check that the composition returns close to the
origin. Pixels where this fails are occluded (or at least untrackable) and must be
masked out of the supervision signal entirely. The spec's `M_cycle` mask is exactly
this.

Note the subtle but important distinction: cycle-consistency is used in *two* places
in the method, for *two* different purposes:

1. As a **validity mask** in `L_dyn`: mask out occluded pixels so they don't contribute
   noise to the primary GT-residual loss. Here the cycle check is binary, threshold
   relatively loose.
2. As a **pseudo-label source** in `L_cyc` (Term 3): soft-label cycle inconsistency
   itself as "probably dynamic". Here the check is sigmoidal. This is the D²USt3R /
   flow-cycle-consistency trick, repurposed as a weak auxiliary label that doesn't
   need GT pose and therefore generalizes to any dataset.

The same computation serves both roles because cycle-inconsistency is the union of
"occluded" and "dynamic discontinuity" — and both of those should reduce the
network's confidence that the pixel is a clean static correspondence.

---

## 4. What inputs the head actually needs

Before the recurrence or the IMU branch, the question is: what should the head read?
There are three plausible inputs and each answers a different question:

1. **The residual map `r` itself.** Feeding `r` as input produces a near-trivial
   network: a learned thresholder. It tells you nothing the residual doesn't already
   tell you, and it *requires* access to GT pose (or ego pose) to compute at inference.
   Not used.

2. **The flow and its covariance.** This tells the network "how fast is each pixel
   moving, and how confident is the flow estimator". A fast-moving pixel with low flow
   covariance is suspicious regardless of what the rest of the scene looks like.

3. **The context feature map `f_ctx`.** This is FlowFormer's 128-channel context
   encoder output. It encodes what the scene *looks like* — textures, shapes, regions
   of coherent appearance. This gives the network access to "what kind of thing is
   this pixel part of", independent of the pixel's local motion.

The head reads (2) and (3). It deliberately does **not** read (1) or `r` or
`f_rigid_gt` as inputs. The residual enters only as a training label, never as an
input. This is the central train/test symmetry constraint: if the head depended on
`r` as input it would need GT pose at inference, which we don't have.

### 4.1 Why frozen features are rich enough

FlowFormerCov was trained on large-scale flow data with covariance supervision. Its
context features have already learned to represent things that matter for motion
reasoning: object extents, texture coherence, depth discontinuities. The static
confidence head is effectively a *small decoder* learning to map those features to a
different output space (static-or-not instead of flow). A ~0.6M-parameter head is
enough for that decoding, and the small parameter count is itself a feature: it trains
fast, needs less data, and doesn't catastrophically forget.

### 4.2 Why frozen — the deeper reason

The user's choice to hard-freeze MAC-VO isn't just a convenience. Training through
FlowFormerCov would mean:

- ~15M additional trainable parameters.
- Needing large, diverse flow datasets to avoid corrupting the pretrained features.
- Risking catastrophic forgetting of the covariance calibration that FlowFormerCov
  already has.
- Paying a 10-50× compute overhead per training step.

Freezing makes the training a *light supervised probe* — we're asking the network
"can you read static-or-not off of these features", not "relearn the whole flow
problem". That's a much easier problem, takes dramatically less data, and doesn't risk
the foundation model's calibration.

This pattern — freeze a strong backbone, train a tiny head — is the same pattern used
in CLIP-probing, SAM prompt tuning, and every modern foundation-model-adapter method.
It works for the same reason here.

---

## 5. The IMU branch and the train/test asymmetry

The elephant in the room: at training time we have GT pose, at inference time we
don't. How can we train a head to do at inference what it only saw at training with
extra information it won't have later?

The answer is: **we don't ever feed GT pose into the head**. GT pose enters the loss
only, as a label-computation signal. The head's *input* at training is identical to
its input at inference. The only difference between train and test is what label the
loss compares against.

But that creates a subtler question: *what information substitutes for GT pose at
inference*? The head needs to know, roughly, how the camera is moving — otherwise it
can't distinguish "flow due to ego-motion" from "flow due to object motion". Without
some motion prior, the head is reduced to "does this look like a dynamic object"
(semantic), which is not what we want.

The answer is **the full IMU preintegration factor**, not merely AirIMU's encoder.
Specifically, the branch is a three-stage pipeline, and all three stages matter:

```
raw IMU samples ──► AirIMU learned corrector ──► (â, ω̂, σ²_a, σ²_g, Δb_a, Δb_g)
                                              │
                                              ▼
                        Differentiable Forster-style Preintegrator
                              (SO(3)×R⁶, 1st-order covariance propagation)
                                              │
                                              ▼
                    (ΔR̂, Δv̂, Δp̂, Σ_imu ∈ R^{9×9}, dt, Δb_a, Δb_g)
                                              │
                                              ▼
                         small feature MLP ─► f_imu ∈ R^{128}
```

What the head actually consumes ("IMU context") is therefore **the same quantity that
a VIO backend would place into a preintegration factor**: a predicted relative
rotation/velocity/translation on SE(3)×R³, its propagated uncertainty, and the bias
health estimates that produced them. This is deliberate — it's what lets the same
IMU signal serve two roles consistently (frontend conditioning + backend factor, §15.8).

### 5.1 Why the preintegration factor, not AirIMU's encoder alone

Feeding "AirIMU's raw encoder state" into the head would be subtly wrong. The head's
job is to disambiguate *flow-due-to-ego-motion* from *flow-due-to-scene-motion*. The
operator that maps ego-motion to expected image-space flow is:

```
f_rigid_imu(x) = π( R_imu · K⁻¹ · D(x) · [u, v, 1]ᵀ + t_imu ) − x
```

This operator consumes a **relative pose** `(R_imu, t_imu) = (ΔR̂, Δp̂)` — not a black-box
context vector. Raw encoder features cannot be projected through this operator; a
preintegrated SE(3) increment can. Using the preintegrator is what makes the IMU
branch *geometrically compatible* with the rest of the method rather than a
hand-wave that "somehow tells the head about motion".

Equally important: preintegration produces a *calibrated uncertainty* Σ_imu that the
raw encoder cannot. That matters because the head's correct response to a
poorly-preintegrated window ("short dt but large jerk; biases drifting") should be to
*distrust the motion prior and fall back on visual cues*. Without Σ_imu as input,
the head has no way to learn that fallback behavior — it has to treat every IMU
output as equally trustworthy. With Σ_imu, it can.

### 5.2 What IMU context the head actually receives

Explicit list of quantities fed from the IMU branch into the static-confidence head
(per flow pair):

| Quantity | Dim | What it means to the head |
|---|---|---|
| `log(ΔR̂)` | 3 | relative rotation as a tangent vector (well-behaved at identity) |
| `Δv̂` | 3 | body-frame velocity change across the pair |
| `Δp̂` | 3 | body-frame translation across the pair |
| `diag(Σ_imu)` or `chol(Σ_imu)` | 9 or 45 | self-reported trust in (ΔR̂, Δv̂, Δp̂) |
| `Δb_g`, `Δb_a` | 3 + 3 | AirIMU-predicted bias corrections (large ⇒ IMU is noisy) |
| `dt` | 1 | window length, so every other quantity is scale-normalizable |

These are flattened into a ~20–50-D vector and projected by a tiny MLP (`FeatureMLP`)
into a 128-D conditioning vector `f_imu`. The MLP is trainable and part of the head;
everything upstream of it (corrector, preintegrator) is frozen or non-parametric.

In addition, the preintegrator's *pose output* `(ΔR̂, Δp̂)` is re-used at the pixel
level — see §5.5.

### 5.4 Why IMU and not a learned visual pose estimate

Another option would be to have a second network predict pose from the stereo pair,
and feed *that* pose's rigid flow as input to the head. This has been tried in the
literature (TartanVO-style architectures). It fails for a specific reason: the
visually-estimated pose is itself corrupted by dynamic objects — the very thing we're
trying to detect — so the head ends up reading a pose that has already been biased
by the phenomenon it's supposed to flag. IMU doesn't have this feedback loop: accel
and gyro measurements are indifferent to visual dynamics.

### 5.5 The IMU-rigid residual proxy: per-pixel closed-loop cue

The conditioning vector `f_imu` gives the head a *global* motion prior. But the
preintegrated pose `(ΔR̂, Δp̂)` also lets us build a *per-pixel* signal that directly
mirrors the GT-based supervision target — turning a portion of the training label
into something the head sees at inference time too.

For every pixel `x` with depth `D(x)` from the frozen stereo backbone:

```
f_rigid_imu(x) = π( ΔR̂ · K⁻¹ · D(x) · [u, v, 1]ᵀ + Δp̂ ) − x
Δf(x)          = f_obs(x) − f_rigid_imu(x)
r_imu(x)       = ||Δf(x)||₂
r_imu_norm(x)  = r_imu(x) / τ(D(x))    # depth-adaptive normalization
```

The head reads three extra input channels: `Δf_x, Δf_y, r_imu_norm`, plus a
validity mask (`inside image ∧ valid depth`). These are the *IMU-analog of the GT
residual* used as a label in `L_dyn`. At training time, the GT residual supervises
the output; at inference time, the IMU residual feeds the input. Both are computed
by the same geometric operator, so the head sees features it already knows how to
interpret.

**Why this is load-bearing and not a refinement.** The central train/test asymmetry
argument (§5 opener) requires that the head's *input distribution* be the same at
train and test. If the head is trained with only `f_imu` and frozen features, it has
no explicit geometric channel to key off — the only signal carrying "this is where
the rigid prediction disagrees with observation" lives in the loss. The proxy input
moves that signal into the forward pass, closing the train/test loop.

**Why IMU (and not visual pose) for the proxy.** Same argument as §5.4: a visually-
estimated pose is corrupted by the dynamic objects it's supposed to help flag. IMU
is indifferent to visual dynamics, so `f_rigid_imu` is an unbiased (if noisy)
estimate of rigid flow even in dynamic-heavy scenes.

**Why Σ_imu still matters even with the proxy.** The proxy's reliability is a
function of Σ_imu. When Σ_imu is large (short dt with jerky motion, bias drift, time
sync issues), `f_rigid_imu` is untrustworthy and the head should ignore the proxy
channels. Feeding Σ_imu through `f_imu` gives the head the global gating signal it
needs to decide how much weight to put on the proxy pixel-wise.

### 5.6 Why the IMU branch is "optional but load-bearing"

The config exposes `imu_fusion: none` as a switch. That ablation is genuinely useful
because it tells you how much of the head's performance comes from visual features
alone vs how much came from the IMU prior. Empirically, the expectation is:

- **On sequences with smooth, mostly translational camera motion**: IMU adds a modest
  bump. Visual features alone do most of the work.
- **On fast rotations, sharp jerks, motion blur**: IMU becomes dominant. Visual
  features are ambiguous in those regimes, and IMU is the only reliable pose signal.

So the ablation isn't pro-forma: it's expected to show a regime-dependent gap, and
that gap is part of the scientific story.

### 5.7 Why FiLM instead of concatenation

FiLM (Feature-wise Linear Modulation) takes a conditioning vector `f_imu` and produces
per-channel scale and shift parameters `(γ, β)` that modulate an activation tensor:
`x' = γ · x + β`. Compared to concatenating `f_imu` as extra input channels, FiLM has
two advantages:

1. **It scales features rather than adding them.** Concatenation is an additive
   injection; FiLM is multiplicative. A multiplicative signal can change the *routing*
   of information through the network (which features are suppressed vs amplified)
   without having to wait for deep layers to learn that routing.
2. **It's spatially uniform, which matches the semantics.** `f_imu` represents the
   whole camera's motion over one frame — it's a scene-level quantity, not a pixel-
   level one. FiLM applies it uniformly across space, which is the correct inductive
   bias. Concatenation would force the network to learn that uniformity.

Both options are left in the config because FiLM can occasionally underperform when
the conditioning signal is high-dimensional and information-rich; for AirIMU's
compact 128-dim summary, FiLM is the right choice.

---

## 6. Why temporal recurrence, and why ConvGRU specifically

The head could be a feed-forward ConvNet run independently on each frame. Why add a
ConvGRU?

### 6.1 Dynamic objects are temporally coherent

A car is a car for more than one frame. A person walking is still walking next frame.
The head's belief about a pixel should, in expectation, be highly correlated with its
belief about the same pixel (warped appropriately) in the previous frame. A feed-
forward head has to re-derive that belief from scratch each frame, burning capacity
to recompute the same answer.

A recurrent hidden state encodes "what did I believe about this region last time",
and the new frame's computation only has to update that belief based on new evidence.
This is information-theoretic compression: it's a strictly smaller problem.

### 6.2 Dynamicness is sometimes only visible over time

Consider a person standing still next to a car. The person has zero motion, the car
has zero motion, both look static to a single-frame analysis. One frame later, the
car starts moving. A feed-forward head flags the car at exactly frame `t+1` (when it
starts moving) and not a moment before. A recurrent head that has been tracking the
car's region over several frames can, in principle, *anticipate* the transition based
on earlier subtle motion cues (engine vibration? wheel turning?). It can also be
slower to flag the person as dynamic when they finally take a step, because the prior
belief was "this region was static".

The gain isn't dramatic, but it's real and it's in the right direction.

### 6.3 Temporal smoothing without a hard constraint

A non-recurrent head that produces jittery frame-to-frame masks would break the PGO
gating downstream — PGO's IRLS iterations expect stable observation weights. You could
add an explicit temporal-consistency loss term, but that's an extra hyperparameter with
its own failure modes. A ConvGRU hidden state smooths temporally *for free*, as a
byproduct of the architecture rather than the loss. This is why the spec deliberately
omits an explicit `L_temporal` term: the recurrence already does the job.

### 6.4 Why ConvGRU, not ConvLSTM

ConvGRU has one hidden state; ConvLSTM has hidden state and cell state. ConvLSTM's
extra cell state is useful when the recurrence needs to maintain a distinct
*long-horizon memory* separable from its *short-horizon output*. For dynamic-object
tracking over a ~4-frame training window, there's no meaningful long-horizon memory
to separate. The ConvGRU is cheaper, has fewer hyperparameters, and empirically
matches ConvLSTM on shorter horizons. The config exposes ConvLSTM as a fallback in
case the hidden-state dynamics turn out to drift over longer sequences.

### 6.5 Windowed BPTT and why `window_len = 4`

Backpropagation Through Time (BPTT) over a window of `W` frames means gradients flow
through `W` recurrence steps before the chain terminates. Longer windows let the head
learn longer-horizon dependencies but cost more memory linearly and more compute
linearly. Shorter windows underfit the temporal aspect.

`W = 4` is a pragmatic choice:
- The relevant temporal structure (object persistence across frames) is captured in
  2-4 frames of camera motion.
- Memory stays well within GPU limits even at full resolution.
- Dynamic-object behavior beyond 4 frames is dominated by scene-level drift that the
  head probably shouldn't try to model — that's ODE integration territory.

The hidden state is explicitly reset at the start of each training window. This has a
small cost (the first frame of each window runs with zero hidden state and therefore
underperforms slightly) and a large benefit: it prevents the training loop from
accidentally learning to rely on infinitely long hidden state that won't be available
at inference.

At inference, there's no such reset — the hidden state runs indefinitely across the
sequence, being reset only at explicit sequence boundaries or when the IMU detects a
long dt gap. This train-test mismatch (windowed training, streaming inference) is
standard for recurrent models and generally doesn't cause problems.

---

## 7. The loss: why three terms and why these three

### 7.1 Why more than one term

A single-term loss (GT-residual BCE) is theoretically sufficient. The reason to add
more is calibration and regularization: the primary signal is informative but noisy,
and secondary signals can reduce the variance of the training process without changing
what the minimum is.

Each additional term in the loss must justify its existence. The spec includes exactly
two more (smoothness and cycle-consistency) and deliberately excludes several plausible
alternatives (proxy segmentation, entropy, temporal L1, Laplace NLL). The justifications:

### 7.2 `L_dyn` — GT-residual focal BCE (primary)

This is the load-bearing signal. It encodes the rigid-flow identity from §2 directly
as a supervised objective.

**Why focal BCE instead of plain BCE.** Dynamic pixels are a small minority in real
scenes — typically 1-10% of pixels in a driving scene, often 0% in an indoor scene.
Plain BCE averages loss over all pixels, which means the tiny dynamic minority gets
overwhelmed by the vast static majority and the network can get 95%+ accuracy by
predicting "static everywhere". Focal BCE multiplies each pixel's BCE contribution by
`(1 - p_correct)^γ`, which downweights easy (confident and correct) pixels and lets
the hard (dynamic minority) pixels dominate the gradient.

With `γ = 2`, a pixel that's correctly classified with probability 0.9 contributes
`0.01 ×` as much as a pixel correctly classified with probability 0. This is enough
to rebalance the signal toward the minority class without completely discarding the
majority.

### 7.3 `L_smooth` — edge-aware smoothness

Smoothness regularizers assume that the output is locally coherent in image space.
Naive smoothness (`||∇c||`) would discourage sharp transitions between dynamic objects
and their static backgrounds — exactly where we want sharp transitions. Edge-aware
smoothness weights the penalty by `exp(-||∇I||)`, which is near 0 on image edges and
near 1 in smooth regions. The effect is: "be smooth where the image is smooth, be
sharp where the image has edges".

This is the same trick used in depth estimation, normal estimation, and semantic
segmentation. It works because image edges tend to align with the boundaries of
actually-separable objects, and the head's output boundaries should follow suit.

The `L_smooth` term is not essential — training without it converges — but it reduces
flicker in the output mask and makes the learned boundaries crisper. Its weight is
small (0.05 by default) because at larger weights it starts oversmoothing into
meaningful transitions.

### 7.4 `L_cyc` — forward-backward cycle-consistency pseudo-label

This is the auxiliary term most people would forget to include, and it's also the one
with the longest-term strategic value.

The core idea: if you compute forward flow `f_fwd(x_t)` from frame `t` to `t+1`, and
then backward flow `f_bwd` from `t+1` to `t`, warping `f_bwd` by `f_fwd` should land
you back at `x_t` up to noise. Formally:

```
warp_err(x_t) = || f_fwd(x_t) + f_bwd(x_t + f_fwd(x_t)) ||₂
```

Where does `warp_err` get large?

1. **Occluded pixels**: no consistent correspondence exists, so either `f_fwd` or
   `f_bwd` is guessing.
2. **Dynamic discontinuities**: along the edge of a moving object, the flow field
   has a discontinuity, and forward/backward flows disagree across that discontinuity.
3. **Interior of moving objects**: slightly less than you'd expect — an object moving
   coherently still has internally-consistent flow, so the interior pixels often
   pass the cycle check. But edges light up reliably.

What this means for us: `warp_err` is a **noisy, biased, but genuinely dataset-
agnostic** label for the dynamic-boundary problem. It requires zero metadata beyond
flow itself. No GT pose, no depth, no segmentation. You can run it on any stereo-pair
dataset in the world.

**Why this is strategically important**: the head as trained with `L_dyn` alone needs
GT pose to train. That means it can only be trained on sequences with GT pose. `L_cyc`
has no such requirement, so in principle you could train on sequences where you only
have stereo — a much larger corpus. More importantly, `L_cyc` is a *self-supervised*
signal that could be used for **test-time adaptation**: at deploy time, collect a few
frames, run `L_cyc`, do a tiny amount of fine-tuning on the fly. The current spec
doesn't implement TTA, but including `L_cyc` from day one keeps the door open.

**Why it has to be combined with `L_dyn`, not used alone**: `L_cyc` is weaker because
it only fires at object boundaries and occlusions, not at the interior of moving
objects. A head trained on `L_cyc` alone would produce boundary masks, not full-object
masks. `L_dyn` provides the "fill in the interior" signal.

### 7.5 What the loss does **not** include

The spec deliberately omits several plausible terms. Each omission is load-bearing:

- **Proxy segmentation BCE**: would require per-pixel dynamic/static class labels, which
  only synthetic datasets provide. Using it would silently make the method unable to
  train or adapt on real data. Even as a "nice-to-have when available" term, it has the
  failure mode that it trains the head to match a specific synthetic-dataset ontology
  rather than to reason about motion — exactly the failure mode we're trying to avoid
  by not using category-based masking in the first place.

- **Entropy regularization `-H(c)`**: encourages the head to produce confident outputs
  (`c` near 0 or 1 rather than 0.5). This is the sort of thing that sounds
  sophisticated but usually harms calibration. A well-calibrated probabilistic output
  has some pixels near 0.5 (the ambiguous ones) and that ambiguity is itself useful
  information for the downstream PGO gating. Encouraging confidence for confidence's
  sake degrades that.

- **Temporal consistency L1 `||c_t - warp(c_{t-1}, f_obs)||₁`**: same role as the
  ConvGRU hidden state (§6.3). Adding both is redundant and introduces a weight that
  fights the BCE term.

- **Laplace NLL over the residual**: the CoProU-VO paper uses this formulation because
  it's predicting an uncertainty, not a probability. The right loss for an uncertainty
  estimate is NLL. The right loss for a probability is BCE. Our head outputs a
  probability. Using NLL here would just be a worse BCE.

- **Photometric reprojection loss**: FlowFormerCov is already trained with flow
  supervision; its flow output is much cleaner than a photometric residual. Adding a
  photometric loss re-litigates a problem that's already been solved upstream.

---

## 8. PGO integration: why the math works

### 8.1 The factor graph setup

MAC-VO's two-frame PGO solves, for a batch of 2D-3D observations, the pose `T` that
minimizes:

```
L(T) = Σ_i  r_i(T)ᵀ · Σ_i⁻¹ · r_i(T)
```

where `r_i(T)` is the reprojection residual of the i-th observation under pose `T` and
`Σ_i` is its 2×2 pixel-space covariance. In the code (see
`Module/Optimization/TwoFramePGO/Graphs.py`), `r_i` is returned by the factor graph's
`forward()` method and `Σ_i` is returned by `covariance_array()`. pypose's Levenberg-
Marquardt solver handles the rest.

### 8.2 What weighting a residual actually does

Suppose we multiply each factor's residual by a per-observation weight `w_i ∈ [0, 1]`:

```
r_i'(T) = w_i · r_i(T)
```

The loss becomes:

```
L'(T) = Σ_i  (w_i · r_i)ᵀ · Σ_i⁻¹ · (w_i · r_i)
      = Σ_i  w_i² · r_iᵀ · Σ_i⁻¹ · r_i
```

Equivalently, we could leave the residual alone and replace `Σ_i` with `Σ_i / w_i²`:

```
L''(T) = Σ_i  r_iᵀ · (Σ_i / w_i²)⁻¹ · r_i
       = Σ_i  w_i² · r_iᵀ · Σ_i⁻¹ · r_i
       = L'(T)
```

So **"multiply the residual by `w`"** is mathematically identical to **"inflate the
covariance by `1/w²`"**. They're the same operation viewed from two different sides of
the whitening transform. The user's stated preference ("multiply dynamicness
probability directly to the whitened residuals") corresponds to the first framing; the
spec uses exactly that implementation path.

### 8.3 Why the `w²` scaling is the right behavior

A factor with `w = 1.0` contributes its full log-likelihood. A factor with `w = 0.5`
contributes 0.25× its log-likelihood — roughly equivalent to saying its covariance is
4× larger in every direction, i.e. "I trust this observation half as much as a
confident one, so its variance is 4× wider". A factor with `w = 0.1` contributes 0.01×
— essentially dead. This quadratic fall-off is exactly what you want: near-static
pixels should count fully, suspicious ones should fade quickly, obviously-dynamic ones
should vanish.

A *linear* scaling (`w_i` instead of `w_i²`) would fall off too slowly — a pixel with
50% dynamic confidence would still contribute half its weight, which is too generous
given how damaging dynamic observations are to pose estimation.

### 8.4 Why the factor cannot be truly zeroed inside the graph

If `w_i = 0`, then the scaled covariance is `Σ / 0 = ∞`, which pypose can't represent.
There are three ways out:

1. **Drop the factor before it enters the graph.** Require `w_i > 0` on entry; filter
   at the keypoint-selection stage. This is clean and explicit but requires an upstream
   hard threshold.
2. **Use a soft floor `w_i = max(w_i, ε)`.** Pypose sees a finite (huge) covariance
   instead of infinity. The factor is near-dead but technically present.
3. **Use Levenberg-Marquardt's trust-region mechanism to absorb infinite covariances
   by mathematical tricks.** Not actually supported by pypose's LM.

The spec uses *both* (1) and (2):

- **Stage 1** (hard mask at `MACVO.run_pair`): drop any observation with
  `c < min_static_conf`. This is where observations that the head is highly confident
  are dynamic get removed entirely. They never make it into the graph. The threshold
  is a config knob.
- **Stage 2** (soft weight inside the factor graph): for observations that *did*
  survive Stage 1, multiply residual by their `c` value. A safety floor `W_EPS = 1e-3`
  clamps `w` to a near-zero-but-finite value, protecting pypose from the infinite-
  covariance edge case in ablation configs that set `min_static_conf = 0`.

### 8.5 Why both stages are needed

Stage 1 alone (hard mask only) is wasteful: it wastes valuable gradient/optimization
signal at boundary pixels that are "probably static but slightly suspicious" by
forcing a binary decision. Some pixels are genuinely ambiguous — a parked motorcycle
that might drive off any second, a pedestrian about to start walking — and the head's
calibrated probability is the right way to express that.

Stage 2 alone (soft weight only) is also wasteful: a pixel with `c = 0.02` contributes
a factor with weight `0.0004`, which is numerically indistinguishable from zero but
still costs CPU time in the LM iterations. Worse, a dynamic factor with tiny weight
but huge residual can still push the LM step in a bad direction numerically.

Doing both gives you: (a) a fast path that drops obviously-dynamic pixels entirely,
and (b) a smooth path that soft-weights everything else according to calibrated
confidence. The threshold between them is set by `min_static_conf`, and the ablation
matrix specifically tests hard-only and soft-only as the endpoints of that spectrum.

---

## 9. Calibration: what temperature scaling actually does

The head outputs a raw logit, which is then passed through `sigmoid(logit / T)` to
produce `c`. `T` is a learned scalar — the *temperature*. At `T = 1` you get the
standard sigmoid; at `T > 1` the output is flattened toward 0.5 (less confident); at
`T < 1` it's sharpened toward 0 or 1 (more confident).

### 9.1 Why calibration is a separate step

During training, the head's logits are optimized to minimize BCE, which incentivizes
the logit distribution that minimizes the training loss — *not* the logit distribution
whose sigmoid is a calibrated probability. Those two goals are similar but not
identical, and deep networks trained with BCE notoriously end up producing *over-
confident* outputs: the raw sigmoid is too close to 0 or 1 relative to the empirical
correctness rate.

Temperature scaling (Guo et al., 2017) is the standard post-hoc fix: freeze the head,
sweep `T` on a held-out split, pick the `T` that minimizes expected calibration error
(ECE) of `sigmoid(logit / T)` against the empirical label distribution.

### 9.2 Why calibration matters *here* specifically

In most classification problems, over-confident outputs are a nuisance but not a
functional bug — you're going to argmax anyway. In our setup, the output `c` is used
as a continuous weight on a factor's residual (§8), and the math only works if `c` is
an actual probability. An over-confident head would over-weight good factors and
over-down-weight bad factors, producing a bimodal mask that behaves like a hard
threshold with extra steps. Calibration is what keeps the soft-weighting regime
meaningful.

### 9.3 Why `T` is a buffer, not a Parameter

During the primary training loop, `T = 1` and the head's logits are trained against
BCE. During the post-hoc calibration step, everything else is frozen and only `T` is
swept — but the sweep is done offline, not via gradient descent. So `T` never needs
to be a `Parameter` (something that gets gradients during training); it's a *learned
scalar* that's set once at the end of training and then never updated. That matches
the semantics of a PyTorch `buffer`, not a `Parameter`. The spec pins this down to
avoid a common footgun where someone wires `T` into the optimizer and it drifts
during training.

---

## 10. Training data and its constraints

The head needs three things per training sample:

1. A stereo image pair (for FlowFormerCov).
2. An IMU window between the two frames' timestamps (for AirIMU).
3. A ground-truth relative pose in the camera frame (for `L_dyn`'s rigid flow).

Nothing else. No segmentation, no instance labels, no manual dynamic annotations.

### 10.1 Which datasets work

- **VIODE** ✓ — synthetic, stereo + IMU + GT poses. Primary training dataset.
- **TartanAir 2** ✓ — synthetic, stereo + (usually) IMU + GT poses. Secondary.
- **EuRoC** ✓ — real, stereo + IMU + GT poses. Used for dataset-generalization eval.
- **KITTI** ✓ (with caveats) — real, stereo + (limited) IMU + GT poses from GNSS.
  Pose frequency is too low for direct IMU windows but GPS-interpolated poses work.
- **TUM-VI** ✓ — real, stereo + IMU + GT poses. Good dataset-generalization target.

Notably excluded: any dataset without IMU, and any dataset without GT camera poses.
The former would break the inference-time architecture; the latter would break
`L_dyn` supervision. (`L_cyc` still works, so `L_cyc`-only training on stereo-only
datasets is possible as a future path.)

### 10.2 Why synthetic primary, real eval

The method is easier to validate on synthetic data because:

- GT poses are exact, not GNSS-noisy.
- Depth is exact from the render engine, not stereo-estimation-noisy.
- Dynamic objects are explicitly scripted, so we know where they are and can sanity-
  check the head's outputs against ground truth even though we don't *use* those
  labels for training.

Training on synthetic and evaluating on real (EuRoC) is the canonical setup for
testing whether the method has learned something transferable vs something that
exploits synthetic-data artifacts. The ablation matrix's A8 row is exactly this test.

### 10.3 The pose frame pitfall

`gt_R`, `gt_t` can mean several things depending on the dataset:

- Body-frame pose (IMU body frame) — VIODE.
- Camera-frame pose — TartanAir.
- World-frame pose of body — EuRoC, KITTI.

The rigid flow formula `X_c_t = R X_c + t` requires `(R, t)` to be the *camera-to-
camera* relative pose. Using the wrong frame produces systematically wrong rigid flow,
which produces systematically wrong soft labels, which trains the head to flag the
wrong pixels as dynamic. This is a silent failure mode: training will "converge", the
loss will look fine, but the head will be useless at inference.

The spec's §4.8 and §12 call this out explicitly as a deferred implementation item.
Every training dataset wrapper needs to audit its pose-frame convention and convert
to camera-frame relative pose before returning.

---

## 11. Deployment: what the full inference path looks like

At deploy time, for each new frame:

1. **Stereo pair + IMU window come in.**
2. **FlowFormerCov** (frozen) produces flow, flow covariance, depth, and (newly tapped)
   the context feature map.
3. **AirIMU** (frozen) produces `f_imu`.
4. **StaticConfidenceHead** reads `(f_ctx, flow, cov, f_imu, h_prev)`, advances its
   hidden state, outputs `c_full`. No GT pose, no GT anything — the head has been
   trained to do without.
5. **MAC-VO's keypoint selector** picks keypoints from the flow as usual.
6. At each keypoint, **sample `c_full`** → per-keypoint static confidence `w_i`.
7. **Stage 1 filter**: drop keypoints with `w_i < min_static_conf`.
8. **Build the factor graph** with surviving keypoints; each factor stores its `w_i`.
9. **Reprojection factor's `forward()`** returns `w_i · r_i(T)`; pypose's LM runs
   as normal.
10. **Pose estimate out.**

The hidden state `h_t` is instance state on the frontend module and gets carried
across calls to `estimate_pair`. It's reset on explicit sequence boundaries or on
long timestamp gaps.

### 11.1 What happens when things go wrong

- **IMU dropout**: runtime config flip to `imu_fusion: none` (visual-only fallback) or
  graceful degradation — the head without IMU is weaker but still functional. A
  well-designed deployment would monitor IMU health and switch modes automatically.
- **Hidden state corruption**: if the head produces NaN or an all-zero output, reset
  the hidden state. This shouldn't happen after GradientClip and logit clamping in
  training, but defensive code helps.
- **Extreme motion**: at very high angular rates, AirIMU's preintegration becomes
  unreliable. The `dt_reset_ms` check fires and the hidden state resets, preventing
  corrupted state from propagating.
- **Cold start**: the first few frames run with zero hidden state and are slightly
  underconfident. Stage 2's soft-gating handles this gracefully: ambiguous-looking
  factors contribute reduced-but-nonzero weight, so the PGO still gets usable
  observations.

### 11.2 Compute budget

At inference, compared to vanilla MAC-VO, the new work is:
- One context-encoder output stash (free, just retains an existing tensor).
- One AirIMU forward pass per frame pair (~2 ms on CPU, much less on GPU).
- One StaticConfidenceHead forward pass (~5-10 ms on GPU for a 640×480 image at
  H/8 resolution).
- Keypoint sampling + Stage 1 filter (sub-ms).
- Per-factor weight multiplication in `Reproj_TwoFramePGO.forward()` (no measurable
  overhead).

Total: well under 20 ms/frame added on a typical GPU. MAC-VO's target frame budget is
~30-50 ms/frame, so this is a 20-50% overhead — acceptable, and probably reducible
through kernel fusion if the frame budget gets tight.

---

## 12. Why this will probably work — and where it might not

### 12.1 Reasons for optimism

- **The rigid-flow identity is tight.** It's a geometric fact, not a learned
  heuristic. The only slack comes from sensor noise and occlusion, both of which are
  addressable with validity masks.
- **The features are rich.** FlowFormerCov is a modern, strong flow estimator, and
  its context features encode scene-level appearance. A small head reading those
  features should have plenty of signal to learn from.
- **The training target is smooth and physically meaningful.** Depth-adaptive
  thresholds and focal BCE turn a naturally-imbalanced problem into a tractable one.
- **The gating mechanism is mathematically clean.** Multiplying whitened residuals by
  `w` is equivalent to covariance scaling and integrates naturally into pypose's IRLS
  loop without hacks.
- **Every architectural choice has an ablation.** If any piece turns out to be
  unnecessary, the ablation matrix will catch it. This isn't a "kitchen sink" design;
  it's a minimum set of components that each earn their keep.

### 12.2 Reasons to be cautious

- **Sim-to-real gap.** Training on VIODE and expecting it to work on EuRoC is a tall
  order. The head might learn to exploit VIODE's rendering artifacts instead of
  genuine motion cues. The A8 ablation is designed to catch this, but it's also
  designed to be a hard test — if A8 fails, the method needs rethinking, not just
  tuning.
- **FOE blind spot.** Pixels directly at the focus of expansion are invisible to
  residual-based supervision. If the eval sequences have lots of straight-ahead
  driving, the method will be weaker there. An acceptable failure mode but worth
  knowing about.
- **Hidden state drift during long sequences.** The ConvGRU is trained on 4-frame
  windows but run on thousands-of-frame sequences at inference. If it starts
  accumulating state that the training distribution never encountered, output quality
  might degrade over long runs. The `dt_reset_ms` and sequence-boundary resets
  mitigate this but don't eliminate it.
- **Interaction with MAC-VO's motion model.** MAC-VO uses a constant-velocity motion
  model to initialize LM iterations. If that initialization is badly biased by a
  dynamic-heavy previous frame, the first LM step might move the pose in the wrong
  direction, then struggle to recover. The gating mitigates this, but it's a feedback
  loop worth watching in practice.
- **Calibration drift between datasets.** `T_calib` is fit on a held-out split of the
  training distribution. On a new dataset with different noise statistics, the
  calibration might be off. This is probably fixable by re-running the calibration
  step on a small labeled subset of each new deployment target, but it's operational
  overhead.

### 12.3 The scientific story

The paper this method would support reads:
*"We show that a small (~0.6M parameter) temporally-recurrent head, trained with GT-
pose-based rigid flow residuals and conditioned on AirIMU features, learns to predict
per-pixel static confidence that meaningfully improves MAC-VO's ATE and RPE on both
synthetic and real dynamic-scene benchmarks. Crucially, the head operates on a hard-
frozen MAC-VO backbone, so training is data-efficient and preserves MAC-VO's
calibrated covariance. Ablations show IMU conditioning is most valuable during fast
camera motion, temporal recurrence smooths the output without requiring explicit
temporal losses, and forward-backward cycle-consistency provides a free auxiliary
signal that enables training on sequences without ground-truth pose."*

If it works, that's a clean and defensible contribution. If it doesn't, the ablation
matrix will tell you which piece failed and where to look next.

---

## 13. Reading guide

If you want to understand the method as quickly as possible:

1. **Read §2** (rigid flow identity) — everything else builds on this.
2. **Read §5** (IMU branch and train/test asymmetry) — this is the single most
   important design decision in the whole method.
3. **Read §8** (PGO math) — the gating mechanism is the simplest part but the one
   most often misunderstood.
4. Skim the rest as needed.

If you want to argue with the method:

- §3.5 (occlusion) and §3.4 (FOE blind spot) are the physics-level failure modes.
- §7.5 (what the loss does *not* include) is where the design most differs from
  standard practice.
- §12.2 (reasons to be cautious) lists the empirical risks.

If you want to extend it:

- **Test-time adaptation** on deploy sequences using `L_cyc` alone. The cycle-
  consistency signal is dataset-agnostic and doesn't need GT pose.
- **Multi-frame PGO gating**. Currently the gating is two-frame; a sliding-window BA
  with the same confidence weights across multiple frames should work identically.
- **Joint training with PGO in the loop**. Freeze FlowFormerCov but differentiate
  through the LM iterations to get a pose-level loss on top of the pixel-level loss.
  This is the natural next step and is explicitly out of scope for this spec.
- **Richer IMU fusion**. AirIMU gives us a compact feature; cross-attention between
  IMU tokens and visual tokens would let the network use IMU in a spatially-varying
  way. Currently FiLM applies it uniformly across space, which is the right prior but
  not the strongest possible fusion.

---

## 14. Common misconceptions (important)

This section is intentionally blunt, because these points are easy to misread when
implementing quickly.

### 14.1 "If GT pose is used in training, inference is impossible without GT."

False. GT pose is used only to *build supervision targets* (`c_target`) in `L_dyn`.
The forward path at inference remains:

1. frozen FlowFormerCov features,
2. frozen AirIMU feature,
3. trainable head,
4. confidence-weighted PGO.

No GT tensor is read at runtime.

### 14.2 "`c` is the same thing as flow covariance."

False. Covariance models **measurement uncertainty** (how noisy a match is under the
rigid-flow estimator). Static confidence models **rigidity likelihood** (how likely a
pixel belongs to the static world). They can correlate, but they are not equivalent:

- High covariance + high static confidence: textureless but static wall.
- Low covariance + low static confidence: moving object with clean texture.

That is exactly why the head reads both `flow/cov` and higher-level context features.

### 14.3 "Hard masking alone is enough."

Usually false in dynamic scenes with boundary ambiguity. Pure hard masking is brittle
to threshold choice and throws away useful but uncertain observations. The two-stage
design (hard + soft) is not redundant:

- hard gate removes obvious dynamic outliers early,
- soft weighting preserves graded trust near motion boundaries.

### 14.4 "Soft weighting alone is enough."

Also usually false. Very low-confidence factors still cost solver time and can create
numerical pathologies if they survive in large numbers. Stage-1 filtering reduces this
burden before optimization.

### 14.5 "This is equivalent to semantic masking."

No. Semantic masks ask "what class is this object?" while this method asks "is this
pixel's motion consistent with rigid ego-motion?" A parked car can be highly static;
a moving object with an unseen category can still be rejected.

### 14.6 "Why not train the whole backbone end-to-end?"

Because the design objective here is *controlled intervention*: preserve MAC-VO's
pretrained flow/covariance behavior and only learn an additional reliability field.
This limits compute, limits overfitting risk, and keeps ablations interpretable.

### 14.7 "The recurrence is optional, so it's probably unimportant."

Not necessarily. Recurrence is optional for ablation hygiene, but it is expected to be
load-bearing whenever dynamic evidence is temporally delayed (e.g., object starts
moving after a short static period) or briefly ambiguous in a single frame pair.

---

## 15. V2 architecture refinements: theory and intuition

This section formalizes five refinements added after the first draft. The design goal is
unchanged: preserve frozen FlowFormerCov + frozen AirIMU, and improve the trainable head
only.

### 15.1 Why split confidence into `p_static` and `p_visible`

A single confidence scalar conflates two different latent variables:

1. rigidity (is this world point static?),
2. observability (is this correspondence reliable right now?).

Those are related but not equivalent. A static point can be untrackable under occlusion;
a moving object can be clearly trackable.

A cleaner factorization is:

```text
c_eff = p_static * p_visible
```

Interpretation:

- `p_static` captures motion consistency with rigid scene assumptions.
- `p_visible` captures correspondence validity (cycle/occlusion/trackability).
- Their product is the probability that a pixel is both static and usable by the solver.

This reduces label noise and improves calibration because each head is trained on a
more specific target.

### 15.2 Why add H/4 refinement on top of H/8 recurrence

`H/8` recurrence is computationally efficient and good for temporal memory, but boundary
precision is limited at that scale. Dynamic outliers that hurt pose often live at thin
structures and object edges.

The refinement branch is a coarse-to-fine compromise:

```text
temporal reasoning at H/8 (cheap) + spatial sharpening at H/4 (targeted)
```

This preserves runtime while improving precisely the pixels that dominate robust-PGO
failure cases.

### 15.3 (superseded by §5.5)

The IMU-rigid residual proxy was originally described here as a V2 refinement. It has
been promoted to §5.5 as a core design element: without it, the train/test symmetry
argument (same forward-pass input distribution at train and test) does not actually
hold, because the geometric discrepancy signal exists only in the training loss.

See §5.5 for the full formulation.

### 15.4 Why spatially adaptive IMU fusion beats global-only FiLM

Global FiLM assumes one motion-conditioning vector should modulate all pixels the same
way. But ego-motion manifestation is depth- and location-dependent. Near and far regions
should react differently to the same motion prior.

Depth-bin FiLM (or tiny cross-attention) introduces the right inductive bias:

- global motion summary from IMU,
- spatially varying modulation over image regions.

This is a synthesis between "IMU is global" and "image evidence is local."

### 15.5 Why learn a solver-aware confidence mapper `w = g(c_eff)`

The solver consumes residual weights, not probabilities directly. If residual is scaled
by `w`, cost scales by `w^2`; therefore, solver sensitivity is nonlinear in confidence.

Using fixed `w = c_eff` assumes a calibration-to-optimization mapping that may be
suboptimal. A monotonic learned map:

```text
w = g(c_eff),   g increasing,   w in [eps, 1]
```

lets the model adapt confidence to LM dynamics while preserving ordering semantics.
Monotonicity prevents pathological inversions ("higher confidence -> lower weight").

### 15.6 Updated loss semantics with factorized heads

A principled decomposition is:

```text
L_static  : supervise p_static from GT-rigid residual target
L_visible : supervise p_visible from cycle/occlusion target
L_smooth  : regularize c_eff = p_static * p_visible
L_joint   : optional weak coupling on c_eff
```

This keeps supervision aligned with latent variable meaning, and keeps the PGO-facing
quantity (`c_eff`) explicitly regularized.

### 15.7 What this changes scientifically

The method's claim becomes sharper:

- We do not predict "dynamicness" as one monolithic scalar.
- We predict *static rigidity* and *visibility reliability* separately.
- We feed the solver a calibrated, mapped reliability weight tailored to LM behavior.

This strengthens interpretability, ablations, and failure analysis while preserving the
original frozen-backbone philosophy.

### 15.8 Why full IMU preintegration in backend is the right next step

The frontend (§5) already consumes the full preintegration factor
`(ΔR̂, Δv̂, Δp̂, Σ_imu)`. The *same* differentiable preintegrator instance produces the
quantity that backend PGO needs in order to add an inertial factor between keyframes.
No second implementation is required — the frontend uses the factor as a conditioning
prior, the backend uses it as a hard constraint, and both share the same numerical
output.

Until the backend consumes the factor, the IMU loop is only half closed: the
frontend is *informed* by IMU but the optimized pose ignores it. Closing the loop
means letting inertial consistency directly constrain the optimized states.

The key Janusian synthesis is:

- keep visual dynamic-robust weighting (`w = g(c_eff)`),
- add inertial preintegration constraints (sourced from the same frontend
  preintegrator) in the same optimizer.

This achieves both robustness to dynamic pixels and inertial consistency, with a
single implementation of preintegration serving both roles.

### 15.9 State-space shift: pose-only -> full VIO state

Pose-only optimization cannot represent IMU bias and velocity explicitly, so inertial
errors leak into pose estimates. Full preintegration requires per-keyframe state:

```text
x_k = {R_k, p_k, v_k, b_gk, b_ak}
```

This is the same modeling choice used by high-reliability VIO systems (e.g., VINS-like
backends) because gyro/accel bias drift is a first-order error source.

### 15.10 IMU preintegration residual structure (implementation form)

For an edge `(i, j)` with preintegrated measurements — the same outputs
`(ΔR̂, Δv̂, Δp̂, Σ_imu)` the frontend consumes in §5.2, plus the bias Jacobians used
for first-order bias correction at optimization time:

```text
delta_R_hat, delta_v_hat, delta_p_hat, dt       # from frontend preintegrator
J_R_bg, J_v_bg, J_v_ba, J_p_bg, J_p_ba          # bias Jacobians (see note)
Sigma_imu                                        # 9×9 (or 15×15 with bias-RW blocks)
```

**Note on bias Jacobians.** A frontend-only preintegrator need not emit the bias
Jacobians because the frontend consumes the factor as a conditioning vector, not as
a constraint that gets re-linearized at every LM step. Backend use *does* require
them, because the optimizer updates bias estimates and the factor must be cheaply
re-evaluated without redoing the full preintegration loop. When promoting the
preintegrator from frontend-only to shared, extend it to return the five Jacobians
in addition to the factor mean and covariance. Do not ship a second, backend-only
preintegrator — that is a well-known source of silent numerical drift between the
two consumers.

Residuals:

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
```

The inertial factor vector is:

```text
r_imu = [r_R, r_v, r_p, r_bg, r_ba]   # 15D
```

### 15.11 Joint objective with dynamic-robust visual terms

The backend objective becomes:

```text
L = sum_i (w_i * r_vis_i)^T * Sigma_vis_i^-1 * (w_i * r_vis_i)
  + sum_edges r_imu^T * Sigma_imu^-1 * r_imu
  + prior terms
```

with `w_i = g(c_eff_i)` from the confidence head. This cleanly combines:

- dynamic-scene visual robustness (learned confidence weighting),
- physically grounded inertial consistency (preintegration factors).

### 15.12 Why this closes the loop on IMU reliability

Without backend IMU factors, poor IMU quality can only be handled heuristically in the
head. With backend factors:

1. IMU consistency directly affects optimization updates,
2. visual residuals can expose inertial mismatch (innovation),
3. covariance inflation / gating policies can be tied to this innovation.

So IMU quality is no longer "assumed"; it is continuously validated by geometric fit.

### 15.13 Practical cautions

- Full preintegration is only as good as time sync and extrinsic calibration.
- Bias priors and random-walk terms are mandatory; otherwise the problem is weakly
  constrained and drifts.
- Start with short sliding windows (5-10 keyframes) for stability and runtime control.
- Keep pose-only IMU factor as a fallback mode for debugging and ablations.
