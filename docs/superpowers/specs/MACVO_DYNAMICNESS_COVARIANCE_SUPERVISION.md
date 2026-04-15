# Dynamicness + Covariance Supervision Design (Stereo+IMU, VIODE)

This document consolidates the full design discussion around:

1. Adding IMU-aware dynamicness reasoning to a MAC-VO-style motion aggregation pipeline.
2. Keeping training losses minimal (2 losses total).
3. Comparing **GT-motion-based** dynamicness supervision vs **ego-inconsistency-based** supervision.
4. Validating first on **VIODE stereo+IMU**.

---

## 1) Repository-grounded context

This design is grounded in the current codebase status:

- `dynamask_vio/train.py` currently uses a **3-loss stack** (`pose`, `pairwise rank`, `smoothness`).
- `dynamask_vio/data/viode_dataset.py`:
  - computes a segmentation-derived `proxy_mask` via `_make_dynamic_mask(...)`,
  - returns `gt_R`, `gt_v`, `gt_p`, IMU window/mask, intrinsics,
  - currently returns `"has_flow": False` and does not expose `proxy_mask` in the sample dict.
- `dynamask_vio/references/MAC-VO/Train/MatchingNet/loss.py` implements covariance-style supervision:
  \[
  \mathcal{L}_{cov}\sim \frac{e^2}{\sigma^2} + \log \sigma^2
  \]
  with per-iteration discounting (`gamma`).

Graphify structure also supports this split:

- FlowFormer/GMA covariance-flow components cluster around the motion module community.
- AirIMU components are in a separate IMU community.
- This motivates explicit fusion rather than assuming existing cross-coupling.

---

## 2) Core goal

Design a fast, robust stereo+IMU training objective for dynamic scenes that:

- remains minimal and stable,
- keeps covariance meaningful (not just large everywhere),
- learns dynamicness without adding many ad-hoc loss terms,
- supports both:
  - **supervised upper-bound** (with GT motion),
  - **practical self-supervised** path (ego inconsistency).

---

## 3) Recommended model factorization

Use a **factorized uncertainty model** instead of collapsing everything into one head:

- **Static covariance head**: \(\Sigma_i^{stat}\) (aleatoric + matching ambiguity under rigidity)
- **Dynamic score head**: \(s_i \in [0,1]\)
- **Dynamic covariance head** (optional but recommended): \(\Sigma_i^{dyn}\)

Compose:
\[
\Sigma_i^{tot} = \Sigma_i^{stat} + s_i \Sigma_i^{dyn}
\]

Interpretation:

- static pixels: \(s_i \approx 0\Rightarrow \Sigma_i^{tot}\approx \Sigma_i^{stat}\)
- dynamic pixels: \(s_i \approx 1\Rightarrow \Sigma_i^{tot}\) inflates in a learned direction/magnitude

This keeps semantics clean:

- dynamicness remains identifiable,
- covariance remains calibrated,
- each output has a clear physical role.

### Why not only one uncertainty head?

You can train one head to absorb dynamicness, but then:

- uncertainty and dynamicness become entangled,
- debugging/calibration becomes harder,
- ablations are less interpretable.

If minimizing complexity is critical, keep \(s_i\) scalar and \(\Sigma_i^{dyn}\) diagonal first.

---

## 4) Where IMU enters (AirIMU-conditioned design)

Inject IMU context into motion aggregation/update, not only at backend BA.

A practical design:

```text
Stereo pair -> visual encoder -> motion features ----+
                                                     | concat/fuse -> recurrent update
IMU window -> AirIMU encoder -> imu context ---------+
                                                        |-> flow update
                                                        |-> static cov head
                                                        |-> dynamic score head
                                                        |-> (optional) dynamic cov head
```

### Fusion choices

1. **FiLM-style conditioning** on aggregator/value tensors (lightweight, stable).
2. **Concatenation + 1x1 conv/MLP** before recurrent update.
3. **Cross-attention** from visual tokens to IMU token(s) (heavier).

Start with (1) or (2) for speed and easier convergence.

---

## 5) Rigid ego-flow and residual definition

Dynamicness supervision is based on inconsistency between observed flow and rigid-motion flow.

For pixel \(x_i=(u_i,v_i)\), depth \(D_i\), intrinsics \(K\), relative pose \((R,t)\):

1. Backproject:
\[
X_i = D_i K^{-1}\tilde{x}_i
\]
2. Transform:
\[
X_i' = RX_i + t
\]
3. Reproject:
\[
\tilde{x}_i' = K X_i'
\]
4. Rigid flow:
\[
f_i^{rigid} = x_i' - x_i
\]
5. Residual against predicted/observed flow:
\[
e_i = f_i^{obs} - f_i^{rigid}
\]

Large residuals indicate either:

- dynamic object motion,
- occlusion/disocclusion,
- depth/pose/time-sync errors.

So supervision should be soft/confidence-weighted, not a brittle hard oracle.

---

## 6) Two supervision sources for dynamicness

## 6.1 GT-motion-based dynamicness target (upper bound)

Use \(R_{gt}, t_{gt}\) to compute \(f^{rigid}_{gt}\), then:
\[
r_i = \lVert f_i^{obs} - f_{i,gt}^{rigid} \rVert_2
\]
Soft label:
\[
d_i = \sigma\left(\frac{r_i-\tau}{\kappa}\right)
\]
or hard label:
\[
d_i = \mathbf{1}[r_i>\tau]
\]

This is the cleanest supervision and should provide best-case segmentation quality.

## 6.2 Ego-inconsistency dynamicness target (deployment-friendly)

Replace GT motion with IMU-estimated motion (or network-estimated ego):
\[
r_i^{ego} = \lVert f_i^{obs} - f_{i,ego}^{rigid} \rVert_2
\]
Construct \(d_i^{ego}\) using the same soft/hard mapping.

This removes dependence on GT motion for scalable training settings.

## 6.3 Expected relation

On well-synchronized sequences, ego-based labels should be close to GT-based labels.
Gap appears mostly in:

- very high angular rate segments,
- severe occlusion,
- poor depth quality,
- timestamp/extrinsic mismatch.

---

## 7) Minimal 2-loss objective (recommended)

Keep exactly two terms:

1. **Covariance supervision loss**
2. **Dynamicness supervision loss**

### 7.1 Loss 1: covariance supervision (MAC-VO style NLL)

Let \(S\) be a static-anchor set (from static proxy labels and/or low-residual points):
\[
\mathcal{L}_{cov}
=
\frac{1}{|S|}\sum_{i\in S}
\left(
e_i^\top (\Sigma_i^{stat})^{-1} e_i
+
\log\det \Sigma_i^{stat}
\right)
\]

If using recurrent iterations, apply MAC-VO-style \(\gamma\)-weighted sum over iterations.

### 7.2 Loss 2: dynamicness supervision

Supervise score head \(s_i\) against target \(d_i\) (GT-based or ego-based):
\[
\mathcal{L}_{dyn}=\text{BCE}(s_i, d_i)
\]
or focal BCE if class imbalance is severe.

Total:
\[
\mathcal{L}
=
\lambda_{cov}\mathcal{L}_{cov}
+
\lambda_{dyn}\mathcal{L}_{dyn}
\]

### Important coupling

Even with only these two losses, dynamicness still affects covariance through:
\[
\Sigma_i^{tot}=\Sigma_i^{stat}+s_i\Sigma_i^{dyn}
\]
used in downstream weighting (BA or robust aggregation).

So you keep minimal loss count but retain behavior coupling.

---

## 8) Dynamicness-as-covariance variant (your idea, formalized)

Instead of scalar-only dynamic score, represent dynamic uncertainty as 2x2 PSD:
\[
\Sigma_i^{dyn}=L_iL_i^\top,\quad
L_i=
\begin{bmatrix}
\exp(a_i) & 0\\
b_i & \exp(c_i)
\end{bmatrix}
\]

Then:
\[
\Sigma_i^{tot}=\Sigma_i^{stat}+s_i\Sigma_i^{dyn}
\]

Practical schedule:

1. start diagonal-only (`uu`,`vv`) for stability,
2. add off-diagonal (`uv`) once training is stable.

---

## 9) GT-vs-Ego ablation plan (clean and convincing)

Use the same architecture, optimizer, and data splits; only change label source.

1. **A0: Baseline** current V2.5 training stack.
2. **A1: Two-loss + GT dynamic labels** (upper bound).
3. **A2: Two-loss + Ego dynamic labels** (practical mode).
4. **A3: Optional mixed curriculum** (start GT, anneal to ego).

### Metrics

- VIO metrics: ATE, RPE.
- Dynamicness quality: IoU/F1/AP vs segmentation proxy mask (or available dynamic annotations).
- Covariance calibration:
  - \( \mathbb{E}[\|e\|^2 / \text{trace}(\Sigma)] \),
  - reliability bins/calibration plots.
- Runtime: FPS on target hardware.

### Claim you want to test

“Ego-inconsistency supervision performs nearly as well as GT-motion supervision.”

Support it with:

- small ATE/RPE gap,
- small dynamic IoU/F1 gap,
- similar covariance calibration metrics.

---

## 10) VIODE-specific implementation notes

## 10.1 Data

In `VIODEDataset.__getitem__`, expose:

- `proxy_mask` (dynamic proxy from segmentation),
- optional quality/confidence mask for valid supervision regions.

Current code computes `proxy_mask` but does not return it.

## 10.2 Training signals available now

Already available in each batch:

- `gt_R`, `gt_p` (relative motion target),
- stereo images,
- IMU window and mask,
- intrinsics.

Not available currently:

- direct dense GT optical flow (`has_flow=False`).

Therefore dynamicness supervision should be rigid-flow residual based (GT or ego pose) rather than direct flow-GT mismatch.

## 10.3 Trainer changes (conceptually)

Replace current `rank` and `smooth` terms with:

- `loss_cov`
- `loss_dyn`

Keep pose-consistency term out if strict 2-loss setup is required for this phase, or reserve it for separate ablation.

---

## 11) Practical defaults for first stable run

- Dynamic label: soft target with \(\tau\) around median residual and temperature \(\kappa\) tuned per resolution.
- Loss weights: start \(\lambda_{cov}=1.0\), \(\lambda_{dyn}\in[0.2,1.0]\).
- Covariance floor: clamp eigenvalues/diagonal with epsilon.
- Ignore top residual percentile early (likely occlusion/sync outliers), then relax.
- Use iteration-discount \(\gamma\) if recurrent outputs are supervised.

---

## 12) Failure modes and mitigations

1. **Covariance inflation everywhere**
   - Mitigate with static-anchor supervision and calibration monitoring.
2. **Dynamic head predicts all-static/all-dynamic**
   - Mitigate with balanced sampling or focal BCE.
3. **Ego-label noise from bad depth/motion**
   - Mitigate with confidence weighting, soft labels, temporal smoothing.
4. **Overfitting to segmentation proxy artifacts**
   - Use proxy only for validation/aux filtering, not as sole hard truth.

---

## 13) Recommended execution order

1. Implement GT-motion dynamic labels first (A1).
2. Validate covariance calibration and dynamic maps.
3. Switch labels to ego-inconsistency (A2).
4. Compare A1 vs A2 gap.
5. Only then test heavier fusion (cross-attention, richer heads).

This order gives rapid debugging and a clear scientific story.

---

## 14) Bottom line

Yes, supervising dynamicness from **ground-truth motion** is not only possible, it is the right upper-bound baseline.
Then showing that **ego-inconsistency supervision is nearly equivalent** is a strong and practical contribution.

The minimal, clean formulation is:

- **Loss 1:** covariance NLL (MAC-VO style),
- **Loss 2:** dynamicness supervision (GT or ego labels),
- with uncertainty composition \(\Sigma^{tot}=\Sigma^{stat}+s\Sigma^{dyn}\) for dynamic-aware weighting.

