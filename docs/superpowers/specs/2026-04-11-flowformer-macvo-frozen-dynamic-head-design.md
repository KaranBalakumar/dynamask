# FlowFormer MAC-VO Frozen Backbone + IMU-Conditioned Dynamic Head

**Date:** 2026-04-11  
**Scope:** Stereo + IMU dynamicness-aware VO where only the dynamic head is trainable.

> Renderer-safe version: this document intentionally avoids LaTeX blocks so it renders cleanly in plain Markdown viewers.

---

## 1) Goal

Use MAC-VO pretrained FlowFormerCov and AirIMU as frozen experts, then train only a lightweight dynamic head that outputs a per-pixel dynamicness probability for:

1. Soft weighting in PGO / GN.
2. Hard masking of dynamic correspondences.
3. Static-biased keypoint selection.

---

## 2) Frozen vs trainable

| Module | Source | Trainable |
|---|---|---|
| FlowFormer context encoder | MAC-VO FlowFormer | No |
| FlowFormer memory encoder | MAC-VO FlowFormer | No |
| FlowFormer memory decoder + GMA update | MAC-VO FlowFormerCov | No |
| Flow covariance branch (`MemoryCovDecoder`) | MAC-VO FlowFormerCov | No |
| AirIMU corrector + preintegrator (`IMUEncoder`) | existing repo | No |
| New dynamic head | new | Yes |
| Optional IMU projection layer inside dynamic head | new | Yes |

Implementation rule:
- `requires_grad=False` for all frozen modules.
- keep frozen modules in eval mode while training dynamic head.

---

## 3) Exactly how IMU input is used in the dynamic head

This is the exact signal path.

## 3.1 IMU outputs from the frozen branch

From `IMUEncoder` (already in repo):

- `f_imu`: global IMU feature vector, shape `[B, 128]`
- `delta_R`: relative rotation, shape `[B, 3, 3]`
- `delta_p`: relative translation, shape `[B, 3]`
- `Sigma_preint`: preintegration covariance, shape `[B, 9, 9]`

## 3.2 Build rigid-flow prior from IMU pose

At low-res grid (`H' = H/8`, `W' = W/8`):

```text
For each pixel x_i:
  X_i   = D_t(x_i) * K^{-1} * [u_i, v_i, 1]^T
  X_i'  = delta_R * X_i + delta_p
  x_i'  = project(K * X_i')
  f_rigid_imu_i = x_i' - x_i
```

This gives an IMU+depth rigid-motion expectation per pixel.

## 3.3 Create IMU uncertainty scalar

```text
c_imu = log(trace(Sigma_preint) + eps)     # shape [B, 1]
```

Broadcast to `[B, 1, H', W']`.

## 3.4 Broadcast IMU feature map

Project `f_imu` to lower channel count and tile spatially:

```text
z_imu = MLP(f_imu)                         # [B, d], recommended d=32
Z_imu = tile(z_imu, H', W')               # [B, d, H', W']
```

## 3.5 Final fusion tensor for dynamic head

Use frozen visual-motion outputs + IMU priors:

```text
f_flow           : [B, 2, H', W']     # frozen flow
log_cov_flow     : [B, 2, H', W']     # log(sigma_u^2), log(sigma_v^2)
f_rigid_imu      : [B, 2, H', W']
r_flow           : [B, 2, H', W']     # f_flow - f_rigid_imu
|r_flow|         : [B, 1, H', W']
Z_imu            : [B, d, H', W']
c_imu_map        : [B, 1, H', W']

X_dyn = concat([f_flow, log_cov_flow, f_rigid_imu, r_flow, |r_flow|, Z_imu, c_imu_map], channel)
```

With `d=32`, `X_dyn` has 42 channels.

Summary: IMU influences dynamic prediction in 3 ways:
1. Pose prior (`delta_R`, `delta_p`) -> rigid flow prior.
2. IMU global context (`f_imu`) -> broadcast conditioning map.
3. IMU confidence (`Sigma_preint`) -> uncertainty-conditioning scalar map.

---

## 4) Dynamic head architecture

Recommended stable head:

```text
Input: X_dyn [B, C_in, H', W']
Conv1x1(C_in -> 128) + SiLU
Conv3x3(128 -> 128) + GroupNorm + SiLU
Conv3x3(128 -> 64)  + GroupNorm + SiLU
Conv1x1(64 -> 1)    -> dyn_logit [B,1,H',W']
p_dyn = sigmoid(dyn_logit)                # [B,1,H',W']
```

Optional extra channel:

```text
alpha_dyn = softplus(head_alpha(...))     # [B,1,H',W']
```

Use only `p_dyn` first for stability; add `alpha_dyn` later if needed.

---

## 5) Losses (no GT dynamic labels required)

No proxy dynamic GT is assumed.

## 5.1 Residual-mixture loss (main)

Let:

```text
r_i = ||f_flow_i - f_rigid_imu_i||_2
p_i = p_dyn_i
sigma_s_i^2 = frozen static variance from flow covariance
sigma_d^2 = fixed large dynamic variance
```

Loss:

```text
L_mix = - mean_i log(
          (1 - p_i) * N(r_i; 0, sigma_s_i^2)
          + p_i     * N(r_i; 0, sigma_d^2)
        )
```

This pushes:
- small explainable residual -> static (low p_i),
- large non-rigid residual -> dynamic (high p_i).

## 5.2 Edge-aware smoothness

```text
L_smooth = EdgeAwareTV(p_dyn, I_t)
```

## 5.3 Optional temporal consistency

```text
L_temp = |p_t - warp(p_{t-1})|_1
```

Total:

```text
L_dyn = 1.0 * L_mix + 0.05 * L_smooth + 0.1 * L_temp
```

Set `L_temp=0` for strict two-frame training.

---

## 6) How dynamic score is used in optimization

## 6.1 Soft weight in PGO / GN

```text
w_i = (1 - p_dyn_i)^gamma        # gamma in [1, 2]
Omega_i = w_i * inv(Sigma_i + lambda * I)
```

Use `Omega_i` in normal equations.

If `alpha_dyn` is used:

```text
Sigma_i' = Sigma_i + alpha_dyn_i * I
Omega_i = w_i * inv(Sigma_i' + lambda * I)
```

## 6.2 Hard dynamic mask

```text
m_i = 1 if p_dyn_i < tau_dyn else 0
```

Use mask to reject dynamic correspondences before factor construction.

## 6.3 Keypoint gating score

```text
kp_score_i =
    (1 - p_dyn_i)
  * 1 / sqrt(det(Sigma_i + lambda * I))
  * texture_i
```

Then apply NMS + top-K.

---

## 7) Training protocol (dynamic-head only)

1. Load frozen FlowFormerCov (MAC-VO checkpoint).
2. Load frozen AirIMU (`IMUEncoder` checkpoint).
3. Build dynamic head (+ optional IMU projection MLP).
4. Forward:
   - frozen flow/cov inference,
   - frozen IMU inference,
   - compute rigid IMU flow,
   - build `X_dyn`,
   - predict `p_dyn`.
5. Compute `L_dyn`.
6. Backprop only through dynamic head params.

Suggested optimizer:
- AdamW, lr `1e-4`, weight decay `1e-4`
- grad clip `1.0`
- mixed precision okay (keep geometry in fp32).

---

## 8) Repo mapping

Use existing modules:

- IMU: `dynamask_vio/models/imu_encoder.py`
- AirIMU: `dynamask_vio/models/airimu_corrector.py`
- existing static weighting pattern: `dynamask_vio/models/differentiable_ba.py`
- FlowFormerCov reference:  
  `dynamask_vio/references/MAC-VO/Module/Network/FlowFormerCov/`

New files to add:

1. `dynamask_vio/models/dynamic_head_flowformer.py`
2. `dynamask_vio/models/flowformer_dynamic_wrapper.py`
3. `dynamask_vio/losses/dynamic_mixture_loss.py`

---

## 9) Recommended ablations

1. Vision-only dynamic head (remove IMU inputs).
2. IMU-conditioned head (full model).
3. Soft-only gating vs hard-only gating vs combined.
4. With/without covariance term in keypoint score.
5. With/without edge-level gating in pose graph.

Report:
- ATE / RPE
- inlier ratio
- residual separation (top dynamic quantile vs static quantile)
- solver stability and runtime.

---

## 10) Final recommendation

Start with one output only:

```text
p_dyn in [0,1], shape [B,1,H',W']
```

It is enough for:
- soft weighting (`w_i = (1 - p_i)^gamma`),
- hard masking (`p_i < tau`),
- keypoint gating.

Keep everything else frozen first. This gives a clean experiment boundary and fastest path to validate dynamic gating gains.

