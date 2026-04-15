# Graph Report - dynamask_vio/models (filtered)  (2026-04-15)

## Corpus Check
- Corpus is ~7,911 words - fits in a single context window. You may not need a graph.

## Summary
- 147 nodes · 195 edges · 10 communities detected
- Extraction: 90% EXTRACTED · 10% INFERRED · 0% AMBIGUOUS · INFERRED: 19 edges (avg confidence: 0.5)
- Token cost: 0 input · 0 output

## Community Hubs (Navigation)
- [[_COMMUNITY_Community 0|Community 0]]
- [[_COMMUNITY_Community 1|Community 1]]
- [[_COMMUNITY_Community 2|Community 2]]
- [[_COMMUNITY_Community 3|Community 3]]
- [[_COMMUNITY_Community 4|Community 4]]
- [[_COMMUNITY_Community 5|Community 5]]
- [[_COMMUNITY_Community 6|Community 6]]
- [[_COMMUNITY_Community 7|Community 7]]
- [[_COMMUNITY_Community 8|Community 8]]
- [[_COMMUNITY_Community 9|Community 9]]

## God Nodes (most connected - your core abstractions)
1. `AirIMUCorrector` - 13 edges
2. `IMUEncoder` - 10 edges
3. `DifferentiablePreintegrator` - 8 edges
4. `DynaMaskVIO` - 8 edges
5. `BasicEncoder` - 8 edges
6. `FlowDecoder` - 7 edges
7. `FeatureMLP` - 7 edges
8. `differentiable_ba()` - 6 edges
9. `CNNEncoder` - 5 edges
10. `CorrBlock` - 5 edges

## Surprising Connections (you probably didn't know these)
- `DynaMaskVIO` --uses--> `IMUEncoder`  [INFERRED]
  dynamask_vio/models/dynamask.py → dynamask_vio/models/imu_encoder.py
- `DynaMask V2.5 model wiring.` --uses--> `IMUEncoder`  [INFERRED]
  dynamask_vio/models/dynamask.py → dynamask_vio/models/imu_encoder.py
- `V2.5 optimizer groups with RAFT layerwise LR decay.` --uses--> `IMUEncoder`  [INFERRED]
  dynamask_vio/models/dynamask.py → dynamask_vio/models/imu_encoder.py
- `FeatureMLP` --uses--> `AirIMUCorrector`  [INFERRED]
  dynamask_vio/models/imu_encoder.py → dynamask_vio/models/airimu_corrector.py
- `IMUEncoder` --uses--> `AirIMUCorrector`  [INFERRED]
  dynamask_vio/models/imu_encoder.py → dynamask_vio/models/airimu_corrector.py

## Communities

### Community 0 - "Community 0"
Cohesion: 0.1
Nodes (23): compute_reprojection_jacobian(), differentiable_ba(), gradient_clip(), _GradientClipFn, Differentiable Two-Frame Bundle Adjustment — training-only module.  Zero learnab, Exponential map so(3) -> SO(3). omega: [B, 3] -> [B, 3, 3].      Rodrigues formu, Logarithmic map SO(3) -> so(3). R: [B, 3, 3] -> [B, 3].      Inverse Rodrigues w, Update SE(3) pose via retraction (left-multiply by exp(δξ)).      Convention: δξ (+15 more)

### Community 1 - "Community 1"
Cohesion: 0.1
Nodes (13): _bilinear_sampler(), FlowHead, GradClipModule, GradientClip, MotionEncoder, RAFT-style Flow Decoder (Update Operator) for DynaMask V2.5.  Replaces the V1 Te, Encodes correlation features + current flow into motion features.      Following, Separable Convolutional GRU — horizontal then vertical updates.      Matches RAF (+5 more)

### Community 2 - "Community 2"
Cohesion: 0.11
Nodes (13): BasicEncoder, load_raft_encoder_weights(), RAFT-style BasicEncoder for DynaMask V2.  Two separate encoder instances are use, Args:             x: [B, 3, H, W] input image             f_imu: [B, imu_dim] IM, Load pretrained RAFT weights into a BasicEncoder.      Args:         encoder: th, Two 3x3 convs with InstanceNorm and residual connection.      When in_ch != out_, RAFT BasicEncoder — produces 128-ch features at 1/8 resolution.      Optionally, ResidualBlock (+5 more)

### Community 3 - "Community 3"
Cohesion: 0.16
Nodes (10): AirIMUCorrector, Broadcast interval-wise features back to per-sample sequence slots., AirIMU CodeNet-style bias/noise corrector.      Input:         accel: [B, N, 3], FeatureMLP, IMUEncoder, IMU Encoder: AirIMU Corrector → Differentiable Preintegrator → Feature MLP.  Dat, Maps preintegrated motion + covariance diagonal to FiLM features., Full IMU encoder pipeline with frozen AirIMU correction. (+2 more)

### Community 4 - "Community 4"
Cohesion: 0.19
Nodes (9): _assert_airimu_load_clean(), CNNEncoder, load_airimu_weights(), AirIMU CodeNet-compatible IMU correction module.  This module mirrors the core A, 1D CNN encoder used by AirIMU CodeNet., Fail loudly on suspicious state-dict mismatches., Resolve AirIMU checkpoint file from a file path or directory path., Load AirIMU CodeNet weights and optionally freeze the encoder. (+1 more)

### Community 5 - "Community 5"
Cohesion: 0.2
Nodes (5): GradientClip, _GradientClipFn, Score head for DynaMask V2.5., Conv score head that outputs clipped logits (no sigmoid)., ScoreHead

### Community 6 - "Community 6"
Cohesion: 0.25
Nodes (7): _project_so3(), Differentiable IMU preintegration using PyPose.  Implements Forster et al. prein, Batch skew-symmetric matrix from [B, 3] vectors → [B, 3, 3]., Project a batch of matrices onto SO(3) via SVD (nearest rotation matrix).      H, Logarithmic map SO(3) → so(3).  R: [B, 3, 3] → [B, 3].      Projects R onto SO(3, _skew_symmetric(), so3_log_map()

### Community 7 - "Community 7"
Cohesion: 0.29
Nodes (5): create_film_layers(), FiLM, FiLM (Feature-wise Linear Modulation) conditioning layers for DynaMask V2.  Appl, Single FiLM layer for one feature scale.      forward(features, f_imu):, Create FiLM layers for RAFT feature encoder's 3 residual stages.      Args:

### Community 8 - "Community 8"
Cohesion: 0.29
Nodes (4): CorrBlock, Lookup local correlation at each pyramid level.          Args:             coord, Args:             fmap1: [B, C, H, W] features from frame t-1             fmap2:, All-pairs correlation volume with multi-level pyramid.      Computes all-pairs d

### Community 9 - "Community 9"
Cohesion: 1.0
Nodes (1): Solve H x = b via Cholesky decomposition.          Args:             H: [B, 6, 6

## Knowledge Gaps
- **47 isolated node(s):** `AirIMU CodeNet-compatible IMU correction module.  This module mirrors the core A`, `1D CNN encoder used by AirIMU CodeNet.`, `AirIMU CodeNet-style bias/noise corrector.      Input:         accel: [B, N, 3]`, `Broadcast interval-wise features back to per-sample sequence slots.`, `Fail loudly on suspicious state-dict mismatches.` (+42 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **Thin community `Community 9`** (1 nodes): `Solve H x = b via Cholesky decomposition.          Args:             H: [B, 6, 6`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `IMUEncoder` connect `Community 3` to `Community 2`?**
  _High betweenness centrality (0.141) - this node is a cross-community bridge._
- **Why does `AirIMUCorrector` connect `Community 3` to `Community 4`?**
  _High betweenness centrality (0.106) - this node is a cross-community bridge._
- **Why does `FlowDecoder` connect `Community 2` to `Community 8`, `Community 1`?**
  _High betweenness centrality (0.101) - this node is a cross-community bridge._
- **Are the 5 inferred relationships involving `AirIMUCorrector` (e.g. with `FeatureMLP` and `IMUEncoder`) actually correct?**
  _`AirIMUCorrector` has 5 INFERRED edges - model-reasoned connections that need verification._
- **Are the 5 inferred relationships involving `IMUEncoder` (e.g. with `AirIMUCorrector` and `DifferentiablePreintegrator`) actually correct?**
  _`IMUEncoder` has 5 INFERRED edges - model-reasoned connections that need verification._
- **Are the 5 inferred relationships involving `DifferentiablePreintegrator` (e.g. with `FeatureMLP` and `IMUEncoder`) actually correct?**
  _`DifferentiablePreintegrator` has 5 INFERRED edges - model-reasoned connections that need verification._
- **Are the 3 inferred relationships involving `DynaMaskVIO` (e.g. with `FlowDecoder` and `BasicEncoder`) actually correct?**
  _`DynaMaskVIO` has 3 INFERRED edges - model-reasoned connections that need verification._