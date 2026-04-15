# Static-Confidence Reimplementation Design (Main Checkout)

## 1. Problem Statement

Implement the static-confidence architecture in `dynamask_vio/` with strict conformance to:

- `dynamask_vio/docs/2026-04-11-static-confidence-head-macvo-spec.md`
- `dynamask_vio/docs/2026-04-11-static-confidence-head-method-theory.md`

Execution constraints:

- Start from current **main checkout** and discard prior worktree implementation.
- Do **not** modify `references/`.
- Prioritize spec/theory correctness over legacy behavior compatibility.
- Support both CUDA and ROCm via runtime/config backend selection.
- Ensure smooth operation for both TartanAir and VIODE (rosbag-native).

## 2. Selected Approach

Chosen approach: **Spec-first clean-room staged build**.

Rationale:

- Lowest discrepancy risk against the two source docs.
- Clear traceability from each requirement to code and tests.
- Enables strict phase gates and explicit mismatch closure.

## 3. Architecture and Sequencing

Implementation is decomposed into five sequential sub-projects:

### A. Foundation and Runtime Controls

- Enforce static-confidence mode where required by docs.
- Add/normalize runtime backend selector: `auto|cuda|rocm|cpu` + device index.
- Ensure ROCm aliases are accepted and routed to valid PyTorch device semantics.

### B. Data and Training Stack

- Build/align DynamicHead training data path.
- Add/align VIODE rosbag dataset ingestion with synchronized camera/IMU extraction.
- Keep TartanAir path aligned under a shared sample schema.
- Integrate AirIMU + DynamicHead training loop, losses, and calibration path.

### C. Frontend/Inference Integration

- Tap FlowFormer context required by DynamicHead.
- Produce frontend outputs:
  - `p_static`
  - `p_visible`
  - `static_conf`
  - `static_weight`
- Propagate confidence fields through map/observation structures.
- Apply Stage-1 hard gating at keypoint selection/tracking path per spec.

### D. Backend Upgrade (Dedicated Phase)

- Extend backend state and factors per spec/theory backend sections.
- Integrate IMU preintegration factorization and sliding-window optimization path.
- Apply Stage-2 residual/Jacobian weighting using static confidence.

### E. Evaluation and Discrepancy Closure

- Run targeted and full regression validation in micromamba env `dynamask`.
- Maintain requirement-by-requirement discrepancy matrix until all resolved.

## 4. Component Contracts

### 4.1 DynamicHead Contract

Inputs:

- Flow context/features from optical flow backbone
- Optional/available IMU latent features from AirIMU branch

Outputs:

- `p_static`: static probability
- `p_visible`: visibility probability
- `static_conf`: confidence score used by gating/weighting
- `static_weight`: optimizer-facing weight tensor

Rules:

- Explicit shape/range checks at module boundary.
- Fail-fast on invalid tensor semantics (shape/dtype/range mismatch).

### 4.2 Frontend Contract

- `StaticConfidenceFrontend` must return all confidence fields for each frame pair.
- No silent fallback to legacy outputs when static-confidence mode is active.
- Invalid backend/device/mode requests raise explicit errors.

### 4.3 Map/Optimization Contract

- Map observation structures persist confidence fields without lossy conversion.
- Optimizer consumes `static_weight` for weighted residual/Jacobian handling.
- Stage-1 hard gate and Stage-2 weighting are both enforced.

### 4.4 Dataset Contract (TartanAir + VIODE)

- Unified sample schema across datasets for train/infer code-path reuse.
- VIODE loader remains rosbag-native with strict topic/calibration validation.
- Timestamp synchronization and missing-stream errors are explicit and actionable.

## 5. Data Flow

Inference:

1. Runtime backend/device resolved from CLI/config.
2. Frame pair (+ IMU context when configured) enters model.
3. Flow backbone produces motion/context features.
4. DynamicHead/AirIMU fusion produces confidence fields.
5. Stage-1 frontend gating filters measurements.
6. Map stores confidence-aware observations.
7. Backend applies Stage-2 weighted optimization.

Training:

1. Sequence windows are prepared from TartanAir or VIODE rosbags.
2. Unified batch schema feeds DynamicHead/AirIMU training.
3. Loss terms follow spec/theory definitions and calibration requirements.
4. Validation ensures output semantics match inference contracts.

## 6. Error Handling Policy

- No broad catch-and-ignore behavior.
- Mode/backend/device errors raise immediately with corrective message.
- Dataset parser errors include topic/calibration/timestamp context.
- Confidence tensor invariant violations fail at the producing boundary.

## 7. Testing and Validation Strategy

- Unit tests:
  - backend normalization/selection
  - dataset parsing/synchronization
  - DynamicHead output semantics and loss behavior
- Integration tests:
  - frontend -> map -> optimizer confidence propagation
  - Stage-1 + Stage-2 behavior correctness
- End-to-end regression:
  - full test suite in micromamba env `dynamask`
  - targeted checks for TartanAir and VIODE pathways

## 8. Scope Boundaries

In scope:

- Changes under `dynamask_vio/` needed for strict doc conformance.
- Runtime backend switching compatibility for CUDA and ROCm.

Out of scope:

- Modifying `references/`.
- Unrelated refactors outside static-confidence implementation needs.

## 9. Acceptance Criteria

- All required static-confidence features from both docs are implemented or explicitly reconciled.
- No unresolved discrepancy entries against the source docs.
- Runtime backend switch works for `auto|cuda|rocm|cpu`.
- TartanAir and VIODE paths execute through the intended static-confidence flow.
- Test suite remains green in `dynamask` environment.
