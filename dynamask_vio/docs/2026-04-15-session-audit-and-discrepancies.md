# 2026-04-15 Session Audit and Spec/Theory Discrepancy Report

## 1) Where the code from this session is

- **Primary implementation location:** `/home/arjun/dynamask/.worktrees/static-confidence-exec`
- **Branch:** `static-confidence-exec`
- **Latest committed implementation SHA:** `7a199b26a12c4fb1af5cd71c6208dc8ab62bee83`
- **Note:** this session also has additional **uncommitted** changes in that worktree.

---

## 2) What was done in this session (full summary)

### 2.1 Process and setup

1. Ran graph extraction/reporting (`graphify`) on the codebase.
2. Created and used isolated git worktree branch `static-confidence-exec`.
3. Stabilized environment/test execution in micromamba env `dynamask`.
4. Fixed baseline test blockers (typechecking/runtime compatibility and eval API mismatch).

### 2.2 Task 1 implementation (committed)

Implemented DynamicHead + AirIMU core modules and unit tests.

**Commit:** `827ec69ac4b9e04c2f422f31472d86bf3eaebb8c`  
**Follow-up quality commit:** `320a7212e1ba11f951ef8ec3b8a36b1e944c25e4`

**Files added/updated (Task 1):**
- `dynamask_vio/Module/Network/DynamicHead/{__init__.py,convgru.py,film.py,head.py,README.md}`
- `dynamask_vio/Module/Network/AirIMU/{__init__.py,corrector.py,preintegration.py,encoder.py}`
- `dynamask_vio/Scripts/UnitTest/{test_dynamic_head.py,test_airimu_modules.py}`

### 2.3 Task 2 implementation (committed)

Integrated static-confidence through frontend/odometry/map/PGO with tests.

**Commit:** `7a199b26a12c4fb1af5cd71c6208dc8ab62bee83`

**Files added/updated (Task 2):**
- `dynamask_vio/Module/Frontend/{Frontend.py,Matching.py}`
- `dynamask_vio/Module/Network/FlowFormerCov/flownet.py`
- `dynamask_vio/Module/Map/{Template.py,VisualMap.py}`
- `dynamask_vio/Odometry/MACVO.py`
- `dynamask_vio/Module/Optimization/TwoFramePGO/Graphs.py`
- `dynamask_vio/Scripts/UnitTest/test_static_confidence_integration.py`

### 2.4 Task 3/4 implementation (currently uncommitted in worktree)

Implemented training/data/config/runtime additions and related tests:

- `dynamask_vio/DataLoader/Dataset/DynamicHeadTrain.py`
- `dynamask_vio/Train/DynamicHead/{__init__.py,train.py,loop.py,loss.py,calibrate.py}`
- `dynamask_vio/Config/Experiment/StaticConfidenceHead/viode.yaml`
- `dynamask_vio/DataLoader/Dataset/VIODE.py`
- `dynamask_vio/Scripts/UnitTest/{test_dynamic_head_training.py,test_viode_dataset.py}`
- Runtime backend switching + device compatibility updates in:
  - `dynamask_vio/MACVO.py`
  - `dynamask_vio/Module/Frontend/{Frontend.py,Matching.py,StereoDepth.py}`
  - `dynamask_vio/Module/Optimization/TwoFramePGO/Optimizer.py`
- Export/wiring updates:
  - `dynamask_vio/DataLoader/__init__.py`
  - `dynamask_vio/Train/__init__.py`
- Docs update:
  - `dynamask_vio/README.md`

### 2.5 Verification done in-session

- Targeted static-confidence/training/data tests passed.
- Full unit test suite in worktree passed:
  - `111 passed`
  - command used with allocator guard:
    - `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True micromamba run -n dynamask python3 -m pytest -q`

---

## 3) Documents compared

Compared implementation against:

1. `dynamask_vio/docs/2026-04-11-static-confidence-head-macvo-spec.md`
2. `dynamask_vio/docs/2026-04-11-static-confidence-head-method-theory.md`

---

## 4) Alignment summary

### 4.1 Implemented/aligned

1. DynamicHead and AirIMU module creation and integration path.
2. Frontend static-confidence outputs (`p_static`, `p_visible`, `static_conf`, `static_weight`).
3. FlowFormer context tap for head conditioning.
4. Map schema extension with per-observation `static_conf`.
5. MACVO Stage-1 gating (`min_static_conf`) and Stage-2 weighted residual/Jacobian path.
6. New training stack and dataset wrapper (`Train/DynamicHead/*`, `DynamicHeadTrainDataset`).
7. VIODE rosbag dataset loader for required topics/calibration files.

### 4.2 Discrepancies (spec/theory gaps or intentional deviations)

1. **Backend full IMU factor upgrade (§14.10–14.13, theory §15.8) is not implemented.**  
   No full `{R,p,v,bg,ba}` state optimization, no IMU preintegration factors in backend graph.

2. **VisualMap frame-state extension for IMU backend path (§14.11) is not implemented.**  
   Missing new frame fields (`vel_w`, `bias_g`, `bias_a`) and dedicated IMU factor edge store.

3. **Sliding-window backend optimizer path (§14.11) is not implemented.**  
   Backend remains two-frame PGO-based in current implementation path.

4. **Hard-freeze verification checklist assertions (§11) are only partially enforced.**  
   Frontend freezing is applied, but explicit assertion set (e.g., gradient-None smoke assertion for frozen backbone) is not fully codified as written.

5. **Factorized loss semantics (theory §15.6) are only partially implemented.**  
   Current training loss composes primarily on effective confidence path; explicit separate `L_static` vs `L_visible` supervision split is not fully realized as formalized in §15.6.

6. **Spatially adaptive IMU fusion (theory §15.4 / spec A12 concept) is not implemented.**  
   Current fusion is global FiLM/concat path, not depth-bin or comparable spatially adaptive fusion.

7. **`convlstm` recurrence option in spec config example (§6) is not implemented in head.**  
   Current head supports `convgru` or `none`.

8. **Spec path layout in §10.1 references `dynamask_vio/references/MAC-VO/...`, but implementation is in `dynamask_vio/...`.**  
   This is an **intentional deviation** to satisfy user direction not to modify `references/`.

9. **`Optimizer.py` was modified despite §10.3 “NOT touched”.**  
   Change made intentionally to broaden runtime device validation for ROCm-compatible routing.

10. **Static-confidence experiment config coverage is incomplete for secondary dataset presets.**  
    Added `viode.yaml`; no parallel dedicated `StaticConfidenceHead` experiment config for TartanAir in this session.

---

## 5) Current state to be aware of

1. Session changes are split between:
   - **Committed** (Task 1/2 commits listed above),
   - **Uncommitted** (Task 3/4 and runtime/backend/docs updates).
2. Main checkout at `/home/arjun/dynamask` does **not** yet contain all worktree code unless explicitly copied/cherry-picked/merged.
