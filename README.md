# MAC-VO (Dynamic-VIO Branch): DRT Initialization + Shared AirIMU + IMU Backend Factors

This README documents the **current branch implementation** in this repository (not the original upstream paper-only README).  
It explains what was added, why it was added, how to run it, and how to configure the new VIO pipeline pieces.

---

## 1. What changed in this branch

This branch adds a VIO-oriented extension on top of MAC-VO:

1. **Shared AirIMU context path**
   - New package: `Module/Network/AirIMU/`
   - Files:
     - `corrector.py`
     - `preintegration.py`
     - `encoder.py`
     - `proxy.py`
   - Goal: produce one consistent IMU preintegration output used by both frontend conditioning and backend factors.

2. **DRT loosely-coupled initializer**
   - New package: `Module/Initialization/DRTLoose/`
   - Files:
     - `drt_loose.py`
     - `gyro_bias_solver.py`
     - `linear_alignment.py`
     - `gravity_refine.py`
   - Integrated into `Odometry/MACVO.py` as staged bootstrap (`method: drt_loose`).

3. **IMU preintegration edges in map state**
   - `Module/Map/Template.py`, `VisualMap.py`, `__init__.py`
   - New edge store: `imu_edges` with `(delta_R, delta_v, delta_p, Sigma, dt, bias_ref, J_*)`.
   - Frame schema extended with `vel`, `bias_g`, `bias_a`.
   - Match schema extended with confidence `c`.

4. **Two-frame IMU backend factor**
   - `graph_type: reproj_imu` in `Module/Optimization/TwoFramePGO/`
   - New graph class: `Reproj_TwoFramePGO_IMU`.
   - Visual residual + 15D IMU residual in one optimization graph.

5. **Sliding-window backend skeleton**
   - New package: `Module/Optimization/SlidingWindow/`
   - Class: `SlidingWindow_VIO_PGO`
   - Windowed multi-pair optimization flow with IMU-capable graph selection.

6. **VIODE memory-safe data path**
   - Rosbag converter: `Scripts/AdHoc/convert_viode_rosbag.py`
   - Stream loader: `DataLoader/Dataset/VIODE.py` (`VIODE_StreamSequence`)
   - Avoids loading rosbag into RAM directly.

7. **Dependency updates**
   - Updated:
     - repo root `requirements.txt`
     - repo root `environment.yml`
     - `MAC-VO/requirements.txt`
    - Added explicit CUDA/ROCm install guidance.

8. **Runtime logging / observability (spec-aligned subset)**
   - New package: `Utility/Observability/`
   - Includes:
     - `DebugLogger` + `NumericalAbort`
     - local / TensorBoard / W&B sink fan-out
     - cadence helpers and runtime dump collector
   - Integrated into:
     - `Odometry/MACVO.py`
     - `Module/Initialization/DRTLoose/drt_loose.py`
     - `Module/Network/AirIMU/preintegration.py`
     - `Module/Optimization/TwoFramePGO/Optimizer.py`
     - `Module/Optimization/SlidingWindow/Optimizer.py`

---

## 2. High-level architecture (current branch)

```text
Stereo + IMU sequence
      |
      v
MACVO bootstrap stage (optional: DRT loose)
  - collect first N init frames
  - estimate bg, R/p/v, gravity
  - write initial frame states + imu_edges
      |
      v
Normal run_pair loop
  - frontend depth/flow
  - match + map updates
  - IMU edge push (shared preintegrator output)
  - backend optimize:
      a) TwoFrame_PGO (legacy)
      b) TwoFrame_PGO with graph_type: reproj_imu
      c) SlidingWindow_VIO_PGO (windowed backend)
```

Core design intent: **single IMU preintegration definition** reused across initialization, frontend conditioning data, and backend factors.

---

## 3. New/updated files you should know first

### IMU and initialization
- `Module/Network/AirIMU/preintegration.py`
- `Module/Network/AirIMU/encoder.py`
- `Module/Initialization/DRTLoose/drt_loose.py`
- `Module/Initialization/DRTLoose/gyro_bias_solver.py`

### Odometry integration
- `Odometry/MACVO.py`

### Backend
- `Module/Optimization/TwoFramePGO/Graphs.py`
- `Module/Optimization/TwoFramePGO/Optimizer.py`
- `Module/Optimization/SlidingWindow/Optimizer.py`

### Map/state
- `Module/Map/Template.py`
- `Module/Map/VisualMap.py`
- `Module/Map/Graph.py`
- `Utility/Extensions/TensorExtension.py`

### Data conversion and loading
- `Scripts/AdHoc/convert_viode_rosbag.py`
- `DataLoader/Dataset/VIODE.py`

### Observability
- `Utility/Observability/debug_logger.py`
- `Utility/Observability/sinks/local_sink.py`
- `Utility/Observability/runtime_collectors.py`
- `Scripts/AdHoc/inspect_runtime_dump.py`
- `Scripts/AdHoc/scrub_partial_dumps.py`

---

## 4. Environment setup

Use Python 3.10+ (tested in this branch with Python 3.12 runtime).

### 4.1 Install PyTorch first (choose one backend)

```bash
# CUDA 12.1
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# CUDA 12.4
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

# ROCm 6.2
pip install torch torchvision --index-url https://download.pytorch.org/whl/rocm6.2

# CPU
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

### 4.2 Install project dependencies

From repository root:

```bash
pip install -r requirements.txt
```

From `MAC-VO/` (module-local requirements):

```bash
pip install -r requirements.txt
```

---

## 5. Running MAC-VO normally

From `MAC-VO/`:

```bash
python3 MACVO.py \
  --odom Config/Experiment/MACVO/MACVO_Performant.yaml \
  --data Config/Sequence/TartanAir_example.yaml
```

For fast mode:

```bash
python3 MACVO.py \
  --odom Config/Experiment/MACVO/MACVO_Fast.yaml \
  --data Config/Sequence/TartanAir_example.yaml
```

---

## 6. Enabling the new VIO components

## 6.1 DRT bootstrap in odometry config

Add this under `Odometry.args` in your odom config:

```yaml
Odometry:
  args:
    # existing fields ...
    init:
      method: drt_loose      # or identity
      drt_window_len: 10
      g_mag: 9.81
      gyro_max_iter: 200
      gyro_cauchy_delta: 1e-5
      track_stride: 16
      min_pair_tracks: 40
      min_pair_tracks_depth: 12
      gravity_refine_iters: 5
      accept_criteria:
        max_bg_norm: 0.5
      imu:
        ckpt_path: null
        jacobian_eps: 1e-5
        sigma_repr_mode: diag
        feature_dim: 64
        emit_jacobians: true
```

Notes:
- `drt_loose` requires IMU-bearing sequence types (e.g., EuRoC with IMU, TartanAirV2, VIODE stream).
- If bootstrap fails repeatedly, odometry falls back to identity bootstrap.

## 6.2 Two-frame backend with IMU factor

Set optimizer graph type:

```yaml
Odometry:
  optimizer:
    type: TwoFrame_PGO
    args:
      device: cpu
      vectorize: true
      parallel: true
      graph_type: reproj_imu
      autodiff: false
```

## 6.3 Sliding-window backend

Switch optimizer type:

```yaml
Odometry:
  optimizer:
    type: SlidingWindow_VIO_PGO
    args:
      device: cpu
      vectorize: true
      parallel: true
      graph_type: reproj_imu
      autodiff: false
      window_size: 5
```

---

## 7. VIODE rosbag -> stream conversion (memory-safe path)

Convert rosbag2 into chunked HDF5:

```bash
python3 Scripts/AdHoc/convert_viode_rosbag.py \
  --bag /path/to/viode_rosbag2 \
  --out /path/to/viode_stream.h5 \
  --left-topic /stereo/left/image_raw \
  --right-topic /stereo/right/image_raw \
  --imu-topic /imu/data \
  --stereo-sync-tol-ms 5.0 \
  --fx 320 --fy 320 --cx 320 --cy 320 \
  --baseline 0.25 \
  --t-bs-translation 0.0 0.0 0.0 \
  --t-bs-quaternion 0.0 0.0 0.0 1.0 \
  --gravity 9.81
```

Key behavior:
- Uses message header time when available (`header.stamp`), otherwise bag timestamp.
- Stereo pairing uses nearest-neighbor within tolerance (not exact timestamp equality).
- `T_BS` can be provided by either:
  - `--t-bs-path /path/to/extrinsic.(txt|json|npy)` containing `[tx ty tz qx qy qz qw]`, or
  - `--t-bs-translation ...` + `--t-bs-quaternion ...` CLI values.
- Writes:
  - `stereo/{left,right,time_ns}`
  - `imu/{acc,gyro,time_ns}`
  - `calib/{K,T_BS,baseline,gravity}`

### 7.1 Sequence config for VIODE stream

Create a sequence config (example):

```yaml
type: VIODE_Stream
name: VIODE_stream_seq
args:
  path: /path/to/viode_stream.h5
```

Then run:

```bash
python3 MACVO.py --odom /path/to/your_odom.yaml --data /path/to/viode_stream.yaml
```

---

## 8. State semantics in this branch

### Frame state (`VisualMap.frames`)
- `pose` (camera pose in world)
- `T_BS` (body->sensor extrinsic)
- `vel` (body velocity in world)
- `bias_g`, `bias_a`
- `need_interp`, `time_ns`, `K`, `baseline`

### Match state (`VisualMap.match`)
- Existing pixel/depth/disparity/covariance fields
- New confidence field: `c`

### IMU edges (`VisualMap.imu_edges`)
- `from_frame`, `to_frame`
- `delta_R`, `delta_v`, `delta_p`
- `Sigma`, `dt`
- `bias_ref`
- `J_R_bg`, `J_v_bg`, `J_v_ba`, `J_p_bg`, `J_p_ba`

---

## 9. Math and validation tests (what was run)

All targeted tests were run from `MAC-VO/` with:

```bash
python3 -m pytest -q -o addopts='' \
  Scripts/UnitTest/test_preintegration_math.py \
  Scripts/UnitTest/test_drt_linear_alignment.py \
  Scripts/UnitTest/test_map_schema_compat.py \
  Scripts/UnitTest/test_reproj_imu_graph.py \
  Scripts/UnitTest/test_config_macvo.py \
  Scripts/UnitTest/test_config_modules.py \
  Scripts/UnitTest/test_config_sequence.py
```

Result in this branch: **42 passed**.

### What these tests cover
- `test_preintegration_math.py`
  - preintegration output shapes
  - zero-motion behavior
  - covariance PSD check
- `test_drt_linear_alignment.py`
  - synthetic linear alignment recovery
  - gravity refine consistency
- `test_reproj_imu_graph.py`
  - `2N + 15` residual dimension
  - IMU covariance block shape
- `test_map_schema_compat.py`
  - backward-compatible map deserialization for new fields
- config tests
  - odometry/module/sequence config loadability

---

## 10. Practical limitations and current status

1. `SlidingWindow_VIO_PGO` is implemented and usable, but this branch does **not** yet ship a full Schur marginalization module/API in production form.
2. `AirIMUCorrector` currently supports fallback identity correction if no compatible checkpoint is provided.
3. VIODE converter currently writes identity `T_BS` by default unless you inject calibrated extrinsics through conversion workflow.
4. This branch focuses on initialization/backend math and integration; large-scale model training scripts are intentionally not run as part of this implementation pass.

---

## 11. Troubleshooting

### Pytest fails due to project addopts / import hook
Use:

```bash
python3 -m pytest -o addopts='' ...
```

### Missing IMU edge error in `reproj_imu`
Ensure:
- sequence has IMU data,
- IMU edges are being pushed (`MACVO.run_pair`),
- graph type is set consistently with initializer/backend.

### DRT bootstrap keeps failing
Try:
- increasing `drt_window_len`,
- relaxing `accept_criteria.max_bg_norm`,
- increasing feature tracks (`track_stride` smaller),
- checking stereo/IMU timestamp quality.

---

## 12. Citation

If you use MAC-VO, please cite the original paper:

```bibtex
@inproceedings{qiu2025mac,
  title={MAC-VO: Metrics-Aware Covariance for Learning-Based Stereo Visual Odometry},
  author={Qiu, Yuheng and Chen, Yutian and Zhang, Zihao and Wang, Wenshan and Scherer, Sebastian},
  booktitle={2025 IEEE International Conference on Robotics and Automation (ICRA)},
  pages={3803--3814},
  year={2025},
  organization={IEEE}
}
```
