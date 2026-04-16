# DynaMask Workspace (MAC-VO + Static-Confidence Integration)

This repository is now centered on **`dynamask_vio/`**, a MAC-VO codebase with an integrated static-confidence path (FlowFormerCov + AirIMU + dynamic head), plus dataset support for **VIODE (rosbag)** and **TartanAir**.

The `references/` directory contains upstream reference code (AirIMU, AirIO, MAC-VO paper code) for comparison only.

---

## Repository layout

| Path | Purpose |
|---|---|
| `dynamask_vio/` | Main implementation (this is where you run everything) |
| `dynamask_vio/docs/` | Static-confidence spec/theory and audit notes |
| `graphify-out/` | Graphify artifacts (`graph.json`, report, chunk files) |
| `references/` | External reference implementations (do not modify for core work) |

---

## 5-minute setup

### 1) Create environment

From repository root:

```bash
micromamba env create -f environment.yml
micromamba activate dynamask
```

### 2) Install PyTorch backend (choose one)

```bash
# CUDA
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

# ROCm
pip install torch torchvision --index-url https://download.pytorch.org/whl/rocm6.2

# CPU (dev only)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

Or use the helper:

```bash
bash install.sh cuda   # or: rocm / cpu
```

### 3) Install Python deps

```bash
pip install -r requirements.txt
```

---

## Required model weights

Create a model directory under `dynamask_vio` and download checkpoints:

```bash
cd dynamask_vio
mkdir -p Model
wget -O Model/MACVO_FrontendCov.pth https://github.com/MAC-VO/MAC-VO/releases/download/model/MACVO_FrontendCov.pth
wget -O Model/MACVO_posenet.pkl https://github.com/MAC-VO/MAC-VO/releases/download/model/MACVO_posenet.pkl
```

If using static-confidence + IMU feature fusion, place AirIMU encoder weights at:

```text
dynamask_vio/Model/AirIMU_encoder.pth
```

---

## How to run

All commands below are executed from:

```bash
cd dynamask_vio
```

### 1) Run MAC-VO on a sequence

```bash
python MACVO.py \
  --odom Config/Experiment/MACVO/MACVO_Performant.yaml \
  --data Config/Sequence/TartanAir_example.yaml \
  --device-backend auto \
  --device-index 0
```

Backend override examples:

```bash
# Force CUDA
python MACVO.py --odom ... --data ... --device-backend cuda --device-index 0

# Force ROCm (mapped to torch cuda namespace internally)
python MACVO.py --odom ... --data ... --device-backend rocm --device-index 0

# CPU
python MACVO.py --odom ... --data ... --device-backend cpu
```

### 2) Run on VIODE (rosbag)

Edit `Config/Sequence/VIODE_example.yaml`:

```yaml
type: VIODE
args:
  root: /absolute/path/to/viode/sequence_0001
  bag: sequence.bag
  cam0_calib: cam0_pinhole.yaml
  cam1_calib: cam1_pinhole.yaml
  calib: calibration.yaml
```

Then run:

```bash
python MACVO.py \
  --odom Config/Experiment/MACVO/MACVO_Performant.yaml \
  --data Config/Sequence/VIODE_example.yaml \
  --device-backend auto
```

### 3) Train static-confidence head

Use one of:

- `Config/Experiment/StaticConfidenceHead/viode.yaml`
- `Config/Experiment/StaticConfidenceHead/tartanair.yaml`

Update dataset paths in that config, then:

```bash
python -m Train.DynamicHead.train \
  --config Config/Experiment/StaticConfidenceHead/viode.yaml \
  --seed 0
```

Checkpoint output defaults to:

```text
./Model/static_conf_head_last.pth
```

### 4) Enable static-confidence frontend in odometry runtime

In your odometry config, set:

- `Odometry.frontend.type: StaticConfidence_FlowFormerCovFrontend`
- `Odometry.frontend.args.imu` (AirIMU weights/config)
- `Odometry.frontend.args.dynamic_head` (head config/weights)
- `Odometry.args.dynamic_gating` (e.g., `enabled`, `min_static_conf`)

Use the dynamic-head checkpoint path in:

```yaml
Odometry:
  frontend:
    args:
      dynamic_head:
        weight: ./Model/static_conf_head_last.pth
```

---

## Evaluation and plotting

After a `MACVO.py` run, results are written under `Results/...` sandbox folders.

```bash
# Trajectory metrics (ATE/RTE/ROE/RPE)
python -m Evaluation.EvalSeq --spaces Results/<SPACE_DIR>

# Plot sequence trajectories/curves
python -m Evaluation.PlotSeq --spaces Results/<SPACE_DIR>
```

---

## Testing

From `dynamask_vio`:

```bash
# Broad non-heavy suite
micromamba run -n dynamask pytest -m "not local and not trt" -q

# Focused static-confidence tests
micromamba run -n dynamask python3 -m pytest -q \
  Scripts/UnitTest/test_dynamic_head_modules.py \
  Scripts/UnitTest/test_dynamic_head_training.py \
  Scripts/UnitTest/test_static_confidence_integration.py \
  Scripts/UnitTest/test_backend_imu_plumbing.py
```

---

## Notes

- Device strings `rocm`/`hip` are accepted and canonicalized to torch `cuda` device strings internally.
- VIODE loader expects rosbag topics: `/cam0/image_raw`, `/cam1/image_raw`, `/imu0`, `/odometry`.
- Static-confidence spec/theory docs are in:
  - `dynamask_vio/docs/2026-04-11-static-confidence-head-macvo-spec.md`
  - `dynamask_vio/docs/2026-04-11-static-confidence-head-method-theory.md`
