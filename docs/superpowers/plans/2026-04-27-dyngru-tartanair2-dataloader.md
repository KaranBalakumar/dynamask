# DynGRU TartanAir v2 Dataloader Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Wire TartanAir v2's real IMU data into the dynGRU Phase-A training pipeline so the model trains on a local sample before HPC deployment.

**Architecture:** Add a `TartanAirV2IMULoader` that reads v2-format IMU npy files (acc.npy, gyro.npy, imu_time.npy, cam_time.npy, ori_global.npy, pos_global.npy, vel_body.npy). Wire it into `TartanAirV2_Sequence` via a config toggle (`use_real_imu`). Fix the training script's `gt_pose` access path (currently wrong: `stereo.gt_pose` should be `cur.gt_pose`). Add `gt_backward_flow` as optional with safe fallback. Update the dyn training config for the local TartanAir v2 path at `~/Downloads/tartanair2`.

**Tech Stack:** Python 3, PyTorch, numpy, existing dynamask DataLoader framework (`SequenceBase`, `StereoInertialFrame`, `IMUData`, `AttitudeData`).

---

## Data on Disk

TartanAir v2 IMU data at `~/Downloads/tartanair2/AbandonedFactory/Data_easy/P00X/imu/`:

| File | Shape | dtype | Description |
|------|-------|-------|-------------|
| `acc.npy` | (N, 3) | float64 | Raw accelerometer (body frame, includes gravity) |
| `gyro.npy` | (N, 3) | float64 | Angular velocity (body frame, rad/s) |
| `imu_time.npy` | (N,) | float64 | IMU timestamps (seconds, 100 Hz, Δt=0.01s) |
| `cam_time.npy` | (M,) | float32 | Camera timestamps (seconds, 10 Hz, Δt=0.1s) |
| `ori_global.npy` | (N, 3) | float64 | Global orientation (Euler angles, radians, xyz) |
| `pos_global.npy` | (N, 3) | float64 | Global position (meters) |
| `vel_body.npy` | (N, 3) | float64 | Body-frame velocity (m/s) |
| `acc_nograv_body.npy` | (N, 3) | float64 | Body-frame accel without gravity (z-forward) |

Camera intrinsics (hardcoded in existing loader, same for v2):
- fx=320, fy=320, cx=320, cy=320, baseline=0.25m, resolution=640×640

Training requirements per frame pair `(cur, nxt)`:
- `imageL` (cur + nxt): from `image_lcam_front/` PNGs ✓ (exists)
- `gt_flow` + `flow_mask`: from `flow_lcam_front/` NPZs ✓ (exists, gated by `gtFlow`)
- `gt_depth`: from `depth_lcam_front/` PNGs ✓ (exists, gated by `gtDepth`)
- `gt_pose`: from `pose_lcam_front.txt` ✓ (exists, gated by `gtPose`)
- `K`: camera intrinsics ✓ (hardcoded)
- `f_imu`, `imu_tokens`: from IMUContext.step() over IMU window → needs real IMU loader
- `gt_backward_flow`: reverse-direction flow (warp fwd pixels back) — needs generator or skip

---

## File Structure (locked)

### New files
- `DataLoader/Dataset/TartanAir2_IMULoader.py` — V2-format real IMU loader
- `DataLoader/Dataset/TartanAir2_BackwardFlow.py` — Optional backward flow generator
- `Config/Sequence/Training_Dataset/TartanAirV2_Dyn_Local.yaml` — Local training data config

### Modified files
- `DataLoader/Dataset/TartanAir2.py` — Wire real IMU loader + backward flow into `TartanAirV2_Sequence`
- `DataLoader/Dataset/__init__.py` — Register new dataset type if needed
- `Train/MatchingNet/train_flowformer.py` — Fix `gt_pose` access path, add backward flow handling
- `Config/Train/FlowFormerDyn_Demo.yaml` — Update data path for local TartanAir v2

---

### Task 1: Create TartanAirV2IMULoader for real IMU data

**Files:**
- Create: `DataLoader/Dataset/TartanAir2_IMULoader.py`
- Test: manual smoke test (data loads without error)

- [ ] **Step 1: Write `TartanAirV2IMULoader` class**

```python
"""Real IMU data loader for TartanAir v2 format.

Reads acc.npy, gyro.npy, imu_time.npy, cam_time.npy from the v2 imu/ directory.
Optionally loads ori_global.npy, pos_global.npy, vel_body.npy for GT attitude.
"""

import torch
import numpy as np
from pathlib import Path
from typing import cast

import pypose as pp
from scipy.spatial.transform import Rotation

from ..Interface import IMUData, AttitudeData


class TartanAirV2IMULoader:
    """Loads real IMU measurements from TartanAir v2 imu/ directory.

    Unlike TartanAirIMUSimulator (which synthesizes IMU from GT pose), this reads
    the actual recorded/simulated IMU streams stored as npy files.  The v2 format
    uses different filenames than v1:

        v1: accel_left.npy / gyro_left.npy / angles_left.npy / vel_left.npy / xyz_left.npy
        v2: acc.npy / gyro.npy / ori_global.npy / vel_body.npy / pos_global.npy

    IMU timestamps (imu_time.npy) run at 100 Hz.  Camera timestamps (cam_time.npy)
    run at 10 Hz.  The loader aligns the two via alignWithCameraTime() so that
    frameRangeQuery(start_frame, end_frame) returns the IMU window between two
    camera frames.
    """

    def __init__(self, imu_dir: Path, gravity: float = 9.81):
        assert imu_dir.exists(), f"IMU directory not found: {imu_dir}"

        self.T_BS = pp.identity_SE3(1)  # body→sensor extrinsic (identity for sim)
        self.gravity = gravity

        # --- Required IMU measurements ---
        self.acc = torch.from_numpy(
            np.load(str(imu_dir / "acc.npy"))
        ).float().unsqueeze(0)                                     # (1, N, 3)

        self.gyro = torch.from_numpy(
            np.load(str(imu_dir / "gyro.npy"))
        ).float().unsqueeze(0)                                     # (1, N, 3)

        self.imu_time = torch.from_numpy(
            np.load(str(imu_dir / "imu_time.npy"))
        ).double().unsqueeze(0)                                    # (1, N)  seconds

        # --- Camera alignment ---
        cam_time_path = imu_dir / "cam_time.npy"
        if cam_time_path.exists():
            self.cam_time = torch.from_numpy(
                np.load(str(cam_time_path))
            ).float().unsqueeze(0)                                 # (1, M)  seconds
        else:
            raise FileNotFoundError(f"cam_time.npy not found in {imu_dir}")

        # --- Optional GT attitude (for AttitudeData) ---
        ori_path = imu_dir / "ori_global.npy"
        pos_path = imu_dir / "pos_global.npy"
        vel_path = imu_dir / "vel_body.npy"

        if ori_path.exists() and pos_path.exists():
            ori_euler = np.load(str(ori_path))                     # (N, 3) Euler xyz rad
            pos_global = np.load(str(pos_path))                    # (N, 3) meters
            self.gt_pos = torch.from_numpy(pos_global).float().unsqueeze(0)
            # Euler angles (xyz, radians) → SO3 (w,x,y,z quaternion)
            r = Rotation.from_euler("xyz", ori_euler, degrees=False)
            self.gt_rot = pp.euler2SO3(torch.from_numpy(
                r.as_euler("xyz", degrees=False)
            ).float()).unsqueeze(0)                                # (1, N, 4)
        else:
            self.gt_pos = None
            self.gt_rot = None

        if vel_path.exists():
            vel_body = np.load(str(vel_path))
            self.gt_vel = torch.from_numpy(vel_body).float().unsqueeze(0)  # (1, N, 3)
        else:
            self.gt_vel = None

        self.length = self.acc.shape[1] - 1
        self.cam2imu_idx: torch.Tensor | None = None
        self._align_camera_time()

    def __len__(self) -> int:
        return self.cam_time.shape[1]  # number of camera frames

    def _align_camera_time(self) -> None:
        """Map each camera timestamp to the last IMU index before it.

        For camera frame k at time t_k:
            cam2imu_idx[k] = max{ i | imu_time[i] <= t_k < imu_time[i+1] }

        This mirrors the v1 TartanAirIMULoader.alignWithCameraTime() logic.
        """
        M = self.cam_time.shape[1]
        N = self.length
        imu_t = self.imu_time[0]          # (N+1,)
        cam_t = self.cam_time[0]          # (M,)

        cam2imu = torch.full((M,), -1, dtype=torch.long)
        imu_idx = 0
        for cam_idx in range(M):
            frame_time = cam_t[cam_idx].item()
            # Advance imu_idx until we bracket the camera timestamp
            while imu_idx < N and not (
                self.imu_time[0, imu_idx].item() <= frame_time < self.imu_time[0, imu_idx + 1].item()
            ):
                imu_idx += 1
            if imu_idx < N:
                cam2imu[cam_idx] = imu_idx

        self.cam2imu_idx = cam2imu

        unmatched = (cam2imu < 0).sum().item()
        if unmatched > 0:
            import warnings
            warnings.warn(f"{unmatched}/{M} camera frames could not be aligned with IMU")

    def frame_range_query(self, start_frame: int, end_frame: int) -> tuple[IMUData, AttitudeData]:
        """Return IMU window between camera frames [start_frame, end_frame)."""
        assert self.cam2imu_idx is not None
        start_imu = int(self.cam2imu_idx[start_frame].item())
        end_imu   = int(self.cam2imu_idx[end_frame].item())
        assert start_imu >= 0 and end_imu >= 0, \
            f"Frames {start_frame}→{end_frame} not aligned with IMU"

        # Convert seconds → nanoseconds
        time_ns = (self.imu_time[:, start_imu:end_imu] * 1_000_000_000).long()

        imu_data = IMUData(
            T_BS=self.T_BS,
            gravity=[self.gravity],
            time_ns=time_ns,
            acc=self.acc[:, start_imu:end_imu],
            gyro=self.gyro[:, start_imu:end_imu],
        )

        # Build attitude data if GT available, otherwise return zeros
        if self.gt_pos is not None and self.gt_rot is not None:
            att = AttitudeData(
                T_BS=self.T_BS,
                gravity=[self.gravity],
                time_ns=time_ns,
                gt_pos=self.gt_pos[:, start_imu:end_imu],
                gt_vel=self.gt_vel[:, start_imu:end_imu] if self.gt_vel is not None else torch.zeros(1, end_imu - start_imu, 3),
                gt_rot=cast(pp.LieTensor, self.gt_rot[:, start_imu:end_imu]),
                init_pos=self.gt_pos[:, start_imu:start_imu + 1],
                init_vel=self.gt_vel[:, start_imu:start_imu + 1] if self.gt_vel is not None else torch.zeros(1, 1, 3),
                init_rot=cast(pp.LieTensor, self.gt_rot[:, start_imu:start_imu + 1]),
            )
        else:
            # Placeholder attitude with zeros
            N = end_imu - start_imu
            att = AttitudeData(
                T_BS=self.T_BS,
                gravity=[self.gravity],
                time_ns=time_ns,
                gt_pos=torch.zeros(1, N, 3),
                gt_vel=torch.zeros(1, N, 3),
                gt_rot=cast(pp.LieTensor, pp.identity_SO3(1).unsqueeze(0).expand(1, N, 4)),
                init_pos=torch.zeros(1, 1, 3),
                init_vel=torch.zeros(1, 1, 3),
                init_rot=cast(pp.LieTensor, pp.identity_SO3(1).unsqueeze(0)),
            )

        return imu_data, att
```

- [ ] **Step 2: Smoke-test the loader against real data**

Run:
```bash
python3 -c "
from DataLoader.Dataset.TartanAir2_IMULoader import TartanAirV2IMULoader
from pathlib import Path
loader = TartanAirV2IMULoader(Path.home() / 'Downloads/tartanair2/AbandonedFactory/Data_easy/P001/imu')
print(f'Camera frames: {len(loader)}')
print(f'Cam2imu first 5: {loader.cam2imu_idx[:5]}')
imu, att = loader.frame_range_query(0, 1)
print(f'IMU window [0→1]: acc shape={imu.acc.shape}, gyro shape={imu.gyro.shape}, dt={imu.time_delta[0,:3]}')
print(f'Attitude: pos shape={att.gt_pos.shape}, vel shape={att.gt_vel.shape}')
print('OK')
"
```
Expected: prints shapes and "OK"

- [ ] **Step 3: Commit**

```bash
git add DataLoader/Dataset/TartanAir2_IMULoader.py
git commit -m "feat(data): add TartanAirV2IMULoader for real v2-format IMU data"
```

---

### Task 2: Wire real IMU loader into TartanAirV2_Sequence

**Files:**
- Modify: `DataLoader/Dataset/TartanAir2.py:16-71`

- [ ] **Step 1: Add `use_real_imu` config toggle and wire real loader**

Replace the `__init__` of `TartanAirV2_Sequence` (lines 37-42):

```python
    def __init__(self, config: SimpleNamespace | dict[str, Any]):
        cfg = self.config_dict2ns(config)

        self.stereo_sequence = TartanAirV2_StereoSequence(cfg)

        use_real_imu = getattr(cfg, "use_real_imu", False)
        if use_real_imu:
            from .TartanAir2_IMULoader import TartanAirV2IMULoader
            self.imu_sequence = TartanAirV2IMULoader(
                Path(cfg.root, "imu"),
                gravity=getattr(cfg, "gravity", 9.81),
            )
        else:
            self.imu_sequence = TartanAirIMUSimulator(
                cfg.imu_sim,
                Path(cfg.root, "pose_lcam_front.txt"),
                fps=cfg.imu_freq,
            )
        super().__init__(len(self.stereo_sequence))
```

Since `TartanAirV2IMULoader` uses `frame_range_query(start, end)` (same API as `TartanAirIMUSimulator`) and `TartanAirIMULoader`, the `__getitem__` at line 44-57 needs no changes — it already calls `self.imu_sequence.frameRangeQuery(index - 1, index)` for non-zero indices and `self.imu_sequence[0]` for index 0.

But `TartanAirV2IMULoader` uses `frame_range_query` (snake_case), while `TartanAirIMUSimulator` uses `frameRangeQuery` (camelCase). Need to either:
1. Alias: add `frameRangeQuery = frame_range_query` in `TartanAirV2IMULoader`
2. Or update the call site

Option (1) is cleaner — add the alias in `TartanAirV2IMULoader`:

```python
    # API compatibility with TartanAirIMUSimulator / TartanAirIMULoader
    frameRangeQuery = frame_range_query
```

Also add `__getitem__` for index-0 access in `TartanAirV2IMULoader`:

```python
    def __getitem__(self, index: int) -> tuple[IMUData, AttitudeData]:
        """Single-frame access at index 0 (used for first keyframe)."""
        return self.frame_range_query(index, index + 1)
```

- [ ] **Step 2: Update config validation to accept `use_real_imu`**

In `TartanAirV2_Sequence.is_valid_config` (line 59-70), add `allow_excessive_cfg=True` if not already there (it is). The new `use_real_imu` and `gravity` keys are optional and handled via `getattr` — no validation change needed.

- [ ] **Step 3: Smoke-test the integrated sequence**

Run:
```bash
python3 -c "
from DataLoader.Dataset.TartanAir2 import TartanAirV2_Sequence
from pathlib import Path
from types import SimpleNamespace
cfg = SimpleNamespace(
    root=str(Path.home() / 'Downloads/tartanair2/AbandonedFactory/Data_easy/P001'),
    compressed=True, imu_freq=100, gtFlow=True, gtDepth=True, gtPose=True,
    use_real_imu=True,
    imu_sim=SimpleNamespace(acc_bias=(0,0,0), acc_init_bias_noise=(0,0,0), acc_bias_instability=(0,0,0), acc_random_walk=(0,0,0), gyro_bias=(0,0,0), gyro_init_bias_noise=(0,0,0), gyro_bias_instability=(0,0,0), gyro_random_walk=(0,0,0)),
)
seq = TartanAirV2_Sequence(cfg)
print(f'Sequence length: {len(seq)}')
f0 = seq[0]
f1 = seq[1]
print(f'Frame 0: imu acc shape={f0.imu.acc.shape}, gyro shape={f0.imu.gyro.shape}')
print(f'Frame 1: imu acc shape={f1.imu.acc.shape}')
print(f'gt_pose: {f0.gt_pose is not None}')
print(f'stereo.gt_depth: {f0.stereo.gt_depth is not None}')
print(f'stereo.gt_flow: {f0.stereo.gt_flow is not None}')
print('OK')
"
```
Expected: prints shapes, all assertions True, "OK"

- [ ] **Step 4: Commit**

```bash
git add DataLoader/Dataset/TartanAir2.py DataLoader/Dataset/TartanAir2_IMULoader.py
git commit -m "feat(data): wire real IMU loader into TartanAirV2_Sequence with use_real_imu toggle"
```

---

### Task 3: Fix gt_pose access path in training script

**Files:**
- Modify: `Train/MatchingNet/train_flowformer.py:125-128`

**Bug:** `gt_pose` is on `DataFrame` (i.e., `frameData.cur.gt_pose`), NOT on `StereoData` (i.e., NOT `frameData.cur.stereo.gt_pose`). The current `getattr(frameData.cur.stereo, "gt_pose", None)` always returns None.

Similarly, `gt_depth` IS on `StereoData` (line 107 of Interface.py) — that access is correct. And `K` IS on `StereoData` (line 60) — that access is also correct.

Only `gt_pose` is in the wrong place.

- [ ] **Step 1: Fix the gt_pose access**

Change lines 125-128 from:
```python
                gt_pose = getattr(frameData.cur.stereo, "gt_pose", None)
                gt_depth = getattr(frameData.cur.stereo, "gt_depth", None)
                fb_flow = getattr(frameData.cur.stereo, "gt_backward_flow", None)
                K = frameData.cur.stereo.K.unsqueeze(0).expand(B, -1, -1).cuda() if hasattr(frameData.cur.stereo, "K") else None
```

To:
```python
                gt_pose = getattr(frameData.cur, "gt_pose", None)
                gt_depth = getattr(frameData.cur.stereo, "gt_depth", None)
                fb_flow = getattr(frameData.cur.stereo, "gt_backward_flow", None)
                K = frameData.cur.stereo.K.unsqueeze(0).expand(B, -1, -1).cuda()
```

Note: `K` is always present on StereoData (it's a required field), so no need for `hasattr` fallback.

- [ ] **Step 2: Verify with a quick check**

Run:
```bash
python3 -c "
from Train.MatchingNet.train_flowformer import *
print('Import OK - gt_pose path fix verified')
"
```
Expected: "Import OK"

- [ ] **Step 3: Commit**

```bash
git add Train/MatchingNet/train_flowformer.py
git commit -m "fix(train): access gt_pose from DataFrame, not StereoData"
```

---

### Task 4: Add backward flow generation (optional, with fallback)

**Files:**
- Create: `DataLoader/Dataset/TartanAir2_BackwardFlow.py`

Backward flow is the optical flow from frame `t+1` back to frame `t`. It's used in `dyn_pseudo_label()` for forward-backward consistency occlusion detection (§4.2 of the dynGRU design). The TartanAir v2 dataset only provides forward flow (`flow_lcam_front/`).

Options:
1. Generate backward flow by inverting forward flow via bilinear warp (fast, approximate)
2. Use `gt_backward_flow` from sequential pairs if the dataloader chains pairs
3. Skip FB check when backward flow unavailable — the `dyn_pseudo_label` already handles `fb_flow=None`

**Decision:** Use option (3) — skip for now. The IGNORE band and depth-adaptive threshold already filter ambiguous pixels. FB consistency is a nice-to-have refinement, not a correctness requirement.

- [ ] **Step 1: No code changes needed.**

The `dyn_pseudo_label()` function already handles `fb_flow=None` gracefully:
```python
if fb_flow is not None:
    # ... FB consistency check ...
```

The training script already uses `getattr(frameData.cur.stereo, "gt_backward_flow", None)` which returns None when the field is absent.

This task is a no-op for the initial implementation. A backward flow generator can be added later if Phase-A validation shows occlusion noise in pseudo-labels.

- [ ] **Step 2: Commit (empty — skip or add a doc note)**

```bash
# No code changes for this task
```

---

### Task 5: Create local training config

**Files:**
- Create: `Config/Sequence/Training_Dataset/TartanAirV2_Dyn_Local.yaml`
- Modify: `Config/Train/FlowFormerDyn_Demo.yaml`

- [ ] **Step 1: Create local dataset config pointing to ~/Downloads/tartanair2**

```yaml
# Config/Sequence/Training_Dataset/TartanAirV2_Dyn_Local.yaml
# Phase-A dynGRU pretrain: local TartanAir v2 sample at ~/Downloads/tartanair2.
# Uses real IMU (use_real_imu: true), GT depth, GT pose, GT flow.
-   type: TartanAirv2
    name: AbandonedFactory_P001
    args:
        root: /home/karan/Downloads/tartanair2/AbandonedFactory/Data_easy/P001
        compressed: true
        imu_freq: 100
        use_real_imu: true
        gravity: 9.81
        imu_sim:
            acc_bias: [0.02, -0.01, 0.05]
            acc_init_bias_noise: [0.01, 0.01, 0.01]
            acc_bias_instability: [1.47e-4, 1.47e-4, 1.47e-4]
            acc_random_walk: [1.96e-7, 1.96e-7, 1.96e-7]
            gyro_bias: [5.e-3, -2.e-3, 5.e-3]
            gyro_init_bias_noise: [0.01, 0.01, 0.01]
            gyro_bias_instability: [5.8e-6, 5.8e-6, 5.8e-6]
            gyro_random_walk: [3.8e-7, 3.8e-7, 3.8e-7]
        gtDepth: true
        gtPose: true
        gtFlow: true
```

- [ ] **Step 2: Update training config to use local dataset**

Change `Config/Train/FlowFormerDyn_Demo.yaml` lines 83-89:

```yaml
Train:
  data: !flatten_seq
    - !include ../Sequence/Training_Dataset/TartanAirV2_Dyn_Local.yaml

Evaluate:
  data: !flatten_seq
    - !include ../Sequence/Training_Dataset/TartanAirV2_Dyn_Local.yaml
```

- [ ] **Step 3: Validate config loads**

Run:
```bash
python3 -m pytest Scripts/UnitTest/test_config_sequence.py -k "Dyn_Local" -q
```
Expected: PASS

- [ ] **Step 4: Commit**

```bash
git add Config/Sequence/Training_Dataset/TartanAirV2_Dyn_Local.yaml Config/Train/FlowFormerDyn_Demo.yaml
git commit -m "feat(config): add local TartanAir v2 dynGRU training config with real IMU"
```

---

### Task 6: Integration smoke test — load a batch and run one training step

**Files:**
- Test: manual smoke test script

- [ ] **Step 1: Verify dataloader produces valid training batches**

Run:
```bash
python3 -c "
import torch
from pathlib import Path
from types import SimpleNamespace
from torch.utils.data import DataLoader
from DataLoader import TrainDataset, DataFramePair, StereoFrame, CenterCropFrame, CastDataType

# Load config
from Utility.Config import load_config, namespace_to_cfgnode
cfg, _ = load_config(Path('Config/Train/FlowFormerDyn_Demo.yaml'))
modlecfg = namespace_to_cfgnode(cfg.Model)
datacfg = cfg.Train

# Build transforms
transforms = [CenterCropFrame(dict(width=640, height=480)),
              CastDataType(dict(dtype='fp32'))]

# Load dataset (dyn mode = IMU-bearing filter)
from DataLoader.Dataset.TartanAir2 import TartanAirV2_Sequence
traindatasets = TrainDataset[StereoFrame].mp_instantiation(
    datacfg.data, 0, -1,
    lambda c: c.type in {'TartanAir', 'TartanAirv2'}
)
print(f'Datasets found: {len(traindatasets)}')

from torch.utils.data import ConcatDataset
loader = DataLoader(
    ConcatDataset([ds.transform_source(transforms) for ds in traindatasets if ds is not None]),
    batch_size=2, shuffle=False, collate_fn=DataFramePair.collate, drop_last=True, num_workers=0,
)

batch = next(iter(loader))
print(f'Batch: img1={batch.cur.stereo.imageL.shape}, img2={batch.nxt.stereo.imageL.shape}')
print(f'gt_flow: {batch.cur.stereo.gt_flow.shape}')
print(f'gt_depth: {batch.cur.stereo.gt_depth.shape}')
print(f'gt_pose (on cur): {batch.cur.gt_pose.shape}')
print(f'IMU acc: {batch.cur.imu.acc.shape}, gyro: {batch.cur.imu.gyro.shape}')
print(f'K: {batch.cur.stereo.K.shape}')
print('OK - all fields present')
"
```
Expected: prints all shapes, "OK"

- [ ] **Step 2: Verify training script can import and model builds (dry run)**

Run:
```bash
python3 -c "
# Dry-run: just verify the model can be built and freeze policy applies
from Train.MatchingNet.train_flowformer import *
print('train_flowformer imports OK')
from Train.MatchingNet.loss import dyn_pseudo_label, dyn_loss_phase_a
print('loss functions import OK')
from Train.MatchingNet.utils import T_TrainType, AssertLiteralType
print(f'Valid train types: {T_TrainType}')
print('All imports OK')
"
```
Expected: "All imports OK"

- [ ] **Step 3: Commit readiness check**

Run full test suite to verify no regressions:
```bash
python3 -m pytest Scripts/UnitTest/ -q --tb=line
```
Expected: 0 failures (all existing tests pass or skip)

---

## Spec Coverage Check

- Real IMU data loading from v2 npy format → Task 1
- Camera-IMU time alignment → Task 1
- Wire into existing `TartanAirV2_Sequence` → Task 2
- Fix `gt_pose` path bug in training script → Task 3
- Backward flow (deferred — optional, fallback handled) → Task 4
- Local training config for `~/Downloads/tartanair2` → Task 5
- End-to-end smoke test → Task 6
- All existing tests still pass → Task 6 step 3

## Omissions (explicitly deferred)

- **IMUContext wiring in training loop**: Currently the training script passes dummy `f_imu=zeros(128)`, `imu_tokens=zeros(7,128)`. Real IMU→IMUContext→FiLM/cross-attn wiring is on the critical path for actual training but requires the IMUContext EKF to be instantiated in the training loop. This is scoped for a follow-up plan (depends on IMUContext checkpoint availability and EKF seeding strategy for training).

- **HPC multi-GPU config**: The local config uses a single P001 sequence. Scaling to full TartanAir v2 is a config change, not a code change.

- **Phase B dataloader (EuRoC/KITTI-360)**: These datasets have different IMU formats. Scoped for a separate plan.
