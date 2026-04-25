"""
Quick debug script: reproduce the (9x0 @ 3x3) matmul error in DRTLooseBootstrap.solve()
by replaying a bootstrap with real EuRoC data and printing full traceback.
"""

import sys, os
sys.path.insert(0, ".")

import torch
import numpy as np
import traceback

EUROC_ROOT = os.path.expanduser(
    "~/Downloads/EuRoC/machine_hall/MH_01_easy/MH_01_easy/mav0"
)
AIRIMU_PKL = "Model/AirIMU_EuRoC/net_output.pickle"

T_BS_data = [
     0.0148655429818, -0.999880929698,  0.00414029679422, -0.0216401454975,
     0.999557249008,   0.0149672133247,  0.025715529948,  -0.064676986768,
    -0.0257744366974,  0.00375618835797, 0.999660727178,   0.00981073058949,
     0.0, 0.0, 0.0, 1.0
]
T_BS = torch.tensor(T_BS_data, dtype=torch.float64).view(4, 4)
R_BC = T_BS[:3, :3]
t_BC = T_BS[:3,  3]

N_KF = 10
TICKS_PER_FRAME = 10

print("Loading AirIMU data...")
import pickle
with open(AIRIMU_PKL, "rb") as f:
    airimu_pkl = pickle.load(f)
airimu_data = airimu_pkl["MH_01_easy"]
corr_acc  = airimu_data["corrected_acc"][0]
corr_gyro = airimu_data["corrected_gyro"][0]
dt_arr    = airimu_data["dt"][:, 0]

print("Loading EuRoC camera timestamps...")
cam_csv = os.path.join(EUROC_ROOT, "cam0", "data.csv")
cam_ts_ns = np.loadtxt(cam_csv, delimiter=",", skiprows=1, usecols=0)
cam_ts_s  = cam_ts_ns * 1e-9

from Module.Initialization.DRTLoose.bootstrap import DRTLooseBootstrap
from Module.Initialization.DRTLoose.types import DRTInitConfig
from Module.Initialization.DRTLoose.tracks import FeatureTrack

cfg = DRTInitConfig(
    min_keyframes=N_KF,
    max_attempts=2,
    window_scales=(1.0, 1.4, 1.8),
    fallback="heuristic",
)
bootstrap = DRTLooseBootstrap(cfg, R_BC=R_BC, t_BC=t_BC, gravity_norm=9.81007)

# Populate with real IMU data
tick_ptr = 0
for kf_i in range(N_KF):
    bootstrap._keyframe_timestamps.append(cam_ts_s[kf_i])
    if kf_i < N_KF - 1:
        segment = []
        for _ in range(TICKS_PER_FRAME):
            if tick_ptr >= len(corr_acc):
                break
            g = corr_gyro[tick_ptr]
            a = corr_acc[tick_ptr]
            dt = float(dt_arr[tick_ptr])
            segment.append((g, a, dt))
            tick_ptr += 1
        bootstrap._imu_segments.append(segment)
        print(f"  segment[{kf_i}]: {len(segment)} ticks")

# Populate with simulated feature tracks: pixels from a camera with
# fx=fy=458, cx=367, cy=248 (EuRoC cam0 approx) over N_KF keyframes
# We'll use static-ish points with small noise to simulate real tracks.
N_TRACKS = 50
torch.manual_seed(42)
# Base 3D points in front of camera (z=2m, scattered around optical axis)
pts_3d_cam0 = torch.randn(N_TRACKS, 3, dtype=torch.float64) * 0.5
pts_3d_cam0[:, 2] = 2.0 + torch.abs(pts_3d_cam0[:, 2])

fx, fy, cx, cy = 458.654, 457.296, 367.215, 248.375
tracks = []
for j in range(N_TRACKS):
    obs = {}
    pt = pts_3d_cam0[j]
    for k in range(N_KF):
        # Project with small random motion (no actual camera pose)
        x_noise = torch.randn((), dtype=torch.float64) * 0.05
        y_noise = torch.randn((), dtype=torch.float64) * 0.05
        u = fx * (pt[0] + x_noise * k * 0.01) / pt[2] + cx
        v = fy * (pt[1] + y_noise * k * 0.01) / pt[2] + cy
        obs[k] = torch.tensor([u.item(), v.item()], dtype=torch.float64)
    tracks.append(FeatureTrack(track_id=j, obs=obs))
bootstrap._tracks = tracks

print(f"\nBootstrap ready: {N_KF} KFs, {len(bootstrap._imu_segments)} segments, {len(bootstrap._tracks)} tracks")
print("Running solve() with full traceback...\n")

try:
    result = bootstrap.solve()
    print(f"solve() returned: success={result.success}, reason={result.failure_reason}")
except Exception:
    print("EXCEPTION in solve():")
    traceback.print_exc()

print("\n--- Now testing with actual pixel observations (not normalized) ---")
print("The _bearing() function treats pixel coords as normalized — checking if that causes issues...")
# Print a bearing vector to see what it looks like with pixel coords
from Module.Initialization.DRTLoose.tracks import _bearing
uv_pixel = torch.tensor([367.0, 248.0], dtype=torch.float64)  # optical axis
b = _bearing(uv_pixel)
print(f"  bearing([367, 248]) = {b.tolist()}")
uv_pixel2 = torch.tensor([0.5, 0.5], dtype=torch.float64)  # normalized
b2 = _bearing(uv_pixel2)
print(f"  bearing([0.5, 0.5]) = {b2.tolist()}")
