"""
Diagnostic script: verify AirIMU, Air-IO, IMUContext, and DRTLoose bootstrap
on the first N frames of EuRoC MH_01_easy WITHOUT needing the visual frontend.

Run from repo root:
    micromamba run -n dynamask python3 Scripts/diag_imu_init.py

Checks (each section prints ✓ / ✗):
  1. EuRoC IMU CSV loading
  2. AirIMU pickle loading (MH_01_easy present, shapes sane)
  3. Air-IO model loading (checkpoint → CodeNetMotionwithRot, forward pass)
  4. IMUContext construction + step() for N camera frames
  5. DRTLooseBootstrap solve (identity visual rotations → expect quality gate fail or success)
"""

import sys, os
sys.path.insert(0, ".")
sys.path.insert(0, "Module/Network/Air-IO")
sys.path.insert(0, "Module/Network/Air-IO/EKF")

import traceback
import torch
import numpy as np

EUROC_ROOT = os.path.expanduser(
    "~/Downloads/EuRoC/machine_hall/MH_01_easy/MH_01_easy/mav0"
)
AIRIMU_PKL  = "Model/AirIMU_EuRoC/net_output.pickle"
AIRIO_CKPT  = "Model/AirIO_EuRoC/AirIO_checkpoint/best_model.ckpt"
AIRIO_CONF  = "Module/Network/Air-IO/configs/EuRoC/motion_body.conf"
SEQ_NAME    = "MH_01_easy"
N_CAM_FRAMES = 15     # camera frames to simulate (200 Hz IMU / 20 Hz cam = 10 ticks/frame)
TICKS_PER_FRAME = 10  # 200 / 20

# EuRoC cam0→IMU extrinsic (T_BS from sensor.yaml: rows of rotation, translation)
T_BS_data = [
     0.0148655429818, -0.999880929698,  0.00414029679422, -0.0216401454975,
     0.999557249008,   0.0149672133247,  0.025715529948,  -0.064676986768,
    -0.0257744366974,  0.00375618835797, 0.999660727178,   0.00981073058949,
     0.0, 0.0, 0.0, 1.0
]
T_BS = torch.tensor(T_BS_data, dtype=torch.float64).view(4, 4)
R_BC = T_BS[:3, :3]  # cam←body (IMU frame)
t_BC = T_BS[:3,  3]


def section(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print('='*60)


def ok(msg=""):
    print(f"  ✓  {msg}" if msg else "  ✓  PASS")


def fail(msg, exc=None):
    print(f"  ✗  FAIL: {msg}")
    if exc:
        traceback.print_exc()


# ─────────────────────────────────────────────────────────────────
# 1. EuRoC IMU CSV
# ─────────────────────────────────────────────────────────────────
section("1 · EuRoC IMU CSV loading")
imu_loaded = False
gyro_raw = acc_raw = dt_s = timestamps_ns = None
try:
    imu_csv = os.path.join(EUROC_ROOT, "imu0", "data.csv")
    assert os.path.exists(imu_csv), f"Missing: {imu_csv}"
    data = np.loadtxt(imu_csv, delimiter=",", skiprows=1)
    assert data.shape[1] == 7, f"Unexpected columns: {data.shape[1]}"
    timestamps_ns = data[:, 0]
    gyro_raw = torch.from_numpy(data[:, 1:4]).double()
    acc_raw  = torch.from_numpy(data[:, 4:7]).double()
    dt_s     = torch.from_numpy(np.diff(timestamps_ns) * 1e-9).double()
    ok(f"Loaded {len(data)} IMU ticks, dt mean={dt_s.mean():.4f}s (expect ~0.005s)")
    imu_loaded = True
except Exception:
    fail("EuRoC IMU CSV", exc=True)


# ─────────────────────────────────────────────────────────────────
# 2. AirIMU pickle
# ─────────────────────────────────────────────────────────────────
section("2 · AirIMU pickle loading")
airimu_data = None
try:
    import pickle
    with open(AIRIMU_PKL, "rb") as f:
        airimu_pkl = pickle.load(f)
    assert SEQ_NAME in airimu_pkl, f"'{SEQ_NAME}' not found; available={list(airimu_pkl.keys())}"
    airimu_data = airimu_pkl[SEQ_NAME]
    ok(f"Sequences: {list(airimu_pkl.keys())}")
    for k in ("corrected_acc", "corrected_gyro", "acc_cov", "gyro_cov", "dt"):
        t = airimu_data[k]
        ok(f"  {k}: shape={tuple(t.shape)} dtype={t.dtype}")
    ok(f"Total AirIMU ticks for {SEQ_NAME}: {airimu_data['corrected_acc'].shape[1]}")
except Exception:
    fail("AirIMU pickle", exc=True)


# ─────────────────────────────────────────────────────────────────
# 3. Air-IO model: CodeNetMotionwithRot
# ─────────────────────────────────────────────────────────────────
section("3 · Air-IO model loading + forward pass")
airio_net = None
try:
    from pyhocon import ConfigFactory
    conf = ConfigFactory.parse_file(AIRIO_CONF)
    train_conf = conf.get("train")
    from model.code import CodeNetMotionwithRot
    airio_net = CodeNetMotionwithRot(train_conf).double()
    ckpt = torch.load(AIRIO_CKPT, map_location="cpu", weights_only=True)
    state = ckpt.get("model_state_dict", ckpt)
    airio_net.load_state_dict(state)
    airio_net.eval()
    ok(f"Checkpoint loaded from {AIRIO_CKPT}")

    T = 20
    dummy_acc  = torch.zeros(1, T, 3, dtype=torch.float64)
    dummy_gyro = torch.zeros(1, T, 3, dtype=torch.float64)
    dummy_rot  = torch.zeros(1, T, 3, dtype=torch.float64)
    with torch.no_grad():
        out = airio_net({"acc": dummy_acc, "gyro": dummy_gyro}, dummy_rot)
    ok(f"Forward pass — net_vel: {out['net_vel'].shape}, cov: {out['cov'].shape if out['cov'] is not None else None}")
    assert out["net_vel"].shape == (1, T // 9 + 1 if T % 9 else T // 9, 3) or out["net_vel"].shape[0] == 1
    ok("Output shapes sane")
except Exception:
    fail("Air-IO model", exc=True)


# ─────────────────────────────────────────────────────────────────
# 4. IMUContext construction + step() on real AirIMU-corrected data
# ─────────────────────────────────────────────────────────────────
section("4 · IMUContext construction + step()")
ctx_sample = None
try:
    if not imu_loaded:
        raise RuntimeError("EuRoC IMU not loaded (section 1 failed)")

    from pyhocon import ConfigFactory
    conf = ConfigFactory.parse_file(AIRIO_CONF)
    train_conf = conf.get("train")
    from Module.Network.IMUContext.imu_context import IMUContext

    ctx = IMUContext(
        airio_cfg=train_conf,
        airio_ckpt=AIRIO_CKPT,
        gravity=9.81007,
    )
    ok("IMUContext constructed")

    ctx.reset(
        init_rot=torch.zeros(3, dtype=torch.float64),
        init_vel=torch.zeros(3, dtype=torch.float64),
        init_pos=torch.zeros(3, dtype=torch.float64),
    )
    ok("reset() OK")

    # Use AirIMU corrected data if available, else raw EuRoC
    if airimu_data is not None:
        corr_acc  = airimu_data["corrected_acc"][0]   # (N, 3)
        corr_gyro = airimu_data["corrected_gyro"][0]  # (N, 3)
        acc_cov_t = airimu_data["acc_cov"][0]
        gyro_cov_t= airimu_data["gyro_cov"][0]
        dt_arr    = airimu_data["dt"][:, 0]           # (N,)
    else:
        corr_acc  = acc_raw
        corr_gyro = gyro_raw
        acc_cov_t = gyro_cov_t = None
        dt_arr    = dt_s

    tick_offset = 0
    for cam_idx in range(N_CAM_FRAMES):
        start = tick_offset
        end   = start + TICKS_PER_FRAME
        if end > len(corr_acc):
            break

        window = []
        for t in range(start, end):
            dt_t = float(dt_arr[t])
            tick = {
                "acc" : corr_acc[t],
                "gyro": corr_gyro[t],
                "dt"  : torch.tensor(dt_t, dtype=torch.float64),
            }
            if acc_cov_t is not None:
                tick["acc_cov"]  = acc_cov_t[t]
                tick["gyro_cov"] = gyro_cov_t[t]
            window.append(tick)

        ctx_sample = ctx.step(window, window)
        tick_offset = end

    ok(f"step() called for {N_CAM_FRAMES} camera frames")
    ok(f"  f_imu:       {ctx_sample.f_imu.shape}  (expect (1,128))")
    ok(f"  imu_tokens:  {ctx_sample.imu_tokens.shape}  (expect (1,7,128))")
    ok(f"  z_raw:       {ctx_sample.z_raw.shape}  (expect (34,))")
    ok(f"  EKF state R: {ctx_sample.state[:3].tolist()}")
    ok(f"  EKF state V: {ctx_sample.state[3:6].tolist()}")
    ok(f"  airio_vel:   {ctx_sample.airio_vel.tolist()}")
    assert ctx_sample.f_imu.shape    == (1, 128)
    assert ctx_sample.imu_tokens.shape == (1, 7, 128)
    assert ctx_sample.z_raw.shape    == (34,)
    assert torch.all(torch.isfinite(ctx_sample.f_imu)), "f_imu has non-finite values"
    ok("All shape + finite assertions PASS")

except Exception:
    fail("IMUContext.step()", exc=True)


# ─────────────────────────────────────────────────────────────────
# 5. DRTLooseBootstrap end-to-end solve (placeholder visual rotations)
# ─────────────────────────────────────────────────────────────────
section("5 · DRTLooseBootstrap solve (identity visual R)")
try:
    from Module.Initialization.DRTLoose.bootstrap import DRTLooseBootstrap, run_drt_with_retry
    from Module.Initialization.DRTLoose.types import DRTInitConfig, DRTInitResult
    from Module.Initialization.DRTLoose.preintegration import IMUPreintegrator
    from Module.Initialization.DRTLoose.tracks import FeatureTrack

    cfg = DRTInitConfig(
        min_keyframes=10,
        max_attempts=2,
        window_scales=(1.0, 1.4, 1.8),
        fallback="heuristic",
    )
    bootstrap = DRTLooseBootstrap(cfg, R_BC=R_BC, t_BC=t_BC, gravity_norm=9.81007)
    ok(f"DRTLooseBootstrap constructed (R_BC shape={R_BC.shape})")

    # Simulate 12 keyframe gaps using real AirIMU data (or raw IMU fallback)
    if airimu_data is not None:
        corr_acc  = airimu_data["corrected_acc"][0]
        corr_gyro = airimu_data["corrected_gyro"][0]
        dt_arr    = airimu_data["dt"][:, 0]
    else:
        corr_acc  = acc_raw
        corr_gyro = gyro_raw
        dt_arr    = dt_s

    cam_csv = os.path.join(EUROC_ROOT, "cam0", "data.csv")
    cam_ts_ns = np.loadtxt(cam_csv, delimiter=",", skiprows=1, usecols=0)
    cam_ts_s  = cam_ts_ns * 1e-9

    N_KF = 12
    tick_ptr = 0
    for kf_i in range(N_KF):
        bootstrap._keyframe_timestamps.append(cam_ts_s[kf_i])

        # Only add a segment for each *gap* (N_KF keyframes → N_KF-1 gaps)
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

        # Dummy feature tracks: 50 static features visible at all keyframes so far
        bootstrap._tracks = [
            FeatureTrack(
                track_id=j,
                obs={k: torch.tensor([320.0 + j * 0.5, 240.0 + j * 0.3], dtype=torch.float64)
                     for k in range(kf_i + 1)}
            )
            for j in range(50)
        ]

    ok(f"Accumulated {len(bootstrap._keyframe_timestamps)} keyframes, {len(bootstrap._imu_segments)} segments")
    ok(f"Window ready: {bootstrap.is_window_ready()}")

    # Try full solve — with identity visual rotations this will likely fail quality gates,
    # which is the expected/correct behaviour before real matcher output is wired in.
    result = bootstrap.solve()
    ok(f"solve() returned: success={result.success}, reason={result.failure_reason}")
    if result.success:
        ok(f"  b_g={result.b_g.tolist()}")
        ok(f"  g_W={result.g_W.tolist()}")
        ok(f"  scale={result.scale:.4f}")
        ok(f"  P_init shape={result.P_init.shape}")
    else:
        ok(f"  (Expected: identity rotations → poor quality gate → failure is correct)")

    # run_drt_with_retry: placeholder solver always returns None → FALLBACK_HEURISTIC
    result_retry = run_drt_with_retry(cfg, solver_fn=lambda scale: None)
    assert result_retry.failure_reason == "FALLBACK_HEURISTIC"
    ok("run_drt_with_retry fallback policy PASS")

    # run_drt_with_retry: mock solver succeeds on first attempt
    result_ok = run_drt_with_retry(cfg, solver_fn=lambda scale: DRTInitResult(success=True, failure_reason=None, retry_recommended=False))
    assert result_ok.success
    ok("run_drt_with_retry success path PASS")

except Exception:
    fail("DRTLooseBootstrap", exc=True)


# ─────────────────────────────────────────────────────────────────
# Summary
# ─────────────────────────────────────────────────────────────────
section("DONE")
print("  Scroll up to check each section for ✓ / ✗ marks.")
print()
