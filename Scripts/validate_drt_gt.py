"""
Validate DRT-loose init results against EuRoC state_groundtruth_estimate0.

Prints a side-by-side comparison of:
  - Gyro bias b_g   (DRT) vs b_w_RS_S  (GT, sensor frame)
  - Velocity v0     (DRT, world=kf0-cam frame) vs v_RS_R rotated into kf0-cam frame
  - Gravity g_W     (DRT) direction and magnitude
  - Scale           DRT scale vs GT inter-frame distance

Run from repo root:
    micromamba run -n dynamask python3 Scripts/validate_drt_gt.py
"""

import sys, os
sys.path.insert(0, ".")

import torch
import numpy as np
import pypose as pp

EUROC_ROOT = os.path.expanduser(
    "~/Downloads/EuRoC/machine_hall/MH_01_easy/MH_01_easy/mav0"
)
ODOM_CFG = "Scripts/UnitTest/assets/test_config/MACVO/MACVO.yaml"


def load_gt():
    gt_csv  = os.path.join(EUROC_ROOT, "state_groundtruth_estimate0", "data.csv")
    cam_csv = os.path.join(EUROC_ROOT, "cam0", "data.csv")
    gt_data = np.loadtxt(gt_csv,  delimiter=",", skiprows=1)
    gt_ts   = gt_data[:, 0].astype(np.int64)
    gt_pos  = gt_data[:, 1:4]
    gt_q    = gt_data[:, 4:8]    # qw, qx, qy, qz
    gt_vel  = gt_data[:, 8:11]
    gt_bg   = gt_data[:, 11:14]
    gt_ba   = gt_data[:, 14:17]
    cam_ts  = np.loadtxt(cam_csv, delimiter=",", skiprows=1, usecols=0).astype(np.int64)

    def interp(ts_ns):
        idx = np.clip(np.searchsorted(gt_ts, ts_ns), 1, len(gt_ts) - 1)
        t0, t1 = gt_ts[idx-1], gt_ts[idx]
        a = (ts_ns - t0) / (t1 - t0 + 1e-9)
        return {
            "pos": (1-a)*gt_pos[idx-1] + a*gt_pos[idx],
            "q":   (1-a)*gt_q[idx-1]   + a*gt_q[idx],
            "vel": (1-a)*gt_vel[idx-1] + a*gt_vel[idx],
            "bg":  (1-a)*gt_bg[idx-1]  + a*gt_bg[idx],
            "ba":  (1-a)*gt_ba[idx-1]  + a*gt_ba[idx],
        }

    kf0 = interp(cam_ts[0])
    kf9 = interp(cam_ts[9])
    return kf0, kf9, cam_ts


def print_gt(kf0, kf9):
    print(f"\n{'='*60}")
    print("  EUROC GT @ kf-0")
    print(f"{'='*60}")
    print(f"  velocity WF   : {kf0['vel'].tolist()}")
    print(f"  speed         : {np.linalg.norm(kf0['vel']):.4f} m/s")
    print(f"  gyro bias (S) : {kf0['bg'].tolist()} rad/s")
    print(f"  accel bias (S): {kf0['ba'].tolist()} m/s²")
    disp = np.linalg.norm(kf9['pos'] - kf0['pos'])
    print(f"  kf0→kf9 disp  : {disp:.4f} m")
    return disp


def compare(res, kf0, gt_disp_09):
    from Module.Initialization.DRTLoose.types import DRTInitResult
    print(f"\n{'='*60}")
    print("  DRT RESULT vs GT")
    print(f"{'='*60}")

    print(f"\n  [Gyro bias] (rad/s)")
    print(f"    DRT b_g      : {[f'{v:.6f}' for v in res.b_g.tolist()]}")
    print(f"    GT  b_w_RS_S : {[f'{v:.6f}' for v in kf0['bg'].tolist()]}")
    err = np.linalg.norm(res.b_g.numpy() - kf0['bg'])
    print(f"    Error norm   : {err:.6f} rad/s")

    print(f"\n  [Gravity vector]")
    print(f"    DRT g_W      : {[f'{v:.4f}' for v in res.g_W.tolist()]}  |g|={res.g_W.norm():.4f}")
    print(f"    Expected |g| : 9.8101 m/s²")
    print(f"    Mag error    : {abs(res.g_W.norm().item() - 9.81007):.4f} m/s²")

    q = kf0['q']
    so3 = pp.SO3(torch.tensor([q[1], q[2], q[3], q[0]], dtype=torch.float64))
    R_WC = so3.matrix()
    v_cam = R_WC.T @ torch.tensor(kf0['vel'], dtype=torch.float64)

    print(f"\n  [Velocity at kf-0] (m/s)")
    print(f"    DRT v0       : {[f'{v:.4f}' for v in res.v0.tolist()]}  |v|={res.v0.norm():.4f}")
    print(f"    GT vel (cam) : {[f'{v:.4f}' for v in v_cam.tolist()]}  |v|={v_cam.norm():.4f}")
    print(f"    Speed error  : {abs(res.v0.norm().item() - v_cam.norm().item()):.4f} m/s")

    print(f"\n  [Scale]")
    print(f"    DRT scale    : {res.scale:.6f}")
    print(f"    GT kf0→kf9   : {gt_disp_09:.4f} m  (reference only)")

    print(f"\n{'='*60}")
    print("DONE — check errors above.")
    print(f"{'='*60}\n")


def main():
    kf0, kf9, cam_ts = load_gt()
    gt_disp_09 = print_gt(kf0, kf9)

    # Write a temporary data config pointing to local EuRoC
    data_cfg_tmp = "/tmp/euroc_mh01_validate.yaml"
    with open(data_cfg_tmp, "w") as f:
        f.write(f"type: EuRoC\nname: MH01\nargs:\n    root: {EUROC_ROOT}\n    gt_pose: true\n")

    print(f"\n{'='*60}")
    print("  RUNNING MACVO DRT BOOTSTRAP (seq_to=15)")
    print(f"{'='*60}")

    captured = {}

    from Utility.Config import load_config, asNamespace
    from Utility.Sandbox import Sandbox
    from DataLoader import SequenceBase, StereoInertialFrame, smart_transform
    from Odometry.MACVO import MACVO
    from pathlib import Path

    orig_seed = MACVO._seed_from_drt

    def patched_seed(self, drt_result, frame):
        captured['result'] = drt_result
        orig_seed(self, drt_result, frame)

    MACVO._seed_from_drt = patched_seed

    cfg, cfg_dict = load_config(Path(ODOM_CFG))
    datacfg, _    = load_config(Path(data_cfg_tmp))
    odomcfg       = cfg.Odometry
    project_name  = odomcfg.name + "@" + datacfg.name

    exp_space = Sandbox.create(Path("./Results"), project_name)
    exp_space.set_autoremove()
    MAX_FRAMES = 120  # temporal subsampling needs ~44 frames before first attempt; allow retries
    exp_space.config = {
        "Project": project_name,
        "Odometry": cfg_dict["Odometry"],
        "Data": {"args": {"root": EUROC_ROOT, "gt_pose": True}, "end_idx": MAX_FRAMES, "start_idx": 0},
    }

    sequence = smart_transform(
        SequenceBase[StereoInertialFrame].instantiate(datacfg.type, datacfg.args).clip(0, MAX_FRAMES),
        cfg.Preprocess
    )
    system = MACVO[StereoInertialFrame].from_config(asNamespace(exp_space.config))

    for frame in sequence:
        system.run(frame)
        if captured:
            break

    try:
        system.Optimizer.terminate()
    except Exception:
        pass

    if not captured:
        print(f"  DRT init did not succeed in {MAX_FRAMES} frames.")
        sys.exit(1)

    compare(captured['result'], kf0, gt_disp_09)


if __name__ == '__main__':
    main()
