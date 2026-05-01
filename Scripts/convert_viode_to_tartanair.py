"""Convert VIODE ROS bags to TartanAir v2 directory format.

Output structure per sequence:
  <out_dir>/<env>/<difficulty>/
    image_lcam_front/  000000.png ...
    image_rcam_front/  000000.png ...
    depth_lcam_front/  000000.png ...   (if --with-depth)
    imu/
      acc.npy, gyro.npy, imu_time.npy, cam_time.npy
    pose_lcam_front.txt
"""
import argparse, cv2, numpy as np, sys
from pathlib import Path
from tqdm import tqdm
from scipy.spatial.transform import Rotation

try:
    from rosbags.highlevel import AnyReader
except ImportError:
    print("pip install rosbags opencv-python scipy tqdm")
    sys.exit(1)

# VIODE camera intrinsics (from calibration files)
VIODE_FX, VIODE_FY = 344.0, 344.0
VIODE_CX, VIODE_CY = 376.0, 240.0
VIODE_BASELINE = 0.12  # meters, cam0 ↔ cam1
VIODE_H, VIODE_W = 480, 752


def _init_stereo_matcher():
    """Return a tuned StereoSGBM matcher for VIODE rectified stereo pairs."""
    matcher = cv2.StereoSGBM_create(
        minDisparity=0,
        numDisparities=128,
        blockSize=7,
        P1=8 * 3 * 7 ** 2,
        P2=32 * 3 * 7 ** 2,
        disp12MaxDiff=1,
        uniquenessRatio=10,
        speckleWindowSize=100,
        speckleRange=2,
        preFilterCap=63,
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY,
    )
    return matcher


def compute_depth(left_img, right_img, matcher):
    """Stereo SGBM → metric depth [m] as float32 (H, W)."""
    disp = matcher.compute(left_img, right_img).astype(np.float32) / 16.0
    disp[disp < 0.5] = 0.0  # invalid / occluded
    depth = np.where(disp > 0, VIODE_FX * VIODE_BASELINE / disp, 0.0)
    return depth.astype(np.float32)


def depth_to_png(depth):
    """Pack float32 depth into TartanAir RGBA-float32 PNG format."""
    rgba = depth.view(np.uint8).reshape(depth.shape[0], depth.shape[1], 4)
    return rgba


def _add_depth_to_existing(out_dir: Path):
    """Add depth_lcam_front/ from already-saved left/right PNGs. Fast path."""
    left_dir = out_dir / "image_lcam_front"
    right_dir = out_dir / "image_rcam_front"
    depth_dir = out_dir / "depth_lcam_front"
    depth_dir.mkdir(exist_ok=True)

    left_files = sorted(left_dir.glob("*.png"))
    right_files = sorted(right_dir.glob("*.png"))
    assert len(left_files) == len(right_files), "Mismatched left/right image counts"

    matcher = _init_stereo_matcher()
    for lf, rf in tqdm(zip(left_files, right_files), total=len(left_files), desc="  Depth"):
        left = cv2.imread(str(lf), cv2.IMREAD_GRAYSCALE)
        right = cv2.imread(str(rf), cv2.IMREAD_GRAYSCALE)
        depth = compute_depth(left, right, matcher)
        cv2.imwrite(str(depth_dir / lf.name), depth_to_png(depth))

    print(f"  → added {len(left_files)} depth frames to {depth_dir}")


def convert_bag(bag_path: Path, out_dir: Path, with_depth: bool = False):
    """Convert a single VIODE ROS bag to TartanAir v2 directory layout."""
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "image_lcam_front").mkdir(exist_ok=True)
    (out_dir / "image_rcam_front").mkdir(exist_ok=True)
    (out_dir / "imu").mkdir(exist_ok=True)
    if with_depth:
        (out_dir / "depth_lcam_front").mkdir(exist_ok=True)

    # --- Pass 1: collect all messages ---
    cam0_times, cam1_times = [], []
    cam0_imgs, cam1_imgs = [], []
    imu_times, imu_gyro, imu_acc = [], [], []
    odom_times, odom_pos, odom_ori = [], [], []

    with AnyReader([bag_path]) as reader:
        for conn, ts, raw in tqdm(reader.messages(), desc=f"  Reading {bag_path.name}", unit="msg"):
            t_ns = ts
            if conn.topic == '/cam0/image_raw':
                msg = reader.deserialize(raw, conn.msgtype)
                img = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, 3)
                cam0_imgs.append(img)
                cam0_times.append(t_ns)
            elif conn.topic == '/cam1/image_raw':
                msg = reader.deserialize(raw, conn.msgtype)
                img = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, 3)
                cam1_imgs.append(img)
                cam1_times.append(t_ns)
            elif conn.topic == '/imu0':
                msg = reader.deserialize(raw, conn.msgtype)
                imu_times.append(t_ns)
                imu_gyro.append([msg.angular_velocity.x, msg.angular_velocity.y, msg.angular_velocity.z])
                imu_acc.append([msg.linear_acceleration.x, msg.linear_acceleration.y, msg.linear_acceleration.z])
            elif conn.topic == '/odometry':
                msg = reader.deserialize(raw, conn.msgtype)
                odom_times.append(t_ns)
                odom_pos.append([msg.pose.pose.position.x, msg.pose.pose.position.y, msg.pose.pose.position.z])
                odom_ori.append([msg.pose.pose.orientation.x, msg.pose.pose.orientation.y,
                                 msg.pose.pose.orientation.z, msg.pose.pose.orientation.w])

    # --- Write images ---
    for i, img in enumerate(tqdm(cam0_imgs, desc="  Writing cam0")):
        cv2.imwrite(str(out_dir / "image_lcam_front" / f"{i:06d}.png"), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    for i, img in enumerate(tqdm(cam1_imgs, desc="  Writing cam1")):
        cv2.imwrite(str(out_dir / "image_rcam_front" / f"{i:06d}.png"), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

    # --- Write depth (stereo SGBM) ---
    if with_depth:
        matcher = _init_stereo_matcher()
        for i in tqdm(range(len(cam0_imgs)), desc="  Writing depth"):
            left = cv2.cvtColor(cam0_imgs[i], cv2.COLOR_RGB2GRAY)
            right = cv2.cvtColor(cam1_imgs[i], cv2.COLOR_RGB2GRAY)
            depth = compute_depth(left, right, matcher)
            depth_rgba = depth_to_png(depth)
            cv2.imwrite(str(out_dir / "depth_lcam_front" / f"{i:06d}.png"), depth_rgba)

    # --- Write IMU ---
    imu_times_np = np.array(imu_times, dtype=np.float64)
    imu_times_sec = (imu_times_np - imu_times_np[0]) / 1e9  # relative seconds
    np.save(str(out_dir / "imu" / "acc.npy"), np.array(imu_acc, dtype=np.float64))
    np.save(str(out_dir / "imu" / "gyro.npy"), np.array(imu_gyro, dtype=np.float64))
    np.save(str(out_dir / "imu" / "imu_time.npy"), imu_times_sec)

    cam0_times_np = np.array(cam0_times, dtype=np.float64)
    cam_times_sec = (cam0_times_np - imu_times_np[0]) / 1e9  # relative to first IMU time
    np.save(str(out_dir / "imu" / "cam_time.npy"), cam_times_sec.astype(np.float32))

    # --- Write poses aligned to camera timestamps ---
    odom_times_np = np.array(odom_times, dtype=np.int64)
    odom_pos_np = np.array(odom_pos, dtype=np.float64)
    odom_ori_np = np.array(odom_ori, dtype=np.float64)

    poses = []
    for cam_t in cam0_times:
        idx = np.searchsorted(odom_times_np, cam_t)
        idx = min(idx, len(odom_times_np) - 1)
        pos = odom_pos_np[idx]
        ori = odom_ori_np[idx]  # xyzw quaternion
        # Write as: tx ty tz qx qy qz qw
        poses.append([pos[0], pos[1], pos[2], ori[0], ori[1], ori[2], ori[3]])

    poses_np = np.array(poses, dtype=np.float64)
    np.savetxt(str(out_dir / "pose_lcam_front.txt"), poses_np,
               fmt="%.6f", delimiter=" ")

    # --- Summary ---
    size_gb = sum(f.stat().st_size for f in out_dir.rglob("*") if f.is_file()) / 1e9
    extra = ", depth" if with_depth else ""
    print(f"  → {out_dir} ({size_gb:.1f} GB, {len(cam0_imgs)} imgs, {len(imu_gyro)} IMU{extra})")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--bag_dir', default=str(Path.home() / 'dynamask/dataset/viode'))
    parser.add_argument('--out_dir', default=str(Path.home() / 'dynamask/dataset/viode_tartan'))
    parser.add_argument('--with-depth', action='store_true',
                        help='Generate dense stereo depth via SGBM and save to depth_lcam_front/')
    args = parser.parse_args()

    bag_dir = Path(args.bag_dir)
    out_root = Path(args.out_dir)

    bags = sorted(bag_dir.rglob('*.bag'))
    print(f"Found {len(bags)} bags")

    for bag_path in bags:
        # viode/city_day/0_none.bag → city_day/0_none
        rel = bag_path.relative_to(bag_dir)
        out_path = out_root / rel.parent.name / rel.stem
        is_done = out_path.exists() and (out_path / "pose_lcam_front.txt").exists()
        if is_done:
            needs_depth = args.with_depth and not (out_path / "depth_lcam_front").exists()
            if not needs_depth:
                print(f"  SKIP {rel} (already converted)")
                continue
            print(f"  ADD-DEPTH {rel}")
            _add_depth_to_existing(out_path)
            continue
        convert_bag(bag_path, out_path, with_depth=args.with_depth)

    print(f"\nDone. Data at {out_root}/")


if __name__ == '__main__':
    main()
