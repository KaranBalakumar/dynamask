"""Convert VIODE ROS bags to TartanAir v2 directory format.

Output structure per sequence:
  <out_dir>/<env>/<difficulty>/
    image_lcam_front/  000000.png ...
    image_rcam_front/  000000.png ...
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

def convert_bag(bag_path: Path, out_dir: Path):
    """Convert a single VIODE ROS bag to TartanAir v2 directory layout."""
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "image_lcam_front").mkdir(exist_ok=True)
    (out_dir / "image_rcam_front").mkdir(exist_ok=True)
    (out_dir / "imu").mkdir(exist_ok=True)

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
    print(f"  → {out_dir} ({size_gb:.1f} GB, {len(cam0_imgs)} imgs, {len(imu_gyro)} IMU)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--bag_dir', default=str(Path.home() / 'dynamask/dataset/viode'))
    parser.add_argument('--out_dir', default=str(Path.home() / 'dynamask/dataset/viode_tartan'))
    args = parser.parse_args()

    bag_dir = Path(args.bag_dir)
    out_root = Path(args.out_dir)

    bags = sorted(bag_dir.rglob('*.bag'))
    print(f"Found {len(bags)} bags")

    for bag_path in bags:
        # viode/city_day/0_none.bag → city_day/0_none
        rel = bag_path.relative_to(bag_dir)
        out_path = out_root / rel.parent.name / rel.stem
        if out_path.exists() and (out_path / "pose_lcam_front.txt").exists():
            print(f"  SKIP {rel} (already converted)")
            continue
        convert_bag(bag_path, out_path)

    print(f"\nDone. Data at {out_root}/")


if __name__ == '__main__':
    main()
