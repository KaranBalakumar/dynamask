"""Convert VIODE ROS bags to HDF5 for efficient random-access training.

Usage: python3 Scripts/convert_viode.py [--bag_dir ~/dynamask/dataset/viode] [--out_dir ~/dynamask/dataset/viode_h5]
"""
import argparse, h5py, numpy as np, sys
from pathlib import Path
from tqdm import tqdm

try:
    from rosbags.highlevel import AnyReader
except ImportError:
    print("pip install rosbags")
    sys.exit(1)

# AirSim / VIODE camera intrinsics (from VINS-Fusion format calibration)
# These are obtained from the VIODE GitHub calibration files
CAM_INTRINSICS = {
    "width": 752, "height": 480,
    "fx": 344.0, "fy": 344.0, "cx": 376.0, "cy": 240.0,  # approximate — will refine from calibration
    "baseline": 0.12,  # meters (T_cam0_cam1 from calibration)
}

def convert_bag(bag_path: Path, out_path: Path):
    """Convert a single VIODE ROS bag to HDF5."""
    print(f"\nConverting {bag_path.name} → {out_path.name}")

    # Storage buffers
    cam0_imgs, cam1_imgs = [], []
    cam0_times, cam1_times = [], []
    imu_times, imu_gyro, imu_acc = [], [], []
    odom_times, odom_pos, odom_ori, odom_vel = [], [], [], []
    seg0_imgs = []  # GT semantic segmentation (left)

    with AnyReader([bag_path]) as reader:
        for conn, ts, raw in tqdm(reader.messages(), desc="Reading", unit="msg"):
            t_ns = ts  # nanosecond timestamp

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

            elif conn.topic == '/cam0/segmentation':
                msg = reader.deserialize(raw, conn.msgtype)
                seg = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, 3)
                seg0_imgs.append(seg)

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
                odom_vel.append([msg.twist.twist.linear.x, msg.twist.twist.linear.y, msg.twist.twist.linear.z])

    # Convert to numpy arrays
    cam0_imgs = np.stack(cam0_imgs)       # (M, H, W, 3) uint8
    cam1_imgs = np.stack(cam1_imgs)
    cam0_times = np.array(cam0_times, dtype=np.int64)
    cam1_times = np.array(cam1_times, dtype=np.int64)
    seg0_imgs = np.stack(seg0_imgs) if seg0_imgs else np.zeros((len(cam0_imgs), 480, 752, 3), dtype=np.uint8)
    imu_times = np.array(imu_times, dtype=np.int64)
    imu_gyro = np.array(imu_gyro, dtype=np.float32)
    imu_acc = np.array(imu_acc, dtype=np.float32)
    odom_times = np.array(odom_times, dtype=np.int64)
    odom_pos = np.array(odom_pos, dtype=np.float64)
    odom_ori = np.array(odom_ori, dtype=np.float64)
    odom_vel = np.array(odom_vel, dtype=np.float64)

    print(f"  cam0: {len(cam0_imgs)} frames, cam1: {len(cam1_imgs)} frames")
    print(f"  IMU: {len(imu_gyro)} samples, Odometry: {len(odom_pos)} samples")
    print(f"  Segmentation: {len(seg0_imgs)} frames")
    print(f"  Duration: {(cam0_times[-1] - cam0_times[0]) / 1e9:.1f}s")

    # --- Write HDF5 ---
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(out_path, 'w') as f:
        # Images (compressed for storage efficiency)
        f.create_dataset('cam0/images', data=cam0_imgs, chunks=(1, 480, 752, 3), compression='lzf')
        f.create_dataset('cam1/images', data=cam1_imgs, chunks=(1, 480, 752, 3), compression='lzf')
        f.create_dataset('cam0/times', data=cam0_times)
        f.create_dataset('cam1/times', data=cam1_times)

        # GT segmentation
        f.create_dataset('cam0/segmentation', data=seg0_imgs, chunks=(1, 480, 752, 3), compression='lzf')

        # IMU
        f.create_dataset('imu/times', data=imu_times)
        f.create_dataset('imu/gyro', data=imu_gyro)
        f.create_dataset('imu/acc', data=imu_acc)

        # Odometry
        f.create_dataset('odometry/times', data=odom_times)
        f.create_dataset('odometry/position', data=odom_pos)
        f.create_dataset('odometry/orientation', data=odom_ori)  # xyzw quaternion
        f.create_dataset('odometry/velocity', data=odom_vel)     # linear velocity

        # Camera intrinsics
        f.attrs['width'] = CAM_INTRINSICS['width']
        f.attrs['height'] = CAM_INTRINSICS['height']
        f.attrs['fx'] = CAM_INTRINSICS['fx']
        f.attrs['fy'] = CAM_INTRINSICS['fy']
        f.attrs['cx'] = CAM_INTRINSICS['cx']
        f.attrs['cy'] = CAM_INTRINSICS['cy']
        f.attrs['baseline'] = CAM_INTRINSICS['baseline']

        # Align camera ↔ IMU indices for efficient frameRangeQuery
        cam2imu = np.searchsorted(imu_times, cam0_times) - 1
        cam2imu = np.clip(cam2imu, 0, len(imu_times) - 2)
        f.create_dataset('align/cam2imu_idx', data=cam2imu)

    size_mb = out_path.stat().st_size / 1e6
    print(f"  → {out_path} ({size_mb:.0f} MB)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--bag_dir', type=str, default=str(Path.home() / 'dynamask/dataset/viode'))
    parser.add_argument('--out_dir', type=str, default=str(Path.home() / 'dynamask/dataset/viode_h5'))
    args = parser.parse_args()

    bag_dir = Path(args.bag_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    bags = sorted(bag_dir.rglob('*.bag'))
    print(f"Found {len(bags)} bag files:")
    for b in bags:
        print(f"  {b.relative_to(bag_dir)}")

    for bag_path in bags:
        rel = bag_path.relative_to(bag_dir)
        out_path = out_dir / rel.with_suffix('.h5')
        if out_path.exists():
            print(f"  SKIP {out_path.name} (already exists)")
            continue
        convert_bag(bag_path, out_path)


if __name__ == '__main__':
    main()
