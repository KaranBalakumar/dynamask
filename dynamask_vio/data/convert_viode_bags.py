#!/usr/bin/env python3
"""Convert VIODE ROS bags to HDF5 for memory-efficient training.

Reads each .bag file once, extracts images / segmentation / IMU / GT poses,
optionally resizes images, and writes a self-contained .h5 file.  During
training the dataset loader reads one frame at a time from HDF5, so RAM usage
stays near zero regardless of dataset size.

Usage (from the project root):
    micromamba run -n dynamask python -m dynamask_vio.data.convert_viode_bags \\
        --viode_root ./dataset/viode \\
        --out_root   ./dataset/viode_hdf5 \\
        [--resize_h 480 --resize_w 640] \\
        [--gzip_level 1]

The output directory mirrors the input tree:
    ./dataset/viode_hdf5/city_day/0_none.h5
    ./dataset/viode_hdf5/city_day/1_low.h5
    ...
"""

import argparse
import os
import sys

import cv2
import h5py
import numpy as np


# ──────────────────────────────────────────────────────────────────────────────
# ROS-bag reader (same logic as viode_dataset._load_rosbag but stream-writes
# to HDF5 instead of collecting everything in RAM).
# ──────────────────────────────────────────────────────────────────────────────

GT_ODOM_TOPICS = {
    "/odometry", "/groundtruth/odometry", "/vio/odom",
    "/ground_truth/odometry", "/odom", "/groundtruth/odom",
}


def _get_ros1_decoder():
    try:
        from rosbags.serde import deserialize_cdr, ros1_to_cdr

        def decode(raw, msgtype):
            return deserialize_cdr(ros1_to_cdr(raw, msgtype), msgtype)
    except ImportError:
        from rosbags.typesys import Stores, get_typestore
        ts = get_typestore(Stores.ROS1_NOETIC)

        def decode(raw, msgtype):
            return ts.deserialize_ros1(raw, msgtype)
    return decode


def convert_bag(bag_path: str, out_path: str,
                resize_h: int | None, resize_w: int | None,
                gzip_level: int = 1):
    """Convert a single .bag to .h5.  Streams data — peak RAM is one frame."""
    from rosbags.rosbag1 import Reader

    decode = _get_ros1_decoder()

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    # ── First pass: collect IMU, GT, image/seg timestamps & raw sizes ──
    print(f"  Scanning {bag_path} …")

    imu_timestamps, imu_data = [], []
    gt_poses = []
    img_meta = []    # (t_sec, h, w, encoding)
    seg_meta = []    # (t_sec, h, w, encoding, ndim)  ndim=2 or 3

    with Reader(bag_path) as reader:
        conns_by_topic = {}
        for conn in reader.connections:
            conns_by_topic.setdefault(conn.topic, []).append(conn)

        avail_gt = GT_ODOM_TOPICS & set(conns_by_topic.keys())

        for conn, timestamp, rawdata in reader.messages():
            t_sec = timestamp / 1e9

            if conn.topic == "/cam0/image_raw":
                msg = decode(rawdata, conn.msgtype)
                img_meta.append((t_sec, msg.height, msg.width, msg.encoding))

            elif conn.topic == "/imu0":
                msg = decode(rawdata, conn.msgtype)
                imu_timestamps.append(t_sec)
                imu_data.append([
                    msg.linear_acceleration.x,
                    msg.linear_acceleration.y,
                    msg.linear_acceleration.z,
                    msg.angular_velocity.x,
                    msg.angular_velocity.y,
                    msg.angular_velocity.z,
                ])

            elif conn.topic == "/cam0/segmentation":
                msg = decode(rawdata, conn.msgtype)
                data = np.frombuffer(msg.data, dtype=np.uint8)
                enc = msg.encoding
                if enc in ("rgb8", "8UC3", "bgr8"):
                    ndim = 3
                else:
                    ndim = 2
                seg_meta.append((t_sec, msg.height, msg.width, enc, ndim))

            elif conn.topic in avail_gt:
                msg = decode(rawdata, conn.msgtype)
                try:
                    pos = msg.pose.pose.position
                    ori = msg.pose.pose.orientation
                    gt_poses.append((t_sec, pos.x, pos.y, pos.z,
                                     ori.x, ori.y, ori.z, ori.w))
                except AttributeError:
                    pass

    img_meta.sort(key=lambda x: x[0])
    seg_meta.sort(key=lambda x: x[0])

    n_img = len(img_meta)
    n_seg = len(seg_meta)
    if n_img == 0:
        print(f"  WARNING: no images found in {bag_path}, skipping")
        return

    # Determine output image shape
    raw_h, raw_w = img_meta[0][1], img_meta[0][2]
    out_h = resize_h if resize_h else raw_h
    out_w = resize_w if resize_w else raw_w

    # Determine seg shape
    if n_seg > 0:
        seg_raw_h, seg_raw_w = seg_meta[0][1], seg_meta[0][2]
        seg_ndim = seg_meta[0][4]
        seg_enc = seg_meta[0][3]
        seg_out_h = resize_h if resize_h else seg_raw_h
        seg_out_w = resize_w if resize_w else seg_raw_w
    else:
        seg_enc = ""

    print(f"  {n_img} images  |  {n_seg} seg frames  |  "
          f"{len(imu_timestamps)} IMU samples  |  {len(gt_poses)} GT poses")
    print(f"  Image size: {raw_h}×{raw_w}  →  {out_h}×{out_w}")

    # ── Second pass: read & write frame data ──────────────────────────────
    compress = dict(compression="gzip", compression_opts=gzip_level)

    with h5py.File(out_path, "w") as hf:
        # Pre-allocate datasets
        ds_images = hf.create_dataset(
            "images", shape=(n_img, out_h, out_w, 3), dtype=np.uint8,
            chunks=(1, out_h, out_w, 3), **compress)
        ds_img_ts = hf.create_dataset(
            "image_timestamps", shape=(n_img,), dtype=np.float64)

        if n_seg > 0:
            if seg_ndim == 3:
                ds_seg = hf.create_dataset(
                    "seg", shape=(n_seg, seg_out_h, seg_out_w, 3), dtype=np.uint8,
                    chunks=(1, seg_out_h, seg_out_w, 3), **compress)
            else:
                ds_seg = hf.create_dataset(
                    "seg", shape=(n_seg, seg_out_h, seg_out_w), dtype=np.uint8,
                    chunks=(1, seg_out_h, seg_out_w), **compress)
            ds_seg_ts = hf.create_dataset(
                "seg_timestamps", shape=(n_seg,), dtype=np.float64)
            hf.attrs["seg_encoding"] = seg_enc

        # IMU & GT (small — write immediately)
        if len(imu_timestamps) > 0:
            imu_ts_arr = np.array(imu_timestamps, dtype=np.float64)
            imu_data_arr = np.array(imu_data, dtype=np.float64)
            hf.create_dataset("imu_timestamps", data=imu_ts_arr)
            hf.create_dataset("imu_data", data=imu_data_arr)

        if len(gt_poses) > 1:
            gt_poses.sort(key=lambda x: x[0])
            gt_arr = np.array(gt_poses, dtype=np.float64)
            hf.create_dataset("gt_timestamps", data=gt_arr[:, 0])
            hf.create_dataset("gt_poses", data=gt_arr[:, 1:8])

        # Second pass: read frames and write one at a time
        img_idx = 0
        seg_idx = 0

        with Reader(bag_path) as reader:
            conns_by_topic = {}
            for conn in reader.connections:
                conns_by_topic.setdefault(conn.topic, []).append(conn)

            for conn, timestamp, rawdata in reader.messages():
                if img_idx >= n_img and seg_idx >= n_seg:
                    break

                if conn.topic == "/cam0/image_raw" and img_idx < n_img:
                    msg = decode(rawdata, conn.msgtype)
                    h, w = msg.height, msg.width
                    enc = msg.encoding
                    data = np.frombuffer(msg.data, dtype=np.uint8)
                    if enc in ("rgb8", "8UC3"):
                        img = data.reshape(h, w, 3)
                    elif enc in ("bgr8",):
                        img = data.reshape(h, w, 3)[:, :, ::-1]
                    elif enc in ("mono8", "8UC1"):
                        img = np.stack([data.reshape(h, w)] * 3, axis=-1)
                    else:
                        img = data.reshape(h, w, -1)[:, :, :3]

                    if img.shape[0] != out_h or img.shape[1] != out_w:
                        img = cv2.resize(img, (out_w, out_h),
                                         interpolation=cv2.INTER_LINEAR)

                    ds_images[img_idx] = img.astype(np.uint8)
                    ds_img_ts[img_idx] = timestamp / 1e9
                    img_idx += 1

                    if img_idx % 500 == 0:
                        print(f"    images: {img_idx}/{n_img}", end="\r")

                elif conn.topic == "/cam0/segmentation" and n_seg > 0 and seg_idx < n_seg:
                    msg = decode(rawdata, conn.msgtype)
                    h, w = msg.height, msg.width
                    enc = msg.encoding
                    data = np.frombuffer(msg.data, dtype=np.uint8)

                    if enc in ("rgb8", "8UC3"):
                        seg = data.reshape(h, w, 3)
                    elif enc in ("bgr8",):
                        seg = data.reshape(h, w, 3)[:, :, ::-1]
                    elif enc in ("mono8", "8UC1"):
                        seg = data.reshape(h, w)
                    else:
                        arr = data.reshape(h, w, -1)
                        seg = arr[:, :, 0] if arr.shape[2] > 0 else np.zeros((h, w), np.uint8)

                    if seg.ndim == 3 and (seg.shape[0] != seg_out_h or seg.shape[1] != seg_out_w):
                        seg = cv2.resize(seg, (seg_out_w, seg_out_h),
                                         interpolation=cv2.INTER_NEAREST)
                    elif seg.ndim == 2 and (seg.shape[0] != seg_out_h or seg.shape[1] != seg_out_w):
                        seg = cv2.resize(seg, (seg_out_w, seg_out_h),
                                         interpolation=cv2.INTER_NEAREST)

                    ds_seg[seg_idx] = seg.astype(np.uint8)
                    ds_seg_ts[seg_idx] = timestamp / 1e9
                    seg_idx += 1

        print(f"    images: {img_idx}/{n_img}   seg: {seg_idx}/{n_seg}")

    size_mb = os.path.getsize(out_path) / 1e6
    print(f"  Wrote {out_path}  ({size_mb:.0f} MB)")


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def find_bags(root: str):
    """Recursively find all .bag files under root, return list of (rel_path, abs_path)."""
    bags = []
    for dirpath, _, filenames in os.walk(root):
        for fn in filenames:
            if fn.endswith(".bag"):
                abs_path = os.path.join(dirpath, fn)
                rel_path = os.path.relpath(abs_path, root)
                bags.append((rel_path, abs_path))
    return sorted(bags)


def main():
    parser = argparse.ArgumentParser(
        description="Convert VIODE ROS bags → HDF5 for fast, low-RAM training")
    parser.add_argument("--viode_root", required=True,
                        help="Root directory containing .bag files")
    parser.add_argument("--out_root", required=True,
                        help="Output directory for .h5 files (mirrors input tree)")
    parser.add_argument("--resize_h", type=int, default=None,
                        help="Resize images/seg to this height (default: keep original)")
    parser.add_argument("--resize_w", type=int, default=None,
                        help="Resize images/seg to this width (default: keep original)")
    parser.add_argument("--gzip_level", type=int, default=1,
                        help="gzip compression level 0-9 (default 1 = fast)")
    parser.add_argument("--skip_existing", action="store_true",
                        help="Skip bags that already have a .h5 output")
    args = parser.parse_args()

    bags = find_bags(args.viode_root)
    if not bags:
        print(f"No .bag files found under {args.viode_root}")
        sys.exit(1)

    print(f"Found {len(bags)} bag(s) under {args.viode_root}")

    for rel_path, abs_path in bags:
        out_rel = rel_path.replace(".bag", ".h5")
        out_path = os.path.join(args.out_root, out_rel)

        if args.skip_existing and os.path.exists(out_path):
            print(f"[SKIP] {rel_path}")
            continue

        print(f"\n[{rel_path}]")
        try:
            convert_bag(abs_path, out_path,
                        resize_h=args.resize_h,
                        resize_w=args.resize_w,
                        gzip_level=args.gzip_level)
        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()

    print("\nDone.")


if __name__ == "__main__":
    main()
