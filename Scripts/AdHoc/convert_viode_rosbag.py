#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import h5py
import numpy as np
from rosbags.highlevel import AnyReader


def decode_image(msg) -> np.ndarray:
    if hasattr(msg, "format"):  # sensor_msgs/CompressedImage
        arr = np.frombuffer(msg.data, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None:
            raise RuntimeError("Failed to decode compressed image")
        return img

    if msg.encoding.lower() not in {"bgr8", "rgb8"}:
        raise ValueError(f"Unsupported image encoding: {msg.encoding}")
    img = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, -1)
    if msg.encoding.lower() == "rgb8":
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    return img


def append_row(ds, row: np.ndarray):
    n = ds.shape[0]
    ds.resize((n + 1, *ds.shape[1:]))
    ds[n] = row


def append_scalar(ds, x):
    n = ds.shape[0]
    ds.resize((n + 1,))
    ds[n] = x


def msg_time_ns(msg, fallback_ns: int) -> int:
    if hasattr(msg, "header") and hasattr(msg.header, "stamp"):
        st = msg.header.stamp
        if hasattr(st, "sec") and hasattr(st, "nanosec"):
            return int(st.sec) * 1_000_000_000 + int(st.nanosec)
    return int(fallback_ns)


def pop_nearest(cache: dict[int, np.ndarray], ts: int, tol_ns: int) -> tuple[int, np.ndarray] | tuple[None, None]:
    if len(cache) == 0:
        return None, None
    keys = list(cache.keys())
    k = min(keys, key=lambda x: abs(x - ts))
    if abs(k - ts) > tol_ns:
        return None, None
    return k, cache.pop(k)


def main():
    parser = argparse.ArgumentParser(description="Convert VIODE rosbag to chunked HDF5 stream format.")
    parser.add_argument("--bag", type=Path, required=True, help="Path to rosbag2 directory")
    parser.add_argument("--out", type=Path, required=True, help="Output .h5 path")
    parser.add_argument("--left-topic", type=str, default="/stereo/left/image_raw")
    parser.add_argument("--right-topic", type=str, default="/stereo/right/image_raw")
    parser.add_argument("--imu-topic", type=str, default="/imu/data")
    parser.add_argument("--stereo-sync-tol-ms", type=float, default=5.0, help="L/R sync tolerance in milliseconds")
    parser.add_argument("--fx", type=float, required=True)
    parser.add_argument("--fy", type=float, required=True)
    parser.add_argument("--cx", type=float, required=True)
    parser.add_argument("--cy", type=float, required=True)
    parser.add_argument("--baseline", type=float, required=True)
    parser.add_argument("--gravity", type=float, default=9.81)
    args = parser.parse_args()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    K = np.array([[args.fx, 0.0, args.cx], [0.0, args.fy, args.cy], [0.0, 0.0, 1.0]], dtype=np.float32)
    T_BS = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0], dtype=np.float32)  # identity SE3 (xyzw quat)

    with h5py.File(args.out, "w") as h5:
        g_st = h5.create_group("stereo")
        g_imu = h5.create_group("imu")
        g_cal = h5.create_group("calib")
        g_cal.create_dataset("K", data=K)
        g_cal.create_dataset("T_BS", data=T_BS)
        g_cal.create_dataset("baseline", data=np.float32(args.baseline))
        g_cal.create_dataset("gravity", data=np.float32(args.gravity))

        ds_st_t = g_st.create_dataset("time_ns", shape=(0,), maxshape=(None,), dtype=np.int64, chunks=(1024,))
        ds_imu_t = g_imu.create_dataset("time_ns", shape=(0,), maxshape=(None,), dtype=np.int64, chunks=(4096,))
        ds_imu_a = g_imu.create_dataset("acc", shape=(0, 3), maxshape=(None, 3), dtype=np.float32, chunks=(4096, 3))
        ds_imu_g = g_imu.create_dataset("gyro", shape=(0, 3), maxshape=(None, 3), dtype=np.float32, chunks=(4096, 3))

        left_cache: dict[int, np.ndarray] = {}
        right_cache: dict[int, np.ndarray] = {}
        left_ds = right_ds = None
        tol_ns = int(args.stereo_sync_tol_ms * 1e6)

        with AnyReader([args.bag]) as reader:
            conns = [c for c in reader.connections if c.topic in {args.left_topic, args.right_topic, args.imu_topic}]
            for conn, timestamp, rawdata in reader.messages(connections=conns):
                msg = reader.deserialize(rawdata, conn.msgtype)
                if conn.topic == args.imu_topic:
                    t_ns = msg_time_ns(msg, timestamp)
                    append_scalar(ds_imu_t, np.int64(t_ns))
                    append_row(ds_imu_a, np.array([msg.linear_acceleration.x, msg.linear_acceleration.y, msg.linear_acceleration.z], dtype=np.float32))
                    append_row(ds_imu_g, np.array([msg.angular_velocity.x, msg.angular_velocity.y, msg.angular_velocity.z], dtype=np.float32))
                    continue

                img = decode_image(msg)
                t_ns = msg_time_ns(msg, timestamp)
                if left_ds is None:
                    H, W, C = img.shape
                    left_ds = g_st.create_dataset("left", shape=(0, H, W, C), maxshape=(None, H, W, C), dtype=np.uint8, chunks=(1, H, W, C), compression="lzf")
                    right_ds = g_st.create_dataset("right", shape=(0, H, W, C), maxshape=(None, H, W, C), dtype=np.uint8, chunks=(1, H, W, C), compression="lzf")

                if conn.topic == args.left_topic:
                    left_cache[t_ns] = img
                    rk, rimg = pop_nearest(right_cache, t_ns, tol_ns)
                    if rk is not None and rimg is not None:
                        pair_ts = min(t_ns, rk)
                        append_scalar(ds_st_t, np.int64(pair_ts))
                        append_row(left_ds, img)
                        append_row(right_ds, rimg)
                else:
                    right_cache[t_ns] = img
                    lk, limg = pop_nearest(left_cache, t_ns, tol_ns)
                    if lk is not None and limg is not None:
                        pair_ts = min(t_ns, lk)
                        append_scalar(ds_st_t, np.int64(pair_ts))
                        append_row(left_ds, limg)
                        append_row(right_ds, img)

    print(f"Wrote stream dataset: {args.out}")


if __name__ == "__main__":
    main()
