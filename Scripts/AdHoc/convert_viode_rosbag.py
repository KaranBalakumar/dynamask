#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
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
    channels = 3
    row_bytes = int(msg.step)
    min_row_bytes = int(msg.width) * channels
    if row_bytes < min_row_bytes:
        raise ValueError(f"Invalid step={msg.step} for {msg.width}x{msg.height} {msg.encoding}")
    flat = np.frombuffer(msg.data, dtype=np.uint8)
    expected = int(msg.height) * row_bytes
    if flat.size < expected:
        raise ValueError(f"Image buffer too small: got {flat.size}, expected at least {expected}")
    rows = flat[:expected].reshape(int(msg.height), row_bytes)
    img = rows[:, :min_row_bytes].reshape(int(msg.height), int(msg.width), channels)
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


def nearest(cache: dict[int, np.ndarray], ts: int, tol_ns: int) -> np.ndarray | None:
    if len(cache) == 0:
        return None
    k = min(cache.keys(), key=lambda x: abs(x - ts))
    if abs(k - ts) > tol_ns:
        return None
    return cache[k]


def prune_cache_before(cache: dict[int, np.ndarray], min_ts: int) -> None:
    stale = [k for k in cache if k < min_ts]
    for k in stale:
        cache.pop(k, None)


def prune_unmatched_cache(cache: dict[int, np.ndarray], min_match_ts: int, max_items: int) -> None:
    prune_cache_before(cache, min_match_ts)
    if len(cache) <= max_items:
        return
    stale = sorted(cache.keys())[: len(cache) - max_items]
    for k in stale:
        cache.pop(k, None)


def decode_pose(msg) -> np.ndarray:
    pose = msg.pose.pose if hasattr(msg, "pose") and hasattr(msg.pose, "pose") else msg.pose
    return np.array(
        [
            pose.position.x,
            pose.position.y,
            pose.position.z,
            pose.orientation.x,
            pose.orientation.y,
            pose.orientation.z,
            pose.orientation.w,
        ],
        dtype=np.float64,
    )


def _parse_t_bs_path(path: Path) -> np.ndarray:
    suffix = path.suffix.lower()
    if suffix == ".npy":
        arr = np.asarray(np.load(path), dtype=np.float32).reshape(-1)
    elif suffix == ".json":
        payload = json.loads(path.read_text())
        if isinstance(payload, dict):
            if "T_BS" in payload:
                payload = payload["T_BS"]
            elif "translation" in payload and "quaternion" in payload:
                payload = [*payload["translation"], *payload["quaternion"]]
        arr = np.asarray(payload, dtype=np.float32).reshape(-1)
    else:
        raw = path.read_text().replace(",", " ").replace("[", " ").replace("]", " ")
        arr = np.fromstring(raw, sep=" ", dtype=np.float32)

    if arr.shape != (7,):
        raise ValueError(f"Expected 7 values for T_BS in {path}, got shape {arr.shape}")
    return arr


def resolve_t_bs(args: argparse.Namespace) -> np.ndarray:
    t = np.asarray(args.t_bs_translation, dtype=np.float32)
    q = np.asarray(args.t_bs_quaternion, dtype=np.float32)
    uses_default = np.allclose(t, np.zeros(3, dtype=np.float32)) and np.allclose(q, np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32))

    if args.t_bs_path is not None:
        if not uses_default:
            raise ValueError("Cannot combine --t-bs-path with --t-bs-translation/--t-bs-quaternion overrides.")
        t_bs = _parse_t_bs_path(args.t_bs_path)
    else:
        if uses_default:
            print("Using default identity T_BS. Override with --t-bs-path or --t-bs-translation/--t-bs-quaternion.")
        q_norm = float(np.linalg.norm(q))
        if q_norm <= 0.0:
            raise ValueError("Invalid --t-bs-quaternion: norm must be > 0.")
        q = q / q_norm
        t_bs = np.concatenate([t, q], axis=0).astype(np.float32)
    return t_bs


def main():
    parser = argparse.ArgumentParser(description="Convert VIODE rosbag to chunked HDF5 stream format.")
    parser.add_argument("--bag", type=Path, required=True, help="Path to rosbag2 directory")
    parser.add_argument("--out", type=Path, required=True, help="Output .h5 path")
    parser.add_argument("--left-topic", type=str, default="/stereo/left/image_raw")
    parser.add_argument("--right-topic", type=str, default="/stereo/right/image_raw")
    parser.add_argument("--imu-topic", type=str, default="/imu/data")
    parser.add_argument("--gt-topic", type=str, default="", help="Optional GT pose topic (PoseStamped/Odometry)")
    parser.add_argument("--stereo-sync-tol-ms", type=float, default=5.0, help="L/R sync tolerance in milliseconds")
    parser.add_argument("--gt-sync-tol-ms", type=float, default=20.0, help="Stereo/GT sync tolerance in milliseconds")
    parser.add_argument("--max-unmatched-cache", type=int, default=4096, help="Max unmatched left/right frames kept in memory")
    parser.add_argument("--fx", type=float, required=True)
    parser.add_argument("--fy", type=float, required=True)
    parser.add_argument("--cx", type=float, required=True)
    parser.add_argument("--cy", type=float, required=True)
    parser.add_argument("--baseline", type=float, required=True)
    parser.add_argument("--gravity", type=float, default=9.81)
    parser.add_argument("--t-bs-path", type=Path, default=None, help="Optional path to T_BS [tx ty tz qx qy qz qw] (.txt/.json/.npy)")
    parser.add_argument("--t-bs-translation", type=float, nargs=3, default=(0.0, 0.0, 0.0), metavar=("TX", "TY", "TZ"))
    parser.add_argument(
        "--t-bs-quaternion",
        type=float,
        nargs=4,
        default=(0.0, 0.0, 0.0, 1.0),
        metavar=("QX", "QY", "QZ", "QW"),
    )
    args = parser.parse_args()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    K = np.array([[args.fx, 0.0, args.cx], [0.0, args.fy, args.cy], [0.0, 0.0, 1.0]], dtype=np.float32)
    T_BS = resolve_t_bs(args)

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
        ds_st_gt = g_st.create_dataset("gt_pose", shape=(0, 7), maxshape=(None, 7), dtype=np.float64, chunks=(1024, 7))

        left_cache: dict[int, np.ndarray] = {}
        right_cache: dict[int, np.ndarray] = {}
        gt_cache: dict[int, np.ndarray] = {}
        left_ds = right_ds = None
        tol_ns = int(args.stereo_sync_tol_ms * 1e6)
        gt_tol_ns = int(args.gt_sync_tol_ms * 1e6)
        max_unmatched_cache = max(1, int(args.max_unmatched_cache))
        latest_left_ts: int | None = None
        latest_right_ts: int | None = None

        with AnyReader([args.bag]) as reader:
            topics = {args.left_topic, args.right_topic, args.imu_topic}
            if args.gt_topic:
                topics.add(args.gt_topic)
            conns = [c for c in reader.connections if c.topic in topics]
            for conn, timestamp, rawdata in reader.messages(connections=conns):
                msg = reader.deserialize(rawdata, conn.msgtype)
                if conn.topic == args.imu_topic:
                    t_ns = msg_time_ns(msg, timestamp)
                    append_scalar(ds_imu_t, np.int64(t_ns))
                    append_row(ds_imu_a, np.array([msg.linear_acceleration.x, msg.linear_acceleration.y, msg.linear_acceleration.z], dtype=np.float32))
                    append_row(ds_imu_g, np.array([msg.angular_velocity.x, msg.angular_velocity.y, msg.angular_velocity.z], dtype=np.float32))
                    continue
                if args.gt_topic and conn.topic == args.gt_topic:
                    gt_cache[msg_time_ns(msg, timestamp)] = decode_pose(msg)
                    continue

                img = decode_image(msg)
                t_ns = msg_time_ns(msg, timestamp)
                if left_ds is None:
                    H, W, C = img.shape
                    left_ds = g_st.create_dataset("left", shape=(0, H, W, C), maxshape=(None, H, W, C), dtype=np.uint8, chunks=(1, H, W, C), compression="lzf")
                    right_ds = g_st.create_dataset("right", shape=(0, H, W, C), maxshape=(None, H, W, C), dtype=np.uint8, chunks=(1, H, W, C), compression="lzf")

                if conn.topic == args.left_topic:
                    left_cache[t_ns] = img
                    latest_left_ts = t_ns
                    rk, rimg = pop_nearest(right_cache, t_ns, tol_ns)
                    if rk is not None and rimg is not None:
                        pair_ts = min(t_ns, rk)
                        append_scalar(ds_st_t, np.int64(pair_ts))
                        append_row(left_ds, img)
                        append_row(right_ds, rimg)
                        gt = nearest(gt_cache, pair_ts, gt_tol_ns)
                        append_row(ds_st_gt, np.full((7,), np.nan, dtype=np.float64) if gt is None else gt)
                        prune_cache_before(gt_cache, pair_ts - gt_tol_ns)
                    prune_unmatched_cache(right_cache, t_ns - tol_ns, max_unmatched_cache)
                    if latest_right_ts is not None:
                        prune_unmatched_cache(left_cache, latest_right_ts - tol_ns, max_unmatched_cache)
                    else:
                        prune_unmatched_cache(left_cache, t_ns - tol_ns, max_unmatched_cache)
                else:
                    right_cache[t_ns] = img
                    latest_right_ts = t_ns
                    lk, limg = pop_nearest(left_cache, t_ns, tol_ns)
                    if lk is not None and limg is not None:
                        pair_ts = min(t_ns, lk)
                        append_scalar(ds_st_t, np.int64(pair_ts))
                        append_row(left_ds, limg)
                        append_row(right_ds, img)
                        gt = nearest(gt_cache, pair_ts, gt_tol_ns)
                        append_row(ds_st_gt, np.full((7,), np.nan, dtype=np.float64) if gt is None else gt)
                        prune_cache_before(gt_cache, pair_ts - gt_tol_ns)
                    prune_unmatched_cache(left_cache, t_ns - tol_ns, max_unmatched_cache)
                    if latest_left_ts is not None:
                        prune_unmatched_cache(right_cache, latest_left_ts - tol_ns, max_unmatched_cache)
                    else:
                        prune_unmatched_cache(right_cache, t_ns - tol_ns, max_unmatched_cache)

    print(f"Wrote stream dataset: {args.out}")


if __name__ == "__main__":
    main()
