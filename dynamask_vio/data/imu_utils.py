"""
IMU utilities: interpolation, windowing, preintegration ground truth.

All functions operate on numpy arrays (dataset level) and return numpy.
Conversion to torch happens in the dataset __getitem__.
"""

import numpy as np
from scipy.spatial.transform import Rotation


def interpolate_imu(imu_timestamps: np.ndarray, imu_data: np.ndarray,
                    query_timestamps: np.ndarray) -> np.ndarray:
    """Linearly interpolate IMU measurements at query timestamps.

    Args:
        imu_timestamps: [M] sorted IMU timestamps (seconds)
        imu_data: [M, 6] (ax, ay, az, gx, gy, gz)
        query_timestamps: [K] timestamps to interpolate at

    Returns:
        [K, 6] interpolated IMU measurements
    """
    result = np.zeros((len(query_timestamps), 6), dtype=np.float64)
    for ch in range(6):
        result[:, ch] = np.interp(query_timestamps, imu_timestamps,
                                   imu_data[:, ch])
    return result


def get_imu_window(imu_timestamps: np.ndarray, imu_data: np.ndarray,
                   t_start: float, t_end: float,
                   max_samples: int = 15) -> tuple:
    """Extract IMU samples between two camera timestamps.

    Args:
        imu_timestamps: [M] sorted (seconds)
        imu_data: [M, 6]
        t_start, t_end: camera frame timestamps
        max_samples: maximum samples to keep (for batching)

    Returns:
        window: [N, 7] — (dt, ax, ay, az, gx, gy, gz) where dt is relative
                          to t_start
        valid_mask: [N] boolean — True for real samples, False for padding
    """
    mask = (imu_timestamps >= t_start) & (imu_timestamps <= t_end)
    idx = np.where(mask)[0]

    if len(idx) == 0:
        # Fallback: interpolate at start and end
        interp = interpolate_imu(imu_timestamps, imu_data,
                                  np.array([t_start, t_end]))
        dt = np.array([[0.0], [t_end - t_start]])
        window = np.hstack([dt, interp])
        N = 2
    else:
        ts = imu_timestamps[idx]
        data = imu_data[idx]
        dt = (ts - t_start).reshape(-1, 1)
        window = np.hstack([dt, data])
        N = len(idx)

    # Truncate if too many samples
    if N > max_samples:
        # Subsample uniformly
        indices = np.linspace(0, N - 1, max_samples, dtype=int)
        window = window[indices]
        N = max_samples

    # Pad to max_samples
    padded = np.zeros((max_samples, 7), dtype=np.float32)
    valid = np.zeros(max_samples, dtype=bool)
    padded[:N] = window.astype(np.float32)
    valid[:N] = True

    return padded, valid


def compute_gt_preintegration(poses: np.ndarray, timestamps: np.ndarray,
                               idx_start: int, idx_end: int,
                               gravity: np.ndarray = None) -> dict:
    """Compute ground truth preintegrated quantities between two poses.

    Uses the definition from Forster et al.:
        ΔR = R_i^T @ R_j
        Δv = R_i^T @ (v_j - v_i - g * dt)
        Δp = R_i^T @ (p_j - p_i - v_i * dt - 0.5 * g * dt^2)

    Args:
        poses: [K, 7] — (px, py, pz, qx, qy, qz, qw) per timestamp
        timestamps: [K] corresponding timestamps
        idx_start, idx_end: indices into poses/timestamps
        gravity: [3] gravity vector in world frame, default [0,0,-9.81]

    Returns:
        dict with delta_R [3,3], delta_v [3], delta_p [3], dt scalar
    """
    if gravity is None:
        gravity = np.array([0.0, 0.0, -9.81])

    p_i = poses[idx_start, 0:3]
    q_i = poses[idx_start, 3:7]  # (qx, qy, qz, qw)
    p_j = poses[idx_end, 0:3]
    q_j = poses[idx_end, 3:7]

    t_i = timestamps[idx_start]
    t_j = timestamps[idx_end]
    dt = t_j - t_i

    R_i = Rotation.from_quat(q_i).as_matrix()  # scipy uses (x,y,z,w)
    R_j = Rotation.from_quat(q_j).as_matrix()

    # Estimate velocities via finite differences if not provided.
    # All divisions guard against near-zero denominators: even though empirical
    # checks show GT timestamps are strictly increasing at the millisecond scale
    # for EuRoC/VIODE, duplicate timestamps can appear at sequence edges and a
    # single NaN velocity here would propagate to loss = inf under NLL.
    eps = 1e-6
    if idx_start > 0 and idx_end < len(poses) - 1:
        dt_prev = timestamps[idx_start] - timestamps[idx_start - 1]
        dt_next = timestamps[idx_start + 1] - timestamps[idx_start]
        v_i = (poses[idx_start + 1, 0:3] - poses[idx_start - 1, 0:3]) / \
              max(dt_prev + dt_next, eps)

        dt_prev_j = timestamps[idx_end] - timestamps[idx_end - 1]
        dt_next_j = timestamps[min(idx_end + 1, len(poses) - 1)] - timestamps[idx_end]
        if dt_next_j > eps:
            v_j = (poses[min(idx_end + 1, len(poses) - 1), 0:3] -
                   poses[idx_end - 1, 0:3]) / max(dt_prev_j + dt_next_j, eps)
        else:
            v_j = (poses[idx_end, 0:3] - poses[idx_end - 1, 0:3]) / \
                  max(dt_prev_j, eps)
    else:
        # Fallback: forward/backward difference
        if idx_end < len(poses) - 1:
            dt_fwd = timestamps[idx_start + 1] - timestamps[idx_start]
            v_i = (poses[idx_start + 1, 0:3] - poses[idx_start, 0:3]) / max(dt_fwd, eps)
        else:
            v_i = np.zeros(3)
        v_j = (poses[idx_end, 0:3] - poses[max(idx_end - 1, 0), 0:3]) / \
              max(timestamps[idx_end] - timestamps[max(idx_end - 1, 0)], eps)

    R_i_T = R_i.T
    delta_R = R_i_T @ R_j
    delta_v = R_i_T @ (v_j - v_i - gravity * dt)
    delta_p = R_i_T @ (p_j - p_i - v_i * dt - 0.5 * gravity * dt ** 2)

    return {
        "delta_R": delta_R.astype(np.float32),
        "delta_v": delta_v.astype(np.float32),
        "delta_p": delta_p.astype(np.float32),
        "dt": float(dt),
    }


def compute_gt_bias(imu_raw: np.ndarray, poses: np.ndarray,
                    pose_timestamps: np.ndarray, imu_timestamps: np.ndarray,
                    gravity: np.ndarray = None) -> np.ndarray:
    """Compute ground truth IMU bias from trajectory and raw readings.

    a_ideal = R_wb^T @ (a_world - g)
    w_ideal = angular velocity from rotation sequence
    bias = raw - ideal

    Args:
        imu_raw: [M, 6] (ax, ay, az, gx, gy, gz)
        poses: [K, 7] (px, py, pz, qx, qy, qz, qw)
        pose_timestamps: [K]
        imu_timestamps: [M]
        gravity: [3] world-frame gravity

    Returns:
        biases: [M, 6] (ba_x, ba_y, ba_z, bg_x, bg_y, bg_z)
    """
    if gravity is None:
        gravity = np.array([0.0, 0.0, -9.81])

    M = len(imu_timestamps)
    biases = np.zeros((M, 6), dtype=np.float32)

    # Interpolate poses at IMU timestamps
    pose_interp = np.zeros((M, 7), dtype=np.float64)
    for ch in range(3):  # position
        pose_interp[:, ch] = np.interp(imu_timestamps, pose_timestamps,
                                        poses[:, ch])
    # Quaternion slerp is complex; use simple linear interp + renormalize
    for ch in range(3, 7):
        pose_interp[:, ch] = np.interp(imu_timestamps, pose_timestamps,
                                        poses[:, ch])
    # Renormalize quaternions
    q_norm = np.linalg.norm(pose_interp[:, 3:7], axis=1, keepdims=True)
    pose_interp[:, 3:7] /= np.maximum(q_norm, 1e-10)

    for i in range(M):
        R = Rotation.from_quat(pose_interp[i, 3:7]).as_matrix()
        R_T = R.T

        # Compute world-frame acceleration from finite differences
        dt_back = 0.005  # 200 Hz
        dt_fwd = 0.005
        t = imu_timestamps[i]

        # Interpolate positions at t-dt, t, t+dt
        p_prev = np.array([np.interp(t - dt_back, pose_timestamps, poses[:, c])
                          for c in range(3)])
        p_curr = pose_interp[i, 0:3]
        p_next = np.array([np.interp(t + dt_fwd, pose_timestamps, poses[:, c])
                          for c in range(3)])

        a_world = (p_next - 2 * p_curr + p_prev) / (dt_back * dt_fwd)
        a_ideal = R_T @ (a_world - gravity)

        # Angular velocity from rotation
        if i < M - 1:
            dt = imu_timestamps[min(i + 1, M - 1)] - imu_timestamps[i]
            if dt > 1e-8:
                R_next = Rotation.from_quat(pose_interp[min(i + 1, M - 1), 3:7]).as_matrix()
                dR = R_T @ R_next
                w_ideal = Rotation.from_matrix(dR).as_rotvec() / dt
            else:
                w_ideal = np.zeros(3)
        else:
            w_ideal = np.zeros(3)

        biases[i, 0:3] = imu_raw[i, 0:3] - a_ideal
        biases[i, 3:6] = imu_raw[i, 3:6] - w_ideal

    return biases
