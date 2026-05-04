"""Virtual KITTI 2 dataset loader — supervised (GT flow, GT depth, GT pose).

VKitti2 format per variant:
    SceneXX/variant/
        frames/rgb/Camera_0/rgb_%05d.jpg         1242×375 RGB
        frames/rgb/Camera_1/rgb_%05d.jpg          stereo right
        frames/depth/Camera_0/depth_%05d.png      16-bit, 1px = 1cm
        frames/forwardFlow/Camera_0/flow_%05d.png  16-bit KITTI-format flow
        intrinsic.txt                              fx fy cx cy per frame
        extrinsic.txt                              4×4 cam-to-world per frame
        pose.txt                                   object bbox poses (unused)
"""

import torch, cv2, numpy as np
import torch.nn.functional as F
from pathlib import Path
from types import SimpleNamespace
import pypose as pp

from DataLoader.Interface import StereoData, StereoInertialFrame, IMUData, AttitudeData
from DataLoader.SequenceBase import SequenceBase


class VKitti2Sequence(SequenceBase[StereoInertialFrame]):
    """Load one variant of a VKitti2 scene as a StereoInertialFrame sequence."""

    @classmethod
    def name(cls) -> str:
        return "VKitti2"

    def __init__(self, cfg: SimpleNamespace):
        self.root = Path(cfg.root)
        self.variant = getattr(cfg, "variant", "clone")
        self.use_real_imu = getattr(cfg, "use_real_imu", False)
        self.gravity = getattr(cfg, "gravity", 9.81)
        self.imu_freq = getattr(cfg, "imu_freq", 200)
        self.variant_dir = self.root / self.variant

        # --- Intrinsics (fx = fy, same for all frames) ---
        intrinsic_path = self.variant_dir / "intrinsic.txt"
        with open(intrinsic_path) as f:
            f.readline()  # header
            line = f.readline().strip().split()
        fx = float(line[2]); fy = float(line[3])
        cx = float(line[4]); cy = float(line[5])
        self.K_raw = torch.tensor(
            [[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=torch.float32
        ).unsqueeze(0)

        self.baseline = float(getattr(cfg, "baseline", 0.532725))

        # --- Read extrinsics (world-to-camera, Camera 0 only) ---
        # Training loop applies NED_R_cam transform unconditionally.
        # VKitti2 extrinsics are already camera-frame, so we pre-rotate
        # by NED^T so the training loop's NED cancels out.
        # NED = [[0,0,1],[1,0,0],[0,1,0]]  →  NED^T @ (NED @ R @ NED^T) @ NED = R
        NED_T = torch.tensor([[0.,1.,0.],[0.,0.,1.],[1.,0.,0.]], dtype=torch.float64)
        NED_R = NED_T.T  # = [[0,0,1],[1,0,0],[0,1,0]]
        self.T_wc = {}
        self.T_w2c = {}
        self.frame_indices = []
        extrinsics_path = self.variant_dir / "extrinsic.txt"
        with open(extrinsics_path) as f:
            f.readline()
            for line in f:
                parts = line.strip().split()
                frame_idx = int(parts[0])
                cam_id = int(parts[1])
                if cam_id != 0:
                    continue
                vals = [float(x) for x in parts[2:18]]
                T_w2c_raw = torch.tensor(
                    [vals[0:4], vals[4:8], vals[8:12], vals[12:16]],
                    dtype=torch.float64,
                )
                self.T_w2c[frame_idx] = T_w2c_raw
                # c2w = inv(w2c), then pre-rotate by NED^T so NED cancels in training
                T_c2w = torch.linalg.inv(T_w2c_raw)
                # Apply pre-rotation: R_c2w_pre = NED @ R_c2w @ NED^T
                R = T_c2w[:3, :3]
                t = T_c2w[:3, 3]
                R_pre = NED_R @ R @ NED_T
                t_pre = NED_R @ t
                T_pre = torch.eye(4, dtype=torch.float64)
                T_pre[:3, :3] = R_pre
                T_pre[:3, 3] = t_pre
                self.T_wc[frame_idx] = T_pre
                self.frame_indices.append(frame_idx)

        self.frame_indices = sorted(set(self.frame_indices))
        # Use only frames that have both flow (t→t+1) and a next frame
        self.num_frames = len(self.frame_indices) - 1  # last frame has no flow

        # --- Image paths ---
        self.rgb_dir = self.variant_dir / "frames" / "rgb" / "Camera_0"
        self.rgb_right_dir = self.variant_dir / "frames" / "rgb" / "Camera_1"
        self.depth_dir = self.variant_dir / "frames" / "depth" / "Camera_0"
        self.flow_dir = self.variant_dir / "frames" / "forwardFlow" / "Camera_0"

        assert self.rgb_dir.exists(), f"Missing: {self.rgb_dir}"
        assert self.depth_dir.exists(), f"Missing: {self.depth_dir}"
        assert self.flow_dir.exists(), f"Missing: {self.flow_dir}"
        assert self.rgb_right_dir.exists(), f"Missing: {self.rgb_right_dir}"

        # --- Precompute synthetic IMU + attitude if requested ---
        self._imu_samples = None
        self._att_samples = None
        if self.use_real_imu:
            self._imu_samples, self._att_samples = _generate_imu_from_poses(
                self.T_wc, self.frame_indices, self.imu_freq, self.gravity
            )

        # Images: non-uniform resize 1242×375 → 640×480 (same as KITTI inference)
        # Residual: computed at original res, then resize scalar map.
        # K scaled accordingly. Geometry is correct because r is computed first.
        self.orig_h, self.orig_w = 375, 1242
        self.target_h, self.target_w = 480, 640
        self.scale_h = self.target_h / self.orig_h
        self.scale_w = self.target_w / self.orig_w
        self.K = self.K_raw.clone()
        self.K[:, 0, 0] *= self.scale_w
        self.K[:, 1, 1] *= self.scale_h
        self.K[:, 0, 2] *= self.scale_w
        self.K[:, 1, 2] *= self.scale_h

        super().__init__(self.num_frames)

    def __len__(self):
        return self.num_frames

    def __getitem__(self, index: int):
        frame_idx = self.frame_indices[index]

        # Load images
        img_l = _load_rgb(self.rgb_dir / f"rgb_{frame_idx:05d}.jpg")
        img_r = _load_rgb(self.rgb_right_dir / f"rgb_{frame_idx:05d}.jpg")
        depth = _load_depth(self.depth_dir / f"depth_{frame_idx:05d}.png")
        flow, flow_mask = _load_flow(self.flow_dir / f"flow_{frame_idx:05d}.png")

        # --- Compute rigid-flow residual at ORIGINAL resolution (correct geometry) ---
        T_wc_cur = self.T_wc[frame_idx]
        T_wc_nxt = self.T_wc[frame_idx + 1] if frame_idx + 1 in self.T_wc else T_wc_cur
        pose_cur = pp.mat2SE3(T_wc_cur.unsqueeze(0))
        pose_nxt = pp.mat2SE3(T_wc_nxt.unsqueeze(0))

        NED_R_cam = torch.tensor([[0,0,1],[1,0,0],[0,1,0]], dtype=torch.float64)
        T_cur_mat = pp.SE3(pose_cur).matrix().double()
        T_nxt_mat = pp.SE3(pose_nxt).matrix().double()
        T_rel = torch.linalg.inv(T_nxt_mat) @ T_cur_mat
        R_rel, t_rel = T_rel[0, :3, :3], T_rel[0, :3, 3:4]
        R_c = NED_R_cam.T @ R_rel @ NED_R_cam
        t_c = NED_R_cam.T @ t_rel
        bottom = torch.tensor([[0.,0.,0.,1.]], dtype=torch.float64)
        T_rel_mat = torch.cat([torch.cat([R_c, t_c], dim=1), bottom], dim=0).unsqueeze(0)

        from Train.MatchingNet.loss import compute_rigid_flow_jacobian
        f_rigid_orig, _, _ = compute_rigid_flow_jacobian(
            depth.float(), self.K_raw.float(), T_rel_mat.float()
        )
        r_abs_orig = (flow.float() - f_rigid_orig.float()).norm(dim=1, keepdim=True)  # (1,1,375,1242)
        # Mask invalid flow pixels (WAFT convention: r=0 where flow is unavailable)
        r_abs_orig = r_abs_orig * flow_mask.float()

        # Resize everything to 640×480 (same as KITTI inference pipeline)
        target_size = (self.target_h, self.target_w)
        img_l  = F.interpolate(img_l,  size=target_size, mode='bilinear', align_corners=False)
        img_r  = F.interpolate(img_r,  size=target_size, mode='bilinear', align_corners=False)
        depth  = F.interpolate(depth,  size=target_size, mode='nearest')
        flow_mask = F.interpolate(flow_mask, size=target_size, mode='nearest')
        # Resize scalar residual (just a heatmap — no geometry)
        r_abs  = F.interpolate(r_abs_orig, size=target_size, mode='bilinear', align_corners=False)
        r_vec = torch.cat([r_abs, torch.zeros_like(r_abs)], dim=1)  # (1,2,480,640)
        # flow_gt is unused for training (precomputed r_vec used instead).
        # Zero it out to avoid misleading debug values from non-uniform resize.
        flow = torch.zeros_like(flow)

        # GT pose as LieTensor
        T_wc = self.T_wc[frame_idx]
        pose = pp.mat2SE3(T_wc.unsqueeze(0))

        # IMU + Attitude data — sliced per-frame (~20 ticks at 200Hz / 10Hz)
        imu_data = None
        att_data = None
        if self.use_real_imu and self._imu_samples is not None:
            ticks_per_frame = self.imu_freq // 10  # 200Hz / 10Hz = 20
            i0 = index * ticks_per_frame
            i1 = i0 + ticks_per_frame
            acc = self._imu_samples["acc"][i0:i1].unsqueeze(0)     # [1, T, 3]
            gyro = self._imu_samples["gyro"][i0:i1].unsqueeze(0)   # [1, T, 3]
            n = acc.shape[1]
            dt_ns = int(1e9 / self.imu_freq)
            t0_ns = i0 * dt_ns
            times_ns = torch.arange(t0_ns, t0_ns + n * dt_ns, dt_ns, dtype=torch.int64).unsqueeze(0).unsqueeze(-1)
            imu_data = IMUData(
                T_BS=pp.identity_SE3(1),
                time_ns=times_ns,
                gravity=[self.gravity],
                acc=acc,
                gyro=gyro,
            )
            # Attitude data for EKF seeding (only first tick needed)
            att = self._att_samples
            pos_0 = att["pos"][i0:i0+1].unsqueeze(0)   # [1, 1, 3]
            vel_0 = att["vel"][i0:i0+1].unsqueeze(0)   # [1, 1, 3]
            rot_0 = att["rot"][i0:i0+1].unsqueeze(0)   # [1, 1, 4] LieTensor
            att_data = AttitudeData(
                T_BS=pp.identity_SE3(1),
                time_ns=times_ns[:, :1],
                gravity=[self.gravity],
                gt_pos=pos_0, gt_vel=vel_0, gt_rot=rot_0,
                init_pos=pos_0, init_vel=vel_0, init_rot=rot_0,
            )

        stereo_data = StereoData(
            T_BS=torch.eye(4).unsqueeze(0),
            K=self.K,
            baseline=torch.tensor([self.baseline]),
            time_ns=[0],
            height=self.target_h,
            width=self.target_w,
            imageL=img_l,
            imageR=img_r,
            gt_depth=depth,
            gt_flow=flow,
            flow_mask=flow_mask.bool(),
            gt_dyn_r_vec=r_vec,
        )

        return StereoInertialFrame(
            idx=[index],
            stereo=stereo_data,
            time_ns=[0],
            gt_pose=pose,
            imu=imu_data,
            gt_attitude=att_data,
        )


# ---- Image loading helpers ----

def _load_rgb(path: Path) -> torch.Tensor:
    img = cv2.imread(str(path))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
    return img.unsqueeze(0)


def _load_depth(path: Path) -> torch.Tensor:
    raw = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    return torch.from_numpy(raw.astype(np.float32) / 100.0).unsqueeze(0).unsqueeze(0)


def _load_flow(path: Path) -> tuple[torch.Tensor, torch.Tensor]:
    """Load VKitti2 16-bit flow PNG.

    Encoding (verified against WAFT reference):
        R channel (idx 2) → u (horizontal flow)
        G channel (idx 1) → v (vertical flow)
        B channel (idx 0) → valid mask (>0 = valid)
        u/v = (value/65535 * 2 - 1) * (size - 1)
    """
    raw = cv2.imread(str(path), cv2.IMREAD_UNCHANGED).astype(np.float64)
    H, W = raw.shape[:2]
    u = (raw[..., 2] / 65535.0 * 2.0 - 1.0) * (W - 1)
    v = (raw[..., 1] / 65535.0 * 2.0 - 1.0) * (H - 1)
    mask = (raw[..., 0] > 0).astype(np.float32)
    flow = np.stack([u, v], axis=0)
    return torch.from_numpy(flow).float().unsqueeze(0), torch.from_numpy(mask).unsqueeze(0).unsqueeze(0)


# ---- Synthetic IMU generation ----

def _generate_imu_from_poses(T_wc: dict, frame_indices: list, imu_freq: int,
                              gravity: float) -> dict:
    from scipy.spatial.transform import Rotation, Slerp
    from scipy.interpolate import interp1d

    frame_rate = 10.0
    times = np.arange(len(frame_indices)) / frame_rate
    positions = np.zeros((len(frame_indices), 3))
    quats = np.zeros((len(frame_indices), 4))

    for i, fi in enumerate(frame_indices):
        T = T_wc[fi].numpy()
        positions[i] = T[:3, 3]
        quats[i] = Rotation.from_matrix(T[:3, :3]).as_quat()

    dt = 1.0 / imu_freq
    imu_times = np.arange(0, times[-1], dt)
    n_imu = len(imu_times)

    pos_interp = interp1d(times, positions, axis=0, kind='cubic', fill_value='extrapolate')
    pos_imu = pos_interp(imu_times)

    vel_w = np.gradient(pos_imu, dt, axis=0)
    acc_w = np.gradient(vel_w, dt, axis=0)

    rots = Rotation.from_quat(quats)
    slerp = Slerp(times, rots)
    rots_imu = slerp(imu_times)

    acc_imu = np.zeros_like(acc_w)
    gyro_imu = np.zeros((n_imu, 3))
    for i in range(n_imu):
        R_w2i = rots_imu[i].as_matrix().T
        acc_imu[i] = R_w2i @ acc_w[i] + R_w2i @ np.array([0, 0, gravity])

    for i in range(1, n_imu):
        dq = rots_imu[i] * rots_imu[i-1].inv()
        angle = dq.magnitude()
        if angle > 1e-10:
            gyro_imu[i] = dq.as_rotvec() / angle * (angle / dt)
        else:
            gyro_imu[i] = gyro_imu[i-1]

    # Add realistic noise
    np.random.seed(42)
    acc_imu += 0.001 + np.cumsum(np.random.randn(n_imu, 3) * 0.0001, axis=0)
    gyro_imu += 0.0001 + np.cumsum(np.random.randn(n_imu, 3) * 1e-5, axis=0)

    # Convert rotation matrices → euler → pypose SO3 LieTensors
    rots_so3 = []
    for i in range(n_imu):
        euler = rots_imu[i].as_euler('xyz', degrees=False)
        so3_i = pp.euler2SO3(torch.from_numpy(euler).float())  # (4,) wxyz
        rots_so3.append(so3_i)
    rots_so3 = torch.stack(rots_so3, dim=0)  # (N, 4)

    return (
        {
            "acc": torch.tensor(acc_imu, dtype=torch.float32),
            "gyro": torch.tensor(gyro_imu, dtype=torch.float32),
        },
        {
            "pos": torch.tensor(pos_imu, dtype=torch.float32),
            "vel": torch.tensor(vel_w, dtype=torch.float32),
            "rot": rots_so3,
        },
    )
