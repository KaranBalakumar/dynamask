import os
import sys
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import pypose as pp

from DataLoader.Interface import AttitudeData

_AIRIO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "Air-IO"))
_AIRIO_EKF_DIR = os.path.join(_AIRIO_ROOT, "EKF")

for p in [_AIRIO_EKF_DIR, _AIRIO_ROOT]:
    if p not in sys.path:
        sys.path.insert(0, p)

from IMUstate import IMUstate
from ekf import IMUEKF
from model.code import CodeNetMotionwithRot

class VelocityEKFDynamics(IMUstate):
    """15-D IMU dynamics for the Velocity EKF."""
    def __init__(self, gravity=9.8107):
        super().__init__()
        self.gravity = torch.tensor([0.0, 0.0, gravity], dtype=torch.float64)

    def state_transition(self, state, input, dt, t=None):
        init_rot = pp.so3(state[..., :3]).Exp()
        bg, ba = input[..., 6:9], input[..., 9:12]

        # Apply biases
        w = input[..., 0:3] - bg
        a = input[..., 3:6] - init_rot.Inv() @ self.gravity.double() - ba

        # Preintegrate kinematics
        Dr = pp.so3(w * dt).Exp()
        Dv = Dr @ a * dt
        Dp = Dv * dt + Dr @ a * 0.5 * dt**2

        # Update absolute state parameters
        R = (init_rot @ Dr).Log()
        V = state[..., 3:6] + init_rot @ Dv
        P = state[..., 6:9] + state[..., 3:6] * dt + init_rot @ Dp

        return torch.cat([R, V, P, bg, ba], dim=-1).tensor()

    def observation(self, state, input, dt, t=None):
        # Observation model: body-frame velocity (v_B)
        nstate = self.state_transition(state, input, dt)
        rot = pp.so3(nstate[..., :3]).Exp()
        return rot.Inv() @ nstate[..., 3:6]

@dataclass
class IMUSample:
    f_imu: torch.Tensor          # (1, 128)   global feature for FiLM (§3.6.1)
    imu_tokens: torch.Tensor     # (1, 7, 128) semantic IMU tokens for CLIP-adapter cross-attn (§3.6.3)
    z_raw: torch.Tensor          # (34,)      raw 34-D IMU state vector (§2.3)
    state: torch.Tensor          # (15,)      EKF state [R, V, P, b_g, b_a]
    P_diag: torch.Tensor         # (15,)      EKF covariance diagonal
    airio_vel: torch.Tensor      # (3,)       Air-IO body-frame velocity
    airio_cov: Optional[torch.Tensor]  # (3,) Air-IO body-frame velocity diag covariance

class IMUContext(nn.Module):
    """Real-time wrapper fusing Velocity EKF and AirIO over temporally aligned windows."""
    def __init__(self, airio_cfg, airio_ckpt=None, gravity=9.8107, bias_noise=1e-12, input_scale=1e2, obs_scale=1e-1):
        super().__init__()
        self.gravity_val = gravity
        self.bias_noise = bias_noise
        self.input_scale = input_scale
        self.obs_scale = obs_scale
        self.register_buffer("gravity_world", torch.tensor([0.0, 0.0, gravity], dtype=torch.float64))

        # Setup Air-IO Model
        self.airio_net = CodeNetMotionwithRot(airio_cfg).double()
        if airio_ckpt is not None:
            ckpt = torch.load(airio_ckpt, map_location="cpu", weights_only=True)
            self.airio_net.load_state_dict(ckpt.get("model_state_dict", ckpt))
            
        self.airio_net.eval()
        self._has_airio = airio_ckpt is not None
        for p in self.airio_net.parameters():
            p.requires_grad = False

        # Setup 15-State PyPose EKF
        self.ekf = IMUEKF(VelocityEKFDynamics(gravity).double(), 
                          Q=torch.eye(12, dtype=torch.float64) * 0.01,
                          R=torch.eye(3, dtype=torch.float64) * 0.01).double()

        # Global 34-D → 128 feature for FiLM (§3.6.1 / §6.4)
        self.feature_mlp = nn.Sequential(
            nn.Linear(34, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 128),
        )

        # Per-slot token projections for the CLIP-adapter cross-attention (§2.3, §3.6.2).
        # Order here defines the token order in imu_tokens: (dR, dv, dp, g, bias, cov, dt).
        self.token_projs = nn.ModuleDict({
            "dR":   nn.Linear(3, 128),
            "dv":   nn.Linear(3, 128),
            "dp":   nn.Linear(3, 128),
            "g":    nn.Linear(3, 128),
            "bias": nn.Linear(6, 128),
            "cov":  nn.Linear(9, 128),
            "dt":   nn.Linear(1, 128),
        })
        self._token_order = ("dR", "dv", "dp", "g", "bias", "cov", "dt")

        self._state = None
        self._P = None
        self._prev_cam_state = None

    def reset(self, init_rot, init_vel, init_pos):
        dev = init_vel.device
        s = torch.zeros(15, dtype=torch.float64, device=dev)

        if hasattr(init_rot, "Log"):
            log = init_rot.Log()
            s[:3] = log.tensor() if hasattr(log, "tensor") else log
        else:
            s[:3] = init_rot.double()

        s[3:6] = init_vel.double()
        s[6:9] = init_pos.double()

        self._state = s
        self._P = torch.eye(15, dtype=torch.float64, device=dev)
        self._prev_cam_state = s.clone()

    def push_biases(self, b_g, b_a):
        """Syncs dynamically updated PGO limits back downward to EKF biases."""
        self._state[9:12] = b_g.double()
        self._state[12:15] = b_a.double()
        self._P[9:15, 9:15] = torch.eye(6, dtype=torch.float64, device=self._P.device) * 1e-4

    def seed_from_drt(self, drt: "DRTInitResult", P_init: torch.Tensor | None = None) -> None:
        """Seed EKF from DRT-loose init result (richer than reset).

        Sets (R0, v0, p0, b_g, b_a) from drt, overwrites gravity_world
        with drt.g_W (which need not be axis-aligned), and initialises
        _P from P_init (or drt.P_init if not supplied).
        Also primes _prev_cam_state so the first step() sees zero dt.
        """
        assert drt.success, "seed_from_drt called with failed DRTInitResult"
        dev = drt.v0.device if drt.v0 is not None else torch.device("cpu")

        s = torch.zeros(15, dtype=torch.float64, device=dev)

        # R0 is (3,3) rotation matrix — store as so3 log in state[0:3]
        R0_so3 = pp.mat2SO3(drt.R0.unsqueeze(0).double())
        log_R0 = R0_so3.Log()
        s[:3] = log_R0.tensor().squeeze(0) if hasattr(log_R0, "tensor") else log_R0.squeeze(0)

        s[3:6]   = drt.v0.to(dtype=torch.float64, device=dev)
        s[6:9]   = drt.p0.to(dtype=torch.float64, device=dev)
        s[9:12]  = drt.b_g.to(dtype=torch.float64, device=dev)
        s[12:15] = drt.b_a.to(dtype=torch.float64, device=dev)

        self._state = s
        cov = P_init if P_init is not None else drt.P_init
        self._P = cov.to(dtype=torch.float64, device=dev) if cov is not None else torch.eye(15, dtype=torch.float64, device=dev)

        # Overwrite gravity_world with DRT-solved gravity (may differ from axis-aligned default)
        self.gravity_world = drt.g_W.to(dtype=torch.float64, device=dev)

        # Prime prev_cam_state so first step() delta starts from DRT state
        self._prev_cam_state = s.clone()

    def seed_from_gt(self, att: AttitudeData, gt_pose: torch.Tensor | None = None) -> None:
        """Seed EKF from ground-truth attitude at the start of a frame pair.

        Used for training on datasets (TartanAir v2) that provide GT
        attitude (init_rot, init_vel, init_pos) and optionally GT pose.
        The EKF is initialised at the *end* of frame k so that step()
        over [t_k, t_{k+1}] produces deltas from a physically correct
        starting point.

        Args:
            att:      ``AttitudeData`` from the dataloader (batched, B=1).
            gt_pose:  optional (1, 7) SE3 LieTensor for gravity alignment.
        """
        assert att.init_rot.shape[0] == 1, "seed_from_gt requires batch_size=1"
        dev = att.init_rot.device

        s = torch.zeros(15, dtype=torch.float64, device=dev)

        # Rotation: SO3 LieTensor (1, 1, 4) → so3 log (3,)
        R0 = att.init_rot.squeeze(0).squeeze(0)          # (4,) SO3 quat
        s[:3] = R0.Log().tensor().double()                # (3,) so3 log

        # Velocity: body-frame velocity (1, 1, 3) → (3,)
        s[3:6] = att.init_vel.squeeze(0).squeeze(0).double()

        # Position: global position (1, 1, 3) → (3,)
        s[6:9] = att.init_pos.squeeze(0).squeeze(0).double()

        # Biases: zero initial (no DRT init in training)
        s[9:12] = torch.zeros(3, dtype=torch.float64, device=dev)
        s[12:15] = torch.zeros(3, dtype=torch.float64, device=dev)

        self._state = s
        self._P = torch.eye(15, dtype=torch.float64, device=dev)

        # Align gravity to GT pose if provided, else use default world gravity
        if gt_pose is not None:
            T_WC = pp.SE3(gt_pose.squeeze(0).double()).matrix()
            R_WC = T_WC[:3, :3]
            new_g = R_WC.T @ torch.tensor([0., 0., self.gravity_val],
                                          dtype=torch.float64, device=dev)
            self.gravity_world.data.copy_(new_g)
        # else: keep the default [0, 0, G] from __init__

        self._prev_cam_state = s.clone()

    def step(self, corrected_imu, raw_imu):
        """Core execution loop processing one camera temporal window."""
        assert self._state is not None
        prev_state = self._prev_cam_state.clone()
        dt_total = 0.0
        ekf_rotations = []

        # Phase 1: EKF Propagate (tick-by-tick physics update)
        for tick in corrected_imu:
            dt = self._to_f64(tick["dt"])
            dt_total += dt.item()
            Q = self._make_Q(tick.get("gyro_cov"), tick.get("acc_cov"))
            inp = torch.cat([tick["gyro"].double(), tick["acc"].double(), self._state[9:12], self._state[12:15]])
            self._state, self._P = self.ekf.state_propogate(state=self._state, input=inp, P=self._P, dt=dt, Q=Q)
            ekf_rotations.append(self._state[:3].clone())

        # Phase 2: Air-IO Vectorized Inference (skipped without checkpoint)
        airio_vel, airio_cov = None, None
        if self._has_airio:
            airio_vel, airio_cov = self._run_airio(raw_imu, ekf_rotations)

        # Phase 3: EKF Velocity Update (skipped when AirIO unavailable)
        if airio_vel is not None:
            self._apply_velocity_update(airio_vel, airio_cov)

        # Phase 4: Final Feature Encoding
        z, slots = self._build_z_and_slots(prev_state, self._state, self._P, airio_vel, airio_cov, dt_total)
        self._prev_cam_state = self._state.clone()

        # Move to the same device as the trainable MLP weights
        mlp_device = next(self.feature_mlp.parameters()).device

        # FiLM input: global 34-D feature, (1, 128)
        z_f = z.float().unsqueeze(0).to(mlp_device)
        f_imu = self.feature_mlp(z_f)

        # CLIP-adapter tokens: 7 semantic slots, (1, 7, 128)
        token_list = []
        for name in self._token_order:
            slot = slots[name].float().unsqueeze(0).to(mlp_device)  # (1, D_slot)
            token_list.append(self.token_projs[name](slot))          # (1, 128)
        imu_tokens = torch.stack(token_list, dim=1)          # (1, 7, 128)

        return IMUSample(
            f_imu=f_imu,
            imu_tokens=imu_tokens,
            z_raw=z.float(),
            state=self._state.float(),
            P_diag=torch.diag(self._P).float(),
            airio_vel=airio_vel.float() if airio_vel is not None else torch.zeros(3),
            airio_cov=airio_cov.float() if airio_cov is not None else None,
        )

    @torch.no_grad()
    def _run_airio(self, raw_imu, ekf_rotations):
        """Batches raw IMU and EKF trajectory through deep learning velocity regressor."""
        if not raw_imu:
            return None, None

        acc = torch.stack([t["acc"] for t in raw_imu]).unsqueeze(0).double()
        gyro = torch.stack([t["gyro"] for t in raw_imu]).unsqueeze(0).double()
        rot = torch.stack(ekf_rotations).unsqueeze(0).double()

        out = self.airio_net({"acc": acc, "gyro": gyro}, rot)
        return out["net_vel"][0, -1, :], (out["cov"][0, -1, :] if out["cov"] is not None else None)

    @torch.no_grad()
    def _apply_velocity_update(self, obs_vel, obs_cov):
        """Integrates Air-IO velocity observations back into the EKF prior mathematically."""
        dev = self._state.device
        P = self._P
        I = torch.eye(15, dtype=torch.float64, device=dev)

        # Project current observation prediction
        y_pred = pp.so3(self._state[:3]).Exp().Inv() @ self._state[3:6]
        
        e = obs_vel.double() - y_pred
        C = self._numerical_obs_jacobian()

        r = obs_cov.double() * self.obs_scale if obs_cov is not None else torch.full((3,), self.obs_scale, dtype=torch.float64, device=dev)
        R_noise = torch.diag(r)

        # Kalman Filter?? 
        S = C @ P @ C.T + R_noise
        K = P @ C.T @ torch.linalg.inv(S)

        self._state += (K @ e)

        IKC = I - K @ C
        self._P = IKC @ P @ IKC.T + K @ R_noise @ K.T

    @torch.no_grad()
    def _numerical_obs_jacobian(self, eps=1e-6):
        """Standard centered numerical jacobian."""
        x = self._state.clone()
        C = torch.zeros(3, 15, dtype=torch.float64, device=x.device)

        y0 = pp.so3(x[:3]).Exp().Inv() @ x[3:6]
        for i in range(15):
            x_p = x.clone()
            x_p[i] += eps
            C[:, i] = (pp.so3(x_p[:3]).Exp().Inv() @ x_p[3:6] - y0) / eps
            
        return C

    def _build_z_and_slots(self, prev_state, curr_state, P, airio_vel, airio_cov, dt_total):
        """Assembles the 34-D z_imu_global AND the per-slot tensors used by the CLIP-adapter tokens.

        Returns:
            z      : (34,) float64 concatenated vector in the order (§2.3 table):
                     dR, dv, dp, cov9, vB, sV, bg, ba, dt, gB
            slots  : dict with per-token slot tensors (float64), keys match `_token_order`:
                     {"dR": (3,), "dv": (3,), "dp": (3,), "g": (3,),
                      "bias": (6,), "cov": (9,), "dt": (1,)}
        """
        dev = curr_state.device

        R_k = pp.so3(prev_state[:3]).Exp()
        R_k1 = pp.so3(curr_state[:3]).Exp()

        dR = (R_k.Inv() @ R_k1).Log()
        dR = dR.tensor() if hasattr(dR, "tensor") else dR

        # Forster-form body-frame deltas
        g_W = self.gravity_world.to(dev)

        dv_world = curr_state[3:6] - prev_state[3:6] - g_W * dt_total
        dv = R_k.Inv() @ dv_world
        dv = dv.tensor() if hasattr(dv, "tensor") else dv

        dp_world = curr_state[6:9] - prev_state[6:9] - prev_state[3:6] * dt_total - 0.5 * g_W * (dt_total**2)
        dp = R_k.Inv() @ dp_world
        dp = dp.tensor() if hasattr(dp, "tensor") else dp

        cov9 = torch.diag(P)[:9]

        vB = airio_vel.double() if airio_vel is not None else torch.zeros(3, dtype=torch.float64, device=dev)
        sV = airio_cov.double() if airio_cov is not None else torch.ones(3, dtype=torch.float64, device=dev)

        bg, ba = curr_state[9:12], curr_state[12:15]
        dt = torch.tensor([dt_total], dtype=torch.float64, device=dev)

        gB = R_k1.Inv() @ g_W
        gB = gB.tensor() if hasattr(gB, "tensor") else gB

        z = torch.cat([dR, dv, dp, cov9, vB, sV, bg, ba, dt, gB])

        slots = {
            "dR":   dR,
            "dv":   dv,
            "dp":   dp,
            "g":    gB,
            "bias": torch.cat([bg, ba]),
            "cov":  cov9,
            "dt":   dt,
        }
        return z, slots

    def _make_Q(self, gyro_cov, acc_cov):
        """Scales inputs covariances to robust process noise standardizations."""
        q = torch.full((12,), self.bias_noise, dtype=torch.float64, device=self._state.device)
        if gyro_cov is not None: q[:3] = gyro_cov.double()
        if acc_cov is not None: q[3:6] = acc_cov.double() * self.input_scale
        return torch.diag(q)

    @staticmethod
    def _to_f64(v):
        return v.double().squeeze() if isinstance(v, torch.Tensor) else torch.tensor(v, dtype=torch.float64)
