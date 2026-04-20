import os
import sys
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import pypose as pp

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
    f_imu: torch.Tensor
    z_raw: torch.Tensor
    state: torch.Tensor
    P_diag: torch.Tensor
    airio_vel: torch.Tensor
    airio_cov: Optional[torch.Tensor]

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
        for p in self.airio_net.parameters():
            p.requires_grad = False

        # Setup 15-State PyPose EKF
        self.ekf = IMUEKF(VelocityEKFDynamics(gravity).double(), 
                          Q=torch.eye(12, dtype=torch.float64) * 0.01,
                          R=torch.eye(3, dtype=torch.float64) * 0.01).double()

        # MLP embedding map
        self.feature_mlp = nn.Sequential(
            nn.Linear(34, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 128),
        )

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

        # Phase 2: Air-IO Vectorized Inference
        airio_vel, airio_cov = self._run_airio(raw_imu, ekf_rotations)

        # Phase 3: EKF Velocity Update
        if airio_vel is not None:
            self._apply_velocity_update(airio_vel, airio_cov)

        # Phase 4: Final Feature Encoding 
        z = self._build_z(prev_state, self._state, self._P, airio_vel, airio_cov, dt_total)
        self._prev_cam_state = self._state.clone()
        
        return IMUSample(
            f_imu=self.feature_mlp(z.float()),
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

    def _build_z(self, prev_state, curr_state, P, airio_vel, airio_cov, dt_total):
        """Assembles the raw 34-D IMUSample mapping for downstream FiLM formatting."""
        dev = curr_state.device

        R_k = pp.so3(prev_state[:3]).Exp()
        R_k1 = pp.so3(curr_state[:3]).Exp()
        
        dR = (R_k.Inv() @ R_k1).Log()
        dR = dR.tensor() if hasattr(dR, "tensor") else dR
        
        # Forster preintegration formulation (i looked it up, pls check)
        g_W = self.gravity_world.to(dev)
        
        dv_world = curr_state[3:6] - prev_state[3:6] - g_W * dt_total
        dv = R_k.Inv() @ dv_world
        
        dp_world = curr_state[6:9] - prev_state[6:9] - prev_state[3:6] * dt_total - 0.5 * g_W * (dt_total**2)
        dp = R_k.Inv() @ dp_world
        
        cov9 = torch.diag(P)[:9]

        vB = airio_vel.double() if airio_vel is not None else torch.zeros(3, dtype=torch.float64, device=dev)
        sV = airio_cov.double() if airio_cov is not None else torch.ones(3, dtype=torch.float64, device=dev)

        bg, ba = curr_state[9:12], curr_state[12:15]
        dt = torch.tensor([dt_total], dtype=torch.float64, device=dev)

        gB = R_k1.Inv() @ g_W
        gB = gB.tensor() if hasattr(gB, "tensor") else gB

        return torch.cat([dR, dv, dp, cov9, vB, sV, bg, ba, dt, gB])

    def _make_Q(self, gyro_cov, acc_cov):
        """Scales inputs covariances to robust process noise standardizations."""
        q = torch.full((12,), self.bias_noise, dtype=torch.float64, device=self._state.device)
        if gyro_cov is not None: q[:3] = gyro_cov.double()
        if acc_cov is not None: q[3:6] = acc_cov.double() * self.input_scale
        return torch.diag(q)

    @staticmethod
    def _to_f64(v):
        return v.double().squeeze() if isinstance(v, torch.Tensor) else torch.tensor(v, dtype=torch.float64)
