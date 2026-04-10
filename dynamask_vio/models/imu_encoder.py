"""IMU Encoder: AirIMU Corrector → Differentiable Preintegrator → Feature MLP.

Data flow:
    Raw IMU ──► AirIMUCorrector (frozen) ──► corrected IMU ──► Preintegrator
             └──► bias corrections + covariances ──────────────────────────────┘
"""

from __future__ import annotations

from contextlib import nullcontext

import torch
import torch.nn as nn

from .airimu_corrector import AirIMUCorrector, load_airimu_weights
from .preintegration import DifferentiablePreintegrator, so3_log_map


class FeatureMLP(nn.Module):
    """Maps preintegrated motion + covariance diagonal to FiLM features."""

    def __init__(self, imu_feature_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(18, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, imu_feature_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class IMUEncoder(nn.Module):
    """Full IMU encoder pipeline with frozen AirIMU correction."""

    def __init__(
        self,
        *,
        imu_feature_dim: int = 128,
        airimu_interval: int = 9,
        airimu_weights_path: str | None = None,
        freeze_airimu: bool = True,
    ):
        super().__init__()
        self.airimu_corrector = AirIMUCorrector(interval=airimu_interval)
        if airimu_weights_path:
            load_airimu_weights(self.airimu_corrector, airimu_weights_path, freeze=freeze_airimu)
        elif freeze_airimu:
            for p in self.airimu_corrector.parameters():
                p.requires_grad_(False)
            self.airimu_corrector.eval()

        self.preintegrator = DifferentiablePreintegrator()
        self.feature_mlp = FeatureMLP(imu_feature_dim)

    def load_airimu_checkpoint(self, ckpt_path: str, *, freeze: bool = True) -> None:
        load_airimu_weights(self.airimu_corrector, ckpt_path, freeze=freeze)

    def forward(self, imu_window: torch.Tensor, imu_mask: torch.Tensor) -> dict:
        B, N, _ = imu_window.shape

        # imu_window format: [dt_rel, ax, ay, az, gx, gy, gz]
        timestamps_rel = imu_window[:, :, 0:1]
        imu_raw = imu_window[:, :, 1:7]

        dt = torch.zeros(B, N, 1, device=imu_window.device, dtype=imu_window.dtype)
        if N > 1:
            dt[:, :-1, 0] = timestamps_rel[:, 1:, 0] - timestamps_rel[:, :-1, 0]
            dt[:, -1, 0] = dt[:, -2, 0]
        else:
            dt[:, 0, 0] = 0.005
        dt = dt.clamp(min=1e-6, max=0.1)

        accel_raw = imu_raw[:, :, 0:3].float()
        gyro_raw = imu_raw[:, :, 3:6].float()

        # AirIMU additive convention: corrected = raw + correction
        delta_ba, delta_bg, sigma2_a, sigma2_g = self.airimu_corrector(
            accel_raw, gyro_raw, imu_mask
        )
        delta_ba = torch.nan_to_num(delta_ba, nan=0.0, posinf=0.0, neginf=0.0).clamp(min=-5.0, max=5.0)
        delta_bg = torch.nan_to_num(delta_bg, nan=0.0, posinf=0.0, neginf=0.0).clamp(min=-2.0, max=2.0)
        sigma2_a = torch.nan_to_num(sigma2_a, nan=1e-4, posinf=1.0, neginf=1e-6).clamp(min=1e-6, max=1e2)
        sigma2_g = torch.nan_to_num(sigma2_g, nan=1e-4, posinf=1.0, neginf=1e-6).clamp(min=1e-6, max=1e2)

        accel_corrected = (accel_raw + delta_ba.float()).clamp(min=-500.0, max=500.0)
        gyro_corrected = (gyro_raw + delta_bg.float()).clamp(min=-200.0, max=200.0)

        if imu_window.device.type == "cuda":
            amp_off_ctx = torch.autocast(device_type="cuda", enabled=False)
        else:
            amp_off_ctx = nullcontext()

        with amp_off_ctx:
            preint_out = self.preintegrator(
                gyro_corrected.float(),
                accel_corrected.float(),
                dt.float(),
                sigma2_g.float(),
                sigma2_a.float(),
                imu_mask,
            )

            delta_R = preint_out["delta_R"]
            delta_v = preint_out["delta_v"]
            delta_p = preint_out["delta_p"]
            Sigma = preint_out["Sigma"]

            log_R = so3_log_map(delta_R)
            sigma_diag = torch.diagonal(Sigma, dim1=-2, dim2=-1)
            mlp_input = torch.cat([log_R, delta_v, delta_p, sigma_diag], dim=-1)
            f_imu = self.feature_mlp(mlp_input)

        return {
            "f_imu": f_imu,
            "delta_bg": delta_bg,
            "delta_ba": delta_ba,
            "sigma2_g": sigma2_g,
            "sigma2_a": sigma2_a,
            "delta_R": delta_R,
            "delta_v": delta_v,
            "delta_p": delta_p,
            "Sigma_preint": Sigma,
        }
