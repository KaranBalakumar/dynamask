"""
IMU Encoder: Noise Corrector → Differentiable Preintegrator → Feature MLP.

Data flow (CRITICAL — do not rewire):
    Raw IMU ──► Noise Corrector ──► CORRECTED IMU ──► Preintegrator ──► ΔR,Δv,Δp,Σ ──► Feature MLP ──► f_imu
                    │                                       │
                    ├──► δb_g, δb_a  (bias corrections)     ├──► ΔR, Δv, Δp (to VIO)
                    └──► σ²_g, σ²_a  (to VIO)               └──► Σ_preint   (to VIO)

The mask loss backpropagates through FiLM → MLP → Preintegrator → Noise Corrector.
"""

import math
from contextlib import nullcontext
import torch
import torch.nn as nn
import torch.nn.functional as F
import pypose as pp

from .preintegration import DifferentiablePreintegrator, so3_log_map


class NoiseCorrector(nn.Module):
    """Dilated 1D CNN that predicts per-sample bias corrections and noise variances.

    Input:  [B, N, 6]  (ax, ay, az, gx, gy, gz)
    Output: delta_bg [B, N, 3], delta_ba [B, N, 3],
            sigma2_g [B, N, 3], sigma2_a [B, N, 3]
    """

    def __init__(self, hidden: int = 64, dilations: list = (1, 2, 4),
                 initial_variance: float = 1e-4):
        super().__init__()
        layers = []
        in_ch = 6
        for d in dilations:
            layers.extend([
                nn.Conv1d(in_ch, hidden, kernel_size=3, padding="same", dilation=d),
                nn.BatchNorm1d(hidden),
                nn.ReLU(inplace=True),
            ])
            in_ch = hidden
        self.backbone = nn.Sequential(*layers)

        # Bias correction head
        self.bias_head = nn.Conv1d(hidden, 6, kernel_size=1)

        # Variance head (Softplus output ≥ 0)
        self.var_head = nn.Conv1d(hidden, 6, kernel_size=1)

        # Initialise variance head bias so Softplus output ≈ initial_variance.
        # Softplus(x) = ln(1 + e^x).  We want ln(1 + e^b) ≈ v  ⟹  b ≈ ln(e^v − 1).
        # For small v, b ≈ ln(v) is a good enough approximation.
        init_bias = math.log(math.exp(initial_variance) - 1.0) if initial_variance > 1e-6 \
            else math.log(initial_variance)
        nn.init.zeros_(self.var_head.weight)
        nn.init.constant_(self.var_head.bias, init_bias)

        # Zero-init bias head so corrections start at zero
        nn.init.zeros_(self.bias_head.weight)
        nn.init.zeros_(self.bias_head.bias)

    def forward(self, imu_raw: torch.Tensor):
        """
        Args:
            imu_raw: [B, N, 6] — (accel_xyz, gyro_xyz)
        Returns:
            dict with delta_ba, delta_bg, sigma2_a, sigma2_g  all [B, N, 3]
        """
        x = imu_raw.transpose(1, 2)            # [B, 6, N]
        feat = self.backbone(x)                 # [B, hidden, N]

        bias = self.bias_head(feat).transpose(1, 2)  # [B, N, 6]
        var = F.softplus(self.var_head(feat)).transpose(1, 2)  # [B, N, 6]

        delta_ba = bias[:, :, 0:3]
        delta_bg = bias[:, :, 3:6]
        sigma2_a = var[:, :, 0:3]
        sigma2_g = var[:, :, 3:6]

        return {
            "delta_ba": delta_ba,
            "delta_bg": delta_bg,
            "sigma2_a": sigma2_a,
            "sigma2_g": sigma2_g,
        }


class FeatureMLP(nn.Module):
    """Maps preintegrated motion + covariance diagonal → FiLM feature vector.

    Input:  [B, 18] = concat(Log(ΔR)[3], Δv[3], Δp[3], diag(Σ)[9])
    Output: [B, imu_feature_dim]
    """

    def __init__(self, imu_feature_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(18, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, imu_feature_dim),
        )

    def forward(self, x):
        return self.net(x)


class IMUEncoder(nn.Module):
    """Full IMU encoder pipeline.

    Input:  imu_window [B, N, 7]  — (timestamp_delta, ax, ay, az, gx, gy, gz)
            imu_mask   [B, N]     — True for valid samples

    Output dict:
        f_imu      : [B, 128]    — feature for FiLM conditioning
        delta_bg   : [B, N, 3]   — gyro bias corrections
        delta_ba   : [B, N, 3]   — accel bias corrections
        sigma2_g   : [B, N, 3]   — gyro noise variance
        sigma2_a   : [B, N, 3]   — accel noise variance
        delta_R    : [B, 3, 3]   — preintegrated rotation
        delta_v    : [B, 3]      — preintegrated velocity
        delta_p    : [B, 3]      — preintegrated position
        Sigma_preint : [B, 9, 9] — preintegration covariance
    """

    def __init__(self, hidden: int = 64, dilations: list = (1, 2, 4),
                 initial_variance: float = 1e-4, imu_feature_dim: int = 128):
        super().__init__()
        self.noise_corrector = NoiseCorrector(hidden, dilations, initial_variance)
        self.preintegrator = DifferentiablePreintegrator()
        self.feature_mlp = FeatureMLP(imu_feature_dim)

    def forward(self, imu_window: torch.Tensor, imu_mask: torch.Tensor):
        B, N, _ = imu_window.shape

        # Split timestamp and measurements
        timestamps = imu_window[:, :, 0:1]      # [B, N, 1]
        imu_raw = imu_window[:, :, 1:7]         # [B, N, 6] (ax,ay,az,gx,gy,gz)

        # Compute per-sample dt
        # dt[i] = t[i+1] - t[i], last sample dt is repeated from previous
        dt = torch.zeros(B, N, 1, device=imu_window.device, dtype=imu_window.dtype)
        if N > 1:
            dt[:, :-1, 0] = timestamps[:, 1:, 0] - timestamps[:, :-1, 0]
            dt[:, -1, 0] = dt[:, -2, 0]  # repeat last valid dt
        else:
            dt[:, 0, 0] = 0.005  # fallback: 200Hz

        # Clamp dt to reasonable range
        dt = dt.clamp(min=1e-6, max=0.1)

        # Noise correction
        nc_out = self.noise_corrector(imu_raw)
        delta_ba = nc_out["delta_ba"]
        delta_bg = nc_out["delta_bg"]
        sigma2_a = nc_out["sigma2_a"]
        sigma2_g = nc_out["sigma2_g"]

        # Corrected measurements
        accel_raw = imu_raw[:, :, 0:3]
        gyro_raw = imu_raw[:, :, 3:6]
        accel_corrected = accel_raw - delta_ba
        gyro_corrected = gyro_raw - delta_bg

        # PyPose SO3 conversion expects consistent floating dtype. Keep this block
        # out of AMP autocast and in fp32 to avoid Half/Float mismatches.
        if imu_window.device.type == "cuda":
            amp_off_ctx = torch.autocast(device_type="cuda", enabled=False)
        else:
            amp_off_ctx = nullcontext()

        with amp_off_ctx:
            preint_out = self.preintegrator(
                gyro_corrected.float(), accel_corrected.float(), dt.float(),
                sigma2_g.float(), sigma2_a.float(), imu_mask,
            )

            # Build feature vector for FiLM
            delta_R = preint_out["delta_R"]         # [B, 3, 3]
            delta_v = preint_out["delta_v"]         # [B, 3]
            delta_p = preint_out["delta_p"]         # [B, 3]
            Sigma = preint_out["Sigma"]             # [B, 9, 9]

            log_R = so3_log_map(delta_R)            # [B, 3]
            sigma_diag = torch.diagonal(Sigma, dim1=-2, dim2=-1)  # [B, 9]

            mlp_input = torch.cat([log_R, delta_v, delta_p, sigma_diag], dim=-1)  # [B, 18]
            f_imu = self.feature_mlp(mlp_input)     # [B, 128]

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
