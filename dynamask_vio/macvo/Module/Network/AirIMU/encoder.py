from __future__ import annotations

from types import SimpleNamespace

import torch
import torch.nn as nn

from .corrector import AirIMUCorrector, load_airimu_weights
from .preintegration import DifferentiablePreintegrator


class FeatureMLP(nn.Module):
    def __init__(self, in_dim: int = 25, hidden_dim: int = 128, out_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class AirIMUEncoder(nn.Module):
    def __init__(
        self,
        airimu_weights: str,
        airimu_interval: int = 9,
        feature_dim: int = 128,
        feature_hidden_dim: int = 128,
    ):
        super().__init__()
        self.corrector = AirIMUCorrector(interval=airimu_interval)
        load_airimu_weights(self.corrector, airimu_weights, freeze=True)
        self.preintegrator = DifferentiablePreintegrator()
        self.feature_mlp = FeatureMLP(in_dim=25, hidden_dim=feature_hidden_dim, out_dim=feature_dim)

    @classmethod
    def from_config(cls, config: SimpleNamespace):
        return cls(
            airimu_weights=str(config.airimu_weights),
            airimu_interval=int(getattr(config, "airimu_interval", 9)),
            feature_dim=int(getattr(config, "feature_dim", 128)),
            feature_hidden_dim=int(getattr(config, "feature_hidden_dim", 128)),
        )

    @staticmethod
    def _extract_dt(imu_window: torch.Tensor, imu_mask: torch.Tensor) -> torch.Tensor:
        t = imu_window[..., 0:1]
        dt = torch.zeros_like(t)
        dt[:, 1:, :] = t[:, 1:, :] - t[:, :-1, :]
        dt = dt.clamp(min=0.0)
        return dt * imu_mask.unsqueeze(-1).to(dtype=dt.dtype)

    def forward(
        self,
        imu_window: torch.Tensor,
        imu_mask: torch.Tensor,
        *,
        share_with_backend: bool = True,
    ) -> dict[str, torch.Tensor]:
        accel = imu_window[..., 1:4]
        gyro = imu_window[..., 4:7]
        dt = self._extract_dt(imu_window, imu_mask)

        with torch.no_grad():
            delta_ba, delta_bg, sigma2_a, sigma2_g = self.corrector(accel, gyro, imu_mask)

        accel_corr = accel + delta_ba
        gyro_corr = gyro + delta_bg
        preint = self.preintegrator(
            gyro=gyro_corr,
            accel=accel_corr,
            dt=dt,
            sigma2_g=sigma2_g,
            sigma2_a=sigma2_a,
            mask=imu_mask,
        )

        sigma_diag = torch.diagonal(preint["Sigma_preint"], dim1=-2, dim2=-1).clamp(min=1e-10)
        dt_sum = (dt.squeeze(-1) * imu_mask.to(dtype=dt.dtype)).sum(dim=1, keepdim=True)
        mlp_in = torch.cat(
            [
                preint["delta_phi"],
                preint["delta_v"],
                preint["delta_p"],
                sigma_diag,
                delta_bg.mean(dim=1),
                delta_ba.mean(dim=1),
                dt_sum,
            ],
            dim=-1,
        )
        f_imu = self.feature_mlp(mlp_in)

        out = {
            "f_imu": f_imu,
            "delta_R": preint["delta_R"],
            "delta_v": preint["delta_v"],
            "delta_p": preint["delta_p"],
            "Sigma_preint": preint["Sigma_preint"],
            "delta_bg": delta_bg.mean(dim=1),
            "delta_ba": delta_ba.mean(dim=1),
        }
        if share_with_backend:
            out.update(
                {
                    "J_R_bg": preint["J_R_bg"],
                    "J_v_bg": preint["J_v_bg"],
                    "J_v_ba": preint["J_v_ba"],
                    "J_p_bg": preint["J_p_bg"],
                    "J_p_ba": preint["J_p_ba"],
                }
            )
        return out

