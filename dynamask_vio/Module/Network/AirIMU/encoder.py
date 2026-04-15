import torch
import torch.nn as nn

from .corrector import AirIMUCorrector
from .preintegration import ForsterPreintegrator


def _so3_log_map(rotation: torch.Tensor) -> torch.Tensor:
    trace = rotation[..., 0, 0] + rotation[..., 1, 1] + rotation[..., 2, 2]
    cos_theta = ((trace - 1.0) * 0.5).clamp(min=-1.0, max=1.0)
    theta = torch.acos(cos_theta)

    skew = 0.5 * (rotation - rotation.transpose(-1, -2))
    vee = torch.stack([skew[..., 2, 1], skew[..., 0, 2], skew[..., 1, 0]], dim=-1)
    scale = torch.where(theta > 1e-5, theta / (torch.sin(theta) + 1e-8), torch.ones_like(theta))
    return vee * scale.unsqueeze(-1)


class AirIMUEncoder(nn.Module):
    def __init__(
        self,
        feature_dim: int = 128,
        hidden_dim: int = 64,
        corrector: AirIMUCorrector | None = None,
        preintegrator: ForsterPreintegrator | None = None,
        freeze_corrector: bool = False,
    ):
        super().__init__()
        self.corrector = corrector if corrector is not None else AirIMUCorrector(hidden_dim=hidden_dim)
        self.preintegrator = preintegrator if preintegrator is not None else ForsterPreintegrator()

        self.feature_mlp = nn.Sequential(
            nn.Linear(25, hidden_dim),
            nn.SiLU(inplace=True),
            nn.Linear(hidden_dim, feature_dim),
        )

        if freeze_corrector:
            self.freeze_corrector()

    def freeze_corrector(self) -> None:
        self.corrector.freeze()

    @staticmethod
    def _dt_summary(dt: torch.Tensor | float, batch: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        if isinstance(dt, (float, int)):
            return torch.full((batch, 1), float(dt), dtype=dtype, device=device)

        if dt.ndim == 1:
            if dt.shape[0] == batch:
                return dt.unsqueeze(-1).to(device=device, dtype=dtype)
            return dt.unsqueeze(0).sum(dim=1, keepdim=True).expand(batch, -1).to(device=device, dtype=dtype)
        if dt.ndim == 2:
            return dt.sum(dim=1, keepdim=True).to(device=device, dtype=dtype)
        raise ValueError("dt must be scalar, [T], [B], [B, 1], or [B, T].")

    def forward(self, acc: torch.Tensor, gyro: torch.Tensor, dt: torch.Tensor | float) -> dict[str, torch.Tensor]:
        corrected = self.corrector(acc=acc, gyro=gyro)
        preint = self.preintegrator(
            acc=corrected["acc"],
            gyro=corrected["gyro"],
            dt=dt,
            delta_bg=corrected["delta_bg"],
            delta_ba=corrected["delta_ba"],
        )

        sigma_diag = torch.diagonal(preint["Sigma_preint"], dim1=-2, dim2=-1)
        dt_sum = self._dt_summary(dt, batch=acc.shape[0], device=acc.device, dtype=acc.dtype)
        feature_input = torch.cat(
            [
                _so3_log_map(preint["delta_R"]),
                preint["delta_v"],
                preint["delta_p"],
                sigma_diag,
                corrected["delta_bg"],
                corrected["delta_ba"],
                dt_sum,
            ],
            dim=-1,
        )
        f_imu = self.feature_mlp(feature_input)

        return {
            "f_imu": f_imu,
            "delta_R": preint["delta_R"],
            "delta_v": preint["delta_v"],
            "delta_p": preint["delta_p"],
            "Sigma_preint": preint["Sigma_preint"],
            "delta_bg": corrected["delta_bg"],
            "delta_ba": corrected["delta_ba"],
            "J_R_bg": preint["J_R_bg"],
            "J_v_bg": preint["J_v_bg"],
            "J_v_ba": preint["J_v_ba"],
            "J_p_bg": preint["J_p_bg"],
            "J_p_ba": preint["J_p_ba"],
        }
