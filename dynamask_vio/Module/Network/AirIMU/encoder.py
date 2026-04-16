import torch
import torch.nn as nn
from pathlib import Path
from types import SimpleNamespace

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
    def _cfg_get(config: SimpleNamespace | dict | None, key: str, default):
        if config is None:
            return default
        if isinstance(config, dict):
            return config.get(key, default)
        return getattr(config, key, default)

    @classmethod
    def from_config(cls, config: SimpleNamespace | dict | None) -> "AirIMUEncoder":
        feature_dim = int(cls._cfg_get(config, "feature_dim", 128))
        hidden_dim = int(cls._cfg_get(config, "hidden_dim", 64))
        freeze_corrector = bool(cls._cfg_get(config, "freeze_corrector", True))

        encoder = cls(
            feature_dim=feature_dim,
            hidden_dim=hidden_dim,
            freeze_corrector=False,
        )

        weight_path = cls._cfg_get(config, "airimu_weights", None)
        if isinstance(weight_path, str) and len(weight_path) > 0 and Path(weight_path).exists():
            state = torch.load(weight_path, map_location="cpu", weights_only=True)
            if isinstance(state, dict) and "state_dict" in state:
                state = state["state_dict"]
            if isinstance(state, dict):
                encoder.load_state_dict(state, strict=False)

        if freeze_corrector:
            encoder.freeze_corrector()
        return encoder

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

    @staticmethod
    def _dt_from_time_ns(time_ns: torch.Tensor, steps: int) -> torch.Tensor:
        # time_ns: [B, T] or [B, T, 1]
        ts = time_ns.squeeze(-1).to(dtype=torch.float32)
        if ts.shape[1] <= 1:
            return torch.full((ts.shape[0], steps), 0.01, device=ts.device, dtype=ts.dtype)
        dt = (ts[:, 1:] - ts[:, :-1]).clamp(min=1.0) * 1e-9
        if dt.shape[1] == steps:
            return dt
        # Most windows provide T samples -> T-1 intervals. Extend with last interval.
        if dt.shape[1] == steps - 1:
            return torch.cat([dt, dt[:, -1:]], dim=1)
        mean_dt = dt.mean(dim=1, keepdim=True)
        return mean_dt.expand(-1, steps)

    def _unpack_imu_inputs(
        self,
        imu_or_acc: torch.Tensor | dict[str, torch.Tensor],
        gyro: torch.Tensor | None,
        dt: torch.Tensor | float | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | float]:
        if isinstance(imu_or_acc, dict):
            acc = imu_or_acc["acc"]
            gyro_val = imu_or_acc["gyro"]
            if dt is None:
                dt = imu_or_acc.get("dt", None)
            if dt is None and "time_ns" in imu_or_acc:
                dt = self._dt_from_time_ns(imu_or_acc["time_ns"], acc.shape[1])
            if dt is None:
                dt = 0.01
            return acc, gyro_val, dt

        if gyro is None:
            raise ValueError("gyro tensor must be provided when imu window dict is not used")
        if dt is None:
            dt = 0.01
        return imu_or_acc, gyro, dt

    def forward(
        self,
        acc: torch.Tensor | dict[str, torch.Tensor],
        gyro: torch.Tensor | None = None,
        dt: torch.Tensor | float | None = None,
        imu_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        acc, gyro, dt = self._unpack_imu_inputs(acc, gyro, dt)

        if imu_mask is not None:
            mask = imu_mask
            if mask.ndim == 2:
                mask = mask.unsqueeze(-1)
            mask = mask.to(device=acc.device, dtype=acc.dtype)
            acc = acc * mask
            gyro = gyro * mask

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
