from __future__ import annotations

import torch
import torch.nn as nn

from .corrector import AirIMUCorrector
from .preintegration import DifferentiablePreintegrator, so3_log


def sigma_repr(Sigma: torch.Tensor, mode: str = "diag") -> torch.Tensor:
    if mode == "diag":
        return torch.diagonal(Sigma, dim1=-2, dim2=-1)
    if mode == "chol":
        L = torch.linalg.cholesky(Sigma)
        tril_i, tril_j = torch.tril_indices(9, 9, offset=0, device=Sigma.device)
        return L[..., tril_i, tril_j]
    raise ValueError(f"Unsupported sigma representation mode: {mode}")


class IMUEncoder(nn.Module):
    def __init__(self, cfg) -> None:
        super().__init__()
        self.cfg = cfg
        self.corrector = AirIMUCorrector.from_ckpt(getattr(cfg, "ckpt_path", None))
        self.preint = DifferentiablePreintegrator(jacobian_eps=getattr(cfg, "jacobian_eps", 1e-5))

        sigma_mode = getattr(cfg, "sigma_repr_mode", "diag")
        imu_dim = 25 if sigma_mode == "diag" else 61
        feat_dim = getattr(cfg, "feature_dim", 64)

        self.head = nn.Sequential(
            nn.Linear(imu_dim, feat_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feat_dim, feat_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, imu_seq: dict[str, torch.Tensor], bias_ref: torch.Tensor) -> tuple[dict, torch.Tensor]:
        corr = self.corrector.inference(imu_seq)
        pre = self.preint(
            corrected_acc=imu_seq["acc"] + corr["correction_acc"],
            corrected_gyro=imu_seq["gyro"] + corr["correction_gyro"],
            acc_cov=corr["cov_state"]["acc_cov"],
            gyro_cov=corr["cov_state"]["gyro_cov"],
            dt=imu_seq["dt"],
            bias_ref=bias_ref,
            emit_jacobians=bool(getattr(self.cfg, "emit_jacobians", False)),
        )

        mode = getattr(self.cfg, "sigma_repr_mode", "diag")
        z_imu = torch.cat(
            [
                so3_log(pre.delta_R),
                pre.delta_v,
                pre.delta_p,
                sigma_repr(pre.Sigma, mode=mode),
                bias_ref,
                pre.dt_total.unsqueeze(-1),
            ],
            dim=-1,
        )
        f_imu = self.head(z_imu)
        return {"preint": pre, "z_imu": z_imu, "corr": corr}, f_imu
