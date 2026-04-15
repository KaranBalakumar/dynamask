import torch
import torch.nn as nn


def _skew(v: torch.Tensor) -> torch.Tensor:
    zeros = torch.zeros_like(v[..., :1])
    vx, vy, vz = v.unbind(dim=-1)
    return torch.stack(
        [
            torch.stack([zeros[..., 0], -vz, vy], dim=-1),
            torch.stack([vz, zeros[..., 0], -vx], dim=-1),
            torch.stack([-vy, vx, zeros[..., 0]], dim=-1),
        ],
        dim=-2,
    )


def so3_exp_map(omega: torch.Tensor) -> torch.Tensor:
    theta = torch.linalg.norm(omega, dim=-1, keepdim=True)
    omega_hat = _skew(omega)
    eye = torch.eye(3, dtype=omega.dtype, device=omega.device).expand(omega.shape[0], 3, 3)

    theta_sq = theta ** 2
    sin_term = torch.where(theta > 1e-5, torch.sin(theta) / theta, 1.0 - theta_sq / 6.0)
    cos_term = torch.where(theta > 1e-5, (1.0 - torch.cos(theta)) / theta_sq, 0.5 - theta_sq / 24.0)

    return eye + sin_term.unsqueeze(-1) * omega_hat + cos_term.unsqueeze(-1) * (omega_hat @ omega_hat)


class ForsterPreintegrator(nn.Module):
    def __init__(self):
        super().__init__()

    @staticmethod
    def _normalize_dt(dt: torch.Tensor | float, batch: int, steps: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        if isinstance(dt, (float, int)):
            return torch.full((batch, steps), float(dt), device=device, dtype=dtype)

        if dt.ndim == 1:
            if dt.shape[0] == steps:
                return dt.unsqueeze(0).expand(batch, -1).to(device=device, dtype=dtype)
            if dt.shape[0] == batch:
                return dt.unsqueeze(1).expand(-1, steps).to(device=device, dtype=dtype)
        if dt.ndim == 2:
            if dt.shape[1] == 1:
                return dt.expand(-1, steps).to(device=device, dtype=dtype)
            return dt.to(device=device, dtype=dtype)

        raise ValueError("dt must be scalar, [T], [B], [B, 1], or [B, T].")

    def forward(
        self,
        acc: torch.Tensor,
        gyro: torch.Tensor,
        dt: torch.Tensor | float,
        delta_bg: torch.Tensor | None = None,
        delta_ba: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        batch, steps, _ = acc.shape
        dt_tensor = self._normalize_dt(dt, batch, steps, acc.device, acc.dtype)
        dt_expanded = dt_tensor.unsqueeze(-1)

        delta_theta = torch.sum(gyro * dt_expanded, dim=1)
        delta_R = so3_exp_map(delta_theta)

        velocity = torch.zeros((batch, 3), dtype=acc.dtype, device=acc.device)
        position = torch.zeros_like(velocity)
        for step in range(steps):
            a_t = acc[:, step]
            dt_t = dt_tensor[:, step].unsqueeze(-1)
            position = position + velocity * dt_t + 0.5 * a_t * (dt_t ** 2)
            velocity = velocity + a_t * dt_t
        delta_v = velocity
        delta_p = position

        gyro_var = torch.var(gyro, dim=1, unbiased=False) + 1e-6
        acc_var = torch.var(acc, dim=1, unbiased=False) + 1e-6
        sigma_diag = torch.cat([gyro_var, acc_var, acc_var], dim=-1)
        sigma_preint = torch.diag_embed(sigma_diag)

        eye = torch.eye(3, dtype=acc.dtype, device=acc.device).unsqueeze(0).expand(batch, -1, -1)
        total_dt = dt_tensor.sum(dim=1, keepdim=True).unsqueeze(-1)
        zeros = torch.zeros_like(eye)
        J_R_bg = -eye * total_dt
        J_v_bg = zeros
        J_v_ba = -eye * total_dt
        J_p_bg = zeros
        J_p_ba = -0.5 * eye * (total_dt ** 2)

        return {
            "delta_R": delta_R,
            "delta_v": delta_v,
            "delta_p": delta_p,
            "Sigma_preint": sigma_preint,
            "J_R_bg": J_R_bg,
            "J_v_bg": J_v_bg,
            "J_v_ba": J_v_ba,
            "J_p_bg": J_p_bg,
            "J_p_ba": J_p_ba,
        }
