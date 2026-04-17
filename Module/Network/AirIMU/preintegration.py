from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


def _skew(v: torch.Tensor) -> torch.Tensor:
    O = torch.zeros_like(v[..., 0])
    vx, vy, vz = v[..., 0], v[..., 1], v[..., 2]
    return torch.stack(
        [
            torch.stack([O, -vz, vy], dim=-1),
            torch.stack([vz, O, -vx], dim=-1),
            torch.stack([-vy, vx, O], dim=-1),
        ],
        dim=-2,
    )


def so3_exp(w: torch.Tensor) -> torch.Tensor:
    theta = torch.linalg.vector_norm(w, dim=-1, keepdim=True).clamp(min=1e-12)
    K = _skew(w / theta)
    I = torch.eye(3, dtype=w.dtype, device=w.device).view(1, 3, 3).repeat(w.shape[0], 1, 1)
    a = torch.sin(theta)[..., None]
    b = (1.0 - torch.cos(theta))[..., None]
    return I + a * K + b * (K @ K)


def so3_log(R: torch.Tensor) -> torch.Tensor:
    tr = R[..., 0, 0] + R[..., 1, 1] + R[..., 2, 2]
    cos_theta = ((tr - 1.0) * 0.5).clamp(-1.0, 1.0)
    theta = torch.acos(cos_theta).unsqueeze(-1)
    denom = (2.0 * torch.sin(theta)).clamp(min=1e-9)
    vee = torch.stack(
        [
            R[..., 2, 1] - R[..., 1, 2],
            R[..., 0, 2] - R[..., 2, 0],
            R[..., 1, 0] - R[..., 0, 1],
        ],
        dim=-1,
    )
    return theta * (vee / denom.squeeze(-1).unsqueeze(-1))


@dataclass
class PreintOut:
    delta_R: torch.Tensor
    delta_v: torch.Tensor
    delta_p: torch.Tensor
    Sigma: torch.Tensor
    dt_total: torch.Tensor
    J_R_bg: torch.Tensor | None
    J_v_bg: torch.Tensor | None
    J_v_ba: torch.Tensor | None
    J_p_bg: torch.Tensor | None
    J_p_ba: torch.Tensor | None
    bias_ref: torch.Tensor


class DifferentiablePreintegrator(nn.Module):
    def __init__(self, jacobian_eps: float = 1e-5) -> None:
        super().__init__()
        self.jacobian_eps = jacobian_eps

    def _integrate_mean(
        self,
        corrected_acc: torch.Tensor,
        corrected_gyro: torch.Tensor,
        dt: torch.Tensor,
        b_g: torch.Tensor,
        b_a: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B, N, _ = corrected_acc.shape
        dt = dt[..., :N]
        if dt.ndim == 3:
            dt = dt.squeeze(-1)
        M = min(N, dt.shape[1])

        R = torch.eye(3, dtype=corrected_acc.dtype, device=corrected_acc.device).view(1, 3, 3).repeat(B, 1, 1)
        v = torch.zeros((B, 3), dtype=corrected_acc.dtype, device=corrected_acc.device)
        p = torch.zeros((B, 3), dtype=corrected_acc.dtype, device=corrected_acc.device)

        for k in range(M):
            dt_k = dt[:, k].unsqueeze(-1)  # [B,1]
            w_k = corrected_gyro[:, k] - b_g
            a_k = corrected_acc[:, k] - b_a

            dR = so3_exp(w_k * dt_k)
            R_prev = R
            v_prev = v
            Ra = torch.bmm(R_prev, a_k.unsqueeze(-1)).squeeze(-1)
            p = p + v_prev * dt_k + 0.5 * Ra * (dt_k ** 2)
            v = v + Ra * dt_k
            R = torch.bmm(R_prev, dR)

        return R, v, p

    def _jacobians_fd(
        self,
        corrected_acc: torch.Tensor,
        corrected_gyro: torch.Tensor,
        dt: torch.Tensor,
        b_g_ref: torch.Tensor,
        b_a_ref: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        B = corrected_acc.shape[0]
        eps = self.jacobian_eps
        J_R_bg = torch.zeros((B, 3, 3), dtype=corrected_acc.dtype, device=corrected_acc.device)
        J_v_bg = torch.zeros((B, 3, 3), dtype=corrected_acc.dtype, device=corrected_acc.device)
        J_v_ba = torch.zeros((B, 3, 3), dtype=corrected_acc.dtype, device=corrected_acc.device)
        J_p_bg = torch.zeros((B, 3, 3), dtype=corrected_acc.dtype, device=corrected_acc.device)
        J_p_ba = torch.zeros((B, 3, 3), dtype=corrected_acc.dtype, device=corrected_acc.device)

        for i in range(3):
            d = torch.zeros_like(b_g_ref)
            d[:, i] = eps
            Rp, vp, pp = self._integrate_mean(corrected_acc, corrected_gyro, dt, b_g_ref + d, b_a_ref)
            Rm, vm, pm = self._integrate_mean(corrected_acc, corrected_gyro, dt, b_g_ref - d, b_a_ref)
            dR = torch.bmm(Rm.transpose(-2, -1), Rp)
            J_R_bg[:, :, i] = so3_log(dR) / (2.0 * eps)
            J_v_bg[:, :, i] = (vp - vm) / (2.0 * eps)
            J_p_bg[:, :, i] = (pp - pm) / (2.0 * eps)

        for i in range(3):
            d = torch.zeros_like(b_a_ref)
            d[:, i] = eps
            _, vp, pp = self._integrate_mean(corrected_acc, corrected_gyro, dt, b_g_ref, b_a_ref + d)
            _, vm, pm = self._integrate_mean(corrected_acc, corrected_gyro, dt, b_g_ref, b_a_ref - d)
            J_v_ba[:, :, i] = (vp - vm) / (2.0 * eps)
            J_p_ba[:, :, i] = (pp - pm) / (2.0 * eps)

        return J_R_bg, J_v_bg, J_v_ba, J_p_bg, J_p_ba

    def forward(
        self,
        corrected_acc: torch.Tensor,
        corrected_gyro: torch.Tensor,
        acc_cov: torch.Tensor,
        gyro_cov: torch.Tensor,
        dt: torch.Tensor,
        bias_ref: torch.Tensor,
        emit_jacobians: bool,
    ) -> PreintOut:
        if dt.ndim == 3:
            dt = dt.squeeze(-1)
        b_g_ref, b_a_ref = bias_ref[:, :3], bias_ref[:, 3:]

        delta_R, delta_v, delta_p = self._integrate_mean(corrected_acc, corrected_gyro, dt, b_g_ref, b_a_ref)

        B, N, _ = corrected_acc.shape
        M = min(N, dt.shape[1])
        rot_var = torch.zeros((B, 3), dtype=corrected_acc.dtype, device=corrected_acc.device)
        vel_var = torch.zeros((B, 3), dtype=corrected_acc.dtype, device=corrected_acc.device)
        pos_var = torch.zeros((B, 3), dtype=corrected_acc.dtype, device=corrected_acc.device)
        dt_total = dt[:, :M].sum(dim=1)

        for k in range(M):
            dt_k = dt[:, k].unsqueeze(-1)
            rot_var = rot_var + gyro_cov[:, k] * (dt_k ** 2)
            vel_var = vel_var + acc_cov[:, k] * (dt_k ** 2)
            pos_var = pos_var + acc_cov[:, k] * (0.5 * dt_k ** 2) ** 2 + vel_var * (dt_k ** 2)

        Sigma = torch.zeros((B, 9, 9), dtype=corrected_acc.dtype, device=corrected_acc.device)
        Sigma[:, 0, 0] = rot_var[:, 0] + 1e-10
        Sigma[:, 1, 1] = rot_var[:, 1] + 1e-10
        Sigma[:, 2, 2] = rot_var[:, 2] + 1e-10
        Sigma[:, 3, 3] = vel_var[:, 0] + 1e-10
        Sigma[:, 4, 4] = vel_var[:, 1] + 1e-10
        Sigma[:, 5, 5] = vel_var[:, 2] + 1e-10
        Sigma[:, 6, 6] = pos_var[:, 0] + 1e-10
        Sigma[:, 7, 7] = pos_var[:, 1] + 1e-10
        Sigma[:, 8, 8] = pos_var[:, 2] + 1e-10

        if emit_jacobians:
            J_R_bg, J_v_bg, J_v_ba, J_p_bg, J_p_ba = self._jacobians_fd(
                corrected_acc, corrected_gyro, dt, b_g_ref, b_a_ref
            )
        else:
            J_R_bg = J_v_bg = J_v_ba = J_p_bg = J_p_ba = None

        return PreintOut(
            delta_R=delta_R,
            delta_v=delta_v,
            delta_p=delta_p,
            Sigma=Sigma,
            dt_total=dt_total,
            J_R_bg=J_R_bg,
            J_v_bg=J_v_bg,
            J_v_ba=J_v_ba,
            J_p_bg=J_p_bg,
            J_p_ba=J_p_ba,
            bias_ref=bias_ref,
        )

