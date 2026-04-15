from __future__ import annotations

import torch
import torch.nn as nn
import pypose as pp


def _skew(v: torch.Tensor) -> torch.Tensor:
    b = v.shape[0]
    x, y, z = v[:, 0:1], v[:, 1:2], v[:, 2:3]
    zero = torch.zeros((b, 1), dtype=v.dtype, device=v.device)
    r0 = torch.cat([zero, -z, y], dim=-1)
    r1 = torch.cat([z, zero, -x], dim=-1)
    r2 = torch.cat([-y, x, zero], dim=-1)
    return torch.stack([r0, r1, r2], dim=1)


def _so3_log(R: torch.Tensor) -> torch.Tensor:
    return pp.mat2SO3(R).Log().tensor()


class DifferentiablePreintegrator(nn.Module):
    def forward(
        self,
        gyro: torch.Tensor,
        accel: torch.Tensor,
        dt: torch.Tensor,
        sigma2_g: torch.Tensor,
        sigma2_a: torch.Tensor,
        mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        gyro = torch.nan_to_num(gyro.float(), nan=0.0, posinf=0.0, neginf=0.0).clamp(-200.0, 200.0)
        accel = torch.nan_to_num(accel.float(), nan=0.0, posinf=0.0, neginf=0.0).clamp(-500.0, 500.0)
        dt = torch.nan_to_num(dt.float(), nan=0.0, posinf=0.0, neginf=0.0).clamp(0.0, 0.1)
        sigma2_g = torch.nan_to_num(sigma2_g.float(), nan=1e-4, posinf=1.0, neginf=1e-6).clamp(1e-6, 1e2)
        sigma2_a = torch.nan_to_num(sigma2_a.float(), nan=1e-4, posinf=1.0, neginf=1e-6).clamp(1e-6, 1e2)
        mask = mask.bool()

        b, n, _ = gyro.shape
        device = gyro.device
        dtype = gyro.dtype
        I3 = torch.eye(3, device=device, dtype=dtype).unsqueeze(0).expand(b, -1, -1)

        delta_R = pp.identity_SO3(b, device=device, dtype=dtype)
        delta_v = torch.zeros((b, 3), device=device, dtype=dtype)
        delta_p = torch.zeros((b, 3), device=device, dtype=dtype)
        Sigma = torch.zeros((b, 9, 9), device=device, dtype=dtype)

        J_R_bg = torch.zeros((b, 3, 3), device=device, dtype=dtype)
        J_v_bg = torch.zeros((b, 3, 3), device=device, dtype=dtype)
        J_v_ba = torch.zeros((b, 3, 3), device=device, dtype=dtype)
        J_p_bg = torch.zeros((b, 3, 3), device=device, dtype=dtype)
        J_p_ba = torch.zeros((b, 3, 3), device=device, dtype=dtype)

        for i in range(n):
            valid = mask[:, i].view(b, 1)
            dt_i = torch.where(valid, dt[:, i, :], torch.zeros_like(dt[:, i, :]))
            gyro_i = torch.where(valid, gyro[:, i, :], torch.zeros_like(gyro[:, i, :]))
            accel_i = torch.where(valid, accel[:, i, :], torch.zeros_like(accel[:, i, :]))

            dt_s = dt_i.squeeze(-1)
            dt_col = dt_i.unsqueeze(-1)
            dt2_col = (dt_i**2).unsqueeze(-1)

            omega = gyro_i * dt_i
            dR = pp.so3(omega).Exp()
            dR_mat = dR.matrix()
            R_mat = delta_R.matrix()

            Ra = torch.bmm(R_mat, accel_i.unsqueeze(-1)).squeeze(-1)
            accel_skew = _skew(accel_i)
            Ra_term = R_mat @ accel_skew

            delta_p_prop = delta_p + delta_v * dt_i + 0.5 * Ra * (dt_i**2)
            delta_v_prop = delta_v + Ra * dt_i
            delta_R_prop = delta_R * dR

            J_R_bg_prop = dR_mat.transpose(-1, -2) @ J_R_bg - I3 * dt_col
            J_v_bg_prop = J_v_bg - (Ra_term @ J_R_bg) * dt_col
            J_v_ba_prop = J_v_ba - R_mat * dt_col
            J_p_bg_prop = J_p_bg + J_v_bg * dt_col - 0.5 * (Ra_term @ J_R_bg) * dt2_col
            J_p_ba_prop = J_p_ba + J_v_ba * dt_col - 0.5 * R_mat * dt2_col

            A_k = torch.zeros((b, 9, 9), device=device, dtype=dtype)
            A_k[:, 0:3, 0:3] = dR_mat.transpose(-1, -2)
            A_k[:, 3:6, 0:3] = -Ra_term * dt_s.view(b, 1, 1)
            A_k[:, 3:6, 3:6] = I3
            A_k[:, 6:9, 0:3] = -0.5 * Ra_term * (dt_s**2).view(b, 1, 1)
            A_k[:, 6:9, 3:6] = I3 * dt_s.view(b, 1, 1)
            A_k[:, 6:9, 6:9] = I3

            B_k = torch.zeros((b, 9, 6), device=device, dtype=dtype)
            B_k[:, 0:3, 0:3] = I3 * dt_s.view(b, 1, 1)
            B_k[:, 3:6, 3:6] = R_mat * dt_s.view(b, 1, 1)
            B_k[:, 6:9, 3:6] = 0.5 * R_mat * (dt_s**2).view(b, 1, 1)

            Q_k = torch.diag_embed(torch.cat([sigma2_g[:, i, :], sigma2_a[:, i, :]], dim=-1))
            Sigma_prop = A_k @ Sigma @ A_k.transpose(-1, -2) + B_k @ Q_k @ B_k.transpose(-1, -2)

            valid_vec = valid.squeeze(-1).view(b, 1, 1)
            valid_mat = valid.squeeze(-1).view(b, 1, 1).expand_as(Sigma)
            delta_p = torch.where(valid, delta_p_prop, delta_p)
            delta_v = torch.where(valid, delta_v_prop, delta_v)
            delta_R = pp.mat2SO3(torch.where(valid_vec, delta_R_prop.matrix(), delta_R.matrix()))
            J_R_bg = torch.where(valid_vec, J_R_bg_prop, J_R_bg)
            J_v_bg = torch.where(valid_vec, J_v_bg_prop, J_v_bg)
            J_v_ba = torch.where(valid_vec, J_v_ba_prop, J_v_ba)
            J_p_bg = torch.where(valid_vec, J_p_bg_prop, J_p_bg)
            J_p_ba = torch.where(valid_vec, J_p_ba_prop, J_p_ba)
            Sigma = torch.where(valid_mat, Sigma_prop, Sigma)

        delta_R_mat = delta_R.matrix()
        Sigma = 0.5 * (Sigma + Sigma.transpose(-1, -2))
        return {
            "delta_R": delta_R_mat,
            "delta_phi": _so3_log(delta_R_mat),
            "delta_v": delta_v,
            "delta_p": delta_p,
            "Sigma_preint": Sigma,
            "J_R_bg": J_R_bg,
            "J_v_bg": J_v_bg,
            "J_v_ba": J_v_ba,
            "J_p_bg": J_p_bg,
            "J_p_ba": J_p_ba,
        }

