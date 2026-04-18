from __future__ import annotations

from dataclasses import dataclass
from typing import Any

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


def so3_right_jacobian(phi: torch.Tensor) -> torch.Tensor:
    """
    SO(3) right Jacobian Jr(phi): Exp(phi + dphi) ≈ Exp(phi) · Exp(Jr(phi) · dphi).
    Jr(phi) = I - (1 - cos θ)/θ² · K + (θ - sin θ)/θ³ · K²  with K = [phi]_×, θ = ||phi||.
    """
    B = phi.shape[0]
    I = torch.eye(3, dtype=phi.dtype, device=phi.device).unsqueeze(0).expand(B, -1, -1)
    theta = torch.linalg.vector_norm(phi, dim=-1, keepdim=True)  # [B, 1]
    K = _skew(phi)
    K2 = torch.bmm(K, K)
    small = theta < 1e-5                                            # [B, 1]
    safe_theta = theta.clamp(min=1e-12)
    coef1_large = (1.0 - torch.cos(safe_theta)) / (safe_theta * safe_theta)
    coef2_large = (safe_theta - torch.sin(safe_theta)) / (safe_theta ** 3)
    # Small-angle Taylor: Jr ≈ I - 0.5 K + (1/6) K²
    coef1 = torch.where(small, torch.full_like(coef1_large, 0.5), coef1_large).view(B, 1, 1)
    coef2 = torch.where(small, torch.full_like(coef2_large, 1.0 / 6.0), coef2_large).view(B, 1, 1)
    return I - coef1 * K + coef2 * K2


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
        logger: Any | None = None,
        log_step: int | None = None,
    ) -> PreintOut:
        if dt.ndim == 3:
            dt = dt.squeeze(-1)
        b_g_ref, b_a_ref = bias_ref[:, :3], bias_ref[:, 3:]

        B, N, _ = corrected_acc.shape
        M = min(N, dt.shape[1])
        device = corrected_acc.device
        dtype = corrected_acc.dtype
        dt_total = dt[:, :M].sum(dim=1)

        # Joint mean + full-covariance propagation (Forster et al., Eq. A.9-A.11):
        #   state error ξ = [δφ, δv, δp] ∈ R^9
        #   Σ_{k+1} = A_k Σ_k A_k^T + B_k Σ_η B_k^T
        # A_k and B_k couple rotation errors into velocity/position via -R_{i,k}·[â]_× · dt.
        R = torch.eye(3, dtype=dtype, device=device).view(1, 3, 3).repeat(B, 1, 1)
        v = torch.zeros((B, 3), dtype=dtype, device=device)
        p = torch.zeros((B, 3), dtype=dtype, device=device)
        Sigma = torch.zeros((B, 9, 9), dtype=dtype, device=device)
        I3 = torch.eye(3, dtype=dtype, device=device).view(1, 3, 3).expand(B, -1, -1)

        for k in range(M):
            dt_k = dt[:, k].view(B, 1, 1)            # [B,1,1]
            dt_k_vec = dt[:, k].unsqueeze(-1)        # [B,1] for mean update
            w_k = corrected_gyro[:, k] - b_g_ref
            a_k = corrected_acc[:, k] - b_a_ref
            phi = w_k * dt_k_vec                     # [B,3]
            dR = so3_exp(phi)
            Jr = so3_right_jacobian(phi)
            Ra_skew = torch.bmm(R, _skew(a_k))       # R_{i,k} · [a_k - b_a]_× : [B,3,3]

            # Build A (9x9)
            A = torch.zeros((B, 9, 9), dtype=dtype, device=device)
            A[:, 0:3, 0:3] = dR.transpose(-2, -1)
            A[:, 3:6, 3:6] = I3
            A[:, 6:9, 6:9] = I3
            A[:, 3:6, 0:3] = -Ra_skew * dt_k
            A[:, 6:9, 0:3] = -0.5 * Ra_skew * (dt_k * dt_k)
            A[:, 6:9, 3:6] = I3 * dt_k

            # Build B_noise (9x6) acting on η = [η_g, η_a]
            Bn = torch.zeros((B, 9, 6), dtype=dtype, device=device)
            Bn[:, 0:3, 0:3] = Jr * dt_k
            Bn[:, 3:6, 3:6] = R * dt_k
            Bn[:, 6:9, 3:6] = 0.5 * R * (dt_k * dt_k)

            Sigma_eta = torch.zeros((B, 6, 6), dtype=dtype, device=device)
            Sigma_eta[:, 0, 0] = gyro_cov[:, k, 0]
            Sigma_eta[:, 1, 1] = gyro_cov[:, k, 1]
            Sigma_eta[:, 2, 2] = gyro_cov[:, k, 2]
            Sigma_eta[:, 3, 3] = acc_cov[:, k, 0]
            Sigma_eta[:, 4, 4] = acc_cov[:, k, 1]
            Sigma_eta[:, 5, 5] = acc_cov[:, k, 2]

            Sigma = (
                torch.bmm(torch.bmm(A, Sigma), A.transpose(-2, -1))
                + torch.bmm(torch.bmm(Bn, Sigma_eta), Bn.transpose(-2, -1))
            )

            Ra = torch.bmm(R, a_k.unsqueeze(-1)).squeeze(-1)
            p = p + v * dt_k_vec + 0.5 * Ra * (dt_k_vec ** 2)
            v = v + Ra * dt_k_vec
            R = torch.bmm(R, dR)

        delta_R, delta_v, delta_p = R, v, p
        Sigma = Sigma + 1e-10 * torch.eye(9, dtype=dtype, device=device).view(1, 9, 9)

        if emit_jacobians:
            J_R_bg, J_v_bg, J_v_ba, J_p_bg, J_p_ba = self._jacobians_fd(
                corrected_acc, corrected_gyro, dt, b_g_ref, b_a_ref
            )
        else:
            J_R_bg = J_v_bg = J_v_ba = J_p_bg = J_p_ba = None

        out = PreintOut(
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
        if logger is not None and log_step is not None:
            self._emit_logging(out, logger, int(log_step))
        return out

    @staticmethod
    def _emit_logging(out: PreintOut, logger: Any, log_step: int) -> None:
        eigvals = torch.linalg.eigvalsh(out.Sigma.double())
        eigmin = float(eigvals.min().item())
        cond = float(torch.linalg.cond(out.Sigma.double()).mean().item())
        rot_angle = torch.linalg.vector_norm(so3_log(out.delta_R.double()), dim=-1)
        j_r_bg = out.J_R_bg if out.J_R_bg is not None else torch.zeros((1, 3, 3), dtype=out.Sigma.dtype, device=out.Sigma.device)
        j_v_ba = out.J_v_ba if out.J_v_ba is not None else torch.zeros((1, 3, 3), dtype=out.Sigma.dtype, device=out.Sigma.device)
        j_p_ba = out.J_p_ba if out.J_p_ba is not None else torch.zeros((1, 3, 3), dtype=out.Sigma.dtype, device=out.Sigma.device)
        logger.log_scalars(
            {
                "frontend.imu.preintegrator.dt_s": float(out.dt_total.mean().item()),
                "frontend.imu.preintegrator.delta_R.angle_rad": float(rot_angle.mean().item()),
                "frontend.imu.preintegrator.delta_v.norm": float(torch.linalg.vector_norm(out.delta_v, dim=-1).mean().item()),
                "frontend.imu.preintegrator.delta_p.norm": float(torch.linalg.vector_norm(out.delta_p, dim=-1).mean().item()),
                "frontend.imu.preintegrator.Sigma.cond": cond,
                "frontend.imu.preintegrator.Sigma.eigmin": eigmin,
                "frontend.imu.preintegrator.Sigma.trace": float(torch.diagonal(out.Sigma, dim1=-2, dim2=-1).sum(dim=-1).mean().item()),
                "frontend.imu.preintegrator.J_R_bg.fro": float(torch.linalg.matrix_norm(j_r_bg, ord="fro").mean().item()),
                "frontend.imu.preintegrator.J_v_ba.fro": float(torch.linalg.matrix_norm(j_v_ba, ord="fro").mean().item()),
                "frontend.imu.preintegrator.J_p_ba.fro": float(torch.linalg.matrix_norm(j_p_ba, ord="fro").mean().item()),
                "diag.frontend.imu.preintegrator.sigma_non_psd": float(1.0 if eigmin <= 0.0 else 0.0),
            },
            log_step,
        )
