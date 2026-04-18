from __future__ import annotations

from dataclasses import dataclass

import torch

from Module.Network.AirIMU.preintegration import so3_exp


def _integrate_rotation(gyro: torch.Tensor, dt: torch.Tensor, bias_g: torch.Tensor) -> torch.Tensor:
    """
    Integrate IMU rotation for one interval.
    gyro: [N, 3], dt: [N] or [N,1], bias_g: [3]
    return: [3, 3]
    """
    if dt.ndim == 2:
        dt = dt.squeeze(-1)
    M = min(gyro.shape[0], dt.shape[0])
    R = torch.eye(3, dtype=gyro.dtype, device=gyro.device)
    for k in range(M):
        phi = (gyro[k] - bias_g) * dt[k]
        dR = so3_exp(phi.unsqueeze(0))[0]
        R = R @ dR
    return R


@dataclass
class GyroBiasSolveResult:
    success: bool
    bias_g: torch.Tensor
    final_loss: float
    message: str


def solve_gyro_bias_lbfgs(
    pair_bearings: list[tuple[torch.Tensor, torch.Tensor]],
    imu_segments: list[dict[str, torch.Tensor]],
    R_bc: torch.Tensor,
    init_bg: torch.Tensor | None = None,
    max_iter: int = 200,
    cauchy_delta: float = 1e-5,
) -> GyroBiasSolveResult:
    """
    Solve DRT-L step-1 gyro bias:
      min_bg sum rho( || (R_bc * dR(bg) * R_bc^T)^T x_i - x_j ||^2 )
    """
    if len(pair_bearings) == 0 or len(pair_bearings) != len(imu_segments):
        return GyroBiasSolveResult(False, torch.zeros(3), float("inf"), "Invalid inputs")

    device = pair_bearings[0][0].device
    dtype = pair_bearings[0][0].dtype
    bg = torch.zeros(3, device=device, dtype=dtype, requires_grad=True) if init_bg is None else init_bg.detach().clone().to(device=device, dtype=dtype).requires_grad_(True)

    R_bc = R_bc.to(device=device, dtype=dtype)
    optimizer = torch.optim.LBFGS(
        [bg],
        max_iter=max_iter,
        tolerance_grad=1e-12,
        tolerance_change=1e-12,
        line_search_fn="strong_wolfe",
    )
    c2 = cauchy_delta * cauchy_delta
    last_loss = torch.tensor(float("inf"), device=device, dtype=dtype)

    def closure():
        nonlocal last_loss
        optimizer.zero_grad()
        loss = torch.zeros((), device=device, dtype=dtype)
        for (x_i, x_j), seg in zip(pair_bearings, imu_segments):
            dR_body = _integrate_rotation(seg["gyro"], seg["dt"], bg)
            # R_bc = R_BS maps camera-frame vectors to body-frame. Given dR_body = R_{B_i B_j},
            # the camera-frame relative rotation is R_{C_i C_j} = R_bc^T @ dR_body @ R_bc.
            dR_cam = R_bc.transpose(-1, -2) @ dR_body @ R_bc
            pred = torch.matmul(dR_cam.transpose(-1, -2), x_i.transpose(0, 1)).transpose(0, 1)
            r2 = ((pred - x_j) ** 2).sum(dim=1)
            loss = loss + (c2 * torch.log1p(r2 / c2)).mean()
        loss.backward()
        last_loss = loss.detach()
        return loss

    try:
        optimizer.step(closure)
    except Exception as exc:
        return GyroBiasSolveResult(False, bg.detach(), float(last_loss.item()), f"LBFGS failed: {exc}")

    if not torch.isfinite(last_loss):
        return GyroBiasSolveResult(False, bg.detach(), float("inf"), "Non-finite objective")
    return GyroBiasSolveResult(True, bg.detach(), float(last_loss.item()), "ok")

