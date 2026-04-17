from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class LinearAlignResult:
    success: bool
    velocity: torch.Tensor
    gravity: torch.Tensor
    message: str


def build_linear_system(
    rotation_body: torch.Tensor,
    position_body: torch.Tensor,
    delta_v: torch.Tensor,
    delta_p: torch.Tensor,
    dt: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Build linear system for unknown [v_0..v_{K-1}, g]:
      v_j - v_i - g*dt = R_i * Δv
      p_j - p_i - v_i*dt - 0.5*g*dt^2 = R_i * Δp
    """
    K = rotation_body.shape[0]
    assert K >= 2
    n_state = 3 * K + 3
    A_rows = []
    b_rows = []
    I3 = torch.eye(3, dtype=rotation_body.dtype, device=rotation_body.device)

    for i in range(K - 1):
        j = i + 1
        dt_ij = dt[i]
        R_i = rotation_body[i]

        row_v = torch.zeros((3, n_state), dtype=rotation_body.dtype, device=rotation_body.device)
        row_v[:, 3 * i : 3 * i + 3] = -I3
        row_v[:, 3 * j : 3 * j + 3] = I3
        row_v[:, 3 * K : 3 * K + 3] = -I3 * dt_ij
        rhs_v = R_i @ delta_v[i]

        row_p = torch.zeros((3, n_state), dtype=rotation_body.dtype, device=rotation_body.device)
        row_p[:, 3 * i : 3 * i + 3] = -I3 * dt_ij
        row_p[:, 3 * K : 3 * K + 3] = -I3 * (0.5 * dt_ij * dt_ij)
        rhs_p = (R_i @ delta_p[i]) - (position_body[j] - position_body[i])

        A_rows.append(row_v)
        A_rows.append(row_p)
        b_rows.append(rhs_v)
        b_rows.append(rhs_p)

    A = torch.cat(A_rows, dim=0)
    b = torch.cat(b_rows, dim=0).unsqueeze(-1)
    return A, b


def solve_linear_alignment(
    rotation_body: torch.Tensor,
    position_body: torch.Tensor,
    delta_v: torch.Tensor,
    delta_p: torch.Tensor,
    dt: torch.Tensor,
    g_mag: float = 9.81,
) -> LinearAlignResult:
    A, b = build_linear_system(rotation_body, position_body, delta_v, delta_p, dt)
    try:
        x = torch.linalg.lstsq(A, b).solution.squeeze(-1)
    except RuntimeError as exc:
        K = rotation_body.shape[0]
        return LinearAlignResult(False, torch.zeros((K, 3), dtype=rotation_body.dtype, device=rotation_body.device), torch.zeros(3, dtype=rotation_body.dtype, device=rotation_body.device), f"lstsq failed: {exc}")

    K = rotation_body.shape[0]
    velocity = x[: 3 * K].view(K, 3)
    g = x[3 * K : 3 * K + 3]
    g_norm = torch.linalg.vector_norm(g).clamp(min=1e-9)
    gravity = g * (g_mag / g_norm)
    return LinearAlignResult(True, velocity, gravity, "ok")

