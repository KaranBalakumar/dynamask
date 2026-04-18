from __future__ import annotations

from dataclasses import dataclass

import torch

from .linear_alignment import build_linear_system


@dataclass
class GravityRefineResult:
    success: bool
    velocity: torch.Tensor
    gravity: torch.Tensor
    message: str


def _tangent_basis(g_unit: torch.Tensor) -> torch.Tensor:
    z = torch.tensor([0.0, 0.0, 1.0], dtype=g_unit.dtype, device=g_unit.device)
    if torch.abs(torch.dot(g_unit, z)) > 0.9:
        z = torch.tensor([1.0, 0.0, 0.0], dtype=g_unit.dtype, device=g_unit.device)
    e1 = z - torch.dot(z, g_unit) * g_unit
    e1 = e1 / torch.linalg.vector_norm(e1).clamp(min=1e-9)
    e2 = torch.linalg.cross(g_unit, e1)
    e2 = e2 / torch.linalg.vector_norm(e2).clamp(min=1e-9)
    return torch.stack([e1, e2], dim=1)  # [3, 2]


def refine_gravity_on_sphere(
    rotation_body: torch.Tensor,
    position_body: torch.Tensor,
    delta_v: torch.Tensor,
    delta_p: torch.Tensor,
    dt: torch.Tensor,
    init_velocity: torch.Tensor,
    init_gravity: torch.Tensor,
    g_mag: float = 9.81,
    iterations: int = 5,
) -> GravityRefineResult:
    """
    Constrained refinement on S^2 (fixed |g|).
    """
    A, b = build_linear_system(rotation_body, position_body, delta_v, delta_p, dt)
    K = rotation_body.shape[0]

    v = init_velocity.clone()
    g = init_gravity.clone()
    g = g * (g_mag / torch.linalg.vector_norm(g).clamp(min=1e-9))

    Av = A[:, : 3 * K]
    Ag = A[:, 3 * K : 3 * K + 3]

    for _ in range(iterations):
        g_unit = g / torch.linalg.vector_norm(g).clamp(min=1e-9)
        U = _tangent_basis(g_unit)  # [3,2]

        rhs_v = b.squeeze(-1) - Ag @ g
        try:
            v = torch.linalg.lstsq(Av, rhs_v.unsqueeze(-1)).solution.squeeze(-1).view(K, 3)
        except RuntimeError as exc:
            return GravityRefineResult(False, init_velocity, init_gravity, f"velocity solve failed: {exc}")

        r = Av @ v.view(-1) + Ag @ g - b.squeeze(-1)
        J_delta = Ag @ (U * g_mag)  # [M,2]
        try:
            delta = -torch.linalg.lstsq(J_delta, r.unsqueeze(-1)).solution.squeeze(-1)
        except RuntimeError as exc:
            return GravityRefineResult(False, v, g, f"tangent solve failed: {exc}")

        g_new = g + (U * g_mag) @ delta
        g = g_new * (g_mag / torch.linalg.vector_norm(g_new).clamp(min=1e-9))

    return GravityRefineResult(True, v, g, "ok")
