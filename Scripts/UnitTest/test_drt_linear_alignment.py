import torch

from Module.Initialization.DRTLoose.linear_alignment import solve_linear_alignment
from Module.Initialization.DRTLoose.gravity_refine import refine_gravity_on_sphere


def test_linear_alignment_recovers_gravity_direction():
    K = 6
    dt = torch.full((K - 1,), 0.1, dtype=torch.float64)
    R = torch.eye(3, dtype=torch.float64).unsqueeze(0).repeat(K, 1, 1)
    g_gt = torch.tensor([0.0, 0.0, -9.81], dtype=torch.float64)

    v = torch.zeros((K, 3), dtype=torch.float64)
    v[:, 0] = 0.3
    p = torch.zeros((K, 3), dtype=torch.float64)
    for i in range(K - 1):
        p[i + 1] = p[i] + v[i] * dt[i] + 0.5 * g_gt * (dt[i] ** 2)

    d_v = []
    d_p = []
    for i in range(K - 1):
        d_v.append(R[i].transpose(0, 1) @ (v[i + 1] - v[i] - g_gt * dt[i]))
        d_p.append(R[i].transpose(0, 1) @ (p[i + 1] - p[i] - v[i] * dt[i] - 0.5 * g_gt * (dt[i] ** 2)))
    d_v = torch.stack(d_v, dim=0)
    d_p = torch.stack(d_p, dim=0)

    la = solve_linear_alignment(R, p, d_v, d_p, dt, g_mag=9.81)
    assert la.success
    cos_sim = torch.dot(la.gravity, g_gt) / (torch.linalg.vector_norm(la.gravity) * torch.linalg.vector_norm(g_gt))
    assert cos_sim > 0.999

    gr = refine_gravity_on_sphere(R, p, d_v, d_p, dt, la.velocity, la.gravity, g_mag=9.81, iterations=4)
    assert gr.success
    cos_sim_ref = torch.dot(gr.gravity, g_gt) / (torch.linalg.vector_norm(gr.gravity) * torch.linalg.vector_norm(g_gt))
    assert cos_sim_ref > 0.999

