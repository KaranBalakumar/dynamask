import torch

from Module.Network.AirIMU.preintegration import DifferentiablePreintegrator


def test_preintegration_shapes_and_psd():
    B, N = 2, 16
    acc = torch.zeros((B, N, 3), dtype=torch.float64)
    gyro = torch.zeros((B, N, 3), dtype=torch.float64)
    acc_cov = torch.full((B, N, 3), 1e-4, dtype=torch.float64)
    gyro_cov = torch.full((B, N, 3), 1e-5, dtype=torch.float64)
    dt = torch.full((B, N), 0.005, dtype=torch.float64)
    bias_ref = torch.zeros((B, 6), dtype=torch.float64)

    pre = DifferentiablePreintegrator(jacobian_eps=1e-6)
    out = pre(
        corrected_acc=acc,
        corrected_gyro=gyro,
        acc_cov=acc_cov,
        gyro_cov=gyro_cov,
        dt=dt,
        bias_ref=bias_ref,
        emit_jacobians=True,
    )

    assert out.delta_R.shape == (B, 3, 3)
    assert out.delta_v.shape == (B, 3)
    assert out.delta_p.shape == (B, 3)
    assert out.Sigma.shape == (B, 9, 9)
    assert out.J_R_bg is not None and out.J_R_bg.shape == (B, 3, 3)
    assert out.J_v_bg is not None and out.J_v_bg.shape == (B, 3, 3)
    assert out.J_p_ba is not None and out.J_p_ba.shape == (B, 3, 3)

    I = torch.eye(3, dtype=torch.float64)
    assert torch.allclose(out.delta_R, I.unsqueeze(0).repeat(B, 1, 1), atol=1e-8)

    eigs = torch.linalg.eigvalsh(out.Sigma)
    assert torch.all(eigs >= -1e-10)

