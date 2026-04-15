import torch

from Module.Network.DynamicHead import FiLM, StaticConfidenceHead
from Module.Network.AirIMU import AirIMUEncoder


def _make_head_inputs(batch: int = 2, height: int = 20, width: int = 30):
    f_ctx = torch.randn(batch, 128, height, width)
    flow = torch.randn(batch, 2, height, width)
    cov = torch.randn(batch, 3, height, width)
    proxy = torch.randn(batch, 4, height, width)
    return f_ctx, flow, cov, proxy


def test_static_confidence_head_outputs_factorized_maps():
    torch.manual_seed(0)
    f_ctx, flow, cov, proxy = _make_head_inputs()
    head = StaticConfidenceHead(hidden_dim=64, imu_feature_dim=32, w_eps=1e-3)

    out = head(
        f_ctx=f_ctx,
        flow=flow,
        cov=cov,
        f_imu=torch.randn(2, 32),
        proxy=proxy,
        h_prev=None,
    )

    assert out["p_static"].shape == (2, 1, 20, 30)
    assert out["p_visible"].shape == (2, 1, 20, 30)
    assert out["static_conf"].shape == (2, 1, 20, 30)
    assert out["static_weight"].shape == (2, 1, 20, 30)
    assert out["h_new"].shape == (2, 64, 20, 30)

    torch.testing.assert_close(
        out["static_conf"],
        out["p_static"] * out["p_visible"],
        rtol=1e-5,
        atol=1e-6,
    )
    assert out["static_weight"].min().item() >= 1e-3
    assert out["static_weight"].max().item() <= 1.0


def test_static_confidence_head_supports_recurrent_pathway():
    torch.manual_seed(1)
    f_ctx, flow, cov, proxy = _make_head_inputs(batch=1)
    head = StaticConfidenceHead(hidden_dim=32, imu_feature_dim=16)

    out_t0 = head(
        f_ctx=f_ctx,
        flow=flow,
        cov=cov,
        f_imu=torch.randn(1, 16),
        proxy=proxy,
        h_prev=None,
    )
    out_t1 = head(
        f_ctx=f_ctx,
        flow=flow,
        cov=cov,
        f_imu=torch.randn(1, 16),
        proxy=proxy,
        h_prev=out_t0["h_new"],
    )

    assert out_t0["h_new"].shape == out_t1["h_new"].shape == (1, 32, 20, 30)
    assert not torch.allclose(out_t0["h_new"], out_t1["h_new"])


def test_film_layer_uses_imu_conditioning():
    torch.manual_seed(2)
    film = FiLM(feature_dim=8, cond_dim=4)
    x = torch.randn(1, 8, 6, 5)

    y0 = film(x, torch.zeros(1, 4))
    y1 = film(x, torch.ones(1, 4))

    assert y0.shape == x.shape
    assert y1.shape == x.shape
    assert not torch.allclose(y0, y1)


def test_airimu_encoder_returns_full_preintegration_dict():
    torch.manual_seed(3)
    encoder = AirIMUEncoder(feature_dim=64, hidden_dim=32)

    acc = torch.randn(2, 5, 3)
    gyro = torch.randn(2, 5, 3)
    dt = torch.full((2, 5), 0.01)

    out = encoder(acc=acc, gyro=gyro, dt=dt)

    expected_keys = {
        "f_imu",
        "delta_R",
        "delta_v",
        "delta_p",
        "Sigma_preint",
        "delta_bg",
        "delta_ba",
        "J_R_bg",
        "J_v_bg",
        "J_v_ba",
        "J_p_bg",
        "J_p_ba",
    }
    assert expected_keys.issubset(out.keys())

    assert out["f_imu"].shape == (2, 64)
    assert out["delta_R"].shape == (2, 3, 3)
    assert out["delta_v"].shape == (2, 3)
    assert out["delta_p"].shape == (2, 3)
    assert out["Sigma_preint"].shape == (2, 9, 9)
    assert out["delta_bg"].shape == (2, 3)
    assert out["delta_ba"].shape == (2, 3)
    for key in ["J_R_bg", "J_v_bg", "J_v_ba", "J_p_bg", "J_p_ba"]:
        assert out[key].shape == (2, 3, 3)


def test_airimu_corrector_freeze_semantics_available():
    encoder = AirIMUEncoder(feature_dim=16, hidden_dim=16, freeze_corrector=False)
    assert any(param.requires_grad for param in encoder.corrector.parameters())

    encoder.freeze_corrector()

    assert not encoder.corrector.training
    assert all(not param.requires_grad for param in encoder.corrector.parameters())
