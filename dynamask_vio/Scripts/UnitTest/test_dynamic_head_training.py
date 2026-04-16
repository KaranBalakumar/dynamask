from types import SimpleNamespace

import torch

from Train.DynamicHead.calibrate import apply_temperature, fit_temperature
from Train.DynamicHead.loop import SequenceWindowSampler
from Train.DynamicHead.loss import compose_factorized_loss
from Train.DynamicHead.train import verify_hard_freeze_post_backward, verify_hard_freeze_preconditions
from Module.Network.DynamicHead.head import StaticConfidenceHead


def _loss_cfg() -> SimpleNamespace:
    return SimpleNamespace(
        static=SimpleNamespace(enabled=True, weight=1.0, focal_gamma=2.0, tau0=1.0, alpha=1.0, kappa=2.0),
        visible=SimpleNamespace(enabled=True, weight=0.5, tau_cyc_bce=1.5, kappa_cyc=1.0),
        joint=SimpleNamespace(enabled=True, weight=0.2),
        smooth=SimpleNamespace(enabled=True, weight=0.05),
        d_min=0.2,
        d_max=80.0,
    )


def test_compose_factorized_loss_returns_scalar_terms() -> None:
    cfg = _loss_cfg()
    b, h, w = 1, 8, 10

    p_static = torch.full((b, 1, h, w), 0.8)
    p_visible = torch.full((b, 1, h, w), 0.7)
    flow = torch.zeros((b, 2, h, w))
    flow_bwd = torch.zeros((b, 2, h, w))
    depth = torch.full((b, 1, h, w), 5.0)
    image = torch.zeros((b, 3, h, w))
    K = torch.tensor([[[120.0, 0.0, 5.0], [0.0, 120.0, 4.0], [0.0, 0.0, 1.0]]], dtype=torch.float32)
    R_gt = torch.eye(3, dtype=torch.float32).unsqueeze(0)
    t_gt = torch.tensor([[0.05, 0.0, 0.0]], dtype=torch.float32)

    out = compose_factorized_loss(
        p_static=p_static,
        p_visible=p_visible,
        flow=flow,
        flow_bwd=flow_bwd,
        depth=depth,
        image=image,
        K=K,
        R_gt=R_gt,
        t_gt=t_gt,
        cfg=cfg,
    )

    assert set(out.keys()) >= {"loss", "L_static", "L_visible", "L_joint", "L_smooth"}
    assert out["loss"].ndim == 0


def test_sequence_window_sampler_stride_and_bounds() -> None:
    sampler = SequenceWindowSampler(length=10, window_len=4, stride=4)
    windows = list(iter(sampler))
    assert windows == [[0, 1, 2, 3], [4, 5, 6, 7]]


def test_temperature_fit_and_apply_updates_head_buffer() -> None:
    logits = torch.tensor([[[-1.2, -0.1], [0.2, 1.0]]], dtype=torch.float32).unsqueeze(1)
    targets = torch.tensor([[[0.0, 0.0], [1.0, 1.0]]], dtype=torch.float32).unsqueeze(1)
    temperature = fit_temperature(logits=logits, targets=targets)
    assert temperature > 0.0

    head = StaticConfidenceHead()
    apply_temperature(head, temperature)
    assert torch.isclose(head.T_calib, torch.tensor(float(temperature)))


class _DummyFrontend:
    def __init__(self, model_trainable: bool):
        self.model = torch.nn.Linear(4, 4)
        self.imu_encoder = torch.nn.Linear(4, 4)
        self.model.eval()
        self.imu_encoder.eval()
        for param in self.model.parameters():
            param.requires_grad_(model_trainable)
        for param in self.imu_encoder.parameters():
            param.requires_grad_(False)


def test_hard_freeze_preconditions_reject_trainable_backbone() -> None:
    frontend = _DummyFrontend(model_trainable=True)
    try:
        verify_hard_freeze_preconditions(frontend)
        assert False, "Expected hard-freeze assertion for trainable frontend.model."
    except AssertionError as exc:
        assert "frontend.model" in str(exc)


def test_hard_freeze_checks_pass_and_head_receives_gradients() -> None:
    frontend = _DummyFrontend(model_trainable=False)
    verify_hard_freeze_preconditions(frontend)

    head = torch.nn.Linear(4, 1)
    out = head(torch.randn(2, 4))
    out.sum().backward()
    verify_hard_freeze_post_backward(frontend, head)
