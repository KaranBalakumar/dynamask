import pypose as pp
import pytest
import torch

from DataLoader import StereoData
from Module.Frontend.Frontend import FlowFormerCovFrontend, IFrontend, StaticConfidence_FlowFormerCovFrontend
from Module.Frontend.Matching import IMatcher
from Module.Frontend.StereoDepth import IStereoDepth
from Module.Map import MatchObs, PointNode, VisualMap
from Module.Optimization.TwoFramePGO.Graphs import Analytic_Reproj_TwoFramePGO, GraphInput, Reproj_TwoFramePGO
from Odometry.MACVO import MACVO


class DummyDynamicHead(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.imu_fusion = "none"
        self.imu_feature_dim = 8
        self.calls = 0
        self.reset_calls = 0

    def reset_state(self) -> None:
        self.reset_calls += 1

    def forward(self, f_ctx, flow, cov, f_imu=None, proxy=None, h_prev=None, depth=None):
        self.calls += 1
        b, _, h, w = f_ctx.shape
        p_static = torch.full((b, 1, h, w), 0.8, dtype=f_ctx.dtype, device=f_ctx.device)
        p_visible = torch.full((b, 1, h, w), 0.5, dtype=f_ctx.dtype, device=f_ctx.device)
        c_eff = p_static * p_visible
        w_map = torch.full((b, 1, h, w), 0.6, dtype=f_ctx.dtype, device=f_ctx.device)
        return {
            "p_static": p_static,
            "p_visible": p_visible,
            "c_eff": c_eff,
            "w": w_map,
            "h_new": torch.zeros((b, 16, h, w), dtype=f_ctx.dtype, device=f_ctx.device),
        }


def _make_stereo(frame_ns: int) -> StereoData:
    h, w = 16, 24
    return StereoData(
        T_BS=pp.identity_SE3(1),
        K=torch.tensor([[[100.0, 0.0, 12.0], [0.0, 100.0, 8.0], [0.0, 0.0, 1.0]]], dtype=torch.float32),
        baseline=torch.tensor([0.2], dtype=torch.float32),
        time_ns=[frame_ns],
        height=h,
        width=w,
        imageL=torch.zeros((1, 3, h, w), dtype=torch.float32),
        imageR=torch.zeros((1, 3, h, w), dtype=torch.float32),
    )


def _match_obs_from_static_conf(static_conf: torch.Tensor) -> MatchObs:
    n = static_conf.shape[0]
    zeros_1 = torch.zeros((n, 1), dtype=torch.float32)
    zeros_2 = torch.zeros((n, 2), dtype=torch.float32)
    zeros_3 = torch.zeros((n, 3), dtype=torch.float32)
    eye_3 = torch.eye(3, dtype=torch.float64).unsqueeze(0).repeat(n, 1, 1)

    return MatchObs(
        index=torch.arange(n, dtype=torch.long),
        data={
            "pixel1_uv": zeros_2,
            "pixel2_uv": zeros_2,
            "pixel1_d": zeros_1,
            "pixel2_d": zeros_1,
            "pixel1_disp": zeros_1,
            "pixel2_disp": zeros_1,
            "pixel1_uv_cov": zeros_3,
            "pixel2_uv_cov": zeros_3,
            "pixel1_d_cov": zeros_1,
            "pixel2_d_cov": zeros_1,
            "pixel1_disp_cov": zeros_1,
            "pixel2_disp_cov": zeros_1,
            "obs1_covTc": eye_3,
            "obs2_covTc": eye_3,
            "static_conf": static_conf.to(dtype=torch.float32),
        },
    )


def _make_graph_input(static_conf: torch.Tensor) -> GraphInput:
    n = static_conf.numel()
    k = torch.tensor([[120.0, 0.0, 0.0], [0.0, 120.0, 0.0], [0.0, 0.0, 1.0]], dtype=torch.float32)

    pts = PointNode(
        index=torch.arange(n, dtype=torch.long),
        data={
            "pos_Tw": torch.tensor([[2.0, 0.2, 0.1], [2.0, -0.1, 0.05]], dtype=torch.float32)[:n],
            "cov_Tw": torch.eye(3, dtype=torch.float64).unsqueeze(0).repeat(n, 1, 1),
            "color": torch.zeros((n, 3), dtype=torch.uint8),
        },
    )

    kp2 = torch.zeros((n, 2), dtype=torch.float32)
    obs = _match_obs_from_static_conf(static_conf.reshape(-1, 1))
    obs.data["pixel2_uv"] = kp2
    obs.data["pixel2_uv_cov"] = torch.tensor([[1.0, 1.0, 0.0], [1.0, 1.0, 0.0]], dtype=torch.float32)[:n]

    return GraphInput(
        frame_idx=torch.tensor([1], dtype=torch.long),
        from_idx=torch.tensor([0], dtype=torch.long),
        init_motion=pp.identity_SE3(1),
        from_pose=pp.identity_SE3(1),
        baseline=torch.tensor([0.2], dtype=torch.float32),
        observations=obs,
        points=pts,
        images_intrinsic=k,
        edges_index=torch.zeros((n,), dtype=torch.long),
        device="cpu",
    )


def test_frontend_static_confidence_contract_and_stream_reset(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_base_estimate_pair(self, frame_t1: StereoData, frame_t2: StereoData):
        h, w = frame_t2.height, frame_t2.width
        self.model.last_context = torch.randn(2, 128, h // 4, w // 4)
        depth = IStereoDepth.Output(
            depth=torch.ones((1, 1, h, w), dtype=torch.float32),
            disparity=torch.ones((1, 1, h, w), dtype=torch.float32),
            cov=torch.ones((1, 1, h, w), dtype=torch.float32),
            disparity_uncertainty=torch.ones((1, 1, h, w), dtype=torch.float32),
        )
        match = IMatcher.Output(
            flow=torch.zeros((1, 2, h, w), dtype=torch.float32),
            cov=torch.ones((1, 3, h, w), dtype=torch.float32),
        )
        return depth, match

    monkeypatch.setattr(FlowFormerCovFrontend, "estimate_pair", fake_base_estimate_pair)

    frontend = StaticConfidence_FlowFormerCovFrontend.__new__(StaticConfidence_FlowFormerCovFrontend)
    frontend.config = type("cfg", (), {"device": "cpu"})()
    frontend.model = type("m", (), {"last_context": None})()
    frontend.dynamic_head = DummyDynamicHead()
    frontend.imu_encoder = None
    frontend._dt_reset_ns = 20_000_000
    frontend._last_frame_ns = None

    frame_t1 = _make_stereo(0)
    frame_t2 = _make_stereo(10_000_000)
    _, match_first = frontend.estimate_pair(frame_t1, frame_t2)

    assert hasattr(match_first, "static_conf")
    assert hasattr(match_first, "static_weight")
    assert hasattr(match_first, "p_static")
    assert hasattr(match_first, "p_visible")
    assert match_first.static_conf.shape == torch.Size([1, frame_t2.height, frame_t2.width])
    assert match_first.static_weight.shape == torch.Size([1, frame_t2.height, frame_t2.width])

    frame_t3 = _make_stereo(100_000_000)
    frontend.estimate_pair(frame_t2, frame_t3)
    assert frontend.dynamic_head.reset_calls == 1


def test_visual_map_match_schema_contains_static_conf_field() -> None:
    visual_map = VisualMap()
    assert "static_conf" in visual_map.match.data
    assert visual_map.match.data["static_conf"].dtype == torch.float32


def test_macvo_stage1_dynamic_gating_drops_low_confidence_observations() -> None:
    macvo = MACVO.__new__(MACVO)
    macvo.device = "cpu"
    macvo.dynamic_gating_enabled = True
    macvo.min_static_conf = 0.5
    macvo.Frontend = IFrontend

    kp0_uv = torch.tensor([[2.0, 2.0], [4.0, 4.0], [6.0, 6.0]], dtype=torch.float32)
    match_out = IMatcher.Output(
        flow=torch.zeros((1, 2, 8, 8), dtype=torch.float32),
        cov=torch.ones((1, 3, 8, 8), dtype=torch.float32),
    )
    conf_map = torch.ones((1, 8, 8), dtype=torch.float32)
    conf_map[0, 2, 2] = 0.1
    conf_map[0, 4, 4] = 0.6
    conf_map[0, 6, 6] = 0.9
    match_out.static_conf = conf_map

    weight_map = torch.ones((1, 8, 8), dtype=torch.float32)
    weight_map[0, 2, 2] = 0.2
    weight_map[0, 4, 4] = 0.7
    weight_map[0, 6, 6] = 0.95
    match_out.static_weight = weight_map

    kp_static_conf, kp_static_weight = macvo._sample_static_confidence(kp0_uv, match_out)
    keep = macvo._stage1_static_keep_mask(kp_static_conf)

    assert keep.tolist() == [False, True, True]
    assert torch.allclose(kp_static_weight.squeeze(-1)[keep], torch.tensor([0.7, 0.95]))


def test_reproj_graph_scales_residuals_and_analytic_jacobian_with_weight_floor() -> None:
    static_conf = torch.tensor([0.0, 0.5], dtype=torch.float32)
    weighted_input = _make_graph_input(static_conf)
    unit_input = _make_graph_input(torch.ones_like(static_conf))

    weighted_graph = Reproj_TwoFramePGO(weighted_input)
    unit_graph = Reproj_TwoFramePGO(unit_input)

    weighted_residual = weighted_graph.forward()
    unit_residual = unit_graph.forward()

    assert weighted_graph.w[0].item() >= weighted_graph.W_EPS
    assert torch.allclose(weighted_graph.w[1], torch.tensor([0.5], dtype=weighted_graph.w.dtype))
    assert torch.allclose(weighted_residual, unit_residual * weighted_graph.w, atol=1.0e-6)

    weighted_analytic = Analytic_Reproj_TwoFramePGO(weighted_input)
    unit_analytic = Analytic_Reproj_TwoFramePGO(unit_input)
    weighted_analytic.forward()
    unit_analytic.forward()

    jw = weighted_analytic.build_jacobian().view(-1, 2, 7)
    ju = unit_analytic.build_jacobian().view(-1, 2, 7)

    assert torch.allclose(jw, ju * weighted_analytic.w.view(-1, 1, 1), atol=1.0e-6)
