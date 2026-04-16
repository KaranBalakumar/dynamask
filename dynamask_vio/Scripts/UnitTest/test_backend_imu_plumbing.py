from types import SimpleNamespace

import pypose as pp
import torch

from Module.Map import FrameNode, MatchObs, PointNode, VisualMap
from Module.Optimization.TwoFramePGO.Graphs import Analytic_ReprojDisp_TwoFramePGO, GraphInput, ReprojDisp_TwoFramePGO
from Module.Optimization.TwoFramePGO.Optimizer import TwoFrame_PGO
from Odometry.MACVO import MACVO


def _make_match_obs(n: int) -> MatchObs:
    zeros_2 = torch.zeros((n, 2), dtype=torch.float32)
    zeros_1 = torch.zeros((n, 1), dtype=torch.float32)
    zeros_3 = torch.zeros((n, 3), dtype=torch.float32)
    eye_3 = torch.eye(3, dtype=torch.float64).unsqueeze(0).repeat(n, 1, 1)
    return MatchObs(
        index=torch.arange(n, dtype=torch.long),
        data={
            "pixel1_uv": zeros_2,
            "pixel2_uv": zeros_2,
            "pixel1_d": torch.ones((n, 1), dtype=torch.float32),
            "pixel2_d": torch.ones((n, 1), dtype=torch.float32),
            "pixel1_disp": torch.ones((n, 1), dtype=torch.float32),
            "pixel2_disp": torch.ones((n, 1), dtype=torch.float32),
            "pixel1_uv_cov": torch.tensor([[1.0, 1.0, 0.0]], dtype=torch.float32).repeat(n, 1),
            "pixel2_uv_cov": torch.tensor([[1.0, 1.0, 0.0]], dtype=torch.float32).repeat(n, 1),
            "pixel1_d_cov": zeros_1 + 0.1,
            "pixel2_d_cov": zeros_1 + 0.1,
            "pixel1_disp_cov": zeros_1 + 0.1,
            "pixel2_disp_cov": zeros_1 + 0.1,
            "obs1_covTc": eye_3,
            "obs2_covTc": eye_3,
            "static_conf": torch.ones((n, 1), dtype=torch.float32),
        },
    )


def _make_imu_payload() -> dict[str, torch.Tensor]:
    eye = torch.eye(3, dtype=torch.float32).unsqueeze(0)
    sigma = torch.eye(9, dtype=torch.float32).unsqueeze(0)
    return {
        "delta_R": eye.clone(),
        "delta_v": torch.zeros((1, 3), dtype=torch.float32),
        "delta_p": torch.tensor([[0.05, 0.0, 0.0]], dtype=torch.float32),
        "Sigma_preint": sigma,
        "J_R_bg": eye.clone(),
        "J_v_bg": eye.clone(),
        "J_v_ba": eye.clone(),
        "J_p_bg": eye.clone(),
        "J_p_ba": eye.clone(),
        "dt": torch.tensor([[0.01]], dtype=torch.float32),
    }


def test_visual_map_imu_factor_roundtrip() -> None:
    gmap = VisualMap()
    payload = _make_imu_payload()
    gmap.push_imu_factor(torch.tensor([0], dtype=torch.long), torch.tensor([1], dtype=torch.long), payload)
    restored = gmap.get_imu_factor(torch.tensor([0], dtype=torch.long), torch.tensor([1], dtype=torch.long))
    assert restored is not None
    assert torch.allclose(restored["delta_p"], payload["delta_p"])
    assert torch.allclose(restored["Sigma_preint"], payload["Sigma_preint"])


def test_reprojdisp_graph_appends_full_imu_state_residual() -> None:
    n = 2
    obs = _make_match_obs(n)
    pts = PointNode(
        index=torch.arange(n, dtype=torch.long),
        data={
            "pos_Tw": torch.tensor([[2.0, 0.0, 0.0], [2.2, 0.1, 0.0]], dtype=torch.float32),
            "cov_Tw": torch.eye(3, dtype=torch.float64).unsqueeze(0).repeat(n, 1, 1),
            "color": torch.zeros((n, 3), dtype=torch.uint8),
        },
    )
    graph_data = GraphInput(
        frame_idx=torch.tensor([1], dtype=torch.long),
        from_idx=torch.tensor([0], dtype=torch.long),
        init_motion=pp.identity_SE3(1),
        from_pose=pp.identity_SE3(1),
        baseline=torch.tensor([0.2], dtype=torch.float32),
        observations=obs,
        points=pts,
        images_intrinsic=torch.tensor([[120.0, 0.0, 0.0], [0.0, 120.0, 0.0], [0.0, 0.0, 1.0]], dtype=torch.float32),
        edges_index=torch.zeros((n,), dtype=torch.long),
        device="cpu",
        imu_factor=_make_imu_payload(),
    )

    graph = ReprojDisp_TwoFramePGO(graph_data)
    residual = graph.forward()
    cov = graph.covariance_array()
    assert residual.shape == torch.Size([n + 5, 3])
    assert cov.shape == torch.Size([n + 5, 3, 3])


def test_analytic_reprojdisp_imu_jacobian_has_full_state_columns() -> None:
    n = 2
    obs = _make_match_obs(n)
    pts = PointNode(
        index=torch.arange(n, dtype=torch.long),
        data={
            "pos_Tw": torch.tensor([[2.0, 0.0, 0.0], [2.2, 0.1, 0.0]], dtype=torch.float32),
            "cov_Tw": torch.eye(3, dtype=torch.float64).unsqueeze(0).repeat(n, 1, 1),
            "color": torch.zeros((n, 3), dtype=torch.uint8),
        },
    )
    graph_data = GraphInput(
        frame_idx=torch.tensor([1], dtype=torch.long),
        from_idx=torch.tensor([0], dtype=torch.long),
        init_motion=pp.identity_SE3(1),
        from_pose=pp.identity_SE3(1),
        baseline=torch.tensor([0.2], dtype=torch.float32),
        observations=obs,
        points=pts,
        images_intrinsic=torch.tensor([[120.0, 0.0, 0.0], [0.0, 120.0, 0.0], [0.0, 0.0, 1.0]], dtype=torch.float32),
        edges_index=torch.zeros((n,), dtype=torch.long),
        device="cpu",
        imu_factor=_make_imu_payload(),
    )

    graph = Analytic_ReprojDisp_TwoFramePGO(graph_data)
    graph.forward()
    jac = graph.build_jacobian()
    assert jac.shape == torch.Size([(n + 5) * 3, 16])


def test_optimizer_get_graph_data_carries_imu_factor() -> None:
    gmap = VisualMap()
    frame0 = FrameNode(
        index=torch.tensor([0], dtype=torch.long),
        data={
            "pose": pp.identity_SE3(1).tensor().float(),
            "T_BS": pp.identity_SE3(1).tensor().float(),
            "vel_w": torch.zeros((1, 3), dtype=torch.float32),
            "bias_g": torch.zeros((1, 3), dtype=torch.float32),
            "bias_a": torch.zeros((1, 3), dtype=torch.float32),
            "need_interp": torch.tensor([False]),
            "time_ns": torch.tensor([0], dtype=torch.long),
            "K": torch.tensor([[[120.0, 0.0, 0.0], [0.0, 120.0, 0.0], [0.0, 0.0, 1.0]]], dtype=torch.float32),
            "baseline": torch.tensor([0.2], dtype=torch.float32),
        },
    )
    frame1 = FrameNode(
        index=torch.tensor([1], dtype=torch.long),
        data={
            "pose": pp.identity_SE3(1).tensor().float(),
            "T_BS": pp.identity_SE3(1).tensor().float(),
            "vel_w": torch.zeros((1, 3), dtype=torch.float32),
            "bias_g": torch.zeros((1, 3), dtype=torch.float32),
            "bias_a": torch.zeros((1, 3), dtype=torch.float32),
            "need_interp": torch.tensor([False]),
            "time_ns": torch.tensor([10_000_000], dtype=torch.long),
            "K": torch.tensor([[[120.0, 0.0, 0.0], [0.0, 120.0, 0.0], [0.0, 0.0, 1.0]]], dtype=torch.float32),
            "baseline": torch.tensor([0.2], dtype=torch.float32),
        },
    )
    f0_idx = gmap.frames.push(frame0)
    f1_idx = gmap.frames.push(frame1)

    obs = _make_match_obs(1)
    pts = PointNode(
        index=torch.tensor([0], dtype=torch.long),
        data={
            "pos_Tw": torch.tensor([[2.0, 0.0, 0.0]], dtype=torch.float32),
            "cov_Tw": torch.eye(3, dtype=torch.float64).unsqueeze(0),
            "color": torch.zeros((1, 3), dtype=torch.uint8),
        },
    )
    pt_idx = gmap.points.push(pts)
    m_idx = gmap.match.push(obs)
    gmap.match2point.set(m_idx, pt_idx)
    gmap.point2match.add(pt_idx, m_idx)
    gmap.match2frame1.set(m_idx, torch.tensor([int(f0_idx.item())], dtype=torch.long))
    gmap.match2frame2.set(m_idx, torch.tensor([int(f1_idx.item())], dtype=torch.long))
    gmap.frame2match.add(f0_idx, torch.tensor([0], dtype=torch.long), torch.tensor([1], dtype=torch.long))
    gmap.frame2match.add(f1_idx, torch.tensor([0], dtype=torch.long), torch.tensor([1], dtype=torch.long))
    gmap.push_imu_factor(f0_idx, f1_idx, _make_imu_payload())

    cfg = SimpleNamespace(graph_type="disp", device="cpu", vectorize=True, parallel=False, autodiff=False)
    optimizer = TwoFrame_PGO(cfg)
    graph_data = optimizer.get_graph_data(gmap, f1_idx)
    assert graph_data.from_pose is not None
    assert graph_data.imu_factor is not None
    assert "delta_p" in graph_data.imu_factor
    assert graph_data.init_vel_w.shape == torch.Size([1, 3])
    assert graph_data.init_bias_g.shape == torch.Size([1, 3])
    assert graph_data.init_bias_a.shape == torch.Size([1, 3])


def test_optimizer_get_graph_data_builds_sliding_window_payload() -> None:
    gmap = VisualMap()
    frames = []
    for i in range(3):
        fidx = gmap.frames.push(FrameNode(
            index=torch.tensor([i], dtype=torch.long),
            data={
                "pose": pp.identity_SE3(1).tensor().float(),
                "T_BS": pp.identity_SE3(1).tensor().float(),
                "vel_w": torch.full((1, 3), float(i), dtype=torch.float32),
                "bias_g": torch.zeros((1, 3), dtype=torch.float32),
                "bias_a": torch.zeros((1, 3), dtype=torch.float32),
                "need_interp": torch.tensor([False]),
                "time_ns": torch.tensor([i * 10_000_000], dtype=torch.long),
                "K": torch.tensor([[[120.0, 0.0, 0.0], [0.0, 120.0, 0.0], [0.0, 0.0, 1.0]]], dtype=torch.float32),
                "baseline": torch.tensor([0.2], dtype=torch.float32),
            },
        ))
        frames.append(fidx)
    gmap.push_imu_factor(frames[0], frames[1], _make_imu_payload())
    gmap.push_imu_factor(frames[1], frames[2], _make_imu_payload())

    obs = _make_match_obs(1)
    pts = PointNode(
        index=torch.tensor([0], dtype=torch.long),
        data={
            "pos_Tw": torch.tensor([[2.0, 0.0, 0.0]], dtype=torch.float32),
            "cov_Tw": torch.eye(3, dtype=torch.float64).unsqueeze(0),
            "color": torch.zeros((1, 3), dtype=torch.uint8),
        },
    )
    pt_idx = gmap.points.push(pts)
    m_idx = gmap.match.push(obs)
    gmap.match2point.set(m_idx, pt_idx)
    gmap.point2match.add(pt_idx, m_idx)
    gmap.match2frame1.set(m_idx, torch.tensor([int(frames[1].item())], dtype=torch.long))
    gmap.match2frame2.set(m_idx, torch.tensor([int(frames[2].item())], dtype=torch.long))
    gmap.frame2match.add(frames[1], torch.tensor([0], dtype=torch.long), torch.tensor([1], dtype=torch.long))
    gmap.frame2match.add(frames[2], torch.tensor([0], dtype=torch.long), torch.tensor([1], dtype=torch.long))

    cfg = SimpleNamespace(
        graph_type="disp",
        device="cpu",
        vectorize=True,
        parallel=False,
        autodiff=True,
        window_size=3,
    )
    optimizer = TwoFrame_PGO(cfg)
    graph_data = optimizer.get_graph_data(gmap, frames[2])
    assert graph_data.window_frame_idx is not None
    assert graph_data.window_frame_idx.shape == torch.Size([3])
    assert graph_data.window_imu_factors is not None
    assert graph_data.window_imu_factors["delta_p"].shape == torch.Size([2, 3])


def test_macvo_imu_payload_adaptive_covariance_inflation() -> None:
    macvo = MACVO.__new__(MACVO)
    macvo.imu_adaptive_enabled = True
    macvo.imu_cov_inflation_base = 1.0
    macvo.imu_cov_inflation_max = 25.0
    macvo.imu_cov_dt_ref = 0.01
    macvo.imu_cov_innovation_ref = 0.2
    macvo.imu_failure_innovation_threshold = 2.0
    macvo.imu_failure_dt_threshold = 0.2

    payload = _make_imu_payload()
    adapted, scale, failed = macvo._adapt_imu_payload(payload, dt=0.08, innovation_norm=1.0)
    assert scale > 1.0
    assert adapted["Sigma_preint"][0, 0, 0] > payload["Sigma_preint"][0, 0, 0]
    assert not failed
