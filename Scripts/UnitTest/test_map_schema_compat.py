import torch

from Module.Map import VisualMap, FrameNode, MatchObs, IMUEdgeNode


def test_visual_map_deserialize_backward_compat():
    m = VisualMap()
    m.frames.push(FrameNode.init({
        "pose": torch.tensor([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]], dtype=torch.float32),
        "T_BS": torch.tensor([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]], dtype=torch.float32),
        "vel": torch.zeros((1, 3), dtype=torch.float32),
        "bias_g": torch.zeros((1, 3), dtype=torch.float32),
        "bias_a": torch.zeros((1, 3), dtype=torch.float32),
        "need_interp": torch.tensor([False]),
        "time_ns": torch.tensor([1], dtype=torch.long),
        "K": torch.eye(3).unsqueeze(0),
        "baseline": torch.tensor([0.2], dtype=torch.float32),
    }))
    m.match.push(MatchObs.init({
        "pixel1_uv": torch.tensor([[10.0, 10.0]], dtype=torch.float32),
        "pixel2_uv": torch.tensor([[11.0, 10.0]], dtype=torch.float32),
        "pixel1_d": torch.tensor([[2.0]], dtype=torch.float32),
        "pixel2_d": torch.tensor([[2.0]], dtype=torch.float32),
        "pixel1_disp": torch.tensor([[1.0]], dtype=torch.float32),
        "pixel2_disp": torch.tensor([[1.0]], dtype=torch.float32),
        "pixel1_disp_cov": torch.tensor([[0.1]], dtype=torch.float32),
        "pixel2_disp_cov": torch.tensor([[0.1]], dtype=torch.float32),
        "obs1_covTc": torch.eye(3, dtype=torch.float64).unsqueeze(0),
        "obs2_covTc": torch.eye(3, dtype=torch.float64).unsqueeze(0),
        "pixel1_uv_cov": torch.tensor([[1.0, 1.0, 0.0]], dtype=torch.float32),
        "pixel2_uv_cov": torch.tensor([[1.0, 1.0, 0.0]], dtype=torch.float32),
        "pixel1_d_cov": torch.tensor([[0.1]], dtype=torch.float32),
        "pixel2_d_cov": torch.tensor([[0.1]], dtype=torch.float32),
        "c": torch.tensor([[1.0]], dtype=torch.float32),
    }))

    dump = m.serialize()
    dump.pop("frames/vel", None)
    dump.pop("frames/bias_g", None)
    dump.pop("frames/bias_a", None)
    dump.pop("match/c", None)

    m2 = VisualMap.deserialize(dump)
    assert "vel" in m2.frames.data and "bias_g" in m2.frames.data and "bias_a" in m2.frames.data
    assert "c" in m2.match.data
    assert len(m2.frames) == 1
    assert len(m2.match) == 1


def test_visual_map_get_imu_edge_handles_plain_tensor_fields_after_deserialize():
    m = VisualMap()
    m.imu_edges.push(IMUEdgeNode.init({
        "from_frame": torch.tensor([0], dtype=torch.long),
        "to_frame": torch.tensor([1], dtype=torch.long),
        "delta_R": torch.eye(3, dtype=torch.float64).unsqueeze(0),
        "delta_v": torch.zeros((1, 3), dtype=torch.float64),
        "delta_p": torch.zeros((1, 3), dtype=torch.float64),
        "Sigma": torch.eye(9, dtype=torch.float64).unsqueeze(0),
        "dt": torch.tensor([0.01], dtype=torch.float64),
        "bias_ref": torch.zeros((1, 6), dtype=torch.float64),
        "J_R_bg": torch.zeros((1, 3, 3), dtype=torch.float64),
        "J_v_bg": torch.zeros((1, 3, 3), dtype=torch.float64),
        "J_v_ba": torch.zeros((1, 3, 3), dtype=torch.float64),
        "J_p_bg": torch.zeros((1, 3, 3), dtype=torch.float64),
        "J_p_ba": torch.zeros((1, 3, 3), dtype=torch.float64),
    }))

    m2 = VisualMap.deserialize(m.serialize())
    assert isinstance(m2.imu_edges.data["from_frame"], torch.Tensor)

    edge = m2.get_imu_edge(0, 1)
    assert edge is not None
    assert edge.data["from_frame"].item() == 0
    assert edge.data["to_frame"].item() == 1
