import torch
import pypose as pp

from Module.Map import MatchObs, PointNode, IMUEdgeNode
from Module.Optimization.TwoFramePGO.Graphs import GraphInput, Reproj_TwoFramePGO_IMU


def test_reproj_imu_graph_output_dims():
    K = torch.tensor([[100.0, 0.0, 50.0], [0.0, 100.0, 50.0], [0.0, 0.0, 1.0]])
    obs = MatchObs.init({
        "pixel1_uv": torch.tensor([[50.0, 50.0]], dtype=torch.float32),
        "pixel2_uv": torch.tensor([[50.0, 50.0]], dtype=torch.float32),
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
    })
    pts = PointNode.init({
        "pos_Tw": torch.tensor([[2.0, 0.0, 0.0]], dtype=torch.float32),
        "cov_Tw": torch.eye(3, dtype=torch.float64).unsqueeze(0),
        "color": torch.tensor([[255, 255, 255]], dtype=torch.uint8),
    })
    imu = IMUEdgeNode.init({
        "from_frame": torch.tensor([0], dtype=torch.long),
        "to_frame": torch.tensor([1], dtype=torch.long),
        "delta_R": torch.eye(3, dtype=torch.float64).unsqueeze(0),
        "delta_v": torch.zeros((1, 3), dtype=torch.float64),
        "delta_p": torch.zeros((1, 3), dtype=torch.float64),
        "Sigma": torch.eye(9, dtype=torch.float64).unsqueeze(0),
        "dt": torch.tensor([0.1], dtype=torch.float64),
        "bias_ref": torch.zeros((1, 6), dtype=torch.float64),
        "J_R_bg": torch.zeros((1, 3, 3), dtype=torch.float64),
        "J_v_bg": torch.zeros((1, 3, 3), dtype=torch.float64),
        "J_v_ba": torch.zeros((1, 3, 3), dtype=torch.float64),
        "J_p_bg": torch.zeros((1, 3, 3), dtype=torch.float64),
        "J_p_ba": torch.zeros((1, 3, 3), dtype=torch.float64),
    })

    g = GraphInput(
        frame_idx=torch.tensor([1], dtype=torch.long),
        from_idx=torch.tensor([0], dtype=torch.long),
        init_motion=pp.identity_SE3(1),
        baseline=torch.tensor([0.2]),
        observations=obs,
        points=pts,
        images_intrinsic=K,
        edges_index=torch.tensor([0], dtype=torch.long),
        device="cpu",
        from_pose=pp.identity_SE3(1).tensor(),
        T_BS=pp.identity_SE3(1).tensor(),
        imu_edge=imu,
        init_v_i=torch.zeros(3),
        init_v_j=torch.zeros(3),
        init_bg_i=torch.zeros(3),
        init_bg_j=torch.zeros(3),
        init_ba_i=torch.zeros(3),
        init_ba_j=torch.zeros(3),
        gravity=torch.tensor([0.0, 0.0, -9.81], dtype=torch.float64),
    )

    graph = Reproj_TwoFramePGO_IMU(g).double()
    r = graph.forward()
    assert r.numel() == (2 * len(obs) + 15)
    cov = graph.covariance_array()
    assert isinstance(cov, list)
    assert cov[-1].shape == (15, 15)

