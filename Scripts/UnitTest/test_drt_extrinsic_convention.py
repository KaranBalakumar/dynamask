import types

import pypose as pp
import torch

from Odometry.MACVO import MACVO


def _make_frame(stereo_t_bs: torch.Tensor, imu_t_bs: pp.LieTensor):
    stereo = types.SimpleNamespace(
        T_BS=pp.from_matrix(stereo_t_bs.unsqueeze(0), pp.SE3_type),
    )
    imu = types.SimpleNamespace(T_BS=imu_t_bs)
    return types.SimpleNamespace(stereo=stereo, imu=imu)


def test_resolve_drt_extrinsic_undoes_edn_ned_when_imu_t_bs_empty():
    # EuRoC-style converted extrinsic in stereo stream: raw_T_BS @ NED2EDN.
    raw_t_bs = torch.tensor([
        [0.0148655429818, -0.999880929698, 0.00414029679422, -0.0216401454975],
        [0.999557249008, 0.0149672133247, 0.025715529948, -0.064676986768],
        [-0.0257744366974, 0.00375618835797, 0.999660727178, 0.00981073058949],
        [0.0, 0.0, 0.0, 1.0],
    ], dtype=torch.float64)
    ned2edn = torch.tensor([
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ], dtype=torch.float64)
    stereo_t_bs = raw_t_bs @ ned2edn

    frame = _make_frame(stereo_t_bs, pp.identity_SE3(0))
    r_bc, t_bc, corrected = MACVO._resolve_drt_extrinsic(frame)

    assert corrected is True
    assert torch.allclose(r_bc, raw_t_bs[:3, :3], atol=1e-7)
    assert torch.allclose(t_bc, stereo_t_bs[:3, 3], atol=1e-7)


def test_resolve_drt_extrinsic_keeps_stereo_rotation_when_imu_t_bs_present():
    stereo_t_bs = torch.eye(4, dtype=torch.float64)
    stereo_t_bs[:3, :3] = torch.tensor([
        [0.0, -1.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
    ], dtype=torch.float64)
    stereo_t_bs[:3, 3] = torch.tensor([0.1, -0.2, 0.3], dtype=torch.float64)

    frame = _make_frame(stereo_t_bs, pp.identity_SE3(1))
    r_bc, t_bc, corrected = MACVO._resolve_drt_extrinsic(frame)

    assert corrected is False
    assert torch.allclose(r_bc, stereo_t_bs[:3, :3], atol=1e-7)
    assert torch.allclose(t_bc, stereo_t_bs[:3, 3], atol=1e-7)
