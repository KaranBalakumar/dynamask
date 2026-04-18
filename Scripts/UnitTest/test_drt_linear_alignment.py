import torch
import pypose as pp
import typing as T
from typing import TYPE_CHECKING
from types import SimpleNamespace

from Module.Initialization.DRTLoose.linear_alignment import solve_linear_alignment
from Module.Initialization.DRTLoose.gravity_refine import refine_gravity_on_sphere
from Module.Initialization.DRTLoose import drt_loose as drt_mod
from Module.Initialization.DRTLoose.drt_loose import DRTLooseInitializer
from DataLoader import StereoData, IMUData, StereoInertialFrame

if TYPE_CHECKING:
    from Module import IFrontend
    from Module.Network.AirIMU.encoder import IMUEncoder
else:
    IFrontend = T.Any
    IMUEncoder = T.Any


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


def _make_frame(idx: int, cam_t_ns: int, imu_samples: int) -> StereoInertialFrame:
    imu_times = torch.arange(imu_samples, dtype=torch.int64).view(1, -1, 1) * 1_000_000 + cam_t_ns
    stereo = StereoData(
        T_BS=pp.identity_SE3(1),
        K=torch.eye(3, dtype=torch.float32).unsqueeze(0),
        baseline=torch.tensor([0.1], dtype=torch.float32),
        time_ns=[cam_t_ns],
        height=2,
        width=2,
        imageL=torch.zeros((1, 3, 2, 2), dtype=torch.float32),
        imageR=torch.zeros((1, 3, 2, 2), dtype=torch.float32),
    )
    imu = IMUData(
        T_BS=pp.identity_SE3(1),
        time_ns=imu_times,
        gravity=[9.81],
        acc=torch.zeros((1, imu_samples, 3), dtype=torch.float32),
        gyro=torch.zeros((1, imu_samples, 3), dtype=torch.float32),
    )
    return StereoInertialFrame(
        idx=[idx],
        time_ns=[cam_t_ns],
        gt_pose=pp.identity_SE3(1),
        stereo=stereo,
        imu=imu,
        gt_attitude=None,
    )


def test_drt_initializer_rejects_insufficient_imu_and_emits_no_edges(monkeypatch):
    class _DummyFrontend:
        def estimate_pair(self, *_args, **_kwargs):
            raise AssertionError("_pair_measurements should be monkeypatched")

    class _DummyCorrector:
        def inference(self, *_args, **_kwargs):
            raise AssertionError("IMU corrector must not run for insufficient IMU window")

    class _DummyEncoder:
        def __init__(self):
            self.corrector = _DummyCorrector()

        def preint(self, *_args, **_kwargs):
            raise AssertionError("Preintegration must not run for insufficient IMU window")

    init = DRTLooseInitializer(
        frontend=T.cast(IFrontend, _DummyFrontend()),
        imu_encoder=T.cast(IMUEncoder, _DummyEncoder()),
        cfg=SimpleNamespace(),
    )

    def _fake_pair_measurements(_self, _fi, _fj, _di, _dj):
        b = torch.tensor([[0.0, 0.0, 1.0], [0.1, 0.0, 0.99]], dtype=torch.float32)
        seg = {"gyro": torch.zeros((2, 3), dtype=torch.float32), "dt": torch.tensor([0.01, 0.01], dtype=torch.float32)}
        return b, b, torch.zeros(3), seg, torch.ones(2)

    monkeypatch.setattr(DRTLooseInitializer, "_pair_measurements", _fake_pair_measurements)
    monkeypatch.setattr(
        drt_mod,
        "solve_gyro_bias_lbfgs",
        lambda **_kwargs: SimpleNamespace(success=True, message="", bias_g=torch.zeros(3), final_loss=0.0),
    )

    frames = [
        _make_frame(0, 0, 3),
        _make_frame(1, 100_000_000, 1),
        _make_frame(2, 200_000_000, 3),
    ]
    depths = [SimpleNamespace(depth=torch.ones((1, 1, 2, 2), dtype=torch.float32)) for _ in frames]
    out = init.run(frames, depths)

    assert not out.ok
    assert "insufficient IMU samples" in out.reason
    assert out.imu_pres == []
