from types import SimpleNamespace

import pypose as pp
import torch

from DataLoader import IMUData
import Odometry.MACVO as macvo_module


def _imu_window_with_samples(n_samples: int) -> IMUData:
    return IMUData(
        T_BS=pp.identity_SE3(1),
        time_ns=torch.arange(n_samples, dtype=torch.long).view(1, n_samples, 1),
        gravity=[-9.81],
        acc=torch.zeros((1, n_samples, 3), dtype=torch.float32),
        gyro=torch.zeros((1, n_samples, 3), dtype=torch.float32),
    )


def test_runtime_keyframe_skips_imu_edge_when_window_has_lt2_samples(monkeypatch):
    imu = _imu_window_with_samples(1)
    macvo = object.__new__(macvo_module.MACVO)
    macvo._imu_since_keyframe = imu
    macvo._imu_encoder = None
    macvo.prev_keyframe = (None, 4, None)
    macvo.init_cfg = SimpleNamespace()
    macvo.debug_logger = None

    pushed_edges: list[tuple] = []
    warnings: list[tuple[str, str]] = []
    macvo._push_imu_edge = lambda *args, **kwargs: pushed_edges.append(args)
    monkeypatch.setattr(macvo_module.Logger, "write", lambda level, msg: warnings.append((level, msg)))

    frame1 = SimpleNamespace(imu=imu)
    macvo._insert_runtime_imu_edge(frame1, torch.tensor([4]), torch.tensor([5]), step=5)

    assert pushed_edges == []
    assert any("Insufficient IMU samples (1)" in msg for level, msg in warnings if level == "warn")


def test_runtime_keyframe_skips_imu_edge_when_window_missing(monkeypatch):
    macvo = object.__new__(macvo_module.MACVO)
    macvo._imu_since_keyframe = None
    macvo._imu_encoder = None
    macvo.prev_keyframe = (None, 7, None)
    macvo.init_cfg = SimpleNamespace()
    macvo.debug_logger = None

    pushed_edges: list[tuple] = []
    warnings: list[tuple[str, str]] = []
    macvo._push_imu_edge = lambda *args, **kwargs: pushed_edges.append(args)
    monkeypatch.setattr(macvo_module.Logger, "write", lambda level, msg: warnings.append((level, msg)))

    frame1 = SimpleNamespace(imu=_imu_window_with_samples(0))
    macvo._insert_runtime_imu_edge(frame1, torch.tensor([7]), torch.tensor([8]), step=8)

    assert pushed_edges == []
    assert any("Missing IMU window" in msg for level, msg in warnings if level == "warn")


def test_runtime_keyframe_inserts_imu_edge_on_boundary_two_samples():
    imu = _imu_window_with_samples(2)
    macvo = object.__new__(macvo_module.MACVO)
    macvo._imu_since_keyframe = imu
    macvo.prev_keyframe = (None, 4, None)
    macvo.init_cfg = SimpleNamespace()
    macvo.debug_logger = None

    class _FakeCorrector:
        @staticmethod
        def inference(inputs):
            return {
                "correction_acc": torch.zeros_like(inputs["acc"]),
                "correction_gyro": torch.zeros_like(inputs["gyro"]),
                "cov_state": {
                    "acc_cov": torch.ones_like(inputs["acc"]),
                    "gyro_cov": torch.ones_like(inputs["gyro"]),
                },
            }

    class _FakeEncoder:
        def __init__(self):
            self.corrector = _FakeCorrector()

        @staticmethod
        def preint(**kwargs):
            return SimpleNamespace(
                delta_R=torch.eye(3).unsqueeze(0),
                delta_v=torch.zeros((1, 3)),
                delta_p=torch.zeros((1, 3)),
                Sigma=torch.eye(9).unsqueeze(0),
                dt_total=torch.tensor([0.01]),
                bias_ref=kwargs["bias_ref"],
                J_R_bg=None,
                J_v_bg=None,
                J_v_ba=None,
                J_p_bg=None,
                J_p_ba=None,
            )

    macvo._imu_encoder = _FakeEncoder()
    macvo.graph = SimpleNamespace(
        frames=SimpleNamespace(
            data={
                "bias_g": torch.zeros((6, 3)),
                "bias_a": torch.zeros((6, 3)),
            }
        )
    )

    pushed_edges: list[tuple] = []
    macvo._push_imu_edge = lambda *args, **kwargs: pushed_edges.append(args)

    frame1 = SimpleNamespace(imu=imu)
    macvo._insert_runtime_imu_edge(frame1, torch.tensor([4]), torch.tensor([5]), step=5)

    assert len(pushed_edges) == 1
    assert pushed_edges[0][0] == 4
    assert pushed_edges[0][1] == 5


def test_append_imu_window_accumulates_without_duplicate_boundary():
    base = _imu_window_with_samples(2)
    window = _imu_window_with_samples(2)
    window.time_ns = torch.tensor([[[1], [2]]], dtype=torch.long)

    merged = macvo_module.MACVO._append_imu_window(base, window)

    assert merged is not None
    assert merged.time_ns.shape[1] == 3
    assert merged.time_ns[..., 0].tolist() == [[0, 1, 2]]
