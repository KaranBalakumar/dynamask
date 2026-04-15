import pytest
import torch
from types import SimpleNamespace

from Module.Frontend.Matching import IMatcher
from Module.Frontend.StereoDepth import IStereoDepth
from Module.Optimization.TwoFramePGO.Optimizer import TwoFrame_PGO
from Utility.Device import (
    canonicalize_torch_device,
    is_supported_device_string,
    normalize_backend_device,
    set_odometry_device_fields,
)


class _DummyMatcher(IMatcher):
    @property
    def provide_cov(self) -> bool:
        return False

    def forward(self, frame_t1, frame_t2):
        raise NotImplementedError


class _DummyDepth(IStereoDepth):
    @property
    def provide_cov(self) -> bool:
        return False

    def estimate(self, frame):
        raise NotImplementedError


def test_rocm_alias_maps_to_cuda_index():
    backend, device = normalize_backend_device("rocm", 1)
    assert backend == "rocm"
    assert device == "cuda:1"


def test_cuda_backend_maps_to_cuda_index():
    backend, device = normalize_backend_device("cuda", 2)
    assert backend == "cuda"
    assert device == "cuda:2"


def test_cpu_backend_maps_to_cpu():
    backend, device = normalize_backend_device("cpu", 3)
    assert backend == "cpu"
    assert device == "cpu"


def test_auto_backend_selects_cuda(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.version, "hip", None, raising=False)
    backend, device = normalize_backend_device("auto", 4)
    assert backend == "cuda"
    assert device == "cuda:4"


def test_auto_backend_selects_rocm_when_hip_available(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.version, "hip", "6.0", raising=False)
    backend, device = normalize_backend_device("auto", 0)
    assert backend == "rocm"
    assert device == "cuda:0"


def test_auto_backend_selects_cpu(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    backend, device = normalize_backend_device("auto", 4)
    assert backend == "cpu"
    assert device == "cpu"


def test_rocm_aliases_are_validated():
    assert is_supported_device_string("rocm")
    assert is_supported_device_string("rocm:1")
    assert is_supported_device_string("hip:2")
    assert is_supported_device_string("cuda:0")
    assert not is_supported_device_string("cpu:0")


def test_device_aliases_are_canonicalized_for_torch():
    assert canonicalize_torch_device("cpu") == "cpu"
    assert canonicalize_torch_device("cuda:3") == "cuda:3"
    assert canonicalize_torch_device("rocm") == "cuda"
    assert canonicalize_torch_device("hip:2") == "cuda:2"
    with pytest.raises(ValueError):
        canonicalize_torch_device("rocm:x")


def test_matcher_and_depth_base_classes_canonicalize_rocm_device():
    matcher = _DummyMatcher(SimpleNamespace(device="rocm:1"))
    depth = _DummyDepth(SimpleNamespace(device="hip"))
    assert matcher.config.device == "cuda:1"
    assert depth.config.device == "cuda"


def test_optimizer_context_canonicalizes_device_alias():
    context = TwoFrame_PGO.init_context(
        SimpleNamespace(
            autodiff=True,
            graph_type="icp",
            vectorize=False,
            device="rocm:2",
        )
    )
    assert context["device"] == "cuda:2"


def test_runtime_override_updates_nested_odometry_device_fields():
    odom_cfg = {
        "args": {"device": "cuda"},
        "frontend": {"args": {"device": "cuda:0"}},
        "outlier": {"args": {"filter_args": [{"args": {"device": "cuda:1"}}]}},
    }
    set_odometry_device_fields(odom_cfg, "cpu")
    assert odom_cfg["args"]["device"] == "cpu"
    assert odom_cfg["frontend"]["args"]["device"] == "cpu"
    assert odom_cfg["outlier"]["args"]["filter_args"][0]["args"]["device"] == "cpu"


def test_runtime_override_normalizes_aliases_in_nested_odometry_device_fields():
    odom_cfg = {
        "args": {"device": "cpu"},
        "frontend": {"args": {"device": "cpu"}},
        "outlier": {"args": {"filter_args": [{"args": {"device": "cpu"}}]}},
    }
    set_odometry_device_fields(odom_cfg, "rocm:4")
    assert odom_cfg["args"]["device"] == "cuda:4"
    assert odom_cfg["frontend"]["args"]["device"] == "cuda:4"
    assert odom_cfg["outlier"]["args"]["filter_args"][0]["args"]["device"] == "cuda:4"
