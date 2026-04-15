import pytest
import torch

from Utility.Device import (
    is_supported_device_string,
    normalize_backend_device,
    set_odometry_device_fields,
)


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
