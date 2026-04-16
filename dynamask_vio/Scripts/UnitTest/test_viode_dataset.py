from pathlib import Path

import pypose as pp
import torch
import pytest

from DataLoader.Dataset.VIODE import REQUIRED_TOPICS, VIODE_Sequence, compute_camera_relative_pose


def test_viode_required_topics_are_rosbag_contract() -> None:
    assert REQUIRED_TOPICS == ["/cam0/image_raw", "/cam1/image_raw", "/imu0", "/odometry"]


def test_viode_config_validation_requires_calibration_files(tmp_path: Path) -> None:
    root = tmp_path / "seq"
    root.mkdir()

    cfg_missing = {
        "root": str(root),
        "bag": "sample.bag",
        "cam0_calib": "cam0_pinhole.yaml",
        "cam1_calib": "cam1_pinhole.yaml",
        "calib": "calibration.yaml",
    }
    VIODE_Sequence.is_valid_config(cfg_missing)

    with pytest.raises(KeyError):
        VIODE_Sequence.is_valid_config({"root": str(root), "bag": "sample.bag"})


def test_compute_camera_relative_pose_matches_manual_formula() -> None:
    T_wb_i = pp.SE3(torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0], dtype=torch.float64))
    T_wb_j = pp.SE3(torch.tensor([0.25, -0.1, 0.05, 0.0, 0.0, 0.0, 1.0], dtype=torch.float64))
    T_bc = pp.SE3(torch.tensor([0.02, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0], dtype=torch.float64))

    rel = compute_camera_relative_pose(T_wb_i=T_wb_i, T_wb_j=T_wb_j, T_bc=T_bc)

    T_wc_i = T_wb_i @ T_bc
    T_wc_j = T_wb_j @ T_bc
    expected = T_wc_i.Inv() @ T_wc_j
    assert torch.allclose(rel.tensor(), expected.tensor(), atol=1e-6, rtol=1e-6)
