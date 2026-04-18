import h5py
import numpy as np
import pypose as pp
import pytest
import torch

from DataLoader.Dataset.DynamicHeadTrain import DynamicHeadDatasetEmptyError, DynamicHeadTrainDataset
from Module.Network.DynamicHead import StaticConfidenceHead
from Train.DynamicHead.loss import compute_dynamic_head_loss


def test_static_confidence_head_shapes():
    head = StaticConfidenceHead(c_ctx=32, c_imu=12, c_hid=24)
    phi8 = torch.randn(2, 32, 8, 8)
    h4 = torch.randn(2, 3, 16, 16)
    z_hat = torch.rand(2, 1, 64, 64) * 4.0 + 0.5
    e_raw = torch.rand(2, 1, 64, 64)
    f_imu = torch.randn(2, 12)
    out = head(phi8, h4, z_hat, e_raw, f_imu, h8_prev=None, return_probs=True)
    assert out.logits_8.shape == (2, 1, 8, 8)
    assert out.logits_4.shape == (2, 1, 16, 16)
    assert out.c.shape == (2, 1, 16, 16)
    assert torch.isfinite(out.c).all()
    assert torch.all((out.c >= 0.0) & (out.c <= 1.0))


def test_dynamic_head_loss_finite():
    B, H, W = 1, 32, 32
    gt_pose_t = pp.SE3(torch.tensor([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]], dtype=torch.float32))
    gt_pose_t1 = pp.SE3(torch.tensor([[0.05, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]], dtype=torch.float32))
    T_BS = pp.identity_SE3(B)
    K = torch.tensor([[[120.0, 0.0, 16.0], [0.0, 120.0, 16.0], [0.0, 0.0, 1.0]]], dtype=torch.float32)
    depth = torch.full((B, 1, H, W), 2.0, dtype=torch.float32)
    flow = torch.zeros((B, 2, H, W), dtype=torch.float32)
    logits = torch.zeros((B, 1, H // 2, W // 2), dtype=torch.float32)
    c_pred = torch.sigmoid(logits)
    loss, metrics = compute_dynamic_head_loss(
        logits_4=logits,
        c_pred=c_pred,
        flow_fwd=flow,
        flow_bwd=None,
        depth_t1=depth,
        K=K,
        gt_pose_t=gt_pose_t,
        gt_pose_t1=gt_pose_t1,
        T_BS=T_BS,
        cfg={"tau0": 0.5, "alpha": 0.25, "kappa": 1.5, "lambda_smooth": 0.03},
    )
    assert torch.isfinite(loss)
    assert metrics["valid_ratio"] > 0.0


def test_dynamic_head_dataset_filters_invalid_gt(tmp_path):
    path = tmp_path / "mini_viode.h5"
    with h5py.File(path, "w") as h5:
        g_st = h5.create_group("stereo")
        g_imu = h5.create_group("imu")
        g_cal = h5.create_group("calib")
        g_st.create_dataset("time_ns", data=np.array([0, 100, 200], dtype=np.int64))
        g_st.create_dataset("left", data=np.zeros((3, 8, 8, 3), dtype=np.uint8))
        g_st.create_dataset("right", data=np.zeros((3, 8, 8, 3), dtype=np.uint8))
        g_st.create_dataset(
            "gt_pose",
            data=np.array(
                [
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
                    [0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
                    [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
                ],
                dtype=np.float64,
            ),
        )
        g_imu.create_dataset("time_ns", data=np.array([0, 50, 100, 150, 200], dtype=np.int64))
        g_imu.create_dataset("acc", data=np.zeros((5, 3), dtype=np.float32))
        g_imu.create_dataset("gyro", data=np.zeros((5, 3), dtype=np.float32))
        g_cal.create_dataset("K", data=np.eye(3, dtype=np.float32))
        g_cal.create_dataset("T_BS", data=np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0], dtype=np.float32))
        g_cal.create_dataset("baseline", data=np.float32(0.1))
        g_cal.create_dataset("gravity", data=np.float32(9.81))

    ds = DynamicHeadTrainDataset(
        {
            "sequence": {"type": "VIODE_Stream", "args": {"path": str(path)}},
            "start": 0,
            "end": None,
            "step": 1,
        }
    )
    assert ds.stats.total_pairs == 2
    assert ds.stats.valid_pairs == 1
    sample = ds[0]
    assert sample.cur.gt_pose is not None
    assert sample.nxt.gt_pose is not None


def test_dynamic_head_dataset_raises_when_no_valid_pairs(tmp_path):
    path = tmp_path / "mini_viode_no_gt.h5"
    with h5py.File(path, "w") as h5:
        g_st = h5.create_group("stereo")
        g_imu = h5.create_group("imu")
        g_cal = h5.create_group("calib")
        g_st.create_dataset("time_ns", data=np.array([0, 100], dtype=np.int64))
        g_st.create_dataset("left", data=np.zeros((2, 8, 8, 3), dtype=np.uint8))
        g_st.create_dataset("right", data=np.zeros((2, 8, 8, 3), dtype=np.uint8))
        g_st.create_dataset("gt_pose", data=np.full((2, 7), np.nan, dtype=np.float64))
        g_imu.create_dataset("time_ns", data=np.array([0, 50, 100], dtype=np.int64))
        g_imu.create_dataset("acc", data=np.zeros((3, 3), dtype=np.float32))
        g_imu.create_dataset("gyro", data=np.zeros((3, 3), dtype=np.float32))
        g_cal.create_dataset("K", data=np.eye(3, dtype=np.float32))
        g_cal.create_dataset("T_BS", data=np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0], dtype=np.float32))
        g_cal.create_dataset("baseline", data=np.float32(0.1))
        g_cal.create_dataset("gravity", data=np.float32(9.81))

    with pytest.raises(DynamicHeadDatasetEmptyError) as exc_info:
        DynamicHeadTrainDataset(
            {
                "sequence": {"type": "VIODE_Stream", "args": {"path": str(path)}},
                "start": 0,
                "end": None,
                "step": 1,
            }
        )

    assert exc_info.value.stats.total_pairs == 1
    assert exc_info.value.stats.valid_pairs == 0
