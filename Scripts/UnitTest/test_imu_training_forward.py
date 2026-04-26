"""Unit tests for IMUContext training-mode forward (seed_from_gt + AirIO skip)."""

import torch
import pytest
from types import SimpleNamespace

from Module.Network.IMUContext.imu_context import IMUContext
from DataLoader.Interface import AttitudeData
import pypose as pp


def make_dummy_attitude(dev="cpu"):
    """Create a minimal AttitudeData for testing seed_from_gt."""
    return AttitudeData(
        T_BS=pp.identity_SE3(1),
        time_ns=torch.zeros(1, 1, 1, dtype=torch.long, device=dev),
        gravity=[9.81],
        gt_pos=torch.zeros(1, 1, 3, device=dev),
        gt_vel=torch.zeros(1, 1, 3, device=dev),
        gt_rot=pp.identity_SO3(1).unsqueeze(0),
        init_pos=torch.tensor([[[1.0, 2.0, 3.0]]], device=dev),
        init_vel=torch.tensor([[[0.5, 0.0, 0.0]]], device=dev),
        init_rot=pp.identity_SO3(1).unsqueeze(0).unsqueeze(0),  # (1,1,4)
    )


class TestIMUContextTraining:
    def test_seed_from_gt_sets_state(self):
        ctx = IMUContext(airio_cfg=SimpleNamespace(propcov=True), airio_ckpt=None)
        att = make_dummy_attitude()
        ctx.seed_from_gt(att)
        assert ctx._state is not None
        assert ctx._prev_cam_state is not None
        assert torch.allclose(ctx._state[6:9].float(),
                              torch.tensor([1.0, 2.0, 3.0]))

    def test_airio_skipped_without_checkpoint(self):
        ctx = IMUContext(airio_cfg=SimpleNamespace(propcov=True), airio_ckpt=None)
        assert ctx._has_airio is False
        att = make_dummy_attitude()
        ctx.seed_from_gt(att)
        tick = {"acc": torch.zeros(3), "gyro": torch.zeros(3), "dt": torch.tensor(0.01)}
        sample = ctx.step([tick], [{"acc": torch.zeros(3), "gyro": torch.zeros(3)}])
        assert torch.allclose(sample.airio_vel, torch.zeros(3))

    def test_step_produces_valid_features(self):
        ctx = IMUContext(airio_cfg=SimpleNamespace(propcov=True), airio_ckpt=None)
        att = make_dummy_attitude()
        ctx.seed_from_gt(att)
        ticks_corrected = [
            {"acc": torch.tensor([0.5, 0.0, 9.81]), "gyro": torch.zeros(3), "dt": torch.tensor(0.01)}
            for _ in range(10)
        ]
        ticks_raw = [{"acc": t["acc"], "gyro": t["gyro"]} for t in ticks_corrected]
        sample = ctx.step(ticks_corrected, ticks_raw)
        assert sample.f_imu.shape == (1, 128)
        assert sample.imu_tokens.shape == (1, 7, 128)
        assert not torch.allclose(sample.f_imu, torch.zeros(1, 128))

    def test_seed_then_step_then_seed_resets(self):
        ctx = IMUContext(airio_cfg=SimpleNamespace(propcov=True), airio_ckpt=None)
        att = make_dummy_attitude()
        tick = {"acc": torch.zeros(3), "gyro": torch.zeros(3), "dt": torch.tensor(0.01)}
        raw = [{"acc": torch.zeros(3), "gyro": torch.zeros(3)}]

        ctx.seed_from_gt(att)
        s1 = ctx.step([tick], raw)
        ctx.seed_from_gt(att)
        s2 = ctx.step([tick], raw)
        assert torch.allclose(s1.f_imu, s2.f_imu)
