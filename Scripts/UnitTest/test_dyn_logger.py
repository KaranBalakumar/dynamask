"""Unit tests for DynTrainLogger offline rendering."""

import os
import tempfile
import torch
import pytest
from types import SimpleNamespace

from Train.DynNet.dyn_logger import DynTrainLogger


class TestDynTrainLogger:
    @pytest.fixture
    def logger_and_dir(self):
        cfg = SimpleNamespace(name="test", wandb=False, visual_freq=500)
        tmp = tempfile.mkdtemp()
        logger = DynTrainLogger(cfg, "test_run", tmp)
        return logger, tmp

    @pytest.fixture
    def dummy_visuals(self):
        return {
            "img1": torch.rand(3, 120, 160),
            "img2": torch.rand(3, 120, 160),
            "dyn_logits_final": torch.randn(1, 30, 40),
            "flow_est": torch.randn(2, 120, 160),
            "flow_rigid": torch.randn(2, 120, 160),
            "M_pseudo": torch.randint(-1, 2, (1, 120, 160)),
            "residual": torch.rand(1, 120, 160) * 20,
            "tau": torch.ones(1, 120, 160) * 5,
            "f_imu": torch.randn(128),
            "imu_tokens": torch.randn(7, 128),
            "alpha": torch.tensor(0.01),
            "token_weights": torch.ones(7),
            "trainable_grad_norm": 0.05,
            "frozen_grad_norm": 0.0,
        }

    def test_log_visuals_creates_pngs(self, logger_and_dir, dummy_visuals):
        logger, tmp = logger_and_dir
        logger.log_visuals(dummy_visuals, step=100)
        step_dir = os.path.join(tmp, "test_run", "debug", "step_000100")
        pngs = sorted(f for f in os.listdir(step_dir) if f.endswith(".png"))
        assert len(pngs) == 5
        assert "dyn_overlay.png" in pngs
        assert "pseudo_labels.png" in pngs
        assert "flow_comparison.png" in pngs
        assert "histograms.png" in pngs
        assert os.path.exists(os.path.join(step_dir, "tensors.pt"))

    def test_log_visuals_no_imu(self, logger_and_dir, dummy_visuals):
        logger, tmp = logger_and_dir
        v = {k: w for k, w in dummy_visuals.items() if k not in ("f_imu", "imu_tokens")}
        logger.log_visuals(v, step=200)
        step_dir = os.path.join(tmp, "test_run", "debug", "step_000200")
        pngs = os.listdir(step_dir)
        assert "histograms.png" in pngs

    def test_frozen_grad_check_returns_zero_for_frozen(self, logger_and_dir):
        import torch.nn as nn
        logger, _ = logger_and_dir
        model = nn.Sequential(
            nn.Linear(10, 10),
            nn.Linear(10, 1),
        )
        for name, p in model.named_parameters():
            p.grad = torch.randn_like(p) if "0" in name else torch.zeros_like(p)
        result = logger.log_frozen_grad_check(model, trainable_pattern="0")
        assert result["frozen_grad_norm"] == 0.0
        assert result["trainable_grad_norm"] > 0.0

    def test_log_step_is_noop_without_wandb(self, logger_and_dir):
        logger, _ = logger_and_dir
        logger.log_step({"loss": 1.5, "lr": 1e-4}, step=10)

    def test_finish_cleans_up(self, logger_and_dir):
        logger, _ = logger_and_dir
        logger.finish()
