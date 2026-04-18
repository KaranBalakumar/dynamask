from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from Utility.Observability import DebugLogger, NumericalAbort
from Utility.Observability.sinks import LocalArtifactSink


def test_debug_logger_scalar_and_key_guard(tmp_path: Path):
    sink = LocalArtifactSink(tmp_path / "run", keep_last_n=10)
    logger = DebugLogger(run_dir=tmp_path / "run", sinks=[sink], fail_on_nan=True, strict_keys=True)
    logger.log_scalar("backend.twoframe.chi2.init", 1.0, 0)
    with pytest.raises(KeyError):
        logger.log_scalar("unknown.key", 1.0, 0)
    logger.close()


def test_debug_logger_nan_abort(tmp_path: Path):
    sink = LocalArtifactSink(tmp_path / "run", keep_last_n=10)
    logger = DebugLogger(run_dir=tmp_path / "run", sinks=[sink], fail_on_nan=True, strict_keys=False)
    with pytest.raises(NumericalAbort):
        logger.log_scalar("any.key", float("nan"), 0)
    logger.close()


def test_debug_logger_from_config(tmp_path: Path):
    cfg = SimpleNamespace(
        run_dir=str(tmp_path / "runs"),
        enabled=True,
        fail_on_nan=False,
        strict_keys=False,
        local=SimpleNamespace(enabled=True, keep_last_n=5, max_file_mb=20),
        tensorboard=SimpleNamespace(enabled=False),
        wandb=SimpleNamespace(enabled=False),
    )
    logger = DebugLogger.from_config(cfg, seed=123)
    logger.log_scalars({"foo": 1.0}, 1)
    logger.dump_artifact("dumps", {"x": torch.tensor([1, 2, 3])}, 1)
    logger.on_step_end(1)
    logger.close()
    run_dirs = list((tmp_path / "runs").glob("*_seed123"))
    assert len(run_dirs) == 1
    assert (run_dirs[0] / "scalars" / "scalars.jsonl").exists()

