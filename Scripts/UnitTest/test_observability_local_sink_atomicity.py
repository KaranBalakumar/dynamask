from __future__ import annotations

from pathlib import Path

import torch

from Utility.Observability.sinks import LocalArtifactSink


def test_local_sink_atomic_dump_write(tmp_path: Path):
    sink = LocalArtifactSink(tmp_path / "run", keep_last_n=10, max_file_mb=100)
    sink.on_artifact("dumps", {"payload": torch.randn(4, 4)}, step=12)
    sink.flush()
    target = tmp_path / "run" / "dumps" / "step_000012.pt"
    assert target.exists()
    assert not target.with_suffix(".pt.partial").exists()
    sink.close()


def test_local_sink_keep_last_n_rolls_regular_dumps(tmp_path: Path):
    sink = LocalArtifactSink(tmp_path / "run", keep_last_n=2, max_file_mb=100)
    for step in [1, 2, 3]:
        sink.on_artifact("dumps", {"step": torch.tensor(step)}, step=step)
    sink.flush()
    files = sorted((tmp_path / "run" / "dumps").glob("step_*.pt"))
    assert len(files) == 2
    assert files[0].name == "step_000002.pt"
    assert files[1].name == "step_000003.pt"
    sink.close()

