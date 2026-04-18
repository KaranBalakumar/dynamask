from __future__ import annotations

from importlib import import_module
from pathlib import Path
from typing import Any

import torch

from .base import Sink

try:
    _tb_mod = import_module("torch.utils.tensorboard")
except Exception:
    _tb_mod = None
SummaryWriter = getattr(_tb_mod, "SummaryWriter", None)


class TensorBoardSink(Sink):
    def __init__(self, run_dir: Path, flush_secs: int = 30, max_queue: int = 2000) -> None:
        self.writer: Any | None
        if SummaryWriter is None:
            self.enabled = False
            self.writer = None
        else:
            self.enabled = True
            self.writer = SummaryWriter(log_dir=str(run_dir / "tensorboard"), flush_secs=flush_secs, max_queue=max_queue)

    def on_scalar(self, key: str, value: float, step: int) -> None:
        if self.writer is not None:
            self.writer.add_scalar(key, value, step)

    def on_hist(self, key: str, tensor: torch.Tensor, step: int) -> None:
        if self.writer is not None:
            self.writer.add_histogram(key, tensor.detach().float().cpu(), step)

    def on_image(self, key: str, image: torch.Tensor, step: int, caption: str | None = None) -> None:
        if self.writer is None:
            return
        img = image.detach().float().cpu()
        if img.ndim == 2:
            img = img.unsqueeze(0)
        self.writer.add_image(key, img, step)

    def on_table(self, key: str, rows: list[dict[str, Any]], step: int) -> None:
        if self.writer is None:
            return
        self.writer.add_text(key, str(rows), step)

    def on_artifact(self, key: str, payload: dict[str, Any], step: int) -> None:
        return

    def on_phase(self, name: str, duration_ms: float, step: int) -> None:
        self.on_scalar(f"timing.phase.{name}", duration_ms, step)

    def flush(self) -> None:
        if self.writer is not None:
            self.writer.flush()

    def close(self) -> None:
        if self.writer is not None:
            self.writer.close()
