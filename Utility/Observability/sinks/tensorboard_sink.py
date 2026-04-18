from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from .base import Sink

try:
    from torch.utils.tensorboard import SummaryWriter
except Exception:  # pragma: no cover
    SummaryWriter = None


class TensorBoardSink(Sink):
    def __init__(self, run_dir: Path, flush_secs: int = 30, max_queue: int = 2000) -> None:
        self.enabled = SummaryWriter is not None
        self.writer = SummaryWriter(log_dir=str(run_dir / "tensorboard"), flush_secs=flush_secs, max_queue=max_queue) if self.enabled else None

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

