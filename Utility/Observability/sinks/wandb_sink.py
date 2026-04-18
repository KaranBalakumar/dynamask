from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from .base import Sink

try:
    import wandb
except Exception:  # pragma: no cover
    wandb = None


class WandbSink(Sink):
    def __init__(
        self,
        run_dir: Path,
        project: str,
        entity: str | None = None,
        group: str | None = None,
        tags: list[str] | None = None,
        mode: str = "online",
        config: dict[str, Any] | None = None,
    ) -> None:
        self.enabled = wandb is not None
        self.run = None
        if self.enabled:
            self.run = wandb.init(
                project=project,
                entity=entity,
                group=group,
                tags=tags or [],
                mode=mode,
                config=config or {},
                dir=str(run_dir),
                settings=wandb.Settings(start_method="thread"),
            )

    def on_scalar(self, key: str, value: float, step: int) -> None:
        if self.run is not None:
            self.run.log({key: value}, step=step)

    def on_hist(self, key: str, tensor: torch.Tensor, step: int) -> None:
        if self.run is not None:
            self.run.log({key: wandb.Histogram(tensor.detach().float().cpu())}, step=step)

    def on_image(self, key: str, image: torch.Tensor, step: int, caption: str | None = None) -> None:
        if self.run is not None:
            self.run.log({key: wandb.Image(image.detach().float().cpu(), caption=caption)}, step=step)

    def on_table(self, key: str, rows: list[dict[str, Any]], step: int) -> None:
        if self.run is None or len(rows) == 0:
            return
        cols = sorted({k for r in rows for k in r.keys()})
        table = wandb.Table(columns=cols)
        for row in rows:
            table.add_data(*[row.get(c) for c in cols])
        self.run.log({key: table}, step=step)

    def on_artifact(self, key: str, payload: dict[str, Any], step: int) -> None:
        return

    def on_phase(self, name: str, duration_ms: float, step: int) -> None:
        self.on_scalar(f"timing.phase.{name}", duration_ms, step)

    def flush(self) -> None:
        return

    def close(self) -> None:
        if self.run is not None:
            self.run.finish()

