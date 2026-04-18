from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import torch


class Sink(ABC):
    @abstractmethod
    def on_scalar(self, key: str, value: float, step: int) -> None: ...

    @abstractmethod
    def on_hist(self, key: str, tensor: torch.Tensor, step: int) -> None: ...

    @abstractmethod
    def on_image(self, key: str, image: torch.Tensor, step: int, caption: str | None = None) -> None: ...

    @abstractmethod
    def on_table(self, key: str, rows: list[dict[str, Any]], step: int) -> None: ...

    @abstractmethod
    def on_artifact(self, key: str, payload: dict[str, Any], step: int) -> None: ...

    @abstractmethod
    def on_phase(self, name: str, duration_ms: float, step: int) -> None: ...

    @abstractmethod
    def flush(self) -> None: ...

    @abstractmethod
    def close(self) -> None: ...

    def assert_writable(self, run_dir: Path) -> None:
        return

