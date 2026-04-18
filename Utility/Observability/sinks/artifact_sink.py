from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from typing import Any

import torch

from .base import Sink


class LocalArtifactSink(Sink):
    def __init__(self, run_dir: Path, keep_last_n: int = 200, max_file_mb: int = 200) -> None:
        self.run_dir = run_dir
        self.keep_last_n = max(1, int(keep_last_n))
        self.max_file_bytes = int(max_file_mb) * 1024 * 1024
        self._lock = threading.RLock()
        self._scalar_lines: list[str] = []

        (self.run_dir / "scalars").mkdir(parents=True, exist_ok=True)
        self.scalars_path = self.run_dir / "scalars" / "scalars.jsonl"

    def on_scalar(self, key: str, value: float, step: int) -> None:
        rec = {"key": key, "value": float(value), "step": int(step), "ts": float(time.time())}
        with self._lock:
            self._scalar_lines.append(json.dumps(rec))

    def on_hist(self, key: str, tensor: torch.Tensor, step: int) -> None:
        return

    def on_image(self, key: str, image: torch.Tensor, step: int, caption: str | None = None) -> None:
        return

    def on_table(self, key: str, rows: list[dict[str, Any]], step: int) -> None:
        return

    def on_artifact(self, key: str, payload: dict[str, Any], step: int) -> None:
        target_dir = self.run_dir / key
        target_dir.mkdir(parents=True, exist_ok=True)
        final_path = target_dir / f"step_{int(step):06d}.pt"
        partial_path = final_path.with_suffix(".pt.partial")

        torch.save(payload, partial_path)
        file_size = partial_path.stat().st_size
        if file_size > self.max_file_bytes:
            partial_path.unlink(missing_ok=True)
            raise RuntimeError(f"Artifact {final_path} exceeds max_file_mb limit ({file_size} bytes).")

        partial_path.replace(final_path)
        self._prune_regular_dumps(target_dir)

    def on_phase(self, name: str, duration_ms: float, step: int) -> None:
        self.on_scalar(f"timing.phase.{name}", float(duration_ms), int(step))

    def flush(self) -> None:
        with self._lock:
            if len(self._scalar_lines) == 0:
                return
            with self.scalars_path.open("a", encoding="utf-8") as f:
                for line in self._scalar_lines:
                    f.write(line + "\n")
            self._scalar_lines.clear()

    def close(self) -> None:
        self.flush()

    def assert_writable(self, run_dir: Path) -> None:
        run_dir.mkdir(parents=True, exist_ok=True)
        marker = run_dir / ".write_test"
        marker.write_text("ok", encoding="utf-8")
        marker.unlink(missing_ok=True)

    def _prune_regular_dumps(self, target_dir: Path) -> None:
        files = sorted(target_dir.glob("step_*.pt"))
        excess = len(files) - self.keep_last_n
        if excess <= 0:
            return
        for path in files[:excess]:
            path.unlink(missing_ok=True)
