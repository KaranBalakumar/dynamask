from __future__ import annotations

import subprocess
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import torch

from .keys_runtime import validate_key
from .sinks import LocalArtifactSink, Sink, TensorBoardSink, WandbSink


class NumericalAbort(RuntimeError):
    pass


@dataclass(frozen=True)
class LoggerConfig:
    run_dir: str = "./runs"
    fail_on_nan: bool = True
    strict_keys: bool = True
    hist_every: int = 50


def _git_sha() -> str:
    try:
        out = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
        return out if out else "unknown"
    except Exception:
        return "unknown"


class DebugLogger:
    def __init__(
        self,
        run_dir: Path,
        sinks: Iterable[Sink],
        fail_on_nan: bool = True,
        strict_keys: bool = True,
        hist_every: int = 50,
    ) -> None:
        self.run_dir = run_dir
        self.sinks = list(sinks)
        self.fail_on_nan = fail_on_nan
        self.strict_keys = strict_keys
        self.hist_every = max(1, int(hist_every))
        self._lock = threading.RLock()
        self._nan_counter: dict[str, int] = {}

    @classmethod
    def from_config(cls, cfg: Any, seed: int = 0) -> "DebugLogger":
        run_base = Path(getattr(cfg, "run_dir", "./runs"))
        ts = datetime.now(timezone.utc).strftime("%Y-%m-%d_%H%M%S")
        sha = _git_sha()
        run_dir = run_base / f"{ts}_{sha}_seed{seed}"
        run_dir.mkdir(parents=True, exist_ok=True)
        sinks: list[Sink] = []

        local_cfg = getattr(cfg, "local", None)
        if local_cfg is None or bool(getattr(local_cfg, "enabled", True)):
            sinks.append(
                LocalArtifactSink(
                    run_dir=run_dir,
                    keep_last_n=getattr(local_cfg, "keep_last_n", 200) if local_cfg is not None else 200,
                    max_file_mb=getattr(local_cfg, "max_file_mb", 200) if local_cfg is not None else 200,
                )
            )

        tb_cfg = getattr(cfg, "tensorboard", None)
        if tb_cfg is not None and bool(getattr(tb_cfg, "enabled", False)):
            sinks.append(
                TensorBoardSink(
                    run_dir=run_dir,
                    flush_secs=int(getattr(tb_cfg, "flush_secs", 30)),
                    max_queue=int(getattr(tb_cfg, "max_queue", 2000)),
                )
            )

        wandb_cfg = getattr(cfg, "wandb", None)
        if wandb_cfg is not None and bool(getattr(wandb_cfg, "enabled", False)):
            sinks.append(
                WandbSink(
                    run_dir=run_dir,
                    project=getattr(wandb_cfg, "project", "macvo"),
                    entity=getattr(wandb_cfg, "entity", None),
                    group=getattr(wandb_cfg, "group", None),
                    tags=list(getattr(wandb_cfg, "tags", [])),
                    mode=getattr(wandb_cfg, "mode", "online"),
                    config={},
                )
            )

        logger = cls(
            run_dir=run_dir,
            sinks=sinks,
            fail_on_nan=bool(getattr(cfg, "fail_on_nan", True)),
            strict_keys=bool(getattr(cfg, "strict_keys", True)),
            hist_every=int(getattr(cfg, "hist_every", 50)),
        )
        logger._write_manifest(cfg)
        return logger

    def _write_manifest(self, cfg: Any) -> None:
        manifest = self.run_dir / "manifest"
        manifest.mkdir(parents=True, exist_ok=True)
        (manifest / "git_sha.txt").write_text(_git_sha() + "\n", encoding="utf-8")
        (manifest / "config.txt").write_text(str(cfg), encoding="utf-8")

    def _assert_key(self, key: str) -> None:
        if not self.strict_keys:
            return
        result = validate_key(key)
        if not result.ok:
            raise KeyError(result.reason)

    def _check_scalar_finite(self, key: str, value: float, step: int) -> None:
        if torch.isfinite(torch.tensor(value)):
            return
        info = {"key": key, "step": step, "value": value}
        if self.fail_on_nan:
            raise NumericalAbort(str(info))
        self._nan_counter[key] = self._nan_counter.get(key, 0) + 1
        self.log_scalar(f"diag.nan.{key}", float(self._nan_counter[key]), step)

    def _check_tensor_finite(self, key: str, tensor: torch.Tensor, step: int) -> None:
        finite = torch.isfinite(tensor)
        if bool(finite.all()):
            return
        info = {
            "key": key,
            "step": step,
            "nan": int(torch.isnan(tensor).sum().item()),
            "inf": int(torch.isinf(tensor).sum().item()),
            "shape": tuple(tensor.shape),
        }
        if self.fail_on_nan:
            raise NumericalAbort(str(info))
        self._nan_counter[key] = self._nan_counter.get(key, 0) + 1
        self.log_scalar(f"diag.nan.{key}", float(self._nan_counter[key]), step)

    def log_scalar(self, key: str, value: float, step: int) -> None:
        with self._lock:
            self._assert_key(key)
            self._check_scalar_finite(key, float(value), step)
            for sink in self.sinks:
                sink.on_scalar(key, float(value), int(step))

    def log_scalars(self, kv: dict[str, float], step: int) -> None:
        for key, value in kv.items():
            self.log_scalar(key, float(value), step)

    def log_hist(self, key: str, tensor: torch.Tensor, step: int, num_bins: int = 64, percentile_clip: float = 99.5) -> None:
        if int(step) % self.hist_every != 0:
            return
        with self._lock:
            self._assert_key(key)
            t = tensor.detach().float()
            self._check_tensor_finite(key, t, step)
            if t.numel() == 0:
                return
            q = torch.quantile(t.flatten().abs(), percentile_clip / 100.0)
            t = t.clamp(min=-q, max=q)
            for sink in self.sinks:
                sink.on_hist(key, t, int(step))

    def log_image(self, key: str, image: torch.Tensor, step: int, caption: str | None = None) -> None:
        with self._lock:
            self._assert_key(key)
            self._check_tensor_finite(key, image, step)
            for sink in self.sinks:
                sink.on_image(key, image, int(step), caption=caption)

    def log_table(self, key: str, rows: list[dict[str, Any]], step: int) -> None:
        with self._lock:
            self._assert_key(key)
            for sink in self.sinks:
                sink.on_table(key, rows, int(step))

    def dump_artifact(self, key: str, payload: dict[str, Any], step: int) -> None:
        with self._lock:
            for sink in self.sinks:
                sink.on_artifact(key, payload, int(step))

    @contextmanager
    def phase(self, name: str, step: int):
        start = time.perf_counter()
        try:
            yield
        finally:
            dt_ms = (time.perf_counter() - start) * 1000.0
            self.log_scalar(f"timing.{name}.ms", float(dt_ms), step)
            for sink in self.sinks:
                sink.on_phase(name, float(dt_ms), int(step))

    def on_step_end(self, step: int) -> None:
        for sink in self.sinks:
            sink.flush()

    def close(self) -> None:
        for sink in self.sinks:
            sink.close()

    def assert_all_sinks_writable(self) -> None:
        for sink in self.sinks:
            sink.assert_writable(self.run_dir)
