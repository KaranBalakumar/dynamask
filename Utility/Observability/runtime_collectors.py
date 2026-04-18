from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import torch


def _cpuify(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        if value.dtype in (torch.float32, torch.float64):
            return value.detach().cpu().to(torch.float16)
        return value.detach().cpu()
    if isinstance(value, dict):
        return {k: _cpuify(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_cpuify(v) for v in value]
    return value


def collect_runtime_dump(
    step: int,
    epoch: int,
    seq_id: str,
    frame_idx: int,
    batch_idx: int,
    cadence: str,
    data: dict[str, Any] | None = None,
    frontend: dict[str, Any] | None = None,
    drt: dict[str, Any] | None = None,
    pgo: dict[str, Any] | None = None,
    writeback: dict[str, Any] | None = None,
    git_sha: str = "unknown",
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "meta": {
            "step": step,
            "epoch": epoch,
            "seq_id": seq_id,
            "frame_idx": frame_idx,
            "batch_idx": batch_idx,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "git_sha": git_sha,
            "cadence": cadence,
        }
    }
    if data is not None:
        payload["data"] = _cpuify(data)
    if frontend is not None:
        payload["frontend"] = _cpuify(frontend)
    if drt is not None:
        payload["drt"] = _cpuify(drt)
    if pgo is not None:
        payload["pgo"] = _cpuify(pgo)
    if writeback is not None:
        payload["writeback"] = _cpuify(writeback)
    return payload


def validate_dump_schema(payload: dict[str, Any]) -> None:
    assert "meta" in payload, "dump payload missing meta"
    meta = payload["meta"]
    required = {"step", "epoch", "seq_id", "frame_idx", "batch_idx", "timestamp", "git_sha", "cadence"}
    missing = required - set(meta.keys())
    assert len(missing) == 0, f"dump payload missing meta keys: {sorted(missing)}"

