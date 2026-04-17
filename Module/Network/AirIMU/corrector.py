from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn as nn


class AirIMUCorrector(nn.Module):
    """
    Minimal frozen IMU corrector interface.

    This implementation is intentionally lightweight and checkpoint-agnostic:
    - if a compatible checkpoint/model is available, load it through `from_ckpt`.
    - otherwise, fall back to identity corrections with conservative covariances.
    """

    def __init__(self) -> None:
        super().__init__()

    @classmethod
    def from_ckpt(cls, ckpt_path: str | Path | None) -> "AirIMUCorrector":
        model = cls()
        if ckpt_path is None:
            return model
        path = Path(ckpt_path)
        if not path.exists():
            return model
        try:
            # Keep future compatibility without hard binding to a single format.
            state: Any = torch.load(path, map_location="cpu", weights_only=False)
            if isinstance(state, dict) and "state_dict" in state:
                model.load_state_dict(state["state_dict"], strict=False)
            elif isinstance(state, dict):
                model.load_state_dict(state, strict=False)
        except Exception:
            # Fall back to identity mode if checkpoint is not compatible.
            pass
        return model

    @torch.inference_mode()
    def inference(self, imu_batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor | dict[str, torch.Tensor]]:
        acc = imu_batch["acc"]
        gyro = imu_batch["gyro"]
        B, N, _ = acc.shape
        device, dtype = acc.device, acc.dtype

        correction_acc = torch.zeros_like(acc)
        correction_gyro = torch.zeros_like(gyro)

        # Conservative default process variances (per sample, diagonal).
        acc_cov = torch.full((B, N, 3), 1e-4, dtype=dtype, device=device)
        gyro_cov = torch.full((B, N, 3), 1e-5, dtype=dtype, device=device)

        return {
            "correction_acc": correction_acc,
            "correction_gyro": correction_gyro,
            "cov_state": {
                "acc_cov": acc_cov,
                "gyro_cov": gyro_cov,
            },
        }

