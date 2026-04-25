from __future__ import annotations
from dataclasses import dataclass


@dataclass
class DRTInitConfig:
    min_keyframes: int = 10
    max_attempts: int = 2
    window_scales: tuple[float, ...] = (1.0, 1.4, 1.8)


@dataclass
class DRTInitResult:
    success: bool
    failure_reason: str | None
    retry_recommended: bool

    @staticmethod
    def failure(reason: str) -> "DRTInitResult":
        return DRTInitResult(success=False, failure_reason=reason, retry_recommended=True)
