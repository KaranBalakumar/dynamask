from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional
import torch


@dataclass
class DRTInitConfig:
    min_keyframes: int = 10
    max_attempts: int = 2
    window_scales: tuple[float, ...] = (1.0, 1.4, 1.8)
    # Quality gate thresholds (from design doc §5.7)
    quality_min_avg_obs: int = 30
    quality_max_cond: float = 1e8
    quality_min_pos_depth_ratio: float = 0.7
    quality_gravity_tol_rel: float = 1e-3
    gravity_norm: float = 9.81007
    huber_delta_gyro_rad: float = 1e-2
    fallback: str = "heuristic"


@dataclass
class DRTInitResult:
    success: bool
    failure_reason: Optional[str]
    retry_recommended: bool
    # Solved state (None when success=False)
    R0: Optional[torch.Tensor] = None        # (3,3) world-frame rotation at kf-0
    v0: Optional[torch.Tensor] = None        # (3,) world-frame velocity at kf-0
    p0: Optional[torch.Tensor] = None        # (3,) world-frame position at kf-0
    b_g: Optional[torch.Tensor] = None       # (3,) gyro bias
    b_a: Optional[torch.Tensor] = None       # (3,) accel bias (zeros from linear alignment)
    g_W: Optional[torch.Tensor] = None       # (3,) gravity in world frame
    scale: Optional[float] = None
    P_init: Optional[torch.Tensor] = None    # (15,15) prior covariance for EKF seed

    @staticmethod
    def failure(reason: str, retry: bool = True) -> "DRTInitResult":
        return DRTInitResult(success=False, failure_reason=reason, retry_recommended=retry)

    @staticmethod
    def success_result(
        R0: torch.Tensor, v0: torch.Tensor, p0: torch.Tensor,
        b_g: torch.Tensor, b_a: torch.Tensor, g_W: torch.Tensor,
        scale: float,
    ) -> "DRTInitResult":
        # Default P_init block-diagonal (from design doc §5.8)
        P = torch.zeros(15, 15, dtype=torch.float64)
        diag_vals = torch.tensor([
            (0.5 * 3.14159/180)**2,  # R (3)
            (0.5 * 3.14159/180)**2,
            (0.5 * 3.14159/180)**2,
            (0.05)**2,               # v (3)
            (0.05)**2,
            (0.05)**2,
            (0.01)**2,               # p (3)
            (0.01)**2,
            (0.01)**2,
            (0.005)**2,              # b_g (3)
            (0.005)**2,
            (0.005)**2,
            (0.05)**2,               # b_a (3)
            (0.05)**2,
            (0.05)**2,
        ], dtype=torch.float64)
        P[range(15), range(15)] = diag_vals
        return DRTInitResult(
            success=True, failure_reason=None, retry_recommended=False,
            R0=R0, v0=v0, p0=p0, b_g=b_g, b_a=b_a, g_W=g_W,
            scale=scale, P_init=P,
        )
