"""
DRT-loose initializer bootstrap orchestrator.

`DRTLooseBootstrap` buffers incoming StereoInertialFrames until a window of
`min_keyframes` is ready, then runs the full solve pipeline:

    1. Preintegrate IMU segments between keyframes
    2. Solve gyro bias (SO3 residual LM)
    3. Compute camera rotations from IMU rotations + extrinsic
    4. Build LiGT LTL + recover translations (up to scale + sign)
    5. Run linear alignment for (v_0..v_{N-1}, scale, gravity)
    6. Check quality gates
    7. Return DRTInitResult (success) or DRTInitResult.failure(reason)

`run_drt_with_retry` wraps this with the retry/fallback policy.
"""

from __future__ import annotations

import torch
import pypose as pp

from .types import DRTInitConfig, DRTInitResult
from .preintegration import IMUPreintegrator, PreintResult
from .gyro_bias import solve_gyro_bias
from .translation import build_LTL, recover_translations, resolve_translation_sign, check_ltl_conditioning
from .alignment import linear_alignment, normalize_gravity
from .quality import run_all_quality_gates, check_gravity_consistency
from .tracks import FeatureTrack


class DRTLooseBootstrap:
    """Stateful bootstrap accumulator."""

    def __init__(
        self,
        cfg: DRTInitConfig,
        R_BC: torch.Tensor,
        t_BC: torch.Tensor,
        gravity_norm: float = 9.81007,
    ):
        self.cfg = cfg
        self.R_BC = R_BC            # (3,3) IMU→camera extrinsic rotation (float64)
        self.t_BC = t_BC            # (3,) IMU→camera extrinsic translation
        self.gravity_norm = gravity_norm
        self._keyframe_timestamps: list[float] = []
        self._imu_segments: list[list] = []    # list of raw IMU ticks per gap
        self._tracks: list[FeatureTrack] = []  # placeholder; real tracks from matcher
        self.reset()

    def reset(self) -> None:
        """Clear accumulated state for a retry."""
        self._keyframe_timestamps.clear()
        self._imu_segments.clear()
        self._tracks.clear()

    def is_window_ready(self) -> bool:
        return len(self._keyframe_timestamps) >= self.cfg.min_keyframes

    def solve(self) -> DRTInitResult:
        """Run full DRT-loose solve on the current window.

        Returns DRTInitResult (success or failure with reason).
        """
        n_kf = len(self._keyframe_timestamps)
        if n_kf < self.cfg.min_keyframes:
            return DRTInitResult.failure("INSUFFICIENT_OBS", retry=True)

        # Step 1: Preintegrate IMU segments
        preint_results: list[PreintResult] = []
        integrator = IMUPreintegrator()
        for segment in self._imu_segments:
            integrator.reset()
            for gyro, acc, dt in segment:
                integrator.integrate(gyro, acc, dt)
            preint_results.append(integrator.result())

        # Step 2: Solve gyro bias
        # (For now: dummy visual rotations from identity — real impl needs matcher)
        R_vis_list = [torch.eye(3, dtype=torch.float64) for _ in preint_results]
        b_g, converged = solve_gyro_bias(
            preint_results, R_vis_list, self.R_BC,
            huber_delta=self.cfg.huber_delta_gyro_rad,
        )

        # Step 3: Camera rotations from IMU (after bias correction)
        integrator_corrected = IMUPreintegrator(b_g=b_g)
        cam_rotations = [torch.eye(3, dtype=torch.float64)]  # kf-0 is identity
        for segment in self._imu_segments:
            integrator_corrected.reset()
            for gyro, acc, dt in segment:
                integrator_corrected.integrate(gyro, acc, dt)
            res = integrator_corrected.result()
            dR = pp.so3(res.dR_log).Exp().matrix().squeeze(0)
            R_prev = cam_rotations[-1]
            R_next = R_prev @ self.R_BC.T @ dR @ self.R_BC
            cam_rotations.append(R_next)

        # Step 4: LiGT translations (up-to-scale)
        LTL, A_lr = build_LTL(self._tracks, cam_rotations, n_keyframes=n_kf)

        if not check_ltl_conditioning(LTL, self.cfg.quality_max_cond):
            return DRTInitResult.failure("ILL_CONDITIONED", retry=True)

        t_flat = recover_translations(LTL)
        # Pad with kf-0 zero translation
        t_full = torch.cat([torch.zeros(3, dtype=torch.float64), t_flat])
        translations = t_full.view(n_kf, 3)

        # Resolve sign and apply
        if A_lr.shape[0] > 0:
            t_fragment = t_flat[:3]
            t_fragment = resolve_translation_sign(A_lr, t_fragment)
            sign = 1.0 if (A_lr @ t_fragment).sum() >= 0 else -1.0
            translations = translations * sign

        # Step 5: Linear alignment
        preint_corrected: list[PreintResult] = []
        integrator_corrected2 = IMUPreintegrator(b_g=b_g)
        for segment in self._imu_segments:
            integrator_corrected2.reset()
            for gyro, acc, dt in segment:
                integrator_corrected2.integrate(gyro, acc, dt)
            preint_corrected.append(integrator_corrected2.result())

        align_result = linear_alignment(
            preint_corrected, translations, cam_rotations, self.gravity_norm
        )

        if not align_result.success:
            return DRTInitResult.failure(align_result.reason or "GRAVITY_BAD", retry=True)

        # Step 6: Quality gates (simplified — full gate needs acc_norms, depth_signs from data)
        g_W = align_result.gravity_world
        passed, reason = check_gravity_consistency(
            g_W, self.gravity_norm, self.cfg.quality_gravity_tol_rel
        )
        if not passed:
            return DRTInitResult.failure("GRAVITY_BAD", retry=False)

        # Build output
        R0 = cam_rotations[0]            # identity (kf-0 is world origin)
        v0 = align_result.velocities[0]  # world-frame velocity at kf-0
        p0 = translations[0]             # zero (kf-0 is world origin)
        b_a = torch.zeros(3, dtype=torch.float64)

        return DRTInitResult.success_result(
            R0=R0, v0=v0, p0=p0,
            b_g=b_g, b_a=b_a, g_W=g_W,
            scale=align_result.scale,
        )


def run_drt_with_retry(
    cfg: DRTInitConfig,
    solver_fn,  # callable(window_scale: float) -> DRTInitResult | None
) -> DRTInitResult:
    """Try DRT solve with progressively larger windows.

    solver_fn(scale) should return a DRTInitResult if successful, or None to
    indicate the window at this scale is not ready / failed before quality gates.

    After all attempts exhausted, returns a failure with reason="FALLBACK_HEURISTIC".
    """
    attempts = min(cfg.max_attempts + 1, len(cfg.window_scales))
    for i in range(attempts):
        res = solver_fn(cfg.window_scales[i])
        if res is not None and res.success:
            return res
    return DRTInitResult(
        success=False,
        failure_reason="FALLBACK_HEURISTIC",
        retry_recommended=False,
    )
