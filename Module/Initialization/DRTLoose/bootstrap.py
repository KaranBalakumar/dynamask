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
from .tracks import _bearing
from .translation import build_LTL, recover_translations, recover_translations_stereo, resolve_translation_sign, check_ltl_conditioning
from .alignment import linear_alignment, normalize_gravity
from .quality import run_all_quality_gates, check_gravity_consistency, check_min_parallax
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
        self._depth_maps: list[torch.Tensor] | None = None  # stereo depth per keyframe
        self._K: torch.Tensor | None = None   # (3,3) camera intrinsics
        self.reset()

    def set_stereo_depth(self, depth_maps: list[torch.Tensor], K: torch.Tensor) -> None:
        """Provide stereo depth maps for metric translation recovery.

        Parameters
        ----------
        depth_maps : list of (1,1,H,W) or (1,H,W) depth tensors, one per keyframe.
        K          : (3,3) camera intrinsics matrix.
        """
        self._depth_maps = depth_maps
        self._K = K

    def reset(self) -> None:
        """Clear accumulated state for a retry."""
        self._keyframe_timestamps.clear()
        self._imu_segments.clear()
        self._tracks.clear()
        self._depth_maps = None
        self._K = None

    def is_window_ready(self) -> bool:
        return len(self._keyframe_timestamps) >= self.cfg.min_keyframes

    def solve(self) -> DRTInitResult:
        """Run full DRT-loose solve on the current window.

        Returns DRTInitResult (success or failure with reason).
        """
        n_kf = len(self._keyframe_timestamps)
        if n_kf < self.cfg.min_keyframes:
            return DRTInitResult.failure("INSUFFICIENT_OBS", retry=True)

        # Gate 0: minimum per-pair parallax — reject windows with near-stationary
        # pairs since the bearing-vector gyro-bias cost is degenerate there.
        passed, reason = check_min_parallax(
            self._tracks, n_kf, self.cfg.quality_min_per_pair_parallax
        )
        if not passed:
            return DRTInitResult.failure(reason, retry=True)

        # Step 1: Preintegrate IMU segments
        preint_results: list[PreintResult] = []
        integrator = IMUPreintegrator()
        for segment in self._imu_segments:
            integrator.reset()
            for gyro, acc, dt in segment:
                integrator.integrate(gyro, acc, dt)
            preint_results.append(integrator.result())

        # Step 2: Solve gyro bias using bearing-vector epipolar cost
        # For each consecutive pair collect matching 3-D bearing vectors from
        # the tracked features (no essential-matrix decomposition needed).
        bearing_pairs: list[tuple[torch.Tensor, torch.Tensor]] = []
        for i in range(n_kf - 1):
            fis, fjs = [], []
            for track in self._tracks:
                if i in track.obs and (i + 1) in track.obs:
                    # _bearing lifts (2,) normalised coords → (3,) unit ray
                    fis.append(_bearing(track.obs[i]))
                    fjs.append(_bearing(track.obs[i + 1]))
            if len(fis) < 8:
                # Insufficient correspondences for this gap — use placeholder zeros;
                # the pair contributes near-zero cost and gradient.
                bearing_pairs.append((
                    torch.zeros(1, 3, dtype=torch.float64),
                    torch.zeros(1, 3, dtype=torch.float64),
                ))
            else:
                bearing_pairs.append((torch.stack(fis), torch.stack(fjs)))

        b_g, converged = solve_gyro_bias(
            preint_results, bearing_pairs, self.R_BC,
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

        # Step 4: LiGT translations (up-to-scale) — good shape from global SVD
        print(f"\n=== Python LiGT Translation Debug ===")
        LTL, A_lr = build_LTL(self._tracks, cam_rotations, n_keyframes=n_kf)

        if not check_ltl_conditioning(LTL, self.cfg.quality_max_cond):
            return DRTInitResult.failure("ILL_CONDITIONED", retry=True)

        t_flat = recover_translations(LTL)
        print(f"t_flat (from SVD):")
        for i in range(min(5, len(t_flat))):
            print(f"  t_flat[{i}] = {t_flat[i].item():.8f}")

        t_full = torch.cat([torch.zeros(3, dtype=torch.float64), t_flat])
        translations = t_full.view(n_kf, 3)
        print(f"translations (reshaped):")
        for i in range(min(3, n_kf)):
            print(f"  t[{i}] = {translations[i].tolist()}")

        # Step 5: Linear alignment
        preint_corrected: list[PreintResult] = []
        integrator_corrected2 = IMUPreintegrator(b_g=b_g)
        for segment in self._imu_segments:
            integrator_corrected2.reset()
            for gyro, acc, dt in segment:
                integrator_corrected2.integrate(gyro, acc, dt)
            preint_corrected.append(integrator_corrected2.result())

        # Try both translation signs; positive scale is a hard physical constraint.
        align_result = linear_alignment(
            preint_corrected, translations, cam_rotations, self.gravity_norm, t_BC=self.t_BC
        )
        if not align_result.success and align_result.reason and "non-positive" in align_result.reason:
            align_result = linear_alignment(
                preint_corrected, -translations, cam_rotations, self.gravity_norm, t_BC=self.t_BC
            )

        # Stereo tiebreaker: if both signs give positive scale, use stereo
        # depth direction to choose.  (Rare edge case — usually positive scale
        # is unambiguous.)
        both_positive = (align_result.success and align_result.scale > 0)
        if both_positive and self._depth_maps is not None and self._K is not None:
            alt_result = linear_alignment(
                preint_corrected, -translations, cam_rotations, self.gravity_norm,
                t_BC=self.t_BC,
            )
            if alt_result.success and alt_result.scale > align_result.scale:
                t_stereo = recover_translations_stereo(
                    self._tracks, cam_rotations, self._depth_maps,
                    self._K.to(dtype=torch.float64), n_kf
                )
                # Compare first non-zero segment direction
                t_ligt_dir = translations[1] / translations[1].norm().clamp_min(1e-10)
                t_stereo_dir = t_stereo[1] / t_stereo[1].norm().clamp_min(1e-10)
                if torch.dot(t_ligt_dir, t_stereo_dir) < 0:
                    align_result = alt_result

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
    last_res: DRTInitResult | None = None
    for i in range(attempts):
        res = solver_fn(cfg.window_scales[i])
        if res is not None and res.success:
            return res
        if res is not None:
            last_res = res
    # If every attempt recommended retry (e.g. LOW_PARALLAX), bubble that up
    # so the caller can slide the window rather than fall back to heuristic.
    if last_res is not None and last_res.retry_recommended:
        return last_res
    return DRTInitResult(
        success=False,
        failure_reason="FALLBACK_HEURISTIC",
        retry_recommended=False,
    )
