from __future__ import annotations

from dataclasses import dataclass


KNOWN_SCALAR_KEYS = {
    "timing.data.ms",
    "timing.frontend.ms",
    "timing.init.ms",
    "timing.backend.ms",
    "frontend.imu.preintegrator.dt_s",
    "frontend.imu.preintegrator.delta_R.angle_rad",
    "frontend.imu.preintegrator.delta_v.norm",
    "frontend.imu.preintegrator.delta_p.norm",
    "frontend.imu.preintegrator.Sigma.cond",
    "frontend.imu.preintegrator.Sigma.eigmin",
    "frontend.imu.preintegrator.Sigma.trace",
    "frontend.imu.preintegrator.J_R_bg.fro",
    "frontend.imu.preintegrator.J_v_ba.fro",
    "frontend.imu.preintegrator.J_p_ba.fro",
    "diag.frontend.imu.preintegrator.sigma_non_psd",
    "init.drt.window_len",
    "init.drt.n_views.total",
    "init.drt.parallax_px.p50",
    "init.drt.parallax_px.p5",
    "init.drt.accept",
    "init.drt.bg_solver.residual",
    "init.drt.bg.norm",
    "init.drt.gW.mag",
    "init.drt.gW.mag_err",
    "init.drt.scale_s",
    "backend.twoframe.lm.iters",
    "backend.twoframe.lm.converged",
    "backend.twoframe.chi2.init",
    "backend.twoframe.chi2.final",
    "backend.twoframe.chi2.delta_frac",
    "backend.twoframe.r_vis.norm.p50",
    "backend.twoframe.r_vis.norm.p95",
    "backend.twoframe.r_vis.maha.p50",
    "backend.twoframe.r_vis.maha.gt_3sig_frac",
    "backend.imu_residual.r_R.norm",
    "backend.imu_residual.r_v.norm",
    "backend.imu_residual.r_p.norm",
    "backend.imu_residual.r_bg.norm",
    "backend.imu_residual.r_ba.norm",
    "backend.imu_residual.maha.total",
    "backend.twoframe.w_eps_hits",
    "backend.twoframe.jacobian.cond",
    "backend.twoframe.time_ms",
    "backend.swf.W_kf",
    "backend.swf.n_visual_factors",
    "backend.swf.n_imu_factors",
    "backend.swf.lm.iters",
    "backend.swf.chi2.init",
    "backend.swf.chi2.final",
    "backend.swf.chi2.delta_frac",
    "backend.swf.time_ms",
    "backend.writeback.pose.delta.trans_m",
    "backend.writeback.pose.delta.rot_deg",
    "backend.writeback.vel.delta_norm",
    "backend.writeback.bias_g.delta_norm",
    "backend.writeback.bias_a.delta_norm",
    "backend.writeback.bias_g.norm",
    "backend.writeback.bias_a.norm",
    "diag.backend.writeback.pose_jump_warn",
}

KNOWN_PREFIXES = (
    "diag.nan.",
    "timing.",
    "timing.phase.",
)


@dataclass(frozen=True)
class KeyValidationResult:
    ok: bool
    reason: str


def validate_key(key: str) -> KeyValidationResult:
    if key in KNOWN_SCALAR_KEYS:
        return KeyValidationResult(True, "exact")
    if any(key.startswith(prefix) for prefix in KNOWN_PREFIXES):
        return KeyValidationResult(True, "prefix")
    return KeyValidationResult(False, f"Unknown logging key: {key}")
