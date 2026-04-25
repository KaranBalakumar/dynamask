import torch

from Module.Initialization.DRTLoose.types import DRTInitConfig, DRTInitResult
from Module.Initialization.DRTLoose.bootstrap import run_drt_with_retry


def test_drt_result_schema_defaults():
    cfg = DRTInitConfig()
    out = DRTInitResult.failure("NO_OBSERVABILITY")
    assert cfg.min_keyframes == 10
    assert out.success is False
    assert out.failure_reason == "NO_OBSERVABILITY"
    assert out.retry_recommended is True


def test_module_exports_drt_types():
    from Module import DRTInitConfig as TopLevelDRTInitConfig
    assert TopLevelDRTInitConfig is DRTInitConfig


def test_drt_init_config_retry_defaults():
    cfg = DRTInitConfig()
    assert cfg.max_attempts == 2
    assert cfg.window_scales == (1.0, 1.4, 1.8)


def test_drt_init_result_success_factory():
    res = DRTInitResult(
        success=True,
        failure_reason=None,
        retry_recommended=False,
    )
    assert res.success is True
    assert res.failure_reason is None


def test_retry_then_fallback_policy():
    cfg = DRTInitConfig(max_attempts=2, window_scales=(1.0, 1.4, 1.8))
    out = run_drt_with_retry(cfg, solver_fn=lambda scale: None)
    assert out.success is False
    assert out.failure_reason == "FALLBACK_HEURISTIC"


def test_retry_succeeds_on_second_attempt():
    attempts = []

    def solver(scale):
        attempts.append(scale)
        if scale >= 1.4:
            return DRTInitResult(success=True, failure_reason=None, retry_recommended=False)
        return None

    cfg = DRTInitConfig(max_attempts=2, window_scales=(1.0, 1.4, 1.8))
    out = run_drt_with_retry(cfg, solver_fn=solver)
    assert out.success is True
    assert len(attempts) == 2   # tried 1.0 and 1.4


def test_drt_init_result_success_factory_has_p_init():
    res = DRTInitResult.success_result(
        R0=torch.eye(3, dtype=torch.float64),
        v0=torch.zeros(3, dtype=torch.float64),
        p0=torch.zeros(3, dtype=torch.float64),
        b_g=torch.zeros(3, dtype=torch.float64),
        b_a=torch.zeros(3, dtype=torch.float64),
        g_W=torch.tensor([0., 0., 9.81], dtype=torch.float64),
        scale=1.0,
    )
    assert res.success is True
    assert res.P_init is not None
    assert res.P_init.shape == (15, 15)
    assert (res.P_init >= 0).all()


def test_drt_init_result_failure_retry_flag():
    out_retry = DRTInitResult.failure("ILL_CONDITIONED", retry=True)
    assert out_retry.retry_recommended is True

    out_no_retry = DRTInitResult.failure("GRAVITY_BAD", retry=False)
    assert out_no_retry.retry_recommended is False


def test_drt_init_config_quality_gate_thresholds():
    cfg = DRTInitConfig()
    assert cfg.quality_min_avg_obs == 30
    assert cfg.quality_max_cond == 1e8
    assert cfg.quality_min_pos_depth_ratio == 0.7
    assert cfg.quality_gravity_tol_rel == 1e-3
    assert cfg.fallback == "heuristic"


def test_retry_first_attempt_succeeds():
    calls = []

    def solver(scale):
        calls.append(scale)
        return DRTInitResult(success=True, failure_reason=None, retry_recommended=False)

    cfg = DRTInitConfig(max_attempts=2, window_scales=(1.0, 1.4, 1.8))
    out = run_drt_with_retry(cfg, solver_fn=solver)
    assert out.success is True
    assert len(calls) == 1   # stops after first success


def test_retry_respects_max_attempts():
    calls = []

    def solver(scale):
        calls.append(scale)
        return None  # always fails

    # max_attempts=1 => attempts = min(2, 3) = 2 window scales tried
    cfg = DRTInitConfig(max_attempts=1, window_scales=(1.0, 1.4, 1.8))
    out = run_drt_with_retry(cfg, solver_fn=solver)
    assert out.success is False
    assert len(calls) == 2  # min(max_attempts+1=2, len(window_scales)=3) = 2
