from Module.Initialization.DRTLoose.types import DRTInitConfig, DRTInitResult


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
    import torch
    import pypose as pp
    res = DRTInitResult(
        success=True,
        failure_reason=None,
        retry_recommended=False,
    )
    assert res.success is True
    assert res.failure_reason is None
