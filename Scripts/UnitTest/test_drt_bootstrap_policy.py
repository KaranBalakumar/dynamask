import torch
from Module.Initialization.DRTLoose.types import DRTInitConfig, DRTInitResult


def test_drt_result_schema_defaults():
    cfg = DRTInitConfig()
    out = DRTInitResult.failure("NO_OBSERVABILITY")
    assert cfg.min_keyframes == 10
    assert out.success is False
    assert out.failure_reason == "NO_OBSERVABILITY"
    assert out.retry_recommended is True
