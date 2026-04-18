from __future__ import annotations

from Utility.Observability.keys_runtime import validate_key


def test_runtime_key_registry_accepts_expected_keys():
    assert validate_key("backend.twoframe.chi2.init").ok
    assert validate_key("timing.backend.ms").ok
    assert validate_key("diag.nan.backend.twoframe.chi2.init").ok


def test_runtime_key_registry_rejects_unknown_key():
    assert not validate_key("some.random.key").ok

