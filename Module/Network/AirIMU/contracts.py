from __future__ import annotations

from collections.abc import Mapping
from typing import TypedDict

import torch


class CorrectorCovState(TypedDict):
    acc_cov: torch.Tensor
    gyro_cov: torch.Tensor


class CorrectorOutput(TypedDict):
    correction_acc: torch.Tensor
    correction_gyro: torch.Tensor
    cov_state: CorrectorCovState


def narrow_corrector_output(raw: Mapping[str, object]) -> CorrectorOutput:
    correction_acc = raw.get("correction_acc")
    correction_gyro = raw.get("correction_gyro")
    cov_state_raw = raw.get("cov_state")
    if not isinstance(correction_acc, torch.Tensor):
        raise TypeError("corrector output is missing tensor field 'correction_acc'")
    if not isinstance(correction_gyro, torch.Tensor):
        raise TypeError("corrector output is missing tensor field 'correction_gyro'")
    if not isinstance(cov_state_raw, Mapping):
        raise TypeError("corrector output is missing mapping field 'cov_state'")

    acc_cov = cov_state_raw.get("acc_cov")
    gyro_cov = cov_state_raw.get("gyro_cov")
    if not isinstance(acc_cov, torch.Tensor):
        raise TypeError("corrector output is missing tensor field 'cov_state.acc_cov'")
    if not isinstance(gyro_cov, torch.Tensor):
        raise TypeError("corrector output is missing tensor field 'cov_state.gyro_cov'")

    return {
        "correction_acc": correction_acc,
        "correction_gyro": correction_gyro,
        "cov_state": {
            "acc_cov": acc_cov,
            "gyro_cov": gyro_cov,
        },
    }
