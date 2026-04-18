from __future__ import annotations

from collections import defaultdict
from types import SimpleNamespace
from typing import Any

from torch.utils.data import DataLoader

from DataLoader import DataFramePair, StereoInertialFrame
from Train.DynamicHead.loop import DynamicHeadTrainer


def evaluate_loader(
    trainer: DynamicHeadTrainer,
    loader: DataLoader[DataFramePair[StereoInertialFrame]],
) -> dict[str, float]:
    acc: dict[str, float] = defaultdict(float)
    n = 0
    for pair in loader:
        out = trainer.eval_step(pair)
        n += 1
        for k, v in out.metrics.items():
            acc[k] += float(v)
    if n == 0:
        return {"loss_total": 0.0}
    return {k: v / n for k, v in acc.items()}

