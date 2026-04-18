from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CadenceConfig:
    dump_every: int = 200
    forced_first_n: int = 10
    on_anomaly: bool = True


def should_dump(step: int, cfg: CadenceConfig, anomaly: bool = False, is_eval: bool = False) -> bool:
    if is_eval:
        return True
    if step < cfg.forced_first_n:
        return True
    if anomaly and cfg.on_anomaly:
        return True
    if cfg.dump_every <= 0:
        return False
    return (step % cfg.dump_every) == 0


def cadence_reason(step: int, cfg: CadenceConfig, anomaly: bool = False, is_eval: bool = False) -> str:
    if is_eval:
        return "eval"
    if step < cfg.forced_first_n:
        return "forced_early"
    if anomaly and cfg.on_anomaly:
        return "anomaly"
    return "regular"

