from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


def fit_temperature(
    *,
    logits: torch.Tensor,
    targets: torch.Tensor,
    temperatures: torch.Tensor | None = None,
) -> float:
    if temperatures is None:
        temperatures = torch.logspace(-1, 1, steps=81, dtype=logits.dtype, device=logits.device)
    targets = targets.to(dtype=logits.dtype, device=logits.device)

    best_t = float(temperatures[0].item())
    best_loss = np.inf
    for t in temperatures:
        probs = torch.sigmoid(logits / t.clamp(min=1e-6))
        loss = F.binary_cross_entropy(
            probs.clamp(min=1e-6, max=1.0 - 1e-6),
            targets.clamp(min=0.0, max=1.0),
        ).item()
        if loss < best_loss:
            best_loss = loss
            best_t = float(t.item())
    return best_t


def apply_temperature(head: torch.nn.Module, temperature: float) -> None:
    t = torch.tensor(float(temperature), dtype=head.T_calib.dtype, device=head.T_calib.device)
    head.T_calib.copy_(t)
