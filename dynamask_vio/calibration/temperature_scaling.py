"""Post-training temperature scaling for score logits."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def apply_temperature(score_logit: torch.Tensor, temperature: float) -> torch.Tensor:
    """Apply scalar temperature and return calibrated score in [0,1]."""
    t = max(float(temperature), 1e-3)
    return torch.sigmoid(score_logit / t)


def fit_temperature(
    score_logit: torch.Tensor,
    pseudo_labels: torch.Tensor,
    *,
    init_temperature: float = 1.0,
    max_iter: int = 50,
) -> float:
    """Fit scalar temperature by minimizing BCE on held-out logits/labels.

    Args:
        score_logit: any shape tensor of raw logits.
        pseudo_labels: same shape tensor with targets in [0,1].
    """
    logits = score_logit.detach().reshape(-1).float()
    labels = pseudo_labels.detach().reshape(-1).float().clamp(0.0, 1.0)
    if logits.numel() == 0:
        return float(init_temperature)

    log_t = torch.tensor([float(init_temperature)]).log().requires_grad_(True)
    optimizer = torch.optim.LBFGS([log_t], lr=0.1, max_iter=max_iter, line_search_fn="strong_wolfe")

    def closure():
        optimizer.zero_grad(set_to_none=True)
        t = torch.exp(log_t).clamp(min=1e-3, max=1e3)
        loss = F.binary_cross_entropy_with_logits(logits / t, labels)
        loss.backward()
        return loss

    optimizer.step(closure)
    temperature = torch.exp(log_t).clamp(min=1e-3, max=1e3).item()
    return float(temperature)
