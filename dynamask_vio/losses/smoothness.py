"""Edge-aware second-order smoothness loss for score logits."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def edge_aware_second_order_smoothness(
    score_logit: torch.Tensor,
    image: torch.Tensor,
    *,
    beta: float = 10.0,
) -> torch.Tensor:
    """Second-order edge-aware smoothness for score logits."""
    if image.shape[-2:] != score_logit.shape[-2:]:
        image = F.interpolate(image, size=score_logit.shape[-2:], mode="bilinear", align_corners=False)

    # First-order image gradients (edge indicator).
    img_dx = (image[:, :, :, 1:] - image[:, :, :, :-1]).abs().mean(dim=1, keepdim=True)
    img_dy = (image[:, :, 1:, :] - image[:, :, :-1, :]).abs().mean(dim=1, keepdim=True)

    # Second-order score derivatives.
    score_dxx = score_logit[:, :, :, 2:] - 2.0 * score_logit[:, :, :, 1:-1] + score_logit[:, :, :, :-2]
    score_dyy = score_logit[:, :, 2:, :] - 2.0 * score_logit[:, :, 1:-1, :] + score_logit[:, :, :-2, :]

    # Align edge-aware weights to second-order derivative shapes.
    weight_x = torch.exp(-beta * img_dx[:, :, :, 1:])
    weight_y = torch.exp(-beta * img_dy[:, :, 1:, :])

    loss_x = (score_dxx.abs() * weight_x).mean()
    loss_y = (score_dyy.abs() * weight_y).mean()
    return loss_x + loss_y
