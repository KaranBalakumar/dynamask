"""V2.5 self-supervised loss primitives."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from ..models.preintegration import so3_log_map


def pose_consistency_loss(
    R_pred: torch.Tensor,
    t_pred: torch.Tensor,
    R_gt: torch.Tensor,
    t_gt: torch.Tensor,
    valid_mask: torch.Tensor | None = None,
    *,
    rot_weight: float = 10.0,
    trans_weight: float = 1.0,
    rot_huber_delta: float = 0.1,
    trans_huber_delta: float = 0.5,
) -> torch.Tensor:
    """SE(3) pose error loss with robust rotation/translation terms."""
    R_err = R_pred @ R_gt.transpose(-1, -2)
    rot_vec = so3_log_map(R_err)
    rot_mag = rot_vec.norm(dim=-1)
    rot_loss = F.huber_loss(
        rot_mag,
        torch.zeros_like(rot_mag),
        delta=rot_huber_delta,
        reduction="none",
    )

    trans_mag = (t_pred - t_gt).norm(dim=-1)
    trans_loss = F.huber_loss(
        trans_mag,
        torch.zeros_like(trans_mag),
        delta=trans_huber_delta,
        reduction="none",
    )

    per_sample = rot_weight * rot_loss + trans_weight * trans_loss
    if valid_mask is not None:
        weight = valid_mask.float()
        return (per_sample * weight).sum() / weight.sum().clamp(min=1.0)
    return per_sample.mean()
