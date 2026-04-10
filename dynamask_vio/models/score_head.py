"""Score head for DynaMask V2.5."""

from __future__ import annotations

import torch
import torch.nn as nn


class _GradientClipFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, clip_val: float):
        ctx.clip_val = clip_val
        return x

    @staticmethod
    def backward(ctx, grad_x: torch.Tensor):
        grad_x = torch.where(torch.isnan(grad_x), torch.zeros_like(grad_x), grad_x)
        return grad_x.clamp(min=-ctx.clip_val, max=ctx.clip_val), None


class GradientClip(nn.Module):
    def __init__(self, clip_val: float = 0.01):
        super().__init__()
        self.clip_val = float(clip_val)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return _GradientClipFn.apply(x, self.clip_val)


class ScoreHead(nn.Module):
    """Conv score head that outputs clipped logits (no sigmoid)."""

    def __init__(self, hidden_dim: int = 128, grad_clip: float = 0.01, logit_clip: float = 10.0):
        super().__init__()
        self.conv1 = nn.Conv2d(hidden_dim, 64, 3, padding=1)
        self.conv2 = nn.Conv2d(64, 1, 1)
        self.relu = nn.ReLU(inplace=True)
        self.grad_clip = GradientClip(clip_val=grad_clip)
        self.logit_clip = float(logit_clip)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        score_logit = self.conv2(self.relu(self.conv1(h)))
        score_logit = self.grad_clip(score_logit)
        # Keep score logits numerically stable for calibration + fp16 export.
        score_logit = score_logit.clamp(min=-self.logit_clip, max=self.logit_clip)
        return score_logit
