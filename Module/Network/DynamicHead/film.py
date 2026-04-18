from __future__ import annotations

import torch
import torch.nn as nn


class FiLMLayer(nn.Module):
    def __init__(self, in_dim: int, out_ch: int, hidden_dim: int = 128) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 2 * out_ch),
        )

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        affine = self.mlp(cond)
        gamma, beta = torch.chunk(affine, chunks=2, dim=-1)
        gamma = gamma.unsqueeze(-1).unsqueeze(-1)
        beta = beta.unsqueeze(-1).unsqueeze(-1)
        return x * (1.0 + gamma) + beta


class DepthBinFiLM(nn.Module):
    def __init__(self, in_dim: int, channels: int, num_bins: int = 16, hidden_dim: int = 128) -> None:
        super().__init__()
        assert num_bins >= 2
        self.num_bins = num_bins
        self.channels = channels
        self.mlp = nn.Sequential(
            nn.Linear(in_dim + num_bins, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 2 * channels),
        )

    def forward(self, x: torch.Tensor, cond: torch.Tensor, z_hat: torch.Tensor, z_min: float, z_max: float) -> torch.Tensor:
        z = z_hat.clamp(min=z_min, max=z_max)
        t = ((z - z_min) / max(z_max - z_min, 1e-6) * (self.num_bins - 1)).long()
        t = t.clamp_(0, self.num_bins - 1)
        onehot = torch.nn.functional.one_hot(t.squeeze(1), num_classes=self.num_bins).to(x.dtype)
        onehot = onehot.permute(0, 3, 1, 2)
        pooled = onehot.mean(dim=(2, 3))
        affine = self.mlp(torch.cat([cond, pooled], dim=-1))
        gamma, beta = torch.chunk(affine, chunks=2, dim=-1)
        gamma = gamma.unsqueeze(-1).unsqueeze(-1)
        beta = beta.unsqueeze(-1).unsqueeze(-1)
        return x * (1.0 + gamma) + beta

