from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class FiLM(nn.Module):
    def __init__(self, channels: int, imu_dim: int):
        super().__init__()
        self.gamma = nn.Linear(imu_dim, channels)
        self.beta = nn.Linear(imu_dim, channels)

    def forward(self, x: torch.Tensor, f_imu: torch.Tensor) -> torch.Tensor:
        g = self.gamma(f_imu).unsqueeze(-1).unsqueeze(-1)
        b = self.beta(f_imu).unsqueeze(-1).unsqueeze(-1)
        return g * x + b


class DepthBinFiLM(nn.Module):
    def __init__(
        self,
        channels: int,
        imu_dim: int,
        depth_bins: tuple[float, ...] = (0.0, 5.0, 20.0, 1e6),
    ):
        super().__init__()
        if len(depth_bins) < 2:
            raise ValueError("depth_bins should contain at least 2 edges")
        self.depth_bins = depth_bins
        self.num_bins = len(depth_bins) - 1
        self.gamma = nn.Linear(imu_dim, channels * self.num_bins)
        self.beta = nn.Linear(imu_dim, channels * self.num_bins)

    def forward(self, x: torch.Tensor, f_imu: torch.Tensor, depth: torch.Tensor | None) -> torch.Tensor:
        b, c, h, w = x.shape
        if depth is None:
            g = self.gamma(f_imu).view(b, self.num_bins, c).mean(dim=1).unsqueeze(-1).unsqueeze(-1)
            z = self.beta(f_imu).view(b, self.num_bins, c).mean(dim=1).unsqueeze(-1).unsqueeze(-1)
            return g * x + z

        if depth.shape[-2:] != (h, w):
            depth = F.interpolate(depth, size=(h, w), mode="bilinear", align_corners=False)

        gamma = self.gamma(f_imu).view(b, self.num_bins, c)
        beta = self.beta(f_imu).view(b, self.num_bins, c)

        mask = []
        for d0, d1 in zip(self.depth_bins[:-1], self.depth_bins[1:]):
            mask.append(((depth >= d0) & (depth < d1)).to(dtype=x.dtype))
        m = torch.stack(mask, dim=1)  # B,K,1,H,W
        norm = m.sum(dim=1, keepdim=True).clamp(min=1.0)
        m = m / norm

        out = torch.zeros_like(x)
        for i in range(self.num_bins):
            g = gamma[:, i, :].unsqueeze(-1).unsqueeze(-1)
            z = beta[:, i, :].unsqueeze(-1).unsqueeze(-1)
            out = out + m[:, i] * (g * x + z)
        return out


class CrossAttentionIMUFusion(nn.Module):
    """
    Tiny spatial cross-attention from visual tokens to an IMU token bank.
    """

    def __init__(self, channels: int, imu_dim: int, num_heads: int = 4, token_bank_size: int = 4):
        super().__init__()
        if channels % num_heads != 0:
            raise ValueError("channels must be divisible by num_heads")
        self.channels = channels
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        self.scale = self.head_dim**-0.5

        self.q_proj = nn.Linear(channels, channels)
        self.k_proj = nn.Linear(imu_dim, channels)
        self.v_proj = nn.Linear(imu_dim, channels)
        self.out_proj = nn.Linear(channels, channels)
        self.norm = nn.LayerNorm(channels)

        self.imu_token_bank = nn.Parameter(torch.zeros(token_bank_size, imu_dim))
        nn.init.normal_(self.imu_token_bank, mean=0.0, std=0.02)

    def forward(self, x: torch.Tensor, f_imu: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        tokens = x.flatten(2).transpose(1, 2)  # B,HW,C
        q = self.q_proj(self.norm(tokens))

        imu_tokens = torch.cat(
            [
                f_imu.unsqueeze(1),
                self.imu_token_bank.unsqueeze(0).expand(b, -1, -1),
            ],
            dim=1,
        )  # B,T,D
        k = self.k_proj(imu_tokens)
        v = self.v_proj(imu_tokens)

        q = q.view(b, h * w, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(b, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(b, -1, self.num_heads, self.head_dim).transpose(1, 2)

        attn = torch.softmax((q @ k.transpose(-2, -1)) * self.scale, dim=-1)
        ctx = attn @ v
        ctx = ctx.transpose(1, 2).contiguous().view(b, h * w, c)
        fused = self.out_proj(ctx) + tokens
        return fused.transpose(1, 2).reshape(b, c, h, w)
