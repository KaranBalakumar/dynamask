from __future__ import annotations

import torch
import torch.nn as nn


class ConvGRUCell(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, kernel_size: int = 3):
        super().__init__()
        self.hidden_dim = hidden_dim
        pad = kernel_size // 2
        self.gates = nn.Conv2d(input_dim + hidden_dim, 2 * hidden_dim, kernel_size, padding=pad)
        self.cand = nn.Conv2d(input_dim + hidden_dim, hidden_dim, kernel_size, padding=pad)

    def forward(self, x: torch.Tensor, h_prev: torch.Tensor | None) -> torch.Tensor:
        b, _, h, w = x.shape
        if h_prev is None:
            h_prev = torch.zeros((b, self.hidden_dim, h, w), device=x.device, dtype=x.dtype)
        z_r = torch.sigmoid(self.gates(torch.cat([h_prev, x], dim=1)))
        z, r = torch.chunk(z_r, chunks=2, dim=1)
        q = torch.tanh(self.cand(torch.cat([r * h_prev, x], dim=1)))
        return (1.0 - z) * h_prev + z * q


class ConvLSTMCell(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, kernel_size: int = 3):
        super().__init__()
        self.hidden_dim = hidden_dim
        pad = kernel_size // 2
        self.conv = nn.Conv2d(input_dim + hidden_dim, 4 * hidden_dim, kernel_size, padding=pad)

    def forward(
        self,
        x: torch.Tensor,
        h_prev: torch.Tensor | None,
        c_prev: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        b, _, h, w = x.shape
        if h_prev is None:
            h_prev = torch.zeros((b, self.hidden_dim, h, w), device=x.device, dtype=x.dtype)
        if c_prev is None:
            c_prev = torch.zeros((b, self.hidden_dim, h, w), device=x.device, dtype=x.dtype)

        i, f, o, g = torch.chunk(self.conv(torch.cat([h_prev, x], dim=1)), chunks=4, dim=1)
        i, f, o = torch.sigmoid(i), torch.sigmoid(f), torch.sigmoid(o)
        g = torch.tanh(g)
        c = f * c_prev + i * g
        h = o * torch.tanh(c)
        return h, c

