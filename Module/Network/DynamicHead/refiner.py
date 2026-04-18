from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ReprojectionRefiner(nn.Module):
    def __init__(self, in_ch: int, hidden_ch: int = 64, out_ch: int = 2) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, hidden_ch, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_ch, hidden_ch, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_ch, out_ch, kernel_size=3, padding=1),
        )

    def forward(self, z_hat: torch.Tensor, e_raw: torch.Tensor) -> torch.Tensor:
        inp = torch.cat([z_hat, e_raw], dim=1)
        return self.net(inp)

