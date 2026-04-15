import torch
import torch.nn as nn


class AirIMUCorrector(nn.Module):
    def __init__(self, hidden_dim: int = 64):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(6, hidden_dim),
            nn.SiLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(inplace=True),
            nn.Linear(hidden_dim, 12),
        )

    def freeze(self) -> None:
        self.eval()
        for param in self.parameters():
            param.requires_grad_(False)

    def forward(self, acc: torch.Tensor, gyro: torch.Tensor) -> dict[str, torch.Tensor]:
        assert acc.shape == gyro.shape and acc.shape[-1] == 3
        raw = torch.cat([acc, gyro], dim=-1)
        corr = self.mlp(raw)

        d_acc, d_gyro, d_bg, d_ba = torch.chunk(corr, 4, dim=-1)
        corrected_acc = acc + d_acc
        corrected_gyro = gyro + d_gyro

        return {
            "acc": corrected_acc,
            "gyro": corrected_gyro,
            "delta_bg": d_bg.mean(dim=1),
            "delta_ba": d_ba.mean(dim=1),
        }
