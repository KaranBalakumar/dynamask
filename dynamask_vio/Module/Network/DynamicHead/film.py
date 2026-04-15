import torch
import torch.nn as nn


class FiLM(nn.Module):
    def __init__(self, feature_dim: int, cond_dim: int):
        super().__init__()
        self.feature_dim = feature_dim
        self.cond_dim = cond_dim
        self.to_scale_shift = nn.Linear(cond_dim, feature_dim * 2)

        nn.init.xavier_uniform_(self.to_scale_shift.weight)
        with torch.no_grad():
            self.to_scale_shift.bias[:feature_dim].fill_(1.0)
            self.to_scale_shift.bias[feature_dim:].zero_()

    def forward(self, x: torch.Tensor, cond: torch.Tensor | None) -> torch.Tensor:
        if cond is None:
            return x
        assert cond.ndim == 2, "FiLM conditioning tensor must be [B, D]."
        gamma_beta = self.to_scale_shift(cond).unsqueeze(-1).unsqueeze(-1)
        gamma, beta = torch.chunk(gamma_beta, 2, dim=1)
        return gamma * x + beta
