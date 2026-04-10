"""
FiLM (Feature-wise Linear Modulation) conditioning layers for DynaMask V2.

Applied after each residual stage of the RAFT feature encoder to modulate
visual features with IMU ego-motion information.

Stage channels (RAFT BasicEncoder):
  Stage 1: 64 channels
  Stage 2: 96 channels
  Stage 3: 128 channels

Initialization: gamma ≈ 1, beta ≈ 0 so FiLM acts as identity at the
start of training. This ensures the encoder behaves identically to the
pretrained RAFT checkpoint before IMU conditioning is learned.
"""

import torch
import torch.nn as nn


class FiLM(nn.Module):
    """Single FiLM layer for one feature scale.

    forward(features, f_imu):
        features: [B, C, H, W]
        f_imu:    [B, imu_dim]
        returns:  gamma * features + beta   (same shape as features)
    """

    def __init__(self, imu_dim: int, feature_channels: int):
        super().__init__()
        self.gamma_proj = nn.Linear(imu_dim, feature_channels)
        self.beta_proj = nn.Linear(imu_dim, feature_channels)

        # Init gamma projection so output ≈ 0 → gamma = 0 + 1 = 1
        nn.init.normal_(self.gamma_proj.weight, std=0.01)
        nn.init.zeros_(self.gamma_proj.bias)

        # Init beta projection so output ≈ 0
        nn.init.zeros_(self.beta_proj.weight)
        nn.init.zeros_(self.beta_proj.bias)

    def forward(self, features: torch.Tensor,
                f_imu: torch.Tensor) -> torch.Tensor:
        # f_imu: [B, imu_dim] → [B, C] → [B, C, 1, 1]
        gamma = self.gamma_proj(f_imu).unsqueeze(-1).unsqueeze(-1) + 1.0
        beta = self.beta_proj(f_imu).unsqueeze(-1).unsqueeze(-1)
        return gamma * features + beta


def create_film_layers(imu_dim: int = 128,
                       stage_channels: list = None) -> nn.ModuleList:
    """Create FiLM layers for RAFT feature encoder's 3 residual stages.

    Args:
        imu_dim: dimensionality of f_imu vector
        stage_channels: channel count at each stage [64, 96, 128]

    Returns:
        nn.ModuleList of 3 FiLM layers
    """
    if stage_channels is None:
        stage_channels = [64, 96, 128]

    return nn.ModuleList([
        FiLM(imu_dim, ch) for ch in stage_channels
    ])
