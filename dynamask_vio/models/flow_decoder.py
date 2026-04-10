"""
RAFT-style Flow Decoder (Update Operator) for DynaMask V2.

Replaces the V1 TemporalDecoder with a proper iterative flow refinement:
  1. All-pairs correlation volume with 4-level pyramid
  2. Motion encoder (correlation features + current flow)
  3. Separable ConvGRU (iterative refinement)
  4. Output heads:
     - FlowHead (internal — drives GRU iterations, feeds BA at train time)
     - MaskHead (exported — dynamic probability per pixel)

The flow is NOT exported at inference. It is an internal mechanism that
drives the GRU state from which the mask is decoded.

Reference: Teed & Deng, "RAFT", ECCV 2020.
GradientClip adopted from DPVO (Teed et al., NeurIPS 2023).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class GradientClip(torch.autograd.Function):
    """Per-element gradient clamp: identity forward, clamp backward to [-c, c].

    Adopted from DPVO (dpvo/blocks.py:74-82).
    """

    @staticmethod
    def forward(ctx, x, clip_val=0.01):
        ctx.clip_val = clip_val
        return x

    @staticmethod
    def backward(ctx, grad_x):
        grad_x = torch.where(torch.isnan(grad_x),
                              torch.zeros_like(grad_x), grad_x)
        return grad_x.clamp(min=-ctx.clip_val, max=ctx.clip_val), None


class GradClipModule(nn.Module):
    """nn.Module wrapper for GradientClip autograd function."""

    def __init__(self, clip_val: float = 0.01):
        super().__init__()
        self.clip_val = clip_val

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return GradientClip.apply(x, self.clip_val)


def _bilinear_sampler(img, coords):
    """Bilinear sampling from img at coords.

    Args:
        img: [B, C, H, W]
        coords: [B, 2, H_out, W_out]  (x, y coordinates)

    Returns:
        [B, C, H_out, W_out]
    """
    H, W = img.shape[-2:]
    xgrid = 2 * coords[:, 0] / max(W - 1, 1) - 1
    ygrid = 2 * coords[:, 1] / max(H - 1, 1) - 1
    grid = torch.stack([xgrid, ygrid], dim=-1)
    return F.grid_sample(img, grid, align_corners=True,
                          mode="bilinear", padding_mode="zeros")


class CorrBlock:
    """All-pairs correlation volume with multi-level pyramid.

    Computes all-pairs dot product between feature maps, then builds a
    pyramid by average-pooling the last two dimensions. Lookups are
    done via bilinear sampling from each level.
    """

    def __init__(self, fmap1: torch.Tensor, fmap2: torch.Tensor,
                 num_levels: int = 4, radius: int = 4):
        self.num_levels = num_levels
        self.radius = radius

        B, C, H, W = fmap1.shape
        self.H, self.W = H, W

        # All-pairs correlation: [B, H*W, 1, H, W]
        fmap1_flat = fmap1.view(B, C, H * W)
        fmap2_flat = fmap2.view(B, C, H * W)

        # Normalise for stable dot products
        fmap1_flat = fmap1_flat / (fmap1_flat.norm(dim=1, keepdim=True) + 1e-8)
        fmap2_flat = fmap2_flat / (fmap2_flat.norm(dim=1, keepdim=True) + 1e-8)

        corr = torch.matmul(fmap1_flat.transpose(1, 2), fmap2_flat)  # [B, H*W, H*W]
        corr = corr.view(B * H * W, 1, H, W)

        # Build pyramid
        self.corr_pyramid = [corr]
        for _ in range(num_levels - 1):
            corr = F.avg_pool2d(corr, kernel_size=2, stride=2)
            self.corr_pyramid.append(corr)

    def __call__(self, coords: torch.Tensor) -> torch.Tensor:
        """Lookup local correlation at each pyramid level.

        Args:
            coords: [B, 2, H, W] absolute coordinates in frame 2

        Returns:
            [B, num_levels * (2r+1)^2, H, W]
        """
        r = self.radius
        B, _, H, W = coords.shape

        # Build local delta grid
        dx = torch.linspace(-r, r, 2 * r + 1, device=coords.device)
        dy = torch.linspace(-r, r, 2 * r + 1, device=coords.device)
        delta = torch.stack(torch.meshgrid(dy, dx, indexing="ij"), dim=-1)
        # delta: [2r+1, 2r+1, 2] — (y, x) but we need (x, y) for grid_sample
        delta = delta.reshape(-1, 2).flip(-1)  # [(2r+1)^2, 2] — (x, y)

        out = []
        for i, corr in enumerate(self.corr_pyramid):
            # corr: [B*H*W, 1, H_i, W_i]
            _, _, H_i, W_i = corr.shape

            # Scale coordinates to this pyramid level
            scale = 1.0 / (2 ** i)
            centroid = coords.permute(0, 2, 3, 1).reshape(B * H * W, 1, 1, 2)
            centroid = centroid * scale

            # Add deltas: [B*H*W, 1, (2r+1)^2, 2]
            sample_coords = centroid + delta.view(1, 1, -1, 2)

            # Normalise to [-1, 1]
            sample_coords[..., 0] = 2 * sample_coords[..., 0] / max(W_i - 1, 1) - 1
            sample_coords[..., 1] = 2 * sample_coords[..., 1] / max(H_i - 1, 1) - 1

            # grid_sample: corr is [B*H*W, 1, H_i, W_i], grid is [B*H*W, 1, K, 2]
            sampled = F.grid_sample(corr, sample_coords,
                                     align_corners=True,
                                     mode="bilinear", padding_mode="zeros")
            # sampled: [B*H*W, 1, 1, K]
            sampled = sampled.view(B, H, W, -1)  # [B, H, W, K]
            sampled = sampled.permute(0, 3, 1, 2)  # [B, K, H, W]
            out.append(sampled)

        return torch.cat(out, dim=1)


class MotionEncoder(nn.Module):
    """Encodes correlation features + current flow into motion features.

    Following RAFT's update block motion encoder structure.
    """

    def __init__(self, corr_channels: int, hidden_dim: int = 128):
        super().__init__()
        self.corr_conv = nn.Sequential(
            nn.Conv2d(corr_channels, 192, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(192, 128, 3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.flow_conv = nn.Sequential(
            nn.Conv2d(2, 64, 7, padding=3),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.out_conv = nn.Sequential(
            nn.Conv2d(128 + 64, hidden_dim - 2, 3, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, corr_features: torch.Tensor,
                flow: torch.Tensor) -> torch.Tensor:
        cor = self.corr_conv(corr_features)
        flo = self.flow_conv(flow)
        out = self.out_conv(torch.cat([cor, flo], dim=1))
        return torch.cat([out, flow], dim=1)  # [B, hidden_dim, H, W]


class SepConvGRU(nn.Module):
    """Separable Convolutional GRU — horizontal then vertical updates.

    Matches RAFT's SepConvGRU for better spatial reasoning.
    """

    def __init__(self, hidden_dim: int = 128, input_dim: int = 256):
        super().__init__()
        # Horizontal
        self.convz1 = nn.Conv2d(hidden_dim + input_dim, hidden_dim,
                                (1, 5), padding=(0, 2))
        self.convr1 = nn.Conv2d(hidden_dim + input_dim, hidden_dim,
                                (1, 5), padding=(0, 2))
        self.convq1 = nn.Conv2d(hidden_dim + input_dim, hidden_dim,
                                (1, 5), padding=(0, 2))
        # Vertical
        self.convz2 = nn.Conv2d(hidden_dim + input_dim, hidden_dim,
                                (5, 1), padding=(2, 0))
        self.convr2 = nn.Conv2d(hidden_dim + input_dim, hidden_dim,
                                (5, 1), padding=(2, 0))
        self.convq2 = nn.Conv2d(hidden_dim + input_dim, hidden_dim,
                                (5, 1), padding=(2, 0))

    def forward(self, h: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        # Horizontal GRU
        hx = torch.cat([h, x], dim=1)
        z = torch.sigmoid(self.convz1(hx))
        r = torch.sigmoid(self.convr1(hx))
        q = torch.tanh(self.convq1(torch.cat([r * h, x], dim=1)))
        h = (1 - z) * h + z * q

        # Vertical GRU
        hx = torch.cat([h, x], dim=1)
        z = torch.sigmoid(self.convz2(hx))
        r = torch.sigmoid(self.convr2(hx))
        q = torch.tanh(self.convq2(torch.cat([r * h, x], dim=1)))
        h = (1 - z) * h + z * q

        return h


class FlowHead(nn.Module):
    """Predicts flow update from GRU hidden state (INTERNAL)."""

    def __init__(self, hidden_dim: int = 128):
        super().__init__()
        self.conv1 = nn.Conv2d(hidden_dim, 256, 3, padding=1)
        self.conv2 = nn.Conv2d(256, 2, 3, padding=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.conv2(self.relu(self.conv1(h)))


class MaskHead(nn.Module):
    """Predicts dynamic probability mask from GRU hidden state (EXPORTED).

    Includes GradientClip after logits to bound BA-derived gradients.
    """

    def __init__(self, hidden_dim: int = 128):
        super().__init__()
        self.conv1 = nn.Conv2d(hidden_dim, 64, 3, padding=1)
        self.conv2 = nn.Conv2d(64, 1, 1)
        self.relu = nn.ReLU(inplace=True)
        self.grad_clip = GradClipModule(clip_val=0.01)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        logits = self.conv2(self.relu(self.conv1(h)))
        logits = self.grad_clip(logits)
        return logits


class FlowDecoder(nn.Module):
    """RAFT-style Update Operator — iterative flow refinement + mask output."""

    def __init__(self, hidden_dim: int = 128, corr_levels: int = 4,
                 corr_radius: int = 4, gru_iters: int = 3):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.corr_levels = corr_levels
        self.corr_radius = corr_radius
        self.gru_iters = gru_iters

        corr_channels = corr_levels * (2 * corr_radius + 1) ** 2

        self.motion_encoder = MotionEncoder(corr_channels, hidden_dim)
        self.gru = SepConvGRU(hidden_dim=hidden_dim,
                              input_dim=hidden_dim + hidden_dim)
        self.flow_head = FlowHead(hidden_dim)
        self.mask_head = MaskHead(hidden_dim)

    def forward(self, fmap1: torch.Tensor, fmap2: torch.Tensor,
                net_init: torch.Tensor, inp: torch.Tensor) -> tuple:
        """
        Args:
            fmap1: [B, C, H, W] features from frame t-1
            fmap2: [B, C, H, W] features from frame t
            net_init: [B, hidden_dim, H, W] GRU hidden init
            inp: [B, hidden_dim, H, W] motion context

        Returns:
            flow_predictions: list of [B, 2, H, W] at each iteration
            mask_logits: [B, 1, H, W] from final GRU state
        """
        B, C, H, W = fmap1.shape

        corr_fn = CorrBlock(fmap1, fmap2,
                            num_levels=self.corr_levels,
                            radius=self.corr_radius)

        flow = torch.zeros(B, 2, H, W, device=fmap1.device, dtype=fmap1.dtype)
        h = net_init

        # Identity coordinate grid
        gy, gx = torch.meshgrid(
            torch.arange(H, device=fmap1.device, dtype=fmap1.dtype),
            torch.arange(W, device=fmap1.device, dtype=fmap1.dtype),
            indexing="ij",
        )
        coords0 = torch.stack([gx, gy], dim=0).unsqueeze(0).expand(B, -1, -1, -1)

        flow_predictions = []

        for _ in range(self.gru_iters):
            flow = flow.detach()
            coords = coords0 + flow

            corr_features = corr_fn(coords)
            motion_feat = self.motion_encoder(corr_features, flow)
            gru_input = torch.cat([motion_feat, inp], dim=1)
            h = self.gru(h, gru_input)

            delta_flow = self.flow_head(h)
            flow = flow + delta_flow
            flow_predictions.append(flow)

        mask_logits = self.mask_head(h)
        return flow_predictions, mask_logits
