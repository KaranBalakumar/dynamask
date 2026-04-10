"""
RAFT-style BasicEncoder for DynaMask V2.

Two separate encoder instances are used:
  - Feature encoder (fnet): called on BOTH frames with shared weights.
    Produces 128-ch features at 1/8 resolution for the correlation volume.
    Has FiLM conditioning points after each residual stage.
  - Context encoder (cnet): called on frame t ONLY with separate weights.
    Produces 128-ch features that are split into GRU hidden init + motion context.

Architecture per encoder:
  Conv2d(3, 64, 7x7, stride=2) + InstanceNorm + ReLU  -> [B, 64,  H/2,  W/2]
  ResidualBlock(64,  64)  x 2                          -> [B, 64,  H/2,  W/2]
  ResidualBlock(64,  96,  stride=2) + ResBlock(96, 96) -> [B, 96,  H/4,  W/4]
  ResidualBlock(96,  128, stride=2) + ResBlock(128,128) -> [B, 128, H/8,  W/8]
  Conv2d(128, 128, 1x1)                                -> [B, 128, H/8,  W/8]

Pretrained weights loaded from raft-things.pth (or other RAFT checkpoints).
FiLM layers start as identity (gamma=1, beta=0) so encoder is identical to
pretrained RAFT at init.

Reference: Teed & Deng, "RAFT: Recurrent All-Pairs Field Transforms for
Optical Flow", ECCV 2020.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    """Two 3x3 convs with InstanceNorm and residual connection.

    When in_ch != out_ch or stride != 1, a 1x1 projection handles the skip.
    """

    def __init__(self, in_ch: int, out_ch: int, stride: int = 1,
                 norm_fn: str = "instance"):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1,
                               bias=False)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, stride=1, padding=1,
                               bias=False)
        self.relu = nn.ReLU(inplace=True)

        if norm_fn == "instance":
            self.norm1 = nn.InstanceNorm2d(out_ch, affine=True)
            self.norm2 = nn.InstanceNorm2d(out_ch, affine=True)
        elif norm_fn == "batch":
            self.norm1 = nn.BatchNorm2d(out_ch)
            self.norm2 = nn.BatchNorm2d(out_ch)
        elif norm_fn == "none":
            self.norm1 = nn.Identity()
            self.norm2 = nn.Identity()
        else:
            raise ValueError(f"Unknown norm_fn: {norm_fn}")

        self.downsample = None
        if stride != 1 or in_ch != out_ch:
            if norm_fn == "instance":
                norm_ds = nn.InstanceNorm2d(out_ch, affine=True)
            elif norm_fn == "batch":
                norm_ds = nn.BatchNorm2d(out_ch)
            else:
                norm_ds = nn.Identity()
            self.downsample = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False),
                norm_ds,
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.relu(self.norm1(self.conv1(x)))
        out = self.norm2(self.conv2(out))
        if self.downsample is not None:
            residual = self.downsample(x)
        return self.relu(out + residual)


class BasicEncoder(nn.Module):
    """RAFT BasicEncoder — produces 128-ch features at 1/8 resolution.

    Optionally accepts FiLM layers to condition features at each residual stage.

    Args:
        output_dim: output feature channels (default 128)
        norm_fn: normalisation type ("instance", "batch", "none")
        film_layers: optional list of 3 FiLM modules (one per residual stage)
    """

    # Channel widths at each residual stage
    STAGE_CHANNELS = [64, 96, 128]

    def __init__(self, output_dim: int = 128, norm_fn: str = "instance",
                 film_layers: nn.ModuleList = None):
        super().__init__()
        self.film_layers = film_layers

        # Stem: 3 -> 64, stride 2
        if norm_fn == "instance":
            stem_norm = nn.InstanceNorm2d(64, affine=True)
        elif norm_fn == "batch":
            stem_norm = nn.BatchNorm2d(64)
        else:
            stem_norm = nn.Identity()

        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            stem_norm,
            nn.ReLU(inplace=True),
        )

        # Stage 1: 64 -> 64 (stride 1, stays at H/2)
        self.stage1 = nn.Sequential(
            ResidualBlock(64, 64, stride=1, norm_fn=norm_fn),
            ResidualBlock(64, 64, stride=1, norm_fn=norm_fn),
        )

        # Stage 2: 64 -> 96 (stride 2, goes to H/4)
        self.stage2 = nn.Sequential(
            ResidualBlock(64, 96, stride=2, norm_fn=norm_fn),
            ResidualBlock(96, 96, stride=1, norm_fn=norm_fn),
        )

        # Stage 3: 96 -> 128 (stride 2, goes to H/8)
        self.stage3 = nn.Sequential(
            ResidualBlock(96, 128, stride=2, norm_fn=norm_fn),
            ResidualBlock(128, 128, stride=1, norm_fn=norm_fn),
        )

        # Output projection
        self.output_conv = nn.Conv2d(128, output_dim, 1)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out",
                                        nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor,
                f_imu: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            x: [B, 3, H, W] input image
            f_imu: [B, imu_dim] IMU feature for FiLM conditioning (optional)

        Returns:
            [B, output_dim, H/8, W/8] feature map
        """
        x = self.stem(x)

        # Stage 1
        x = self.stage1(x)
        if self.film_layers is not None and f_imu is not None:
            x = self.film_layers[0](x, f_imu)

        # Stage 2
        x = self.stage2(x)
        if self.film_layers is not None and f_imu is not None:
            x = self.film_layers[1](x, f_imu)

        # Stage 3
        x = self.stage3(x)
        if self.film_layers is not None and f_imu is not None:
            x = self.film_layers[2](x, f_imu)

        x = self.output_conv(x)
        return x


def load_raft_encoder_weights(encoder: BasicEncoder, ckpt_path: str,
                              prefix: str = "module.fnet."):
    """Load pretrained RAFT weights into a BasicEncoder.

    Args:
        encoder: the BasicEncoder instance to load into
        ckpt_path: path to RAFT checkpoint (e.g. raft-things.pth)
        prefix: key prefix in the checkpoint ("module.fnet." or "module.cnet.")

    Returns:
        (loaded_count, missing_keys, unexpected_keys) for diagnostics
    """
    import os
    if not os.path.exists(ckpt_path):
        print(f"[RAFT] Checkpoint not found: {ckpt_path}, using random init")
        return 0, [], []

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)

    # Extract weights matching the prefix
    state = {}
    for k, v in ckpt.items():
        if k.startswith(prefix):
            new_key = k[len(prefix):]
            state[new_key] = v

    if not state:
        print(f"[RAFT] No keys found with prefix '{prefix}' in {ckpt_path}")
        return 0, [], []

    # Map RAFT key names to our names
    # RAFT uses: conv1, norm1, layer1, layer2, layer3, conv2
    # We use:    stem.0, stem.1, stage1, stage2, stage3, output_conv
    mapped_state = {}
    key_mapping = {
        "conv1.": "stem.0.",
        "norm1.": "stem.1.",
        "layer1.": "stage1.",
        "layer2.": "stage2.",
        "layer3.": "stage3.",
        "conv2.": "output_conv.",
    }

    for k, v in state.items():
        mapped = False
        for old_prefix, new_prefix in key_mapping.items():
            if k.startswith(old_prefix):
                new_key = new_prefix + k[len(old_prefix):]
                mapped_state[new_key] = v
                mapped = True
                break
        if not mapped:
            mapped_state[k] = v

    # Also need to map ResidualBlock internals
    # RAFT ResBlock: conv1, conv2, norm1, norm2, downsample
    # Ours: same names, should work directly

    missing, unexpected = encoder.load_state_dict(mapped_state, strict=False)
    loaded = len(encoder.state_dict()) - len(missing)

    print(f"[RAFT] Loaded {loaded} params from {ckpt_path} (prefix={prefix}, "
          f"missing={len(missing)}, unexpected={len(unexpected)})")

    if missing:
        # Filter out FiLM-related missing keys (expected)
        non_film_missing = [k for k in missing if "film" not in k.lower()]
        if non_film_missing:
            print(f"[RAFT] Missing non-FiLM keys: {non_film_missing[:10]}")

    return loaded, missing, unexpected
