from __future__ import annotations

import torch
import torch.nn as nn


def _maybe_spectral(module: nn.Module, enabled: bool) -> nn.Module:
    return nn.utils.spectral_norm(module) if enabled else module


class ConvGRUCell(nn.Module):
    def __init__(self, in_ch: int, hid_ch: int, kernel: int = 3, use_spectral_norm: bool = False) -> None:
        super().__init__()
        pad = kernel // 2
        gate_in = in_ch + hid_ch
        self.hid_ch = hid_ch
        self.conv_z = _maybe_spectral(nn.Conv2d(gate_in, hid_ch, kernel_size=kernel, padding=pad), use_spectral_norm)
        self.conv_r = _maybe_spectral(nn.Conv2d(gate_in, hid_ch, kernel_size=kernel, padding=pad), use_spectral_norm)
        self.conv_n = _maybe_spectral(nn.Conv2d(gate_in, hid_ch, kernel_size=kernel, padding=pad), use_spectral_norm)

    def forward(self, x: torch.Tensor, h_prev: torch.Tensor | None) -> torch.Tensor:
        if h_prev is None:
            h_prev = torch.zeros((x.shape[0], self.hid_ch, x.shape[2], x.shape[3]), device=x.device, dtype=x.dtype)
        xr = torch.cat([x, h_prev], dim=1)
        z = torch.sigmoid(self.conv_z(xr))
        r = torch.sigmoid(self.conv_r(xr))
        n = torch.tanh(self.conv_n(torch.cat([x, r * h_prev], dim=1)))
        return (1.0 - z) * h_prev + z * n


class ConvGRUStack(nn.Module):
    def __init__(self, in_ch: int, hid_ch: int, num_layers: int = 2, kernel: int = 3, use_spectral_norm: bool = False) -> None:
        super().__init__()
        assert num_layers >= 1
        layers = []
        for i in range(num_layers):
            layers.append(
                ConvGRUCell(
                    in_ch if i == 0 else hid_ch,
                    hid_ch,
                    kernel=kernel,
                    use_spectral_norm=use_spectral_norm,
                )
            )
        self.layers = nn.ModuleList(layers)

    def forward(self, x: torch.Tensor, h_prev: torch.Tensor | None) -> tuple[torch.Tensor, torch.Tensor]:
        h = h_prev
        for layer in self.layers:
            h = layer(x, h)
            x = h
        assert h is not None
        return h, h
