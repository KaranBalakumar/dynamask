"""AirIMU CodeNet-compatible IMU correction module.

This module mirrors the core AirIMU CodeNet structure so public checkpoints can
be loaded directly for frozen IMU correction in DynaMask V2.5.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import torch
import torch.nn as nn


class CNNEncoder(nn.Module):
    """1D CNN encoder used by AirIMU CodeNet."""

    def __init__(
        self,
        *,
        c_list: list[int] | tuple[int, ...] = (6, 32, 64),
        k_list: list[int] | tuple[int, ...] = (7, 7),
        s_list: list[int] | tuple[int, ...] = (3, 3),
        p_list: list[int] | tuple[int, ...] = (3, 3),
    ):
        super().__init__()
        if not (len(c_list) - 1 == len(k_list) == len(s_list) == len(p_list)):
            raise ValueError("CNNEncoder lists must have compatible lengths")

        layers: list[nn.Module] = []
        for i in range(len(c_list) - 1):
            layers.extend(
                [
                    nn.Conv1d(
                        c_list[i],
                        c_list[i + 1],
                        kernel_size=k_list[i],
                        stride=s_list[i],
                        padding=p_list[i],
                    ),
                    nn.BatchNorm1d(c_list[i + 1]),
                    nn.GELU(),
                    nn.Dropout(0.1),
                ]
            )
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class AirIMUCorrector(nn.Module):
    """AirIMU CodeNet-style bias/noise corrector.

    Input:
        accel: [B, N, 3]
        gyro:  [B, N, 3]
    Output:
        correction_acc : [B, N, 3]
        correction_gyro: [B, N, 3]
        acc_cov        : [B, N, 3]
        gyro_cov       : [B, N, 3]
    """

    def __init__(
        self,
        *,
        interval: int = 9,
        gyro_std: float = float(np.pi / 180.0),
        acc_std: float = 0.1,
    ):
        super().__init__()
        self.interval = int(interval)
        self.inter_head = int(np.floor(self.interval / 2.0))
        self.inter_tail = int(self.interval - self.inter_head)

        self.cnn = CNNEncoder(c_list=(6, 32, 64), k_list=(7, 7), s_list=(3, 3), p_list=(3, 3))
        self.gru1 = nn.GRU(input_size=64, hidden_size=128, num_layers=1, batch_first=True)
        self.gru2 = nn.GRU(input_size=128, hidden_size=256, num_layers=1, batch_first=True)

        self.accdecoder = nn.Sequential(nn.Linear(256, 128), nn.GELU(), nn.Linear(128, 3))
        self.acccov_decoder = nn.Sequential(nn.Linear(256, 128), nn.GELU(), nn.Linear(128, 3))
        self.gyrodecoder = nn.Sequential(nn.Linear(256, 128), nn.GELU(), nn.Linear(128, 3))
        self.gyrocov_decoder = nn.Sequential(nn.Linear(256, 128), nn.GELU(), nn.Linear(128, 3))

        self.register_buffer("gyro_std", torch.tensor(float(gyro_std)))
        self.register_buffer("acc_std", torch.tensor(float(acc_std)))

    def encoder(self, x: torch.Tensor) -> torch.Tensor:
        x = self.cnn(x.transpose(-1, -2)).transpose(-1, -2)
        x, _ = self.gru1(x)
        x, _ = self.gru2(x)
        return x

    def cov_decoder(self, x: torch.Tensor) -> torch.Tensor:
        acc_cov = torch.exp(self.acccov_decoder(x) - 5.0)
        gyro_cov = torch.exp(self.gyrocov_decoder(x) - 5.0)
        return torch.cat([acc_cov, gyro_cov], dim=-1)

    def decoder(self, x: torch.Tensor) -> torch.Tensor:
        acc_corr = self.accdecoder(x) * self.acc_std
        gyro_corr = self.gyrodecoder(x) * self.gyro_std
        return torch.cat([acc_corr, gyro_corr], dim=-1)

    def _update(self, to_update: torch.Tensor, feat: torch.Tensor, frame_len: int) -> torch.Tensor:
        """Broadcast interval-wise features back to per-sample sequence slots."""

        def _clip(v: int, upper: int) -> int:
            if v < 0:
                return 0
            if v > upper:
                return upper
            return v

        updated = to_update
        feat_range = int(np.ceil((frame_len - self.inter_head) / self.interval)) + 1
        for i in range(feat_range):
            s_p = _clip(i * self.interval - self.inter_head, frame_len)
            e_p = _clip(i * self.interval + self.inter_tail, frame_len)
            idx = _clip(i, feat.shape[1] - 1)
            updated[:, s_p:e_p, :] = updated[:, s_p:e_p, :] + feat[:, idx : idx + 1, :]
        return updated

    def forward(
        self,
        accel: torch.Tensor,
        gyro: torch.Tensor,
        valid_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        B, N, _ = accel.shape
        feature = torch.cat([accel, gyro], dim=-1)
        feature = self.encoder(feature)
        if feature.shape[1] > 1:
            feature = feature[:, 1:, :]

        correction = self.decoder(feature)
        covariance = self.cov_decoder(feature)

        correction_acc = torch.zeros_like(accel)
        correction_gyro = torch.zeros_like(gyro)
        acc_cov = torch.full_like(accel, 1e-4)
        gyro_cov = torch.full_like(gyro, 1e-4)

        frame_len = max(int(N - self.interval), 0)
        if frame_len > 0 and correction.shape[1] > 0:
            corr_acc_tail = self._update(
                torch.zeros_like(accel[:, self.interval :, :]),
                correction[..., :3],
                frame_len,
            )
            corr_gyro_tail = self._update(
                torch.zeros_like(gyro[:, self.interval :, :]),
                correction[..., 3:],
                frame_len,
            )
            cov_acc_tail = self._update(
                torch.zeros_like(accel[:, self.interval :, :]),
                covariance[..., :3],
                frame_len,
            )
            cov_gyro_tail = self._update(
                torch.zeros_like(gyro[:, self.interval :, :]),
                covariance[..., 3:],
                frame_len,
            )

            correction_acc[:, self.interval :, :] = corr_acc_tail
            correction_gyro[:, self.interval :, :] = corr_gyro_tail
            acc_cov[:, self.interval :, :] = cov_acc_tail
            gyro_cov[:, self.interval :, :] = cov_gyro_tail

        if valid_mask is not None:
            valid = valid_mask.unsqueeze(-1).to(dtype=correction_acc.dtype)
            correction_acc = correction_acc * valid
            correction_gyro = correction_gyro * valid
            acc_cov = acc_cov * valid + (1.0 - valid) * 1e-4
            gyro_cov = gyro_cov * valid + (1.0 - valid) * 1e-4

        acc_cov = acc_cov.clamp(min=1e-6, max=1e2)
        gyro_cov = gyro_cov.clamp(min=1e-6, max=1e2)
        return correction_acc, correction_gyro, acc_cov, gyro_cov


def _assert_airimu_load_clean(missing: Iterable[str], unexpected: Iterable[str]) -> None:
    """Fail loudly on suspicious state-dict mismatches."""
    allowed_missing = {"acc_std", "gyro_std"}
    allowed_unexpected_prefixes = (
        "integrator.",
        "model.",
        "module.integrator.",
    )

    missing = list(missing)
    unexpected = list(unexpected)

    critical_missing = [k for k in missing if k not in allowed_missing]
    critical_unexpected = [
        k for k in unexpected if not any(k.startswith(prefix) for prefix in allowed_unexpected_prefixes)
    ]

    if critical_missing or critical_unexpected:
        raise RuntimeError(
            "AirIMU checkpoint mismatch.\n"
            f"Missing: {critical_missing}\n"
            f"Unexpected: {critical_unexpected}"
        )


def load_airimu_weights(encoder: AirIMUCorrector, ckpt_path: str, *, freeze: bool = True) -> AirIMUCorrector:
    """Load AirIMU CodeNet weights and optionally freeze the encoder."""
    ckpt_file = Path(ckpt_path)
    if not ckpt_file.exists():
        raise FileNotFoundError(f"AirIMU checkpoint not found: {ckpt_path}")

    ckpt = torch.load(str(ckpt_file), map_location="cpu", weights_only=False)
    state = ckpt.get("model_state_dict", ckpt.get("state_dict", ckpt))

    cleaned: dict[str, torch.Tensor] = {}
    prefixes = (
        "module.",
        "model.",
        "network.",
        "codenet.",
    )
    for key, value in state.items():
        clean_key = key
        for prefix in prefixes:
            if clean_key.startswith(prefix):
                clean_key = clean_key[len(prefix) :]
        cleaned[clean_key] = value

    missing, unexpected = encoder.load_state_dict(cleaned, strict=False)
    _assert_airimu_load_clean(missing, unexpected)

    if freeze:
        for p in encoder.parameters():
            p.requires_grad_(False)
        encoder.eval()
    return encoder
