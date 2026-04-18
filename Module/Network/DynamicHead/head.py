from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any
import typing as T

import torch
import torch.nn as nn
import torch.nn.functional as F

from .convgru import ConvGRUStack
from .film import DepthBinFiLM, FiLMLayer
from .refiner import ReprojectionRefiner


def _ns_get(cfg: SimpleNamespace | dict[str, Any], key: str, default: Any) -> Any:
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


@dataclass
class HeadOut:
    c: torch.Tensor
    logits_8: torch.Tensor
    logits_4: torch.Tensor
    h8_new: torch.Tensor
    p_static: torch.Tensor | None = None
    p_visible: torch.Tensor | None = None


class StaticConfidenceHead(nn.Module):
    def __init__(
        self,
        c_ctx: int = 256,
        c_imu: int = 128,
        c_hid: int = 128,
        z_min: float = 0.5,
        z_max: float = 30.0,
        visible_eps: float = 0.02,
        max_logit: float = 10.0,
        max_flow_norm: float = 200.0,
        use_spectral_norm: bool = False,
        factorize: bool = False,
    ) -> None:
        super().__init__()
        self.z_min = float(z_min)
        self.z_max = float(z_max)
        self.visible_eps = float(visible_eps)
        self.max_logit = float(max_logit)
        self.max_flow_norm = float(max_flow_norm)
        self.factorize = bool(factorize)
        self.register_buffer("temperature", torch.tensor(1.0))
        self.register_buffer("temperature_visible", torch.tensor(1.0))

        self.depth_film = DepthBinFiLM(c_imu, c_ctx)
        self.res_film = FiLMLayer(c_imu, c_ctx)
        self.proj = nn.Sequential(
            nn.Conv2d(c_ctx + 6, c_hid, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.gru = ConvGRUStack(c_hid, c_hid, num_layers=2, use_spectral_norm=use_spectral_norm)
        out_ch = 2 if self.factorize else 1
        self.classifier8 = nn.Conv2d(c_hid, out_ch, kernel_size=3, padding=1)
        # Residual refinement at 1/4: logits_4 = upsample(logits_8) + refine4(upsample(h8)).
        # Zero-init the final conv so at start of training logits_4 ≡ upsampled logits_8.
        self.refine4 = nn.Sequential(
            nn.Conv2d(c_hid, c_hid // 2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(c_hid // 2, out_ch, kernel_size=3, padding=1),
        )
        nn.init.zeros_(self.refine4[-1].weight)
        nn.init.zeros_(self.refine4[-1].bias)
        self.refiner = ReprojectionRefiner(in_ch=2, hidden_ch=32, out_ch=2)

    @staticmethod
    def _sobel_grad(x: torch.Tensor) -> torch.Tensor:
        kx = torch.tensor(
            [[1.0, 0.0, -1.0], [2.0, 0.0, -2.0], [1.0, 0.0, -1.0]],
            device=x.device,
            dtype=x.dtype,
        ).view(1, 1, 3, 3) / 8.0
        ky = kx.transpose(-1, -2)
        gx = F.conv2d(x, kx, padding=1)
        gy = F.conv2d(x, ky, padding=1)
        return torch.sqrt(gx * gx + gy * gy + 1e-12)

    @staticmethod
    def _resize_to(ref: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-2:] == ref.shape[-2:]:
            return x
        return F.interpolate(x, size=ref.shape[-2:], mode="bilinear", align_corners=False)

    def forward(
        self,
        phi8: torch.Tensor,
        h4: torch.Tensor,
        z_hat: torch.Tensor,
        e_raw: torch.Tensor,
        f_imu: torch.Tensor,
        h8_prev: torch.Tensor | None = None,
        return_probs: bool = False,
    ) -> HeadOut:
        z8 = self._resize_to(phi8, z_hat)
        e8 = self._resize_to(phi8, e_raw)
        h4_8 = self._resize_to(phi8, h4)

        z8 = z8.clamp(min=self.z_min, max=self.z_max)
        e8 = e8.clamp(min=0.0, max=self.max_flow_norm)
        grad = self._sobel_grad(h4_8.mean(dim=1, keepdim=True))

        phi_mod = self.depth_film(phi8, f_imu, z8, self.z_min, self.z_max)
        r_mod = self.res_film(phi8, f_imu)
        e_ref = self.refiner(z8, e8)
        feat = torch.cat([phi_mod, z8, e8, e_ref, grad, (r_mod - phi8).norm(dim=1, keepdim=True)], dim=1)
        h_in = self.proj(feat)
        h8, h8_new = self.gru(h_in, h8_prev)

        logits_8 = self.classifier8(h8).clamp(min=-self.max_logit, max=self.max_logit)
        logits_8_up = F.interpolate(logits_8, scale_factor=2.0, mode="bilinear", align_corners=False)
        h8_up = F.interpolate(h8, scale_factor=2.0, mode="bilinear", align_corners=False)
        logits_4 = (logits_8_up + self.refine4(h8_up)).clamp(min=-self.max_logit, max=self.max_logit)
        temp_static = torch.clamp(T.cast(torch.Tensor, self.temperature), min=1e-3)
        temp_visible = torch.clamp(T.cast(torch.Tensor, self.temperature_visible), min=1e-3)
        if self.factorize:
            p_static = torch.sigmoid(logits_4[:, :1] / temp_static)
            p_visible = torch.sigmoid(logits_4[:, 1:2] / temp_visible)
            c = p_static * p_visible
            return HeadOut(c=c, logits_8=logits_8, logits_4=logits_4, h8_new=h8_new, p_static=p_static, p_visible=p_visible)

        c = torch.sigmoid(logits_4 / temp_static)
        if return_probs:
            p_visible = (z_hat > self.visible_eps).to(dtype=z_hat.dtype)
            return HeadOut(c=c, logits_8=logits_8, logits_4=logits_4, h8_new=h8_new, p_static=c, p_visible=p_visible)
        return HeadOut(c=c, logits_8=logits_8, logits_4=logits_4, h8_new=h8_new)


def build_head(config: SimpleNamespace | dict[str, Any] | None) -> StaticConfidenceHead:
    cfg = config if config is not None else {}
    return StaticConfidenceHead(
        c_ctx=int(_ns_get(cfg, "c_ctx", 256)),
        c_imu=int(_ns_get(cfg, "c_imu", 128)),
        c_hid=int(_ns_get(cfg, "c_hid", 128)),
        z_min=float(_ns_get(cfg, "z_min", 0.5)),
        z_max=float(_ns_get(cfg, "z_max", 30.0)),
        visible_eps=float(_ns_get(cfg, "visible_eps", 0.02)),
        max_logit=float(_ns_get(cfg, "max_logit", 10.0)),
        max_flow_norm=float(_ns_get(cfg, "max_flow_norm", 200.0)),
        use_spectral_norm=bool(_ns_get(cfg, "spectral_norm", False)),
        factorize=bool(_ns_get(cfg, "factorize", False)),
    )
