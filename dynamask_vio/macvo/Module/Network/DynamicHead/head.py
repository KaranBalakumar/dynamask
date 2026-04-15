from __future__ import annotations

from types import SimpleNamespace

import torch
import torch.nn as nn
import torch.nn.functional as F

from .convgru import ConvGRUCell, ConvLSTMCell
from .film import CrossAttentionIMUFusion, DepthBinFiLM, FiLM


class _GradClipFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, clip: float):
        ctx.clip = clip
        return x

    @staticmethod
    def backward(ctx, grad_x: torch.Tensor):
        grad_x = torch.where(torch.isnan(grad_x), torch.zeros_like(grad_x), grad_x)
        return grad_x.clamp(min=-ctx.clip, max=ctx.clip), None


class GradientClip(nn.Module):
    def __init__(self, clip: float):
        super().__init__()
        self.clip = float(clip)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return _GradClipFn.apply(x, self.clip)


class MonotonicWeightMapper(nn.Module):
    def __init__(self, num_knots: int = 8, w_eps: float = 1e-3):
        super().__init__()
        if num_knots < 2:
            raise ValueError("num_knots should be >= 2")
        self.num_knots = int(num_knots)
        self.w_eps = float(w_eps)
        self.inc_logits = nn.Parameter(torch.zeros(self.num_knots - 1))
        self.register_buffer("x_knots", torch.linspace(0.0, 1.0, self.num_knots))

    def _y_knots(self) -> torch.Tensor:
        inc = F.softplus(self.inc_logits)
        inc = inc / inc.sum().clamp(min=1e-8)
        y = torch.cat(
            [torch.zeros(1, dtype=inc.dtype, device=inc.device), torch.cumsum(inc, dim=0)],
            dim=0,
        )
        return self.w_eps + (1.0 - self.w_eps) * y

    def forward(self, c_eff: torch.Tensor) -> torch.Tensor:
        c = c_eff.clamp(0.0, 1.0)
        x = self.x_knots.to(dtype=c.dtype, device=c.device)
        y = self._y_knots().to(dtype=c.dtype, device=c.device)
        idx = torch.bucketize(c, x[1:-1], right=False)
        x0, x1 = x[idx], x[idx + 1]
        y0, y1 = y[idx], y[idx + 1]
        t = (c - x0) / (x1 - x0).clamp(min=1e-8)
        return ((1.0 - t) * y0 + t * y1).clamp(min=self.w_eps, max=1.0)


class StaticConfidenceHead(nn.Module):
    def __init__(
        self,
        hidden_dim: int = 128,
        recurrence: str = "convgru",
        imu_fusion: str = "depth_bin_film",
        imu_feature_dim: int = 128,
        grad_clip: float = 0.01,
        logit_clip: float = 10.0,
        outputs: tuple[str, ...] = ("static",),
        refinement_enabled: bool = False,
        refinement_width: int = 64,
        mapper_enabled: bool = True,
        mapper_knots: int = 8,
        w_eps: float = 1e-3,
        proxy_input_enabled: bool = True,
        depth_bins: tuple[float, ...] = (0.0, 5.0, 20.0, 1e6),
    ):
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.recurrence = recurrence
        self.imu_fusion = imu_fusion
        self.logit_clip = float(logit_clip)
        self.outputs = tuple(outputs)
        self.refinement_enabled = bool(refinement_enabled)
        self.refinement_width = int(refinement_width)
        self.proxy_input_enabled = bool(proxy_input_enabled)

        out_ch = 2 if ("static" in self.outputs and "visible" in self.outputs) else 1

        in_ch = 128 + 2 + 3 + (5 if self.proxy_input_enabled else 0)
        if imu_fusion == "concat":
            in_ch += int(imu_feature_dim)

        self.stem = nn.Sequential(
            nn.Conv2d(in_ch, hidden_dim, kernel_size=1, padding=0),
            nn.GroupNorm(8, hidden_dim),
            nn.SiLU(inplace=True),
        )

        self.film = FiLM(hidden_dim, imu_feature_dim) if imu_fusion == "film" else None
        self.depth_bin_film = (
            DepthBinFiLM(hidden_dim, imu_feature_dim, depth_bins=depth_bins) if imu_fusion == "depth_bin_film" else None
        )
        self.cross_attention_film = (
            CrossAttentionIMUFusion(hidden_dim, imu_feature_dim) if imu_fusion == "cross_attention" else None
        )

        if recurrence == "convgru":
            self.recurrent = ConvGRUCell(hidden_dim, hidden_dim, kernel_size=3)
        elif recurrence == "convlstm":
            self.recurrent = ConvLSTMCell(hidden_dim, hidden_dim, kernel_size=3)
        elif recurrence == "none":
            self.recurrent = None
        else:
            raise ValueError(f"Unsupported recurrence mode: {recurrence}")

        self.out_head = nn.Sequential(
            nn.Conv2d(hidden_dim, 64, kernel_size=3, padding=1),
            nn.SiLU(inplace=True),
            nn.Conv2d(64, out_ch, kernel_size=1, padding=0),
        )
        self.out_grad_clip = GradientClip(grad_clip)

        if self.refinement_enabled:
            refine_in_ch = out_ch + 2 + 3 + 2
            self.refine_h4 = nn.Sequential(
                nn.Conv2d(refine_in_ch, refine_in_ch, kernel_size=3, padding=1, groups=refine_in_ch),
                nn.Conv2d(refine_in_ch, self.refinement_width, kernel_size=1, padding=0),
                nn.SiLU(inplace=True),
                nn.Conv2d(self.refinement_width, out_ch, kernel_size=1, padding=0),
            )
        else:
            self.refine_h4 = None

        self.weight_mapper = MonotonicWeightMapper(num_knots=mapper_knots, w_eps=w_eps) if mapper_enabled else None
        self.register_buffer("T_calib", torch.tensor(1.0))
        self._c_prev: torch.Tensor | None = None

    @classmethod
    def from_config(cls, config: SimpleNamespace):
        refinement_cfg = getattr(config, "refinement", SimpleNamespace(enabled=False))
        mapper_cfg = getattr(config, "weight_mapper", SimpleNamespace(enabled=True, num_knots=8, w_eps=1e-3))
        proxy_cfg = getattr(config, "proxy_input", SimpleNamespace(enabled=True))
        outputs = tuple(getattr(config, "outputs", ["static"]))
        depth_bins = tuple(getattr(config, "depth_bins", [0.0, 5.0, 20.0, 1e6]))
        return cls(
            hidden_dim=int(getattr(config, "hidden_dim", 128)),
            recurrence=str(getattr(config, "recurrence", "convgru")),
            imu_fusion=str(getattr(config, "imu_fusion", "depth_bin_film")),
            imu_feature_dim=int(getattr(config, "imu_feature_dim", 128)),
            grad_clip=float(getattr(config, "grad_clip", 0.01)),
            logit_clip=float(getattr(config, "logit_clip", 10.0)),
            outputs=outputs,
            refinement_enabled=bool(getattr(refinement_cfg, "enabled", False)),
            refinement_width=int(getattr(refinement_cfg, "width", 64)),
            mapper_enabled=bool(getattr(mapper_cfg, "enabled", True)),
            mapper_knots=int(getattr(mapper_cfg, "num_knots", 8)),
            w_eps=float(getattr(mapper_cfg, "w_eps", 1e-3)),
            proxy_input_enabled=bool(getattr(proxy_cfg, "enabled", True)),
            depth_bins=depth_bins,
        )

    def _apply_imu_fusion(self, x: torch.Tensor, f_imu: torch.Tensor | None, depth: torch.Tensor | None) -> torch.Tensor:
        if f_imu is None or self.imu_fusion == "none":
            return x
        if self.imu_fusion == "film" and self.film is not None:
            return self.film(x, f_imu)
        if self.imu_fusion == "depth_bin_film" and self.depth_bin_film is not None:
            return self.depth_bin_film(x, f_imu, depth)
        if self.imu_fusion == "cross_attention" and self.cross_attention_film is not None:
            return self.cross_attention_film(x, f_imu)
        return x

    def forward(
        self,
        f_ctx: torch.Tensor,
        flow: torch.Tensor,
        cov: torch.Tensor,
        f_imu: torch.Tensor | None,
        proxy: torch.Tensor | None,
        h_prev: torch.Tensor | None,
        *,
        depth: torch.Tensor | None = None,
        image: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        x_inputs = [f_ctx, flow, cov]
        if self.proxy_input_enabled:
            if proxy is None:
                raise ValueError("proxy_input is enabled but proxy tensor is None")
            x_inputs.append(proxy)
        if self.imu_fusion == "concat" and f_imu is not None:
            b, _, h, w = f_ctx.shape
            imu_map = f_imu.unsqueeze(-1).unsqueeze(-1).expand(b, -1, h, w)
            x_inputs.append(imu_map)

        x = torch.cat(x_inputs, dim=1)
        x = self.stem(x)
        x = self._apply_imu_fusion(x, f_imu, depth)

        if self.recurrent is None:
            h = x
        elif isinstance(self.recurrent, ConvGRUCell):
            h = self.recurrent(x, h_prev)
        else:
            h, self._c_prev = self.recurrent(x, h_prev, self._c_prev)

        logits = self.out_grad_clip(self.out_head(h)).clamp(min=-self.logit_clip, max=self.logit_clip)

        if self.refine_h4 is not None:
            logits_h4 = F.interpolate(logits, scale_factor=2.0, mode="bilinear", align_corners=False)
            flow_h4 = F.interpolate(flow, size=logits_h4.shape[-2:], mode="bilinear", align_corners=False)
            cov_h4 = F.interpolate(cov, size=logits_h4.shape[-2:], mode="bilinear", align_corners=False)
            if image is None:
                image_h4 = torch.zeros(
                    (logits_h4.size(0), 1, logits_h4.size(-2), logits_h4.size(-1)),
                    dtype=logits_h4.dtype,
                    device=logits_h4.device,
                )
            else:
                if image.shape[-2:] != logits_h4.shape[-2:]:
                    image_h4 = F.interpolate(image, size=logits_h4.shape[-2:], mode="bilinear", align_corners=False)
                else:
                    image_h4 = image
                image_h4 = image_h4.mean(dim=1, keepdim=True)
            grad_x = torch.zeros_like(image_h4)
            grad_y = torch.zeros_like(image_h4)
            grad_x[..., :, 1:] = image_h4[..., :, 1:] - image_h4[..., :, :-1]
            grad_y[..., 1:, :] = image_h4[..., 1:, :] - image_h4[..., :-1, :]
            img_grad_h4 = torch.cat([grad_x.abs(), grad_y.abs()], dim=1)
            logits = (logits_h4 + self.refine_h4(torch.cat([logits_h4, flow_h4, cov_h4, img_grad_h4], dim=1))).clamp(
                min=-self.logit_clip, max=self.logit_clip
            )

        return logits, h

    def effective_confidence(self, logits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if logits.size(1) == 2:
            p_static = torch.sigmoid(logits[:, 0:1] / self.T_calib.clamp(min=1e-3))
            p_visible = torch.sigmoid(logits[:, 1:2] / self.T_calib.clamp(min=1e-3))
        else:
            p_static = torch.sigmoid(logits / self.T_calib.clamp(min=1e-3))
            p_visible = torch.ones_like(p_static)
        c_eff = p_static * p_visible
        return p_static, p_visible, c_eff

    def map_weight(self, c_eff: torch.Tensor) -> torch.Tensor:
        if self.weight_mapper is None:
            return c_eff
        return self.weight_mapper(c_eff)

    def reset_state(self) -> None:
        self._c_prev = None
