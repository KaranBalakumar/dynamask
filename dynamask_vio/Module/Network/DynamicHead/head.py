import torch
import torch.nn as nn
import torch.nn.functional as F

from .convgru import ConvGRUCell, ConvLSTMCell
from .film import FiLM


def _pick_group_count(channels: int) -> int:
    for g in (32, 16, 8, 4, 2, 1):
        if channels % g == 0:
            return g
    return 1


class MonotonicPiecewiseMapper(nn.Module):
    def __init__(self, num_knots: int = 8):
        super().__init__()
        assert num_knots >= 2
        self.num_knots = num_knots
        self.raw_increments = nn.Parameter(torch.zeros(num_knots - 1))
        self.register_buffer("x_knots", torch.linspace(0.0, 1.0, num_knots))

    def _y_knots(self) -> torch.Tensor:
        increments = F.softplus(self.raw_increments) + 1e-6
        cumulative = torch.cumsum(increments, dim=0)
        return torch.cat(
            [torch.zeros(1, device=cumulative.device), cumulative / cumulative[-1]],
            dim=0,
        )

    def forward(self, confidence: torch.Tensor) -> torch.Tensor:
        x = confidence.clamp(0.0, 1.0)
        y_knots = self._y_knots()

        flat = x.reshape(-1)
        idx = torch.bucketize(flat, self.x_knots[1:-1])
        x0 = self.x_knots[idx]
        x1 = self.x_knots[idx + 1]
        y0 = y_knots[idx]
        y1 = y_knots[idx + 1]
        alpha = (flat - x0) / (x1 - x0 + 1e-8)
        mapped = y0 + alpha * (y1 - y0)
        return mapped.reshape_as(x)


class StaticConfidenceHead(nn.Module):
    def __init__(
        self,
        ctx_dim: int = 128,
        flow_dim: int = 2,
        cov_dim: int = 3,
        proxy_dim: int = 4,
        hidden_dim: int = 128,
        imu_feature_dim: int = 128,
        recurrence: str = "convgru",
        imu_fusion: str = "film",
        weight_mapper_knots: int = 8,
        w_eps: float = 1e-3,
    ):
        super().__init__()
        assert recurrence in {"convgru", "convlstm", "none"}
        assert imu_fusion in {"film", "concat", "none"}

        self.hidden_dim = hidden_dim
        self.proxy_dim = proxy_dim
        self.imu_feature_dim = imu_feature_dim
        self.recurrence = recurrence
        self.imu_fusion = imu_fusion
        self.w_eps = w_eps

        in_channels = ctx_dim + flow_dim + cov_dim + proxy_dim
        if imu_fusion == "concat":
            in_channels += imu_feature_dim

        self.input_proj = nn.Conv2d(in_channels, hidden_dim, kernel_size=1)
        self.input_norm = nn.GroupNorm(_pick_group_count(hidden_dim), hidden_dim)
        self.input_act = nn.SiLU(inplace=True)

        self.film = FiLM(hidden_dim, imu_feature_dim) if imu_fusion == "film" else None

        if recurrence == "convgru":
            self.recurrent = ConvGRUCell(hidden_dim=hidden_dim, input_dim=hidden_dim)
        elif recurrence == "convlstm":
            self.recurrent = ConvLSTMCell(hidden_dim=hidden_dim, input_dim=hidden_dim)
        else:
            self.recurrent = None

        mid_channels = max(hidden_dim // 2, 16)
        self.output_head = nn.Sequential(
            nn.Conv2d(hidden_dim, mid_channels, kernel_size=3, padding=1),
            nn.SiLU(inplace=True),
            nn.Conv2d(mid_channels, 2, kernel_size=1),
        )

        self.weight_mapper = MonotonicPiecewiseMapper(num_knots=weight_mapper_knots)
        self.register_buffer("T_calib", torch.tensor(1.0))

    def _prepare_proxy(self, f_ctx: torch.Tensor, proxy: torch.Tensor | None) -> torch.Tensor:
        if proxy is not None:
            return proxy
        batch, _, height, width = f_ctx.shape
        return torch.zeros(
            batch,
            self.proxy_dim,
            height,
            width,
            dtype=f_ctx.dtype,
            device=f_ctx.device,
        )

    def _prepare_imu_for_concat(self, f_ctx: torch.Tensor, f_imu: torch.Tensor | None) -> torch.Tensor:
        batch, _, height, width = f_ctx.shape
        if f_imu is None:
            f_imu = torch.zeros(
                batch,
                self.imu_feature_dim,
                dtype=f_ctx.dtype,
                device=f_ctx.device,
            )
        return f_imu.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, height, width)

    def _apply_recurrence(self, x: torch.Tensor, h_prev: torch.Tensor | None) -> torch.Tensor:
        if self.recurrence == "none" or self.recurrent is None:
            return x

        if h_prev is None:
            h_prev = torch.zeros_like(x)

        if self.recurrence == "convlstm":
            c_prev = torch.zeros_like(h_prev)
            h_new, _ = self.recurrent(h_prev, c_prev, x)
            return h_new

        return self.recurrent(h_prev, x)

    def forward(
        self,
        f_ctx: torch.Tensor,
        flow: torch.Tensor,
        cov: torch.Tensor,
        f_imu: torch.Tensor | None,
        proxy: torch.Tensor | None,
        h_prev: torch.Tensor | None,
    ) -> dict[str, torch.Tensor]:
        inputs = [f_ctx, flow, cov, self._prepare_proxy(f_ctx, proxy)]

        if self.imu_fusion == "concat":
            inputs.append(self._prepare_imu_for_concat(f_ctx, f_imu))

        x = torch.cat(inputs, dim=1)
        x = self.input_act(self.input_norm(self.input_proj(x)))

        if self.film is not None:
            x = self.film(x, f_imu)

        h_new = self._apply_recurrence(x, h_prev)
        logits = self.output_head(h_new).clamp(min=-10.0, max=10.0)

        logits_static = logits[:, :1]
        logits_visible = logits[:, 1:]
        temperature = self.T_calib.clamp(min=1e-3)
        p_static = torch.sigmoid(logits_static / temperature)
        p_visible = torch.sigmoid(logits_visible / temperature)
        c_eff = p_static * p_visible
        static_weight = self.weight_mapper(c_eff).clamp(min=self.w_eps, max=1.0)

        return {
            "p_static": p_static,
            "p_visible": p_visible,
            "static_conf": c_eff,
            "static_weight": static_weight,
            "h_new": h_new,
        }
