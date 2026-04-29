import torch
import torch.nn as nn
import torch.nn.functional as F

from ..FlowFormer.core.gru import SepConvGRU
from ..FlowFormer.core.decoder import MemoryDecoder, initialize_flow
# CovUpdateBlock is a standalone sibling block — imported for reuse, not its decoder wrapper.
from ..FlowFormerCov.covhead import CovUpdateBlock


# ---------------------------------------------------------------------------
# IMU-conditioning sublayers (§3.6)
# ---------------------------------------------------------------------------


class FiLMLayer(nn.Module):
    """Feature-wise Linear Modulation (FiLM) from a global conditioner (§3.6.1).

        h' = (1 + γ) · h + β,          γ, β = chunk(proj(cond), 2)

    cond ∈ ℝ^{B×cond_dim}; h ∈ ℝ^{B×feat_dim×H×W}. `proj` produces (γ, β) via a single
    linear layer of width 2·feat_dim, then split along channel.
    """

    def __init__(self, cond_dim: int = 128, feat_dim: int = 128):
        super().__init__()
        self.proj = nn.Linear(cond_dim, feat_dim * 2)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        # (B, cond_dim) → (B, 2·feat_dim) → (B, 2·feat_dim, 1, 1) to broadcast against (B, C, H, W).
        params = self.proj(cond).unsqueeze(-1).unsqueeze(-1)
        gamma, beta = params.chunk(2, dim=1)
        return x * (gamma) + beta


class IMUCrossAttn(nn.Module):
    """Single-head scaled dot-product cross-attention over N semantic IMU tokens (§3.6.3).

    Given feature map h ∈ ℝ^{B×C×H×W} and tokens T ∈ ℝ^{B×N×C}:

        T' = w ⊙ T                             (per-token learnable gain, w init 1)
        Q = h.flatten_spatial · W_Q ∈ ℝ^{B × HW × C}
        K = T' · W_K           ∈ ℝ^{B × N × C}
        V = T' · W_V           ∈ ℝ^{B × N × C}
        A = softmax(Q·K^T / √C) ∈ ℝ^{B × HW × N}   (attention is per-pixel over N tokens)
        Δ = A · V              ∈ ℝ^{B × HW × C}
        return Δ reshaped to (B, C, H, W)

    Single head: 7 tokens is too few to split meaningfully. Per-token weights `w`
    let the model mute weak slots (e.g., `dt`, `diag(Σ)`) without changing the
    attention shape — see §3.6.2.
    """

    def __init__(self, feat_dim: int = 128, n_tokens: int = 7):
        super().__init__()
        self.n_tokens = n_tokens
        self.scale = feat_dim ** -0.5
        self.q_proj = nn.Linear(feat_dim, feat_dim)
        self.k_proj = nn.Linear(feat_dim, feat_dim)
        self.v_proj = nn.Linear(feat_dim, feat_dim)
        # Per-token gain, applied before K/V projections. Init 1 preserves the
        # starting behavior; α in DynUpdateBlock still gates the whole residual.
        self.token_weights = nn.Parameter(torch.ones(n_tokens))

    def forward(self, h: torch.Tensor, imu_tokens: torch.Tensor) -> torch.Tensor:
        B, C, H, W = h.shape
        tokens = imu_tokens * self.token_weights.view(1, -1, 1)              # (B, N, C)
        q = self.q_proj(h.flatten(2).transpose(1, 2))                        # (B, HW, C)
        k = self.k_proj(tokens)                                              # (B, N, C)
        v = self.v_proj(tokens)                                              # (B, N, C)
        attn = torch.softmax((q @ k.transpose(-2, -1)) * self.scale, dim=-1)  # (B, HW, N)
        out  = attn @ v                                                       # (B, HW, C)
        return out.transpose(1, 2).reshape(B, C, H, W)


# ---------------------------------------------------------------------------
# DynHead + DynUpdateBlock (sibling of CovUpdateBlock)
# ---------------------------------------------------------------------------


class DynHead(nn.Module):
    """Per-iteration 1-logit head at H/8, mirrors CovHead structure with 1-ch output."""

    def __init__(self, input_dim: int = 128, hidden_dim: int = 256):
        super().__init__()
        self.conv1 = nn.Conv2d(input_dim, hidden_dim, 3, padding=1)
        self.conv2 = nn.Conv2d(hidden_dim, hidden_dim // 2, 3, padding=1)
        self.conv3 = nn.Conv2d(hidden_dim // 2, hidden_dim // 4, 3, padding=1)
        self.conv4 = nn.Conv2d(hidden_dim // 4, 1, 3, padding=1)
        self.relu  = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv2(self.relu(self.conv1(x)))
        x = self.conv4(self.relu(self.conv3(x)))
        return x


class DynUpdateBlock(nn.Module):
    """Sibling of CovUpdateBlock with IMU conditioning (§3.3, §3.6).

    Pipeline per decoder iteration k:
        1. h  = SepConvGRU(h, inp_cat)                 # same 384-ch input as flow/cov GRU
        2. h' = FiLM(h, f_imu)                         # global modulation from IMU
        3. h  = h' + tanh(α) · IMUCrossAttn(h', tokens) # CLIP-adapter on FiLM'd base
        4. logits = DynHead(h) ∈ ℝ^{B×1×H/8×W/8}
        5. mask   = 0.25 · mask_conv(h) ∈ ℝ^{B×36×H/8×W/8}   (2× convex upsample, §3.3)

    Note on step 3: the adapter takes the FiLM'd `h` (the "base representation") as its
    input, matching CLIP-adapter's `h' + tanh(α)·adapter(h')` form. α is initialized to
    zero so at training start the block behaves as pure FiLM (§3.6.5).
    """

    def __init__(
        self,
        args,
        hidden_dim: int = 128,
        imu_dim: int = 128,
        n_imu_tokens: int = 7,
    ):
        super().__init__()
        self.args = args
        # Same inp_cat width as flow/cov GRUs: 128 (flow_inp) + 128 (motion) + 128 (motion_g) = 384
        self.gru      = SepConvGRU(hidden_dim=hidden_dim, input_dim=128 + hidden_dim + hidden_dim)
        self.film     = FiLMLayer(cond_dim=imu_dim, feat_dim=hidden_dim)
        self.imu_attn = IMUCrossAttn(feat_dim=hidden_dim, n_tokens=n_imu_tokens)
        # Gated residual scalar — initialized to 0 (critical for CLIP-adapter behavior).
        self.alpha    = nn.Parameter(torch.zeros(1))
        self.dyn_head = DynHead(hidden_dim, hidden_dim=256)
        # Mask for 2× convex upsample (H/8 → H/4): C=1 × 2×2 sub-pixels × 9 neighbors = 36 channels.
        self.mask = nn.Sequential(
            nn.Conv2d(hidden_dim, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 1 * 4 * 9, 1, padding=0),
        )

    def forward(
        self,
        dyn_net: torch.Tensor,          # (B, 128, H/8, W/8)
        inp_cat: torch.Tensor,          # (B, 384, H/8, W/8)
        f_imu: torch.Tensor,            # (B, 128)
        imu_tokens: torch.Tensor,       # (B, 7, 128)
    ):
        h = self.gru(dyn_net, inp_cat)
        h = self.film(h, f_imu)
        h = h + torch.tanh(self.alpha) * self.imu_attn(h, imu_tokens)
        delta_dyn = self.dyn_head(h)
        mask      = 0.25 * self.mask(h)
        return h, delta_dyn, mask


# ---------------------------------------------------------------------------
# Upsample: H/8 → H/4 by 2× convex combination
# ---------------------------------------------------------------------------


def upsample_dyn_logits(dyn: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Upsample a 1-channel H/8 logit map to H/4 using a 9-neighbor convex combination.

    Args:
        dyn  : (N, 1, H, W)                     per-pixel logits at H/8
        mask : (N, 1·2·2·9 = 36, H, W)          raw mask weights (softmax applied here)

    Layout: mask is reshaped to (N, 1, 9, 2, 2, H, W). For every pixel (h, w) and every
    sub-pixel (sh, sw) ∈ {0,1}², softmax over the 9 spatial neighbors produces a valid
    probability distribution; the resulting 2×2 block of outputs is placed at
    (2h+sh, 2w+sw) in the upsampled map. Identical in spirit to RAFT's 8× upsample,
    with block size 2 instead of 8.
    """
    N, C, H, W = dyn.shape
    # (N, 36, H, W) → (N, C, 9, 2, 2, H, W)
    mask = mask.view(N, C, 9, 2, 2, H, W)
    mask = torch.softmax(mask, dim=2)                 # normalize over 9 neighbors

    # unfold(3×3, pad=1): (N, C, H, W) → (N, C·9, H·W) → (N, C, 9, 1, 1, H, W)
    up_dyn = F.unfold(dyn, kernel_size=(3, 3), padding=1)
    up_dyn = up_dyn.view(N, C, 9, 1, 1, H, W)

    up_dyn = (mask * up_dyn).sum(dim=2)               # (N, C, 2, 2, H, W)
    up_dyn = up_dyn.permute(0, 1, 4, 2, 5, 3)          # (N, C, H, 2, W, 2)
    return up_dyn.reshape(N, C, 2 * H, 2 * W)


# ---------------------------------------------------------------------------
# Inter-frame warp (§3.4)
# ---------------------------------------------------------------------------


def warp_prev_dyn_net(
    prev_dyn_net: torch.Tensor,        # (B, C, H/8, W/8)
    prev_flow:    torch.Tensor,        # (B, 2,  H/8, W/8) — at the same resolution
) -> torch.Tensor:
    """Warp the previous pair's `dyn_net_{K-1}` forward by `prev_flow` (§3.4).

    Out-of-bounds samples are zeroed via `grid_sample(padding_mode='zeros')` — that's
    the only geometry-certain reset signal available. Beyond that, per-pixel trust is
    left to DynGRU to learn from `inp_cat` over the K decoder iterations.
    """
    B, _, H, W = prev_dyn_net.shape

    device, dtype = prev_dyn_net.device, prev_dyn_net.dtype
    ys, xs = torch.meshgrid(
        torch.arange(H, device=device, dtype=dtype),
        torch.arange(W, device=device, dtype=dtype),
        indexing="ij",
    )
    base = torch.stack([xs, ys], dim=-1).unsqueeze(0).expand(B, -1, -1, -1)  # (B, H, W, 2)
    disp = prev_flow.permute(0, 2, 3, 1)                                     # (B, H, W, 2)
    grid = base + disp
    grid_x = 2.0 * grid[..., 0] / max(W - 1, 1) - 1.0
    grid_y = 2.0 * grid[..., 1] / max(H - 1, 1) - 1.0
    norm_grid = torch.stack([grid_x, grid_y], dim=-1)

    return F.grid_sample(
        prev_dyn_net, norm_grid, mode="bilinear", padding_mode="zeros", align_corners=True,
    )


# ---------------------------------------------------------------------------
# MemoryDynDecoder (subclasses MemoryDecoder directly — mirrors MemoryCovDecoder)
# ---------------------------------------------------------------------------


class MemoryDynDecoder(MemoryDecoder):
    """Extends the FlowFormer MemoryDecoder with two sibling branches: cov + dyn.

    Structurally parallel to MemoryCovDecoder (which only adds `cov_update`). We add
    both `cov_update` and `dyn_update` as siblings of the built-in flow `update_block`,
    all three sharing the same `inp_cat = [flow_inp, motion_feat, motion_feat_global]`.

    This keeps FlowFormerDyn a direct sibling of FlowFormerCov — both subclass FlowFormer
    via MemoryDecoder rather than chaining through MemoryCovDecoder.
    """

    def __init__(self, cfg, decoder_dtype: torch.dtype):
        super().__init__(cfg)
        self.decoder_dtype = decoder_dtype

        self.cov_update = CovUpdateBlock(self.cfg, hidden_dim=128)
        self.dyn_update = DynUpdateBlock(self.cfg, hidden_dim=128, imu_dim=128, n_imu_tokens=7)

        # Cast all decoder-internal modules to decoder_dtype (mirrors MemoryCovDecoder).
        self.delta              = self.delta.to(dtype=self.decoder_dtype)
        self.att                = self.att.to(dtype=self.decoder_dtype)
        self.decoder_layer      = self.decoder_layer.to(dtype=self.decoder_dtype)
        self.flow_token_encoder = self.flow_token_encoder.to(dtype=self.decoder_dtype)
        self.update_block       = self.update_block.to(dtype=self.decoder_dtype)
        self.cov_update         = self.cov_update.to(dtype=self.decoder_dtype)
        self.dyn_update         = self.dyn_update.to(dtype=self.decoder_dtype)

    def forward(                                                # pyright: ignore[reportIncompatibleMethodOverride]
        self,
        cost_memory: torch.Tensor,
        context: torch.Tensor,
        cost_maps: torch.Tensor,
        f_imu: torch.Tensor,
        imu_tokens: torch.Tensor,
        prev_dyn_net: torch.Tensor | None = None,
        prev_flow:    torch.Tensor | None = None,
    ):
        """
        Args:
            cost_memory : (B·H1·W1, H2'·W2', C)  memory from MemoryEncoder
            context     : (B, D, H1, W1)          context features (fp32)
            cost_maps   : cost volume from MemoryEncoder
            f_imu       : (B, 128)     FiLM conditioner (§3.6.1)
            imu_tokens  : (B, 7, 128)  semantic IMU tokens for cross-attn (§3.6.2)
            prev_dyn_net : optional (B, 128, H/8, W/8) previous pair's final dyn_net.
            prev_flow    : optional (B, 2,   H/8, W/8) previous pair's final forward flow,
                           in pixel units at H/8 resolution (raw coords, not normalized).

            When both prev_* args are supplied, `fdyn_net` is initialized by warping
            `prev_dyn_net` along `prev_flow` (out-of-bounds pixels zeroed, §3.4).
            Otherwise `fdyn_net` defaults to `flow_net.clone()` — first-pair behavior,
            matching the cov branch.
        Returns: (flow_predictions, cov_predictions, dyn_predictions)
            Training mode: three lists of K per-iter predictions.
            Eval mode    : (last, final_coord) pairs for flow/cov and (last,) for dyn,
                           matching the shape MemoryCovDecoder uses.
        """
        cost_memory = cost_memory.to(dtype=self.decoder_dtype)

        flow_coords0, flow_coords1 = initialize_flow(context)
        cov_coords0,  cov_coords1  = flow_coords0, flow_coords1.clone()

        flow_predictions: list[torch.Tensor] = []
        cov_predictions:  list[torch.Tensor] = []
        dyn_predictions:  list[torch.Tensor] = []

        context  = self.proj(context)
        flow_net, flow_inp = torch.split(context, [128, 128], dim=1)
        flow_net = flow_net.tanh().to(dtype=self.decoder_dtype)
        fcov_net = flow_net.clone().to(dtype=self.decoder_dtype)

        if prev_dyn_net is not None and prev_flow is not None:
            fdyn_net = warp_prev_dyn_net(
                prev_dyn_net.to(dtype=self.decoder_dtype),
                prev_flow.to(dtype=self.decoder_dtype),
            )
        else:
            fdyn_net = flow_net.clone().to(dtype=self.decoder_dtype)

        flow_inp = flow_inp.relu().to(dtype=self.decoder_dtype)

        attention = self.att(flow_inp)

        size = flow_net.shape
        key, value = None, None

        f_imu_dec      = f_imu.to(dtype=self.decoder_dtype)
        imu_tokens_dec = imu_tokens.to(dtype=self.decoder_dtype)

        for _ in range(self.depth):
            flow_coords1      = flow_coords1.detach()
            bf16_flow_coords1 = flow_coords1.to(dtype=self.decoder_dtype)
            flow              = (flow_coords1 - flow_coords0).to(dtype=self.decoder_dtype)

            with torch.cuda.nvtx.range("Encode Flow Token"):
                # This module MUST run in fp32 precision.
                cost_forward = self.encode_flow_token(cost_maps, flow_coords1)
                cost_forward = cost_forward.to(dtype=self.decoder_dtype)

            with torch.cuda.nvtx.range("CNN Encoder"):
                query = self.flow_token_encoder(cost_forward)
                query = query.permute(0, 2, 3, 1).view(size[0] * size[2] * size[3], 1, self.dim)

            with torch.cuda.nvtx.range("Cross Attention"):
                cost_global, key, value = self.decoder_layer(
                    query, key, value, cost_memory, bf16_flow_coords1, size, self.cfg.query_latent_dim
                )
                corr = torch.cat([cost_global, cost_forward], dim=1)

            with torch.cuda.nvtx.range("GMA Update Block"):
                motion_feat        = self.update_block.encoder(flow, corr)
                motion_feat_global = self.update_block.aggregator(attention, motion_feat)

            inp_cat = torch.cat([flow_inp, motion_feat, motion_feat_global], dim=1)

            with torch.cuda.nvtx.range("Flow Update Block"):
                flow_net   = self.update_block.gru(flow_net, inp_cat)
                delta_flow = self.update_block.flow_head(flow_net)
                up_mask    = self.update_block.mask(flow_net)

            with torch.cuda.nvtx.range("Cov Update Block"):
                fcov_net, delta_cov, cov_mask = self.cov_update(fcov_net, inp_cat)

            with torch.cuda.nvtx.range("Dyn Update Block"):
                fdyn_net, delta_dyn, dyn_mask = self.dyn_update(
                    fdyn_net, inp_cat, f_imu_dec, imu_tokens_dec
                )

            with torch.cuda.nvtx.range("Flow Upsample"):
                # fp32 from here.
                delta_flow = delta_flow.to(dtype=torch.float32)
                up_mask    = 0.25 * up_mask.to(dtype=torch.float32)
                flow_coords1 = flow_coords1 + delta_flow
                flow_up      = self.upsample_flow(flow_coords1 - flow_coords0, up_mask)
                flow_predictions.append(flow_up)

            with torch.cuda.nvtx.range("Cov Upsample"):
                delta_cov = delta_cov.to(dtype=torch.float32)
                cov_mask  = cov_mask.to(dtype=torch.float32)
                cov_coords1 = cov_coords1 + delta_cov
                cov_up = self.upsample_flow(cov_coords1 - cov_coords0, cov_mask)
                cov_predictions.append(cov_up)

            with torch.cuda.nvtx.range("Dyn Upsample"):
                # 2× convex upsample: H/8 → H/4 (NOT 8× — see §3.3).
                delta_dyn = delta_dyn.to(dtype=torch.float32)
                dyn_mask  = dyn_mask.to(dtype=torch.float32)
                dyn_up    = upsample_dyn_logits(delta_dyn, dyn_mask)
                dyn_predictions.append(dyn_up)

        if self.training:
            return flow_predictions, cov_predictions, dyn_predictions
        return (
            (flow_predictions[-1], flow_coords1 - flow_coords0),
            (cov_predictions[-1],  cov_coords1  - cov_coords0),
            (dyn_predictions[-1],),
        )
