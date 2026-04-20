import torch
import torch.nn as nn

from ..FlowFormer.core.gru import SepConvGRU
from ..FlowFormerCov.covhead import MemoryCovDecoder

class FiLMLayer(nn.Module):
    """Conditions the GRU hidden state using the 128-D IMU feature mapping."""
    def __init__(self, cond_dim=128, feat_dim=128):
        super().__init__()
        # Linear layer mapping f_imu -> Gamma and Beta for FiLM
        self.proj = nn.Linear(cond_dim, feat_dim * 2)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        # Generate spatial parameters. [B, 128] -> [B, 256] -> [B, 256, 1, 1]
        params = self.proj(cond).unsqueeze(-1).unsqueeze(-1)
        gamma, beta = params.chunk(2, dim=1)
        
        # Apply standard Feature-wise Linear Modulation
        return x * (1 + gamma) + beta


class DynHead(nn.Module):
    """Per-iteration 1-logit head at H/8, mirrors CovHead structure."""
    def __init__(self, input_dim: int = 128, hidden_dim: int = 256):
        super().__init__()
        self.conv1 = nn.Conv2d(input_dim, hidden_dim, 3, padding=1)
        self.conv2 = nn.Conv2d(hidden_dim, hidden_dim // 2, 3, padding=1)
        self.conv3 = nn.Conv2d(hidden_dim // 2, hidden_dim // 4, 3, padding=1)
        self.conv4 = nn.Conv2d(hidden_dim // 4, 1, 3, padding=1)   # 1 logit (static/dynamic)
        self.relu  = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv2(self.relu(self.conv1(x)))
        x = self.conv4(self.relu(self.conv3(x)))
        return x


class DynUpdateBlock(nn.Module):
    """Sibling of CovUpdateBlock: own SepConvGRU + DynHead + upsample mask, FiLM-conditioned on f_imu."""
    def __init__(self, args, hidden_dim: int = 128, imu_dim: int = 128):
        super().__init__()
        self.args = args
        # Same inp_cat width as flow/cov GRUs: 128 (flow_inp) + 128 (motion) + 128 (motion_g) = 384
        self.gru  = SepConvGRU(hidden_dim=hidden_dim, input_dim=128 + hidden_dim + hidden_dim)
        
        # FiLM modulates the GRU hidden state post-update
        self.film = FiLMLayer(cond_dim=imu_dim, feat_dim=hidden_dim)
        self.dyn_head = DynHead(hidden_dim, hidden_dim=256)
        
        self.mask = nn.Sequential(
            nn.Conv2d(hidden_dim, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            # MATH FIX: H/8 -> H/4 requires 2x convex upsample. 
            # 2x2 pixels per block * 9 neighborhood = 36 channels for C=1 logit
            nn.Conv2d(256, 1 * 4 * 9, 1, padding=0),  
        )

    def forward(self, dyn_net: torch.Tensor, inp_cat: torch.Tensor, f_imu: torch.Tensor):
        dyn_net   = self.gru(dyn_net, inp_cat)                     # co-iterative update
        dyn_net   = self.film(dyn_net, f_imu)                      # IMU conditioning
        delta_dyn = self.dyn_head(dyn_net)                         # 1-ch logits @ H/8
        mask      = 0.25 * self.mask(dyn_net)                      # RAFT convex upsample
        return dyn_net, delta_dyn, mask


def upsample_dyn_logits(dyn: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Upsample dynamic logits using 2x convex combination (H/8 -> H/4)."""
    N, C, H, W = dyn.shape
    # mask shape: [N, C*36, H, W] -> [N, 1, 9, 2, 2, H, W]
    mask = mask.view(N, 1, 9, 2, 2, H, W)
    mask = torch.softmax(mask, dim=2)

    up_dyn = nn.functional.unfold(dyn, [3,3], padding=1)
    up_dyn = up_dyn.view(N, C, 9, 1, 1, H, W)

    up_dyn = torch.sum(mask * up_dyn, dim=2)
    up_dyn = up_dyn.permute(0, 1, 4, 2, 5, 3)
    return up_dyn.reshape(N, C, 2 * H, 2 * W)


class MemoryDynDecoder(MemoryCovDecoder):
    """Extends MemoryCovDecoder with a third sibling update branch."""
    def __init__(self, cfg, decoder_dtype: torch.dtype):
        super().__init__(cfg, decoder_dtype)
        self.dyn_update = DynUpdateBlock(self.cfg, hidden_dim=128, imu_dim=128)
        self.dyn_update = self.dyn_update.to(dtype=decoder_dtype)

    def forward(self, cost_memory, context, cost_maps, f_imu):
        """
        Implementation Node: To fully embed this, you override the inner 
        MemoryCovDecoder loop to step self.dyn_update(fdyn_net, ...)
        alongside the classical flow_update() and cov_update().
        """
        
        # 1. Boilerplate initialization (mirroring MemoryCovDecoder)
        dyn_predictions = []
        flow_predictions, cov_predictions = super().forward(cost_memory, context, cost_maps, query_latent_dim, flow_init)
        
        # Note: Code integration goes here, stepping through `self.depth` tracking fdyn_net = context.split()
        
        if self.training:
            return flow_predictions, cov_predictions, dyn_predictions
            
        return (flow_predictions[-1], ...), (cov_predictions[-1], ...), (dyn_predictions[-1],)
