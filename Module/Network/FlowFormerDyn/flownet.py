# hopefully same as FlowFormerCov but without the covariance shit. images go in, optical flow comes out.

import torch
import torch.nn as nn
from collections import OrderedDict

from ..FlowFormer.core.utils import InputPadder
from ..FlowFormer.core.twins_svt import TwinsSVTLarge
from ..FlowFormer.core.encoder import MemoryEncoder
from ..FlowFormer.core.decoder import MemoryDecoder


class FlowFormerDyn(nn.Module):

    def __init__(self, cfg, encoder_dtype: torch.dtype = torch.float32):
        super().__init__()
        self.cfg = cfg
        self.enc_dtype = encoder_dtype

        self.context_encoder = TwinsSVTLarge(pretrained=cfg.pretrain).to(dtype=encoder_dtype)
        self.memory_encoder = MemoryEncoder(cfg).to(dtype=encoder_dtype)
        self.memory_decoder = MemoryDecoder(cfg)

    def forward(self, image1: torch.Tensor, image2: torch.Tensor) -> list[torch.Tensor]:
        image1 = ((2 * image1) - 1.0).to(dtype=self.enc_dtype)
        image2 = ((2 * image2) - 1.0).to(dtype=self.enc_dtype)

        context = self.context_encoder(image1)
        cost_memory, cost_maps = self.memory_encoder(image1, image2, context)
        cost_maps = cost_maps.float()
        context = context.float()

        flow_predictions, _ = self.memory_decoder(
            cost_memory, context, cost_maps, self.cfg.query_latent_dim, flow_init=None
        )
        return flow_predictions

    @torch.no_grad()
    @torch.inference_mode()
    def inference(self, image1: torch.Tensor, image2: torch.Tensor) -> torch.Tensor:
        padder = InputPadder(image1.shape)
        image1, image2 = padder.pad(image1, image2)
        flow_predictions = self.forward(image1, image2)
        return padder.unpad(flow_predictions[-1])

    def load_ddp_state_dict(self, ckpt: OrderedDict):
        cvt_ckpt = OrderedDict()
        for k in ckpt:
            key = k[7:] if k.startswith("module.") else k
            cvt_ckpt[key] = ckpt[k]
        self.load_state_dict(cvt_ckpt, strict=False)
