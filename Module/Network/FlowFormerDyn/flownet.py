import torch
import torch.nn.functional as F
from collections import OrderedDict

from ..FlowFormer.core.utils import InputPadder
from ..FlowFormer.core.transformer import FlowFormer
from .dynhead import MemoryDynDecoder


class FlowFormerDyn(FlowFormer):
    """FlowFormer with a joint flow + cov + dyn decoder.

    Structurally mirrors FlowFormerCov: subclass FlowFormer, replace its `memory_decoder`
    with `MemoryDynDecoder` (which itself subclasses MemoryDecoder and adds cov + dyn
    siblings to the flow update block). The dyn branch is conditioned on
    (f_imu, imu_tokens) produced by IMUContext.
    """

    def __init__(
        self,
        cfg,
        encoder_dtype: torch.dtype = torch.float32,
        decoder_dtype: torch.dtype = torch.float32,
    ):
        super().__init__(cfg)
        self.memory_decoder = MemoryDynDecoder(self.cfg, decoder_dtype)

        self.enc_dtype = encoder_dtype
        self.context_encoder = self.context_encoder.to(dtype=self.enc_dtype)
        self.memory_encoder  = self.memory_encoder.to(dtype=self.enc_dtype)

    def forward(                                                # pyright: ignore[reportIncompatibleMethodOverride]
        self,
        image1: torch.Tensor,
        image2: torch.Tensor,
        f_imu: torch.Tensor,
        imu_tokens: torch.Tensor,
        prev_dyn_net: torch.Tensor | None = None,
        prev_flow:    torch.Tensor | None = None,
    ):
        image1 = ((2 * image1) - 1.0).to(dtype=self.enc_dtype)
        image2 = ((2 * image2) - 1.0).to(dtype=self.enc_dtype)

        with torch.cuda.nvtx.range("Context Encoder"):
            context = self.context_encoder(image1)

        with torch.cuda.nvtx.range("Memory Encoder"):
            cost_memory, cost_maps = self.memory_encoder(image1, image2, context)
            cost_maps = cost_maps.float()
            context   = context.float()

        with torch.cuda.nvtx.range("Memory Decoder"):
            flow_predictions, cov_predictions, dyn_predictions = self.memory_decoder(
                cost_memory, context, cost_maps, f_imu, imu_tokens,
                prev_dyn_net=prev_dyn_net,
                prev_flow=prev_flow,
            )

        return flow_predictions, cov_predictions, dyn_predictions

    @torch.no_grad()
    @torch.inference_mode()
    def inference(                                              # pyright: ignore[reportIncompatibleMethodOverride]
        self,
        image1: torch.Tensor,
        image2: torch.Tensor,
        f_imu: torch.Tensor,
        imu_tokens: torch.Tensor,
    ):
        padder = InputPadder(image1.shape)
        image1, image2 = padder.pad(image1, image2)
        flow_pre, cov_pre, dyn_pre = self.forward(image1, image2, f_imu, imu_tokens)

        flow_pre = padder.unpad(flow_pre[0])
        cov_pre  = padder.unpad(cov_pre[0])
        # Dyn logits are at H/4 — bilinear-upsample to full resolution, then unpad.
        dyn_full = F.interpolate(
            dyn_pre[0].float(),
            size=(image1.shape[-2], image1.shape[-1]),
            mode="bilinear",
            align_corners=False,
        )
        dyn_full = padder.unpad(dyn_full)
        return flow_pre, torch.exp(cov_pre * 2), dyn_full

    def load_ddp_state_dict(self, ckpt: OrderedDict):
        cvt_ckpt: OrderedDict = OrderedDict()
        for k in ckpt:
            key = k[7:] if k.startswith("module.") else k
            cvt_ckpt[key] = ckpt[k]
        self.load_state_dict(cvt_ckpt, strict=False)
