import torch


def build_flowformer_dyn(cfg, encoder_dtype: torch.dtype = torch.float32):
    from .flownet import FlowFormerDyn
    return FlowFormerDyn(cfg["latentcostformer"], encoder_dtype)
