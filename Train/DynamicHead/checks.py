from __future__ import annotations

import torch.nn as nn


def assert_module_frozen(module: nn.Module, name: str) -> None:
    trainable = [n for n, p in module.named_parameters() if p.requires_grad]
    if trainable:
        raise RuntimeError(f"{name} is expected to be frozen, but has trainable params: {trainable[:8]}")


def assert_module_trainable(module: nn.Module, name: str) -> None:
    trainable = [n for n, p in module.named_parameters() if p.requires_grad]
    if len(trainable) == 0:
        raise RuntimeError(f"{name} has no trainable parameters")


def freeze_module(module: nn.Module) -> None:
    module.eval()
    for p in module.parameters():
        p.requires_grad_(False)

