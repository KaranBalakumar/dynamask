from .convgru import ConvGRUCell, ConvLSTMCell
from .head import StaticConfidenceHead


def build_head(config):
    return StaticConfidenceHead.from_config(config)


__all__ = [
    "ConvGRUCell",
    "ConvLSTMCell",
    "StaticConfidenceHead",
    "build_head",
]

