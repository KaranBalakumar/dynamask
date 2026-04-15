from .convgru import ConvGRUCell, ConvLSTMCell
from .film import FiLM
from .head import StaticConfidenceHead


def build_head(**kwargs) -> StaticConfidenceHead:
    return StaticConfidenceHead(**kwargs)


__all__ = [
    "ConvGRUCell",
    "ConvLSTMCell",
    "FiLM",
    "StaticConfidenceHead",
    "build_head",
]
