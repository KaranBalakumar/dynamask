from .convgru import ConvGRUCell, ConvLSTMCell
from .film import FiLM
from .head import StaticConfidenceHead


def build_head(config=None, **kwargs) -> StaticConfidenceHead:
    if kwargs:
        return StaticConfidenceHead(**kwargs)
    if config is None:
        return StaticConfidenceHead()
    return StaticConfidenceHead.from_config(config)


__all__ = [
    "ConvGRUCell",
    "ConvLSTMCell",
    "FiLM",
    "StaticConfidenceHead",
    "build_head",
]
