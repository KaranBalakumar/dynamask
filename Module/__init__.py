from __future__ import annotations


__all__ = [
    "ICovariance2to3",
    "IKeypointSelector",
    "IMotionModel",
    "IObservationFilter",
    "IMapProcessor",
    "IKeyframeSelector",
    "IOptimizer",
    "IStereoDepth",
    "IMatcher",
    "IFrontend",
    "DRTInitConfig",
    "DRTInitResult",
]


def __getattr__(name: str):
    if name == "ICovariance2to3":
        from .Covariance import ICovariance2to3
        return ICovariance2to3
    if name == "IKeypointSelector":
        from .KeypointSelector import IKeypointSelector
        return IKeypointSelector
    if name == "IMotionModel":
        from .MotionModel import IMotionModel
        return IMotionModel
    if name == "IObservationFilter":
        from .OutlierFilter import IObservationFilter
        return IObservationFilter
    if name == "IMapProcessor":
        from .MapProcessor import IMapProcessor
        return IMapProcessor
    if name == "IKeyframeSelector":
        from .KeyframeSelector import IKeyframeSelector
        return IKeyframeSelector
    if name == "IOptimizer":
        from .Optimization import IOptimizer
        return IOptimizer
    if name == "IStereoDepth":
        from .Frontend.StereoDepth import IStereoDepth
        return IStereoDepth
    if name == "IMatcher":
        from .Frontend.Matching import IMatcher
        return IMatcher
    if name == "IFrontend":
        from .Frontend.Frontend import IFrontend
        return IFrontend
    if name in {"DRTInitConfig", "DRTInitResult"}:
        from .Initialization import DRTInitConfig, DRTInitResult
        return {"DRTInitConfig": DRTInitConfig, "DRTInitResult": DRTInitResult}[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
