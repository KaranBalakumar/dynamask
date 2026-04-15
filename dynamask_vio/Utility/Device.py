from __future__ import annotations

import torch


def canonicalize_torch_device(device: str | torch.device) -> str:
    """
    Map accepted config device aliases onto torch-compatible device strings.
    Returned values are always "cpu", "cuda", or "cuda:<idx>".
    """
    value = str(device).lower().strip()

    if value == "cpu":
        return "cpu"

    for prefix in ("cuda", "rocm", "hip"):
        if value == prefix:
            return "cuda"
        if value.startswith(f"{prefix}:"):
            _, index = value.split(":", 1)
            if index.isdigit():
                return f"cuda:{int(index)}"
            break

    raise ValueError(f"Unsupported device string: {device}")


def normalize_backend_device(backend: str, device_index: int) -> tuple[str, str]:
    if not isinstance(device_index, int) or device_index < 0:
        raise ValueError(f"Invalid device index: {device_index}")

    normalized_backend = backend.lower()
    if normalized_backend == "auto":
        if torch.cuda.is_available():
            normalized_backend = "rocm" if torch.version.hip is not None else "cuda"
        else:
            normalized_backend = "cpu"

    if normalized_backend == "cpu":
        return "cpu", "cpu"
    if normalized_backend == "cuda":
        return "cuda", canonicalize_torch_device(f"cuda:{device_index}")
    if normalized_backend == "rocm":
        # ROCm is routed via torch's cuda device namespace.
        return "rocm", canonicalize_torch_device(f"rocm:{device_index}")

    raise ValueError(f"Unsupported backend: {backend}")


def is_supported_device_string(device: str) -> bool:
    try:
        canonicalize_torch_device(device)
    except ValueError:
        return False
    return True


def set_odometry_device_fields(cfg: dict, device: str) -> None:
    canonical_device = canonicalize_torch_device(device)
    for key, value in cfg.items():
        if key == "device":
            cfg[key] = canonical_device
        elif isinstance(value, dict):
            set_odometry_device_fields(value, canonical_device)
        elif isinstance(value, list):
            for item in value:
                if isinstance(item, dict):
                    set_odometry_device_fields(item, canonical_device)
