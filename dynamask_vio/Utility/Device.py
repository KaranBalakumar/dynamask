from __future__ import annotations

import torch


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
        return "cuda", f"cuda:{device_index}"
    if normalized_backend == "rocm":
        # ROCm is routed via torch's cuda device namespace.
        return "rocm", f"cuda:{device_index}"

    raise ValueError(f"Unsupported backend: {backend}")


def is_supported_device_string(device: str) -> bool:
    if not isinstance(device, str):
        return False

    value = device.lower().strip()
    if value == "cpu":
        return True

    def _matches(prefix: str) -> bool:
        if value == prefix:
            return True
        if not value.startswith(f"{prefix}:"):
            return False
        _, index = value.split(":", 1)
        return index.isdigit()

    return _matches("cuda") or _matches("rocm") or _matches("hip")


def set_odometry_device_fields(cfg: dict, device: str) -> None:
    for key, value in cfg.items():
        if key == "device":
            cfg[key] = device
        elif isinstance(value, dict):
            set_odometry_device_fields(value, device)
        elif isinstance(value, list):
            for item in value:
                if isinstance(item, dict):
                    set_odometry_device_fields(item, device)
