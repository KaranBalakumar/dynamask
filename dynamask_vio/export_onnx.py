"""
Export DynaMask-VIO to ONNX for deployment.

Usage:
    python -m dynamask_vio.export_onnx \
        --checkpoint checkpoints/best.ckpt \
        --output dynamask_vio.onnx
"""

import argparse

import torch
import yaml

from .models import DynaMaskVIO


def _load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


class DynaMaskONNXWrapper(torch.nn.Module):
    """Thin wrapper that flattens the dict output for ONNX export."""

    def __init__(self, model: DynaMaskVIO):
        super().__init__()
        self.model = model

    def forward(self, img_prev, img_curr, imu_window, imu_mask):
        out = self.model(img_prev, img_curr, imu_window, imu_mask)
        return (
            out["dynamic_mask"],
            out["flow_residual"],
            out["delta_R"],
            out["delta_v"],
            out["delta_p"],
        )


def export(checkpoint_path: str, config_path: str, output_path: str,
           opset: int = 17):
    cfg = _load_config(config_path)
    model = DynaMaskVIO(cfg)

    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state = ckpt.get("state_dict", ckpt)
    state = {k.replace("model.", "", 1): v for k, v in state.items()
             if k.startswith("model.")} or state
    model.load_state_dict(state, strict=False)
    model.eval()

    wrapper = DynaMaskONNXWrapper(model)

    H = cfg.get("data", {}).get("image_height", 480)
    W = cfg.get("data", {}).get("image_width", 640)
    N = cfg.get("data", {}).get("imu_max_window_size", 15)

    img_prev = torch.randn(1, 3, H, W)
    img_curr = torch.randn(1, 3, H, W)
    imu_window = torch.randn(1, N, 7)
    imu_mask = torch.ones(1, N, dtype=torch.bool)

    torch.onnx.export(
        wrapper,
        (img_prev, img_curr, imu_window, imu_mask),
        output_path,
        opset_version=opset,
        input_names=["img_prev", "img_curr", "imu_window", "imu_mask"],
        output_names=["dynamic_mask", "flow_residual",
                       "delta_R", "delta_v", "delta_p"],
        dynamic_axes={
            "img_prev": {0: "batch"},
            "img_curr": {0: "batch"},
            "imu_window": {0: "batch", 1: "imu_len"},
            "imu_mask": {0: "batch", 1: "imu_len"},
            "dynamic_mask": {0: "batch"},
            "flow_residual": {0: "batch"},
            "delta_R": {0: "batch"},
            "delta_v": {0: "batch"},
            "delta_p": {0: "batch"},
        },
    )

    # Verify
    import onnx
    onnx_model = onnx.load(output_path)
    onnx.checker.check_model(onnx_model)
    print(f"ONNX model exported and verified: {output_path}")
    print(f"  Inputs:  {[i.name for i in onnx_model.graph.input]}")
    print(f"  Outputs: {[o.name for o in onnx_model.graph.output]}")


def main():
    parser = argparse.ArgumentParser(description="Export DynaMask-VIO to ONNX")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--config", type=str,
                        default="dynamask_vio/configs/default.yaml")
    parser.add_argument("--output", type=str, default="dynamask_vio.onnx")
    parser.add_argument("--opset", type=int, default=17)
    args = parser.parse_args()

    export(args.checkpoint, args.config, args.output, args.opset)


if __name__ == "__main__":
    main()
