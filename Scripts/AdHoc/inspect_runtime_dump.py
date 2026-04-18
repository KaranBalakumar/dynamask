#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import torch


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect a runtime observability dump")
    parser.add_argument("dump", type=Path, help="Path to step_XXXXXX.pt dump")
    args = parser.parse_args()
    try:
        payload = torch.load(args.dump, map_location="cpu", weights_only=True)
    except TypeError as exc:
        raise RuntimeError(
            "This script requires a PyTorch build that supports safe loading "
            "(torch.load(..., weights_only=True))."
        ) from exc
    if not isinstance(payload, dict):
        raise ValueError(f"Invalid dump format in {args.dump}: expected dict root.")
    print("Top-level keys:", sorted(payload.keys()))
    meta = payload.get("meta", {})
    print("Meta:")
    for k in sorted(meta.keys()):
        print(f"  {k}: {meta[k]}")
    if "pgo" in payload:
        print("PGO keys:", sorted(payload["pgo"].keys()))
    if "drt" in payload:
        print("DRT keys:", sorted(payload["drt"].keys()))


if __name__ == "__main__":
    main()
