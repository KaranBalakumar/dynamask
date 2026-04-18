#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Delete stale *.partial dump files")
    parser.add_argument("run_dir", type=Path, help="Run directory (contains dumps/)")
    args = parser.parse_args()
    dumps = args.run_dir / "dumps"
    if not dumps.exists():
        print("No dumps directory found.")
        return
    removed = 0
    for p in dumps.glob("*.partial"):
        p.unlink(missing_ok=True)
        removed += 1
    print(f"Removed {removed} partial dump files from {dumps}")


if __name__ == "__main__":
    main()

