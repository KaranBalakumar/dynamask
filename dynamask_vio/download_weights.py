"""Download pretrained weights required for DynaMask V2.5."""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import zipfile


RAFT_THINGS_URL = "https://dl.dropboxusercontent.com/s/4j4z58wuv8o0mfz/models.zip"
RAFT_THINGS_FILENAME = "raft-things.pth"

AIRIMU_EUROC_URL = (
    "https://github.com/sleepycan/AirIMU/releases/download/pretrained_model_euroc/EuRoCWholeaug.zip"
)
AIRIMU_FILENAME = "airimu_codenet_euroc.ckpt"

RAFT_LOCAL_PATHS = [
    "references/DPVO/thirdparty/RAFT/models/raft-things.pth",
    "references/RAFT/models/raft-things.pth",
    "../RAFT/models/raft-things.pth",
]


def _download_url(url: str, out_path: str) -> None:
    import torch

    torch.hub.download_url_to_file(url, out_path)


def _extract_first_matching(zip_path: str, target_path: str, candidates: tuple[str, ...]) -> str:
    with zipfile.ZipFile(zip_path) as zf:
        names = zf.namelist()
        for name in names:
            lower = name.lower()
            if any(lower.endswith(cand) for cand in candidates):
                with open(target_path, "wb") as f:
                    f.write(zf.read(name))
                return name
    raise FileNotFoundError(f"No matching file {candidates} found in {zip_path}")


def download_raft_weights(output_dir: str) -> str:
    target = os.path.join(output_dir, RAFT_THINGS_FILENAME)
    if os.path.exists(target):
        print(f"[weights] {RAFT_THINGS_FILENAME} already exists at {target}")
        return target

    for local_path in RAFT_LOCAL_PATHS:
        if os.path.exists(local_path):
            print(f"[weights] Found RAFT weights at {local_path}")
            os.makedirs(output_dir, exist_ok=True)
            shutil.copy2(local_path, target)
            print(f"[weights] Copied to {target}")
            return target

    os.makedirs(output_dir, exist_ok=True)
    zip_path = os.path.join(output_dir, "raft_models.zip")
    print("[weights] Downloading RAFT-Things weights...")
    _download_url(RAFT_THINGS_URL, zip_path)

    try:
        extracted = _extract_first_matching(zip_path, target, ("/raft-things.pth", "raft-things.pth"))
        print(f"[weights] Extracted {extracted} -> {target}")
    finally:
        if os.path.exists(zip_path):
            os.remove(zip_path)
    return target


def download_airimu_weights(output_dir: str) -> str:
    target = os.path.join(output_dir, AIRIMU_FILENAME)
    if os.path.exists(target):
        print(f"[weights] {AIRIMU_FILENAME} already exists at {target}")
        return target

    os.makedirs(output_dir, exist_ok=True)
    zip_path = os.path.join(output_dir, "airimu_euroc.zip")
    print("[weights] Downloading AirIMU EuRoC checkpoint...")
    _download_url(AIRIMU_EUROC_URL, zip_path)

    try:
        extracted = _extract_first_matching(
            zip_path,
            target,
            (
                ".ckpt",
                ".pth",
                ".pt",
            ),
        )
        print(f"[weights] Extracted {extracted} -> {target}")
    finally:
        if os.path.exists(zip_path):
            os.remove(zip_path)
    return target


def _verify_torch_load(path: str, label: str) -> None:
    try:
        import torch

        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        if isinstance(ckpt, dict):
            print(f"[verify] {label}: loaded dict with {len(ckpt)} top-level keys")
        else:
            print(f"[verify] {label}: loaded object type {type(ckpt).__name__}")
    except Exception as e:
        print(f"[verify] Warning: could not verify {label} checkpoint: {e}")


def main():
    parser = argparse.ArgumentParser(description="Download pretrained weights for DynaMask V2.5")
    parser.add_argument("--output-dir", type=str, default="dynamask_vio/weights")
    parser.add_argument("--skip-raft", action="store_true")
    parser.add_argument("--skip-airimu", action="store_true")
    args = parser.parse_args()

    print("=" * 60)
    print("DynaMask V2.5 — Pretrained Weight Download")
    print("=" * 60)

    raft_path = None
    airimu_path = None
    try:
        if not args.skip_raft:
            raft_path = download_raft_weights(args.output_dir)
            _verify_torch_load(raft_path, "RAFT")
        if not args.skip_airimu:
            airimu_path = download_airimu_weights(args.output_dir)
            _verify_torch_load(airimu_path, "AirIMU")
    except Exception as e:
        print(f"[weights] Download failed: {e}")
        sys.exit(1)

    print("\n[Done]")
    if raft_path:
        print(f"  RAFT checkpoint:   {raft_path}")
    if airimu_path:
        print(f"  AirIMU checkpoint: {airimu_path}")

    if raft_path or airimu_path:
        print("\nSuggested config:")
        print("model:")
        if raft_path:
            print(f'  raft_checkpoint: "{raft_path}"')
        if airimu_path:
            print(f'  airimu_weights_path: "{airimu_path}"')


if __name__ == "__main__":
    main()
