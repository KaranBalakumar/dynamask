"""Download pretrained weights required for DynaMask V2.

RAFT pretrained weights (raft-things.pth):
  - Trained on FlyingThings3D by Teed & Deng (ECCV 2020)
  - Used to initialize feature encoder (fnet) and context encoder (cnet)
  - ~5.3M params total; our encoder is a subset

Usage:
    python -m dynamask_vio.download_weights
    python -m dynamask_vio.download_weights --output-dir ./weights
"""

import argparse
import os
import hashlib
import sys

# RAFT pretrained checkpoint hosted on the RAFT authors' Dropbox
# This is the "raft-things.pth" model trained on FlyingThings3D
RAFT_THINGS_URL = "https://dl.dropboxusercontent.com/s/4j4z58wuv8o0mfz/models.zip"
RAFT_THINGS_FILENAME = "raft-things.pth"

# Alternative: if user already has the RAFT repo cloned
RAFT_LOCAL_PATHS = [
    "references/DPVO/thirdparty/RAFT/models/raft-things.pth",
    "references/RAFT/models/raft-things.pth",
    "../RAFT/models/raft-things.pth",
]


def _sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def download_raft_weights(output_dir: str) -> str:
    """Download or locate RAFT-Things pretrained weights.

    Returns path to the checkpoint file.
    """
    target = os.path.join(output_dir, RAFT_THINGS_FILENAME)

    # Already downloaded?
    if os.path.exists(target):
        print(f"[weights] {RAFT_THINGS_FILENAME} already exists at {target}")
        return target

    # Check local reference paths
    for local_path in RAFT_LOCAL_PATHS:
        if os.path.exists(local_path):
            print(f"[weights] Found RAFT weights at {local_path}")
            import shutil
            os.makedirs(output_dir, exist_ok=True)
            shutil.copy2(local_path, target)
            print(f"[weights] Copied to {target}")
            return target

    # Download from torch hub or direct URL
    os.makedirs(output_dir, exist_ok=True)
    print(f"[weights] Downloading RAFT-Things weights...")
    print(f"[weights] This may take a few minutes (~20MB)")

    try:
        import torch
        # Try torch.hub.download_url_to_file (handles redirects, shows progress)
        zip_path = os.path.join(output_dir, "raft_models.zip")
        torch.hub.download_url_to_file(RAFT_THINGS_URL, zip_path)

        # Extract the specific checkpoint
        import zipfile
        with zipfile.ZipFile(zip_path) as zf:
            # Find raft-things.pth inside the zip
            for name in zf.namelist():
                if name.endswith("raft-things.pth"):
                    print(f"[weights] Extracting {name}...")
                    data = zf.read(name)
                    with open(target, "wb") as f:
                        f.write(data)
                    break
            else:
                # If exact name not found, list contents
                print(f"[weights] Zip contents: {zf.namelist()}")
                raise FileNotFoundError("raft-things.pth not found in archive")

        os.remove(zip_path)
        print(f"[weights] Saved to {target}")
        return target

    except Exception as e:
        print(f"[weights] Download failed: {e}")
        print(f"[weights] Please manually download RAFT weights:")
        print(f"  1. Clone https://github.com/princeton-vl/RAFT")
        print(f"  2. Run: ./download_models.sh")
        print(f"  3. Copy models/raft-things.pth to {target}")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="Download pretrained weights for DynaMask V2")
    parser.add_argument("--output-dir", type=str,
                        default="dynamask_vio/weights",
                        help="Directory to save weights")
    args = parser.parse_args()

    print("=" * 60)
    print("DynaMask V2 — Pretrained Weight Download")
    print("=" * 60)

    path = download_raft_weights(args.output_dir)
    print(f"\n[Done] RAFT weights: {path}")

    # Verify the checkpoint can be loaded
    try:
        import torch
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        n_keys = len(ckpt) if isinstance(ckpt, dict) else 0
        print(f"[Verify] Checkpoint has {n_keys} keys")

        # Show encoder-relevant keys
        fnet_keys = [k for k in ckpt if "fnet" in k or "cnet" in k]
        print(f"[Verify] Found {len(fnet_keys)} encoder keys (fnet/cnet)")
    except Exception as e:
        print(f"[Verify] Warning: could not verify checkpoint: {e}")

    print(f"\nTo use these weights, set in your config:")
    print(f"  model:")
    print(f"    raft_checkpoint: \"{path}\"")


if __name__ == "__main__":
    main()
