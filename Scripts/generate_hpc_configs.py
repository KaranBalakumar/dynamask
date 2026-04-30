#!/usr/bin/env python3
"""
Generate training dataset configs for the HPC cluster.

Scans the actual directory structures on disk and produces YAML configs
with correct paths.  Run this ON the HPC, not locally.

Usage:
    python3 Scripts/generate_hpc_configs.py

Output:
    Config/Sequence/Training_Dataset/TartanAirV2_HPC_All.yaml
    Config/Sequence/Training_Dataset/VIODE_HPC_SelfSup.yaml
"""

import os
import sys
from pathlib import Path


# ---------------------------------------------------------------------------
# Paths (HPC-specific)
# ---------------------------------------------------------------------------
TARTANAIR2_ROOT = Path("/scratch/amukherjee/karan/tartan2/tartanair2")
VIODE_ROOT = Path("/home/amukherjee/Workbenches/balu/viode/viode_tartan")

OUT_DIR = Path(__file__).resolve().parent.parent / "Config" / "Sequence" / "Training_Dataset"


# ---------------------------------------------------------------------------
# Common TartanAirv2 entry template
# ---------------------------------------------------------------------------
TARTANAIR_ENTRY = """\
-   type: TartanAirv2
    name: {name}
    args:
        root: {root}
        compressed: true
        use_real_imu: true
        gravity: 9.81
        fixed_imu_samples: 10
        {cam_k_line}
        baseline: {baseline}
        imu_sim:
            acc_bias: [0.0, 0.0, 0.0]
            acc_init_bias_noise: [0.0, 0.0, 0.0]
            acc_bias_instability: [0.0, 0.0, 0.0]
            acc_random_walk: [0.0, 0.0, 0.0]
            gyro_bias: [0.0, 0.0, 0.0]
            gyro_init_bias_noise: [0.0, 0.0, 0.0]
            gyro_bias_instability: [0.0, 0.0, 0.0]
            gyro_random_walk: [0.0, 0.0, 0.0]
        gtDepth: {gt_depth}
        gtPose: true
        gtFlow: {gt_flow}
"""

# VIODE dataset: 752×480 images, specific intrinsics (verified from H5 attrs)
VIODE_CAM_K = [[344.0, 0.0, 376.0], [0.0, 344.0, 240.0], [0.0, 0.0, 1.0]]
VIODE_BASELINE = 0.12

# TartanAir2 640×640: default K is correct for geometry (cx=W/2, cy=H/2).
# fx=fy=320 matches the TartanAir v2 convention (90° HFOV).
TARTANAIR2_CAM_K = [[320.0, 0.0, 320.0], [0.0, 320.0, 320.0], [0.0, 0.0, 1.0]]
TARTANAIR2_BASELINE = 0.25


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def scan_tartanair2_scenes(root: Path) -> list[dict]:
    """Walk the TartanAir2 directory tree and return a list of entries.

    Each entry dict has keys: name, root, gt_depth, gt_flow
    """
    entries: list[dict] = []
    if not root.exists():
        print(f"[ERROR] TartanAir2 root not found: {root}", file=sys.stderr)
        return entries

    for scene_dir in sorted(root.iterdir()):
        if not scene_dir.is_dir():
            continue
        scene_name = scene_dir.name

        for difficulty in ("Data_easy", "Data_hard"):
            diff_dir = scene_dir / difficulty
            if not diff_dir.is_dir():
                continue

            # P000, P001, ...
            for p_dir in sorted(diff_dir.iterdir()):
                if not p_dir.is_dir() or not p_dir.name.startswith("P"):
                    continue

                seq_name = f"{scene_name}_{p_dir.name}"
                root_path = str(p_dir)

                entries.append({
                    "name": seq_name,
                    "root": root_path,
                    "cam_k": TARTANAIR2_CAM_K,
                    "baseline": TARTANAIR2_BASELINE,
                })

    return entries


def scan_viode_scenes(root: Path) -> list[dict]:
    """Walk the VIODE directory tree and return a list of entries.

    VIODE on HPC is in TartanAirv2-style layout:
        viode_tartan/city_day/3_high/image_lcam_front/
                                         /imu/
                                         /pose_lcam_front.txt
    """
    entries: list[dict] = []
    if not root.exists():
        print(f"[ERROR] VIODE root not found: {root}", file=sys.stderr)
        return entries

    for env_dir in sorted(root.iterdir()):
        if not env_dir.is_dir():
            continue
        env_name = env_dir.name  # city_day, city_night, parking_lot

        for diff_dir in sorted(env_dir.iterdir()):
            if not diff_dir.is_dir():
                continue
            diff_name = diff_dir.name  # 0_none, 1_low, 2_mid, 3_high

            # Verify this looks like a valid sequence directory
            lcam = diff_dir / "image_lcam_front"
            imu = diff_dir / "imu"
            pose = diff_dir / "pose_lcam_front.txt"
            if not (lcam.exists() or imu.exists() or pose.exists()):
                continue

            seq_name = f"{env_name}_{diff_name}"
            root_path = str(diff_dir)

            entries.append({
                "name": seq_name,
                "root": root_path,
                "cam_k": VIODE_CAM_K,
                "baseline": VIODE_BASELINE,
            })

    return entries


def write_yaml_config(entries: list[dict], out_path: Path, gt_depth: bool, gt_flow: bool, header_comment: str) -> None:
    """Write a YAML dataset config file from a list of entries."""
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "w") as f:
        f.write(f"# {header_comment}\n")
        f.write(f"# {len(entries)} sequences, auto-generated from HPC directory scan\n")
        f.write("\n")

        for i, entry in enumerate(entries):
            cam_k = entry.get("cam_k")
            if cam_k is not None:
                cam_k_line = f"cam_K: {cam_k}"
            else:
                cam_k_line = ""
            f.write(TARTANAIR_ENTRY.format(
                name=entry["name"],
                root=entry["root"],
                cam_k_line=cam_k_line,
                baseline=entry.get("baseline", 0.25),
                gt_depth=str(gt_depth).lower(),
                gt_flow=str(gt_flow).lower(),
            ))
            if i < len(entries) - 1:
                f.write("\n")

    print(f"[OK] Wrote {len(entries)} sequences → {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    print("=" * 60)
    print("HPC Config Generator")
    print("=" * 60)

    # 1. TartanAir2 — supervised (GT depth + GT flow)
    print("\n[1/2] Scanning TartanAir2 scenes...")
    ta2_entries = scan_tartanair2_scenes(TARTANAIR2_ROOT)
    print(f"       Found {len(ta2_entries)} sequences across all scenes")

    ta2_out = OUT_DIR / "TartanAirV2_HPC_All.yaml"
    write_yaml_config(
        ta2_entries, ta2_out,
        gt_depth=True, gt_flow=True,
        header_comment="TartanAir2 — ALL scenes, Data_easy + Data_hard, supervised (GT depth + GT flow + GT pose), real IMU."
    )

    # 2. VIODE — self-supervised (no GT depth, no GT flow)
    print("\n[2/2] Scanning VIODE scenes...")
    viode_entries = scan_viode_scenes(VIODE_ROOT)
    print(f"       Found {len(viode_entries)} sequences")

    viode_out = OUT_DIR / "VIODE_HPC_SelfSup.yaml"
    write_yaml_config(
        viode_entries, viode_out,
        gt_depth=False, gt_flow=False,
        header_comment="VIODE — TartanAirv2 format, self-supervised (no GT depth/flow), real IMU, GT pose."
    )

    print("\n" + "=" * 60)
    print("Done. Generated:")
    print(f"  {ta2_out}")
    print(f"  {viode_out}")
    print("=" * 60)


if __name__ == "__main__":
    main()
