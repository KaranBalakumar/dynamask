from __future__ import annotations

from pathlib import Path

import torch

from Utility.Observability import collect_runtime_dump, validate_dump_schema


def test_runtime_dump_schema_roundtrip(tmp_path: Path):
    payload = collect_runtime_dump(
        step=10,
        epoch=2,
        seq_id="seq-a",
        frame_idx=42,
        batch_idx=0,
        cadence="regular",
        data={
            "image_L": torch.randn(3, 16, 16),
            "imu_window": torch.randn(8, 6),
        },
        frontend={"depth": torch.rand(1, 16, 16)},
        pgo={"chi2_init": torch.tensor([1.0]), "chi2_final": torch.tensor([0.5])},
        drt={"accept": torch.tensor([True])},
    )
    validate_dump_schema(payload)
    dump_file = tmp_path / "dump.pt"
    torch.save(payload, dump_file)
    loaded = torch.load(dump_file, map_location="cpu")
    validate_dump_schema(loaded)
    assert loaded["meta"]["step"] == 10
    assert loaded["data"]["image_L"].dtype == torch.float16
    assert loaded["frontend"]["depth"].dtype == torch.float16

