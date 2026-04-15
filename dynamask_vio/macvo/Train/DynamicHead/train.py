from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

import Module
from DataLoader import DynamicHeadTrainDataset
from Utility.Config import load_config
from Utility.PrettyPrint import Logger
from .calibrate import fit_temperature
from .loop import run_window


def _build_loader(data_cfg, batch_size: int, num_workers: int) -> DataLoader:
    dataset = DynamicHeadTrainDataset(data_cfg)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=True)


def _assert_freeze(frontend) -> None:
    for p in frontend.model.parameters():
        assert not p.requires_grad, "FlowFormer backbone must be frozen."
    for p in frontend.imu_encoder.corrector.parameters():
        assert not p.requires_grad, "AirIMU corrector must be frozen."
    assert any(p.requires_grad for p in frontend.imu_encoder.feature_mlp.parameters()), "FeatureMLP must remain trainable."
    assert frontend.model.training is False, "Frozen FlowFormer must stay in eval mode."


def main():
    parser = argparse.ArgumentParser(description="Train static-confidence head")
    parser.add_argument("--config", type=str, default="Config/Experiment/StaticConfidenceHead/viode.yaml")
    args = parser.parse_args()

    cfg, _ = load_config(Path(args.config))
    device = torch.device(cfg.model.frontend.args.device)

    frontend = Module.IFrontend.instantiate(cfg.model.frontend.type, cfg.model.frontend.args)
    frontend.dynamic_head.train()
    frontend.imu_encoder.feature_mlp.train()
    _assert_freeze(frontend)

    train_loader = _build_loader(cfg.data.train, int(cfg.train.batch_size), int(cfg.train.num_workers))
    val_loader = _build_loader(cfg.data.val, int(cfg.train.batch_size), int(cfg.train.num_workers))

    params = list(frontend.dynamic_head.parameters()) + list(frontend.imu_encoder.feature_mlp.parameters())
    optimizer = torch.optim.AdamW(params, lr=float(cfg.train.lr), weight_decay=float(cfg.train.weight_decay))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=int(cfg.train.total_steps))

    total_steps = 0
    smoke_test_checked = False
    for epoch in range(int(cfg.train.epochs)):
        for batch in train_loader:
            optimizer.zero_grad(set_to_none=True)
            loss, metrics = run_window(frontend, batch, cfg.loss, device)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, float(cfg.train.grad_clip_norm))
            optimizer.step()
            scheduler.step()
            total_steps += 1

            if not smoke_test_checked:
                has_grad = any(p.grad is not None for p in frontend.model.parameters())
                assert not has_grad, "Frozen FlowFormer unexpectedly received gradients."
                smoke_test_checked = True

            if total_steps % 20 == 0:
                Logger.write("info", f"step={total_steps} loss={loss.item():.5f} metrics={metrics}")
            if total_steps >= int(cfg.train.total_steps):
                break
        if total_steps >= int(cfg.train.total_steps):
            break

    if hasattr(cfg, "calibration") and bool(cfg.calibration.enabled):
        t_new = fit_temperature(
            frontend=frontend,
            loader=val_loader,
            loss_cfg=cfg.loss,
            device=device,
            optimizer_steps=int(cfg.calibration.optimizer_steps),
            t_min=float(cfg.calibration.temperature_min),
            t_max=float(cfg.calibration.temperature_max),
        )
        Logger.write("info", f"Calibrated temperature={t_new:.5f}")

    save_path = Path(getattr(cfg.train, "save_path", "Model/StaticConfidenceHead/latest.pth"))
    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "dynamic_head": frontend.dynamic_head.state_dict(),
            "feature_mlp": frontend.imu_encoder.feature_mlp.state_dict(),
            "temperature": float(frontend.dynamic_head.T_calib.item()),
            "config_path": args.config,
        },
        save_path,
    )
    Logger.write("info", f"Saved checkpoint to {save_path}")


if __name__ == "__main__":
    main()
