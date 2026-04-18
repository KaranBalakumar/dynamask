from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
import torch

from Train.DynamicHead.calibrate import calibrate_temperature
from Train.DynamicHead.eval import evaluate_loader
from Train.DynamicHead.loop import DynamicHeadTrainer, build_dataloader
from Utility.Config import load_config
from Utility.PrettyPrint import Logger


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_dynamic_head(config_path: Path) -> None:
    cfg, _ = load_config(config_path)
    model_cfg = cfg.Model
    train_cfg = cfg.Train
    eval_cfg = getattr(cfg, "Evaluate", None)

    _set_seed(int(getattr(model_cfg, "seed", 1234)))
    trainer = DynamicHeadTrainer(model_cfg)
    train_loader = build_dataloader(train_cfg, shuffle=True)
    val_loader = build_dataloader(eval_cfg, shuffle=False) if eval_cfg is not None else None

    epochs = int(getattr(model_cfg, "epochs", 5))
    eval_every = int(getattr(model_cfg, "eval_every", 1))
    save_dir = Path(getattr(model_cfg, "save_dir", "Model"))
    save_dir.mkdir(parents=True, exist_ok=True)

    global_step = 0
    for epoch in range(epochs):
        epoch_loss = 0.0
        n = 0
        for pair in train_loader:
            out = trainer.train_step(pair)
            epoch_loss += out.loss
            n += 1
            global_step += 1
            if global_step % int(getattr(model_cfg, "log_every", 20)) == 0:
                Logger.write("info", f"[train] step={global_step} loss={out.loss:.5f} valid={out.metrics.get('valid_ratio', 0.0):.3f}")

        Logger.write("info", f"[train] epoch={epoch + 1}/{epochs} mean_loss={(epoch_loss / max(n, 1)):.5f}")
        if val_loader is not None and ((epoch + 1) % eval_every == 0):
            metrics = evaluate_loader(trainer, val_loader)
            Logger.write("info", f"[eval] epoch={epoch + 1} metrics={metrics}")

        ckpt_path = save_dir / f"dynamic_head_epoch{epoch + 1}.pth"
        torch.save(
            {
                "head": trainer.head.state_dict(),
                "imu_encoder_head": trainer.imu_encoder.head.state_dict(),
                "optimizer": trainer.optimizer.state_dict(),
                "epoch": epoch + 1,
            },
            ckpt_path,
        )
        Logger.write("info", f"Saved checkpoint: {ckpt_path}")

    if val_loader is not None and bool(getattr(model_cfg, "calibrate_temperature", True)):
        cal = calibrate_temperature(trainer, val_loader, max_batches=int(getattr(model_cfg, "calibration_batches", 100)))
        Logger.write("info", f"[calibrate] T={cal.temperature:.4f}, T_vis={cal.temperature_visible}, nll={cal.nll:.6f}, ece={cal.ece:.6f}")
        cal_path = save_dir / "dynamic_head_calibrated.pth"
        temperature = trainer.head.get_buffer("temperature")
        temperature_visible = trainer.head.get_buffer("temperature_visible")
        torch.save(
            {
                "head": trainer.head.state_dict(),
                "imu_encoder_head": trainer.imu_encoder.head.state_dict(),
                "temperature": float(temperature.item()),
                "temperature_visible": float(temperature_visible.item()),
            },
            cal_path,
        )
        Logger.write("info", f"Saved calibrated checkpoint: {cal_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train static confidence head on frozen MAC-VO frontend + AirIMU context")
    parser.add_argument("--config", type=Path, default=Path("Config/Train/StaticConfidenceHeadVIODE.yaml"))
    args = parser.parse_args()
    train_dynamic_head(args.config)


if __name__ == "__main__":
    main()
