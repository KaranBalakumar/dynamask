"""DynaMask V2.5 training script (single-phase, 3-loss stack)."""

from __future__ import annotations

import argparse
import os
import time

import pytorch_lightning as pl
import torch
import yaml
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import ConcatDataset, DataLoader

from .data import EuRoCDataset, TartanAirDataset, VIODEDataset
from .losses.pairwise_rank import pairwise_rank_loss
from .losses.self_supervised import pose_consistency_loss
from .losses.smoothness import edge_aware_second_order_smoothness
from .models import DynaMaskVIO
from .models.differentiable_ba import differentiable_ba, select_static_correspondences

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass


def _load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def _merge_configs(base: dict, override: dict) -> dict:
    merged = base.copy()
    for k, v in override.items():
        if isinstance(v, dict) and k in merged and isinstance(merged[k], dict):
            merged[k] = _merge_configs(merged[k], v)
        else:
            merged[k] = v
    return merged


class DynaMaskLitModule(pl.LightningModule):
    """Lightning module for V2.5 single-phase training."""

    def __init__(self, cfg: dict):
        super().__init__()
        self.cfg = cfg
        self.save_hyperparameters(cfg)
        self.model = DynaMaskVIO(cfg)

        loss_cfg = cfg.get("loss", {})
        self.lambda_pose = float(loss_cfg.get("lambda_pose", 1.0))
        self.lambda_rank = float(loss_cfg.get("lambda_rank", 0.3))
        self.lambda_smooth = float(loss_cfg.get("lambda_smooth", 0.05))
        self.pose_warmup_epochs = int(loss_cfg.get("pose_warmup_epochs", 2))

        self.pose_rot_weight = float(loss_cfg.get("pose_rotation_weight", 10.0))
        self.pose_trans_weight = float(loss_cfg.get("pose_translation_weight", 1.0))
        self.pose_rot_delta = float(loss_cfg.get("pose_rotation_huber_delta", 0.1))
        self.pose_trans_delta = float(loss_cfg.get("pose_translation_huber_delta", 0.5))

        self.rank_pairs = int(loss_cfg.get("rank_pairs", 512))
        self.rank_margin = float(loss_cfg.get("rank_margin", 1.0))
        self.rank_static_quantile = float(loss_cfg.get("rank_static_quantile", 0.5))
        self.rank_min_spread = float(loss_cfg.get("rank_min_spread", 1e-4))
        self.iter_gamma = float(loss_cfg.get("iter_gamma", 0.8))
        self.smooth_beta = float(loss_cfg.get("smooth_beta", 10.0))

        ba_cfg = cfg.get("ba", {})
        self.ba_n_iters = int(ba_cfg.get("n_iters", 3))
        self.ba_base_damping = float(ba_cfg.get("base_damping", 1e-4))
        self.ba_ep_initial = float(ba_cfg.get("ep_initial", 10.0))
        self.ba_ep_final = float(ba_cfg.get("ep_final", 1.0))
        self.ba_ep_decay_epochs = int(ba_cfg.get("ep_decay_epochs", 10))
        self.ba_outlier_threshold = float(ba_cfg.get("outlier_threshold", 100.0))
        self.ba_convergence_threshold = float(ba_cfg.get("convergence_threshold", 50.0))
        self.ba_num_correspondences = int(ba_cfg.get("num_correspondences", 256))
        self.ba_min_static_points = int(ba_cfg.get("min_static_points", 64))

    def _pose_warmup(self) -> float:
        epoch = self.current_epoch + 1
        if self.pose_warmup_epochs <= 1:
            return 1.0
        alpha = (epoch - 1) / max(self.pose_warmup_epochs - 1, 1)
        return float(max(0.0, min(1.0, alpha)))

    def _get_phase(self) -> str:
        """Backward-compatible hook used by debug callbacks."""
        return "single"

    def forward(self, batch: dict) -> dict:
        return self.model(
            batch["img_prev"],
            batch["img_curr"],
            batch["imu_window"],
            batch["imu_mask"],
        )

    def _run_ba(self, outputs: dict, batch: dict) -> dict | None:
        if "intrinsics" not in batch:
            return None

        flow = outputs["flow"]
        score_prob = torch.sigmoid(outputs["score_logits_lowres"])
        correspondences, weights, valid_batch = select_static_correspondences(
            flow,
            score_prob,
            num_points=self.ba_num_correspondences,
            min_points=self.ba_min_static_points,
        )
        if not valid_batch.any():
            return None

        K = batch["intrinsics"]
        K_scaled = K.clone()
        K_scaled[:, 0, :] = K_scaled[:, 0, :] / 8.0
        K_scaled[:, 1, :] = K_scaled[:, 1, :] / 8.0
        K_scaled[:, 2, 2] = 1.0

        ba_result = differentiable_ba(
            correspondences=correspondences,
            weights=weights,
            K=K_scaled,
            R_init=outputs["delta_R"],
            epoch=self.current_epoch + 1,
            phase2_start=0,
            n_iters=self.ba_n_iters,
            base_damping=self.ba_base_damping,
            ep_initial=self.ba_ep_initial,
            ep_final=self.ba_ep_final,
            ep_decay_epochs=self.ba_ep_decay_epochs,
            outlier_threshold=self.ba_outlier_threshold,
            convergence_threshold=self.ba_convergence_threshold,
        )
        ba_result["weights"] = weights
        ba_result["valid_batch"] = valid_batch
        return ba_result

    def _compute_iter_losses(
        self,
        score_logits_per_iter: list[torch.Tensor],
        flow_predictions: list[torch.Tensor],
        img_curr: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        total_rank = flow_predictions[-1].sum() * 0.0
        total_smooth = flow_predictions[-1].sum() * 0.0
        total_w = 0.0
        n_iters = len(score_logits_per_iter)

        for i, score_i in enumerate(score_logits_per_iter):
            w = self.iter_gamma ** (n_iters - 1 - i)
            flow_i = flow_predictions[min(i, len(flow_predictions) - 1)]
            l_rank = pairwise_rank_loss(
                score_i,
                flow_i,
                n_pairs=self.rank_pairs,
                margin=self.rank_margin,
                static_quantile=self.rank_static_quantile,
                min_spread=self.rank_min_spread,
            )
            l_smooth = edge_aware_second_order_smoothness(score_i, img_curr, beta=self.smooth_beta)
            total_rank = total_rank + w * l_rank
            total_smooth = total_smooth + w * l_smooth
            total_w += w

        if total_w <= 0:
            return total_rank, total_smooth
        return total_rank / total_w, total_smooth / total_w

    def _compute_loss(self, batch: dict, outputs: dict) -> dict:
        losses: dict[str, torch.Tensor | float | dict] = {}

        img_curr_norm = 2.0 * (batch["img_curr"] / 255.0) - 0.5
        l_rank, l_smooth = self._compute_iter_losses(
            outputs["score_logits_per_iter"],
            outputs["flow_predictions"],
            img_curr_norm,
        )
        losses["rank"] = self.lambda_rank * l_rank
        losses["smooth"] = self.lambda_smooth * l_smooth

        ba_result = self._run_ba(outputs, batch)
        warmup = self._pose_warmup()
        losses["_pose_warmup"] = warmup

        if ba_result is not None and "gt_R" in batch and "gt_p" in batch:
            l_pose = pose_consistency_loss(
                ba_result["R"],
                ba_result["t"],
                batch["gt_R"],
                batch["gt_p"],
                valid_mask=ba_result["converged"],
                rot_weight=self.pose_rot_weight,
                trans_weight=self.pose_trans_weight,
                rot_huber_delta=self.pose_rot_delta,
                trans_huber_delta=self.pose_trans_delta,
            )
            losses["pose"] = warmup * self.lambda_pose * l_pose

        ba_diag: dict[str, float] = {}
        if ba_result is not None:
            ba_diag["convergence_rate"] = ba_result["converged"].mean().item()
            ba_diag["reproj_error_mean"] = ba_result["reproj_error"].pow(2).sum(-1).sqrt().mean().item()
            ba_diag["n_static_points"] = float(ba_result["weights"].shape[1])
            ba_diag["static_weight_mean"] = ba_result["weights"].mean().item()
            ba_diag["ran"] = 1.0
        else:
            ba_diag["convergence_rate"] = 0.0
            ba_diag["ran"] = 0.0
        losses["_ba_diag"] = ba_diag

        losses["total"] = sum(
            v for k, v in losses.items() if not k.startswith("_") and isinstance(v, torch.Tensor)
        )
        return losses

    def training_step(self, batch, batch_idx):
        outputs = self(batch)
        losses = self._compute_loss(batch, outputs)
        total = losses["total"]

        is_bad = isinstance(total, torch.Tensor) and not torch.isfinite(total).all().item()
        if is_bad:
            bad_terms = {
                k: float(v.detach().cpu())
                for k, v in losses.items()
                if isinstance(v, torch.Tensor) and v.numel() == 1
            }
            print(f"[train] non-finite loss at batch {batch_idx}; skipping. per-term={bad_terms}")
            anchor_param = next((p for p in self.model.parameters() if p.requires_grad), None)
            if anchor_param is not None:
                total = anchor_param.sum() * 0.0
            else:
                total = torch.zeros((), device=self.device, dtype=torch.float32, requires_grad=True)
            safe_zero = torch.zeros((), device=self.device, dtype=total.dtype)
            losses = {
                k: (
                    safe_zero
                    if isinstance(v, torch.Tensor) and v.numel() == 1 and not torch.isfinite(v)
                    else v
                )
                for k, v in losses.items()
            }
            losses["total"] = safe_zero

        for name, val in losses.items():
            if name.startswith("_"):
                continue
            self.log(f"train/loss_{name}", val, prog_bar=(name == "total"), sync_dist=True)
        self.log("train/pose_warmup", losses.get("_pose_warmup", 1.0))

        diag = losses.get("_ba_diag", {})
        if isinstance(diag, dict):
            for k, v in diag.items():
                self.log(f"train/ba_{k}", v, sync_dist=True)

        return total

    def validation_step(self, batch, batch_idx):
        outputs = self(batch)
        losses = self._compute_loss(batch, outputs)

        for name, val in losses.items():
            if name.startswith("_"):
                continue
            if isinstance(val, torch.Tensor) and val.numel() == 1 and not torch.isfinite(val):
                val = torch.zeros((), device=self.device)
                losses[name] = val
            self.log(f"val/loss_{name}", val, sync_dist=True)

        return losses["total"]

    def configure_optimizers(self):
        train_cfg = self.cfg.get("training", {})
        lr = float(train_cfg.get("lr", 2e-4))
        wd = float(train_cfg.get("weight_decay", 1e-4))
        encoder_lr_mult = float(train_cfg.get("encoder_lr_multiplier", 0.1))
        total_steps = int(train_cfg.get("total_steps", 240000))

        param_groups = self.model.get_parameter_groups(lr, encoder_lr_mult)
        optimizer = torch.optim.AdamW(param_groups, lr=lr, weight_decay=wd, betas=(0.9, 0.999))

        max_lrs = [group["lr"] for group in param_groups]
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=max_lrs,
            total_steps=total_steps,
            pct_start=0.1,
            cycle_momentum=False,
            anneal_strategy="cos",
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
        }


class V25DataModule(pl.LightningDataModule):
    """Single-phase datamodule with optional multi-dataset concatenation."""

    def __init__(self, cfg: dict):
        super().__init__()
        self.cfg = cfg
        self.batch_size = cfg.get("training", {}).get("batch_size", 8)
        self.num_workers = cfg.get("training", {}).get("num_workers", 8)
        data_cfg = cfg.get("data", {})
        self.enabled_datasets = {
            str(x).lower() for x in data_cfg.get("enabled_datasets", ["viode", "tartanair"])
        }

        self.viode_train = None
        self.viode_val = None
        self.euroc_train = None
        self.euroc_val = None
        self.tartanair_train = None

    @staticmethod
    def _concat_nonempty(*datasets):
        non_empty = [d for d in datasets if d is not None and len(d) > 0]
        if not non_empty:
            return None
        if len(non_empty) == 1:
            return non_empty[0]
        return ConcatDataset(non_empty)

    def setup(self, stage=None):
        paths = self.cfg.get("paths", {})

        if "viode" in self.enabled_datasets:
            viode_cfg_path = os.path.join(os.path.dirname(__file__), "configs", "viode.yaml")
            viode_cfg = _load_config(viode_cfg_path) if os.path.exists(viode_cfg_path) else {}
            viode_merged = _merge_configs(viode_cfg, self.cfg)
            viode_splits = viode_merged.get("splits", {})

            viode_hdf5_root = paths.get("viode_hdf5_root", paths.get("viode_root", "./dataset/viode_hdf5"))
            self.viode_train = VIODEDataset(
                viode_hdf5_root,
                viode_splits.get("train", []),
                viode_merged,
                split="train",
            )
            self.viode_val = VIODEDataset(
                viode_hdf5_root,
                viode_splits.get("val", []),
                viode_merged,
                split="val",
            )

        if "euroc" in self.enabled_datasets:
            euroc_cfg_path = os.path.join(os.path.dirname(__file__), "configs", "euroc.yaml")
            euroc_cfg = _load_config(euroc_cfg_path) if os.path.exists(euroc_cfg_path) else {}
            euroc_merged = _merge_configs(euroc_cfg, self.cfg)
            euroc_splits = euroc_merged.get("splits", {})

            self.euroc_train = EuRoCDataset(
                paths.get("euroc_root", "./dataset/euroc"),
                euroc_splits.get("train", []),
                euroc_merged,
                split="train",
            )
            self.euroc_val = EuRoCDataset(
                paths.get("euroc_root", "./dataset/euroc"),
                euroc_splits.get("val", []),
                euroc_merged,
                split="val",
            )

        if "tartanair" in self.enabled_datasets:
            ta_cfg_path = os.path.join(os.path.dirname(__file__), "configs", "tartanair.yaml")
            ta_cfg = _load_config(ta_cfg_path) if os.path.exists(ta_cfg_path) else {}
            ta_merged = _merge_configs(ta_cfg, self.cfg)

            self.tartanair_train = TartanAirDataset(
                paths.get("tartanair_root", "./dataset/tartanair"),
                ta_merged.get("environments", []),
                ta_merged.get("difficulties", []),
                ta_merged,
                split="train",
            )

    def train_dataloader(self):
        train_ds = self._concat_nonempty(self.tartanair_train, self.viode_train, self.euroc_train)
        if train_ds is None:
            raise RuntimeError("No non-empty training dataset found.")
        return DataLoader(
            train_ds,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True,
            persistent_workers=self.num_workers > 0,
        )

    def val_dataloader(self):
        val_ds = self._concat_nonempty(self.viode_val, self.euroc_val)
        if val_ds is None:
            val_ds = self._concat_nonempty(self.tartanair_train, self.viode_train, self.euroc_train)
        if val_ds is None:
            raise RuntimeError("No non-empty validation dataset found.")
        return DataLoader(
            val_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )


def main():
    # Avoid OpenMP fork deadlock: loading AirIMU's GRU weights initializes the
    # intraop thread pool in the main process, and DataLoader workers forked
    # afterwards inherit corrupted mutex state and hang in futex_wait_queue.
    # Resetting to 1 thread rebuilds the pool cleanly before workers fork.
    torch.set_num_threads(1)

    parser = argparse.ArgumentParser(description="DynaMask V2.5 Training")
    parser.add_argument("--config", type=str, default="dynamask_vio/configs/default.yaml")
    parser.add_argument("--gpus", type=int, default=1)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--no-wandb", action="store_true")
    parser.add_argument("--wandb-name", type=str, default=None)
    parser.add_argument("--wandb-tags", nargs="*", default=None)
    args = parser.parse_args()

    cfg = _load_config(args.config)
    train_cfg = cfg.get("training", {})
    paths = cfg.get("paths", {})
    wandb_cfg = cfg.get("wandb", {})
    offline_cfg = cfg.get("offline_logging", {})

    airimu_path = cfg.get("model", {}).get("airimu_weights_path", None)
    if not airimu_path:
        raise ValueError("model.airimu_weights_path is required for V2.5 training")
    if not os.path.exists(airimu_path):
        raise FileNotFoundError(f"AirIMU checkpoint not found: {airimu_path}")

    dm = V25DataModule(cfg)
    model = DynaMaskLitModule(cfg)

    checkpoint_cb = ModelCheckpoint(
        dirpath=paths.get("checkpoint_dir", "./checkpoints"),
        filename="dynamask-v25-{epoch:02d}-{val/loss_total:.3f}",
        monitor="val/loss_total",
        mode="min",
        save_top_k=3,
        save_last=True,
    )
    lr_monitor = LearningRateMonitor(logging_interval="step")
    callbacks = [checkpoint_cb, lr_monitor]

    loggers = []
    tb_logger = TensorBoardLogger(save_dir=paths.get("log_dir", "./logs"), name="dynamask_v25")
    loggers.append(tb_logger)

    use_wandb = not args.no_wandb
    if use_wandb:
        try:
            from pytorch_lightning.loggers import WandbLogger

            wandb_project = os.environ.get("WANDB_PROJECT", wandb_cfg.get("project", "dynamask-vio"))
            wandb_entity = os.environ.get("WANDB_ENTITY", wandb_cfg.get("entity", None)) or None
            wandb_logger = WandbLogger(
                project=wandb_project,
                entity=wandb_entity,
                name=args.wandb_name,
                tags=args.wandb_tags or wandb_cfg.get("tags", []),
                config=cfg,
                log_model="all",
                save_dir=paths.get("log_dir", "./logs"),
            )
            loggers.append(wandb_logger)
        except (ImportError, Exception) as e:
            print(f"[W&B] Failed: {e}, using TensorBoard only")
            use_wandb = False

    offline_enabled = offline_cfg.get("enabled", True)
    if offline_enabled:
        try:
            from .offline_debug import OfflineDebugCallback

            run_name = args.wandb_name or f"offline_{int(time.time())}"
            offline_dir = offline_cfg.get(
                "output_dir",
                os.path.join(paths.get("log_dir", "./logs"), "offline"),
            )
            callbacks.append(
                OfflineDebugCallback(
                    output_dir=offline_dir,
                    cfg=cfg,
                    run_name=run_name,
                    log_every_n_steps=offline_cfg.get("log_every_n_steps", 1),
                    hist_every_n_steps=offline_cfg.get("hist_every_n_steps", 100),
                    image_every_n_steps=offline_cfg.get("image_every_n_steps", 200),
                    max_images=offline_cfg.get("max_images", 4),
                    save_histograms=offline_cfg.get("save_histograms", True),
                    save_images=offline_cfg.get("save_images", True),
                    save_alerts=offline_cfg.get("save_alerts", True),
                )
            )
        except Exception as e:
            print(f"[OFFLINE] Failed: {e}")

    trainer = pl.Trainer(
        max_epochs=train_cfg.get("epochs", 20),
        accelerator="auto",
        devices=args.gpus,
        precision="16-mixed" if train_cfg.get("mixed_precision", True) else 32,
        gradient_clip_val=train_cfg.get("grad_clip_norm", 10.0),
        callbacks=callbacks,
        logger=loggers,
        log_every_n_steps=10,
        check_val_every_n_epoch=1,
    )

    trainer.fit(model, datamodule=dm, ckpt_path=args.resume)

    if use_wandb:
        try:
            import wandb

            wandb.finish()
        except Exception:
            pass


if __name__ == "__main__":
    main()
