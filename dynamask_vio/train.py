"""
DynaMask V2 Training Script — PyTorch Lightning + Weights & Biases.

Three-phase training with self-supervised losses:
  Phase 1  (epochs 1-20):   IMU head pretraining on EuRoC. L_imu only.
  Phase 2a (epochs 21-30):  Flow + photometric warmup. BA runs but detached.
                             L_photo + L_smooth + L_reg + L_imu
  Phase 2b (epochs 31-70):  Full self-supervised. BA gradient enabled.
                             ramp * L_pose + L_photo + L_reproj + L_reg + L_imu + L_smooth
  Phase 3  (epochs 71-80):  Fine-tune on dynamic scenes. Full loss, ramp=1.

Usage:
    python -m dynamask_vio.train --config configs/default.yaml
    python -m dynamask_vio.train --config configs/default.yaml --no-wandb
"""

import argparse
import os
import time

import yaml
import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning.callbacks import (
    ModelCheckpoint, EarlyStopping, LearningRateMonitor,
)
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader, ConcatDataset

from .models import DynaMaskVIO
from .models.differentiable_ba import (
    differentiable_ba, select_static_correspondences,
)
from .losses.imu_loss import imu_integration_loss, covariance_nll_loss
from .losses.self_supervised import (
    pose_consistency_loss, photometric_loss, reprojection_loss,
    mask_regularisation_loss, flow_smoothness_loss,
)
from .data import VIODEDataset, TartanAirDataset, EuRoCDataset

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
    """Lightning module for V2 self-supervised training."""

    def __init__(self, cfg: dict):
        super().__init__()
        self.cfg = cfg
        self.save_hyperparameters(cfg)

        self.model = DynaMaskVIO(cfg)

        # ── Loss config ──
        loss_cfg = cfg.get("loss", {})
        self.lambda_pose = loss_cfg.get("lambda_pose", 1.0)
        self.lambda_photo = loss_cfg.get("lambda_photo", 0.5)
        self.lambda_reproj = loss_cfg.get("lambda_reproj", 0.3)
        self.lambda_reg = loss_cfg.get("lambda_reg", 0.1)
        self.lambda_imu = loss_cfg.get("lambda_imu", 1.0)
        self.lambda_cov = loss_cfg.get("lambda_cov", 0.5)
        self.lambda_smooth = loss_cfg.get("lambda_smooth", 0.2)
        self.pose_w_rot = loss_cfg.get("pose_rotation_weight", 5.0)
        self.pose_w_dir = loss_cfg.get("pose_direction_weight", 1.0)
        self.pose_ramp_epochs = loss_cfg.get("pose_ramp_epochs", 5)
        self.w_rot = loss_cfg.get("imu_rotation_weight", 5.0)
        self.w_vel = loss_cfg.get("imu_velocity_weight", 1.0)
        self.w_pos = loss_cfg.get("imu_position_weight", 1.0)

        # ── BA config ──
        ba_cfg = cfg.get("ba", {})
        self.ba_n_iters = ba_cfg.get("n_iters", 3)
        self.ba_base_damping = ba_cfg.get("base_damping", 1e-4)
        self.ba_ep_initial = ba_cfg.get("ep_initial", 10.0)
        self.ba_ep_final = ba_cfg.get("ep_final", 1.0)
        self.ba_ep_decay_epochs = ba_cfg.get("ep_decay_epochs", 10)
        self.ba_outlier_threshold = ba_cfg.get("outlier_threshold", 100.0)
        self.ba_convergence_threshold = ba_cfg.get("convergence_threshold", 50.0)
        self.ba_num_correspondences = ba_cfg.get("num_correspondences", 256)
        self.ba_min_static_points = ba_cfg.get("min_static_points", 64)

        # ── Training phase boundaries ──
        train_cfg = cfg.get("training", {})
        self.phase1_end = train_cfg.get("phase1_epochs", 20)
        self.phase2a_end = self.phase1_end + train_cfg.get("phase2a_epochs", 10)
        self.phase2b_end = self.phase2a_end + train_cfg.get("phase2b_epochs", 40)
        # Phase 3 continues until max_epochs

    def _get_phase(self) -> str:
        """Returns current training phase as string."""
        epoch = self.current_epoch + 1
        if epoch <= self.phase1_end:
            return "1"
        elif epoch <= self.phase2a_end:
            return "2a"
        elif epoch <= self.phase2b_end:
            return "2b"
        return "3"

    def _get_pose_ramp(self) -> float:
        """L_pose weight ramp: 0 -> 1 over pose_ramp_epochs at Phase 2b start."""
        epoch = self.current_epoch + 1
        if epoch <= self.phase2a_end:
            return 0.0
        epochs_into_2b = epoch - self.phase2a_end
        return min(1.0, epochs_into_2b / max(self.pose_ramp_epochs, 1))

    def on_train_epoch_start(self):
        phase = self._get_phase()

        if phase == "1":
            # Phase 1: freeze everything except IMU encoder
            for name, param in self.model.named_parameters():
                param.requires_grad = name.startswith("imu_encoder")
        else:
            # Phase 2+: unfreeze all
            for param in self.model.parameters():
                param.requires_grad = True

    def forward(self, batch):
        return self.model(
            batch["img_prev"], batch["img_curr"],
            batch["imu_window"], batch["imu_mask"],
        )

    def _run_ba(self, outputs, batch, detach_ba: bool = False):
        """Run differentiable BA on model outputs.

        Args:
            outputs: model forward outputs
            batch: data batch (must contain "intrinsics" [B, 3, 3])
            detach_ba: if True, detach BA pose from graph (Phase 2a)

        Returns:
            ba_result dict or None if BA cannot run
        """
        if "intrinsics" not in batch:
            return None

        flow = outputs["flow"]  # [B, 2, H/8, W/8]
        mask_logits = outputs["mask_logits"]  # [B, 1, H/8, W/8]
        mask_prob = torch.sigmoid(mask_logits)

        # Select static correspondences
        correspondences, weights, valid_batch = select_static_correspondences(
            flow, mask_prob,
            num_points=self.ba_num_correspondences,
            min_points=self.ba_min_static_points,
        )

        if not valid_batch.any():
            return None

        K = batch["intrinsics"]  # [B, 3, 3]

        # Scale intrinsics to 1/8 resolution (flow is at 1/8)
        K_scaled = K.clone()
        K_scaled[:, 0, :] = K_scaled[:, 0, :] / 8.0
        K_scaled[:, 1, :] = K_scaled[:, 1, :] / 8.0
        K_scaled[:, 2, 2] = 1.0

        # IMU rotation as warm start
        R_init = outputs["delta_R"]  # [B, 3, 3]

        if detach_ba:
            correspondences = correspondences.detach()
            weights = weights.detach()
            R_init = R_init.detach()

        ba_result = differentiable_ba(
            correspondences=correspondences,
            weights=weights,
            K=K_scaled,
            R_init=R_init,
            epoch=self.current_epoch + 1,
            phase2_start=self.phase2a_end,
            n_iters=self.ba_n_iters,
            base_damping=self.ba_base_damping,
            ep_initial=self.ba_ep_initial,
            ep_final=self.ba_ep_final,
            ep_decay_epochs=self.ba_ep_decay_epochs,
            outlier_threshold=self.ba_outlier_threshold,
            convergence_threshold=self.ba_convergence_threshold,
        )

        if detach_ba:
            ba_result["R"] = ba_result["R"].detach()
            ba_result["t"] = ba_result["t"].detach()

        ba_result["weights"] = weights
        ba_result["valid_batch"] = valid_batch

        return ba_result

    def _cov_warmup(self) -> float:
        """Linearly ramp covariance NLL weight from 0 → 1 over the first
        ``cov_warmup_steps`` global steps.  At init, the preintegrated Σ is
        O(1e-8) and the err is O(1), so the NLL loss is dominated by the
        Mahalanobis term; even with err detached, the raw loss scale is huge
        and steals gradient budget from the L2 term.  A short warmup lets
        the L2 drive the err to a sane scale first, then NLL takes over
        calibrating Σ."""
        warmup_steps = self.cfg.get("loss", {}).get("cov_warmup_steps", 500)
        if warmup_steps <= 0:
            return 1.0
        return min(1.0, float(self.global_step) / float(warmup_steps))

    def _compute_loss(self, batch, outputs, phase: str) -> dict:
        losses = {}

        # ── L_imu: always active ──
        if "gt_R" in batch:
            l_imu = imu_integration_loss(
                outputs["delta_R"], outputs["delta_v"], outputs["delta_p"],
                batch["gt_R"], batch["gt_v"], batch["gt_p"],
                w_rot=self.w_rot, w_vel=self.w_vel, w_pos=self.w_pos,
            )
            losses["imu"] = self.lambda_imu * l_imu

            # ── L_cov: NLL for covariance calibration ──
            # NOT used in Phase 1.  Reason: Phase 1 trains the bias corrector
            # on EuRoC to match GT preintegration.  When L2 drives err → 0,
            # NLL = ½(err² Σ⁻¹ + log|Σ|) is minimized by Σ → 0, which sends
            # log|Σ| → -∞ and loss → -∞ (logdet collapse).  Phase 1 has no
            # dynamic objects and no strong covariance signal anyway; the
            # variance head will be calibrated in Phase 2+ when scene diversity
            # keeps err bounded away from zero.
            if phase != "1":
                l_cov = covariance_nll_loss(
                    outputs["delta_R"], outputs["delta_v"], outputs["delta_p"],
                    batch["gt_R"], batch["gt_v"], batch["gt_p"],
                    outputs["Sigma_preint"],
                )
                losses["cov"] = self.lambda_cov * self._cov_warmup() * l_cov

        if phase == "1":
            # Phase 1: IMU-only
            losses["total"] = sum(losses.values())
            return losses

        # ── Phase 2a, 2b, 3: vision losses ──

        flow = outputs["flow"]  # [B, 2, H/8, W/8]
        mask_logits = outputs["mask_logits"]  # [B, 1, H/8, W/8]
        mask_prob_8 = torch.sigmoid(mask_logits)  # at 1/8 res
        mask_full = outputs["dynamic_mask"]  # [B, 1, H, W]

        img_prev = batch["img_prev"]
        img_curr = batch["img_curr"]

        # Normalise images for photometric loss
        img_prev_norm = 2.0 * (img_prev / 255.0) - 0.5
        img_curr_norm = 2.0 * (img_curr / 255.0) - 0.5

        # L_photo
        l_photo = photometric_loss(img_prev_norm, img_curr_norm,
                                    flow, mask_prob_8)
        losses["photo"] = self.lambda_photo * l_photo

        # L_smooth
        l_smooth = flow_smoothness_loss(flow, img_curr_norm)
        losses["smooth"] = self.lambda_smooth * l_smooth

        # L_reg
        l_reg = mask_regularisation_loss(mask_full)
        losses["reg"] = self.lambda_reg * l_reg

        # ── BA losses (Phase 2a: detached, Phase 2b+: connected) ──
        detach_ba = (phase == "2a")
        ba_result = self._run_ba(outputs, batch, detach_ba=detach_ba)

        if ba_result is not None and phase in ("2b", "3"):
            ramp = self._get_pose_ramp()

            # L_pose: BA pose vs IMU preintegrated pose
            if ramp > 0:
                l_pose = pose_consistency_loss(
                    ba_result["R"], ba_result["t"],
                    outputs["delta_R"], outputs["delta_p"],
                    ba_result["converged"],
                    w_rot=self.pose_w_rot, w_dir=self.pose_w_dir,
                )
                losses["pose"] = ramp * self.lambda_pose * l_pose

            # L_reproj
            l_reproj = reprojection_loss(
                ba_result["reproj_error"],
                ba_result["weights"],
                ba_result["converged"],
            )
            losses["reproj"] = self.lambda_reproj * l_reproj

        # BA diagnostics (attached to losses dict, popped before sum)
        ba_diag = {}
        if ba_result is not None:
            ba_diag["convergence_rate"] = ba_result["converged"].mean().item()
            ba_diag["reproj_error_mean"] = ba_result["reproj_error"].pow(2).sum(-1).sqrt().mean().item()
            ba_diag["n_static_points"] = float(ba_result["weights"].shape[1])
            ba_diag["static_weight_mean"] = ba_result["weights"].mean().item()
        else:
            ba_diag["convergence_rate"] = 0.0
            ba_diag["ran"] = 0.0
        if ba_result is not None:
            ba_diag["ran"] = 1.0
        losses["_ba_diag"] = ba_diag

        losses["total"] = sum(v for k, v in losses.items()
                              if k != "_ba_diag" and isinstance(v, (int, float, torch.Tensor)))
        return losses

    def training_step(self, batch, batch_idx):
        phase = self._get_phase()
        outputs = self(batch)
        losses = self._compute_loss(batch, outputs, phase)

        total = losses["total"]
        # NaN/inf guard: non-finite loss poisons weights if backpropped and
        # crashes wandb's checkpoint artifact save (json.dumps with
        # allow_nan=False).  Replace the batch's total with a parameter-tethered
        # zero so backward is effectively a no-op while AMP scaler checks remain valid.
        is_bad = isinstance(total, torch.Tensor) and not torch.isfinite(total).all().item()
        if is_bad:
            bad_terms = {k: float(v.detach().cpu()) for k, v in losses.items()
                         if isinstance(v, torch.Tensor) and v.numel() == 1}
            print(f"[train] non-finite loss at batch {batch_idx} "
                  f"(phase={phase}); skipping. per-term={bad_terms}")
            # Keep the zero loss attached to an optimizer parameter so AMP's
            # GradScaler still records inf checks for this step.
            anchor_param = next(
                (p for p in self.model.parameters() if p.requires_grad),
                None,
            )
            if anchor_param is not None:
                total = anchor_param.sum() * 0.0
            else:
                total = torch.zeros(
                    (),
                    device=self.device,
                    dtype=total.dtype,
                    requires_grad=True,
                )
            safe_zero = torch.zeros((), device=self.device, dtype=total.dtype)
            # Sanitize individual loss scalars for logging
            losses = {k: (safe_zero
                          if isinstance(v, torch.Tensor)
                             and v.numel() == 1
                             and not torch.isfinite(v)
                          else v)
                      for k, v in losses.items()}
            losses["total"] = safe_zero

        for name, val in losses.items():
            if name == "_ba_diag":
                continue
            self.log(f"train/loss_{name}", val,
                     prog_bar=(name == "total"), sync_dist=True)
        self.log("train/phase", float({"1": 1, "2a": 2.0,
                                        "2b": 2.5, "3": 3}[phase]),
                 prog_bar=True)
        self.log("train/pose_ramp", self._get_pose_ramp())

        # BA diagnostics (logged when BA runs)
        if "_ba_diag" in losses:
            diag = losses.pop("_ba_diag")
            for k, v in diag.items():
                self.log(f"train/ba_{k}", v, sync_dist=True)

        return total

    def validation_step(self, batch, batch_idx):
        phase = self._get_phase()
        outputs = self(batch)
        losses = self._compute_loss(batch, outputs, phase)

        # Sanitize non-finite values — the ModelCheckpoint monitor is
        # val/loss_total, which gets serialised into wandb artifact metadata
        # (json.dumps with allow_nan=False) and crashes if NaN leaks through.
        for name, val in losses.items():
            if name == "_ba_diag":
                continue
            if isinstance(val, torch.Tensor) and val.numel() == 1 \
                    and not torch.isfinite(val):
                val = torch.zeros((), device=self.device)
                losses[name] = val
            self.log(f"val/loss_{name}", val, sync_dist=True)

        # Compute mask IoU for validation (using GT masks if available)
        if "gt_mask" in batch and phase != "1":
            pred_mask = (outputs["dynamic_mask"] > 0.5).float()
            gt_mask = batch["gt_mask"]
            intersection = (pred_mask * gt_mask).sum()
            union = ((pred_mask + gt_mask) > 0).float().sum()
            iou = intersection / (union + 1e-6)
            self.log("val/mask_iou", iou, prog_bar=True, sync_dist=True)

        return losses["total"]

    def configure_optimizers(self):
        train_cfg = self.cfg.get("training", {})
        lr = train_cfg.get("lr", 1e-4)
        wd = train_cfg.get("weight_decay", 1e-4)
        encoder_lr_mult = train_cfg.get("encoder_lr_multiplier", 0.5)
        imu_lr_mult = train_cfg.get("imu_lr_multiplier", 0.1)

        param_groups = self.model.get_parameter_groups(
            lr, encoder_lr_mult, imu_lr_mult)
        optimizer = torch.optim.AdamW(param_groups, lr=lr, weight_decay=wd,
                                       betas=(0.9, 0.999))

        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=[lr, lr * encoder_lr_mult, lr * imu_lr_mult],
            total_steps=train_cfg.get("total_steps", 240000),
            pct_start=0.01,
            cycle_momentum=False,
            anneal_strategy="linear",
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
            },
        }


class PhaseDataModule(pl.LightningDataModule):
    """Switches datasets according to training phase."""

    def __init__(self, cfg: dict):
        super().__init__()
        self.cfg = cfg
        self.batch_size = cfg.get("training", {}).get("batch_size", 8)
        self.num_workers = cfg.get("training", {}).get("num_workers", 8)
        data_cfg = cfg.get("data", {})

        self.enabled_datasets = {
            str(x).lower()
            for x in data_cfg.get("enabled_datasets",
                                   ["euroc", "viode", "tartanair"])
        }
        self.current_phase = "1"

    def setup(self, stage=None):
        paths = self.cfg.get("paths", {})

        # VIODE
        viode_cfg_path = os.path.join(os.path.dirname(__file__),
                                       "configs", "viode.yaml")
        viode_cfg = _load_config(viode_cfg_path) if os.path.exists(viode_cfg_path) else {}
        viode_merged = _merge_configs(viode_cfg, self.cfg)
        viode_splits = viode_merged.get("splits", {})

        viode_hdf5_root = paths.get("viode_hdf5_root",
                                     paths.get("viode_root", "./dataset/viode_hdf5"))
        self.viode_train = VIODEDataset(
            viode_hdf5_root,
            viode_splits.get("train", []),
            viode_merged, split="train",
        )
        self.viode_val = VIODEDataset(
            viode_hdf5_root,
            viode_splits.get("val", []),
            viode_merged, split="val",
        )

        self.euroc_train = None
        self.euroc_val = None
        self.tartanair_train = None

        if "euroc" in self.enabled_datasets:
            euroc_cfg_path = os.path.join(os.path.dirname(__file__),
                                           "configs", "euroc.yaml")
            euroc_cfg = _load_config(euroc_cfg_path) if os.path.exists(euroc_cfg_path) else {}
            euroc_merged = _merge_configs(euroc_cfg, self.cfg)
            euroc_splits = euroc_merged.get("splits", {})

            self.euroc_train = EuRoCDataset(
                paths.get("euroc_root", "./dataset/euroc"),
                euroc_splits.get("train", []),
                euroc_merged, split="train",
            )
            self.euroc_val = EuRoCDataset(
                paths.get("euroc_root", "./dataset/euroc"),
                euroc_splits.get("val", []),
                euroc_merged, split="val",
            )

        if "tartanair" in self.enabled_datasets:
            ta_cfg_path = os.path.join(os.path.dirname(__file__),
                                        "configs", "tartanair.yaml")
            ta_cfg = _load_config(ta_cfg_path) if os.path.exists(ta_cfg_path) else {}
            ta_merged = _merge_configs(ta_cfg, self.cfg)

            self.tartanair_train = TartanAirDataset(
                paths.get("tartanair_root", "./dataset/tartanair"),
                ta_merged.get("environments", []),
                ta_merged.get("difficulties", []),
                ta_merged, split="train",
            )

    def set_phase(self, phase: str):
        self.current_phase = phase

    def _pick_train_dataset(self):
        candidates = {
            "1": lambda: self.euroc_train,
            "2a": lambda: self._concat_nonempty(self.viode_train, self.tartanair_train),
            "2b": lambda: self._concat_nonempty(self.viode_train, self.tartanair_train),
            "3": lambda: self.viode_train,
        }
        for phase in [self.current_phase, "1", "2a", "3"]:
            ds = candidates[phase]()
            if ds is not None and len(ds) > 0:
                return ds
        raise RuntimeError("All datasets empty — check data paths.")

    @staticmethod
    def _concat_nonempty(*datasets):
        non_empty = [d for d in datasets if d is not None and len(d) > 0]
        if not non_empty:
            return None
        if len(non_empty) == 1:
            return non_empty[0]
        return ConcatDataset(non_empty)

    def train_dataloader(self):
        return DataLoader(
            self._pick_train_dataset(),
            batch_size=self.batch_size, shuffle=True,
            num_workers=self.num_workers, pin_memory=True,
            drop_last=True, persistent_workers=self.num_workers > 0,
        )

    def val_dataloader(self):
        if self.current_phase == "1" and self.euroc_val and len(self.euroc_val) > 0:
            dataset = self.euroc_val
        elif len(self.viode_val) > 0:
            dataset = self.viode_val
        elif self.euroc_val and len(self.euroc_val) > 0:
            dataset = self.euroc_val
        else:
            dataset = self._pick_train_dataset()

        return DataLoader(
            dataset, batch_size=self.batch_size, shuffle=False,
            num_workers=self.num_workers, pin_memory=True,
        )


class PhaseCallback(pl.Callback):
    """Switches data module phase at epoch boundaries."""

    def __init__(self, phase1_end: int, phase2a_end: int, phase2b_end: int):
        self.phase1_end = phase1_end
        self.phase2a_end = phase2a_end
        self.phase2b_end = phase2b_end

    def on_train_epoch_start(self, trainer, pl_module):
        epoch = trainer.current_epoch + 1
        if epoch <= self.phase1_end:
            phase = "1"
        elif epoch <= self.phase2a_end:
            phase = "2a"
        elif epoch <= self.phase2b_end:
            phase = "2b"
        else:
            phase = "3"

        dm = trainer.datamodule
        if hasattr(dm, "set_phase") and dm.current_phase != phase:
            dm.set_phase(phase)
            trainer.reset_train_dataloader()


def main():
    parser = argparse.ArgumentParser(description="DynaMask V2 Training")
    parser.add_argument("--config", type=str,
                        default="dynamask_vio/configs/default.yaml")
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

    phase1_end = train_cfg.get("phase1_epochs", 20)
    phase2a_end = phase1_end + train_cfg.get("phase2a_epochs", 10)
    phase2b_end = phase2a_end + train_cfg.get("phase2b_epochs", 40)

    dm = PhaseDataModule(cfg)
    model = DynaMaskLitModule(cfg)

    # Callbacks
    checkpoint_cb = ModelCheckpoint(
        dirpath=paths.get("checkpoint_dir", "./checkpoints"),
        filename="dynamask-v2-{epoch:02d}-{val/mask_iou:.3f}",
        monitor="val/loss_total",
        mode="min",
        save_top_k=3,
        save_last=True,
    )

    lr_monitor = LearningRateMonitor(logging_interval="step")
    phase_cb = PhaseCallback(phase1_end, phase2a_end, phase2b_end)

    callbacks = [checkpoint_cb, lr_monitor, phase_cb]

    # Logger
    loggers = []
    tb_logger = TensorBoardLogger(
        save_dir=paths.get("log_dir", "./logs"),
        name="dynamask_v2",
    )
    loggers.append(tb_logger)

    use_wandb = not args.no_wandb
    if use_wandb:
        try:
            from pytorch_lightning.loggers import WandbLogger
            wandb_project = os.environ.get(
                "WANDB_PROJECT", wandb_cfg.get("project", "dynamask-vio"))
            wandb_entity = os.environ.get(
                "WANDB_ENTITY", wandb_cfg.get("entity", None)) or None
            wandb_logger = WandbLogger(
                project=wandb_project, entity=wandb_entity,
                name=args.wandb_name,
                tags=args.wandb_tags or wandb_cfg.get("tags", []),
                config=cfg, log_model="all",
                save_dir=paths.get("log_dir", "./logs"),
            )
            loggers.append(wandb_logger)
        except (ImportError, Exception) as e:
            print(f"[W&B] Failed: {e}, using TensorBoard only")
            use_wandb = False

    # Offline debug logging
    offline_enabled = offline_cfg.get("enabled", True)
    if offline_enabled:
        try:
            from .offline_debug import OfflineDebugCallback, OfflineModelWatchCallback
            run_name = args.wandb_name or f"offline_{int(time.time())}"
            offline_dir = offline_cfg.get(
                "output_dir", os.path.join(paths.get("log_dir", "./logs"), "offline"))
            callbacks.append(OfflineDebugCallback(
                output_dir=offline_dir, cfg=cfg, run_name=run_name,
                log_every_n_steps=offline_cfg.get("log_every_n_steps", 1),
                hist_every_n_steps=offline_cfg.get("hist_every_n_steps", 100),
                image_every_n_steps=offline_cfg.get("image_every_n_steps", 200),
                max_images=offline_cfg.get("max_images", 4),
                save_histograms=offline_cfg.get("save_histograms", True),
                save_images=offline_cfg.get("save_images", True),
                save_alerts=offline_cfg.get("save_alerts", True),
            ))
        except Exception as e:
            print(f"[OFFLINE] Failed: {e}")

    trainer = pl.Trainer(
        max_epochs=train_cfg.get("epochs", 80),
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
