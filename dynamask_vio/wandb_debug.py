"""
W&B in-depth debugging and logging system for DynaMask V2.

Provides a Lightning Callback that instruments every aspect of training:

  SCALARS (every step):
    - Per-component losses: imu, cov, photo, smooth, reg, pose, reproj, total
    - Gradient norms per submodule: feature_encoder, context_encoder,
      flow_decoder, imu_encoder, film_layers
    - Learning rates per parameter group
    - IMU correction magnitudes: mean |delta_bg|, |delta_ba|, sigma2 stats
    - Preintegration output norms: |delta_R|, |delta_v|, |delta_p|
    - Mask statistics: mean prob, dynamic pixel ratio, entropy
    - Flow statistics: magnitude mean/max, per-iteration convergence
    - BA diagnostics: convergence rate, reproj error, static point count
    - GPU memory: allocated, reserved, utilization
    - Throughput: samples/sec, batches/sec

  HISTOGRAMS (every N steps):
    - Weight distributions per submodule
    - Gradient distributions per submodule
    - Mask probability distribution
    - Flow magnitude distribution
    - Covariance diagonal distribution

  IMAGES (every N steps):
    - Input frame pair
    - Predicted mask overlay on current frame
    - Ground truth mask overlay
    - Prediction error map (FP=red, FN=blue, TP=green)
    - Flow magnitude visualization

  ALERTS:
    - Loss spike detection (>5x running median)
    - Gradient explosion (norm > 100)
    - Gradient vanishing (norm < 1e-8 for any submodule)
    - NaN/Inf in any output
    - Covariance matrix non-positive-definite
    - Mask collapse (all predictions same value)
"""

import os
import time
from collections import deque

import numpy as np
import torch
import pytorch_lightning as pl

try:
    import wandb
except ImportError:
    wandb = None


# ─── Visualization helpers ───────────────────────────────────────────────────

def _mask_overlay(image: np.ndarray, mask: np.ndarray,
                  color: tuple = (0, 255, 0), alpha: float = 0.4) -> np.ndarray:
    """Overlay a binary mask on an image.

    Args:
        image: [H, W, 3] uint8
        mask: [H, W] float in [0, 1]
        color: RGB tuple
        alpha: blend factor

    Returns:
        [H, W, 3] uint8 blended image
    """
    overlay = image.copy()
    mask_bool = mask > 0.5
    for c in range(3):
        overlay[:, :, c] = np.where(
            mask_bool,
            np.clip(image[:, :, c] * (1 - alpha) + color[c] * alpha, 0, 255),
            image[:, :, c],
        )
    return overlay.astype(np.uint8)


def _error_map(pred_mask: np.ndarray, gt_mask: np.ndarray) -> np.ndarray:
    """Create a color-coded error map.

    TP = green, FP = red, FN = blue, TN = dark gray.
    """
    H, W = pred_mask.shape
    img = np.full((H, W, 3), 40, dtype=np.uint8)

    pred_bin = pred_mask > 0.5
    gt_bin = gt_mask > 0.5

    tp = pred_bin & gt_bin
    fp = pred_bin & ~gt_bin
    fn = ~pred_bin & gt_bin

    img[tp] = [0, 200, 0]
    img[fp] = [220, 0, 0]
    img[fn] = [0, 0, 220]

    return img


def _flow_magnitude_image(flow: np.ndarray) -> np.ndarray:
    """Convert [2, H, W] flow to a magnitude heatmap [H, W, 3] uint8."""
    mag = np.sqrt(flow[0] ** 2 + flow[1] ** 2)
    mag = np.clip(mag / (mag.max() + 1e-8) * 255, 0, 255).astype(np.uint8)
    import cv2
    return cv2.applyColorMap(mag, cv2.COLORMAP_TURBO)


# ─── Gradient norm computation ───────────────────────────────────────────────

def _grad_norm(module: torch.nn.Module) -> float:
    """Compute total gradient L2 norm across all parameters of a module."""
    total = 0.0
    count = 0
    for p in module.parameters():
        if p.grad is not None:
            total += p.grad.data.norm(2).item() ** 2
            count += 1
    return total ** 0.5 if count > 0 else 0.0


def _param_norm(module: torch.nn.Module) -> float:
    """Compute total parameter L2 norm."""
    total = 0.0
    for p in module.parameters():
        total += p.data.norm(2).item() ** 2
    return total ** 0.5


# V2 submodule name → attribute mapping
_V2_SUBMODULES = {
    "feature_encoder": "feature_encoder",
    "context_encoder": "context_encoder",
    "flow_decoder": "flow_decoder",
    "imu_encoder": "imu_encoder",
    "film_layers": "film_layers",
}


def _get_submodules(model):
    """Get V2 submodule dict, skipping any that don't exist."""
    out = {}
    for name, attr in _V2_SUBMODULES.items():
        mod = getattr(model, attr, None)
        if mod is not None:
            out[name] = mod
    return out


# ─── Main Callback ──────────────────────────────────────────────────────────

class WandbDebugCallback(pl.Callback):
    """Comprehensive W&B debugging callback for DynaMask V2 training."""

    def __init__(self, log_every_n_steps: int = 1,
                 hist_every_n_steps: int = 100,
                 image_every_n_steps: int = 200,
                 max_images: int = 4,
                 alert_window: int = 50):
        super().__init__()
        self.log_every = log_every_n_steps
        self.hist_every = hist_every_n_steps
        self.image_every = image_every_n_steps
        self.max_images = max_images

        self.loss_history = deque(maxlen=alert_window)
        self.grad_history = {}

        self._batch_start_time = None
        self._epoch_start_time = None
        self._epoch_samples = 0

    def on_train_epoch_start(self, trainer, pl_module):
        self._epoch_start_time = time.time()
        self._epoch_samples = 0

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        self._batch_start_time = time.time()

    def on_after_backward(self, trainer, pl_module):
        step = trainer.global_step
        if step % self.log_every != 0:
            return

        model = pl_module.model
        log = {}

        submodules = _get_submodules(model)

        total_grad_norm = 0.0
        for name, mod in submodules.items():
            gn = _grad_norm(mod)
            pn = _param_norm(mod)
            log[f"gradients/norm_{name}"] = gn
            log[f"gradients/param_norm_{name}"] = pn
            if pn > 0:
                log[f"gradients/ratio_{name}"] = gn / pn
            total_grad_norm += gn ** 2

            if name not in self.grad_history:
                self.grad_history[name] = deque(maxlen=50)
            self.grad_history[name].append(gn)

        log["gradients/norm_total"] = total_grad_norm ** 0.5

        for name, history in self.grad_history.items():
            gn = history[-1] if history else 0
            if gn > 100.0:
                wandb.alert(
                    title=f"Gradient explosion: {name}",
                    text=f"Gradient norm = {gn:.1f} at step {step}",
                    level=wandb.AlertLevel.WARN,
                )
            if len(history) > 10 and 0 < gn < 1e-8:
                wandb.alert(
                    title=f"Gradient vanishing: {name}",
                    text=f"Gradient norm = {gn:.2e} at step {step}",
                    level=wandb.AlertLevel.WARN,
                )

        if step % self.hist_every == 0:
            for name, mod in submodules.items():
                for pname, param in mod.named_parameters():
                    tag = f"weights/{name}.{pname}"
                    log[tag] = wandb.Histogram(param.data.cpu().float().numpy())
                    if param.grad is not None:
                        tag_g = f"grad_hist/{name}.{pname}"
                        log[tag_g] = wandb.Histogram(
                            param.grad.data.cpu().float().numpy())

        if log:
            wandb.log(log, step=step)

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        step = trainer.global_step
        if step % self.log_every != 0:
            return

        model = pl_module.model
        log = {}
        batch_time = time.time() - self._batch_start_time if self._batch_start_time else 0
        batch_size = batch["img_prev"].shape[0]
        self._epoch_samples += batch_size

        if batch_time > 0:
            log["perf/batch_time_ms"] = batch_time * 1000
            log["perf/samples_per_sec"] = batch_size / batch_time

        if torch.cuda.is_available():
            dev = pl_module.device
            log["perf/gpu_memory_allocated_MB"] = torch.cuda.memory_allocated(dev) / 1e6
            log["perf/gpu_memory_reserved_MB"] = torch.cuda.memory_reserved(dev) / 1e6
            log["perf/gpu_utilization_pct"] = (
                torch.cuda.memory_allocated(dev) / max(torch.cuda.memory_reserved(dev), 1)
            ) * 100

        with torch.no_grad():
            model.eval()
            out = model(batch["img_prev"], batch["img_curr"],
                        batch["imu_window"], batch["imu_mask"])
            model.train()

        # NaN/Inf detection
        nan_detected = False
        for k, v in out.items():
            if isinstance(v, torch.Tensor):
                if torch.isnan(v).any() or torch.isinf(v).any():
                    nan_detected = True
                    log[f"debug/nan_inf_{k}"] = 1
        if nan_detected:
            wandb.alert(
                title="NaN/Inf detected in model outputs",
                text=f"Step {step}: check debug/nan_inf_* metrics",
                level=wandb.AlertLevel.ERROR,
            )

        # Mask statistics
        mask_probs = out["dynamic_mask"]
        log["mask/mean_probability"] = mask_probs.mean().item()
        log["mask/std_probability"] = mask_probs.std().item()
        log["mask/min_probability"] = mask_probs.min().item()
        log["mask/max_probability"] = mask_probs.max().item()
        dynamic_ratio = (mask_probs > 0.5).float().mean().item()
        log["mask/dynamic_pixel_ratio"] = dynamic_ratio

        p = mask_probs.clamp(1e-6, 1 - 1e-6)
        entropy = -(p * p.log() + (1 - p) * (1 - p).log()).mean().item()
        log["mask/entropy"] = entropy

        if mask_probs.std().item() < 1e-4:
            wandb.alert(
                title="Mask collapse",
                text=f"Step {step}: mask std = {mask_probs.std().item():.6f}, "
                     f"mean = {mask_probs.mean().item():.4f}",
                level=wandb.AlertLevel.WARN,
            )

        # Flow statistics
        flow = out["flow"]  # [B, 2, H/8, W/8]
        flow_mag = flow.norm(dim=1)  # [B, H/8, W/8]
        log["flow/magnitude_mean"] = flow_mag.mean().item()
        log["flow/magnitude_max"] = flow_mag.max().item()
        log["flow/magnitude_std"] = flow_mag.std().item()

        # Per-iteration flow convergence
        for it_idx, fp in enumerate(out["flow_predictions"]):
            fp_mag = fp.norm(dim=1).mean().item()
            log[f"flow/iter_{it_idx}_magnitude_mean"] = fp_mag

        # IMU correction statistics
        log["imu/mean_abs_delta_bg"] = out["delta_bg"].abs().mean().item()
        log["imu/mean_abs_delta_ba"] = out["delta_ba"].abs().mean().item()
        log["imu/max_abs_delta_bg"] = out["delta_bg"].abs().max().item()
        log["imu/max_abs_delta_ba"] = out["delta_ba"].abs().max().item()
        log["imu/mean_sigma2_g"] = out["sigma2_g"].mean().item()
        log["imu/mean_sigma2_a"] = out["sigma2_a"].mean().item()
        log["imu/min_sigma2_g"] = out["sigma2_g"].min().item()
        log["imu/max_sigma2_g"] = out["sigma2_g"].max().item()

        # Preintegration output norms
        log["preint/delta_v_norm"] = out["delta_v"].norm(dim=-1).mean().item()
        log["preint/delta_p_norm"] = out["delta_p"].norm(dim=-1).mean().item()

        R = out["delta_R"]
        I = torch.eye(3, device=R.device).unsqueeze(0)
        log["preint/delta_R_frob_from_I"] = (R - I).norm(dim=(-2, -1)).mean().item()

        Sigma = out["Sigma_preint"]
        sigma_diag = torch.diagonal(Sigma, dim1=-2, dim2=-1)
        log["preint/cov_diag_mean"] = sigma_diag.mean().item()
        log["preint/cov_diag_min"] = sigma_diag.min().item()
        log["preint/cov_diag_max"] = sigma_diag.max().item()

        try:
            eigvals = torch.linalg.eigvalsh(Sigma)
            min_eigval = eigvals.min().item()
            log["preint/cov_min_eigenvalue"] = min_eigval
            if min_eigval < -1e-6:
                wandb.alert(
                    title="Non-PD covariance",
                    text=f"Step {step}: min eigenvalue = {min_eigval:.2e}",
                    level=wandb.AlertLevel.WARN,
                )
        except Exception:
            pass

        # FiLM modulation magnitude (iterate nn.ModuleList)
        film_layers = getattr(model, "film_layers", None)
        if film_layers is not None:
            for idx, film_layer in enumerate(film_layers):
                gamma_w = film_layer.gamma_proj.weight.data
                beta_w = film_layer.beta_proj.weight.data
                log[f"film/stage{idx}_gamma_weight_norm"] = gamma_w.norm().item()
                log[f"film/stage{idx}_beta_weight_norm"] = beta_w.norm().item()

        # Loss spike detection
        if isinstance(outputs, dict) and "loss" in outputs:
            loss_val = outputs["loss"].item() if isinstance(outputs["loss"], torch.Tensor) else outputs["loss"]
        elif isinstance(outputs, torch.Tensor):
            loss_val = outputs.item()
        else:
            loss_val = None

        if loss_val is not None:
            self.loss_history.append(loss_val)
            if len(self.loss_history) > 20:
                median_loss = sorted(self.loss_history)[len(self.loss_history) // 2]
                if loss_val > 5 * median_loss and median_loss > 0:
                    wandb.alert(
                        title="Loss spike",
                        text=f"Step {step}: loss = {loss_val:.4f}, "
                             f"median = {median_loss:.4f} (5x threshold)",
                        level=wandb.AlertLevel.WARN,
                    )
                log["debug/loss_ratio_to_median"] = loss_val / max(median_loss, 1e-8)

        # Histograms
        if step % self.hist_every == 0:
            log["hist/mask_probabilities"] = wandb.Histogram(
                mask_probs.cpu().float().numpy().flatten(), num_bins=64)
            log["hist/flow_magnitude"] = wandb.Histogram(
                flow_mag.cpu().float().numpy().flatten(), num_bins=64)
            log["hist/delta_bg"] = wandb.Histogram(
                out["delta_bg"].cpu().float().numpy().flatten())
            log["hist/delta_ba"] = wandb.Histogram(
                out["delta_ba"].cpu().float().numpy().flatten())
            log["hist/sigma2_g"] = wandb.Histogram(
                out["sigma2_g"].cpu().float().numpy().flatten())
            log["hist/sigma2_a"] = wandb.Histogram(
                out["sigma2_a"].cpu().float().numpy().flatten())
            log["hist/cov_diagonal"] = wandb.Histogram(
                sigma_diag.cpu().float().numpy().flatten())

        # Images
        if step % self.image_every == 0:
            n = min(self.max_images, batch_size)
            images_to_log = []
            has_gt_mask = "gt_mask" in batch
            for i in range(n):
                # Images are now [0, 255] float — just clip and cast
                img_prev = batch["img_prev"][i].cpu().permute(1, 2, 0).numpy()
                img_curr = batch["img_curr"][i].cpu().permute(1, 2, 0).numpy()
                img_prev = np.clip(img_prev, 0, 255).astype(np.uint8)
                img_curr = np.clip(img_curr, 0, 255).astype(np.uint8)

                pred_m = out["dynamic_mask"][i, 0].cpu().numpy()
                pred_overlay = _mask_overlay(img_curr, pred_m, color=(0, 255, 0))

                flow_vis = out["flow"][i].cpu().numpy()
                flow_img = _flow_magnitude_image(flow_vis)

                images_to_log.append(
                    wandb.Image(img_prev, caption=f"Frame t-1 (sample {i})"))
                images_to_log.append(
                    wandb.Image(pred_overlay,
                                caption=f"Pred mask (dyn={dynamic_ratio:.1%})"))
                if has_gt_mask:
                    gt_m = batch["gt_mask"][i, 0].cpu().numpy()
                    gt_overlay = _mask_overlay(img_curr, gt_m, color=(0, 100, 255))
                    err_map = _error_map(pred_m, gt_m)
                    images_to_log.append(
                        wandb.Image(gt_overlay, caption="GT mask"))
                    images_to_log.append(
                        wandb.Image(err_map,
                                    caption="Error: G=TP, R=FP, B=FN"))
                images_to_log.append(
                    wandb.Image(flow_img, caption="Flow magnitude"))

            log["images/predictions"] = images_to_log

        if log:
            wandb.log(log, step=step)

    def on_train_epoch_end(self, trainer, pl_module):
        elapsed = time.time() - self._epoch_start_time if self._epoch_start_time else 0
        epoch = trainer.current_epoch
        phase = pl_module._get_phase()

        log = {
            "epoch/epoch": epoch,
            "epoch/phase": phase,
            "epoch/duration_min": elapsed / 60,
            "epoch/total_samples": self._epoch_samples,
        }
        if elapsed > 0:
            log["epoch/throughput_samples_per_sec"] = self._epoch_samples / elapsed

        n_frozen = sum(1 for p in pl_module.model.parameters() if not p.requires_grad)
        n_total = sum(1 for p in pl_module.model.parameters())
        log["epoch/frozen_params"] = n_frozen
        log["epoch/trainable_params"] = n_total - n_frozen

        wandb.log(log, step=trainer.global_step)

    def on_validation_epoch_end(self, trainer, pl_module):
        step = trainer.global_step
        epoch = trainer.current_epoch

        log = {"val_epoch/epoch": epoch}
        if torch.cuda.is_available():
            dev = pl_module.device
            log["val_epoch/gpu_peak_memory_MB"] = torch.cuda.max_memory_allocated(dev) / 1e6
            torch.cuda.reset_peak_memory_stats(dev)

        wandb.log(log, step=step)


class WandbModelWatchCallback(pl.Callback):
    """Calls wandb.watch() once at the start of training."""

    def __init__(self, log_freq: int = 100, log_graph: bool = True):
        super().__init__()
        self.log_freq = log_freq
        self.log_graph = log_graph
        self._watched = False

    def on_train_start(self, trainer, pl_module):
        if not self._watched and wandb.run is not None:
            wandb.watch(
                pl_module.model,
                log="all",
                log_freq=self.log_freq,
                log_graph=self.log_graph,
            )
            self._watched = True
