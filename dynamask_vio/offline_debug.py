"""Offline debug logging callbacks for DynaMask V2.

These callbacks mirror the W&B debug payloads to local files so the same
information is available without an online service.
"""

import json
import os
import time
from collections import deque
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np
import pytorch_lightning as pl
import torch

from .wandb_debug import (
    _error_map,
    _flow_magnitude_image,
    _grad_norm,
    _mask_overlay,
    _param_norm,
    _get_submodules,
)


class OfflineWriter:
    """Append-only writer for offline training debug artifacts."""

    def __init__(self, output_dir: str, run_name: Optional[str] = None):
        self.output_dir = output_dir
        self.run_name = run_name or f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.run_dir = os.path.join(self.output_dir, self.run_name)
        self.images_dir = os.path.join(self.run_dir, "images")
        os.makedirs(self.images_dir, exist_ok=True)

        self.scalars_path = os.path.join(self.run_dir, "scalars.jsonl")
        self.hist_path = os.path.join(self.run_dir, "histograms.jsonl")
        self.alerts_path = os.path.join(self.run_dir, "alerts.jsonl")
        self.watch_path = os.path.join(self.run_dir, "watch.jsonl")
        self.images_index_path = os.path.join(self.run_dir, "images.jsonl")

    def write_manifest(self, cfg: dict):
        manifest = {
            "created_at": datetime.now().isoformat(),
            "schema_version": "1.0",
            "run_name": self.run_name,
            "pid": os.getpid(),
            "config": cfg,
        }
        self._write_json(os.path.join(self.run_dir, "manifest.json"), manifest)

    def log_scalars(self, step: int, metrics: Dict[str, Any]):
        row = {
            "ts": time.time(),
            "step": int(step),
            "metrics": self._to_jsonable(metrics),
        }
        self._append_jsonl(self.scalars_path, row)

    def log_histograms(self, step: int, hist_data: Dict[str, Any], bins: int = 64):
        packed = {}
        for key, value in hist_data.items():
            arr = self._to_numpy(value)
            if arr.size == 0:
                continue

            arr = arr[np.isfinite(arr)]
            if arr.size == 0:
                continue

            vmin = float(arr.min())
            vmax = float(arr.max())

            if vmin == vmax:
                counts = np.array([int(arr.size)], dtype=np.int64)
                edges = np.array([vmin - 0.5, vmax + 0.5], dtype=np.float64)
            else:
                try:
                    counts, edges = np.histogram(arr, bins=bins, range=(vmin, vmax))
                except Exception:
                    counts = np.array([], dtype=np.int64)
                    edges = np.array([], dtype=np.float64)

            packed[key] = {
                "min": vmin,
                "max": vmax,
                "mean": float(arr.mean()),
                "std": float(arr.std()),
                "count": int(arr.size),
                "counts": counts.tolist(),
                "bin_edges": edges.tolist(),
            }

        if packed:
            self._append_jsonl(self.hist_path, {
                "ts": time.time(),
                "step": int(step),
                "histograms": packed,
            })

    def log_alert(self, step: int, title: str, text: str, level: str):
        self._append_jsonl(self.alerts_path, {
            "ts": time.time(),
            "step": int(step),
            "title": title,
            "text": text,
            "level": level,
        })

    def log_watch(self, step: int, payload: Dict[str, Any]):
        self._append_jsonl(self.watch_path, {
            "ts": time.time(),
            "step": int(step),
            "payload": self._to_jsonable(payload),
        })

    def log_images(self, step: int, images: List[Dict[str, Any]]):
        import cv2

        step_dir = os.path.join(self.images_dir, f"step_{int(step):08d}")
        os.makedirs(step_dir, exist_ok=True)

        index_rows = []
        for i, item in enumerate(images):
            caption = item.get("caption", "")
            image = item["image"]
            fname = f"{i:03d}.png"
            fpath = os.path.join(step_dir, fname)

            if image.ndim == 3 and image.shape[2] == 3:
                cv2.imwrite(fpath, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
            else:
                cv2.imwrite(fpath, image)

            index_rows.append({
                "ts": time.time(),
                "step": int(step),
                "caption": caption,
                "path": os.path.relpath(fpath, self.run_dir),
            })

        for row in index_rows:
            self._append_jsonl(self.images_index_path, row)

    def close(self):
        return

    @staticmethod
    def _write_json(path: str, payload: dict):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

    @staticmethod
    def _append_jsonl(path: str, payload: dict):
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(payload) + "\n")

    @staticmethod
    def _to_numpy(value: Any) -> np.ndarray:
        if isinstance(value, torch.Tensor):
            arr = value.detach().cpu().float().numpy()
        elif isinstance(value, np.ndarray):
            arr = value
        elif isinstance(value, (list, tuple)):
            arr = np.asarray(value)
        else:
            arr = np.asarray([value])
        return arr.reshape(-1)

    @staticmethod
    def _to_jsonable(value: Any) -> Any:
        if isinstance(value, dict):
            return {k: OfflineWriter._to_jsonable(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [OfflineWriter._to_jsonable(v) for v in value]
        if isinstance(value, torch.Tensor):
            if value.numel() == 1:
                return float(value.detach().cpu().item())
            return value.detach().cpu().float().tolist()
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, (np.floating, np.integer)):
            return value.item()
        if isinstance(value, (float, int, bool, str)) or value is None:
            return value
        return str(value)


class OfflineDebugCallback(pl.Callback):
    """Offline equivalent of WandbDebugCallback with parity metric names."""

    def __init__(
        self,
        output_dir: str,
        cfg: dict,
        run_name: Optional[str] = None,
        log_every_n_steps: int = 1,
        hist_every_n_steps: int = 100,
        image_every_n_steps: int = 200,
        max_images: int = 4,
        alert_window: int = 50,
        save_histograms: bool = True,
        save_images: bool = True,
        save_alerts: bool = True,
    ):
        super().__init__()
        self.writer = OfflineWriter(output_dir=output_dir, run_name=run_name)
        self.cfg = cfg

        self.log_every = log_every_n_steps
        self.hist_every = hist_every_n_steps
        self.image_every = image_every_n_steps
        self.max_images = max_images

        self.save_histograms = save_histograms
        self.save_images = save_images
        self.save_alerts = save_alerts

        self.loss_history = deque(maxlen=alert_window)
        self.grad_history = {}

        self._batch_start_time = None
        self._epoch_start_time = None
        self._epoch_samples = 0

    def on_train_start(self, trainer, pl_module):
        self.writer.write_manifest(self.cfg)

    def on_train_end(self, trainer, pl_module):
        self.writer.close()

    def on_train_epoch_start(self, trainer, pl_module):
        self._epoch_start_time = time.time()
        self._epoch_samples = 0

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        self._batch_start_time = time.time()

    def _alert(self, step: int, title: str, text: str, level: str = "WARN"):
        if self.save_alerts:
            self.writer.log_alert(step=step, title=title, text=text, level=level)

    def on_after_backward(self, trainer, pl_module):
        step = trainer.global_step
        if step % self.log_every != 0:
            return

        model = pl_module.model
        metrics = {}
        histograms = {}

        submodules = _get_submodules(model)

        total_grad_norm = 0.0
        for name, mod in submodules.items():
            gn = _grad_norm(mod)
            pn = _param_norm(mod)
            metrics[f"gradients/norm_{name}"] = gn
            metrics[f"gradients/param_norm_{name}"] = pn
            if pn > 0:
                metrics[f"gradients/ratio_{name}"] = gn / pn
            total_grad_norm += gn ** 2

            if name not in self.grad_history:
                self.grad_history[name] = deque(maxlen=50)
            self.grad_history[name].append(gn)

            if self.save_histograms and step % self.hist_every == 0:
                for pname, param in mod.named_parameters():
                    histograms[f"weights/{name}.{pname}"] = param.data
                    if param.grad is not None:
                        histograms[f"grad_hist/{name}.{pname}"] = param.grad.data

        metrics["gradients/norm_total"] = total_grad_norm ** 0.5

        for name, history in self.grad_history.items():
            gn = history[-1] if history else 0
            if gn > 100.0:
                self._alert(
                    step,
                    title=f"Gradient explosion: {name}",
                    text=f"Gradient norm = {gn:.1f} at step {step}",
                )
            if len(history) > 10 and 0 < gn < 1e-8:
                self._alert(
                    step,
                    title=f"Gradient vanishing: {name}",
                    text=f"Gradient norm = {gn:.2e} at step {step}",
                )

        self.writer.log_scalars(step, metrics)
        if histograms:
            self.writer.log_histograms(step, histograms)

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        step = trainer.global_step
        if step % self.log_every != 0:
            return

        model = pl_module.model
        metrics = {}
        histograms = {}
        batch_time = time.time() - self._batch_start_time if self._batch_start_time else 0
        batch_size = batch["img_prev"].shape[0]
        self._epoch_samples += batch_size

        if batch_time > 0:
            metrics["perf/batch_time_ms"] = batch_time * 1000
            metrics["perf/samples_per_sec"] = batch_size / batch_time

        if torch.cuda.is_available():
            dev = pl_module.device
            metrics["perf/gpu_memory_allocated_MB"] = torch.cuda.memory_allocated(dev) / 1e6
            metrics["perf/gpu_memory_reserved_MB"] = torch.cuda.memory_reserved(dev) / 1e6
            metrics["perf/gpu_utilization_pct"] = (
                torch.cuda.memory_allocated(dev) / max(torch.cuda.memory_reserved(dev), 1)
            ) * 100

        with torch.no_grad():
            model.eval()
            out = model(batch["img_prev"], batch["img_curr"],
                        batch["imu_window"], batch["imu_mask"])
            model.train()

        nan_detected = False
        for k, v in out.items():
            if isinstance(v, torch.Tensor):
                if torch.isnan(v).any() or torch.isinf(v).any():
                    nan_detected = True
                    metrics[f"debug/nan_inf_{k}"] = 1
        if nan_detected:
            self._alert(
                step,
                title="NaN/Inf detected in model outputs",
                text=f"Step {step}: check debug/nan_inf_* metrics",
                level="ERROR",
            )

        # Mask statistics
        mask_probs = out["dynamic_mask"]
        metrics["mask/mean_probability"] = mask_probs.mean().item()
        metrics["mask/std_probability"] = mask_probs.std().item()
        metrics["mask/min_probability"] = mask_probs.min().item()
        metrics["mask/max_probability"] = mask_probs.max().item()
        dynamic_ratio = (mask_probs > 0.5).float().mean().item()
        metrics["mask/dynamic_pixel_ratio"] = dynamic_ratio

        p = mask_probs.clamp(1e-6, 1 - 1e-6)
        metrics["mask/entropy"] = -(p * p.log() + (1 - p) * (1 - p).log()).mean().item()

        if mask_probs.std().item() < 1e-4:
            self._alert(
                step,
                title="Mask collapse",
                text=(
                    f"Step {step}: mask std = {mask_probs.std().item():.6f}, "
                    f"mean = {mask_probs.mean().item():.4f}"
                ),
            )

        # Flow statistics
        flow = out["flow"]
        flow_mag = flow.norm(dim=1)
        metrics["flow/magnitude_mean"] = flow_mag.mean().item()
        metrics["flow/magnitude_max"] = flow_mag.max().item()
        metrics["flow/magnitude_std"] = flow_mag.std().item()

        for it_idx, fp in enumerate(out["flow_predictions"]):
            fp_mag = fp.norm(dim=1).mean().item()
            metrics[f"flow/iter_{it_idx}_magnitude_mean"] = fp_mag

        # IMU
        metrics["imu/mean_abs_delta_bg"] = out["delta_bg"].abs().mean().item()
        metrics["imu/mean_abs_delta_ba"] = out["delta_ba"].abs().mean().item()
        metrics["imu/max_abs_delta_bg"] = out["delta_bg"].abs().max().item()
        metrics["imu/max_abs_delta_ba"] = out["delta_ba"].abs().max().item()
        metrics["imu/mean_sigma2_g"] = out["sigma2_g"].mean().item()
        metrics["imu/mean_sigma2_a"] = out["sigma2_a"].mean().item()
        metrics["imu/min_sigma2_g"] = out["sigma2_g"].min().item()
        metrics["imu/max_sigma2_g"] = out["sigma2_g"].max().item()

        # Preintegration
        metrics["preint/delta_v_norm"] = out["delta_v"].norm(dim=-1).mean().item()
        metrics["preint/delta_p_norm"] = out["delta_p"].norm(dim=-1).mean().item()

        R = out["delta_R"]
        I = torch.eye(3, device=R.device).unsqueeze(0)
        metrics["preint/delta_R_frob_from_I"] = (R - I).norm(dim=(-2, -1)).mean().item()

        Sigma = out["Sigma_preint"]
        sigma_diag = torch.diagonal(Sigma, dim1=-2, dim2=-1)
        metrics["preint/cov_diag_mean"] = sigma_diag.mean().item()
        metrics["preint/cov_diag_min"] = sigma_diag.min().item()
        metrics["preint/cov_diag_max"] = sigma_diag.max().item()

        try:
            eigvals = torch.linalg.eigvalsh(Sigma)
            min_eigval = eigvals.min().item()
            metrics["preint/cov_min_eigenvalue"] = min_eigval
            if min_eigval < -1e-6:
                self._alert(
                    step,
                    title="Non-PD covariance",
                    text=f"Step {step}: min eigenvalue = {min_eigval:.2e}",
                )
        except Exception:
            pass

        # FiLM modulation (nn.ModuleList in V2)
        film_layers = getattr(model, "film_layers", None)
        if film_layers is not None:
            for idx, film_layer in enumerate(film_layers):
                gamma_w = film_layer.gamma_proj.weight.data
                beta_w = film_layer.beta_proj.weight.data
                metrics[f"film/stage{idx}_gamma_weight_norm"] = gamma_w.norm().item()
                metrics[f"film/stage{idx}_beta_weight_norm"] = beta_w.norm().item()

        # Loss spike detection
        loss_val = None
        if isinstance(outputs, dict) and "loss" in outputs:
            loss_val = outputs["loss"].item() if isinstance(outputs["loss"], torch.Tensor) else outputs["loss"]
        elif isinstance(outputs, torch.Tensor):
            loss_val = outputs.item()

        if loss_val is not None:
            self.loss_history.append(loss_val)
            if len(self.loss_history) > 20:
                median_loss = sorted(self.loss_history)[len(self.loss_history) // 2]
                if loss_val > 5 * median_loss and median_loss > 0:
                    self._alert(
                        step,
                        title="Loss spike",
                        text=(
                            f"Step {step}: loss = {loss_val:.4f}, "
                            f"median = {median_loss:.4f} (5x threshold)"
                        ),
                    )
                metrics["debug/loss_ratio_to_median"] = loss_val / max(median_loss, 1e-8)

        # Histograms
        if self.save_histograms and step % self.hist_every == 0:
            histograms["hist/mask_probabilities"] = mask_probs
            histograms["hist/flow_magnitude"] = flow_mag
            histograms["hist/delta_bg"] = out["delta_bg"]
            histograms["hist/delta_ba"] = out["delta_ba"]
            histograms["hist/sigma2_g"] = out["sigma2_g"]
            histograms["hist/sigma2_a"] = out["sigma2_a"]
            histograms["hist/cov_diagonal"] = sigma_diag

        # Images
        if self.save_images and step % self.image_every == 0:
            n = min(self.max_images, batch_size)
            images_to_log = []
            for i in range(n):
                # Images are [0, 255] float — clip and cast
                img_prev = batch["img_prev"][i].cpu().permute(1, 2, 0).numpy()
                img_curr = batch["img_curr"][i].cpu().permute(1, 2, 0).numpy()
                img_prev = np.clip(img_prev, 0, 255).astype(np.uint8)
                img_curr = np.clip(img_curr, 0, 255).astype(np.uint8)

                pred_m = out["dynamic_mask"][i, 0].cpu().numpy()
                gt_m = batch["gt_mask"][i, 0].cpu().numpy()

                pred_overlay = _mask_overlay(img_curr, pred_m, color=(0, 255, 0))
                gt_overlay = _mask_overlay(img_curr, gt_m, color=(0, 100, 255))
                err_map = _error_map(pred_m, gt_m)

                flow_vis = out["flow"][i].cpu().numpy()
                flow_img = _flow_magnitude_image(flow_vis)

                images_to_log.append({"image": img_prev, "caption": f"Frame t-1 (sample {i})"})
                images_to_log.append({"image": pred_overlay, "caption": f"Pred mask (dyn={dynamic_ratio:.1%})"})
                images_to_log.append({"image": gt_overlay, "caption": "GT mask"})
                images_to_log.append({"image": err_map, "caption": "Error: G=TP, R=FP, B=FN"})
                images_to_log.append({"image": flow_img, "caption": "Flow magnitude"})

            self.writer.log_images(step, images_to_log)

        self.writer.log_scalars(step, metrics)
        if histograms:
            self.writer.log_histograms(step, histograms)

    def on_train_epoch_end(self, trainer, pl_module):
        elapsed = time.time() - self._epoch_start_time if self._epoch_start_time else 0
        epoch = trainer.current_epoch
        phase = pl_module._get_phase()

        metrics = {
            "epoch/epoch": epoch,
            "epoch/phase": phase,
            "epoch/duration_min": elapsed / 60,
            "epoch/total_samples": self._epoch_samples,
        }
        if elapsed > 0:
            metrics["epoch/throughput_samples_per_sec"] = self._epoch_samples / elapsed

        n_frozen = sum(1 for p in pl_module.model.parameters() if not p.requires_grad)
        n_total = sum(1 for p in pl_module.model.parameters())
        metrics["epoch/frozen_params"] = n_frozen
        metrics["epoch/trainable_params"] = n_total - n_frozen

        self.writer.log_scalars(trainer.global_step, metrics)

    def on_validation_epoch_end(self, trainer, pl_module):
        metrics = {"val_epoch/epoch": trainer.current_epoch}
        if torch.cuda.is_available():
            dev = pl_module.device
            metrics["val_epoch/gpu_peak_memory_MB"] = torch.cuda.max_memory_allocated(dev) / 1e6
            torch.cuda.reset_peak_memory_stats(dev)
        self.writer.log_scalars(trainer.global_step, metrics)


class OfflineModelWatchCallback(pl.Callback):
    """Offline equivalent of wandb.watch periodic parameter/gradient snapshots."""

    def __init__(self, output_dir: str, run_name: Optional[str] = None, log_freq: int = 100):
        super().__init__()
        self.log_freq = log_freq
        self._watched = False
        self.writer = OfflineWriter(output_dir=output_dir, run_name=run_name)

    def on_train_start(self, trainer, pl_module):
        self._watched = True

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if not self._watched:
            return
        step = trainer.global_step
        if step == 0 or step % self.log_freq != 0:
            return

        payload = {}
        for name, p in pl_module.model.named_parameters():
            payload[f"param_norm/{name}"] = p.data.norm().item()
            payload[f"param_mean/{name}"] = p.data.mean().item()
            payload[f"param_std/{name}"] = p.data.std(unbiased=False).item()
            if p.grad is not None:
                payload[f"grad_norm/{name}"] = p.grad.data.norm().item()
                payload[f"grad_mean/{name}"] = p.grad.data.mean().item()
                payload[f"grad_std/{name}"] = p.grad.data.std(unbiased=False).item()

        self.writer.log_watch(step=step, payload=payload)

    def on_train_end(self, trainer, pl_module):
        self.writer.close()
