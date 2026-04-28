"""Online (wandb) and offline (file) logging for dynGRU training."""

from __future__ import annotations

import json
import os
import subprocess
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import yaml

from Utility.PrettyPrint import Logger


def _get_git_hash() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=Path(__file__).resolve().parents[2],
        ).decode().strip()
    except Exception:
        return "unknown"


class DynTrainLogger:
    """Handles online (wandb) metrics and offline (file) artifact dumps.

    Usage:
        logger = DynTrainLogger(cfg, run_name="dyngru_demo_0427")
        # In training loop:
        logger.log_step(step_metrics, total_steps)
        if total_steps % visual_freq == 0:
            logger.log_visuals(visuals, total_steps)
        logger.finish()
    """

    def __init__(self, cfg, run_name: str, log_dir: str = "Results"):
        self.cfg = cfg
        self.run_name = run_name
        self.log_dir = Path(log_dir) / run_name
        self.debug_dir = self.log_dir / "debug"
        self.debug_dir.mkdir(parents=True, exist_ok=True)

        self.visual_freq = getattr(cfg, "visual_freq", 500)
        self.use_wandb = getattr(cfg, "wandb", False)
        self._git_hash = _get_git_hash()

        self._console_step = 0

        # --- Wandb ---
        self._wandb_run = None
        if self.use_wandb:
            import wandb
            self._wandb_run = wandb.init(
                project=getattr(cfg, "name", "dynGRU"),
                name=run_name,
                config=cfg if isinstance(cfg, dict) else vars(cfg) if hasattr(cfg, "__dict__") else {},
            )

        # Save config for reproducibility
        meta = {
            "run_name": run_name,
            "git_hash": self._git_hash,
            "start_time": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        with open(self.log_dir / "metadata.json", "w") as f:
            json.dump(meta, f, indent=2)
        if hasattr(cfg, "__dict__"):
            with open(self.log_dir / "config.yaml", "w") as f:
                yaml.dump(vars(cfg), f)

    # ------------------------------------------------------------------
    # Online: scalar metrics -> wandb + console
    # ------------------------------------------------------------------

    def log_step(self, metrics: dict[str, float], step: int) -> None:
        """Log scalars at each training step."""
        self._console_step = step
        if self._wandb_run is not None:
            self._wandb_run.log(metrics, step=step)

    def log_console(self, msg: str) -> None:
        Logger.write("info", f"[step {self._console_step}] {msg}")

    # ------------------------------------------------------------------
    # Offline: visual artifacts -> file
    # ------------------------------------------------------------------

    @torch.no_grad()
    def log_visuals(self, visuals: dict[str, Any], step: int) -> None:
        """Save debug artifacts and render diagnostic PNGs.

        Args:
            visuals: dict with keys:
                img1, img2          -- (3, H, W)  input images
                dyn_logits_final    -- (1, H/4, W/4)  final dyn head logits
                flow_est            -- (2, H, W)  estimated flow
                flow_rigid          -- (2, H, W)  rigid flow from GT
                M_pseudo            -- (1, H, W)  pseudo labels {-1,0,1}
                residual            -- (1, H, W)  ||f_est - f_rigid||
                tau                 -- (1, H, W)  adaptive threshold
                f_imu               -- (128,)     IMU global feature
                imu_tokens          -- (7, 128)   IMU semantic tokens
                alpha               -- scalar     adapter gate
                token_weights       -- (7,)       per-token gains
                frozen_grad_norm    -- scalar     gradient norm in frozen params
                trainable_grad_norm -- scalar     gradient norm in trainable params
            step: current training step.
        """
        step_dir = self.debug_dir / f"step_{step:06d}"
        step_dir.mkdir(parents=True, exist_ok=True)

        # Save raw tensors
        tensors = {k: v.cpu() for k, v in visuals.items() if isinstance(v, torch.Tensor)}
        torch.save(tensors, step_dir / "tensors.pt")

        # Save metadata
        meta = {
            "step": step,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "git_hash": self._git_hash,
        }
        with open(step_dir / "metadata.json", "w") as f:
            json.dump(meta, f, indent=2)

        # Render PNGs — each in its own try/except so one failure doesn't skip others
        for renderer, name in [
            (self._render_dyn_overlay, "dyn_overlay"),
            (self._render_residual_map, "residual_map"),
            (self._render_flow_comparison, "flow_comparison"),
        ]:
            try:
                renderer(visuals, step_dir)
            except Exception as e:
                self.log_console(f"WARNING: {name} render failed: {e}")
        if "f_imu" in visuals and "imu_tokens" in visuals:
            try:
                self._render_imu_features(visuals, step_dir)
            except Exception as e:
                self.log_console(f"WARNING: imu_features render failed: {e}")
        try:
            self._render_histograms(visuals, step_dir)
        except Exception as e:
            self.log_console(f"WARNING: histograms render failed: {e}")

        self.log_console(f"Saved debug artifacts to {step_dir}")

        # --- Also log rendered PNGs to wandb ---
        if self._wandb_run is not None:
            import wandb
            wandb_images = {}
            for png_file in sorted(step_dir.glob("*.png")):
                wandb_images[png_file.stem] = wandb.Image(str(png_file))
            if wandb_images:
                self._wandb_run.log(wandb_images, step=step)

    # ------------------------------------------------------------------
    # PNG renderers (private)
    # ------------------------------------------------------------------

    def _render_dyn_overlay(self, v: dict, step_dir: Path) -> None:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        img = v["img1"].cpu()
        dyn_logits = v["dyn_logits_final"].cpu()

        dyn_full = torch.nn.functional.interpolate(
            dyn_logits.unsqueeze(0), size=img.shape[-2:],
            mode="bilinear", align_corners=False,
        ).squeeze()
        c = torch.sigmoid(dyn_full)

        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        img_np = img.permute(1, 2, 0).clamp(0, 1).numpy()
        axes[0].imshow(img_np)
        axes[0].set_title("Input image")
        axes[0].axis("off")
        axes[1].imshow(img_np, alpha=0.5)
        heat = axes[1].imshow(c.numpy(), cmap="RdYlBu_r", vmin=0, vmax=1, alpha=0.6)
        plt.colorbar(heat, ax=axes[1], label="static confidence c")
        axes[1].set_title("DynGRU static confidence")
        axes[1].axis("off")
        fig.tight_layout()
        fig.savefig(step_dir / "dyn_overlay.png", dpi=100)
        plt.close(fig)

    def _render_residual_map(self, v: dict, step_dir: Path) -> None:
        """Residual heatmaps: ||f_est - f_rigid|| and ||f_gt - f_rigid|| with dyn head overlay."""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        r_est = v["residual"].cpu().squeeze()  # ||f_est - f_rigid||
        img = v["img1"].cpu()
        dyn_logits = v.get("dyn_logits_final")
        if dyn_logits is not None:
            c = dyn_logits.cpu().sigmoid()
            if c.ndim == 3:
                c = torch.nn.functional.interpolate(
                    c.unsqueeze(0), size=img.shape[-2:], mode="bilinear", align_corners=False
                ).squeeze(0).squeeze(0)

        has_gt = "flow_gt" in v
        n_cols = 3 if has_gt else 2
        fig, axes = plt.subplots(1, n_cols, figsize=(7 * n_cols, 5))
        if n_cols == 2:
            axes = [axes[0], axes[1], None]  # normalize indexing

        im1 = axes[0].imshow(r_est.numpy(), cmap="hot")
        plt.colorbar(im1, ax=axes[0])
        axes[0].set_title(f"||f_est - f_rigid|| (mean={r_est.mean():.2f} px)")
        axes[0].axis("off")

        if has_gt:
            flow_gt_t = v["flow_gt"].cpu().float()
            flow_rigid_t = v["flow_rigid"].cpu().float()
            if flow_gt_t.shape[-2:] != flow_rigid_t.shape[-2:]:
                flow_gt_t = torch.nn.functional.interpolate(
                    flow_gt_t.unsqueeze(0) if flow_gt_t.dim() == 3 else flow_gt_t.unsqueeze(0).unsqueeze(0),
                    size=flow_rigid_t.shape[-2:], mode="bilinear", align_corners=False,
                ).squeeze(0)
            r_true = (flow_gt_t - flow_rigid_t).norm(dim=0)
            im2 = axes[1].imshow(r_true.numpy(), cmap="hot")
            plt.colorbar(im2, ax=axes[1])
            axes[1].set_title(f"||f_gt - f_rigid|| true dynamic (mean={r_true.mean():.2f} px)")
            axes[1].axis("off")
            dyn_ax = axes[2]
        else:
            dyn_ax = axes[1]

        if dyn_logits is not None:
            img_np = img.permute(1, 2, 0).clamp(0, 1).numpy()
            dyn_ax.imshow(img_np, alpha=0.5)
            heat = dyn_ax.imshow(c.numpy(), cmap="RdYlBu_r", vmin=0, vmax=1, alpha=0.6)
            plt.colorbar(heat, ax=dyn_ax, label="static confidence c")
            dyn_ax.set_title(f"DynGRU c (mean={c.mean():.3f})")
        dyn_ax.axis("off")
        fig.tight_layout()
        fig.savefig(step_dir / "residual_map.png", dpi=100)
        plt.close(fig)

    def _render_flow_comparison(self, v: dict, step_dir: Path) -> None:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from torchvision.utils import flow_to_image

        flow_est = v["flow_est"].cpu().float().unsqueeze(0)
        flow_rigid_t = v["flow_rigid"].cpu().float()
        if flow_rigid_t.dim() == 2:
            flow_rigid_t = flow_rigid_t.unsqueeze(0).unsqueeze(0)  # (H,W) -> (1,2,H,W)
        elif flow_rigid_t.dim() == 3:
            flow_rigid_t = flow_rigid_t.unsqueeze(0)               # (2,H,W) -> (1,2,H,W)

        flow_est_img = flow_to_image(flow_est)[0].permute(1, 2, 0).numpy() / 255.0
        flow_rigid_img = flow_to_image(flow_rigid_t)[0].permute(1, 2, 0).numpy() / 255.0

        # GT flow (optional)
        has_gt = "flow_gt" in v
        if has_gt:
            flow_gt_t = v["flow_gt"].cpu().float()
            if flow_gt_t.dim() == 2:
                flow_gt_t = flow_gt_t.unsqueeze(0).unsqueeze(0)
            elif flow_gt_t.dim() == 3:
                flow_gt_t = flow_gt_t.unsqueeze(0)
            flow_gt_img = flow_to_image(flow_gt_t)[0].permute(1, 2, 0).numpy() / 255.0

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        axes[0, 0].imshow(flow_est_img)
        axes[0, 0].set_title("Estimated flow")
        axes[0, 0].axis("off")
        axes[0, 1].imshow(flow_rigid_img)
        axes[0, 1].set_title("Rigid flow (GT pose + depth)")
        axes[0, 1].axis("off")

        if has_gt:
            axes[1, 0].imshow(flow_gt_img)
            axes[1, 0].set_title("GT flow")
            axes[1, 0].axis("off")
            # True dynamic signal: where real flow deviates from rigid
            diff_dyn = (flow_gt_t.squeeze(0) - flow_rigid_t.squeeze(0)).norm(dim=0)
            im4 = axes[1, 1].imshow(diff_dyn.numpy(), cmap="hot")
            plt.colorbar(im4, ax=axes[1, 1])
            axes[1, 1].set_title(f"|f_gt - f_rigid| true dynamic (mean={diff_dyn.mean():.2f})")
            axes[1, 1].axis("off")
        else:
            diff = (v["flow_est"].cpu().float() - flow_rigid_t.squeeze(0)).norm(dim=0)
            im3 = axes[1, 0].imshow(diff.numpy(), cmap="hot")
            plt.colorbar(im3, ax=axes[1, 0])
            axes[1, 0].set_title(f"|f_est - f_rigid| (mean={diff.mean():.2f})")
            axes[1, 0].axis("off")
            axes[1, 1].axis("off")

        fig.tight_layout()
        fig.savefig(step_dir / "flow_comparison.png", dpi=100)
        plt.close(fig)

    def _render_imu_features(self, v: dict, step_dir: Path) -> None:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        tokens = v["imu_tokens"].cpu()
        f_imu = v["f_imu"].cpu()
        token_names = ["dR", "dv", "dp", "R^Tg", "bias", "Sig", "dt"]

        fig, axes = plt.subplots(1, 2, figsize=(16, 5))
        im = axes[0].imshow(tokens.numpy(), aspect="auto", cmap="RdBu_r")
        axes[0].set_yticks(range(7))
        axes[0].set_yticklabels(token_names)
        axes[0].set_xlabel("Feature dim")
        axes[0].set_title("IMU tokens (7x128)")
        plt.colorbar(im, ax=axes[0])
        axes[1].plot(f_imu.numpy())
        axes[1].set_xlabel("Feature dim")
        axes[1].set_title(f"f_imu (128-D, mean={f_imu.mean():.3f}, std={f_imu.std():.3f})")
        fig.tight_layout()
        fig.savefig(step_dir / "imu_features.png", dpi=100)
        plt.close(fig)

    def _render_histograms(self, v: dict, step_dir: Path) -> None:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        if "dyn_logits_final" in v:
            c = torch.sigmoid(v["dyn_logits_final"].cpu().flatten())
            axes[0, 0].hist(c.numpy(), bins=50, range=(0, 1), color="steelblue", edgecolor="white")
            axes[0, 0].set_xlabel("static confidence c")
            axes[0, 0].set_title(f"c histogram (mean={c.mean().item():.3f})")

        if "residual" in v:
            r_est = v["residual"].cpu().flatten()
            axes[0, 1].hist(r_est.clamp(0, 50).numpy(), bins=50, color="coral", edgecolor="white",
                           alpha=0.7, label="||f_est - f_rigid||")
            if "flow_gt" in v and "flow_rigid" in v:
                flow_gt_t = v["flow_gt"].cpu().float()
                flow_rigid_t = v["flow_rigid"].cpu().float()
                if flow_gt_t.shape[-2:] != flow_rigid_t.shape[-2:]:
                    flow_gt_t = torch.nn.functional.interpolate(
                        flow_gt_t.unsqueeze(0) if flow_gt_t.dim() == 3 else flow_gt_t,
                        size=flow_rigid_t.shape[-2:], mode="bilinear", align_corners=False,
                    ).squeeze(0)
                r_true = (flow_gt_t - flow_rigid_t).norm(dim=0).flatten()
                axes[0, 1].hist(r_true.clamp(0, 50).numpy(), bins=50, color="steelblue", edgecolor="white",
                               alpha=0.5, label="||f_gt - f_rigid||")
                axes[0, 1].legend(fontsize=7)
                axes[0, 1].set_title(f"Residual (est mean={r_est.mean().item():.1f}, true mean={r_true.mean().item():.1f})")
            else:
                axes[0, 1].set_title(f"||f_est - f_rigid|| (mean={r_est.mean().item():.2f})")
            axes[0, 1].set_xlabel("Residual [px]")

        if "token_weights" in v:
            w = v["token_weights"].cpu().numpy()
            names = ["dR", "dv", "dp", "R^Tg", "bias", "Sig", "dt"]
            axes[1, 0].bar(names, w, color="steelblue")
            axes[1, 0].axhline(y=1.0, color="gray", linestyle="--")
            alpha_val = v.get("alpha", 0)
            if isinstance(alpha_val, torch.Tensor):
                alpha_val = alpha_val.item() if alpha_val.numel() == 1 else float(alpha_val)
            axes[1, 0].set_title(f"Token weights (alpha={alpha_val:.4f})")

        if "trainable_grad_norm" in v:
            axes[1, 1].bar(
                ["trainable", "frozen"],
                [v.get("trainable_grad_norm", 0), v.get("frozen_grad_norm", 0)],
                color=["green", "red"],
            )
            axes[1, 1].set_yscale("log")
            axes[1, 1].set_title("Gradient norms per partition")
            axes[1, 1].set_ylabel("log scale")

        fig.tight_layout()
        fig.savefig(step_dir / "histograms.png", dpi=100)
        plt.close(fig)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def log_frozen_grad_check(self, model: nn.Module, trainable_pattern: str = "dyn_update") -> dict[str, float]:
        """Verify freeze policy: frozen params must have zero grad.

        Returns dict with trainable_grad_norm and frozen_grad_norm.
        """
        frozen_norm = 0.0
        trainable_norm = 0.0
        for name, param in model.named_parameters():
            if param.grad is None:
                continue
            gnorm = param.grad.norm().item()
            if trainable_pattern in name:
                trainable_norm += gnorm ** 2
            else:
                frozen_norm += gnorm ** 2
        return {
            "trainable_grad_norm": trainable_norm ** 0.5,
            "frozen_grad_norm": frozen_norm ** 0.5,
        }

    def finish(self) -> None:
        if self._wandb_run is not None:
            self._wandb_run.finish()
