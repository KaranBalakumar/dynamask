from __future__ import annotations

import argparse
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F


def _cfg_get(config: SimpleNamespace | dict | None, key: str, default):
    if config is None:
        return default
    if isinstance(config, dict):
        return config.get(key, default)
    return getattr(config, key, default)


def verify_hard_freeze_preconditions(frontend) -> None:
    model = getattr(frontend, "model", None)
    if model is None:
        raise AssertionError("frontend must expose a frozen `model`.")
    if any(param.requires_grad for param in model.parameters()):
        raise AssertionError("Hard-freeze violated: frontend.model has trainable parameters.")

    imu_encoder = getattr(frontend, "imu_encoder", None)
    if imu_encoder is not None and any(param.requires_grad for param in imu_encoder.parameters()):
        raise AssertionError("Hard-freeze violated: frontend.imu_encoder has trainable parameters.")

    if model.training:
        raise AssertionError("Hard-freeze violated: frontend.model must stay in eval mode.")


def verify_hard_freeze_post_backward(frontend, head: torch.nn.Module) -> None:
    model = getattr(frontend, "model", None)
    if model is None:
        raise AssertionError("frontend must expose a frozen `model`.")
    for param in model.parameters():
        if param.grad is not None:
            raise AssertionError("Hard-freeze violated: frontend.model received gradients.")

    imu_encoder = getattr(frontend, "imu_encoder", None)
    if imu_encoder is not None:
        for param in imu_encoder.parameters():
            if param.grad is not None:
                raise AssertionError("Hard-freeze violated: frontend.imu_encoder received gradients.")

    head_grads = [param.grad for param in head.parameters() if param.requires_grad]
    if not head_grads or all(grad is None for grad in head_grads):
        raise AssertionError("Head update violated: dynamic head has no gradients after backward.")


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--log-every", type=int, default=20)
    return parser.parse_args()


def _build_optimizer(head: torch.nn.Module, train_cfg: SimpleNamespace) -> torch.optim.Optimizer:
    trainable = [param for param in head.parameters() if param.requires_grad]
    if not trainable:
        raise ValueError("Dynamic head has no trainable parameters.")

    lr = float(_cfg_get(train_cfg, "lr", 3e-4))
    weight_decay = float(_cfg_get(train_cfg, "weight_decay", 1e-4))
    optimizer_name = str(_cfg_get(train_cfg, "optimizer", "adamw")).lower()
    if optimizer_name == "adam":
        return torch.optim.Adam(trainable, lr=lr, weight_decay=weight_decay)
    if optimizer_name == "adamw":
        return torch.optim.AdamW(trainable, lr=lr, weight_decay=weight_decay)
    raise ValueError(f"Unsupported optimizer `{optimizer_name}` for dynamic-head training.")


def _build_scheduler(optimizer: torch.optim.Optimizer, train_cfg: SimpleNamespace):
    schedule = str(_cfg_get(train_cfg, "schedule", "cosine")).lower()
    total_steps = int(_cfg_get(train_cfg, "total_steps", 0))
    if schedule == "cosine" and total_steps > 0:
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)
    return None


def _run_window_step(
    *,
    frontend,
    head: torch.nn.Module,
    window_pairs: list,
    loss_cfg: SimpleNamespace,
    optimizer: torch.optim.Optimizer,
    grad_clip_norm: float,
) -> dict[str, float]:
    from Module.Frontend.Frontend import FlowFormerCovFrontend
    from Train.DynamicHead.loss import compose_factorized_loss

    frontend.reset_stream()
    h_prev = None
    term_logs: dict[str, list[float]] = {"L_static": [], "L_visible": [], "L_joint": [], "L_smooth": []}
    window_losses: list[torch.Tensor] = []

    for pair in window_pairs:
        if frontend.model.training:
            raise AssertionError("Hard-freeze violated: frontend.model switched to train mode.")

        with torch.no_grad():
            depth_out, match_out = FlowFormerCovFrontend.estimate_pair(frontend, pair.cur.stereo, pair.nxt.stereo)
            _, match_bwd = FlowFormerCovFrontend.estimate_pair(frontend, pair.nxt.stereo, pair.cur.stereo)

            if match_out.cov is None:
                raise RuntimeError("Training requires FlowFormerCov matcher covariance.")

            f_ctx = frontend._resolve_temporal_context(pair.cur.stereo.imageL.shape[0]).to(match_out.flow)
            flow_lo, cov_lo = frontend._prepare_flow_and_cov(match_out.flow, match_out.cov, f_ctx.shape[-2:])
            f_imu, imu_out = frontend._extract_imu_feature(pair.nxt.stereo, f_ctx)

            if imu_out is not None and ("delta_R" in imu_out) and ("delta_p" in imu_out):
                proxy_full = frontend._build_proxy_channels(
                    depth=depth_out.depth.to(match_out.flow),
                    flow_obs=match_out.flow,
                    K=pair.nxt.stereo.K.to(match_out.flow),
                    delta_R=imu_out["delta_R"].to(match_out.flow),
                    delta_p=imu_out["delta_p"].to(match_out.flow),
                )
            else:
                proxy_full = torch.zeros(
                    (match_out.flow.shape[0], 4, *match_out.flow.shape[-2:]),
                    dtype=match_out.flow.dtype,
                    device=match_out.flow.device,
                )

            proxy_lo = F.interpolate(proxy_full, size=f_ctx.shape[-2:], mode="bilinear", align_corners=False)
            scale_x = float(f_ctx.shape[-1]) / float(match_out.flow.shape[-1])
            scale_y = float(f_ctx.shape[-2]) / float(match_out.flow.shape[-2])
            proxy_lo[:, 0] *= scale_x
            proxy_lo[:, 1] *= scale_y

        head_out = head(
            f_ctx=f_ctx,
            flow=flow_lo,
            cov=cov_lo,
            f_imu=f_imu,
            proxy=proxy_lo,
            h_prev=h_prev,
            depth=depth_out.depth.to(match_out.flow),
        )
        h_prev = head_out["h_new"]

        h_full, w_full = match_out.flow.shape[-2:]
        p_static = F.interpolate(head_out["p_static"], size=(h_full, w_full), mode="bilinear", align_corners=False)
        p_visible = F.interpolate(head_out["p_visible"], size=(h_full, w_full), mode="bilinear", align_corners=False)

        loss_terms = compose_factorized_loss(
            p_static=p_static,
            p_visible=p_visible,
            flow=match_out.flow,
            flow_bwd=match_bwd.flow,
            depth=depth_out.depth.to(match_out.flow),
            image=pair.nxt.stereo.imageL.to(match_out.flow),
            K=pair.nxt.stereo.K.to(match_out.flow),
            R_gt=pair.gt_R_rel.to(match_out.flow),
            t_gt=pair.gt_t_rel.to(match_out.flow),
            cfg=loss_cfg,
        )
        window_losses.append(loss_terms["loss"])
        for key in term_logs:
            term_logs[key].append(float(loss_terms[key].detach().item()))

    loss = torch.stack(window_losses).mean()
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    verify_hard_freeze_post_backward(frontend, head)

    if grad_clip_norm > 0:
        torch.nn.utils.clip_grad_norm_(head.parameters(), grad_clip_norm)
    optimizer.step()

    return {
        "loss": float(loss.detach().item()),
        "L_static": float(np.mean(term_logs["L_static"])),
        "L_visible": float(np.mean(term_logs["L_visible"])),
        "L_joint": float(np.mean(term_logs["L_joint"])),
        "L_smooth": float(np.mean(term_logs["L_smooth"])),
    }


def main() -> None:
    from Utility.Config import load_config
    from Utility.PrettyPrint import Logger
    from DataLoader.Dataset.DynamicHeadTrain import DynamicHeadTrainDataset
    from Train.DynamicHead.loop import SequenceWindowSampler
    from Module.Frontend.Frontend import IFrontend, StaticConfidence_FlowFormerCovFrontend

    args = get_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    cfg, _ = load_config(Path(args.config))
    train_cfg = cfg.train
    data_cfg = cfg.data
    loss_cfg = cfg.loss

    dataset = DynamicHeadTrainDataset(data_cfg)
    sampler = SequenceWindowSampler(
        length=len(dataset),
        window_len=int(_cfg_get(train_cfg, "window_len", 4)),
        stride=int(_cfg_get(train_cfg, "stride", _cfg_get(train_cfg, "window_len", 4))),
    )
    windows = list(iter(sampler))
    if not windows:
        raise RuntimeError("No training windows were generated.")

    frontend_cfg = cfg.model.frontend
    frontend = IFrontend.instantiate(frontend_cfg.type, frontend_cfg.args)
    if not isinstance(frontend, StaticConfidence_FlowFormerCovFrontend):
        raise TypeError(
            f"Expected StaticConfidence_FlowFormerCovFrontend, got `{type(frontend).__name__}`."
        )
    verify_hard_freeze_preconditions(frontend)

    head = frontend.dynamic_head
    head.train()
    optimizer = _build_optimizer(head, train_cfg)
    scheduler = _build_scheduler(optimizer, train_cfg)

    grad_clip_norm = float(_cfg_get(train_cfg, "grad_clip_norm", 1.0))
    total_steps = int(_cfg_get(train_cfg, "total_steps", 1))
    log_every = max(int(args.log_every), 1)
    if total_steps <= 0:
        raise ValueError("train.total_steps must be positive.")

    start = time.time()
    step = 0
    while step < total_steps:
        for win_idx in np.random.permutation(len(windows)):
            window_ids = windows[int(win_idx)]
            window_pairs = [dataset[i] for i in window_ids]
            metrics = _run_window_step(
                frontend=frontend,
                head=head,
                window_pairs=window_pairs,
                loss_cfg=loss_cfg,
                optimizer=optimizer,
                grad_clip_norm=grad_clip_norm,
            )
            if scheduler is not None:
                scheduler.step()

            step += 1
            if (step % log_every) == 0 or step == 1 or step == total_steps:
                Logger.write(
                    "info",
                    (
                        f"[dynamic-head-train] step={step}/{total_steps} "
                        f"loss={metrics['loss']:.6f} "
                        f"L_static={metrics['L_static']:.6f} "
                        f"L_visible={metrics['L_visible']:.6f} "
                        f"L_joint={metrics['L_joint']:.6f} "
                        f"L_smooth={metrics['L_smooth']:.6f}"
                    ),
                )
            if step >= total_steps:
                break

    out_path = Path(str(_cfg_get(train_cfg, "checkpoint_out", "./Model/static_conf_head_last.pth")))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "head_state_dict": head.state_dict(),
            "T_calib": head.T_calib.detach().cpu(),
            "steps": total_steps,
        },
        out_path,
    )
    duration = time.time() - start
    Logger.write("info", f"Dynamic-head training finished in {duration:.1f}s. checkpoint={out_path}")


if __name__ == "__main__":
    main()
