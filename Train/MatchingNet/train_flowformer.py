import argparse
import os
import numpy as np
import time
import torch
import torch.distributed
import torch.nn as nn

from typing import get_args
from pathlib import Path
from torch.amp.grad_scaler import GradScaler
from torch.utils.data import ConcatDataset, DataLoader
from DataLoader import TrainDataset, DataFramePair, StereoFrame, CenterCropFrame, CastDataType, AddImageNoise, ScaleFrame
from Train.MatchingNet.loss import sequence_loss, sequence_metric, dyn_pseudo_label
from Utility.Config import load_config, namespace_to_cfgnode
from Utility.PrettyPrint import ColoredTqdm, Logger

from .utils import (
    T_TrainType, 
    AssertLiteralType,
    get_scheduler, get_optimizer
)


def write_wandb(header, objs, epoch_i):
    if isinstance(objs, dict):
        for k, v in objs.items():
            if isinstance(v, float):
                wandb.log({os.path.join(header, k): v}, epoch_i)
    else:
        wandb.log({header: objs}, step = epoch_i)


def merge_matrices(matrices):
    _matric = matrices[0]
    for i, m in enumerate(matrices):
        if i == 0:
            continue
        for k, v in m.items():
            _matric[k] += v

    for k, v in _matric.items():
        _matric[k] /= len(matrices)
    return _matric


def _imu_data_to_ticks(imu) -> tuple[list[dict], list[dict]]:
    """Convert batched IMUData (B=1 slice) to IMUContext.step() input format.

    Args:
        imu: ``IMUData`` with .acc (1, N, 3), .gyro (1, N, 3), .time_delta (1, N-1, 1).

    Returns:
        (corrected_imu, raw_imu) — lists of per-tick dicts with acc/gyro/dt keys.
    """
    acc = imu.acc.squeeze(0)                     # (N, 3)
    gyro = imu.gyro.squeeze(0)                   # (N, 3)
    dt = imu.time_delta.squeeze(0).squeeze(-1)   # (N-1,) in nanoseconds
    N = acc.size(0)
    corrected = [
        {"acc": acc[t], "gyro": gyro[t], "dt": dt[t].item() * 1e-9}
        for t in range(N - 1)
    ]
    raw = [{"acc": acc[t], "gyro": gyro[t]} for t in range(N)]
    return corrected, raw


def _imu_data_unbatch(imu, index: int):
    """Extract a single batch element from batched IMUData."""
    from DataLoader.Interface import IMUData as _IMUData
    return _IMUData(
        T_BS=imu.T_BS[index:index+1],
        time_ns=imu.time_ns[index:index+1],
        gravity=imu.gravity,
        acc=imu.acc[index:index+1],
        gyro=imu.gyro[index:index+1],
    )


def _attitude_unbatch(att, index: int):
    """Extract a single batch element from batched AttitudeData."""
    from DataLoader.Interface import AttitudeData as _AttitudeData
    return _AttitudeData(
        T_BS=att.T_BS[index:index+1],
        time_ns=att.time_ns[index:index+1],
        gravity=att.gravity,
        gt_pos=att.gt_pos[index:index+1],
        gt_vel=att.gt_vel[index:index+1],
        gt_rot=att.gt_rot[index:index+1],
        init_pos=att.init_pos[index:index+1],
        init_vel=att.init_vel[index:index+1],
        init_rot=att.init_rot[index:index+1],
    )


def train(modelcfg, cfg, loader: DataLoader[DataFramePair[StereoFrame]], eval_loader=None):
    train_mode: T_TrainType = modelcfg.training_mode
    AssertLiteralType(train_mode, T_TrainType)

    if train_mode == "dyn":
        from Module.Network.FlowFormerDyn import build_flowformer_dyn
        model = build_flowformer_dyn(modelcfg, torch.float32, torch.float32)
        if hasattr(modelcfg, "restore_ckpt") and modelcfg.restore_ckpt:
            ckpt = torch.load(modelcfg.restore_ckpt, map_location="cpu", weights_only=True)
            model.load_ddp_state_dict(ckpt)
        model = model.cuda()
    else:
        from Module.Network.FlowFormerCov import build_flowformer
        model = build_flowformer(modelcfg, torch.float32, torch.float32)
        if modelcfg.restore_ckpt:
            model.load_ddp_state_dict(torch.load(modelcfg.restore_ckpt, weights_only=True))

    model = nn.DataParallel(model)
    model.cuda()
    model.train()
    
    optimizer = get_optimizer(cfg.Model.optimizer.type)(
        model.parameters(),
        **vars(cfg.Model.optimizer.args)
    )
    scheduler = get_scheduler(cfg.Model.scheduler.type)(
        optimizer,
        **vars(cfg.Model.scheduler.args)
    )
    scaler = GradScaler(enabled=modelcfg.mixed_precision)
    model_ptr = model.module if isinstance(model, nn.DataParallel) else model
    match train_mode:
        case "flow":
            for param in model_ptr.memory_decoder.cov_update.parameters():
                param.requires_grad = False
        case "cov":
            for param in model_ptr.parameters():
                param.requires_grad = False
            for param in model_ptr.memory_decoder.cov_update.parameters():
                param.requires_grad = True
        case "dyn":
            # Freeze everything first
            for param in model_ptr.parameters():
                param.requires_grad = False
            # Unfreeze only the dyn branch (dyn_head is nested inside dyn_update)
            for param in model_ptr.memory_decoder.dyn_update.parameters():
                param.requires_grad = True
            # Assert cov_update is frozen (invariant check)
            assert all(
                not p.requires_grad
                for p in model_ptr.memory_decoder.cov_update.parameters()
            ), "cov_update must be frozen in dyn training mode"

    # --- Build IMUContext for dyn training (no AirIO checkpoint needed) ---
    if train_mode == "dyn":
        from types import SimpleNamespace as _SNS
        from Module.Network.IMUContext.imu_context import IMUContext
        _grav = getattr(modelcfg, "gravity", 9.81) if hasattr(modelcfg, "gravity") else 9.81
        imu_context = IMUContext(
            airio_cfg=_SNS(propcov=True),
            airio_ckpt=None,
            gravity=_grav,
        )
        imu_context.cuda()
    else:
        imu_context = None

    # --- Build logger ---
    from Train.DynNet.dyn_logger import DynTrainLogger
    _time = getattr(modelcfg, "time", time.strftime("%m-%d-%H-%M-%S", time.localtime()))
    run_name = f"{modelcfg.name}_{_time}" if hasattr(modelcfg, "name") else f"dyngru_{_time}"
    logger = DynTrainLogger(modelcfg, run_name)
    logger.log_console(f"Training mode: {train_mode}, steps: {modelcfg.num_steps}")

    if modelcfg.wandb:
        wandb.init(project=modelcfg.name, config=modelcfg)
        wandb.watch(model, log=None)
        
    total_steps = 0
    should_keep_training = True
    while should_keep_training:
        frameData: DataFramePair[StereoFrame]
        for frameData in ColoredTqdm(loader):
            assert frameData.cur.stereo.gt_flow   is not None
            assert frameData.cur.stereo.flow_mask is not None
            try:
                optimizer.zero_grad()
                img1, img2 = frameData.cur.stereo.imageL.cuda(), frameData.nxt.stereo.imageL.cuda()
                gt_flow = frameData.cur.stereo.gt_flow.cuda()
                flow_mask = frameData.cur.stereo.flow_mask.cuda()

                dyn = None
                dyn_pseudo = None
                if train_mode == "dyn":
                    B = img1.shape[0]

                    # Process IMU window through IMUContext for each batch element
                    f_imu_list, imu_tokens_list = [], []
                    for b in range(B):
                        imu_b = _imu_data_unbatch(frameData.cur.imu, b)
                        att_b = _attitude_unbatch(frameData.cur.gt_attitude, b)
                        pose_b = frameData.cur.gt_pose[b] if frameData.cur.gt_pose is not None else None
                        corrected, raw = _imu_data_to_ticks(imu_b)
                        imu_context.seed_from_gt(att_b, pose_b)
                        sample = imu_context.step(corrected, raw)
                        f_imu_list.append(sample.f_imu)
                        imu_tokens_list.append(sample.imu_tokens)

                    f_imu = torch.cat(f_imu_list, dim=0).cuda()
                    imu_tokens = torch.cat(imu_tokens_list, dim=0).cuda()
                    flow, cov, dyn = model(img1, img2, f_imu, imu_tokens)

                    # Pseudo-label from rigid-flow residual (§4.2).
                    gt_pose = getattr(frameData.cur, "gt_pose", None)
                    gt_depth = getattr(frameData.cur.stereo, "gt_depth", None)
                    fb_flow = getattr(frameData.cur.stereo, "gt_backward_flow", None)
                    K = frameData.cur.stereo.K.unsqueeze(0).expand(B, -1, -1).cuda()

                    if gt_pose is not None and gt_depth is not None and K is not None:
                        M_pseudo, residual = dyn_pseudo_label(
                            flow[-1], gt_pose.cuda(), gt_depth.cuda(), K,
                            fb_flow=fb_flow.cuda() if fb_flow is not None else None,
                        )
                    else:
                        # Placeholder: all-static, zero residual (dataloader not yet wired
                        # with GT depth + pose — TartanAir configs have gtDepth: false).
                        _, _, Hf, Wf = flow[-1].shape
                        M_pseudo = torch.ones(B, 1, Hf, Wf, device=img1.device, dtype=torch.long)
                        residual = torch.zeros(B, 1, Hf, Wf, device=img1.device)
                    dyn_pseudo = (M_pseudo, residual)

                    loss, _ = sequence_loss(cfg=modelcfg, preds=flow, gt=gt_flow, flow_mask=flow_mask,
                                            cov_preds=cov, dyn_preds=dyn, dyn_pseudo=dyn_pseudo)
                else:
                    flow, cov = model(img1, img2)
                    loss, _ = sequence_loss(cfg=modelcfg, preds=flow, gt=gt_flow, flow_mask=flow_mask, cov_preds=cov)
                
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), modelcfg.clip)
                scaler.step(optimizer)
                scheduler.step()
                lr = optimizer.param_groups[0]["lr"]
                scaler.update()
                if total_steps % int(modelcfg.log_freq) == 0:
                    Logger.write("info", "Iter: %d, Loss: %.4f" % (total_steps, loss.item()))
                    if modelcfg.wandb:
                        _, metric = sequence_metric(modelcfg, flow, cov, gt_flow, flow_mask, dyn_preds=dyn, dyn_pseudo=dyn_pseudo)
                        metrics = merge_matrices([metric])
                        metrics["lr"] = lr

                        # --- Dyn-specific metrics ---
                        if train_mode == "dyn" and dyn is not None:
                            c_final = dyn[-1].sigmoid().detach()
                            metrics["train/dyn_mean_c"] = c_final.mean().item()
                            metrics["train/dyn_static_frac"] = (c_final > 0.5).float().mean().item()
                            metrics["dyngru/alpha"] = model_ptr.memory_decoder.dyn_update.alpha.item()
                            metrics["dyngru/token_weights_mean"] = model_ptr.memory_decoder.dyn_update.imu_attn.token_weights.mean().item()
                            if f_imu is not None:
                                metrics["imu/f_imu_mean"] = f_imu.mean().item()
                                metrics["imu/f_imu_std"] = f_imu.std().item()
                            if dyn_pseudo is not None:
                                M_pseudo, _ = dyn_pseudo
                                valid = M_pseudo >= 0
                                if valid.sum() > 0:
                                    metrics["pseudo/static_pct"] = (M_pseudo == 1).float().mean().item()
                                    metrics["pseudo/dynamic_pct"] = (M_pseudo == 0).float().mean().item()
                                    metrics["pseudo/ignore_pct"] = (M_pseudo == -1).float().mean().item()
                            grad_info = logger.log_frozen_grad_check(model_ptr)
                            metrics["dyngru/grad_dyn_update"] = grad_info["trainable_grad_norm"]
                            metrics["dyngru/grad_frozen"] = grad_info["frozen_grad_norm"]

                        logger.log_step(metrics, total_steps)
                    
                total_steps += 1

                # --- Offline visual debug dump ---
                visual_freq = getattr(modelcfg, "visual_freq", 500)
                if train_mode == "dyn" and dyn is not None and dyn_pseudo is not None and total_steps % visual_freq == 0:
                    with torch.no_grad():
                        M_pseudo, residual = dyn_pseudo
                        visuals = {
                            "img1": img1[0].cpu().clamp(0, 1),
                            "img2": img2[0].cpu().clamp(0, 1),
                            "dyn_logits_final": dyn[-1][0].cpu(),
                            "flow_est": flow[-1][0].cpu(),
                            "flow_rigid": flow[-1][0].cpu(),
                            "M_pseudo": M_pseudo[0].cpu(),
                            "residual": residual[0].cpu(),
                            "tau": torch.ones_like(residual[0].cpu()) * 0.5,
                            "f_imu": f_imu[0].cpu(),
                            "imu_tokens": imu_tokens[0].cpu(),
                            "alpha": model_ptr.memory_decoder.dyn_update.alpha.detach().cpu(),
                            "token_weights": model_ptr.memory_decoder.dyn_update.imu_attn.token_weights.detach().cpu(),
                        }
                        grad_info = logger.log_frozen_grad_check(model_ptr)
                        visuals["trainable_grad_norm"] = grad_info["trainable_grad_norm"]
                        visuals["frozen_grad_norm"] = grad_info["frozen_grad_norm"]
                        logger.log_visuals(visuals, total_steps)
            except Exception as e:
                logger.log_console(f"CRASH at step {total_steps}: {e}")
                crash_dir = logger.debug_dir / f"crash_step_{total_steps:06d}"
                crash_dir.mkdir(parents=True, exist_ok=True)
                crash = {
                    "img1": img1.cpu(), "img2": img2.cpu(),
                    "gt_flow": gt_flow.cpu(), "flow_mask": flow_mask.cpu(),
                    "error": str(e),
                }
                if dyn is not None:
                    crash["dyn"] = [d.cpu() for d in dyn]
                if dyn_pseudo is not None:
                    crash["M_pseudo"] = dyn_pseudo[0].cpu()
                    crash["residual"] = dyn_pseudo[1].cpu()
                torch.save(crash, crash_dir / "crash_dump.pt")
                import traceback
                with open(crash_dir / "traceback.txt", "w") as f:
                    traceback.print_exc(file=f)
                logger.log_console(f"Crash dump saved to {crash_dir}")
                raise

            if total_steps > modelcfg.num_steps:
                should_keep_training = False
                break

            if modelcfg.autosave_freq and total_steps % modelcfg.autosave_freq == 0:
                PATH = "%s/%s/%d.pth" % (modelcfg.autosave_dir, modelcfg.name + modelcfg.time, total_steps)  
                Logger.write("info", f"Save model to {PATH}")
                if isinstance(model, nn.DataParallel):
                    # We don't want to have a layer of `module.` on all weights. Since we are definitely not
                    # using DDP during inference, I will just save the "real weights" of the model.
                    torch.save(model.module.state_dict(), PATH)
                else:
                    torch.save(model.state_dict(), PATH)
                
    logger.log_console(f"Training complete at step {total_steps}")
    logger.finish()

    PATH = "%s/%s/%d.pth" % (modelcfg.autosave_dir, modelcfg.name + modelcfg.time, total_steps)
    if isinstance(model, nn.DataParallel):
        torch.save(model.module.state_dict(), PATH)
    else:
        torch.save(model.state_dict(), PATH)
    
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="Config/Train/Demo.yaml")
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--autosave_dir", type=str, default="Model")
    parser.add_argument("--training_mode", type=str, choices=get_args(T_TrainType),
                        default="cov", help=f"Training mode: {get_args(T_TrainType)}")
    args = parser.parse_args()
    cfg, _ = load_config(Path(args.config))
    modlecfg = namespace_to_cfgnode(cfg.Model)
    modlecfg.update(vars(args))
    datacfg = cfg.Train
    modlecfg.time = time.strftime("%m-%d-%H-%M-%S", time.localtime())
    
    os.makedirs("%s/%s" % (args.autosave_dir, modlecfg.name + modlecfg.time), exist_ok=True)
    torch.manual_seed(modlecfg.seed)
    np.random.seed(modlecfg.seed)
    transforms = [CenterCropFrame(dict(width=640, height=480)),
                  CastDataType(dict(dtype=cfg.Model.datatype)),
                  AddImageNoise(dict(stdv=5.0)),
                  ScaleFrame(dict(scale_u=cfg.Model.image_scale, scale_v=cfg.Model.image_scale, interp='nearest'))]
    
    train_mode = modlecfg.training_mode
    if train_mode == "dyn":
        # DynGRU training needs IMU-bearing datasets (TartanAirv2, not _NoIMU)
        # and GT depth + GT pose for rigid-flow pseudo-labels.
        traindatasets = TrainDataset[StereoFrame].mp_instantiation(
            datacfg.data, 0, -1,
            lambda cfg: cfg.type in {"TartanAir", "TartanAirv2"}
        )
    else:
        traindatasets = TrainDataset[StereoFrame].mp_instantiation(
            datacfg.data, 0, -1,
            lambda cfg: cfg.type in {"TartanAir_NoIMU", "TartanAirv2_NoIMU"}
        )
    trainloader = DataLoader[DataFramePair[StereoFrame]](
        ConcatDataset([
            ds.transform_source(transforms)
            for ds in traindatasets
            if ds is not None
        ]),
        batch_size=modlecfg.batch_size,
        shuffle=True,
        collate_fn=DataFramePair.collate,
        drop_last=True,
        num_workers=4,
    )
    
    if args.wandb:
        try:
            import wandb
            wandb.init(project="FlowFormerCov", name = modlecfg.name,  config=modlecfg)
        except ImportError:
            Logger.write("warn", "Wandb is not installed, disabling it.")
            modlecfg.wandb = False

    train(modlecfg, cfg, trainloader)
