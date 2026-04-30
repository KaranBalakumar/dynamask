import argparse
import os
import numpy as np
import time
import torch
import torch.distributed
import torch.nn as nn
import torch.nn.functional as F

from typing import get_args
from pathlib import Path
from torch.amp.grad_scaler import GradScaler
from torch.utils.data import ChainDataset, DataLoader
from DataLoader import TrainDataset, DataFramePair, StereoFrame, CenterCropFrame, CastDataType, AddImageNoise, ScaleFrame
from Train.MatchingNet.loss import sequence_loss, sequence_metric, compute_rigid_flow_jacobian, compute_rigid_flow_jacobian_3d, dyn_residual_loss
import DataLoader.Dataset.VIODE as _viode_dl  # noqa: F401 — register VIODESequence
from Utility.Config import load_config, namespace_to_cfgnode
from Utility.PrettyPrint import ColoredTqdm, Logger

from .utils import (
    T_TrainType, 
    AssertLiteralType,
    get_scheduler, get_optimizer
)




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
    # Auto-compute num_steps from num_epochs if set
    if hasattr(modelcfg, "num_epochs") and modelcfg.num_epochs:
        frames_per_epoch = len(loader.dataset)
        steps_per_epoch = frames_per_epoch // modelcfg.batch_size
        modelcfg.num_steps = modelcfg.num_epochs * steps_per_epoch
        cfg.Model.scheduler.args.total_steps = modelcfg.num_steps

    train_mode: T_TrainType = modelcfg.training_mode
    AssertLiteralType(train_mode, T_TrainType)

    if train_mode in ("dyn", "dyn_selfsup"):
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
        case "dyn" | "dyn_selfsup":
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

    # --- Build IMUContext for dyn training ---
    if train_mode in ("dyn", "dyn_selfsup"):
        from types import SimpleNamespace as _SNS
        from Module.Network.IMUContext.imu_context import IMUContext
        _grav = getattr(modelcfg, "gravity", 9.81) if hasattr(modelcfg, "gravity") else 9.81
        _airio_ckpt = getattr(modelcfg, "airio_ckpt", None)
        imu_context = IMUContext(
            airio_cfg=_SNS(propcov=True),
            airio_ckpt=_airio_ckpt,
            gravity=_grav,
        )
        imu_context.cuda()
        imu_context.train()
    else:
        imu_context = None

    # Build optimizer with all trainable parameters (model + IMUContext)
    _trainable = list(model.parameters())
    if imu_context is not None:
        _trainable += list(imu_context.parameters())
    optimizer = get_optimizer(cfg.Model.optimizer.type)(
        _trainable,
        **vars(cfg.Model.optimizer.args)
    )
    scheduler = get_scheduler(cfg.Model.scheduler.type)(
        optimizer,
        **vars(cfg.Model.scheduler.args)
    )
    scaler = GradScaler(enabled=modelcfg.mixed_precision)

    # --- Build logger ---
    from Train.DynNet.dyn_logger import DynTrainLogger
    _time = getattr(modelcfg, "time", time.strftime("%m-%d-%H-%M-%S", time.localtime()))
    run_name = f"{modelcfg.name}_{_time}" if hasattr(modelcfg, "name") else f"dyngru_{_time}"
    logger = DynTrainLogger(modelcfg, run_name)
    logger.log_console(f"Training mode: {train_mode}, steps: {modelcfg.num_steps}")

    if modelcfg.wandb and logger._wandb_run is not None:
        logger._wandb_run.watch(model, log=None)

    # --- Resume from checkpoint if requested ---
    resume_ckpt = getattr(modelcfg, 'resume_ckpt', None)
    total_steps = 0
    if resume_ckpt:
        logger.log_console(f"Resuming from checkpoint: {resume_ckpt}")
        ckpt = torch.load(resume_ckpt, map_location='cpu', weights_only=False)
        if 'model_state_dict' in ckpt:
            model_ptr.load_state_dict(ckpt['model_state_dict'], strict=False)
            if 'optimizer_state_dict' in ckpt:
                optimizer.load_state_dict(ckpt['optimizer_state_dict'])
            if 'scheduler_state_dict' in ckpt:
                scheduler.load_state_dict(ckpt['scheduler_state_dict'])
            total_steps = ckpt.get('step', 0)
            if 'imu_context_state_dict' in ckpt and imu_context is not None:
                imu_context.load_state_dict(ckpt['imu_context_state_dict'])
        else:
            # Legacy checkpoint: just model state_dict
            model_ptr.load_state_dict(ckpt, strict=False)
        logger.log_console(f"Resumed at step {total_steps}")

    should_keep_training = True
    while should_keep_training:
        frameData: DataFramePair[StereoFrame]
        for frameData in ColoredTqdm(loader):
            if train_mode not in ("dyn_selfsup",):
                assert frameData.cur.stereo.gt_flow   is not None
                assert frameData.cur.stereo.flow_mask is not None
            try:
                optimizer.zero_grad()
                img1, img2 = frameData.cur.stereo.imageL.cuda(), frameData.nxt.stereo.imageL.cuda()
                gt_flow = frameData.cur.stereo.gt_flow
                gt_flow = gt_flow.cuda() if gt_flow is not None else None
                flow_mask = frameData.cur.stereo.flow_mask
                flow_mask = flow_mask.cuda() if flow_mask is not None else None

                dyn = None
                dyn_data = None
                if train_mode in ("dyn", "dyn_selfsup"):
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

                    # Continuous cov-gated target: c_target = exp(-r² / 2σ²)
                    gt_pose_cur = getattr(frameData.cur, "gt_pose", None)
                    gt_pose_nxt = getattr(frameData.nxt, "gt_pose", None)
                    gt_depth = getattr(frameData.cur.stereo, "gt_depth", None)
                    K = frameData.cur.stereo.K.cuda()

                    if gt_pose_cur is not None and gt_pose_nxt is not None:
                        import pypose as pp
                        # When GT depth is missing, compute stereo depth via FlowFormer
                        sigma_depth = None  # depth uncertainty from stereo cov
                        if gt_depth is None and hasattr(frameData.cur.stereo, 'imageR'):
                            with torch.no_grad():
                                flow_stereo, cov_stereo, _ = model(
                                    img1, frameData.cur.stereo.imageR.cuda(),
                                    torch.zeros(B, 128, device=img1.device),
                                    torch.zeros(B, 7, 128, device=img1.device),
                                )
                                disparity = flow_stereo[-1][:, :1].abs()
                                disp_cov = cov_stereo[-1][:, :1].abs()  # disparity std from cov head
                                baseline = frameData.cur.stereo.baseline.to(dtype=torch.float32, device=img1.device)
                                fx = K[:, 0, 0].unsqueeze(1).unsqueeze(2).unsqueeze(3)
                                gt_depth = (fx * baseline.view(-1,1,1,1)) / (disparity.clamp_min(0.1))
                                # Propagate disparity uncertainty → depth uncertainty
                                # depth = fx*baseline/disp → σ_depth ≈ (depth/disp) * σ_disp
                                sigma_depth = gt_depth.clamp_min(0.1) / disparity.clamp_min(0.1) * disp_cov.abs()
                                if gt_depth.shape[-2:] != flow[-1].shape[-2:]:
                                    gt_depth = torch.nn.functional.interpolate(
                                        gt_depth, size=flow[-1].shape[-2:], mode='bilinear', align_corners=False)
                                    sigma_depth = torch.nn.functional.interpolate(
                                        sigma_depth, size=flow[-1].shape[-2:], mode='bilinear', align_corners=False)

                        if gt_depth is not None:
                            NED_R_cam = torch.tensor([[0,0,1],[1,0,0],[0,1,0]], dtype=torch.float64)
                            T_cur_mat = pp.SE3(gt_pose_cur).matrix().to(dtype=torch.float64, device=img1.device)
                            T_nxt_mat = pp.SE3(gt_pose_nxt).matrix().to(dtype=torch.float64, device=img1.device)
                            T_rel = torch.linalg.inv(T_nxt_mat) @ T_cur_mat
                            R_rel, t_rel = T_rel[:, :3, :3], T_rel[:, :3, 3:4]
                            R_c = NED_R_cam.T.to(R_rel.device) @ R_rel @ NED_R_cam.to(R_rel.device)
                            t_c = NED_R_cam.T.to(R_rel.device) @ t_rel
                            B = R_c.size(0)
                            bottom_row = torch.tensor([0.,0.,0.,1.], device=R_rel.device, dtype=torch.float64).repeat(B, 1, 1)
                            T_rel_mat = torch.cat([torch.cat([R_c, t_c], dim=2), bottom_row], dim=1)
                            # 1D depth Jacobian + 2×3 3D Jacobian (for 3D loss modes)
                            f_rigid, J_d, _ = compute_rigid_flow_jacobian(
                                gt_depth.cuda(), K, T_rel_mat)
                            _, J_3d, _ = compute_rigid_flow_jacobian_3d(
                                gt_depth.cuda(), K, T_rel_mat)
                            # Use GT flow as residual target when available, fall back to estimated flow
                            if gt_flow is not None:
                                flow_target = gt_flow
                            else:
                                flow_target = flow[-1]
                            if flow_target.shape[-2:] != f_rigid.shape[-2:]:
                                flow_target = torch.nn.functional.interpolate(
                                    flow_target, size=f_rigid.shape[-2:], mode="bilinear", align_corners=False)
                            residual = (flow_target - f_rigid).norm(dim=1, keepdim=True)
                            r_vec = flow_target - f_rigid  # (B, 2, H, W) for loss
                        else:
                            _, _, Hf, Wf = flow[-1].shape
                            residual = torch.zeros(B, 1, Hf, Wf, device=img1.device)
                            f_rigid = torch.zeros(B, 2, Hf, Wf, device=img1.device)
                            r_vec = torch.zeros(B, 2, Hf, Wf, device=img1.device)
                            J_d = torch.zeros(B, 2, Hf, Wf, device=img1.device)
                            J_3d = torch.zeros(B, 2, 3, Hf, Wf, device=img1.device)
                    else:
                        _, _, Hf, Wf = flow[-1].shape
                        residual = torch.zeros(B, 1, Hf, Wf, device=img1.device)
                        f_rigid = torch.zeros(B, 2, Hf, Wf, device=img1.device)
                        r_vec = torch.zeros(B, 2, Hf, Wf, device=img1.device)
                        J_d = torch.zeros(B, 2, Hf, Wf, device=img1.device)
                        J_3d = torch.zeros(B, 2, 3, Hf, Wf, device=img1.device)
                    # dyn_data tuple extended: (residual, f_rigid, r_vec, J_d, sigma_depth, J_3d, depth_val, K)
                    dyn_data = (residual, f_rigid, r_vec, J_d, sigma_depth,
                                J_3d, gt_depth, K)

                    # GT flow for EPE metrics; dyn loss uses r_vec = flow_target - f_rigid (above)
                    _gt = gt_flow if gt_flow is not None else flow[-1].detach()
                    if flow_mask is not None:
                        _fm = flow_mask
                    else:
                        _fm = torch.ones(_gt.size(0), 1, *_gt.shape[-2:], device=_gt.device, dtype=torch.bool)
                    loss, _ = sequence_loss(cfg=modelcfg, preds=flow, gt=_gt, flow_mask=_fm,
                                            cov_preds=cov, dyn_preds=dyn, dyn_data=dyn_data)
                else:
                    flow, cov = model(img1, img2)
                    loss, _ = sequence_loss(cfg=modelcfg, preds=flow, gt=gt_flow, flow_mask=flow_mask, cov_preds=cov)
                
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), modelcfg.clip)
                if imu_context is not None:
                    torch.nn.utils.clip_grad_norm_(imu_context.parameters(), modelcfg.clip)
                scaler.step(optimizer)
                scheduler.step()
                lr = optimizer.param_groups[0]["lr"]
                scaler.update()
                if total_steps % int(modelcfg.log_freq) == 0:
                    Logger.write("info", "Iter: %d, Loss: %.4f" % (total_steps, loss.item()))
                    if modelcfg.wandb:
                        _, metric = sequence_metric(modelcfg, flow, cov, _gt, _fm, dyn_preds=dyn, dyn_data=dyn_data)
                        metrics = merge_matrices([metric])
                        metrics["lr"] = lr

                        # --- Dyn-specific metrics ---
                        if train_mode in ("dyn", "dyn_selfsup") and dyn is not None:
                            r_hat = F.softplus(dyn[-1]).detach()
                            metrics["train/dyn_r_mean"] = r_hat.mean().item()
                            metrics["train/dyn_r_max"]  = r_hat.max().item()
                            metrics["dyngru/alpha"] = model_ptr.memory_decoder.dyn_update.alpha.item()
                            metrics["dyngru/token_weights_mean"] = model_ptr.memory_decoder.dyn_update.imu_attn.token_weights.mean().item()
                            if f_imu is not None:
                                metrics["imu/f_imu_mean"] = f_imu.mean().item()
                                metrics["imu/f_imu_std"] = f_imu.std().item()
                            if dyn_data is not None:
                                metrics["pseudo/residual_mean"] = dyn_data[0].mean().item()
                            grad_info = logger.log_frozen_grad_check(model_ptr)
                            metrics["dyngru/grad_dyn_update"] = grad_info["trainable_grad_norm"]
                            metrics["dyngru/grad_frozen"] = grad_info["frozen_grad_norm"]

                        logger.log_step(metrics, total_steps)
                    
                total_steps += 1

                # --- Offline visual debug dump ---
                visual_freq = getattr(modelcfg, "visual_freq", 500)
                if train_mode in ("dyn", "dyn_selfsup") and dyn is not None and dyn_data is not None and total_steps % visual_freq == 0:
                    with torch.no_grad():
                        residual, f_rigid, r_vec, J_d, sigma_depth = dyn_data[:5]
                        flow_cov = cov[-1][0].cpu() if cov is not None else None
                        r_hat = F.softplus(dyn[-1][0].cpu())
                        visuals = {
                            "img1": img1[0].cpu().clamp(0, 1),
                            "img2": img2[0].cpu().clamp(0, 1),
                            "dyn_r_hat": r_hat,          # (1, H, W) predicted residual magnitude [px]
                            "dyn_target": residual[0].cpu(),  # (1, H, W) target residual [px]
                            "flow_est": flow[-1][0].cpu(),
                            "flow_rigid": f_rigid[0].cpu(),
                            "flow_gt": gt_flow[0].cpu() if gt_flow is not None else torch.zeros_like(flow[-1][0].cpu()),
                            "flow_cov": flow_cov,  # (2, H, W) — var_u, var_v from cov head
                            "f_imu": f_imu[0].cpu(),
                            "imu_tokens": imu_tokens[0].cpu(),
                            "alpha": model_ptr.memory_decoder.dyn_update.alpha.detach().cpu().item(),
                            "token_weights": model_ptr.memory_decoder.dyn_update.imu_attn.token_weights.detach().cpu(),
                        }
                        grad_info = logger.log_frozen_grad_check(model_ptr)
                        visuals["trainable_grad_norm"] = grad_info["trainable_grad_norm"]
                        visuals["frozen_grad_norm"] = grad_info["frozen_grad_norm"]
                        logger.log_visuals(visuals, total_steps, training_mode=train_mode)
            except Exception as e:
                logger.log_console(f"CRASH at step {total_steps}: {e}")
                crash_dir = logger.debug_dir / f"crash_step_{total_steps:06d}"
                crash_dir.mkdir(parents=True, exist_ok=True)
                crash = {
                    "img1": img1.cpu(), "img2": img2.cpu(),
                    "gt_flow": gt_flow.cpu() if gt_flow is not None else torch.zeros(1),
                    "flow_mask": flow_mask.cpu() if flow_mask is not None else torch.zeros(1),
                    "error": str(e),
                }
                if dyn is not None:
                    crash["dyn"] = [d.cpu() for d in dyn]
                if dyn_data is not None:
                    crash["residual"] = dyn_data[0].cpu()
                    crash["f_rigid"] = dyn_data[1].cpu()
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
                Logger.write("info", f"Save checkpoint to {PATH}")
                state = {
                    'model_state_dict': model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'step': total_steps,
                }
                if imu_context is not None:
                    state['imu_context_state_dict'] = imu_context.state_dict()
                torch.save(state, PATH)

    logger.log_console(f"Training complete at step {total_steps}")
    logger.finish()

    PATH = "%s/%s/%d.pth" % (modelcfg.autosave_dir, modelcfg.name + modelcfg.time, total_steps)
    state = {
        'model_state_dict': model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'step': total_steps,
    }
    if imu_context is not None:
        state['imu_context_state_dict'] = imu_context.state_dict()
    torch.save(state, PATH)


if __name__ == "__main__":
    import torch.backends.cuda
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="Config/Train/Demo.yaml")
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--autosave_dir", type=str, default="Model")
    parser.add_argument("--training_mode", type=str, choices=get_args(T_TrainType),
                        default="cov", help=f"Training mode: {get_args(T_TrainType)}")
    parser.add_argument("--resume_ckpt", type=str, default=None,
                        help="Path to checkpoint to resume training from")
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
    if train_mode in ("dyn", "dyn_selfsup"):
        # DynGRU training needs IMU-bearing datasets
        # and GT depth + GT pose for rigid-flow pseudo-labels.
        traindatasets = TrainDataset[StereoFrame].mp_instantiation(
            datacfg.data, 0, -1,
            lambda cfg: cfg.type in {"TartanAir", "TartanAirv2", "VIODE"}
        )
    else:
        traindatasets = TrainDataset[StereoFrame].mp_instantiation(
            datacfg.data, 0, -1,
            lambda cfg: cfg.type in {"TartanAir_NoIMU", "TartanAirv2_NoIMU"}
        )
    valid_datasets = [ds.transform_source(transforms) for ds in traindatasets if ds is not None]
    if len(valid_datasets) == 0:
        raise RuntimeError("No valid training datasets found")
    total_frames = sum(len(ds) for ds in valid_datasets)
    train_dataset = ChainDataset(valid_datasets)
    train_dataset.__len__ = lambda: total_frames  # ChainDataset has no __len__
    Logger.write("info", f"Training on {len(valid_datasets)} sequences ({total_frames} frame pairs total)")
    trainloader = DataLoader[DataFramePair[StereoFrame]](
        train_dataset,
        batch_size=modlecfg.batch_size,
        collate_fn=DataFramePair.collate,
        drop_last=True,
        num_workers=modlecfg.num_workers,
    )
    
    if args.wandb:
        try:
            import wandb
            wandb.init(project="FlowFormerCov", name = modlecfg.name,  config=modlecfg)
        except ImportError:
            Logger.write("warn", "Wandb is not installed, disabling it.")
            modlecfg.wandb = False

    train(modlecfg, cfg, trainloader)
