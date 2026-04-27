import torch
import torch.nn.functional as F
from Utility.Extensions import OnCallCompiler

@OnCallCompiler()
def flow_loss(gamma: float, preds: torch.Tensor, gt: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    n_predictions = len(preds)
    
    flow_loss = torch.tensor(0.0, device=gt.device)
    for i in range(n_predictions):
        i_weight = gamma**(n_predictions - i - 1)
        i_loss = (preds[i] - gt).abs()
        flow_loss += i_weight * (mask * i_loss).nanmean()

    return flow_loss


@OnCallCompiler()
def cov_loss(gamma: float, preds: torch.Tensor, gt: torch.Tensor, cov_preds: list[torch.Tensor], 
             flow_mask: torch.Tensor | None = None, max_cov: float = 10., eps: float = 1e-7) -> tuple[torch.Tensor, torch.Tensor]:
    n_predictions = len(preds)
    cov_loss = torch.zeros_like(gt)
    error = torch.zeros_like(gt)
    exp_cov = torch.zeros_like(gt)
    
    error = None
    for i in range(n_predictions):
        exp_cov = cov_preds[i] + eps
        error = ((preds[i] - gt)**2).detach()
        i_weight = gamma**(n_predictions - i - 1)
        i_loss = ((error / exp_cov ) + torch.log(exp_cov)) * i_weight
        cov_loss += i_loss
    assert error is not None
    
    return cov_loss.mean(), error

@OnCallCompiler()
def final_cov_loss(preds: torch.Tensor, gt: torch.Tensor, cov_preds: list[torch.Tensor], 
                   flow_mask: torch.Tensor | None = None, max_cov: float = 10., eps: float = 1e-7) -> tuple[torch.Tensor, torch.Tensor]:
    cov = cov_preds[-1:]
    pred = preds[-1:]
    return cov_loss(1.0, pred, gt, cov, flow_mask, max_cov, eps)

@OnCallCompiler()
def depth_loss(gamma: float, preds: torch.Tensor, gt_depth: torch.Tensor, Ks: torch.Tensor, bls: torch.Tensor) -> torch.Tensor:
    n_predictions = len(preds)
    
    fxs = Ks[:, 0, 0]
    gt_disparity  = (fxs * bls).unsqueeze(-1).unsqueeze(-1) / gt_depth 
    est_disparity = preds[..., 0]
    
    depth_loss = torch.tensor(0.0, device=gt_disparity.device)
    for i in range(n_predictions):
        i_weight = gamma ** (n_predictions - i - 1)
        i_loss   = (est_disparity[i] + gt_disparity)
        depth_loss = i_weight * i_loss.mean()
    
    return depth_loss

# ---- Rigid-flow residual + continuous cov-gated target ----------------

def compute_rigid_flow_residual(
    flow_pred: torch.Tensor,        # (B, 2, H, W)  flow prediction
    pose_GT: torch.Tensor,          # (B, 4, 4)  SE(3) relative pose T_{t→t+1}
    depth: torch.Tensor,            # (B, 1, H, W)  depth map
    K: torch.Tensor,                # (B, 3, 3)  camera intrinsics
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute rigid-flow residual r = ‖f_est − f_rigid‖ and f_rigid.

    Returns:
        residual : (B, 1, H, W)  float  ‖f_est − f_rigid‖₂ per pixel
        f_rigid  : (B, 2, H, W)  float  rigid flow from pose + depth
    """
    B, _, H, W = flow_pred.shape
    device, dtype = flow_pred.device, flow_pred.dtype

    ys, xs = torch.meshgrid(
        torch.arange(H, device=device, dtype=dtype),
        torch.arange(W, device=device, dtype=dtype), indexing="ij",
    )
    ones = torch.ones(H, W, device=device, dtype=dtype)
    uv_homo = torch.stack([xs, ys, ones], dim=0).unsqueeze(0).expand(B, -1, -1, -1)

    K_inv = torch.linalg.inv(K.to(dtype))
    cam_pts = (K_inv @ uv_homo.flatten(2))
    X_cam_flat = cam_pts * depth.flatten(2)

    if pose_GT.shape[-1] == 4:
        T = pose_GT.to(dtype)
    else:
        raise ValueError(f"pose_GT must be (B,4,4), got {pose_GT.shape}")
    R_GT, t_GT = T[:, :3, :3], T[:, :3, 3:4]

    X_next_flat = R_GT @ X_cam_flat + t_GT
    X_next = X_next_flat.view(B, 3, H, W)
    Z_next = X_next[:, 2:3].clamp_min(1e-10)

    u_proj = K[:, 0, 0].view(-1, 1, 1, 1) * X_next[:, 0:1] / Z_next + K[:, 0, 2].view(-1, 1, 1, 1)
    v_proj = K[:, 1, 1].view(-1, 1, 1, 1) * X_next[:, 1:2] / Z_next + K[:, 1, 2].view(-1, 1, 1, 1)
    f_rigid = torch.cat([u_proj, v_proj], dim=1) - torch.stack([xs, ys], dim=0).unsqueeze(0).expand(B, -1, -1, -1)

    residual = (flow_pred - f_rigid).norm(dim=1, keepdim=True)
    return residual, f_rigid


# ---- Continuous cov-gated target + γ-weighted BCE loss -----------------

def dyn_loss_phase_a(
    dyn_predictions: list[torch.Tensor],   # K logit maps at H/4
    cov_predictions: list[torch.Tensor],   # K cov maps (read-only, frozen)
    residual: torch.Tensor,                # (B, 1, H, W)  ‖f_est − f_rigid‖
    gamma: float = 0.85,
) -> dict[str, torch.Tensor]:
    """γ-weighted BCE against continuous cov-gated target c_target = exp(−r² / 2σ²).

    The cov head (frozen) is calibrated to model flow estimation noise.  With GT
    pose + depth, that is the only noise source in the residual.  The continuous
    target eliminates tau_0, alpha, IGNORE bands, and focal-BCE — replaced by
    plain BCE against exp(−r² / 2σ²).
    """
    K = len(dyn_predictions)
    L_total = torch.tensor(0.0, device=residual.device)

    for i in range(K):
        i_weight = gamma ** (K - i - 1)
        logit = dyn_predictions[i]

        # Cov std per pixel (use L2 norm of 2-channel cov as proxy for σ_epe)
        cov_epe = cov_predictions[i].norm(dim=1, keepdim=True).clamp_min(1e-6)
        r = residual
        if r.shape[-2:] != cov_epe.shape[-2:]:
            r = F.interpolate(r, size=cov_epe.shape[-2:], mode="bilinear", align_corners=False)
        if logit.shape[-2:] != cov_epe.shape[-2:]:
            logit = F.interpolate(logit, size=cov_epe.shape[-2:], mode="bilinear", align_corners=False)

        c_target = torch.exp(-(r ** 2) / (2.0 * cov_epe ** 2 + 1e-8))
        L_total += i_weight * F.binary_cross_entropy_with_logits(logit, c_target)

    return {"L_total": L_total}


def sequence_loss(cfg, preds: torch.Tensor, gt: torch.Tensor, flow_mask: torch.Tensor | None, cov_preds: list[torch.Tensor] | None, dyn_preds: list[torch.Tensor] | None = None, dyn_data: tuple | None = None):
    loss = torch.tensor(0.0, device=gt.device)  # default, overwritten by match cases
    gt_mag = gt.norm(dim=1, keepdim=True)
    mask = gt_mag < cfg.max_flow
    if flow_mask is not None: mask &= flow_mask.bool()
    
    metrics = dict()
    
    match cfg.training_mode:
        case "flow":
            loss = flow_loss(cfg.gamma, preds, gt, mask)
        
        case "finalcov":
            assert cov_preds is not None
            if cfg.cov_mask:
                loss, error = final_cov_loss(preds, gt, cov_preds, mask)
            else:
                loss, error = final_cov_loss(preds, gt, cov_preds)
            metrics["error"] = error.mean().item()
            metrics["cov"] = cov_preds[-1].mean().item()
            metrics["cov_ratioe"] = (error/cov_preds[-1]).mean().item()
        
        case "cov":
            assert cov_preds is not None
            if cfg.cov_mask:
                loss, error = cov_loss(cfg.gamma, preds, gt, cov_preds, mask)
            else:
                loss, error = cov_loss(cfg.gamma, preds, gt, cov_preds)
            metrics["error"] = error.mean().item()
            metrics["cov"] = cov_preds[-1].mean().item()
            cov_mag = cov_preds[-1].sum(dim=-1).sqrt()
            epe = error.sum(dim=-1).sqrt()
            metrics["cov_ratioe"] = (cov_mag / epe).mean().item()

        case "dyn" | "dyn_selfsup":
            assert dyn_preds is not None and dyn_data is not None
            residual, f_rigid = dyn_data
            loss_dict = dyn_loss_phase_a(dyn_preds, cov_preds, residual, gamma=cfg.gamma)
            loss = loss_dict["L_total"]
            c_mean = dyn_preds[-1].sigmoid().mean().item()
            metrics["dyn_c_mean"] = c_mean

        case default:
            raise ValueError(f"Unavailable training mode {default}")      
    return loss, metrics


def sequence_metric(cfg, preds: torch.Tensor, cov_preds: list[torch.Tensor] | None, gt: torch.Tensor, flow_mask: torch.Tensor | None, dyn_preds: list[torch.Tensor] | None = None, dyn_data: tuple | None = None):
    loss = torch.tensor(0.0, device=gt.device)  # default, overwritten by match cases
    sqe = (preds[-1] - gt)**2
    epe = torch.sum(sqe, dim=1).sqrt()
    
    gt_mag = gt.norm(dim=1)
    mask = flow_mask.bool() & (gt_mag < cfg.max_flow) if flow_mask is not None else gt_mag < cfg.max_flow
    masked_epe = epe.view(-1)[mask.view(-1)]
    
    metrics = {
        'epe': masked_epe.mean().item(),
        '1px': (masked_epe < 1).float().mean().item(),
        '3px': (masked_epe < 3).float().mean().item(),
        '5px': (masked_epe < 5).float().mean().item(),
    }
    
    flow_gt_thresholds = [5, 10, 20]
    gt_mag = gt_mag.view(-1)[mask.view(-1)]
    for t in flow_gt_thresholds:
        e = masked_epe[gt_mag < t]
        metrics.update({f"{t}-th-5px": (e < 5).float().mean().item()})

    match cfg.training_mode:
        case "flow":
            loss = flow_loss(cfg.gamma, preds, gt, mask)
            
        case "cov":
            assert cov_preds is not None
            loss, _ = cov_loss(cfg.gamma, preds, gt, cov_preds)
            
            cov_mag = cov_preds[-1].sum(dim=1).sqrt()
            cov_ratio = (cov_mag / epe)
            metrics.update({"cov_loss": loss.mean().item()})
            metrics.update({"cov_mag": cov_mag.mean().item()})
            metrics.update({"cov_ratio": cov_ratio.nanmean().item()})

        case "dyn" | "dyn_selfsup":
            assert dyn_preds is not None and dyn_data is not None
            residual, f_rigid = dyn_data
            loss_dict = dyn_loss_phase_a(dyn_preds, cov_preds, residual, gamma=cfg.gamma)
            metrics.update({"dyn_loss": loss_dict["L_total"].item()})
            metrics.update({"dyn_c_mean": dyn_preds[-1].sigmoid().mean().item()})

        case default:
            raise ValueError(f"Unavailable training mode {default}")      
    
    return loss, metrics
