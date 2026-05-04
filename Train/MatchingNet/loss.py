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


# ---- Analytic Jacobian ∂f_rigid/∂d for Mahalanobis loss ---------------

def compute_rigid_flow_jacobian(
    depth: torch.Tensor,          # (B, 1, H, W)
    K: torch.Tensor,              # (B, 3, 3)
    T_rel: torch.Tensor,          # (B, 4, 4)
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute f_rigid and analytic Jacobian J_d = ∂f_rigid/∂d per pixel.

    f_rigid = π(K · T_rel · X_cam) − uv   where X_cam = d · K⁻¹ · [u,v,1]ᵀ

    Returns:
        f_rigid : (B, 2, H, W)  rigid flow in pixels
        J_d     : (B, 2, H, W)  ∂f_rigid/∂d  per pixel
        X_next_z: (B, 1, H, W)  Z-coordinate of transformed points (for depth cov propagation)
    """
    B, _, H, W = depth.shape
    device, dtype = depth.device, depth.dtype
    K = K.to(dtype=dtype, device=device)

    fx, fy = K[:, 0, 0], K[:, 1, 1]
    cx, cy = K[:, 0, 2], K[:, 1, 2]

    # Unit ray r̂ = [(u−cx)/fx, (v−cy)/fy, 1]  per pixel
    ys, xs = torch.meshgrid(
        torch.arange(H, device=device, dtype=dtype),
        torch.arange(W, device=device, dtype=dtype), indexing="ij",
    )
    rx = (xs - cx.view(-1, 1, 1)) / fx.view(-1, 1, 1)
    ry = (ys - cy.view(-1, 1, 1)) / fy.view(-1, 1, 1)
    rz = torch.ones(B, H, W, device=device, dtype=dtype)

    T_rel = T_rel.to(dtype=dtype)
    R, t = T_rel[:, :3, :3], T_rel[:, :3, 3]

    # R·r̂ per component
    Rrx = R[:, 0, 0].view(B,1,1)*rx + R[:, 0, 1].view(B,1,1)*ry + R[:, 0, 2].view(B,1,1)*rz
    Rry = R[:, 1, 0].view(B,1,1)*rx + R[:, 1, 1].view(B,1,1)*ry + R[:, 1, 2].view(B,1,1)*rz
    Rrz = R[:, 2, 0].view(B,1,1)*rx + R[:, 2, 1].view(B,1,1)*ry + R[:, 2, 2].view(B,1,1)*rz

    d_sq = depth.squeeze(1)  # (B, H, W)
    Xx = d_sq * Rrx + t[:, 0].view(B, 1, 1)
    Xy = d_sq * Rry + t[:, 1].view(B, 1, 1)
    Xz = d_sq * Rrz + t[:, 2].view(B, 1, 1)
    Z_clamp = Xz.clamp_min(1e-10)
    Z_sq = Z_clamp * Z_clamp

    # Project to pixels
    u_proj = fx.view(-1,1,1) * Xx / Z_clamp + cx.view(-1,1,1)
    v_proj = fy.view(-1,1,1) * Xy / Z_clamp + cy.view(-1,1,1)
    f_rigid = torch.stack([u_proj - xs, v_proj - ys], dim=1)  # (B, 2, H, W)

    # Analytic Jacobian J_d = ∂f_rigid/∂d
    du_dd = fx.view(-1,1,1) * (Rrx * Z_clamp - Xx * Rrz) / Z_sq
    dv_dd = fy.view(-1,1,1) * (Rry * Z_clamp - Xy * Rrz) / Z_sq
    J_d = torch.stack([du_dd, dv_dd], dim=1)  # (B, 2, H, W)

    return f_rigid, J_d, Xz.unsqueeze(1)


# ---- 2×3 Jacobian ∂f_rigid/∂X_3D for 3D covariance projection ---------

def compute_rigid_flow_jacobian_3d(
    depth: torch.Tensor,          # (B, 1, H, W)
    K: torch.Tensor,              # (B, 3, 3)
    T_rel: torch.Tensor,          # (B, 4, 4)
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute f_rigid and 2×3 Jacobian J_{f→X} = ∂f_rigid/∂X_3D per pixel.

    X_3D = d · K⁻¹ · [u,v,1]ᵀ  is the 3D point in camera frame.
    f_rigid = π(K · T_rel · [X_3D; 1]) − uv

    J_{f→X} ∈ ℝ^{2×3}:
        ∂u'/∂X = fx/Z' · R₀ − fx·X'_x/Z'² · R₂
        ∂v'/∂X = fy/Z' · R₁ − fy·X'_y/Z'² · R₂

    Returns:
        f_rigid  : (B, 2, H, W)   rigid flow in pixels
        J_3d     : (B, 2, 3, H, W)  ∂f_rigid/∂X  per pixel
        X_next_z : (B, 1, H, W)   Z-coordinate of transformed points
    """
    B, _, H, W = depth.shape
    device, dtype = depth.device, depth.dtype
    K = K.to(dtype=dtype, device=device)
    T_rel = T_rel.to(dtype=dtype)

    fx, fy = K[:, 0, 0], K[:, 1, 1]
    cx, cy = K[:, 0, 2], K[:, 1, 2]

    # Unit ray r̂ = [(u−cx)/fx, (v−cy)/fy, 1]
    ys, xs = torch.meshgrid(
        torch.arange(H, device=device, dtype=dtype),
        torch.arange(W, device=device, dtype=dtype), indexing="ij",
    )
    rx = (xs - cx.view(-1, 1, 1)) / fx.view(-1, 1, 1)
    ry = (ys - cy.view(-1, 1, 1)) / fy.view(-1, 1, 1)
    rz = torch.ones(B, H, W, device=device, dtype=dtype)

    R, t = T_rel[:, :3, :3], T_rel[:, :3, 3]

    # R·r̂ per component
    Rrx = R[:, 0, 0].view(B,1,1)*rx + R[:, 0, 1].view(B,1,1)*ry + R[:, 0, 2].view(B,1,1)*rz
    Rry = R[:, 1, 0].view(B,1,1)*rx + R[:, 1, 1].view(B,1,1)*ry + R[:, 1, 2].view(B,1,1)*rz
    Rrz = R[:, 2, 0].view(B,1,1)*rx + R[:, 2, 1].view(B,1,1)*ry + R[:, 2, 2].view(B,1,1)*rz

    d_sq = depth.squeeze(1)
    Xx = d_sq * Rrx + t[:, 0].view(B, 1, 1)
    Xy = d_sq * Rry + t[:, 1].view(B, 1, 1)
    Xz = d_sq * Rrz + t[:, 2].view(B, 1, 1)
    Z_clamp = Xz.clamp_min(1e-10)
    Z_sq = Z_clamp * Z_clamp

    # Project to pixels
    u_proj = fx.view(-1,1,1) * Xx / Z_clamp + cx.view(-1,1,1)
    v_proj = fy.view(-1,1,1) * Xy / Z_clamp + cy.view(-1,1,1)
    f_rigid = torch.stack([u_proj - xs, v_proj - ys], dim=1)

    # ---- 2×3 Jacobian J_{f→X} = ∂f_rigid/∂X_3D ----
    # ∂u'/∂X = fx/Z' · R_row0 − fx·Xx'/Z'² · R_row2   (1×3 per pixel)
    # ∂v'/∂X = fy/Z' · R_row1 − fy·Xy'/Z'² · R_row2   (1×3 per pixel)
    J_u = (fx.view(-1,1,1) / Z_clamp).reshape(B, 1, 1, H, W) * R[:, 0:1].view(B, 1, 3, 1, 1) \
        - (fx.view(-1,1,1) * Xx / Z_sq).reshape(B, 1, 1, H, W) * R[:, 2:3].view(B, 1, 3, 1, 1)  # (B, 1, 3, H, W)
    J_v = (fy.view(-1,1,1) / Z_clamp).reshape(B, 1, 1, H, W) * R[:, 1:2].view(B, 1, 3, 1, 1) \
        - (fy.view(-1,1,1) * Xy / Z_sq).reshape(B, 1, 1, H, W) * R[:, 2:3].view(B, 1, 3, 1, 1)  # (B, 1, 3, H, W)
    J_3d = torch.cat([J_u, J_v], dim=1)  # (B, 2, 3, H, W)

    return f_rigid, J_3d, Xz.unsqueeze(1)


# ---- Dense 3D→2D covariance projection (MAC-VO §III.C, Appendix A) --------

def _build_sigma_3d_dense(
    sigma_uu: torch.Tensor,   # (B, 1, H, W)
    sigma_vv: torch.Tensor,   # (B, 1, H, W)
    sigma_dd: torch.Tensor,   # (B, 1, H, W)
    depth: torch.Tensor,       # (B, 1, H, W)
    K: torch.Tensor,           # (B, 3, 3)
) -> torch.Tensor:
    """Build dense per-pixel Σ_3D ∈ ℝ^{3×3} via Covariance_2to3 formulation.

    Returns Σ_3D as (B, 3, 3, H, W). Layout: rows are (z, x, y) in camera frame.
    Uses diagonal flow covariance (sigma_uv=0).
    """
    B, _, H, W = sigma_uu.shape
    device, dtype = sigma_uu.device, sigma_uu.dtype
    fx = K[:, 0, 0].view(B, 1, 1, 1)
    fy = K[:, 1, 1].view(B, 1, 1, 1)
    cx = K[:, 0, 2].view(B, 1, 1, 1)
    cy = K[:, 1, 2].view(B, 1, 1, 1)

    ys, xs = torch.meshgrid(
        torch.arange(H, device=device, dtype=dtype),
        torch.arange(W, device=device, dtype=dtype), indexing="ij",
    )
    u = xs.unsqueeze(0)  # (1, H, W)
    v = ys.unsqueeze(0)

    d = depth  # (B, 1, H, W)
    sd = sigma_dd  # (B, 1, H, W)
    su = sigma_uu
    sv = sigma_vv

    s_zz = sd
    s_xx = (su * sd + su * d.square() + (u - cx).square() * sd) / (fx * fx)
    s_yy = (sv * sd + sv * d.square() + (v - cy).square() * sd) / (fy * fy)
    s_xy = ((u - cx) * (v - cy) * sd) / (fx * fy)
    s_xz = ((u - cx) * sd) / fx
    s_yz = ((v - cy) * sd) / fy

    # Assemble 3×3 per pixel: rows (z, x, y)
    S = torch.zeros(B, 3, 3, H, W, device=device, dtype=dtype)
    S[:, 0, 0] = s_zz.squeeze(1);  S[:, 0, 1] = s_xz.squeeze(1);  S[:, 0, 2] = s_yz.squeeze(1)
    S[:, 1, 0] = s_xz.squeeze(1);  S[:, 1, 1] = s_xx.squeeze(1);  S[:, 1, 2] = s_xy.squeeze(1)
    S[:, 2, 0] = s_yz.squeeze(1);  S[:, 2, 1] = s_xy.squeeze(1);  S[:, 2, 2] = s_yy.squeeze(1)
    return S


def _project_3d_cov_to_2x2(
    var_u: torch.Tensor,      # (B, 1, H, W)
    var_v: torch.Tensor,      # (B, 1, H, W)
    sigma_dd: torch.Tensor,   # (B, 1, H, W)  depth variance (σ_d², not std)
    depth: torch.Tensor,       # (B, 1, H, W)
    J_3d: torch.Tensor,        # (B, 2, 3, H, W)
    K: torch.Tensor,           # (B, 3, 3)
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Project MAC-VO 3D point covariance to 2D flow covariance.

    Σ_3D = Covariance_2to3(var_u, var_v, sigma_dd, depth, K)
    Σ_2D = J_{f→X} · Σ_3D · J_{f→X}ᵀ

    Returns (var_u_3d, var_v_3d, var_uv_3d) -- the FULL 2x2 covariance from 3D projection Σ.
    """
    B, _, H, W = var_u.shape
    S_3d = _build_sigma_3d_dense(var_u, var_v, sigma_dd, depth, K)
    N = B * H * W
    J_flat = J_3d.permute(0, 3, 4, 1, 2).reshape(N, 2, 3)
    S_flat = S_3d.permute(0, 3, 4, 1, 2).reshape(N, 3, 3)
    JS = torch.bmm(J_flat, S_flat)
    S_2d_flat = torch.bmm(JS, J_flat.transpose(-2, -1))
    S_2d = S_2d_flat.reshape(B, H, W, 2, 2).permute(0, 3, 4, 1, 2)

    du  = S_2d[:, 0, 0].unsqueeze(1)  # Σ_2D[0,0] — additional u-variance
    dv  = S_2d[:, 1, 1].unsqueeze(1)  # Σ_2D[1,1] — additional v-variance
    duv = S_2d[:, 0, 1].unsqueeze(1)  # Σ_2D[0,1] — additional uv-covariance
    return du, dv, duv


def _project_3d_cov_to_scalar(
    var_u: torch.Tensor, var_v: torch.Tensor,
    sigma_dd: torch.Tensor, depth: torch.Tensor,
    J_3d: torch.Tensor, K: torch.Tensor,
) -> torch.Tensor:
    """Project 3D cov to 2D and return isotropic scalar variance addition = trace(Σ_2D)/2."""
    du, dv, _ = _project_3d_cov_to_2x2(var_u, var_v, sigma_dd, depth, J_3d, K)
    return (du + dv) / 2


# ---- Residual magnitude prediction (direct regression) ----------------

def dyn_residual_loss(
    dyn_predictions: list[torch.Tensor],   # K logit maps at H/4
    r_vec: torch.Tensor,                   # (B, 2, H, W)  flow_target − f_rigid vector
    gamma: float = 0.85,
) -> dict[str, torch.Tensor]:
    """γ-weighted smooth-L1 loss for residual magnitude prediction.

    Target:  r     = ‖f_target − f_rigid‖     (residual magnitude in pixels)
    Pred:    r_hat = softplus(dyn_logits)     (positive magnitude in pixels)
    Loss:    smooth_L1(r_hat, r)

    With GT depth, f_rigid is accurate → r isolates true non-rigid motion.
    No cov head, no normalization, no Mahalanobis.
    """
    n_iter = len(dyn_predictions)
    L_total = torch.tensor(0.0, device=r_vec.device)

    r_u = r_vec[:, 0:1]  # (B, 1, H, W)
    r_v = r_vec[:, 1:2]
    target = torch.sqrt(r_u * r_u + r_v * r_v + 1e-8)  # (B, 1, H, W)
    target = torch.where(target < 1.0, torch.zeros_like(target), target)  # <1px → static noise
    target = target.clamp(max=80.0)  # cap explosions

    for i in range(n_iter):
        i_weight = gamma ** (n_iter - i - 1)
        logit = dyn_predictions[i]

        if logit.shape[-2:] != target.shape[-2:]:
            logit = F.interpolate(logit, size=target.shape[-2:], mode="bilinear", align_corners=False)

        r_hat = F.softplus(logit)  # → [0, ∞)
        L_total += i_weight * F.smooth_l1_loss(r_hat, target, reduction="mean")

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
            r_vec = dyn_data[2] if len(dyn_data) > 2 else torch.zeros_like(dyn_data[1])
            loss_dict = dyn_residual_loss(dyn_preds, r_vec, gamma=cfg.gamma)
            loss = loss_dict["L_total"]
            r_hat = F.softplus(dyn_preds[-1]).detach()
            metrics["dyn_r_mean"] = r_hat.mean().item()
            metrics["dyn_r_max"]  = r_hat.max().item()

        case default:
            raise ValueError(f"Unavailable training mode {default}")
    return loss, metrics


def sequence_metric(cfg, preds: torch.Tensor, cov_preds: list[torch.Tensor] | None, gt: torch.Tensor, flow_mask: torch.Tensor | None, dyn_preds: list[torch.Tensor] | None = None, dyn_data: tuple | None = None):
    loss = torch.tensor(0.0, device=gt.device)  # default, overwritten by match cases
    sqe = (preds[-1] - gt)**2
    epe = torch.sum(sqe, dim=1).sqrt()
    
    gt_mag = gt.norm(dim=1, keepdim=True)
    mask = flow_mask.bool() & (gt_mag < cfg.max_flow) if flow_mask is not None else gt_mag < cfg.max_flow
    mask = mask.squeeze(1)  # (B,1,H,W) → (B,H,W) for indexing
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
            r_vec = dyn_data[2] if len(dyn_data) > 2 else torch.zeros_like(dyn_data[1])
            loss_dict = dyn_residual_loss(dyn_preds, r_vec, gamma=cfg.gamma)
            r_hat = F.softplus(dyn_preds[-1]).detach()
            metrics.update({"dyn_loss": loss_dict["L_total"].item()})
            metrics.update({"dyn_r_mean": r_hat.mean().item()})

        case default:
            raise ValueError(f"Unavailable training mode {default}")

    return loss, metrics
