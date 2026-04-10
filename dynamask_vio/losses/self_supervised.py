"""
Self-supervised loss functions for DynaMask V2.

No GT masks. No GT flow. Only: images, IMU, camera intrinsics (training datasets).

Loss components:
  L_pose:   BA pose vs IMU preintegrated pose (primary self-supervised signal)
  L_photo:  Photometric consistency on static regions
  L_reproj: BA reprojection error on static correspondences
  L_reg:    Mask regularisation (ratio bounds + total variation)
  L_smooth: Edge-aware flow smoothness
"""

import torch
import torch.nn.functional as F

from ..models.preintegration import so3_log_map


def pose_consistency_loss(R_ba: torch.Tensor, t_ba: torch.Tensor,
                          R_imu: torch.Tensor, t_imu: torch.Tensor,
                          converged: torch.Tensor,
                          w_rot: float = 5.0,
                          w_dir: float = 1.0) -> torch.Tensor:
    """BA-estimated pose should agree with IMU-preintegrated pose.

    Rotation: geodesic distance on SO(3).
    Translation: direction-only comparison (monocular scale ambiguity).

    Args:
        R_ba:  [B, 3, 3] BA-estimated rotation
        t_ba:  [B, 3] BA-estimated translation
        R_imu: [B, 3, 3] IMU preintegrated rotation
        t_imu: [B, 3] IMU preintegrated translation (if available)
        converged: [B] float mask (1.0 for converged BA samples)
        w_rot: rotation weight
        w_dir: translation direction weight

    Returns:
        scalar loss
    """
    # Rotation error: ||Log(R_ba^T @ R_imu)||^2
    R_err = R_ba.transpose(-1, -2) @ R_imu
    rot_err = so3_log_map(R_err)  # [B, 3]
    loss_rot = rot_err.pow(2).sum(dim=-1)  # [B]

    # Translation direction error (ignore magnitude — scale ambiguity)
    t_ba_norm = F.normalize(t_ba, dim=-1, eps=1e-8)
    t_imu_norm = F.normalize(t_imu, dim=-1, eps=1e-8)
    loss_dir = (t_ba_norm - t_imu_norm).pow(2).sum(dim=-1)  # [B]

    # Only count converged samples
    loss = (w_rot * loss_rot + w_dir * loss_dir) * converged
    return loss.mean()


def photometric_loss(img_prev: torch.Tensor, img_curr: torch.Tensor,
                     flow: torch.Tensor, mask: torch.Tensor,
                     tau: float = 0.1) -> torch.Tensor:
    """Photometric consistency on static regions + dynamic penalty.

    Warps frame t-1 to frame t using flow. Static regions should have low
    photometric error. Dynamic regions should have HIGH error (tau penalty).

    Args:
        img_prev: [B, 3, H, W] previous frame (normalised)
        img_curr: [B, 3, H, W] current frame (normalised)
        flow: [B, 2, H_f, W_f] predicted flow at 1/8 resolution
        mask: [B, 1, H_f, W_f] dynamic probability (after sigmoid) at 1/8 res
        tau: minimum expected error for dynamic regions

    Returns:
        scalar loss
    """
    B, _, H_f, W_f = flow.shape

    # Downsample images to flow resolution
    img_prev_ds = F.interpolate(img_prev, size=(H_f, W_f),
                                 mode="bilinear", align_corners=False)
    img_curr_ds = F.interpolate(img_curr, size=(H_f, W_f),
                                 mode="bilinear", align_corners=False)

    # Build warping grid
    gy, gx = torch.meshgrid(
        torch.arange(H_f, device=flow.device, dtype=flow.dtype),
        torch.arange(W_f, device=flow.device, dtype=flow.dtype),
        indexing="ij",
    )
    coords = torch.stack([gx, gy], dim=0).unsqueeze(0).expand(B, -1, -1, -1)
    warped_coords = coords + flow  # [B, 2, H_f, W_f]

    # Normalise to [-1, 1] for grid_sample
    grid_x = 2.0 * warped_coords[:, 0] / max(W_f - 1, 1) - 1.0
    grid_y = 2.0 * warped_coords[:, 1] / max(H_f - 1, 1) - 1.0
    grid = torch.stack([grid_x, grid_y], dim=-1)  # [B, H_f, W_f, 2]

    warped = F.grid_sample(img_prev_ds, grid, mode="bilinear",
                            padding_mode="border", align_corners=True)

    # Per-pixel photometric error (robust L1)
    photo_err = (warped - img_curr_ds).abs().mean(dim=1, keepdim=True)  # [B, 1, H_f, W_f]

    mask_sq = mask  # [B, 1, H_f, W_f]
    static_weight = 1.0 - mask_sq

    # Static regions: low error
    loss_static = (photo_err * static_weight).sum() / (static_weight.sum() + 1e-6)

    # Dynamic regions: should have HIGH error (penalise if too low)
    loss_dynamic = (F.relu(tau - photo_err) * mask_sq).sum() / (mask_sq.sum() + 1e-6)

    return loss_static + loss_dynamic


def reprojection_loss(reproj_error: torch.Tensor,
                      weights: torch.Tensor,
                      converged: torch.Tensor) -> torch.Tensor:
    """BA reprojection error on static correspondences should be low.

    Args:
        reproj_error: [B, K, 2] from differentiable BA
        weights: [B, K] soft static confidence
        converged: [B] float mask

    Returns:
        scalar loss
    """
    err_sq = reproj_error.pow(2).sum(dim=-1)  # [B, K]
    # Weighted mean per sample
    w_sum = weights.sum(dim=-1).clamp(min=1e-6)  # [B]
    per_sample = (err_sq * weights).sum(dim=-1) / w_sum  # [B]
    loss = (per_sample * converged).mean()
    return loss


def mask_regularisation_loss(mask: torch.Tensor,
                              min_ratio: float = 0.05,
                              max_ratio: float = 0.6,
                              lambda_tv: float = 0.1) -> torch.Tensor:
    """Prevent trivial mask solutions + spatial smoothness.

    Args:
        mask: [B, 1, H, W] dynamic probability (after sigmoid)
        min_ratio: minimum expected dynamic fraction
        max_ratio: maximum expected dynamic fraction
        lambda_tv: total variation weight

    Returns:
        scalar loss
    """
    # Ratio constraint
    mask_ratio = mask.mean(dim=(1, 2, 3))  # [B]
    loss_low = F.relu(min_ratio - mask_ratio).pow(2).mean()
    loss_high = F.relu(mask_ratio - max_ratio).pow(2).mean()

    # Total variation (spatial smoothness)
    tv_h = (mask[:, :, 1:, :] - mask[:, :, :-1, :]).abs().mean()
    tv_w = (mask[:, :, :, 1:] - mask[:, :, :, :-1]).abs().mean()
    loss_tv = lambda_tv * (tv_h + tv_w)

    return loss_low + loss_high + loss_tv


def flow_smoothness_loss(flow: torch.Tensor,
                          image: torch.Tensor) -> torch.Tensor:
    """Edge-aware flow smoothness (UnFlow/DDFlow style).

    Flow should be smooth except at image edges.

    Args:
        flow: [B, 2, H, W] predicted flow
        image: [B, 3, H, W] current frame (for edge detection)

    Returns:
        scalar loss
    """
    # Downsample image to flow resolution if needed
    if image.shape[2:] != flow.shape[2:]:
        image = F.interpolate(image, size=flow.shape[2:],
                               mode="bilinear", align_corners=False)

    # Image gradients (mean across channels)
    img_dx = (image[:, :, :, 1:] - image[:, :, :, :-1]).abs().mean(dim=1, keepdim=True)
    img_dy = (image[:, :, 1:, :] - image[:, :, :-1, :]).abs().mean(dim=1, keepdim=True)

    # Flow gradients
    flow_dx = (flow[:, :, :, 1:] - flow[:, :, :, :-1]).abs()
    flow_dy = (flow[:, :, 1:, :] - flow[:, :, :-1, :]).abs()

    # Edge-aware weighting: flow should be smooth where image is smooth
    weight_x = torch.exp(-img_dx)
    weight_y = torch.exp(-img_dy)

    loss_x = (flow_dx * weight_x).mean()
    loss_y = (flow_dy * weight_y).mean()

    return loss_x + loss_y
