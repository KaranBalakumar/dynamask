"""Pairwise ranking loss for score logits using Sampson distance ordering."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def _to_homogeneous(points: torch.Tensor) -> torch.Tensor:
    ones = torch.ones(points.shape[0], 1, device=points.device, dtype=points.dtype)
    return torch.cat([points, ones], dim=-1)


def _normalize_points(points: torch.Tensor, eps: float = 1e-6) -> tuple[torch.Tensor, torch.Tensor]:
    """Normalize 2D points for numerically stable 8-point estimation."""
    mean = points.mean(dim=0, keepdim=True)
    centered = points - mean
    scale = torch.sqrt((centered.pow(2).sum(dim=-1).mean() / 2.0).clamp(min=eps))
    scale_val = 1.0 / scale

    zero = torch.zeros((), device=points.device, dtype=points.dtype)
    one = torch.ones((), device=points.device, dtype=points.dtype)
    T = torch.stack(
        [
            torch.stack([scale_val, zero, -scale_val * mean[0, 0]]),
            torch.stack([zero, scale_val, -scale_val * mean[0, 1]]),
            torch.stack([zero, zero, one]),
        ]
    )
    points_h = _to_homogeneous(points)
    norm_h = (T @ points_h.T).T
    return norm_h[:, :2], T


def _estimate_fundamental(points1: torch.Tensor, points2: torch.Tensor) -> torch.Tensor | None:
    """Estimate a fundamental matrix with normalized 8-point algorithm."""
    if points1.shape[0] < 8 or points2.shape[0] < 8:
        return None

    p1n, T1 = _normalize_points(points1)
    p2n, T2 = _normalize_points(points2)
    x1, y1 = p1n[:, 0], p1n[:, 1]
    x2, y2 = p2n[:, 0], p2n[:, 1]

    A = torch.stack(
        [
            x2 * x1,
            x2 * y1,
            x2,
            y2 * x1,
            y2 * y1,
            y2,
            x1,
            y1,
            torch.ones_like(x1),
        ],
        dim=-1,
    )
    try:
        _, _, Vh = torch.linalg.svd(A, full_matrices=False)
        F_norm = Vh[-1].reshape(3, 3)

        Uf, Sf, Vhf = torch.linalg.svd(F_norm, full_matrices=False)
        Sf[-1] = 0.0
        F_rank2 = Uf @ torch.diag(Sf) @ Vhf

        F_denorm = T2.T @ F_rank2 @ T1
        return F_denorm / (F_denorm.norm() + 1e-8)
    except RuntimeError:
        return None


def sampson_distance(correspondences: torch.Tensor, F_mat: torch.Tensor) -> torch.Tensor:
    """Compute Sampson distance for correspondences [K,4]."""
    x1 = _to_homogeneous(correspondences[:, 0:2])
    x2 = _to_homogeneous(correspondences[:, 2:4])

    Fx1 = (F_mat @ x1.T).T
    Ftx2 = (F_mat.T @ x2.T).T
    x2tFx1 = (x2 * Fx1).sum(dim=-1)

    denom = Fx1[:, 0].pow(2) + Fx1[:, 1].pow(2) + Ftx2[:, 0].pow(2) + Ftx2[:, 1].pow(2)
    return x2tFx1.pow(2) / (denom + 1e-8)


def pairwise_rank_loss(
    score_logit: torch.Tensor,
    flow: torch.Tensor,
    *,
    n_pairs: int = 512,
    margin: float = 1.0,
    static_quantile: float = 0.5,
    min_spread: float = 1e-4,
) -> torch.Tensor:
    """Pairwise margin ranking loss driven by Sampson distance ordering."""
    B, _, H, W = score_logit.shape
    device = score_logit.device
    dtype = score_logit.dtype

    gy, gx = torch.meshgrid(
        torch.arange(H, device=device, dtype=dtype),
        torch.arange(W, device=device, dtype=dtype),
        indexing="ij",
    )
    coords = torch.stack([gx, gy], dim=0).reshape(2, -1)

    batch_losses: list[torch.Tensor] = []
    num_points = H * W
    k_static = max(8, int(num_points * static_quantile))
    k_pairs = max(1, min(int(n_pairs), num_points // 2))

    for b in range(B):
        scores = score_logit[b, 0].reshape(-1)
        flow_b = flow[b].reshape(2, -1)
        corr = torch.stack(
            [
                coords[0],
                coords[1],
                coords[0] + flow_b[0],
                coords[1] + flow_b[1],
            ],
            dim=-1,
        )

        static_idx = torch.topk(-scores, k=k_static).indices
        F_mat = _estimate_fundamental(corr[static_idx, 0:2], corr[static_idx, 2:4])
        if F_mat is None:
            continue

        sampson = sampson_distance(corr, F_mat)
        spread = torch.quantile(sampson, 0.9) - torch.quantile(sampson, 0.1)
        if spread < min_spread:
            continue

        low_idx = torch.topk(-sampson, k=k_pairs).indices
        high_idx = torch.topk(sampson, k=k_pairs).indices

        score_low = scores[low_idx]
        score_high = scores[high_idx]
        rank_loss = F.relu(margin - (score_high - score_low)).mean()
        batch_losses.append(rank_loss)

    if not batch_losses:
        return score_logit.sum() * 0.0
    return torch.stack(batch_losses).mean()
