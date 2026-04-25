"""
Feature track and keyframe data structures for the DRT-loose initializer.

These are lightweight data containers with no heavy dependencies.
All tensors use float64 to match the rest of the DRT-loose pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import torch


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class FeatureTrack:
    """A feature point tracked across multiple keyframes.

    obs: dict mapping keyframe_idx -> (u, v) in normalized image coordinates.
         Each value is a (2,) float64 tensor.
    """
    track_id: int
    obs: dict  # {kf_idx: torch.Tensor shape (2,)}


@dataclass
class KeyframeBundle:
    """Minimal per-keyframe data needed by the DRT-loose initializer."""
    kf_idx: int
    timestamp: float
    R_cam: Optional[torch.Tensor] = None  # (3,3) rotation from DRT solve, initially None


# ---------------------------------------------------------------------------
# Track utilities
# ---------------------------------------------------------------------------

def tracks_with_min_obs(tracks: list[FeatureTrack], min_obs: int = 3) -> list[FeatureTrack]:
    """Filter to only tracks with at least *min_obs* observations."""
    return [t for t in tracks if len(t.obs) >= min_obs]


def select_base_views(track: FeatureTrack, rotations: list) -> tuple[int, int]:
    """Select the two keyframe indices with maximum parallax for LiGT base views.

    Parameters
    ----------
    track    : FeatureTrack whose .obs keys give the set of keyframe indices.
    rotations: list of (3,3) camera rotation matrices indexed by kf_idx.
               Entry may be None; if a rotation is missing we fall back to
               ranking by keyframe index distance alone.

    Returns
    -------
    (l, r) : pair of kf_idx values (l < r) with the largest inter-view
             parallax, measured as the magnitude of the cross product between
             the rotated bearing vectors R_i @ x_i and R_j @ x_j.

    Notes
    -----
    - If the track has only two observations those two are always returned.
    - The bearing vector for each observation is computed as
      [u, v, 1] / ||[u, v, 1]|| (unit norm).
    """
    kf_indices = sorted(track.obs.keys())

    if len(kf_indices) <= 2:
        return kf_indices[0], kf_indices[-1]

    best_pair = (kf_indices[0], kf_indices[1])
    best_score = -1.0

    for ii in range(len(kf_indices)):
        for jj in range(ii + 1, len(kf_indices)):
            ki = kf_indices[ii]
            kj = kf_indices[jj]

            uv_i = track.obs[ki].to(dtype=torch.float64)
            uv_j = track.obs[kj].to(dtype=torch.float64)

            # Bearing vectors (unit norm)
            b_i = _bearing(uv_i)
            b_j = _bearing(uv_j)

            # Rotate into world frame if rotations are available
            if (rotations is not None
                    and ki < len(rotations)
                    and kj < len(rotations)
                    and rotations[ki] is not None
                    and rotations[kj] is not None):
                Ri = rotations[ki].to(dtype=torch.float64)
                Rj = rotations[kj].to(dtype=torch.float64)
                wb_i = Ri @ b_i
                wb_j = Rj @ b_j
            else:
                wb_i = b_i
                wb_j = b_j

            score = torch.linalg.norm(torch.linalg.cross(wb_i, wb_j)).item()

            if score > best_score:
                best_score = score
                best_pair = (ki, kj)

    return best_pair


# ---------------------------------------------------------------------------
# Internal helper (also re-exported for use in translation.py)
# ---------------------------------------------------------------------------

def _bearing(uv: torch.Tensor) -> torch.Tensor:
    """Convert a (2,) normalized-image-coordinate observation to a unit bearing vector (3,)."""
    u, v = uv[0], uv[1]
    ray = torch.stack([u, v, torch.ones_like(u)]).to(dtype=torch.float64)
    return ray / torch.linalg.norm(ray)
