from __future__ import annotations

from typing import Tuple

import numpy as np
import torch

try:
    from skimage import measure
except Exception:  # pragma: no cover - optional dependency
    measure = None


def sample_sdf_queries(
    surface_points: torch.Tensor,
    num_queries: int,
    near_surface_ratio: float = 0.7,
    jitter_std: float = 0.03,
) -> torch.Tensor:
    """Sample query points for SDF supervision.

    70% near-surface (jittered from surface points) + 30% uniform in [0, 1]^3.
    """
    batch_size, n_points, _ = surface_points.shape
    num_near = int(num_queries * near_surface_ratio)
    num_uniform = num_queries - num_near

    near_idx = torch.randint(0, n_points, (batch_size, num_near), device=surface_points.device)
    near_pts = torch.gather(
        surface_points,
        1,
        near_idx.unsqueeze(-1).expand(-1, -1, surface_points.shape[-1]),
    )
    near_pts = near_pts + torch.randn_like(near_pts) * jitter_std
    # uniform queries in [0, 1]^3 (matching paper normalization range)
    uniform_pts = torch.rand(batch_size, num_uniform, 3, device=surface_points.device)
    return torch.cat([near_pts[..., :3], uniform_pts], dim=1)


def estimate_point_sdf(
    query_xyz: torch.Tensor,
    surface_points: torch.Tensor,
    surface_normals: torch.Tensor,
) -> torch.Tensor:
    """Estimate SDF values at query points using nearest-point normal projection.

    This approximates the true SDF by computing the signed distance as the
    projection of (query - nearest_surface_point) onto the surface normal.
    """
    distances = torch.cdist(query_xyz, surface_points)
    nn_index = distances.argmin(dim=-1)
    nearest_points = torch.gather(
        surface_points,
        1,
        nn_index.unsqueeze(-1).expand(-1, -1, surface_points.shape[-1]),
    )
    nearest_normals = torch.gather(
        surface_normals,
        1,
        nn_index.unsqueeze(-1).expand(-1, -1, surface_normals.shape[-1]),
    )
    signed = torch.sum((query_xyz - nearest_points) * nearest_normals, dim=-1, keepdim=True)
    return signed


def build_query_grid(
    resolution: int,
    bounds: Tuple[float, float] = (0.0, 1.0),
    device: str | torch.device = "cpu",
) -> torch.Tensor:
    """Build a 3D query grid for SDF evaluation and Marching Cubes.

    Uses [0, 1] bounds to match the paper's normalization range.
    """
    lo, hi = bounds
    axis = torch.linspace(lo, hi, resolution, device=device)
    zz, yy, xx = torch.meshgrid(axis, axis, axis, indexing="ij")
    grid = torch.stack([xx, yy, zz], dim=-1)
    return grid.view(-1, 3)


def extract_mesh_from_sdf_grid(
    sdf_grid: np.ndarray,
    level: float = 0.0,
    spacing: Tuple[float, float, float] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract mesh from SDF grid via Marching Cubes.

    Args:
        sdf_grid: (R, R, R) SDF values on a regular grid
        level: iso-surface level (0.0 for zero-level set)
        spacing: voxel spacing; if None, normalized to [0, 1]
    """
    if measure is None:
        raise ImportError("scikit-image is required for marching cubes.")
    res = sdf_grid.shape[0]
    if spacing is None:
        spacing = (1.0 / res, 1.0 / res, 1.0 / res)
    verts, faces, _normals, _values = measure.marching_cubes(
        sdf_grid, level=level, spacing=spacing
    )
    return verts.astype(np.float32), faces.astype(np.int32)
