from __future__ import annotations

from typing import Tuple

import numpy as np
import torch

try:
    import open3d as o3d
except Exception:  # pragma: no cover - optional dependency
    o3d = None


def normalize_point_cloud(points: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
    xyz = points.copy().astype(np.float32)
    mask = np.any(xyz != 0, axis=1)
    if not np.any(mask):
        return xyz, np.zeros(3, dtype=np.float32), 1.0

    valid = xyz[mask]
    centroid = valid.mean(axis=0)
    xyz[mask] = xyz[mask] - centroid

    scale = np.linalg.norm(xyz[mask], axis=1).max()
    scale = float(scale) if scale > 1e-8 else 1.0
    xyz[mask] = xyz[mask] / scale
    return xyz, centroid.astype(np.float32), scale


def estimate_normals_open3d(
    points_xyz: np.ndarray,
    radius: float = 0.1,
    max_nn: int = 30,
) -> np.ndarray:
    if o3d is None:
        raise ImportError("open3d is required to estimate normals.")
    if points_xyz.shape[0] == 0:
        return np.zeros((0, 3), dtype=np.float32)

    centroid = np.mean(points_xyz, axis=0)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_xyz.astype(np.float64))
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=max_nn)
    )
    normals = np.asarray(pcd.normals, dtype=np.float32)

    vec_to_centroid = centroid - points_xyz
    inward = np.sum(normals * vec_to_centroid, axis=1) > 0
    normals[inward] = -normals[inward]
    return normals


def farthest_point_sample(points: torch.Tensor, num_samples: int) -> torch.Tensor:
    if points.ndim != 2:
        raise ValueError(f"Expected [N, C], got {tuple(points.shape)}")
    n_points = points.shape[0]
    if num_samples >= n_points:
        return points

    xyz = points[:, :3]
    centroids = torch.zeros(num_samples, dtype=torch.long, device=points.device)
    distances = torch.full((n_points,), 1e10, device=points.device)
    farthest = torch.randint(0, n_points, (1,), device=points.device).item()

    for idx in range(num_samples):
        centroids[idx] = farthest
        centroid = xyz[farthest].view(1, 3)
        dist = torch.sum((xyz - centroid) ** 2, dim=1)
        distances = torch.minimum(distances, dist)
        farthest = torch.argmax(distances).item()
    return points[centroids]


def toothwise_fps(blocks: torch.Tensor, points_per_tooth: int) -> torch.Tensor:
    if blocks.ndim != 4:
        raise ValueError(f"Expected [B, T, P, C], got {tuple(blocks.shape)}")
    sampled = []
    for tooth_idx in range(blocks.shape[1]):
        tooth = blocks[:, tooth_idx]
        tooth_samples = []
        for batch_idx in range(blocks.shape[0]):
            tooth_samples.append(farthest_point_sample(tooth[batch_idx], points_per_tooth))
        sampled.append(torch.stack(tooth_samples, dim=0))
    return torch.stack(sampled, dim=1)
