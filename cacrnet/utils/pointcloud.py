from __future__ import annotations

from typing import Tuple

import numpy as np
import torch

try:
    import open3d as o3d
except Exception:  # pragma: no cover - optional dependency
    o3d = None


def normalize_point_cloud(points: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
    """Normalize point cloud to [0, 1] range (paper Sec IV-A-1).

    Returns (normalized_xyz, min_corner, scale) so that denormalization is:
        original = normalized * scale + min_corner
    """
    xyz = points.copy().astype(np.float32)
    mask = np.any(xyz != 0, axis=1)
    if not np.any(mask):
        return xyz, np.zeros(3, dtype=np.float32), 1.0

    valid = xyz[mask]
    mins = valid.min(axis=0)
    maxs = valid.max(axis=0)
    scale = float((maxs - mins).max())
    scale = scale if scale > 1e-8 else 1.0

    xyz[mask] = (xyz[mask] - mins) / scale
    return xyz, mins.astype(np.float32), scale


def estimate_normals_open3d(
    points_xyz: np.ndarray,
    radius: float = 0.1,
    max_nn: int = 30,
) -> np.ndarray:
    """Estimate surface normals using Open3D with outward-facing orientation."""
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

    # Orient normals outward (away from centroid)
    vec_to_centroid = centroid - points_xyz
    inward = np.sum(normals * vec_to_centroid, axis=1) > 0
    normals[inward] = -normals[inward]
    return normals


def estimate_normals_torch(xyz: torch.Tensor, k: int = 32) -> torch.Tensor:
    """Estimate normals via PCA on kNN neighborhoods (NVC in paper Sec III-A-4).

    Args:
        xyz: (B, N, 3) point coordinates
        k: number of neighbors

    Returns:
        normals: (B, N, 3) unit normal vectors
    """
    B, N, _ = xyz.shape
    k_actual = min(k + 1, N)
    dist = torch.cdist(xyz, xyz)
    knn_idx = dist.topk(k_actual, largest=False).indices[:, :, 1:]  # exclude self
    # (B, N, k, 3)
    neighbors = torch.gather(
        xyz.unsqueeze(2).expand(-1, -1, N, -1),
        2,
        knn_idx.unsqueeze(-1).expand(-1, -1, -1, 3),
    )
    # center
    centered = neighbors - xyz.unsqueeze(2)  # (B, N, k, 3)
    # covariance
    cov = torch.matmul(centered.transpose(-1, -2), centered)  # (B, N, 3, 3)
    # smallest eigenvector = normal
    _, eigvecs = torch.linalg.eigh(cov)  # eigenvalues ascending
    normals = eigvecs[:, :, :, 0]  # smallest eigenvalue -> normal direction
    # normalize
    normals = normals / (normals.norm(dim=-1, keepdim=True).clamp_min(1e-8))
    # orient consistently (toward centroid of local patch → flip if needed)
    centroid = xyz.mean(dim=1, keepdim=True)  # (B, 1, 3)
    to_centroid = centroid - xyz  # (B, N, 3)
    flip = (normals * to_centroid).sum(dim=-1) > 0  # should point away
    normals[flip] = -normals[flip]
    return normals


def farthest_point_sample(points: torch.Tensor, num_samples: int) -> torch.Tensor:
    """Farthest point sampling on a single point cloud [N, C]."""
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
    """Tooth-wise Iterative Farthest Point Sampling (TIFPS, paper Sec III-A-1).

    Performs FPS independently within each tooth region.

    Args:
        blocks: (B, T, P, C) per-tooth point clouds
        points_per_tooth: target number of points per tooth
    """
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
