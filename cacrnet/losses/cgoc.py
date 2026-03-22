from __future__ import annotations

import torch

from cacrnet.utils.pointcloud import estimate_normals_torch


def _curvature(xyz: torch.Tensor, normals: torch.Tensor, k: int = 32) -> torch.Tensor:
    """Curvature indicator (Eq. 5): C_i = (1/k) sum_j (1 - <n_i, n_j>)."""
    dist = torch.cdist(xyz, xyz)
    knn_idx = dist.topk(k=min(k + 1, xyz.shape[1]), largest=False).indices[..., 1:]
    neigh_normals = torch.gather(
        normals.unsqueeze(1).expand(-1, xyz.shape[1], -1, -1),
        2,
        knn_idx.unsqueeze(-1).expand(-1, -1, -1, normals.shape[-1]),
    )
    center_normals = normals.unsqueeze(2)
    return (1.0 - (center_normals * neigh_normals).sum(dim=-1)).mean(dim=-1)


def cgoc_loss(
    pred_points: torch.Tensor,
    gt_points: torch.Tensor,
    top_ratio: float = 0.3,
    temperature: float = 0.1,
    k: int = 32,
) -> torch.Tensor:
    """Curvature-Guided Occlusal Constraint loss (Eq. 6-7).

    If pred_points has only 3 channels (xyz), normals are estimated via NVC
    (Normal Vector Calculation, paper Sec III-A-4).

    Args:
        pred_points: (B, N, 3) or (B, N, 6) predicted crown points
        gt_points: (B, N, 6) ground-truth crown points (xyz + normals)
        top_ratio: fraction of highest-curvature points to select (T/N)
        temperature: SoftMax temperature tau (Eq. 6)
        k: kNN neighborhood size for curvature estimation
    """
    pred_xyz = pred_points[..., :3]
    gt_xyz = gt_points[..., :3]

    # Estimate normals via NVC if not provided (paper Sec III-A-4)
    if pred_points.shape[-1] >= 6:
        pred_normals = pred_points[..., 3:6]
    else:
        pred_normals = estimate_normals_torch(pred_xyz, k=k)

    gt_normals = gt_points[..., 3:6]

    pred_curv = _curvature(pred_xyz, pred_normals, k=k)
    gt_curv = _curvature(gt_xyz, gt_normals, k=k)

    top_T = max(1, int(pred_xyz.shape[1] * top_ratio))

    pred_idx = pred_curv.topk(top_T, dim=-1).indices
    gt_idx = gt_curv.topk(top_T, dim=-1).indices

    pred_top_xyz = torch.gather(pred_xyz, 1, pred_idx.unsqueeze(-1).expand(-1, -1, 3))
    gt_top_xyz = torch.gather(gt_xyz, 1, gt_idx.unsqueeze(-1).expand(-1, -1, 3))
    pred_top_curv = torch.gather(pred_curv, 1, pred_idx)
    gt_top_curv = torch.gather(gt_curv, 1, gt_idx)

    # Eq. 6: dist_ij = ||p_i - g_j||^2_2, w_ij = softmax(-dist_ij / tau)
    dists_sq = torch.cdist(pred_top_xyz, gt_top_xyz, p=2).pow(2)
    weights = torch.softmax(-dists_sq / max(temperature, 1e-6), dim=-1)

    # Eq. 7: curvature + positional differences
    curvature_term = torch.abs(pred_top_curv.unsqueeze(-1) - gt_top_curv.unsqueeze(1))
    position_term = dists_sq

    return (weights * (curvature_term + position_term)).mean()
