from __future__ import annotations

import torch


def _curvature(points: torch.Tensor, normals: torch.Tensor, k: int = 32) -> torch.Tensor:
    xyz = points[..., :3]
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
    pred_xyz = pred_points[..., :3]
    gt_xyz = gt_points[..., :3]
    pred_normals = pred_points[..., 3:6]
    gt_normals = gt_points[..., 3:6]

    pred_curv = _curvature(pred_points, pred_normals, k=k)
    gt_curv = _curvature(gt_points, gt_normals, k=k)

    top_k_pred = max(1, int(pred_xyz.shape[1] * top_ratio))
    top_k_gt = max(1, int(gt_xyz.shape[1] * top_ratio))

    pred_idx = pred_curv.topk(top_k_pred, dim=-1).indices
    gt_idx = gt_curv.topk(top_k_gt, dim=-1).indices

    pred_top_xyz = torch.gather(pred_xyz, 1, pred_idx.unsqueeze(-1).expand(-1, -1, 3))
    gt_top_xyz = torch.gather(gt_xyz, 1, gt_idx.unsqueeze(-1).expand(-1, -1, 3))
    pred_top_curv = torch.gather(pred_curv, 1, pred_idx)
    gt_top_curv = torch.gather(gt_curv, 1, gt_idx)

    dists = torch.cdist(pred_top_xyz, gt_top_xyz, p=2)
    weights = torch.softmax(-dists / max(temperature, 1e-6), dim=-1)
    curvature_term = torch.abs(pred_top_curv.unsqueeze(-1) - gt_top_curv.unsqueeze(1))
    position_term = dists.pow(2)
    return (weights * (curvature_term + position_term)).mean()
