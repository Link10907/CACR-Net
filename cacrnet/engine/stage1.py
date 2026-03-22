from __future__ import annotations

from typing import Dict

import torch

from cacrnet.losses import cgoc_loss, multi_resolution_chamfer, penetration_loss, penetration_rate
from cacrnet.models.cmdent_net import CMDenNet
from cacrnet.utils.pointcloud import farthest_point_sample


def build_stage1_model(cfg) -> CMDenNet:
    return CMDenNet(
        point_dim=cfg.point_dim,
        hidden_dim=cfg.hidden_dim,
        global_dim=cfg.global_dim,
        voxel_resolution=cfg.voxel_resolution,
        k_neighbors=cfg.k_neighbors,
        low_points=cfg.low_points,
        mid_points=cfg.mid_points,
        high_points=cfg.high_points,
        schemes=cfg.schemes,
    )


def _gt_pyramid(
    target_points: torch.Tensor,
    low_points: int,
    mid_points: int,
    high_points: int,
):
    """Build ground-truth point pyramids at three resolutions via IFPS."""
    gt_low, gt_mid, gt_high = [], [], []
    for batch_idx in range(target_points.shape[0]):
        gt = target_points[batch_idx]
        gt_high.append(farthest_point_sample(gt, high_points))
        gt_mid.append(farthest_point_sample(gt, mid_points))
        gt_low.append(farthest_point_sample(gt, low_points))
    return torch.stack(gt_low), torch.stack(gt_mid), torch.stack(gt_high)


def compute_stage1_losses(
    stage1_out: Dict[str, torch.Tensor],
    batch: Dict[str, torch.Tensor],
    cfg,
) -> Dict[str, torch.Tensor]:
    gt_low, gt_mid, gt_high = _gt_pyramid(
        batch["target_points"],
        low_points=cfg.low_points,
        mid_points=cfg.mid_points,
        high_points=cfg.high_points,
    )

    # Multi-resolution CD loss (Eq. 3)
    cd_terms = multi_resolution_chamfer(
        stage1_out["low"],
        stage1_out["mid"],
        stage1_out["high"],
        gt_low[..., :3],
        gt_mid[..., :3],
        gt_high[..., :3],
        alpha_low=cfg.alpha_low,
        alpha_mid=cfg.alpha_mid,
    )

    # CGOC loss (Eq. 7) — pass predicted xyz only; cgoc_loss will estimate
    # normals via NVC (normal vector calculation) internally
    cgoc = cgoc_loss(
        stage1_out["high"],  # (B, N, 3) — normals estimated by NVC inside
        gt_high,             # (B, N, 6) — xyz + normals from GT
        top_ratio=cfg.cgoc_top_ratio,
        temperature=cfg.cgoc_temperature,
        k=cfg.k_neighbors,
    )

    # SDF-based non-penetration loss (Eq. 4)
    sdf_penalty = penetration_loss(stage1_out["high"], batch["environment"])

    # Total loss (Eq. 8)
    total = cfg.lambda_cd * cd_terms["loss"] + cfg.lambda_cgoc * cgoc + cfg.lambda_sdf * sdf_penalty

    return {
        "loss": total,
        "loss_cd": cd_terms["loss"],
        "loss_cgoc": cgoc,
        "loss_sdf": sdf_penalty,
        "penetration_rate": penetration_rate(stage1_out["high"], batch["environment"]),
    }
