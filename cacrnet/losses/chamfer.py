from __future__ import annotations

from typing import Dict

import torch


def chamfer_l1(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Chamfer distance with L1 norm (||.||_1)."""
    dist = torch.cdist(pred, target, p=1)
    return dist.min(dim=-1).values.mean() + dist.min(dim=-2).values.mean()


def chamfer_l2_squared(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Squared Chamfer distance with L2 norm as defined in the paper (Eq. 2).

    d_CD(P, G) = (1/|P|) sum_{x in P} min_{y in G} ||x-y||^2_2
               + (1/|G|) sum_{y in G} min_{x in P} ||y-x||^2_2
    """
    # torch.cdist with p=2 gives L2 distance; square it for ||.||^2_2
    dist_sq = torch.cdist(pred, target, p=2).pow(2)
    return dist_sq.min(dim=-1).values.mean() + dist_sq.min(dim=-2).values.mean()


# Keep old name as alias
chamfer_l2 = chamfer_l2_squared


def multi_resolution_chamfer(
    pred_low: torch.Tensor,
    pred_mid: torch.Tensor,
    pred_high: torch.Tensor,
    gt_low: torch.Tensor,
    gt_mid: torch.Tensor,
    gt_high: torch.Tensor,
    alpha_low: float = 0.2,
    alpha_mid: float = 0.4,
) -> Dict[str, torch.Tensor]:
    """Multi-resolution CD loss (Eq. 3): L_CD = d^H + alpha_L * d^L + alpha_M * d^M."""
    loss_low = chamfer_l2_squared(pred_low, gt_low)
    loss_mid = chamfer_l2_squared(pred_mid, gt_mid)
    loss_high = chamfer_l2_squared(pred_high, gt_high)
    total = loss_high + alpha_low * loss_low + alpha_mid * loss_mid
    return {
        "loss": total,
        "loss_low": loss_low,
        "loss_mid": loss_mid,
        "loss_high": loss_high,
    }
