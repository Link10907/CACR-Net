from __future__ import annotations

from typing import Dict

import torch


def _pairwise_dist(pred: torch.Tensor, target: torch.Tensor, p: int = 2) -> torch.Tensor:
    return torch.cdist(pred, target, p=p)


def chamfer_l1(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    dist = _pairwise_dist(pred, target, p=1)
    return dist.min(dim=-1).values.mean() + dist.min(dim=-2).values.mean()


def chamfer_l2(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    dist = _pairwise_dist(pred, target, p=2)
    return dist.min(dim=-1).values.mean() + dist.min(dim=-2).values.mean()


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
    loss_low = chamfer_l2(pred_low, gt_low)
    loss_mid = chamfer_l2(pred_mid, gt_mid)
    loss_high = chamfer_l2(pred_high, gt_high)
    total = loss_high + alpha_low * loss_low + alpha_mid * loss_mid
    return {
        "loss": total,
        "loss_low": loss_low,
        "loss_mid": loss_mid,
        "loss_high": loss_high,
    }
