from __future__ import annotations

import torch


def _nearest_environment(
    pred_xyz: torch.Tensor,
    environment: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Find nearest environment surface point for each predicted crown point.

    Args:
        pred_xyz: (B, N, 3) predicted crown coordinates
        environment: (B, M, >=6) environment points (xyz + normals)

    Returns:
        nearest_xyz: (B, N, 3) nearest surface coordinates
        nearest_normals: (B, N, 3) surface normals at nearest points
    """
    env_xyz = environment[..., :3]
    env_normals = environment[..., 3:6]
    dists = torch.cdist(pred_xyz, env_xyz, p=2)
    # mask out zero-padded environment points
    valid = (env_xyz.abs().sum(dim=-1) > 0).unsqueeze(1)
    dists = dists.masked_fill(~valid, float("inf"))
    idx = dists.argmin(dim=-1)
    nearest_xyz = torch.gather(env_xyz, 1, idx.unsqueeze(-1).expand(-1, -1, 3))
    nearest_normals = torch.gather(env_normals, 1, idx.unsqueeze(-1).expand(-1, -1, 3))
    return nearest_xyz, nearest_normals


def penetration_loss(pred_points: torch.Tensor, environment: torch.Tensor) -> torch.Tensor:
    """SDF-based non-penetration loss (Eq. 4).

    L_SDF = (1/N) sum max(0, -d_env(p_i))

    Approximates d_env via signed projection onto nearest surface normal.
    Negative signed distance indicates penetration.

    Args:
        pred_points: (B, N, 3) or (B, N, >=3) predicted crown points
        environment: (B, M, 6) environment mesh points (xyz + normals)
    """
    pred_xyz = pred_points[..., :3]
    nearest_xyz, nearest_normals = _nearest_environment(pred_xyz, environment)
    signed = ((pred_xyz - nearest_xyz) * nearest_normals).sum(dim=-1)
    return torch.relu(-signed).mean()


def penetration_rate(pred_points: torch.Tensor, environment: torch.Tensor) -> torch.Tensor:
    """Fraction of predicted crown points that penetrate the environment.

    Penetration is defined as d_env(p_i) < 0.
    """
    pred_xyz = pred_points[..., :3]
    nearest_xyz, nearest_normals = _nearest_environment(pred_xyz, environment)
    signed = ((pred_xyz - nearest_xyz) * nearest_normals).sum(dim=-1)
    return (signed < 0).float().mean()
