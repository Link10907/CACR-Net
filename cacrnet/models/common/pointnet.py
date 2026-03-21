from __future__ import annotations

import torch
from torch import nn


class PointNetEncoder(nn.Module):
    def __init__(self, in_channels: int = 3, hidden_dim: int = 128, out_dim: int = 256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Conv1d(in_channels, hidden_dim, 1),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Conv1d(hidden_dim, hidden_dim, 1),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Conv1d(hidden_dim, out_dim, 1),
            nn.BatchNorm1d(out_dim),
            nn.GELU(),
        )
        self.out_dim = out_dim

    def forward(self, points: torch.Tensor) -> torch.Tensor:
        if points.ndim != 3:
            raise ValueError(f"Expected [B, N, C], got {tuple(points.shape)}")
        features = self.mlp(points.transpose(1, 2))
        return torch.max(features, dim=-1).values
