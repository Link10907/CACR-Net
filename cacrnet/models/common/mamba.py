from __future__ import annotations

import torch
from torch import nn

try:
    from mamba_ssm import Mamba
except Exception:  # pragma: no cover - optional dependency
    Mamba = None


class _FallbackMixer(nn.Module):
    def __init__(self, dim: int, kernel_size: int = 5):
        super().__init__()
        self.dwconv = nn.Conv1d(dim, dim, kernel_size, padding=kernel_size // 2, groups=dim)
        self.gate = nn.Linear(dim, dim * 2)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        conv = self.dwconv(x.transpose(1, 2)).transpose(1, 2)
        gate, value = self.gate(conv).chunk(2, dim=-1)
        return self.proj(torch.sigmoid(gate) * value)


class ResidualMambaBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        if Mamba is not None:
            self.mixer = Mamba(d_model=dim, d_state=d_state, d_conv=d_conv, expand=expand)
        else:
            self.mixer = _FallbackMixer(dim, kernel_size=max(3, d_conv + 1))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.dropout(self.mixer(self.norm(x)))


class FiLMMambaBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        cond_dim: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.mamba = ResidualMambaBlock(dim, d_state=d_state, d_conv=d_conv, expand=expand, dropout=dropout)
        self.to_scale_shift = nn.Linear(cond_dim, dim * 2)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        scale, shift = self.to_scale_shift(cond).chunk(2, dim=-1)
        x = x * (1.0 + scale.unsqueeze(1)) + shift.unsqueeze(1)
        return self.mamba(x)
