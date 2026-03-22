from __future__ import annotations

import torch
from torch import nn

try:
    from mamba_ssm import Mamba
except Exception:  # pragma: no cover - optional dependency
    Mamba = None


class _FallbackMixer(nn.Module):
    """Gated depthwise-conv mixer used when mamba_ssm is not installed."""

    def __init__(self, dim: int, kernel_size: int = 5):
        super().__init__()
        self.dwconv = nn.Conv1d(dim, dim, kernel_size, padding=kernel_size // 2, groups=dim)
        self.gate = nn.Linear(dim, dim * 2)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        conv = self.dwconv(x.transpose(1, 2)).transpose(1, 2)
        gate, value = self.gate(conv).chunk(2, dim=-1)
        return self.proj(torch.sigmoid(gate) * value)


def _build_ssm(dim: int, d_state: int, d_conv: int, expand: int) -> nn.Module:
    if Mamba is not None:
        return Mamba(d_model=dim, d_state=d_state, d_conv=d_conv, expand=expand)
    return _FallbackMixer(dim, kernel_size=max(3, d_conv + 1))


class BidirectionalMambaBlock(nn.Module):
    """Bidirectional SSM block as described in the paper (Sec III-A-2).

    The serialized sequence and its reversed copy are processed by SSMs in the
    forward and backward directions; a 1x1 Conv1D performs channel projection
    and the two directional representations are fused into a single feature.
    Residual connections link adjacent blocks with optional channel alignment.
    """

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
        self.forward_ssm = _build_ssm(dim, d_state, d_conv, expand)
        self.backward_ssm = _build_ssm(dim, d_state, d_conv, expand)
        self.channel_proj = nn.Conv1d(dim, dim, 1)
        self.fusion = nn.Linear(dim * 2, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm(x)
        # channel projection (1x1 Conv1D)
        h = self.channel_proj(h.transpose(1, 2)).transpose(1, 2)
        # forward SSM
        h_fwd = self.forward_ssm(h)
        # backward SSM on reversed sequence
        h_bwd = self.backward_ssm(h.flip(dims=[1])).flip(dims=[1])
        # fuse bidirectional features
        h = self.fusion(torch.cat([h_fwd, h_bwd], dim=-1))
        return x + self.dropout(h)


# Keep alias for backward compatibility
ResidualMambaBlock = BidirectionalMambaBlock


class FiLMMambaBlock(nn.Module):
    """FiLM-modulated Mamba block as described in the paper (Eq. 10).

    h^(l+1) = Mamba(h^(l)) * sigma(Linear(ctx)) + Linear(ctx)
    FiLM scale/shift are applied AFTER Mamba processing.
    """

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
        self.norm = nn.LayerNorm(dim)
        self.ssm = _build_ssm(dim, d_state, d_conv, expand)
        self.to_scale = nn.Linear(cond_dim, dim)
        self.to_shift = nn.Linear(cond_dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        h = self.ssm(self.norm(x))
        # FiLM: scale and shift AFTER Mamba (Eq. 10)
        scale = torch.sigmoid(self.to_scale(cond)).unsqueeze(1)
        shift = self.to_shift(cond).unsqueeze(1)
        return x + self.dropout(h * scale + shift)
