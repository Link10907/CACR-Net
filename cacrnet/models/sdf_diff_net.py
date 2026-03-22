from __future__ import annotations

import torch
from torch import nn

from cacrnet.diffusion.ddpm import SinusoidalTimeEmbedding
from cacrnet.models.common.mamba import FiLMMambaBlock
from cacrnet.models.common.pointnet import PointNetEncoder


class SDFDiffNet(nn.Module):
    """Latent Conditional SDF Diffusion Dental Crown Network (paper Sec III-B).

    Conditioning (paper Sec III-B):
    - Cond_prep: PointNet encoding of stage-1 predicted crown point cloud
    - Cond_ant: PointNet encoding of antagonist teeth point cloud
    - Cond_final = concat(Cond_prep, Cond_ant)
    - ctx_emb = concat(Cond_final, time_embedding)

    Denoising network: FiLM-modulated Mamba layers (Eq. 10).
    """

    def __init__(
        self,
        latent_dim: int = 512,
        latent_tokens: int = 16,
        token_dim: int = 32,
        condition_dim: int = 256,
        hidden_dim: int = 256,
        num_layers: int = 6,
    ):
        super().__init__()
        if latent_tokens * token_dim != latent_dim:
            raise ValueError("latent_tokens * token_dim must equal latent_dim")
        self.latent_dim = latent_dim
        self.latent_tokens = latent_tokens
        self.token_dim = token_dim

        # Condition encoders (PointNet-based, ref [10])
        self.pred_encoder = PointNetEncoder(in_channels=3, hidden_dim=condition_dim, out_dim=condition_dim)
        self.ant_encoder = PointNetEncoder(in_channels=6, hidden_dim=condition_dim, out_dim=condition_dim)

        # Timestep embedding
        self.time_embed = nn.Sequential(
            SinusoidalTimeEmbedding(condition_dim),
            nn.Linear(condition_dim, condition_dim),
            nn.GELU(),
        )

        # ctx_emb = MLP(concat(Cond_prep, Cond_ant, time_embed))
        self.ctx_proj = nn.Sequential(
            nn.Linear(condition_dim * 3, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Latent token projection
        self.latent_in = nn.Linear(token_dim, hidden_dim)
        # FiLM-modulated Mamba layers (Eq. 10)
        self.layers = nn.ModuleList([FiLMMambaBlock(hidden_dim, hidden_dim) for _ in range(num_layers)])
        self.latent_out = nn.Linear(hidden_dim, token_dim)

    def condition(
        self, pred_crown: torch.Tensor, antagonist: torch.Tensor, timesteps: torch.Tensor
    ) -> torch.Tensor:
        cond_pred = self.pred_encoder(pred_crown[..., :3])
        cond_ant = self.ant_encoder(antagonist[..., :6])
        cond_time = self.time_embed(timesteps)
        return self.ctx_proj(torch.cat([cond_pred, cond_ant, cond_time], dim=-1))

    def forward(
        self,
        latent_noisy: torch.Tensor,
        timesteps: torch.Tensor,
        pred_crown: torch.Tensor,
        antagonist: torch.Tensor,
    ) -> torch.Tensor:
        """Predict noise from noisy latent conditioned on crown + antagonist + timestep."""
        tokens = latent_noisy.view(latent_noisy.shape[0], self.latent_tokens, self.token_dim)
        hidden = self.latent_in(tokens)
        cond = self.condition(pred_crown, antagonist, timesteps)
        for layer in self.layers:
            hidden = layer(hidden, cond)
        return self.latent_out(hidden).reshape(latent_noisy.shape[0], -1)
