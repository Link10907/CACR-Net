from __future__ import annotations

from typing import Dict

import torch
from torch import nn

from cacrnet.models.common.pointnet import PointNetEncoder


class SDFDecoder(nn.Module):
    """Decoder D: maps latent z + query xyz -> SDF value (paper Sec III-B)."""

    def __init__(self, latent_dim: int = 512, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim + 3, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, latent: torch.Tensor, query_xyz: torch.Tensor) -> torch.Tensor:
        """
        Args:
            latent: (B, latent_dim) latent representation
            query_xyz: (B, Q, 3) query 3D coordinates

        Returns:
            sdf: (B, Q, 1) predicted SDF values
        """
        latent_expanded = latent.unsqueeze(1).expand(-1, query_xyz.shape[1], -1)
        return self.net(torch.cat([latent_expanded, query_xyz], dim=-1))


class LatentSDFVAE(nn.Module):
    """Variational autoencoder for latent SDF space (paper Sec III-B, Eq. 9).

    Encoder E: crown point cloud -> (mu, logvar) -> z via reparameterization
    Decoder D: z + query_xyz -> SDF values
    """

    def __init__(self, latent_dim: int = 512, hidden_dim: int = 256, in_channels: int = 6):
        super().__init__()
        self.encoder = PointNetEncoder(in_channels=in_channels, hidden_dim=hidden_dim, out_dim=hidden_dim)
        self.to_mu = nn.Linear(hidden_dim, latent_dim)
        self.to_logvar = nn.Linear(hidden_dim, latent_dim)
        self.decoder = SDFDecoder(latent_dim=latent_dim, hidden_dim=hidden_dim)
        self.latent_dim = latent_dim

    def encode(self, points: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        feat = self.encoder(points)
        return self.to_mu(feat), self.to_logvar(feat)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, latent: torch.Tensor, query_xyz: torch.Tensor) -> torch.Tensor:
        return self.decoder(latent, query_xyz)

    def forward(
        self,
        points: torch.Tensor,
        query_xyz: torch.Tensor,
        sdf_targets: torch.Tensor | None = None,
    ) -> Dict[str, torch.Tensor]:
        mu, logvar = self.encode(points)
        latent = self.reparameterize(mu, logvar)
        sdf_pred = self.decode(latent, query_xyz)
        output = {"latent": latent, "mu": mu, "logvar": logvar, "sdf_pred": sdf_pred}
        if sdf_targets is not None:
            # Eq. 9: L1 reconstruction + KL divergence
            recon = torch.nn.functional.l1_loss(sdf_pred, sdf_targets)
            kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
            output["recon_loss"] = recon
            output["kl_loss"] = kl
        return output
