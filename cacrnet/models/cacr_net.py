from __future__ import annotations

from typing import Dict

import numpy as np
import torch
from torch import nn

from cacrnet.diffusion.ddpm import DDPMScheduler
from cacrnet.models.cmdent_net import CMDenNet
from cacrnet.models.sdf_diff_net import SDFDiffNet
from cacrnet.models.sdf_vae import LatentSDFVAE
from cacrnet.utils.sdf import build_query_grid, extract_mesh_from_sdf_grid


class CACRNet(nn.Module):
    def __init__(
        self,
        stage1: CMDenNet,
        stage2_vae: LatentSDFVAE,
        stage2_diffusion: SDFDiffNet,
        scheduler: DDPMScheduler,
    ):
        super().__init__()
        self.stage1 = stage1
        self.stage2_vae = stage2_vae
        self.stage2_diffusion = stage2_diffusion
        self.scheduler = scheduler

    def forward_stage1(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        return self.stage1(batch["arch"], batch["arch_blocks"])

    def forward_stage2(
        self,
        latent_noisy: torch.Tensor,
        timesteps: torch.Tensor,
        pred_crown: torch.Tensor,
        antagonist: torch.Tensor,
    ) -> torch.Tensor:
        return self.stage2_diffusion(latent_noisy, timesteps, pred_crown, antagonist)

    @torch.no_grad()
    def reconstruct_mesh(
        self,
        batch: Dict[str, torch.Tensor],
        grid_resolution: int = 128,
    ) -> Dict[str, np.ndarray]:
        stage1_out = self.forward_stage1(batch)
        pred_crown = stage1_out["high"]

        latent = torch.randn(pred_crown.shape[0], self.stage2_vae.latent_dim, device=pred_crown.device)
        for step in reversed(range(self.scheduler.steps)):
            timestep = torch.full((pred_crown.shape[0],), step, device=pred_crown.device, dtype=torch.long)
            noise_pred = self.forward_stage2(latent, timestep, pred_crown, batch["antagonist"])
            latent = self.scheduler.p_step(latent, timestep, noise_pred)

        query_grid = build_query_grid(grid_resolution, device=pred_crown.device).unsqueeze(0).expand(pred_crown.shape[0], -1, -1)
        sdf = self.stage2_vae.decode(latent, query_grid)
        sdf_grid = sdf[0].reshape(grid_resolution, grid_resolution, grid_resolution).detach().cpu().numpy()
        verts, faces = extract_mesh_from_sdf_grid(sdf_grid, level=0.0)
        return {"vertices": verts, "faces": faces}
