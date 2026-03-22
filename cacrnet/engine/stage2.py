from __future__ import annotations

from typing import Dict

import torch

from cacrnet.diffusion.ddpm import DDPMScheduler
from cacrnet.models.sdf_diff_net import SDFDiffNet
from cacrnet.models.sdf_vae import LatentSDFVAE
from cacrnet.utils.sdf import estimate_point_sdf, sample_sdf_queries


def build_stage2_components(cfg) -> tuple[LatentSDFVAE, SDFDiffNet, DDPMScheduler]:
    vae = LatentSDFVAE(latent_dim=cfg.latent_dim, hidden_dim=cfg.hidden_dim, in_channels=6)
    denoiser = SDFDiffNet(
        latent_dim=cfg.latent_dim,
        latent_tokens=cfg.latent_tokens,
        token_dim=cfg.token_dim,
        condition_dim=cfg.condition_dim,
        hidden_dim=cfg.hidden_dim,
        num_layers=cfg.num_layers,
    )
    scheduler = DDPMScheduler(
        steps=cfg.diffusion_steps,
        beta_start=cfg.beta_start,
        beta_end=cfg.beta_end,
    )
    return vae, denoiser, scheduler


def compute_stage2_losses(
    vae: LatentSDFVAE,
    denoiser: SDFDiffNet,
    scheduler: DDPMScheduler,
    stage1_out: Dict[str, torch.Tensor],
    batch: Dict[str, torch.Tensor],
    cfg,
) -> Dict[str, torch.Tensor]:
    device = batch["target_points"].device

    # Sample SDF query points and estimate GT SDF values
    query_xyz = sample_sdf_queries(batch["target_xyz"], num_queries=cfg.num_query_points)
    sdf_target = estimate_point_sdf(query_xyz, batch["target_xyz"], batch["target_points"][..., 3:6])

    # VAE forward: encode GT crown -> latent -> decode to SDF
    vae_out = vae(batch["target_points"], query_xyz, sdf_targets=sdf_target)

    # Diffusion: add noise to latent and train denoiser
    timesteps = scheduler.sample_timesteps(batch["target_points"].shape[0], device)
    latent_noisy, noise = scheduler.q_sample(vae_out["latent"].detach(), timesteps)
    noise_pred = denoiser(
        latent_noisy,
        timesteps,
        stage1_out["high"].detach(),
        batch["antagonist"],
    )
    diffusion_loss = torch.nn.functional.mse_loss(noise_pred, noise)

    # VAE loss (Eq. 9)
    vae_loss = vae_out["recon_loss"] + cfg.vae_kl_weight * vae_out["kl_loss"]

    return {
        "loss": vae_loss + diffusion_loss,
        "loss_vae": vae_loss,
        "loss_recon": vae_out["recon_loss"],
        "loss_kl": vae_out["kl_loss"],
        "loss_diffusion": diffusion_loss,
    }
