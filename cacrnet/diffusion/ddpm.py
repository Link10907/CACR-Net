from __future__ import annotations

import math

import torch
from torch import nn


class SinusoidalTimeEmbedding(nn.Module):
    """Sinusoidal positional embedding for diffusion timesteps."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        half_dim = self.dim // 2
        scale = math.log(10000) / max(half_dim - 1, 1)
        exp = torch.exp(torch.arange(half_dim, device=timesteps.device) * -scale)
        emb = timesteps.float().unsqueeze(1) * exp.unsqueeze(0)
        emb = torch.cat([emb.sin(), emb.cos()], dim=-1)
        if emb.shape[-1] < self.dim:
            emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
        return emb


class DDPMScheduler(nn.Module):
    """DDPM noise scheduler following Ho et al. 2020 (ref [24] in paper).

    Uses the standard posterior variance beta_tilde_t for the reverse step.
    """

    def __init__(self, steps: int, beta_start: float, beta_end: float):
        super().__init__()
        betas = torch.linspace(beta_start, beta_end, steps, dtype=torch.float64)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = torch.cat([torch.ones(1, dtype=torch.float64), alphas_cumprod[:-1]], dim=0)

        # posterior variance: beta_tilde_t = beta_t * (1 - alpha_bar_{t-1}) / (1 - alpha_bar_t)
        posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        # clip first element to avoid log(0)
        posterior_variance = torch.clamp(posterior_variance, min=1e-20)

        self.steps = steps
        self.register_buffer("betas", betas.float())
        self.register_buffer("alphas", alphas.float())
        self.register_buffer("alphas_cumprod", alphas_cumprod.float())
        self.register_buffer("alphas_cumprod_prev", alphas_cumprod_prev.float())
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod).float())
        self.register_buffer("sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod).float())
        self.register_buffer("posterior_variance", posterior_variance.float())
        self.register_buffer("posterior_log_variance", torch.log(posterior_variance).float())

    def sample_timesteps(self, batch_size: int, device: torch.device | str) -> torch.Tensor:
        return torch.randint(0, self.steps, (batch_size,), device=device, dtype=torch.long)

    def q_sample(
        self, x0: torch.Tensor, timesteps: torch.Tensor, noise: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward diffusion: q(x_t | x_0) = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * eps."""
        if noise is None:
            noise = torch.randn_like(x0)
        sqrt_alpha = self.sqrt_alphas_cumprod[timesteps]
        sqrt_one_minus = self.sqrt_one_minus_alphas_cumprod[timesteps]
        # broadcast to match x0 shape
        while sqrt_alpha.ndim < x0.ndim:
            sqrt_alpha = sqrt_alpha.unsqueeze(-1)
            sqrt_one_minus = sqrt_one_minus.unsqueeze(-1)
        return sqrt_alpha * x0 + sqrt_one_minus * noise, noise

    def predict_x0(
        self, xt: torch.Tensor, timesteps: torch.Tensor, noise_pred: torch.Tensor
    ) -> torch.Tensor:
        """Predict x_0 from x_t and predicted noise."""
        sqrt_alpha = self.sqrt_alphas_cumprod[timesteps]
        sqrt_one_minus = self.sqrt_one_minus_alphas_cumprod[timesteps]
        while sqrt_alpha.ndim < xt.ndim:
            sqrt_alpha = sqrt_alpha.unsqueeze(-1)
            sqrt_one_minus = sqrt_one_minus.unsqueeze(-1)
        return (xt - sqrt_one_minus * noise_pred) / sqrt_alpha.clamp_min(1e-6)

    def p_step(
        self, xt: torch.Tensor, timesteps: torch.Tensor, noise_pred: torch.Tensor
    ) -> torch.Tensor:
        """Reverse diffusion step using DDPM posterior (standard beta_tilde variance)."""
        x0_pred = self.predict_x0(xt, timesteps, noise_pred)
        # clamp x0 for stability
        x0_pred = x0_pred.clamp(-5.0, 5.0)

        alpha_bar_t = self.alphas_cumprod[timesteps]
        alpha_bar_prev = self.alphas_cumprod_prev[timesteps]
        beta_t = self.betas[timesteps]
        while alpha_bar_t.ndim < xt.ndim:
            alpha_bar_t = alpha_bar_t.unsqueeze(-1)
            alpha_bar_prev = alpha_bar_prev.unsqueeze(-1)
            beta_t = beta_t.unsqueeze(-1)

        # posterior mean: mu_tilde = (sqrt(alpha_bar_prev) * beta_t / (1 - alpha_bar_t)) * x0
        #                          + (sqrt(alpha_t) * (1 - alpha_bar_prev) / (1 - alpha_bar_t)) * xt
        coeff_x0 = torch.sqrt(alpha_bar_prev) * beta_t / (1.0 - alpha_bar_t)
        coeff_xt = torch.sqrt(1.0 - beta_t) * (1.0 - alpha_bar_prev) / (1.0 - alpha_bar_t)
        mean = coeff_x0 * x0_pred + coeff_xt * xt

        # posterior variance: beta_tilde
        var = self.posterior_variance[timesteps]
        while var.ndim < xt.ndim:
            var = var.unsqueeze(-1)

        noise = torch.randn_like(xt)
        mask = (timesteps > 0).float()
        while mask.ndim < xt.ndim:
            mask = mask.unsqueeze(-1)
        return mean + mask * torch.sqrt(var) * noise
