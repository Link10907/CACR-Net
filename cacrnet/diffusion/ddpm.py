from __future__ import annotations

import math

import torch
from torch import nn


class SinusoidalTimeEmbedding(nn.Module):
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
    def __init__(self, steps: int, beta_start: float, beta_end: float):
        super().__init__()
        betas = torch.linspace(beta_start, beta_end, steps, dtype=torch.float32)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_prev = torch.cat([torch.ones(1), alphas_cumprod[:-1]], dim=0)

        self.steps = steps
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("alphas_prev", alphas_prev)
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer(
            "sqrt_one_minus_alphas_cumprod",
            torch.sqrt(1.0 - alphas_cumprod),
        )

    def sample_timesteps(self, batch_size: int, device: torch.device | str) -> torch.Tensor:
        return torch.randint(0, self.steps, (batch_size,), device=device, dtype=torch.long)

    def q_sample(self, x0: torch.Tensor, timesteps: torch.Tensor, noise: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        if noise is None:
            noise = torch.randn_like(x0)
        sqrt_alpha = self.sqrt_alphas_cumprod[timesteps].unsqueeze(-1)
        sqrt_one_minus = self.sqrt_one_minus_alphas_cumprod[timesteps].unsqueeze(-1)
        return sqrt_alpha * x0 + sqrt_one_minus * noise, noise

    def predict_x0(self, xt: torch.Tensor, timesteps: torch.Tensor, noise_pred: torch.Tensor) -> torch.Tensor:
        sqrt_alpha = self.sqrt_alphas_cumprod[timesteps].unsqueeze(-1)
        sqrt_one_minus = self.sqrt_one_minus_alphas_cumprod[timesteps].unsqueeze(-1)
        return (xt - sqrt_one_minus * noise_pred) / sqrt_alpha.clamp_min(1e-6)

    def p_step(self, xt: torch.Tensor, timesteps: torch.Tensor, noise_pred: torch.Tensor) -> torch.Tensor:
        betas_t = self.betas[timesteps].unsqueeze(-1)
        alpha_t = self.alphas[timesteps].unsqueeze(-1)
        alpha_bar_t = self.alphas_cumprod[timesteps].unsqueeze(-1)
        alpha_bar_prev = self.alphas_prev[timesteps].unsqueeze(-1)
        x0 = self.predict_x0(xt, timesteps, noise_pred)
        mean = (
            torch.sqrt(alpha_bar_prev) * betas_t / (1.0 - alpha_bar_t) * x0
            + torch.sqrt(alpha_t) * (1.0 - alpha_bar_prev) / (1.0 - alpha_bar_t) * xt
        )
        noise = torch.randn_like(xt)
        mask = (timesteps > 0).float().unsqueeze(-1)
        return mean + mask * torch.sqrt(betas_t) * noise
