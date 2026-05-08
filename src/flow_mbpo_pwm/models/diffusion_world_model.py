import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .mlp import mlp
from .world_model import weight_init, zero_, symexp


class DiffusionWorldModel(nn.Module):
    """Minimal latent-delta diffusion world model for controlled sidecar studies."""

    diffusion_dynamics = True

    def __init__(
        self,
        observation_dim,
        action_dim,
        latent_dim,
        units,
        encoder_units,
        encoder,
        dynamics,
        reward,
        action_dims=None,
        num_bins=None,
        vmin=None,
        vmax=None,
        multitask=False,
        tasks=None,
        task_dim=0,
        diffusion_steps=8,
        beta_start=1e-4,
        beta_end=2e-2,
        sample_clip=5.0,
    ):
        super().__init__()
        self.multitask = multitask
        self.num_bins = num_bins
        self.vmin = vmin
        self.vmax = vmax
        self.latent_dim = latent_dim
        self.diffusion_steps = int(diffusion_steps)
        self.sample_clip = float(sample_clip)
        if self.diffusion_steps < 2:
            raise ValueError("diffusion_steps must be at least 2")

        if self.multitask:
            self._task_emb = nn.Embedding(len(tasks), task_dim, max_norm=1)
            self._action_masks = torch.zeros(len(tasks), action_dim)
            for i in range(len(tasks)):
                self._action_masks[i, : action_dims[i]] = 1.0

        self._encoder = mlp(
            observation_dim + task_dim,
            encoder_units,
            latent_dim,
            last_layer=encoder["last_layer"],
            last_layer_kwargs=encoder["last_layer_kwargs"],
        )
        # Input: current latent z, action a, noisy delta, normalized timestep.
        self._denoiser = mlp(
            latent_dim + action_dim + latent_dim + 1 + task_dim,
            units,
            latent_dim,
            last_layer=dynamics["last_layer"],
            last_layer_kwargs=dynamics["last_layer_kwargs"],
        )
        self._reward = mlp(
            latent_dim + action_dim + task_dim,
            units,
            max(num_bins, 1) if num_bins else 1,
            last_layer=reward["last_layer"],
            last_layer_kwargs=reward["last_layer_kwargs"],
        )

        betas = torch.linspace(float(beta_start), float(beta_end), self.diffusion_steps)
        alphas = 1.0 - betas
        alpha_bars = torch.cumprod(alphas, dim=0)
        alpha_bars_prev = torch.cat([torch.ones(1), alpha_bars[:-1]], dim=0)
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alpha_bars", alpha_bars)
        self.register_buffer("alpha_bars_prev", alpha_bars_prev)

        self.apply(weight_init)
        zero_([self._reward[-1].weight])

    @property
    def total_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def to(self, *args, **kwargs):
        super().to(*args, **kwargs)
        if self.multitask:
            self._action_masks = self._action_masks.to(*args, **kwargs)
        return self

    def train(self, mode=True):
        super().train(mode)
        return self

    def task_emb(self, x, task):
        if isinstance(task, int):
            task = torch.tensor([task], device=x.device)
        emb = self._task_emb(task.long())
        if x.ndim == 3:
            emb = emb.unsqueeze(0).repeat(x.shape[0], 1, 1)
        elif emb.shape[0] == 1:
            emb = emb.repeat(x.shape[0], 1)
        return torch.cat([x, emb], dim=-1)

    def encode(self, obs, task):
        if self.multitask:
            obs = self.task_emb(obs, task)
        return self._encoder(obs)

    def _denoiser_input(self, z, a, noisy_delta, tau, task):
        x = torch.cat([z, a, noisy_delta, tau], dim=-1)
        if self.multitask:
            x = self.task_emb(x, task)
        return x

    def predict_noise(self, z, a, noisy_delta, tau, task):
        x = self._denoiser_input(z, a, noisy_delta, tau, task)
        return self._denoiser(x)

    def diffusion_loss(self, z, a, target_z, task):
        delta = target_z - z
        batch_size = z.shape[0]
        device = z.device
        t_idx = torch.randint(0, self.diffusion_steps, (batch_size,), device=device)
        alpha_bar_t = self.alpha_bars[t_idx].unsqueeze(-1)
        noise = torch.randn_like(delta)
        noisy_delta = torch.sqrt(alpha_bar_t) * delta + torch.sqrt(1.0 - alpha_bar_t) * noise
        tau = t_idx.float().unsqueeze(-1) / float(self.diffusion_steps - 1)
        pred_noise = self.predict_noise(z, a, noisy_delta, tau, task)
        return F.mse_loss(pred_noise, noise)

    def _predict_x0_from_noise(self, x_t, eps_pred, t_idx):
        alpha_bar_t = self.alpha_bars[t_idx].unsqueeze(-1)
        return (x_t - torch.sqrt(1.0 - alpha_bar_t) * eps_pred) / torch.sqrt(alpha_bar_t)

    @torch.no_grad()
    def next(self, z, a, task):
        x = torch.randn_like(z)
        for t in reversed(range(self.diffusion_steps)):
            t_idx = torch.full((z.shape[0],), t, device=z.device, dtype=torch.long)
            tau = t_idx.float().unsqueeze(-1) / float(self.diffusion_steps - 1)
            eps = self.predict_noise(z, a, x, tau, task)
            x0 = self._predict_x0_from_noise(x, eps, t_idx)
            x0 = torch.clamp(x0, -self.sample_clip, self.sample_clip)
            if t == 0:
                x = x0
            else:
                alpha_bar_prev = self.alpha_bars_prev[t_idx].unsqueeze(-1)
                x = torch.sqrt(alpha_bar_prev) * x0 + torch.sqrt(1.0 - alpha_bar_prev) * eps
        return z + x

    def reward(self, z, a, task):
        if self.multitask:
            z = self.task_emb(z, task)
        z = torch.cat([z, a], dim=-1)
        return self._reward(z)

    def step(self, z, a, task):
        z_next = self.next(z, a, task)
        r = self.reward(z, a, task)
        return z_next, r

    def two_hot_inv(self, x):
        if self.num_bins == 0:
            return x
        elif self.num_bins == 1:
            return symexp(x)
        vals = torch.linspace(self.vmin, self.vmax, self.num_bins, device=x.device)
        x = F.softmax(x, dim=-1)
        x = torch.sum(x * vals, dim=-1, keepdim=True)
        return symexp(x)

    def almost_two_hot_inv(self, x):
        if self.num_bins == 0 or self.num_bins is None:
            return x
        elif self.num_bins == 1:
            return symexp(x)
        vals = torch.linspace(self.vmin, self.vmax, self.num_bins, device=x.device)
        x = F.softmax(x, dim=-1)
        x = torch.sum(x * vals, dim=-1, keepdim=True)
        return x
