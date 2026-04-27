import torch
import torch.nn as nn
import torch.nn.functional as F

from .mlp import mlp
from .world_model import weight_init, zero_, symexp


class ChunkedWorldModel(nn.Module):
    """Predicts latent endpoints after a fixed action chunk."""

    chunked_dynamics = True

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
        chunk_size=2,
        rollout_consistency_weight=0.0,
        rollout_consistency_steps=0,
        rollout_reward_consistency_weight=0.0,
    ):
        super().__init__()
        self.multitask = multitask
        self.num_bins = num_bins
        self.vmin = vmin
        self.vmax = vmax
        self.latent_dim = latent_dim
        self.action_dim = action_dim
        self.chunk_size = int(chunk_size)
        self.rollout_consistency_weight = float(rollout_consistency_weight)
        self.rollout_consistency_steps = int(rollout_consistency_steps)
        self.rollout_reward_consistency_weight = float(rollout_reward_consistency_weight)
        self._units_cfg = units
        self._dynamics_last_layer = dynamics["last_layer"]
        self._dynamics_last_layer_kwargs = dynamics["last_layer_kwargs"]

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
        self._dynamics = mlp(
            latent_dim + action_dim * self.chunk_size + task_dim,
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
        self.apply(weight_init)
        zero_([self._reward[-1].weight])

    @property
    def total_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def task_emb(self, x, task):
        if isinstance(task, int):
            task = torch.tensor([task], device=x.device)
        emb = self._task_emb(task.long())
        if x.ndim == 3:
            emb = emb.unsqueeze(0).repeat(x.shape[0], 1, 1)
        elif emb.shape[0] == 1:
            emb = emb.repeat(x.shape[0], 1)
        return torch.cat([x, emb], dim=-1)

    def _pad_task(self, x, expected):
        task_dim = expected - x.shape[-1]
        if task_dim > 0:
            pad = torch.zeros(*x.shape[:-1], task_dim, device=x.device, dtype=x.dtype)
            x = torch.cat([x, pad], dim=-1)
        return x

    def encode(self, obs, task):
        if self.multitask:
            obs = self.task_emb(obs, task)
        else:
            obs = self._pad_task(obs, self._encoder[0].weight.shape[1])
        return self._encoder(obs)

    def _flatten_action_chunk(self, action_chunk):
        if action_chunk.ndim != 3:
            raise ValueError("action_chunk must have shape [chunk, batch, action_dim] or [batch, chunk, action_dim]")
        if action_chunk.shape[0] == self.chunk_size:
            action_chunk = action_chunk.permute(1, 0, 2)
        elif action_chunk.shape[1] != self.chunk_size:
            raise ValueError(f"Expected chunk_size={self.chunk_size}, got shape={tuple(action_chunk.shape)}")
        return action_chunk.reshape(action_chunk.shape[0], self.chunk_size * self.action_dim)

    def next_chunk(self, z, action_chunk, task):
        x = z
        if self.multitask:
            x = self.task_emb(x, task)
        act_flat = self._flatten_action_chunk(action_chunk)
        x = torch.cat([x, act_flat], dim=-1)
        x = self._pad_task(x, self._dynamics[0].weight.shape[1])
        return self._dynamics(x)

    def next(self, z, a, task):
        action_chunk = a.unsqueeze(0).repeat(self.chunk_size, 1, 1)
        return self.next_chunk(z, action_chunk, task)

    def reward(self, z, a, task):
        x = z
        if self.multitask:
            x = self.task_emb(x, task)
        x = torch.cat([x, a], dim=-1)
        x = self._pad_task(x, self._reward[0].weight.shape[1])
        return self._reward(x)

    def step(self, z, a, task):
        return self.next(z, a, task), self.reward(z, a, task)

    def two_hot_inv(self, x):
        if self.num_bins == 0:
            return x
        if self.num_bins == 1:
            return symexp(x)
        vals = torch.linspace(self.vmin, self.vmax, self.num_bins, device=x.device)
        x = F.softmax(x, dim=-1)
        x = torch.sum(x * vals, dim=-1, keepdim=True)
        return symexp(x)

    def almost_two_hot_inv(self, x):
        if self.num_bins == 0 or self.num_bins is None:
            return x
        if self.num_bins == 1:
            return symexp(x)
        vals = torch.linspace(self.vmin, self.vmax, self.num_bins, device=x.device)
        x = F.softmax(x, dim=-1)
        return torch.sum(x * vals, dim=-1, keepdim=True)


class ChunkedResidualFlowWorldModel(ChunkedWorldModel):
    """Chunked endpoint MLP plus residual flow correction."""

    chunked_residual_flow_dynamics = True

    def __init__(self, *args, residual_substeps=4, residual_integrator="heun", **kwargs):
        super().__init__(*args, **kwargs)
        self.residual_substeps = int(residual_substeps)
        self.residual_integrator = residual_integrator
        task_dim = self._dynamics[0].weight.shape[1] - (self.latent_dim + self.action_dim * self.chunk_size)
        self._velocity = mlp(
            self.latent_dim * 2 + self.action_dim * self.chunk_size + 1 + max(task_dim, 0),
            self._units_cfg,
            self.latent_dim,
            last_layer=self._dynamics_last_layer,
            last_layer_kwargs=self._dynamics_last_layer_kwargs,
        )
        self._velocity.apply(weight_init)

    def dynamics_parameters(self):
        return list(self._dynamics.parameters()) + list(self._velocity.parameters())

    def residual_velocity(self, z_context, r, action_chunk, tau, task):
        act_flat = self._flatten_action_chunk(action_chunk)
        x = torch.cat([z_context, r, act_flat, tau], dim=-1)
        x = self._pad_task(x, self._velocity[0].weight.shape[1])
        return self._velocity(x)

    def residual_vector(self, z, action_chunk, task):
        dt = 1.0 / max(1, self.residual_substeps)
        r = torch.zeros_like(z)
        for k in range(max(1, self.residual_substeps)):
            tau = torch.full((z.shape[0], 1), k * dt, device=z.device, dtype=z.dtype)
            k1 = self.residual_velocity(z, r, action_chunk, tau, task)
            if self.residual_integrator == "euler":
                r = r + dt * k1
            elif self.residual_integrator == "heun":
                r_pred = r + dt * k1
                tau_next = torch.full(
                    (z.shape[0], 1),
                    min(1.0, (k + 1) * dt),
                    device=z.device,
                    dtype=z.dtype,
                )
                k2 = self.residual_velocity(z, r_pred, action_chunk, tau_next, task)
                r = r + 0.5 * dt * (k1 + k2)
            else:
                raise ValueError(f"Unknown residual integrator: {self.residual_integrator}")
        return r

    def base_next_chunk(self, z, action_chunk, task):
        return super().next_chunk(z, action_chunk, task)

    def next_chunk(self, z, action_chunk, task):
        return self.base_next_chunk(z, action_chunk, task) + self.residual_vector(z, action_chunk, task)
