import torch
import torch.nn as nn
import torch.nn.functional as F

from .mlp import mlp
from .world_model import weight_init, zero_, symexp


class EnsembleFlowWorldModel(nn.Module):
    """Flow world model with an ensemble of velocity heads."""

    ensemble_flow_dynamics = True

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
        ensemble_size=3,
    ):
        super().__init__()
        self.multitask = multitask
        self.num_bins = num_bins
        self.vmin = vmin
        self.vmax = vmax
        self.latent_dim = latent_dim
        self.ensemble_size = int(ensemble_size)

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

        self._velocity = nn.ModuleList(
            [
                mlp(
                    latent_dim + action_dim + 1 + task_dim,
                    units,
                    latent_dim,
                    last_layer=dynamics["last_layer"],
                    last_layer_kwargs=dynamics["last_layer_kwargs"],
                )
                for _ in range(self.ensemble_size)
            ]
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
        else:
            task_dim = self._encoder[0].weight.shape[1] - obs.shape[-1]
            if task_dim > 0:
                zero_pad = torch.zeros(*obs.shape[:-1], task_dim, device=obs.device)
                obs = torch.cat([obs, zero_pad], dim=-1)
        return self._encoder(obs)

    def _velocity_input(self, z, a, tau, task):
        x = torch.cat([z, a, tau], dim=-1)
        if self.multitask:
            x = self.task_emb(x, task)
        else:
            task_dim = self._velocity[0][0].weight.shape[1] - x.shape[-1]
            if task_dim > 0:
                zero_pad = torch.zeros(*x.shape[:-1], task_dim, device=x.device)
                x = torch.cat([x, zero_pad], dim=-1)
        return x

    def velocity_ensemble(self, z, a, tau, task):
        x = self._velocity_input(z, a, tau, task)
        preds = [head(x) for head in self._velocity]
        return torch.stack(preds, dim=0)

    def velocity(self, z, a, tau, task):
        return self.velocity_ensemble(z, a, tau, task).mean(dim=0)

    def next(self, z, a, task, integrator=None, substeps=1, reverse=False):
        from flow_mbpo_pwm.utils.integrators import euler_step, heun_step

        if integrator is None or integrator == "heun":
            return heun_step(self.velocity, z, a, task, substeps, reverse=reverse)
        if integrator == "euler":
            return euler_step(self.velocity, z, a, task, substeps, reverse=reverse)
        raise ValueError(f"Unknown integrator: {integrator}")

    def reward(self, z, a, task):
        z = torch.cat([z, a], dim=-1)
        if self.multitask:
            z = self.task_emb(z, task)
        else:
            task_dim = self._reward[0].weight.shape[1] - z.shape[-1]
            if task_dim > 0:
                zero_pad = torch.zeros(*z.shape[:-1], task_dim, device=z.device)
                z = torch.cat([z, zero_pad], dim=-1)
        return self._reward(z)

    def step(self, z, a, task, integrator=None, substeps=1):
        z_next = self.next(z, a, task, integrator, substeps)
        r = self.reward(z, a, task)
        return z_next, r

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
        x = torch.sum(x * vals, dim=-1, keepdim=True)
        return x
