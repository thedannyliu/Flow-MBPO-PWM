import torch
import torch.nn as nn
import torch.nn.functional as F

from .mlp import mlp
from .world_model import weight_init, zero_, symexp


class EnsembleWorldModel(nn.Module):
    """World model with an ensemble of deterministic dynamics heads."""

    ensemble_dynamics = True

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
        ensemble_size=5,
    ):
        super().__init__()
        self.multitask = multitask
        self.num_bins = num_bins
        self.vmin = vmin
        self.vmax = vmax
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
        self._dynamics = nn.ModuleList(
            [
                mlp(
                    latent_dim + action_dim + task_dim,
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
        return self._encoder(obs)

    def _dynamics_input(self, z, a, task):
        if self.multitask:
            z = self.task_emb(z, task)
        return torch.cat([z, a], dim=-1)

    def next_ensemble(self, z, a, task):
        x = self._dynamics_input(z, a, task)
        preds = [head(x) for head in self._dynamics]
        return torch.stack(preds, dim=0)

    def next(self, z, a, task):
        return self.next_ensemble(z, a, task).mean(dim=0)

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
