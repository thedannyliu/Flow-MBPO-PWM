import torch
import torch.nn as nn
import torch.nn.functional as F

from .mlp import mlp
from .world_model import weight_init, zero_, symexp


class _BaseStateWM(nn.Module):
    def _init_common(
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
        rollout_consistency_weight=0.0,
        rollout_consistency_steps=0,
        rollout_reward_consistency_weight=0.0,
    ):
        self.multitask = multitask
        self.num_bins = num_bins
        self.vmin = vmin
        self.vmax = vmax
        self.latent_dim = latent_dim
        self.action_dim = action_dim
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
        self._reward = mlp(
            latent_dim + action_dim + task_dim,
            units,
            max(num_bins, 1) if num_bins else 1,
            last_layer=reward["last_layer"],
            last_layer_kwargs=reward["last_layer_kwargs"],
        )

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
        return symexp(torch.sum(x * vals, dim=-1, keepdim=True))

    def almost_two_hot_inv(self, x):
        if self.num_bins == 0 or self.num_bins is None:
            return x
        if self.num_bins == 1:
            return symexp(x)
        vals = torch.linspace(self.vmin, self.vmax, self.num_bins, device=x.device)
        x = F.softmax(x, dim=-1)
        return torch.sum(x * vals, dim=-1, keepdim=True)


class LatentTransformerWorldModel(_BaseStateWM):
    """Small causal latent transformer for one-step residual dynamics."""

    sequence_dynamics = True

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
        transformer_dim=256,
        transformer_layers=2,
        transformer_heads=4,
        context_len=8,
        residual_prediction=True,
        **kwargs,
    ):
        super().__init__()
        self._init_common(
            observation_dim,
            action_dim,
            latent_dim,
            units,
            encoder_units,
            encoder,
            dynamics,
            reward,
            **kwargs,
        )
        self.context_len = int(context_len)
        self.residual_prediction = bool(residual_prediction)
        self._token = nn.Linear(latent_dim + action_dim + 1, transformer_dim)
        self._pos = nn.Parameter(torch.zeros(1, self.context_len, transformer_dim))
        layer = nn.TransformerEncoderLayer(
            d_model=transformer_dim,
            nhead=transformer_heads,
            dim_feedforward=transformer_dim * 4,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self._transformer = nn.TransformerEncoder(layer, num_layers=transformer_layers)
        self._out = mlp(
            transformer_dim,
            [transformer_dim],
            latent_dim,
            last_layer=dynamics["last_layer"],
            last_layer_kwargs=dynamics["last_layer_kwargs"],
        )
        self.apply(weight_init)
        zero_([self._reward[-1].weight])

    def dynamics_parameters(self):
        return list(self._token.parameters()) + list(self._transformer.parameters()) + list(self._out.parameters()) + [self._pos]

    def _causal_transform(self, tokens):
        if tokens.shape[1] > self.context_len:
            tokens = tokens[:, -self.context_len :]
        h = self._token(tokens) + self._pos[:, : tokens.shape[1]]
        mask = torch.triu(torch.ones(tokens.shape[1], tokens.shape[1], device=tokens.device, dtype=torch.bool), diagonal=1)
        return self._transformer(h, mask=mask)

    def predict_next_sequence(self, z_seq, act_seq, rew_seq=None, task=None):
        if rew_seq is None:
            rew_seq = torch.zeros(*z_seq.shape[:2], 1, device=z_seq.device, dtype=z_seq.dtype)
        preds = []
        for t in range(z_seq.shape[0]):
            lo = max(0, t - self.context_len + 1)
            tokens = torch.cat([z_seq[lo : t + 1], act_seq[lo : t + 1], rew_seq[lo : t + 1]], dim=-1).permute(1, 0, 2)
            h = self._causal_transform(tokens)[:, -1]
            delta = self._out(h)
            preds.append(z_seq[t] + delta if self.residual_prediction else delta)
        return torch.stack(preds)

    def next(self, z, a, task):
        rew = torch.zeros(z.shape[0], 1, device=z.device, dtype=z.dtype)
        return self.predict_next_sequence(z.unsqueeze(0), a.unsqueeze(0), rew.unsqueeze(0), task)[0]


class ChunkedLatentTransformerWorldModel(LatentTransformerWorldModel):
    """Causal latent transformer that predicts chunk endpoints."""

    chunked_dynamics = True
    chunked_sequence_dynamics = True

    def __init__(self, *args, chunk_size=2, **kwargs):
        self.chunk_size = int(chunk_size)
        super().__init__(*args, **kwargs)
        d_model = self._token.out_features
        self._chunk_token = nn.Linear(self.latent_dim + self.action_dim * self.chunk_size + 1, d_model)
        self._chunk_token.apply(weight_init)

    def _flatten_action_chunk(self, action_chunk):
        if action_chunk.ndim != 3:
            raise ValueError("action_chunk must have shape [chunk, batch, action_dim] or [batch, chunk, action_dim]")
        if action_chunk.shape[0] == self.chunk_size:
            action_chunk = action_chunk.permute(1, 0, 2)
        elif action_chunk.shape[1] != self.chunk_size:
            raise ValueError(f"Expected chunk_size={self.chunk_size}, got shape={tuple(action_chunk.shape)}")
        return action_chunk.reshape(action_chunk.shape[0], self.chunk_size * self.action_dim)

    def next_chunk_with_context(self, z_context, act_context, rew_context, action_chunk, task):
        act_flat = self._flatten_action_chunk(action_chunk)
        z0 = z_context[-1]
        flag = torch.ones(z0.shape[0], 1, device=z0.device, dtype=z0.dtype)
        chunk_tok = torch.cat([z0, act_flat, flag], dim=-1).unsqueeze(1)

        if rew_context is None:
            rew_context = torch.zeros(*z_context.shape[:2], 1, device=z_context.device, dtype=z_context.dtype)
        context_tokens = torch.cat([z_context, act_context, rew_context], dim=-1).permute(1, 0, 2)
        if context_tokens.shape[1] > self.context_len - 1:
            context_tokens = context_tokens[:, -(self.context_len - 1) :]
        h_context = self._token(context_tokens)
        h_chunk = self._chunk_token(chunk_tok)
        h = torch.cat([h_context, h_chunk], dim=1)
        h = h + self._pos[:, : h.shape[1]]
        mask = torch.triu(torch.ones(h.shape[1], h.shape[1], device=h.device, dtype=torch.bool), diagonal=1)
        y = self._transformer(h, mask=mask)[:, -1]
        delta = self._out(y)
        return z0 + delta if self.residual_prediction else delta

    def predict_chunk_sequence(self, z_all, act_seq, rew_seq=None, task=None):
        preds = []
        max_start = act_seq.shape[0] - self.chunk_size + 1
        for start in range(max_start):
            lo = max(0, start - self.context_len + 2)
            z_context = z_all[lo : start + 1]
            act_context = act_seq[lo : start + 1]
            rew_context = rew_seq[lo : start + 1] if rew_seq is not None else None
            preds.append(self.next_chunk_with_context(z_context, act_context, rew_context, act_seq[start : start + self.chunk_size], task))
        return torch.stack(preds)

    def next_chunk(self, z, action_chunk, task):
        first_action = action_chunk[0] if action_chunk.shape[0] == self.chunk_size else action_chunk[:, 0]
        rew = torch.zeros(z.shape[0], 1, device=z.device, dtype=z.dtype)
        return self.next_chunk_with_context(z.unsqueeze(0), first_action.unsqueeze(0), rew.unsqueeze(0), action_chunk, task)

    def next(self, z, a, task):
        action_chunk = a.unsqueeze(0).repeat(self.chunk_size, 1, 1)
        return self.next_chunk(z, action_chunk, task)


class ChunkedLatentActionWorldModel(_BaseStateWM):
    """Chunk endpoint model with a learned action-effect bottleneck."""

    chunked_dynamics = True
    latent_action_dynamics = True

    def __init__(self, *args, chunk_size=2, latent_action_dim=64, **kwargs):
        super().__init__()
        self._init_common(*args, **kwargs)
        self.chunk_size = int(chunk_size)
        self.latent_action_dim = int(latent_action_dim)
        self._action_encoder = mlp(
            self.action_dim * self.chunk_size,
            [max(self.latent_action_dim, 64)],
            self.latent_action_dim,
            last_layer="linear",
            last_layer_kwargs={},
        )
        self._dynamics = mlp(
            self.latent_dim + self.latent_action_dim,
            self._units_cfg,
            self.latent_dim,
            last_layer=self._dynamics_last_layer,
            last_layer_kwargs=self._dynamics_last_layer_kwargs,
        )
        self.apply(weight_init)
        zero_([self._reward[-1].weight])

    def _flatten_action_chunk(self, action_chunk):
        if action_chunk.shape[0] == self.chunk_size:
            action_chunk = action_chunk.permute(1, 0, 2)
        return action_chunk.reshape(action_chunk.shape[0], self.chunk_size * self.action_dim)

    def latent_action(self, action_chunk):
        return self._action_encoder(self._flatten_action_chunk(action_chunk))

    def next_chunk(self, z, action_chunk, task):
        u = self.latent_action(action_chunk)
        return self._dynamics(torch.cat([z, u], dim=-1))

    def next(self, z, a, task):
        return self.next_chunk(z, a.unsqueeze(0).repeat(self.chunk_size, 1, 1), task)

    def dynamics_parameters(self):
        return list(self._action_encoder.parameters()) + list(self._dynamics.parameters())


class ChunkedLatentActionTransformerWorldModel(ChunkedLatentTransformerWorldModel):
    """Chunked transformer that uses a learned latent action chunk token."""

    latent_action_dynamics = True

    def __init__(self, *args, latent_action_dim=64, **kwargs):
        self.latent_action_dim = int(latent_action_dim)
        super().__init__(*args, **kwargs)
        self._action_encoder = mlp(
            self.action_dim * self.chunk_size,
            [max(self.latent_action_dim, 64)],
            self.latent_action_dim,
            last_layer="linear",
            last_layer_kwargs={},
        )
        d_model = self._token.out_features
        self._chunk_token = nn.Linear(self.latent_dim + self.latent_action_dim + 1, d_model)
        self._action_encoder.apply(weight_init)
        self._chunk_token.apply(weight_init)

    def latent_action(self, action_chunk):
        return self._action_encoder(self._flatten_action_chunk(action_chunk))

    def next_chunk_with_context(self, z_context, act_context, rew_context, action_chunk, task):
        u = self.latent_action(action_chunk)
        z0 = z_context[-1]
        flag = torch.ones(z0.shape[0], 1, device=z0.device, dtype=z0.dtype)
        chunk_tok = torch.cat([z0, u, flag], dim=-1).unsqueeze(1)
        if rew_context is None:
            rew_context = torch.zeros(*z_context.shape[:2], 1, device=z_context.device, dtype=z_context.dtype)
        context_tokens = torch.cat([z_context, act_context, rew_context], dim=-1).permute(1, 0, 2)
        if context_tokens.shape[1] > self.context_len - 1:
            context_tokens = context_tokens[:, -(self.context_len - 1) :]
        h = torch.cat([self._token(context_tokens), self._chunk_token(chunk_tok)], dim=1)
        h = h + self._pos[:, : h.shape[1]]
        mask = torch.triu(torch.ones(h.shape[1], h.shape[1], device=h.device, dtype=torch.bool), diagonal=1)
        y = self._transformer(h, mask=mask)[:, -1]
        delta = self._out(y)
        return z0 + delta if self.residual_prediction else delta


class ChunkedLatentActionResidualFlowWorldModel(ChunkedLatentActionWorldModel):
    """Latent-action chunk MLP endpoint plus a residual flow correction."""

    chunked_residual_flow_dynamics = True

    def __init__(self, *args, residual_substeps=2, residual_integrator="heun", **kwargs):
        super().__init__(*args, **kwargs)
        self.residual_substeps = int(residual_substeps)
        self.residual_integrator = residual_integrator
        self._velocity = mlp(
            self.latent_dim * 2 + self.latent_action_dim + 1,
            self._units_cfg,
            self.latent_dim,
            last_layer=self._dynamics_last_layer,
            last_layer_kwargs=self._dynamics_last_layer_kwargs,
        )
        self._velocity.apply(weight_init)

    def dynamics_parameters(self):
        return super().dynamics_parameters() + list(self._velocity.parameters())

    def base_next_chunk(self, z, action_chunk, task):
        return super().next_chunk(z, action_chunk, task)

    def residual_velocity(self, z_context, r, latent_action, tau):
        return self._velocity(torch.cat([z_context, r, latent_action, tau], dim=-1))

    def residual_vector(self, z, action_chunk):
        u = self.latent_action(action_chunk)
        dt = 1.0 / max(1, self.residual_substeps)
        r = torch.zeros_like(z)
        for k in range(max(1, self.residual_substeps)):
            tau = torch.full((z.shape[0], 1), k * dt, device=z.device, dtype=z.dtype)
            k1 = self.residual_velocity(z, r, u, tau)
            if self.residual_integrator == "euler":
                r = r + dt * k1
            else:
                r_pred = r + dt * k1
                tau_next = torch.full((z.shape[0], 1), min(1.0, (k + 1) * dt), device=z.device, dtype=z.dtype)
                k2 = self.residual_velocity(z, r_pred, u, tau_next)
                r = r + 0.5 * dt * (k1 + k2)
        return r

    def next_chunk(self, z, action_chunk, task):
        return self.base_next_chunk(z, action_chunk, task) + self.residual_vector(z, action_chunk)


class GatedResidualWorldModel(_BaseStateWM):
    """MLP base with two learned residual experts and a soft regime gate."""

    residual_flow_dynamics = True
    gated_residual_dynamics = True

    def __init__(self, *args, num_experts=2, **kwargs):
        super().__init__()
        self._init_common(*args, **kwargs)
        self.num_experts = int(num_experts)
        self._dynamics = mlp(
            self.latent_dim + self.action_dim,
            self._units_cfg,
            self.latent_dim,
            last_layer=self._dynamics_last_layer,
            last_layer_kwargs=self._dynamics_last_layer_kwargs,
        )
        self._experts = nn.ModuleList(
            [
                mlp(
                    self.latent_dim + self.action_dim,
                    self._units_cfg,
                    self.latent_dim,
                    last_layer=self._dynamics_last_layer,
                    last_layer_kwargs=self._dynamics_last_layer_kwargs,
                )
                for _ in range(self.num_experts)
            ]
        )
        self._gate = mlp(self.latent_dim + self.action_dim, [128], self.num_experts, last_layer="linear", last_layer_kwargs={})
        self.apply(weight_init)
        zero_([self._reward[-1].weight])

    def dynamics_parameters(self):
        params = list(self._dynamics.parameters()) + list(self._gate.parameters())
        for expert in self._experts:
            params += list(expert.parameters())
        return params

    def base_next(self, z, a, task):
        return self._dynamics(torch.cat([z, a], dim=-1))

    def gate_probs(self, z, a):
        return F.softmax(self._gate(torch.cat([z, a], dim=-1)), dim=-1)

    def residual_vector(self, z, a, task=None):
        x = torch.cat([z, a], dim=-1)
        experts = torch.stack([expert(x) for expert in self._experts], dim=0)
        gate = self.gate_probs(z, a).transpose(0, 1).unsqueeze(-1)
        return (gate * experts).sum(dim=0)

    def next(self, z, a, task):
        return self.base_next(z, a, task) + self.residual_vector(z, a, task)


class GatedResidualFlowWorldModel(GatedResidualWorldModel):
    """Gated residual model whose experts are short residual flow fields."""

    def __init__(self, *args, residual_substeps=2, residual_integrator="heun", **kwargs):
        super().__init__(*args, **kwargs)
        self.residual_substeps = int(residual_substeps)
        self.residual_integrator = residual_integrator
        self._experts = nn.ModuleList(
            [
                mlp(
                    self.latent_dim * 2 + self.action_dim + 1,
                    self._units_cfg,
                    self.latent_dim,
                    last_layer=self._dynamics_last_layer,
                    last_layer_kwargs=self._dynamics_last_layer_kwargs,
                )
                for _ in range(self.num_experts)
            ]
        )
        for expert in self._experts:
            expert.apply(weight_init)

    def _expert_flow(self, expert, z, a):
        dt = 1.0 / max(1, self.residual_substeps)
        r = torch.zeros_like(z)
        for k in range(max(1, self.residual_substeps)):
            tau = torch.full((z.shape[0], 1), k * dt, device=z.device, dtype=z.dtype)
            x = torch.cat([z, r, a, tau], dim=-1)
            k1 = expert(x)
            if self.residual_integrator == "euler":
                r = r + dt * k1
            else:
                r_pred = r + dt * k1
                tau_next = torch.full((z.shape[0], 1), min(1.0, (k + 1) * dt), device=z.device, dtype=z.dtype)
                k2 = expert(torch.cat([z, r_pred, a, tau_next], dim=-1))
                r = r + 0.5 * dt * (k1 + k2)
        return r

    def residual_vector(self, z, a, task=None):
        experts = torch.stack([self._expert_flow(expert, z, a) for expert in self._experts], dim=0)
        gate = self.gate_probs(z, a).transpose(0, 1).unsqueeze(-1)
        return (gate * experts).sum(dim=0)
