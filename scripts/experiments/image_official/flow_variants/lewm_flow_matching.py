"""LeWorldModel flow-matching ODE predictor variants."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class ODEARPredictor(nn.Module):
    """Autoregressive velocity field with time conditioning."""

    flow_matching_ode_predictor = True

    def __init__(
        self,
        *,
        num_frames,
        depth,
        heads,
        mlp_dim,
        input_dim,
        hidden_dim,
        output_dim=None,
        dim_head=64,
        dropout=0.0,
        emb_dropout=0.0,
    ):
        super().__init__()
        from module import Transformer, ConditionalBlock

        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim or input_dim)
        self.pos_embedding = nn.Parameter(torch.randn(1, num_frames, input_dim))
        self.tau_proj = nn.Sequential(nn.Linear(1, input_dim), nn.SiLU(), nn.Linear(input_dim, input_dim))
        self.dropout = nn.Dropout(emb_dropout)
        self.transformer = Transformer(
            input_dim,
            hidden_dim,
            self.output_dim,
            depth,
            heads,
            dim_head,
            mlp_dim,
            dropout,
            block_class=ConditionalBlock,
        )

    def forward(self, x: torch.Tensor, c: torch.Tensor, tau: torch.Tensor | None = None) -> torch.Tensor:
        if tau is None:
            tau = torch.zeros(*x.shape[:-1], 1, device=x.device, dtype=x.dtype)
        while tau.ndim < x.ndim:
            tau = tau.unsqueeze(-1)
        tau = tau.expand(*x.shape[:-1], 1)
        t = x.size(1)
        x = x + self.pos_embedding[:, :t] + self.tau_proj(tau)
        x = self.dropout(x)
        return self.transformer(x, c)


class FlowMatchingJEPA(nn.Module):
    """JEPA model whose predictor is an ODE-integrated velocity field."""

    flow_matching_ode_model = True

    def __init__(
        self,
        encoder,
        predictor,
        action_encoder,
        projector=None,
        pred_proj=None,
        ode_substeps: int = 4,
        ode_integrator: str = "heun",
    ):
        super().__init__()
        self.encoder = encoder
        self.predictor = predictor
        self.action_encoder = action_encoder
        self.projector = projector or nn.Identity()
        self.pred_proj = pred_proj or nn.Identity()
        self.ode_substeps = int(ode_substeps)
        self.ode_integrator = ode_integrator

    def encode(self, info):
        pixels = info["pixels"].float()
        b = pixels.size(0)
        pixels = rearrange(pixels, "b t ... -> (b t) ...")
        output = self.encoder(pixels, interpolate_pos_encoding=True)
        pixels_emb = output.last_hidden_state[:, 0]
        emb = self.projector(pixels_emb)
        info["emb"] = rearrange(emb, "(b t) d -> b t d", b=b)
        if "action" in info:
            info["act_emb"] = self.action_encoder(info["action"])
        return info

    def _apply_pred_proj(self, x: torch.Tensor) -> torch.Tensor:
        b = x.size(0)
        x = self.pred_proj(rearrange(x, "b t d -> (b t) d"))
        return rearrange(x, "(b t) d -> b t d", b=b)

    def _velocity(self, emb: torch.Tensor, act_emb: torch.Tensor, tau: torch.Tensor) -> torch.Tensor:
        return self.predictor(emb, act_emb, tau=tau)

    def predict(self, emb, act_emb):
        z = emb
        dt = 1.0 / max(1, self.ode_substeps)
        for step in range(max(1, self.ode_substeps)):
            tau = torch.full((*z.shape[:-1], 1), step * dt, device=z.device, dtype=z.dtype)
            if self.ode_integrator == "euler":
                z = z + dt * self._velocity(z, act_emb, tau)
            elif self.ode_integrator == "heun":
                v0 = self._velocity(z, act_emb, tau)
                z_euler = z + dt * v0
                tau_next = torch.full((*z.shape[:-1], 1), (step + 1) * dt, device=z.device, dtype=z.dtype)
                v1 = self._velocity(z_euler, act_emb, tau_next)
                z = z + 0.5 * dt * (v0 + v1)
            else:
                raise ValueError(f"unknown integrator: {self.ode_integrator}")
        return self._apply_pred_proj(z)

    def flow_matching_loss(self, emb: torch.Tensor, act_emb: torch.Tensor, target_emb: torch.Tensor) -> torch.Tensor:
        tau = torch.rand(*emb.shape[:-1], 1, device=emb.device, dtype=emb.dtype)
        z_tau = (1.0 - tau) * emb + tau * target_emb.detach()
        target_velocity = target_emb.detach() - emb.detach()
        pred_velocity = self._velocity(z_tau, act_emb, tau)
        return F.mse_loss(pred_velocity, target_velocity)
