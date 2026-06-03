"""Runtime NEWT flow-matching ODE dynamics patch.

The patch replaces NEWT's one-step latent dynamics head with a velocity field.
Training adds a flow-matching loss on random linear interpolants, while
``WorldModel.next`` continues to return an endpoint produced by ODE integration.
"""

from __future__ import annotations

import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from tensordict import TensorDict


def _enabled(name: str) -> bool:
    return os.environ.get(name, "0").strip().lower() in {"1", "true", "yes", "fm_ode", "flow"}


class FlowMatchingODEDynamics(nn.Module):
    """Latent velocity field with Euler/Heun integration."""

    flow_matching_ode_dynamics = True

    def __init__(
        self,
        latent_dim: int,
        condition_dim: int,
        hidden_dim: int,
        out_act: nn.Module | None = None,
        substeps: int = 4,
        integrator: str = "heun",
    ):
        super().__init__()
        self.latent_dim = int(latent_dim)
        self.condition_dim = int(condition_dim)
        self.substeps = int(substeps)
        self.integrator = integrator
        self.out_act = out_act
        self.net = nn.Sequential(
            nn.Linear(self.latent_dim + self.condition_dim + 1, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Mish(inplace=False),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Mish(inplace=False),
            nn.Linear(hidden_dim, self.latent_dim),
        )

    def _velocity(self, z: torch.Tensor, cond: torch.Tensor, tau: torch.Tensor) -> torch.Tensor:
        while tau.ndim < z.ndim:
            tau = tau.unsqueeze(-1)
        tau = tau.expand(*z.shape[:-1], 1)
        return self.net(torch.cat([z, cond, tau], dim=-1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = x[..., : self.latent_dim]
        cond = x[..., self.latent_dim :]
        dt = 1.0 / max(1, self.substeps)
        for step in range(max(1, self.substeps)):
            tau = torch.full((*z.shape[:-1], 1), step * dt, device=z.device, dtype=z.dtype)
            if self.integrator == "euler":
                z = z + dt * self._velocity(z, cond, tau)
            elif self.integrator == "heun":
                v0 = self._velocity(z, cond, tau)
                z_euler = z + dt * v0
                tau_next = torch.full((*z.shape[:-1], 1), (step + 1) * dt, device=z.device, dtype=z.dtype)
                v1 = self._velocity(z_euler, cond, tau_next)
                z = z + 0.5 * dt * (v0 + v1)
            else:
                raise ValueError(f"unknown integrator: {self.integrator}")
        if self.out_act is not None:
            z = self.out_act(z)
        return z

    def flow_matching_loss(self, x0: torch.Tensor, z1: torch.Tensor) -> torch.Tensor:
        z0 = x0[..., : self.latent_dim]
        cond = x0[..., self.latent_dim :]
        tau = torch.rand(*z0.shape[:-1], 1, device=z0.device, dtype=z0.dtype)
        z_tau = (1.0 - tau) * z0 + tau * z1.detach()
        target_velocity = z1.detach() - z0.detach()
        pred_velocity = self._velocity(z_tau, cond, tau)
        return F.mse_loss(pred_velocity, target_velocity)

    def __repr__(self) -> str:
        return (
            "FlowMatchingODEDynamics("
            f"latent_dim={self.latent_dim}, condition_dim={self.condition_dim}, "
            f"substeps={self.substeps}, integrator={self.integrator})"
        )


def apply_patch() -> None:
    if not _enabled("NEWT_FM_ODE_DYNAMICS"):
        return

    from common import init, layers
    from common import world_model as world_model_module
    import tdmpc2 as tdmpc2_module

    if getattr(world_model_module.WorldModel, "_fm_ode_patched", False):
        return

    original_wm_init = world_model_module.WorldModel.__init__
    original_loss_fn = tdmpc2_module.TDMPC2._loss_fn

    def patched_wm_init(self, cfg):
        original_wm_init(self, cfg)
        substeps = int(os.environ.get("NEWT_FM_ODE_SUBSTEPS", "4"))
        integrator = os.environ.get("NEWT_FM_ODE_INTEGRATOR", "heun")
        condition_dim = cfg.task_dim + cfg.action_dim
        self._dynamics = FlowMatchingODEDynamics(
            latent_dim=cfg.latent_dim,
            condition_dim=condition_dim,
            hidden_dim=cfg.mlp_dim,
            out_act=layers.SimNorm(cfg),
            substeps=substeps,
            integrator=integrator,
        )
        self._dynamics.apply(init.weight_init)
        self.fm_ode_arch = {
            "dynamics_arch": "fm_ode",
            "substeps": substeps,
            "integrator": integrator,
        }
        print(
            "[fm_ode] NEWT dynamics architecture=fm_ode "
            f"substeps={substeps} integrator={integrator}"
        )

    def patched_loss_fn(self, obs, action, reward, task=None):
        if not getattr(self.model._dynamics, "flow_matching_ode_dynamics", False):
            return original_loss_fn(self, obs, action, reward, task)

        with torch.no_grad():
            next_z = self.model.encode(obs[1:], task)
            td_targets = self._td_target(next_z, reward, task)

        zs = torch.empty(self.cfg.horizon + 1, self.cfg.batch_size, self.cfg.latent_dim, device=self.device)
        z = self.model.encode(obs[0], task[0])
        zs[0] = z
        endpoint_consistency_loss = 0
        flow_matching_loss = 0
        for t, (_action, _next_z, _task) in enumerate(zip(action.unbind(0), next_z.unbind(0), task.unbind(0))):
            cond_z = self.model.task_emb(z, _task)
            x0 = torch.cat([cond_z, _action], dim=-1)
            flow_matching_loss = flow_matching_loss + self.model._dynamics.flow_matching_loss(x0, _next_z) * self.rho[t]
            z = self.model.next(z, _action, _task)
            endpoint_consistency_loss = endpoint_consistency_loss + F.mse_loss(z, _next_z) * self.rho[t]
            zs[t + 1] = z

        flow_weight = float(os.environ.get("NEWT_FM_LOSS_WEIGHT", "1.0"))
        endpoint_weight = float(os.environ.get("NEWT_FM_ENDPOINT_WEIGHT", "1.0"))
        consistency_loss = endpoint_weight * endpoint_consistency_loss + flow_weight * flow_matching_loss

        _zs = zs[:-1]
        qs = self.model.Q(_zs, action, task, return_type="all")
        reward_preds = self.model.reward(_zs, action, task)

        reward_loss, value_loss = 0, 0
        for t, (rew_pred_unbind, rew_unbind, td_targets_unbind, qs_unbind) in enumerate(
            zip(reward_preds.unbind(0), reward.unbind(0), td_targets.unbind(0), qs.unbind(1))
        ):
            reward_loss = reward_loss + self.rho[t] * tdmpc2_module.math.soft_ce(
                rew_pred_unbind, rew_unbind, self.cfg
            ).mean()
            for qs_unbind_unbind in qs_unbind.unbind(0):
                value_loss = value_loss + self.rho[t] * tdmpc2_module.math.soft_ce(
                    qs_unbind_unbind, td_targets_unbind, self.cfg
                ).mean()
        value_loss = value_loss / self.cfg.num_q

        total_loss = (
            self.cfg.consistency_coef * consistency_loss
            + self.cfg.reward_coef * reward_loss
            + self.cfg.value_coef * value_loss
        )

        info = TensorDict(
            {
                "consistency_loss": consistency_loss,
                "endpoint_consistency_loss": endpoint_consistency_loss,
                "flow_matching_loss": flow_matching_loss,
                "reward_loss": reward_loss,
                "value_loss": value_loss,
                "total_loss": total_loss,
            }
        )
        return total_loss, zs.detach(), info.detach()

    world_model_module.WorldModel.__init__ = patched_wm_init
    world_model_module.WorldModel._fm_ode_patched = True
    tdmpc2_module.TDMPC2._loss_fn = patched_loss_fn
    tdmpc2_module.TDMPC2._fm_ode_patched = True


apply_patch()

