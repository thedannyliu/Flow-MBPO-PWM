"""Runtime NEWT architecture patch for flow 2x2 probes.

This module is loaded through ``sitecustomize`` in Slurm jobs. It leaves the
external NEWT checkout untouched and replaces selected modules after the
official ``WorldModel`` constructor runs.
"""

from __future__ import annotations

import os

import torch
import torch.nn as nn


def _enabled(name: str) -> bool:
    return os.environ.get(name, "0").strip().lower() in {"1", "true", "yes", "flow"}


class FlowResidualBlock(nn.Module):
    """Small residual velocity block used as a flow-style replacement MLP core."""

    def __init__(self, dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, 4 * dim),
            nn.Mish(inplace=False),
            nn.Linear(4 * dim, dim),
        )
        self.gate = nn.Parameter(torch.zeros(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + torch.tanh(self.gate) * self.net(x)


class FlowMLP(nn.Module):
    """Drop-in flow-style MLP with the same input/output contract as NEWT MLPs."""

    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, act: nn.Module | None = None, steps: int = 2):
        super().__init__()
        self.in_proj = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Mish(inplace=False),
        )
        self.blocks = nn.ModuleList([FlowResidualBlock(hidden_dim) for _ in range(max(1, int(steps)))])
        self.final = nn.Linear(hidden_dim, out_dim)
        self.act = act

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.in_proj(x)
        for block in self.blocks:
            h = block(h)
        out = self.final(h)
        if self.act is not None:
            out = self.act(out)
        return out

    def __repr__(self) -> str:
        return (
            f"FlowMLP(in_features={self.in_proj[0].in_features}, "
            f"hidden_features={self.in_proj[0].out_features}, "
            f"out_features={self.final.out_features}, "
            f"steps={len(self.blocks)}, act={self.act.__class__.__name__ if self.act else None})"
        )


def apply_patch() -> None:
    flow_wm = _enabled("NEWT_FLOW_WM")
    flow_policy = _enabled("NEWT_FLOW_POLICY")
    if not flow_wm and not flow_policy:
        return

    from common import init, layers
    from common import world_model as world_model_module

    if getattr(world_model_module.WorldModel, "_flow_2x2_patched", False):
        return

    original_init = world_model_module.WorldModel.__init__

    def patched_init(self, cfg):
        original_init(self, cfg)
        flow_steps = int(os.environ.get("NEWT_FLOW_STEPS", "2"))
        wm_in_dim = cfg.latent_dim + cfg.action_dim + cfg.task_dim
        policy_in_dim = cfg.latent_dim + cfg.task_dim
        if flow_wm:
            self._dynamics = FlowMLP(
                wm_in_dim,
                cfg.mlp_dim,
                cfg.latent_dim,
                act=layers.SimNorm(cfg),
                steps=flow_steps,
            )
            self._reward = FlowMLP(
                wm_in_dim,
                cfg.mlp_dim,
                max(cfg.num_bins, 1),
                steps=flow_steps,
            )
            self._dynamics.apply(init.weight_init)
            self._reward.apply(init.weight_init)
            init.zero_(self._reward.final.weight)
            print(f"[flow_2x2] NEWT world-model architecture=flow steps={flow_steps}")
        else:
            print("[flow_2x2] NEWT world-model architecture=mlp")

        if flow_policy:
            self._pi = FlowMLP(
                policy_in_dim,
                cfg.mlp_dim,
                2 * cfg.action_dim,
                steps=flow_steps,
            )
            self._pi.apply(init.weight_init)
            print(f"[flow_2x2] NEWT policy architecture=flow steps={flow_steps}")
        else:
            print("[flow_2x2] NEWT policy architecture=mlp")

        self.flow_2x2_arch = {
            "wm_arch": "flow" if flow_wm else "mlp",
            "policy_arch": "flow" if flow_policy else "mlp",
            "flow_steps": flow_steps,
        }

    world_model_module.WorldModel.__init__ = patched_init
    world_model_module.WorldModel._flow_2x2_patched = True


apply_patch()

