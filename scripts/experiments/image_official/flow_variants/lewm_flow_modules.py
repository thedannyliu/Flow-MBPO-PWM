"""LeWorldModel flow modules for architecture 2x2 probes."""

from __future__ import annotations

import torch
import torch.nn as nn


class FlowResidualBlock(nn.Module):
    def __init__(self, dim: int, hidden_mult: int = 4):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_mult * dim),
            nn.GELU(),
            nn.Linear(hidden_mult * dim, dim),
        )
        self.gate = nn.Parameter(torch.zeros(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + torch.tanh(self.gate) * self.net(x)


class FlowActionEmbedder(nn.Module):
    """Action embedder with residual-flow refinement in embedding space."""

    def __init__(self, input_dim=10, smoothed_dim=10, emb_dim=10, mlp_scale=4, flow_steps=2):
        super().__init__()
        from module import Embedder

        self.base = Embedder(
            input_dim=input_dim,
            smoothed_dim=smoothed_dim,
            emb_dim=emb_dim,
            mlp_scale=mlp_scale,
        )
        self.blocks = nn.ModuleList([FlowResidualBlock(emb_dim) for _ in range(max(1, int(flow_steps)))])
        self.flow_steps = int(flow_steps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.base(x)
        for block in self.blocks:
            h = block(h)
        return h


class FlowARPredictor(nn.Module):
    """ARPredictor plus residual-flow refinement on predicted embeddings."""

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
        flow_steps=2,
    ):
        super().__init__()
        from module import ARPredictor

        out_dim = output_dim or input_dim
        self.base = ARPredictor(
            num_frames=num_frames,
            depth=depth,
            heads=heads,
            mlp_dim=mlp_dim,
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=out_dim,
            dim_head=dim_head,
            dropout=dropout,
            emb_dropout=emb_dropout,
        )
        self.blocks = nn.ModuleList([FlowResidualBlock(out_dim) for _ in range(max(1, int(flow_steps)))])
        self.flow_steps = int(flow_steps)

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        h = self.base(x, c)
        for block in self.blocks:
            h = block(h)
        return h

