#!/usr/bin/env python3
"""Standalone A2.5 world-model feasibility runner.

The runner uses normalized physical observations as a frozen reference state.
This deliberately avoids separate learned encoder confounds for A2.5.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb


class MLP(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden: int = 512, depth: int = 3):
        super().__init__()
        layers = []
        dim = in_dim
        for _ in range(depth):
            layers += [nn.Linear(dim, hidden), nn.SiLU()]
            dim = hidden
        layers.append(nn.Linear(dim, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MLPWM(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, command_dim: int, hidden: int):
        super().__init__()
        self.dyn = MLP(state_dim + action_dim, state_dim, hidden=hidden)
        self.rew = MLP(state_dim + action_dim + command_dim, 1, hidden=hidden)

    def next(self, z: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        return z + self.dyn(torch.cat([z, a], dim=-1))

    def reward(self, z: torch.Tensor, a: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        return self.rew(torch.cat([z, a, c], dim=-1)).squeeze(-1)


class FlowWM(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, command_dim: int, hidden: int, substeps: int = 4):
        super().__init__()
        self.state_dim = state_dim
        self.substeps = int(substeps)
        self.vel = MLP(state_dim + action_dim + 1, state_dim, hidden=hidden)
        self.rew = MLP(state_dim + action_dim + command_dim, 1, hidden=hidden)

    def velocity(self, z: torch.Tensor, a: torch.Tensor, tau: torch.Tensor) -> torch.Tensor:
        if tau.ndim == 0:
            tau = tau.expand(z.shape[0], 1)
        elif tau.ndim == 1:
            tau = tau[:, None]
        return self.vel(torch.cat([z, a, tau.to(z.device, z.dtype)], dim=-1))

    def fm_loss(self, z0: torch.Tensor, z1: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        tau = torch.rand(z0.shape[0], 1, device=z0.device, dtype=z0.dtype)
        ztau = (1.0 - tau) * z0 + tau * z1
        target = z1 - z0
        return F.mse_loss(self.velocity(ztau, a, tau), target)

    def next(self, z: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        dt = 1.0 / max(1, self.substeps)
        out = z
        for i in range(self.substeps):
            tau = torch.full((z.shape[0], 1), (i + 0.5) * dt, device=z.device, dtype=z.dtype)
            out = out + dt * self.velocity(out, a, tau)
        return out

    def reward(self, z: torch.Tensor, a: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        return self.rew(torch.cat([z, a, c], dim=-1)).squeeze(-1)


class ResidualFlowWM(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, command_dim: int, hidden: int, substeps: int = 4):
        super().__init__()
        self.base = MLPWM(state_dim, action_dim, command_dim, hidden)
        self.residual = FlowWM(state_dim, action_dim, command_dim, hidden, substeps=substeps)

    def next(self, z: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        base = self.base.next(z, a)
        return base + (self.residual.next(z, a) - z)

    def reward(self, z: torch.Tensor, a: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        return self.base.reward(z, a, c)

    def fm_loss(self, z0: torch.Tensor, z1: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            base = self.base.next(z0, a)
        return self.residual.fm_loss(z0, z1 - base + z0, a)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--metadata", required=True)
    parser.add_argument("--normalization", required=True)
    parser.add_argument("--method", choices=["mlp_ref", "flow_ref", "residual_flow_frozen_mlp"], required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--train-iters", type=int, default=50000)
    parser.add_argument("--base-pretrain-iters", type=int, default=50000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--eval-batch-size", type=int, default=1024)
    parser.add_argument("--eval-every", type=int, default=1000)
    parser.add_argument("--hidden", type=int, default=512)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--flow-substeps", type=int, default=4)
    parser.add_argument("--wandb-project", default="flow-mbpo-mjlab-phaseA-wm-feasibility")
    parser.add_argument("--wandb-group", default="a25_mini_feasibility")
    parser.add_argument("--wandb-name", default=None)
    parser.add_argument("--disable-wandb", action="store_true")
    return parser.parse_args()


def load_norm(path: Path, device: torch.device) -> Dict[str, torch.Tensor]:
    raw = json.loads(path.read_text())
    return {k: torch.tensor(v, dtype=torch.float32, device=device) for k, v in raw.items() if isinstance(v, list)}


def norm(x: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    return (x - mean) / std.clamp_min(1e-6)


def batch(data: Dict[str, torch.Tensor], indices: torch.Tensor, device: torch.device, nrm: Dict[str, torch.Tensor]):
    z = norm(data["phys_obs"][indices].to(device).float(), nrm["phys_obs_mean"], nrm["phys_obs_std"])
    a = data["policy_action"][indices].to(device).float()
    r = norm(data["reward"][indices].to(device).float()[..., None], nrm["reward_mean"], nrm["reward_std"]).squeeze(-1)
    c = data["command"][indices].to(device).float()
    if c.shape[-1] and "command_mean" in nrm:
        c = norm(c, nrm["command_mean"], nrm["command_std"])
    done = data["done"][indices].to(device).bool()
    return z, a, r, c, done


def sample_train_indices(data: Dict[str, torch.Tensor], train_idx: torch.Tensor, batch_size: int) -> torch.Tensor:
    qids = data["quality_bin_id"][train_idx]
    unique = torch.unique(qids)
    per = max(1, batch_size // max(1, len(unique)))
    pieces = []
    for q in unique:
        candidates = train_idx[qids == q]
        pick = candidates[torch.randint(0, candidates.numel(), (per,))]
        pieces.append(pick)
    out = torch.cat(pieces)
    if out.numel() < batch_size:
        extra = train_idx[torch.randint(0, train_idx.numel(), (batch_size - out.numel(),))]
        out = torch.cat([out, extra])
    return out[torch.randperm(out.numel())[:batch_size]]


def rollout_losses(model: nn.Module, z: torch.Tensor, a: torch.Tensor, r: torch.Tensor, c: torch.Tensor, done: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    horizon = a.shape[1]
    pred = z[:, 0]
    dyn_losses = []
    rew_losses = []
    mask = torch.ones(z.shape[0], device=z.device)
    for h in range(horizon):
        rew_pred = model.reward(pred, a[:, h], c[:, h])
        rew_losses.append(((rew_pred - r[:, h]) ** 2 * mask).sum() / mask.sum().clamp_min(1.0))
        pred = model.next(pred, a[:, h])
        err = ((pred - z[:, h + 1]) ** 2).mean(dim=-1)
        dyn_losses.append((err * mask).sum() / mask.sum().clamp_min(1.0))
        mask = mask * (~done[:, h]).float()
    return torch.stack(dyn_losses), torch.stack(rew_losses), pred


def train_loss(model: nn.Module, method: str, z: torch.Tensor, a: torch.Tensor, r: torch.Tensor, c: torch.Tensor, done: torch.Tensor) -> torch.Tensor:
    if method in {"flow_ref", "residual_flow_frozen_mlp"}:
        fm = 0.0
        for h in range(a.shape[1]):
            fm = fm + model.fm_loss(z[:, h], z[:, h + 1], a[:, h])
        fm = fm / a.shape[1]
        _, rew, _ = rollout_losses(model, z, a, r, c, done)
        return fm + rew.mean()
    dyn, rew, _ = rollout_losses(model, z, a, r, c, done)
    return dyn.mean() + rew.mean()


@torch.no_grad()
def evaluate(model: nn.Module, method: str, data: Dict[str, torch.Tensor], idx: torch.Tensor, device: torch.device, nrm: Dict[str, torch.Tensor], eval_batch_size: int) -> Dict[str, float]:
    if idx.numel() == 0:
        return {
            "one_step_dyn_mse": float("nan"),
            "rollout_dyn_mse_H16": float("nan"),
            "reward_mse": float("nan"),
            "rollout_error_e1": float("nan"),
            "rollout_error_e16": float("nan"),
            "rollout_error_ratio_e16_e1": float("nan"),
        }
    dyn_all = []
    rew_all = []
    one_all = []
    weights = []
    for start in range(0, idx.numel(), eval_batch_size):
        ids = idx[start : start + eval_batch_size]
        z, a, r, c, done = batch(data, ids, device, nrm)
        dyn, rew, _ = rollout_losses(model, z, a, r, c, done)
        dyn_all.append(dyn.detach().cpu())
        rew_all.append(rew.detach().cpu())
        one_all.append(dyn[0].detach().cpu())
        weights.append(ids.numel())
    w = torch.tensor(weights, dtype=torch.float32)
    w = w / w.sum().clamp_min(1.0)
    dyn = (torch.stack(dyn_all) * w[:, None]).sum(dim=0)
    rew = (torch.stack(rew_all) * w[:, None]).sum(dim=0)
    return {
        "one_step_dyn_mse": float((torch.stack(one_all) * w).sum().item()),
        "rollout_dyn_mse_H16": float(dyn.mean().item()),
        "reward_mse": float(rew.mean().item()),
        "rollout_error_e1": float(dyn[0].item()),
        "rollout_error_e16": float(dyn[-1].item()),
        "rollout_error_ratio_e16_e1": float((dyn[-1] / dyn[0].clamp_min(1e-8)).item()),
    }


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    data = torch.load(args.dataset, map_location="cpu", weights_only=False)
    metadata = json.loads(Path(args.metadata).read_text())
    nrm = load_norm(Path(args.normalization), device)
    train_idx = (data["split_id"] == 0).nonzero(as_tuple=False).squeeze(-1)
    val_idx = (data["split_id"] == 1).nonzero(as_tuple=False).squeeze(-1)
    test_idx = (data["split_id"] == 2).nonzero(as_tuple=False).squeeze(-1)
    state_dim = int(data["phys_obs"].shape[-1])
    action_dim = int(data["policy_action"].shape[-1])
    command_dim = int(data["command"].shape[-1])
    if args.method == "mlp_ref":
        model: nn.Module = MLPWM(state_dim, action_dim, command_dim, args.hidden)
    elif args.method == "flow_ref":
        model = FlowWM(state_dim, action_dim, command_dim, args.hidden, substeps=args.flow_substeps)
    else:
        model = ResidualFlowWM(state_dim, action_dim, command_dim, args.hidden, substeps=args.flow_substeps)
    model.to(device)
    if args.method == "residual_flow_frozen_mlp":
        base_opt = torch.optim.Adam(model.base.parameters(), lr=args.lr)
        for _ in range(args.base_pretrain_iters):
            ids = sample_train_indices(data, train_idx, args.batch_size)
            z, a, r, c, done = batch(data, ids, device, nrm)
            loss = train_loss(model.base, "mlp_ref", z, a, r, c, done)
            base_opt.zero_grad()
            loss.backward()
            base_opt.step()
        for p in model.base.parameters():
            p.requires_grad_(False)
    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=args.lr)

    run = None
    if not args.disable_wandb:
        run = wandb.init(
            project=args.wandb_project,
            group=args.wandb_group,
            name=args.wandb_name or f"{args.method}_seed{args.seed}",
            job_type="formal",
            config={**vars(args), "dataset_metadata": metadata},
        )
    best_val = math.inf
    best = None
    t0 = time.time()
    for it in range(args.train_iters + 1):
        if it > 0:
            ids = sample_train_indices(data, train_idx, args.batch_size)
            z, a, r, c, done = batch(data, ids, device, nrm)
            loss = train_loss(model, args.method, z, a, r, c, done)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 100.0)
            opt.step()
        if it % args.eval_every == 0 or it == args.train_iters:
            train_m = evaluate(model, args.method, data, train_idx[: min(train_idx.numel(), 4096)], device, nrm, args.eval_batch_size)
            val_m = evaluate(model, args.method, data, val_idx, device, nrm, args.eval_batch_size)
            metrics = {f"train/{k}": v for k, v in train_m.items()}
            metrics.update({f"val/{k}": v for k, v in val_m.items()})
            metrics["iter"] = it
            metrics["wall_clock_seconds"] = time.time() - t0
            if val_m["rollout_dyn_mse_H16"] < best_val:
                best_val = val_m["rollout_dyn_mse_H16"]
                best = {"iter": it, **metrics}
                torch.save({"model": model.state_dict(), "args": vars(args), "best": best}, output_dir / "best.pt")
            if run is not None:
                wandb.log(metrics, step=it)
            print(json.dumps(metrics, sort_keys=True))
    test_m = evaluate(model, args.method, data, test_idx, device, nrm, args.eval_batch_size)
    summary = {
        "method": args.method,
        "seed": args.seed,
        "best_val_rollout_dyn_mse_H16": best_val,
        "best_iter": best["iter"] if best else None,
        **{f"test/{k}": v for k, v in test_m.items()},
        "wall_clock_seconds": time.time() - t0,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    if run is not None:
        run.summary.update(summary)
        run.finish()


if __name__ == "__main__":
    main()
