#!/usr/bin/env python3
"""Generate a small Flow-MBPO v0 synthetic rollout diagnostic buffer.

This smoke script does not update a policy. It verifies that existing BC policy
and world-model checkpoints can produce bounded short synthetic rollouts from
real MJLab-QS dataset states.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from scripts.experiments.mjlab_qs.render_policy_rollout import (  # noqa: E402
    build_actor,
    command_line,
    git_branch,
    git_sha,
)
from scripts.experiments.mjlab_qs.run_phaseA_wm_feasibility import (  # noqa: E402
    FlowTrajectoryChunkWM,
    FlowWM,
    MLPWM,
    ResidualFlowWM,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True)
    p.add_argument("--metadata", required=True)
    p.add_argument("--normalization", required=True)
    p.add_argument("--policy-checkpoint", required=True)
    p.add_argument("--wm-checkpoint", action="append", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--num-starts", type=int, default=256)
    p.add_argument("--horizon", type=int, default=1)
    p.add_argument("--split", default="train", choices=["train", "val", "test"])
    p.add_argument("--quality-filter", default="expert,expert_noisy")
    p.add_argument("--wandb-project", default="")
    p.add_argument("--wandb-group", default="")
    p.add_argument("--wandb-name", default="")
    p.add_argument("--enable-wandb", action="store_true")
    return p.parse_args()


def load_norm(path: Path, device: torch.device) -> dict[str, torch.Tensor]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    return {k: torch.tensor(v, dtype=torch.float32, device=device) for k, v in raw.items() if isinstance(v, list)}


def norm(x: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    return (x - mean) / std.clamp_min(1e-6)


def summarize_tensor(x: torch.Tensor) -> dict[str, float]:
    x = x.detach().float().reshape(-1).cpu()
    finite = x[torch.isfinite(x)]
    if finite.numel() == 0:
        return {"mean": math.nan, "std": math.nan, "min": math.nan, "p90": math.nan, "max": math.nan}
    return {
        "mean": float(finite.mean().item()),
        "std": float(finite.std(unbiased=False).item()),
        "min": float(finite.min().item()),
        "p90": float(torch.quantile(finite, 0.90).item()),
        "max": float(finite.max().item()),
    }


def model_from_checkpoint(path: Path, state_dim: int, action_dim: int, command_dim: int, device: torch.device) -> tuple[str, nn.Module, dict[str, Any]]:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    ckpt_args = ckpt.get("args", {})
    if not isinstance(ckpt_args, dict):
        raise TypeError(f"{path} checkpoint args must be a dict")
    method = str(ckpt_args.get("method", ""))
    hidden = int(ckpt_args.get("hidden", 512))
    flow_substeps = int(ckpt_args.get("flow_substeps", 4))
    chunk_size = int(ckpt_args.get("chunk_size", 3))
    if method == "mlp_ref":
        model: nn.Module = MLPWM(state_dim, action_dim, command_dim, hidden)
    elif method in {"flow_ref", "flow_endpoint"}:
        model = FlowWM(state_dim, action_dim, command_dim, hidden, substeps=flow_substeps)
    elif method == "residual_flow_frozen_mlp":
        model = ResidualFlowWM(state_dim, action_dim, command_dim, hidden, substeps=flow_substeps)
    elif method == "flow_trajectory_chunk":
        model = FlowTrajectoryChunkWM(
            state_dim,
            action_dim,
            command_dim,
            hidden,
            chunk_size=chunk_size,
            substeps=flow_substeps,
        )
    else:
        raise ValueError(f"Unsupported world-model method {method!r} in {path}")
    model.load_state_dict(ckpt["model"])
    model.to(device)
    model.eval()
    return method, model, ckpt_args


def select_start_indices(data: dict[str, torch.Tensor], metadata: dict[str, Any], args: argparse.Namespace) -> torch.Tensor:
    split_map = metadata["split_id_map"]
    quality_map = metadata["quality_id_map"]
    split_id = int(split_map[args.split])
    quality_names = [item.strip() for item in args.quality_filter.split(",") if item.strip()]
    quality_ids = torch.tensor([int(quality_map[name]) for name in quality_names], dtype=torch.long)
    mask = data["split_id"].long() == split_id
    mask = mask & torch.isin(data["quality_bin_id"].long(), quality_ids)
    candidates = mask.nonzero(as_tuple=False).reshape(-1)
    if candidates.numel() == 0:
        raise ValueError(f"No windows match split={args.split!r}, quality_filter={quality_names!r}")
    count = min(int(args.num_starts), int(candidates.numel()))
    pick = torch.randperm(candidates.numel())[:count]
    return candidates[pick]


@torch.no_grad()
def generate_synthetic_buffer(
    data: dict[str, torch.Tensor],
    nrm: dict[str, torch.Tensor],
    actor: nn.Module,
    models: list[nn.Module],
    ids: torch.Tensor,
    horizon: int,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    phys = data["phys_obs"][ids].to(device).float()
    commands = data["command"][ids].to(device).float()
    z = norm(phys[:, 0], nrm["phys_obs_mean"], nrm["phys_obs_std"])
    horizon = min(int(horizon), int(commands.shape[1]))
    rows: dict[str, list[torch.Tensor]] = {
        "start_index": [],
        "horizon_step": [],
        "state": [],
        "command": [],
        "action": [],
        "reward": [],
        "next_state": [],
        "next_state_uncertainty": [],
        "reward_uncertainty": [],
        "done_probability": [],
        "done": [],
    }
    ids_device = ids.to(device)
    for h in range(horizon):
        c_raw = commands[:, h]
        c = c_raw
        if c.shape[-1] and "command_mean" in nrm:
            c = norm(c, nrm["command_mean"], nrm["command_std"])
        action = actor(z, c, deterministic=True).clamp(-1.0, 1.0)
        next_preds = torch.stack([model.next(z, action) for model in models], dim=0)
        reward_preds = torch.stack([model.reward(z, action, c).reshape(z.shape[0]) for model in models], dim=0)
        done_prob_preds = []
        for model in models:
            done_probability = getattr(model, "done_probability", None)
            if done_probability is not None:
                done_prob_preds.append(done_probability(z, action, c).reshape(z.shape[0]))
        next_mean = next_preds.mean(dim=0)
        reward_mean = reward_preds.mean(dim=0)
        if done_prob_preds:
            done_probability_mean = torch.stack(done_prob_preds, dim=0).mean(dim=0)
            done_model = done_probability_mean >= 0.5
        else:
            done_probability_mean = torch.zeros(z.shape[0], device=device)
            done_model = torch.zeros(z.shape[0], dtype=torch.bool, device=device)
        if len(models) > 1:
            next_uncertainty = next_preds.var(dim=0, unbiased=False).mean(dim=-1).sqrt()
            reward_uncertainty = reward_preds.var(dim=0, unbiased=False).sqrt()
        else:
            next_uncertainty = torch.zeros(z.shape[0], device=device)
            reward_uncertainty = torch.zeros(z.shape[0], device=device)
        rows["start_index"].append(ids_device)
        rows["horizon_step"].append(torch.full_like(ids_device, h))
        rows["state"].append(z.detach())
        rows["command"].append(c.detach())
        rows["action"].append(action.detach())
        rows["reward"].append(reward_mean.detach())
        rows["next_state"].append(next_mean.detach())
        rows["next_state_uncertainty"].append(next_uncertainty.detach())
        rows["reward_uncertainty"].append(reward_uncertainty.detach())
        rows["done_probability"].append(done_probability_mean.detach())
        rows["done"].append(done_model.detach())
        z = next_mean
    return {key: torch.cat(value, dim=0).detach().cpu() for key, value in rows.items()}


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit(f"CUDA device requested ({args.device}) but torch.cuda.is_available() is false")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    complete_paths = [output_dir / "summary.json", output_dir / "synthetic_buffer.pt"]
    if all(path.exists() for path in complete_paths):
        print(f"flow-mbpo v0 smoke already complete; skipping {output_dir}", flush=True)
        return
    lock_file = (output_dir / ".flow_mbpo_v0_smoke.lock").open("w", encoding="utf-8")
    try:
        fcntl.flock(lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        print(f"flow-mbpo v0 smoke already running; skipping {output_dir}", flush=True)
        return
    if all(path.exists() for path in complete_paths):
        print(f"flow-mbpo v0 smoke already complete; skipping {output_dir}", flush=True)
        return
    t0 = time.time()

    data = torch.load(args.dataset, map_location="cpu", weights_only=False)
    metadata = json.loads(Path(args.metadata).read_text(encoding="utf-8"))
    nrm = load_norm(Path(args.normalization), device)
    state_dim = int(data["phys_obs"].shape[-1])
    action_dim = int(data["policy_action"].shape[-1])
    command_dim = int(data["command"].shape[-1])

    policy_ckpt = torch.load(args.policy_checkpoint, map_location="cpu", weights_only=False)
    actor = build_actor(policy_ckpt, state_dim, command_dim, action_dim, device)
    methods: list[str] = []
    wm_args: list[dict[str, Any]] = []
    models: list[nn.Module] = []
    for raw_path in args.wm_checkpoint:
        method, model, ckpt_args = model_from_checkpoint(Path(raw_path), state_dim, action_dim, command_dim, device)
        methods.append(method)
        wm_args.append(ckpt_args)
        models.append(model)
    if len(set(methods)) != 1:
        raise ValueError(f"All WM checkpoints must use the same method for v0 smoke uncertainty, got {methods}")

    ids = select_start_indices(data, metadata, args)
    buffer = generate_synthetic_buffer(data, nrm, actor, models, ids, args.horizon, device)
    torch.save(buffer, output_dir / "synthetic_buffer.pt")
    summary = {
        "git_sha": git_sha(),
        "git_branch": git_branch(),
        "command": command_line(),
        "dataset": args.dataset,
        "metadata": args.metadata,
        "normalization": args.normalization,
        "policy_checkpoint": args.policy_checkpoint,
        "wm_checkpoints": args.wm_checkpoint,
        "wm_method": methods[0],
        "wm_model_count": len(models),
        "uncertainty_defined": len(models) > 1,
        "seed": args.seed,
        "num_starts": int(ids.numel()),
        "horizon": int(args.horizon),
        "transitions": int(buffer["reward"].numel()),
        "split": args.split,
        "quality_filter": args.quality_filter,
        "reward_is_normalized": True,
        "synthetic_reward": summarize_tensor(buffer["reward"]),
        "next_state_uncertainty": summarize_tensor(buffer["next_state_uncertainty"]),
        "reward_uncertainty": summarize_tensor(buffer["reward_uncertainty"]),
        "done_probability": summarize_tensor(buffer["done_probability"]),
        "action_l2": summarize_tensor(buffer["action"].pow(2).mean(dim=-1).sqrt()),
        "next_state_delta_l2": summarize_tensor((buffer["next_state"] - buffer["state"]).pow(2).mean(dim=-1).sqrt()),
        "predicted_done_fraction": float(buffer["done"].float().mean().item()),
        "wall_clock_seconds": time.time() - t0,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    if args.enable_wandb:
        import wandb

        run = wandb.init(
            project=args.wandb_project or "flow-mbpo-mjlab-flow-mbpo-v0-smoke",
            group=args.wandb_group or f"{methods[0]}_h{args.horizon}",
            name=args.wandb_name or f"{methods[0]}_seed{args.seed}_smoke",
            job_type="flow_mbpo_v0_smoke",
            config=summary,
        )
        run.log({
            "synthetic/reward_mean": summary["synthetic_reward"]["mean"],
            "synthetic/next_state_uncertainty_mean": summary["next_state_uncertainty"]["mean"],
            "synthetic/reward_uncertainty_mean": summary["reward_uncertainty"]["mean"],
            "synthetic/done_probability_mean": summary["done_probability"]["mean"],
            "synthetic/action_l2_mean": summary["action_l2"]["mean"],
            "synthetic/next_state_delta_l2_mean": summary["next_state_delta_l2"]["mean"],
            "synthetic/predicted_done_fraction": summary["predicted_done_fraction"],
        })
        run.finish()
    print(json.dumps(summary, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
