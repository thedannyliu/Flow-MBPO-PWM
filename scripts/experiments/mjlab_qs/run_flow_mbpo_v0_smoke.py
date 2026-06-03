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
from types import SimpleNamespace
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
from scripts.experiments.mjlab_qs.add_flow_mbpo_support_penalty import (  # noqa: E402
    load_norm as load_support_norm,
    nearest_l2_per_dim,
    real_features,
    select_indices as select_support_indices,
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
    p.add_argument(
        "--support-risk-termination",
        action="store_true",
        help="Stop synthetic rollout branches when generated state/command/action leaves calibrated real-data support.",
    )
    p.add_argument("--support-max-rows", type=int, default=20000)
    p.add_argument("--support-probe-rows", type=int, default=4096)
    p.add_argument("--support-threshold", type=float, default=-1.0)
    p.add_argument("--support-threshold-quantile", type=float, default=0.90)
    p.add_argument("--support-distance-batch-size", type=int, default=256)
    p.add_argument("--support-state-weight", type=float, default=1.0)
    p.add_argument("--support-command-weight", type=float, default=1.0)
    p.add_argument("--support-action-weight", type=float, default=1.0)
    p.add_argument("--wandb-project", default="")
    p.add_argument("--wandb-group", default="")
    p.add_argument("--wandb-name", default="")
    p.add_argument("--enable-wandb", action="store_true")
    p.add_argument("--notes", default="")
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


def tensor_shapes(buffer: dict[str, torch.Tensor]) -> dict[str, list[int]]:
    return {key: list(value.shape) for key, value in buffer.items() if isinstance(value, torch.Tensor)}


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


def normalize_command(c_raw: torch.Tensor, nrm: dict[str, torch.Tensor]) -> torch.Tensor:
    if c_raw.shape[-1] and "command_mean" in nrm:
        return norm(c_raw, nrm["command_mean"], nrm["command_std"])
    return c_raw


def support_args(args: argparse.Namespace) -> SimpleNamespace:
    return SimpleNamespace(
        split=args.split,
        quality_filter=args.quality_filter,
        state_weight=float(args.support_state_weight),
        command_weight=float(args.support_command_weight),
        action_weight=float(args.support_action_weight),
    )


def build_support_state(
    data: dict[str, torch.Tensor],
    metadata: dict[str, Any],
    normalization: str,
    args: argparse.Namespace,
    device: torch.device,
) -> dict[str, Any] | None:
    if not bool(args.support_risk_termination):
        return None
    sargs = support_args(args)
    selected = select_support_indices(data, metadata, sargs)
    generator = torch.Generator().manual_seed(int(args.seed))
    perm = selected[torch.randperm(selected.numel(), generator=generator)]
    support_n = min(int(args.support_max_rows), int(perm.numel()))
    probe_n = min(int(args.support_probe_rows), max(0, int(perm.numel()) - support_n))
    support_indices = perm[:support_n]
    probe_indices = perm[support_n : support_n + probe_n]
    if support_indices.numel() == 0:
        raise ValueError("Need a non-empty support set")
    if float(args.support_threshold) < 0.0 and probe_indices.numel() == 0:
        raise ValueError("Need non-empty support probe rows when --support-threshold is not set")
    support_nrm = load_support_norm(Path(normalization))
    support_feat = real_features(data, support_indices, support_nrm, sargs)
    if float(args.support_threshold) >= 0.0:
        threshold = torch.tensor(float(args.support_threshold), dtype=torch.float32)
        probe_distance = torch.empty(0, dtype=torch.float32)
    else:
        probe_feat = real_features(data, probe_indices, support_nrm, sargs)
        probe_distance = nearest_l2_per_dim(probe_feat, support_feat, int(args.support_distance_batch_size))
        q = min(max(float(args.support_threshold_quantile), 0.0), 1.0)
        threshold = torch.quantile(probe_distance[torch.isfinite(probe_distance)], q)
    return {
        "support_feat": support_feat.to(device),
        "threshold": threshold.to(device),
        "probe_distance": probe_distance,
        "support_rows": int(support_indices.numel()),
        "probe_rows": int(probe_indices.numel()),
        "feature_dim": int(support_feat.shape[-1]),
    }


def support_distance_for_batch(
    state: torch.Tensor,
    command: torch.Tensor,
    action: torch.Tensor,
    support_state: dict[str, Any],
    args: argparse.Namespace,
) -> torch.Tensor:
    query = torch.cat(
        [
            state.float() * float(args.support_state_weight),
            command.float() * float(args.support_command_weight),
            action.float() * float(args.support_action_weight),
        ],
        dim=-1,
    )
    support = support_state["support_feat"]
    denom = math.sqrt(float(query.shape[-1]))
    return torch.cdist(query, support, p=2).min(dim=1).values / denom


@torch.no_grad()
def generate_synthetic_buffer(
    data: dict[str, torch.Tensor],
    nrm: dict[str, torch.Tensor],
    actor: nn.Module,
    models: list[nn.Module],
    ids: torch.Tensor,
    horizon: int,
    device: torch.device,
    support_state: dict[str, Any] | None = None,
    args: argparse.Namespace | None = None,
) -> dict[str, torch.Tensor]:
    phys = data["phys_obs"][ids].to(device).float()
    commands = data["command"][ids].to(device).float()
    z = norm(phys[:, 0], nrm["phys_obs_mean"], nrm["phys_obs_std"])
    stopped = torch.zeros(z.shape[0], dtype=torch.bool, device=device)
    horizon = min(int(horizon), int(commands.shape[1]))
    rows: dict[str, list[torch.Tensor]] = {
        "start_index": [],
        "horizon_step": [],
        "rollout_active": [],
        "state": [],
        "command": [],
        "action": [],
        "reward": [],
        "next_state": [],
        "next_state_uncertainty": [],
        "reward_uncertainty": [],
        "done_probability": [],
        "support_risk_distance": [],
        "support_risk_threshold": [],
        "support_risk_done": [],
        "done": [],
    }
    ids_device = ids.to(device)
    stop_branches = support_state is not None
    for h in range(horizon):
        active = ~stopped
        c_raw = commands[:, h]
        c = normalize_command(c_raw, nrm)
        chunk_models = all(hasattr(model, "next_trajectory") for model in models)
        if chunk_models:
            chunk_size = int(getattr(models[0], "chunk_size"))
            command_chunk = []
            action_chunk = []
            for offset in range(chunk_size):
                idx = min(h + offset, int(commands.shape[1]) - 1)
                c_future = normalize_command(commands[:, idx], nrm)
                command_chunk.append(c_future)
                action_chunk.append(actor(z, c_future, deterministic=True).clamp(-1.0, 1.0))
            command_chunk_t = torch.stack(command_chunk, dim=1)
            action_chunk_t = torch.stack(action_chunk, dim=1)
            action = action_chunk_t[:, 0]
            next_items = []
            reward_items = []
            done_prob_preds = []
            for model in models:
                pred_states, pred_rewards, done_logits = model.next_trajectory(z, action_chunk_t, command_chunk_t)
                next_items.append(pred_states[:, 0])
                reward_items.append(pred_rewards[:, 0])
                done_prob_preds.append(torch.sigmoid(done_logits[:, 0]))
            next_preds = torch.stack(next_items, dim=0)
            reward_preds = torch.stack(reward_items, dim=0)
        else:
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
        if support_state is not None:
            if args is None:
                raise ValueError("args is required when support_state is provided")
            support_distance = support_distance_for_batch(z, c, action, support_state, args)
            support_threshold = support_state["threshold"].repeat(z.shape[0])
            support_done = support_distance > support_state["threshold"]
        else:
            support_distance = torch.zeros(z.shape[0], device=device)
            support_threshold = torch.full((z.shape[0],), float("inf"), device=device)
            support_done = torch.zeros(z.shape[0], dtype=torch.bool, device=device)
        if stop_branches:
            done = (~active) | done_model | support_done
            next_z = torch.where(done.unsqueeze(-1), z, next_mean)
            stopped = stopped | done_model | support_done
        else:
            done = done_model
            next_z = next_mean
        rows["start_index"].append(ids_device)
        rows["horizon_step"].append(torch.full_like(ids_device, h))
        rows["rollout_active"].append(active.detach())
        rows["state"].append(z.detach())
        rows["command"].append(c.detach())
        rows["action"].append(action.detach())
        rows["reward"].append(reward_mean.detach())
        rows["next_state"].append(next_z.detach())
        rows["next_state_uncertainty"].append(next_uncertainty.detach())
        rows["reward_uncertainty"].append(reward_uncertainty.detach())
        rows["done_probability"].append(done_probability_mean.detach())
        rows["support_risk_distance"].append(support_distance.detach())
        rows["support_risk_threshold"].append(support_threshold.detach())
        rows["support_risk_done"].append(support_done.detach())
        rows["done"].append(done.detach())
        z = next_z
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
    support_state = build_support_state(data, metadata, args.normalization, args, device)
    buffer = generate_synthetic_buffer(data, nrm, actor, models, ids, args.horizon, device, support_state, args)
    buffer_path = output_dir / "synthetic_buffer.pt"
    torch.save(buffer, buffer_path)
    summary = {
        "git_sha": git_sha(),
        "git_branch": git_branch(),
        "command": command_line(),
        "notes": args.notes,
        "enable_wandb": bool(args.enable_wandb),
        "wandb_project": args.wandb_project,
        "wandb_group": args.wandb_group,
        "wandb_name": args.wandb_name,
        "wandb_run_id": "",
        "wandb_run_url": "",
        "dataset": args.dataset,
        "metadata": args.metadata,
        "normalization": args.normalization,
        "policy_checkpoint": args.policy_checkpoint,
        "wm_checkpoints": args.wm_checkpoint,
        "synthetic_buffer_path": str(buffer_path),
        "synthetic_buffer_metadata_path": str(output_dir / "synthetic_buffer_metadata.json"),
        "wm_method": methods[0],
        "wm_model_count": len(models),
        "uncertainty_defined": len(models) > 1,
        "seed": args.seed,
        "num_starts": int(ids.numel()),
        "horizon": int(args.horizon),
        "transitions": int(buffer["reward"].numel()),
        "split": args.split,
        "quality_filter": args.quality_filter,
        "support_risk_termination": bool(args.support_risk_termination),
        "support_rows": int(support_state["support_rows"]) if support_state is not None else 0,
        "support_probe_rows": int(support_state["probe_rows"]) if support_state is not None else 0,
        "support_feature_dim": int(support_state["feature_dim"]) if support_state is not None else 0,
        "support_threshold": float(support_state["threshold"].item()) if support_state is not None else None,
        "support_threshold_quantile": float(args.support_threshold_quantile),
        "support_state_weight": float(args.support_state_weight),
        "support_command_weight": float(args.support_command_weight),
        "support_action_weight": float(args.support_action_weight),
        "support_probe_distance": summarize_tensor(support_state["probe_distance"])
        if support_state is not None and support_state["probe_distance"].numel() > 0
        else None,
        "reward_is_normalized": True,
        "synthetic_reward": summarize_tensor(buffer["reward"]),
        "next_state_uncertainty": summarize_tensor(buffer["next_state_uncertainty"]),
        "reward_uncertainty": summarize_tensor(buffer["reward_uncertainty"]),
        "done_probability": summarize_tensor(buffer["done_probability"]),
        "support_risk_distance": summarize_tensor(buffer["support_risk_distance"]),
        "support_risk_done_fraction": float(buffer["support_risk_done"].float().mean().item()),
        "rollout_active_fraction": float(buffer["rollout_active"].float().mean().item()),
        "action_l2": summarize_tensor(buffer["action"].pow(2).mean(dim=-1).sqrt()),
        "next_state_delta_l2": summarize_tensor((buffer["next_state"] - buffer["state"]).pow(2).mean(dim=-1).sqrt()),
        "predicted_done_fraction": float(buffer["done"].float().mean().item()),
        "wall_clock_seconds": time.time() - t0,
    }
    buffer_metadata = {
        "artifact": "synthetic_buffer.pt",
        "artifact_path": str(buffer_path),
        "git_sha": summary["git_sha"],
        "git_branch": summary["git_branch"],
        "command": summary["command"],
        "notes": args.notes,
        "wandb_run_id": "",
        "wandb_run_url": "",
        "dataset": args.dataset,
        "metadata": args.metadata,
        "normalization": args.normalization,
        "policy_checkpoint": args.policy_checkpoint,
        "wm_checkpoints": args.wm_checkpoint,
        "wm_method": methods[0],
        "wm_model_count": len(models),
        "seed": int(args.seed),
        "split": args.split,
        "quality_filter": args.quality_filter,
        "num_starts": int(ids.numel()),
        "horizon": int(args.horizon),
        "transitions": int(buffer["reward"].numel()),
        "support_risk_termination": bool(args.support_risk_termination),
        "support_threshold": summary["support_threshold"],
        "tensor_shapes": tensor_shapes(buffer),
    }
    (output_dir / "synthetic_buffer_metadata.json").write_text(
        json.dumps(buffer_metadata, indent=2),
        encoding="utf-8",
    )
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    if args.enable_wandb:
        import wandb

        wandb_init = getattr(wandb, "init", None)
        if wandb_init is None:
            raise RuntimeError("W&B logging requested but the imported wandb module has no init()")
        run = wandb_init(
            project=args.wandb_project or "flow-mbpo-mjlab-flow-mbpo-v0-smoke",
            group=args.wandb_group or f"{methods[0]}_h{args.horizon}",
            name=args.wandb_name or f"{methods[0]}_seed{args.seed}_smoke",
            job_type="flow_mbpo_v0_smoke",
            config=summary,
        )
        summary["wandb_run_id"] = str(getattr(run, "id", "") or "")
        summary["wandb_run_url"] = str(getattr(run, "url", "") or "")
        buffer_metadata["wandb_run_id"] = summary["wandb_run_id"]
        buffer_metadata["wandb_run_url"] = summary["wandb_run_url"]
        (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        (output_dir / "synthetic_buffer_metadata.json").write_text(
            json.dumps(buffer_metadata, indent=2),
            encoding="utf-8",
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
