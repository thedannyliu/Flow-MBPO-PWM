#!/usr/bin/env python3
"""Score real rollout steps against the expert/noisy real-data support set."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from add_flow_mbpo_support_penalty import (  # noqa: E402
    command_line,
    git_branch,
    git_sha,
    load_norm,
    nearest_l2_per_dim,
    real_features,
    select_indices,
    summarize_tensor,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--rollout-support-features", required=True)
    p.add_argument("--dataset", required=True)
    p.add_argument("--metadata", required=True)
    p.add_argument("--normalization", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--support-max-rows", type=int, default=20000)
    p.add_argument("--support-probe-rows", type=int, default=4096)
    p.add_argument("--support-threshold", type=float, default=-1.0)
    p.add_argument("--support-threshold-quantile", type=float, default=0.90)
    p.add_argument("--distance-batch-size", type=int, default=256)
    p.add_argument("--split", default="train", choices=["train", "val", "test"])
    p.add_argument("--quality-filter", default="expert,expert_noisy")
    p.add_argument("--state-weight", type=float, default=1.0)
    p.add_argument("--command-weight", type=float, default=1.0)
    p.add_argument("--action-weight", type=float, default=1.0)
    p.add_argument("--lambda-support", type=float, default=1.0)
    p.add_argument("--use-raw-action", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def require_tensor(data: dict[str, Any], key: str) -> torch.Tensor:
    value = data.get(key)
    if not torch.is_tensor(value):
        raise ValueError(f"Rollout support features missing tensor key {key!r}")
    return value.cpu()


def rollout_features(rollout: dict[str, Any], args: argparse.Namespace) -> torch.Tensor:
    action_key = "raw_action" if args.use_raw_action else "action"
    parts = [
        require_tensor(rollout, "state").float() * float(args.state_weight),
        require_tensor(rollout, "command").float() * float(args.command_weight),
        require_tensor(rollout, action_key).float() * float(args.action_weight),
    ]
    rows = {part.shape[0] for part in parts}
    if len(rows) != 1:
        raise ValueError(f"Feature row mismatch: {[tuple(part.shape) for part in parts]}")
    return torch.cat(parts, dim=-1).contiguous()


def scalar_at(x: torch.Tensor, idx: int, default: float | int | bool = 0.0) -> Any:
    if x.numel() <= idx:
        return default
    value = x[idx]
    if value.dtype == torch.bool:
        return bool(value.item())
    if value.dtype.is_floating_point:
        return float(value.item())
    return int(value.item())


def finite_quantile(values: torch.Tensor, q: float) -> float:
    values = values.detach().float().reshape(-1)
    finite = values[torch.isfinite(values)]
    if finite.numel() == 0:
        return math.nan
    return float(torch.quantile(finite, min(max(float(q), 0.0), 1.0)).item())


def finite_mean(values: torch.Tensor) -> float:
    values = values.detach().float().reshape(-1)
    finite = values[torch.isfinite(values)]
    if finite.numel() == 0:
        return math.nan
    return float(finite.mean().item())


def finite_max(values: torch.Tensor) -> float:
    values = values.detach().float().reshape(-1)
    finite = values[torch.isfinite(values)]
    if finite.numel() == 0:
        return math.nan
    return float(finite.max().item())


def write_step_csv(
    path: Path,
    rollout: dict[str, Any],
    support_distance: torch.Tensor,
    support_penalty: torch.Tensor,
    lambda_support: float,
) -> None:
    reward = require_tensor(rollout, "reward").float()
    episode_slot = require_tensor(rollout, "episode_slot").long()
    step = require_tensor(rollout, "step").long()
    done = require_tensor(rollout, "done").bool()
    terminated = require_tensor(rollout, "terminated").bool()
    truncated = require_tensor(rollout, "truncated").bool()
    action = require_tensor(rollout, "action").float()
    raw_action = rollout.get("raw_action")
    if torch.is_tensor(raw_action):
        raw_action = raw_action.float().cpu()
    else:
        raw_action = action
    action_l2 = action.norm(dim=-1)
    raw_action_l2 = raw_action.norm(dim=-1)

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "episode_slot",
                "step",
                "reward",
                "done",
                "terminated",
                "truncated",
                "action_l2",
                "raw_action_l2",
                "support_distance",
                "support_penalty",
                "support_reward_penalty",
            ],
        )
        writer.writeheader()
        for idx in range(support_distance.shape[0]):
            writer.writerow(
                {
                    "episode_slot": int(episode_slot[idx].item()),
                    "step": int(step[idx].item()),
                    "reward": float(reward[idx].item()),
                    "done": bool(done[idx].item()),
                    "terminated": bool(terminated[idx].item()),
                    "truncated": bool(truncated[idx].item()),
                    "action_l2": float(action_l2[idx].item()),
                    "raw_action_l2": float(raw_action_l2[idx].item()),
                    "support_distance": float(support_distance[idx].item()),
                    "support_penalty": float(support_penalty[idx].item()),
                    "support_reward_penalty": float(lambda_support * support_penalty[idx].item()),
                }
            )


def episode_summaries(
    rollout: dict[str, Any],
    support_distance: torch.Tensor,
    support_penalty: torch.Tensor,
    lambda_support: float,
) -> list[dict[str, Any]]:
    reward = require_tensor(rollout, "reward").float()
    episode_slot = require_tensor(rollout, "episode_slot").long()
    step = require_tensor(rollout, "step").long()
    done = require_tensor(rollout, "done").bool()
    terminated = require_tensor(rollout, "terminated").bool()
    truncated = require_tensor(rollout, "truncated").bool()

    rows: list[dict[str, Any]] = []
    for slot in torch.unique(episode_slot).tolist():
        mask = episode_slot == int(slot)
        idx = mask.nonzero(as_tuple=False).reshape(-1)
        if idx.numel() == 0:
            continue
        dist = support_distance[idx]
        penalty = support_penalty[idx]
        tail_n = min(10, int(idx.numel()))
        tail_dist = dist[-tail_n:]
        tail_penalty = penalty[-tail_n:]
        rows.append(
            {
                "episode_slot": int(slot),
                "length": int(idx.numel()),
                "return": float(reward[idx].sum().item()),
                "last_step": int(step[idx[-1]].item()),
                "done": bool(done[idx].any().item()),
                "terminated": bool(terminated[idx].any().item()),
                "truncated": bool(truncated[idx].any().item()),
                "support_distance_mean": finite_mean(dist),
                "support_distance_p90": finite_quantile(dist, 0.90),
                "support_distance_max": finite_max(dist),
                "support_distance_last": float(dist[-1].item()),
                "support_distance_tail10_mean": finite_mean(tail_dist),
                "support_distance_tail10_max": finite_max(tail_dist),
                "support_penalty_mean": finite_mean(penalty),
                "support_penalty_p90": finite_quantile(penalty, 0.90),
                "support_penalty_max": finite_max(penalty),
                "support_penalty_last": float(penalty[-1].item()),
                "support_penalty_tail10_mean": finite_mean(tail_penalty),
                "support_penalty_tail10_max": finite_max(tail_penalty),
                "support_reward_penalty_sum": float(lambda_support * penalty.sum().item()),
            }
        )
    return rows


def write_episode_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "episode_slot",
        "length",
        "return",
        "last_step",
        "done",
        "terminated",
        "truncated",
        "support_distance_mean",
        "support_distance_p90",
        "support_distance_max",
        "support_distance_last",
        "support_distance_tail10_mean",
        "support_distance_tail10_max",
        "support_penalty_mean",
        "support_penalty_p90",
        "support_penalty_max",
        "support_penalty_last",
        "support_penalty_tail10_mean",
        "support_penalty_tail10_max",
        "support_reward_penalty_sum",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def grouped_episode_stats(rows: list[dict[str, Any]], key: str) -> dict[str, Any]:
    group = [row for row in rows if bool(row[key])]
    if not group:
        return {"episodes": 0}
    lengths = torch.tensor([row["length"] for row in group], dtype=torch.float32)
    returns = torch.tensor([row["return"] for row in group], dtype=torch.float32)
    dist = torch.tensor([row["support_distance_max"] for row in group], dtype=torch.float32)
    penalty = torch.tensor([row["support_penalty_max"] for row in group], dtype=torch.float32)
    return {
        "episodes": len(group),
        "length_mean": finite_mean(lengths),
        "return_mean": finite_mean(returns),
        "support_distance_max_mean": finite_mean(dist),
        "support_distance_max_max": finite_max(dist),
        "support_penalty_max_mean": finite_mean(penalty),
        "support_penalty_max_max": finite_max(penalty),
    }


def main() -> None:
    args = parse_args()
    t0 = time.time()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rollout = torch.load(args.rollout_support_features, map_location="cpu", weights_only=False)
    if not isinstance(rollout, dict):
        raise TypeError(f"{args.rollout_support_features} must contain a dict")
    query_feat = rollout_features(rollout, args)

    data = torch.load(args.dataset, map_location="cpu", weights_only=False)
    if not isinstance(data, dict):
        raise TypeError(f"{args.dataset} must contain a dict")
    metadata = json.loads(Path(args.metadata).read_text(encoding="utf-8"))
    nrm = load_norm(Path(args.normalization))
    selected = select_indices(data, metadata, args)
    generator = torch.Generator().manual_seed(int(args.seed))
    perm = selected[torch.randperm(selected.numel(), generator=generator)]
    support_n = min(int(args.support_max_rows), int(perm.numel()))
    probe_n = min(int(args.support_probe_rows), max(0, int(perm.numel()) - support_n))
    support_indices = perm[:support_n]
    probe_indices = perm[support_n : support_n + probe_n]
    if support_indices.numel() == 0 or probe_indices.numel() == 0:
        raise ValueError("Need non-empty support and disjoint probe sets; reduce --support-max-rows if needed")

    support_feat = real_features(data, support_indices, nrm, args)
    probe_feat = real_features(data, probe_indices, nrm, args)
    real_probe_distance = nearest_l2_per_dim(probe_feat, support_feat, int(args.distance_batch_size))
    if float(args.support_threshold) >= 0.0:
        threshold = torch.tensor(float(args.support_threshold), dtype=torch.float32)
    else:
        q = min(max(float(args.support_threshold_quantile), 0.0), 1.0)
        threshold = torch.quantile(real_probe_distance[torch.isfinite(real_probe_distance)], q)
    support_distance = nearest_l2_per_dim(query_feat, support_feat, int(args.distance_batch_size))
    support_penalty = F.relu(support_distance - threshold)

    scored = dict(rollout)
    scored["support_distance"] = support_distance
    scored["support_threshold"] = threshold.repeat(support_distance.shape[0])
    scored["support_penalty"] = support_penalty
    scored["support_lambda"] = torch.full_like(support_distance, float(args.lambda_support))
    scored_path = output_dir / "rollout_support_scores.pt"
    torch.save(scored, scored_path)

    step_csv = output_dir / "support_steps.csv"
    episode_csv = output_dir / "support_episode_summary.csv"
    write_step_csv(step_csv, rollout, support_distance, support_penalty, float(args.lambda_support))
    episode_rows = episode_summaries(rollout, support_distance, support_penalty, float(args.lambda_support))
    write_episode_csv(episode_csv, episode_rows)

    summary = {
        "git_sha": git_sha(),
        "git_branch": git_branch(),
        "command": command_line(),
        "rollout_support_features": args.rollout_support_features,
        "dataset": args.dataset,
        "metadata": args.metadata,
        "normalization": args.normalization,
        "output_dir": str(output_dir),
        "scored_features": str(scored_path),
        "support_steps_csv": str(step_csv),
        "support_episode_summary_csv": str(episode_csv),
        "split": args.split,
        "quality_filter": args.quality_filter,
        "selected_real_rows": int(selected.numel()),
        "support_rows": int(support_indices.numel()),
        "support_probe_rows": int(probe_indices.numel()),
        "support_feature_dim": int(support_feat.shape[-1]),
        "rollout_rows": int(support_distance.shape[0]),
        "episode_count": int(len(episode_rows)),
        "lambda_support": float(args.lambda_support),
        "support_threshold": float(threshold.item()),
        "support_threshold_quantile": float(args.support_threshold_quantile),
        "state_weight": float(args.state_weight),
        "command_weight": float(args.command_weight),
        "action_weight": float(args.action_weight),
        "use_raw_action": bool(args.use_raw_action),
        "real_probe_distance": summarize_tensor(real_probe_distance),
        "rollout_support_distance": summarize_tensor(support_distance),
        "rollout_support_penalty": summarize_tensor(support_penalty),
        "terminated_episode_stats": grouped_episode_stats(episode_rows, "terminated"),
        "truncated_episode_stats": grouped_episode_stats(episode_rows, "truncated"),
        "done_episode_stats": grouped_episode_stats(episode_rows, "done"),
        "wall_clock_seconds": time.time() - t0,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
