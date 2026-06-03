#!/usr/bin/env python3
"""Add a real-data support/OOD penalty to Flow-MBPO synthetic replay."""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--synthetic-replay", required=True)
    p.add_argument("--dataset", required=True)
    p.add_argument("--metadata", required=True)
    p.add_argument("--normalization", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--lambda-support", type=float, default=0.0)
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
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def command_line() -> str:
    return " ".join(sys.argv)


def git_value(args: list[str]) -> str:
    try:
        return subprocess.check_output(["git", *args], cwd=PROJECT_ROOT, text=True).strip()
    except Exception:
        return ""


def git_sha() -> str:
    return git_value(["rev-parse", "HEAD"])


def git_branch() -> str:
    return git_value(["rev-parse", "--abbrev-ref", "HEAD"])


def summarize_tensor(x: torch.Tensor) -> dict[str, float]:
    x = x.detach().float().reshape(-1).cpu()
    finite = x[torch.isfinite(x)]
    if finite.numel() == 0:
        return {"mean": math.nan, "std": math.nan, "min": math.nan, "p50": math.nan, "p90": math.nan, "max": math.nan}
    return {
        "mean": float(finite.mean().item()),
        "std": float(finite.std(unbiased=False).item()),
        "min": float(finite.min().item()),
        "p50": float(torch.quantile(finite, 0.50).item()),
        "p90": float(torch.quantile(finite, 0.90).item()),
        "max": float(finite.max().item()),
    }


def load_norm(path: Path) -> dict[str, torch.Tensor]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    return {key: torch.tensor(value, dtype=torch.float32) for key, value in raw.items() if isinstance(value, list)}


def norm(x: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    return (x.float() - mean) / std.clamp_min(1.0e-6)


def isin(values: torch.Tensor, candidates: torch.Tensor) -> torch.Tensor:
    mask = torch.zeros(values.shape, dtype=torch.bool)
    for candidate in candidates.tolist():
        mask |= values == int(candidate)
    return mask


def select_indices(data: dict[str, torch.Tensor], metadata: dict[str, Any], args: argparse.Namespace) -> torch.Tensor:
    split_id = int(metadata["split_id_map"][args.split])
    quality_names = [item.strip() for item in args.quality_filter.split(",") if item.strip()]
    quality_ids = torch.tensor([int(metadata["quality_id_map"][name]) for name in quality_names], dtype=torch.long)
    mask = data["split_id"].long() == split_id
    mask = mask & isin(data["quality_bin_id"].long(), quality_ids)
    indices = mask.nonzero(as_tuple=False).reshape(-1)
    if indices.numel() == 0:
        raise ValueError(f"No rows match split={args.split!r}, quality_filter={quality_names!r}")
    return indices


def real_features(
    data: dict[str, torch.Tensor],
    indices: torch.Tensor,
    nrm: dict[str, torch.Tensor],
    args: argparse.Namespace,
) -> torch.Tensor:
    state = norm(data["phys_obs"][indices, 0], nrm["phys_obs_mean"], nrm["phys_obs_std"])
    command = data["command"][indices, 0].float()
    if command.shape[-1] and "command_mean" in nrm:
        command = norm(command, nrm["command_mean"], nrm["command_std"])
    action = data["policy_action"][indices, 0].float()
    parts = [
        state * float(args.state_weight),
        command * float(args.command_weight),
        action * float(args.action_weight),
    ]
    return torch.cat(parts, dim=-1).contiguous()


def replay_features(replay: dict[str, torch.Tensor], args: argparse.Namespace) -> torch.Tensor:
    parts = [
        replay["state"].float() * float(args.state_weight),
        replay["command"].float() * float(args.command_weight),
        replay["action"].float() * float(args.action_weight),
    ]
    return torch.cat(parts, dim=-1).contiguous()


def nearest_l2_per_dim(query: torch.Tensor, support: torch.Tensor, batch_size: int) -> torch.Tensor:
    if query.shape[-1] != support.shape[-1]:
        raise ValueError(f"Feature dimension mismatch: query={query.shape[-1]}, support={support.shape[-1]}")
    denom = math.sqrt(float(query.shape[-1]))
    out = []
    for start in range(0, query.shape[0], batch_size):
        chunk = query[start : start + batch_size]
        dist = torch.cdist(chunk, support, p=2).min(dim=1).values / denom
        out.append(dist.cpu())
    return torch.cat(out, dim=0)


def main() -> None:
    args = parse_args()
    t0 = time.time()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    replay = torch.load(args.synthetic_replay, map_location="cpu", weights_only=False)
    if not isinstance(replay, dict):
        raise TypeError(f"{args.synthetic_replay} must contain a dict")
    for key in ["state", "command", "action", "reward_conservative", "done"]:
        if key not in replay:
            raise ValueError(f"Synthetic replay is missing required key {key!r}")

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
    synthetic_feat = replay_features(replay, args)
    real_probe_distance = nearest_l2_per_dim(probe_feat, support_feat, int(args.distance_batch_size))
    if float(args.support_threshold) >= 0.0:
        threshold = torch.tensor(float(args.support_threshold), dtype=torch.float32)
    else:
        q = min(max(float(args.support_threshold_quantile), 0.0), 1.0)
        threshold = torch.quantile(real_probe_distance[torch.isfinite(real_probe_distance)], q)
    support_distance = nearest_l2_per_dim(synthetic_feat, support_feat, int(args.distance_batch_size))
    support_penalty = F.relu(support_distance - threshold)

    out = dict(replay)
    previous_reward = replay["reward_conservative"].float()
    out["reward_pre_support"] = previous_reward
    out["support_distance"] = support_distance
    out["support_threshold"] = threshold.repeat(support_distance.shape[0])
    out["support_penalty"] = support_penalty
    out["support_lambda"] = torch.full_like(support_distance, float(args.lambda_support))
    out["reward_conservative"] = previous_reward - float(args.lambda_support) * support_penalty

    replay_path = output_dir / "synthetic_replay.pt"
    torch.save(out, replay_path)
    summary = {
        "git_sha": git_sha(),
        "git_branch": git_branch(),
        "command": command_line(),
        "synthetic_replay": args.synthetic_replay,
        "dataset": args.dataset,
        "metadata": args.metadata,
        "normalization": args.normalization,
        "output_dir": str(output_dir),
        "replay_path": str(replay_path),
        "split": args.split,
        "quality_filter": args.quality_filter,
        "selected_real_rows": int(selected.numel()),
        "support_rows": int(support_indices.numel()),
        "support_probe_rows": int(probe_indices.numel()),
        "support_feature_dim": int(support_feat.shape[-1]),
        "lambda_support": float(args.lambda_support),
        "support_threshold": float(threshold.item()),
        "support_threshold_quantile": float(args.support_threshold_quantile),
        "state_weight": float(args.state_weight),
        "command_weight": float(args.command_weight),
        "action_weight": float(args.action_weight),
        "real_probe_distance": summarize_tensor(real_probe_distance),
        "support_distance": summarize_tensor(support_distance),
        "support_penalty": summarize_tensor(support_penalty),
        "reward_pre_support": summarize_tensor(previous_reward),
        "reward_conservative": summarize_tensor(out["reward_conservative"]),
        "synthetic_transitions": int(support_distance.shape[0]),
        "wall_clock_seconds": time.time() - t0,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
