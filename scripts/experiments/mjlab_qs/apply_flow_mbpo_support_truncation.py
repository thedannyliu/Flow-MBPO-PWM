#!/usr/bin/env python3
"""Apply support-risk early termination to a scored Flow-MBPO synthetic replay."""

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


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--synthetic-replay", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument(
        "--risk-threshold",
        type=float,
        default=-1.0,
        help="Support-distance termination threshold. Defaults to per-row support_threshold from replay.",
    )
    p.add_argument("--risk-penalty-weight", type=float, default=0.0)
    p.add_argument("--truncate-post-risk", action=argparse.BooleanOptionalAction, default=True)
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


def required_tensor(replay: dict[str, Any], key: str) -> torch.Tensor:
    value = replay.get(key)
    if not torch.is_tensor(value):
        raise ValueError(f"Synthetic replay is missing tensor key {key!r}")
    return value.cpu()


def risk_threshold(replay: dict[str, Any], args: argparse.Namespace, rows: int) -> torch.Tensor:
    if float(args.risk_threshold) >= 0.0:
        return torch.full((rows,), float(args.risk_threshold), dtype=torch.float32)
    threshold = required_tensor(replay, "support_threshold").float().reshape(-1)
    if threshold.numel() == 1:
        threshold = threshold.repeat(rows)
    if threshold.shape[0] != rows:
        raise ValueError(f"support_threshold row mismatch: expected {rows}, got {threshold.shape[0]}")
    return threshold


def branch_truncation_mask(risk: torch.Tensor, replay: dict[str, Any], truncate_post_risk: bool) -> torch.Tensor:
    if not truncate_post_risk:
        return risk.clone()
    rows = risk.shape[0]
    if "start_index" not in replay or "horizon_step" not in replay:
        return risk.clone()
    start_index = required_tensor(replay, "start_index").long().reshape(-1)
    horizon_step = required_tensor(replay, "horizon_step").long().reshape(-1)
    if start_index.shape[0] != rows or horizon_step.shape[0] != rows:
        raise ValueError("start_index/horizon_step must match support_distance rows")
    out = torch.zeros(rows, dtype=torch.bool)
    for start in torch.unique(start_index).tolist():
        idx = (start_index == int(start)).nonzero(as_tuple=False).reshape(-1)
        order = torch.argsort(horizon_step[idx])
        branch_idx = idx[order]
        branch_risk = risk[branch_idx]
        if bool(branch_risk.any().item()):
            first = int(branch_risk.nonzero(as_tuple=False)[0].item())
            out[branch_idx[first:]] = True
    return out


def branch_stats(mask: torch.Tensor, replay: dict[str, Any]) -> dict[str, Any]:
    if "start_index" not in replay:
        return {}
    start_index = required_tensor(replay, "start_index").long().reshape(-1)
    total = int(torch.unique(start_index).numel())
    risky = int(torch.unique(start_index[mask]).numel()) if bool(mask.any().item()) else 0
    return {
        "branches": total,
        "risk_branches": risky,
        "risk_branch_fraction": float(risky / total) if total else math.nan,
    }


def main() -> None:
    args = parse_args()
    t0 = time.time()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    replay = torch.load(args.synthetic_replay, map_location="cpu", weights_only=False)
    if not isinstance(replay, dict):
        raise TypeError(f"{args.synthetic_replay} must contain a dict")
    support_distance = required_tensor(replay, "support_distance").float().reshape(-1)
    previous_reward = required_tensor(replay, "reward_conservative").float().reshape(-1)
    previous_done = required_tensor(replay, "done").bool().reshape(-1)
    rows = int(support_distance.shape[0])
    threshold = risk_threshold(replay, args, rows)
    risk = support_distance > threshold
    truncated = branch_truncation_mask(risk, replay, bool(args.truncate_post_risk))
    penalty = F.relu(support_distance - threshold)

    out = dict(replay)
    out["done_pre_support_truncation"] = previous_done
    out["support_risk_threshold"] = threshold
    out["support_risk_crossing"] = risk
    out["support_risk_truncated"] = truncated
    out["support_risk_penalty"] = penalty
    out["support_risk_penalty_weight"] = torch.full((rows,), float(args.risk_penalty_weight), dtype=torch.float32)
    out["reward_pre_support_truncation"] = previous_reward
    out["reward_conservative"] = previous_reward - float(args.risk_penalty_weight) * penalty
    out["done"] = previous_done | truncated

    replay_path = output_dir / "synthetic_replay.pt"
    torch.save(out, replay_path)
    summary = {
        "git_sha": git_sha(),
        "git_branch": git_branch(),
        "command": command_line(),
        "synthetic_replay": args.synthetic_replay,
        "output_dir": str(output_dir),
        "replay_path": str(replay_path),
        "risk_threshold": float(args.risk_threshold),
        "risk_penalty_weight": float(args.risk_penalty_weight),
        "truncate_post_risk": bool(args.truncate_post_risk),
        "support_distance": summarize_tensor(support_distance),
        "support_risk_threshold": summarize_tensor(threshold),
        "support_risk_penalty": summarize_tensor(penalty),
        "reward_pre_support_truncation": summarize_tensor(previous_reward),
        "reward_conservative": summarize_tensor(out["reward_conservative"]),
        "done_pre_support_truncation_fraction": float(previous_done.float().mean().item()),
        "support_risk_crossing_fraction": float(risk.float().mean().item()),
        "support_risk_truncated_fraction": float(truncated.float().mean().item()),
        "done_post_support_truncation_fraction": float(out["done"].float().mean().item()),
        "branch_stats": branch_stats(truncated, replay),
        "synthetic_transitions": rows,
        "wall_clock_seconds": time.time() - t0,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
