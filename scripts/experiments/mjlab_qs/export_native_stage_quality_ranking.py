#!/usr/bin/env python3
"""Rank native checkpoint-stage probes into empirical QS roles."""

from __future__ import annotations

import argparse
import csv
import math
import re
from pathlib import Path
from statistics import mean, pstdev
from typing import Dict, Iterable, List

import torch


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--raw", nargs="+", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--max-expert-fall-rate", type=float, default=0.10)
    p.add_argument("--min-expert-length", type=float, default=800.0)
    p.add_argument("--min-expert-return-margin", type=float, default=1.0)
    p.add_argument("--max-medium-fall-rate", type=float, default=0.60)
    p.add_argument("--min-medium-score", type=float, default=0.35)
    p.add_argument("--min-weak-score", type=float, default=0.08)
    return p.parse_args()


def expand(inputs: Iterable[str]) -> List[Path]:
    out: List[Path] = []
    for raw in inputs:
        p = Path(raw)
        if p.is_dir():
            out.extend(sorted(p.glob("*.pt")))
        else:
            out.append(p)
    return sorted(dict.fromkeys(out))


def finite_mean(vals: List[float]) -> float:
    clean = [v for v in vals if math.isfinite(v)]
    return mean(clean) if clean else float("nan")


def finite_std(vals: List[float]) -> float:
    clean = [v for v in vals if math.isfinite(v)]
    return pstdev(clean) if len(clean) > 1 else 0.0


def tensor_nan_count(x) -> int:
    if not torch.is_tensor(x) or not x.is_floating_point():
        return 0
    return int(torch.isnan(x).sum().item())


def checkpoint_iter(checkpoint: str, collector_id: str) -> int:
    for raw in (checkpoint, collector_id):
        m = re.search(r"(?:model_|iter)(\d+)", raw)
        if m:
            return int(m.group(1))
    return -1


def main() -> None:
    args = parse_args()
    rows: List[Dict[str, object]] = []
    for path in expand(args.raw):
        payload = torch.load(path, map_location="cpu", weights_only=False)
        episodes = payload.get("episodes", [])
        if not episodes:
            continue
        meta = payload.get("metadata", {})
        returns = [float(ep.get("episode_return", float("nan"))) for ep in episodes]
        lengths = [float(ep.get("episode_length", 0.0)) for ep in episodes]
        falls = [float(ep.get("fall_rate_episode", 0.0)) for ep in episodes]
        clips = [float(ep.get("clip_fraction", 0.0)) for ep in episodes]
        nan_reward = sum(tensor_nan_count(ep.get("reward")) for ep in episodes)
        nan_action = sum(tensor_nan_count(ep.get("policy_action")) for ep in episodes)
        checkpoint = str(meta.get("checkpoint", ""))
        collector_id = str(meta.get("collector_id", ""))
        rows.append(
            {
                "raw_path": str(path),
                "task_id": meta.get("task_id_resolved") or meta.get("task_id_requested"),
                "method": meta.get("method", ""),
                "collector_id": collector_id,
                "collector_mode": meta.get("collector_mode", ""),
                "source_quality_bin": meta.get("quality_bin", ""),
                "checkpoint": checkpoint,
                "checkpoint_iter": checkpoint_iter(checkpoint, collector_id),
                "seed": meta.get("seed", ""),
                "n": len(episodes),
                "return_mean": finite_mean(returns),
                "return_std": finite_std(returns),
                "episode_length_mean": finite_mean(lengths),
                "episode_length_std": finite_std(lengths),
                "fall_rate_mean": finite_mean(falls),
                "clip_fraction_mean": finite_mean(clips),
                "nan_reward_count": nan_reward,
                "nan_policy_action_count": nan_action,
            }
        )
    if not rows:
        raise RuntimeError("No probe rows found.")

    random_by_task = {
        r["task_id"]: float(r["return_mean"])
        for r in rows
        if r["source_quality_bin"] == "random_smooth" and math.isfinite(float(r["return_mean"]))
    }
    best_by_task: Dict[object, float] = {}
    for r in rows:
        task = r["task_id"]
        ret = float(r["return_mean"])
        if math.isfinite(ret):
            best_by_task[task] = max(best_by_task.get(task, ret), ret)

    for row in rows:
        random_mean = random_by_task.get(row["task_id"], float("nan"))
        best = best_by_task.get(row["task_id"], float("nan"))
        denom = max(best - random_mean, 1e-6) if math.isfinite(best) and math.isfinite(random_mean) else float("nan")
        score = (float(row["return_mean"]) - random_mean) / denom if math.isfinite(denom) else float("nan")
        row["random_return_mean"] = random_mean
        row["best_return_mean"] = best
        row["quality_score"] = score
        clean = int(row["nan_reward_count"]) == 0 and int(row["nan_policy_action_count"]) == 0
        role = "random_or_failed"
        if (
            clean
            and float(row["fall_rate_mean"]) <= args.max_expert_fall_rate
            and float(row["episode_length_mean"]) >= args.min_expert_length
            and float(row["return_mean"]) >= random_mean + args.min_expert_return_margin
            and score >= 0.80
        ):
            role = "expert"
        elif clean and float(row["fall_rate_mean"]) <= args.max_medium_fall_rate and score >= args.min_medium_score:
            role = "medium"
        elif clean and score >= args.min_weak_score:
            role = "weak"
        row["empirical_role"] = role

    role_rank = {"expert": 3, "medium": 2, "weak": 1, "random_or_failed": 0}
    rows.sort(
        key=lambda r: (
            str(r["task_id"]),
            role_rank.get(str(r["empirical_role"]), 0),
            float(r["quality_score"]) if math.isfinite(float(r["quality_score"])) else -1.0,
            float(r["episode_length_mean"]),
        ),
        reverse=True,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} stage-ranking rows to {output}")


if __name__ == "__main__":
    main()
