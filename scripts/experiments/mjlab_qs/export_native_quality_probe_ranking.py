#!/usr/bin/env python3
"""Rank MJLab-native quality-probe shards by empirical rollout quality."""

from __future__ import annotations

import argparse
import csv
import math
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
    p.add_argument("--min-episodes", type=int, default=50)
    return p.parse_args()


def expand(inputs: Iterable[str]) -> List[Path]:
    paths: List[Path] = []
    for raw in inputs:
        p = Path(raw)
        if p.is_dir():
            paths.extend(sorted(p.glob("*.pt")))
        else:
            paths.append(p)
    return sorted(dict.fromkeys(paths))


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
        rows.append(
            {
                "raw_path": str(path),
                "task_id": meta.get("task_id_resolved") or meta.get("task_id_requested"),
                "method": meta.get("method", ""),
                "collector_id": meta.get("collector_id", ""),
                "collector_mode": meta.get("collector_mode", ""),
                "quality_bin": meta.get("quality_bin", ""),
                "checkpoint": meta.get("checkpoint", ""),
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

    random_by_task = {
        r["task_id"]: float(r["return_mean"])
        for r in rows
        if r["quality_bin"] == "random_smooth" and math.isfinite(float(r["return_mean"]))
    }
    for row in rows:
        random_mean = random_by_task.get(row["task_id"], float("nan"))
        row["random_return_mean"] = random_mean
        row["expert_gate_pass"] = bool(
            row["quality_bin"] == "expert"
            and int(row["n"]) >= args.min_episodes
            and int(row["nan_reward_count"]) == 0
            and int(row["nan_policy_action_count"]) == 0
            and float(row["fall_rate_mean"]) <= args.max_expert_fall_rate
            and float(row["episode_length_mean"]) >= args.min_expert_length
            and math.isfinite(random_mean)
            and float(row["return_mean"]) >= random_mean + args.min_expert_return_margin
        )

    rows.sort(
        key=lambda r: (
            str(r["task_id"]),
            bool(r["expert_gate_pass"]),
            float(r["episode_length_mean"]),
            float(r["return_mean"]),
        ),
        reverse=True,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise RuntimeError("No probe rows found.")
    with output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} rows to {output}")


if __name__ == "__main__":
    main()
