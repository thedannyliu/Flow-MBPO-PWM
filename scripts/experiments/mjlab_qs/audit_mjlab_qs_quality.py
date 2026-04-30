#!/usr/bin/env python3
"""Audit empirical quality of MJLab-QS raw episode shards.

The old Phase-A collector labels buckets by how data was collected
(`expert`, `medium`, etc.). This audit deliberately treats those labels only as
source metadata. It recomputes empirical quality from return, episode length,
and fall rate before a shard can be used as a canonical QS dataset.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean, pstdev
from typing import Dict, Iterable, List

import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", nargs="+", required=True, help="Raw .pt episode shards or directories containing shards.")
    parser.add_argument("--csv-output", required=True)
    parser.add_argument("--json-output", required=True)
    parser.add_argument("--md-output", required=True)
    parser.add_argument("--max-expert-fall-rate", type=float, default=0.10)
    parser.add_argument("--min-expert-length", type=float, default=800.0)
    parser.add_argument("--min-expert-return-margin", type=float, default=1.0)
    parser.add_argument("--min-expert-return-ratio", type=float, default=1.50)
    parser.add_argument("--min-episodes-per-empirical-expert", type=int, default=50)
    parser.add_argument("--allow-fail", action="store_true")
    return parser.parse_args()


def expand_raw(inputs: Iterable[str]) -> List[Path]:
    paths: List[Path] = []
    for item in inputs:
        p = Path(item)
        if p.is_dir():
            paths.extend(sorted(p.glob("*.pt")))
        else:
            paths.append(p)
    return sorted(dict.fromkeys(paths))


def safe_float(x: object) -> float:
    try:
        v = float(x)
    except Exception:
        return float("nan")
    return v


def tensor_nan_count(x: object) -> int:
    if not torch.is_tensor(x):
        return 0
    if not x.is_floating_point():
        return 0
    return int(torch.isnan(x).sum().item())


def load_episode_rows(raw_paths: List[Path]) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for raw_path in raw_paths:
        payload = torch.load(raw_path, map_location="cpu", weights_only=False)
        for ep_idx, ep in enumerate(payload.get("episodes", [])):
            reward = ep.get("reward")
            action = ep.get("policy_action")
            episode_return = safe_float(ep.get("episode_return", float("nan")))
            episode_length = int(ep.get("episode_length", 0))
            fall_rate = safe_float(ep.get("fall_rate_episode", float("nan")))
            if torch.is_tensor(reward) and reward.numel() > 1:
                reward_no_nan = torch.nan_to_num(reward[1:].float(), nan=0.0)
                episode_return = float(reward_no_nan.sum().item())
                episode_length = int(reward_no_nan.shape[0])
            rows.append(
                {
                    "raw_path": str(raw_path),
                    "episode_index": ep_idx,
                    "task_id_resolved": str(ep.get("task_id_resolved", "")),
                    "collector_quality_bin": str(ep.get("quality_bin", "")),
                    "collector_id": str(ep.get("collector_id", "")),
                    "collector_mode": str(ep.get("collector_mode", "")),
                    "episode_return": episode_return,
                    "episode_length": episode_length,
                    "fall_rate_episode": fall_rate,
                    "clip_fraction": safe_float(ep.get("clip_fraction", float("nan"))),
                    "nan_reward_count": tensor_nan_count(reward),
                    "nan_policy_action_count": tensor_nan_count(action),
                }
            )
    return rows


def summarize(vals: List[float]) -> Dict[str, float]:
    clean = [v for v in vals if math.isfinite(v)]
    if not clean:
        return {"mean": float("nan"), "std": float("nan"), "min": float("nan"), "max": float("nan")}
    return {
        "mean": mean(clean),
        "std": pstdev(clean) if len(clean) > 1 else 0.0,
        "min": min(clean),
        "max": max(clean),
    }


def empirical_bin(row: Dict[str, object], random_mean: float, best_return: float, args: argparse.Namespace) -> str:
    ret = float(row["episode_return"])
    length = float(row["episode_length"])
    fall = float(row["fall_rate_episode"])
    denom = max(best_return - random_mean, 1e-6)
    q = (ret - random_mean) / denom
    row["empirical_quality_score"] = q
    passes_expert_return = ret >= random_mean + args.min_expert_return_margin
    if random_mean > 0:
        passes_expert_return = passes_expert_return and ret >= random_mean * args.min_expert_return_ratio
    if fall <= args.max_expert_fall_rate and length >= args.min_expert_length and passes_expert_return and q >= 0.8:
        return "expert"
    if fall <= 0.50 and q >= 0.45:
        return "medium"
    if q >= 0.10:
        return "weak"
    return "random_or_failed"


def main() -> None:
    args = parse_args()
    raw_paths = expand_raw(args.raw)
    if not raw_paths:
        raise RuntimeError("No raw .pt shards found.")
    rows = load_episode_rows(raw_paths)
    if not rows:
        raise RuntimeError("No episodes found in raw shards.")

    by_task = defaultdict(list)
    for row in rows:
        by_task[str(row["task_id_resolved"])].append(row)

    task_refs: Dict[str, Dict[str, float]] = {}
    for task, task_rows in by_task.items():
        random_returns = [
            float(r["episode_return"])
            for r in task_rows
            if str(r["collector_quality_bin"]) == "random_smooth" and math.isfinite(float(r["episode_return"]))
        ]
        all_returns = [float(r["episode_return"]) for r in task_rows if math.isfinite(float(r["episode_return"]))]
        if not random_returns:
            raise RuntimeError(f"Task {task} has no random_smooth episodes for reference.")
        task_refs[task] = {
            "random_return_mean": mean(random_returns),
            "best_return": max(all_returns) if all_returns else mean(random_returns),
        }

    for row in rows:
        refs = task_refs[str(row["task_id_resolved"])]
        row["random_return_mean"] = refs["random_return_mean"]
        row["best_return"] = refs["best_return"]
        row["empirical_quality_bin"] = empirical_bin(row, refs["random_return_mean"], refs["best_return"], args)

    summary_rows: List[Dict[str, object]] = []
    groups = defaultdict(list)
    for row in rows:
        groups[(row["task_id_resolved"], row["collector_quality_bin"], row["empirical_quality_bin"])].append(row)
    for (task, collector_bin, empirical_qbin), group in sorted(groups.items()):
        rets = [float(r["episode_return"]) for r in group]
        lengths = [float(r["episode_length"]) for r in group]
        falls = [float(r["fall_rate_episode"]) for r in group]
        clips = [float(r["clip_fraction"]) for r in group]
        ret_s = summarize(rets)
        len_s = summarize(lengths)
        summary_rows.append(
            {
                "task": task,
                "collector_quality_bin": collector_bin,
                "empirical_quality_bin": empirical_qbin,
                "n": len(group),
                "return_mean": ret_s["mean"],
                "return_std": ret_s["std"],
                "return_min": ret_s["min"],
                "return_max": ret_s["max"],
                "episode_length_mean": len_s["mean"],
                "episode_length_min": len_s["min"],
                "episode_length_max": len_s["max"],
                "fall_rate_mean": summarize(falls)["mean"],
                "clip_fraction_mean": summarize(clips)["mean"],
                "nan_reward_count": sum(int(r["nan_reward_count"]) for r in group),
                "nan_policy_action_count": sum(int(r["nan_policy_action_count"]) for r in group),
            }
        )

    gate_failures: List[str] = []
    for task, task_rows in sorted(by_task.items()):
        empirical_expert = [r for r in task_rows if r["empirical_quality_bin"] == "expert"]
        if len(empirical_expert) < args.min_episodes_per_empirical_expert:
            gate_failures.append(
                f"{task}: empirical expert episodes {len(empirical_expert)} < {args.min_episodes_per_empirical_expert}"
            )
        source_expert = [r for r in task_rows if r["collector_quality_bin"] == "expert"]
        if source_expert:
            fall_mean = mean(float(r["fall_rate_episode"]) for r in source_expert)
            length_mean = mean(float(r["episode_length"]) for r in source_expert)
            return_mean = mean(float(r["episode_return"]) for r in source_expert)
            random_mean = task_refs[task]["random_return_mean"]
            if fall_mean > args.max_expert_fall_rate:
                gate_failures.append(f"{task}: collector expert fall_rate {fall_mean:.3f} > {args.max_expert_fall_rate:.3f}")
            if length_mean < args.min_expert_length:
                gate_failures.append(f"{task}: collector expert episode_length {length_mean:.1f} < {args.min_expert_length:.1f}")
            if return_mean < random_mean + args.min_expert_return_margin:
                gate_failures.append(
                    f"{task}: collector expert return {return_mean:.3f} < random_mean+margin {random_mean + args.min_expert_return_margin:.3f}"
                )

    total_nan_reward = sum(int(r["nan_reward_count"]) for r in rows)
    total_nan_action = sum(int(r["nan_policy_action_count"]) for r in rows)
    if total_nan_reward > 0:
        gate_failures.append(f"raw episodes contain reward NaNs: {total_nan_reward}")
    if total_nan_action > 0:
        gate_failures.append(f"raw episodes contain policy_action NaNs: {total_nan_action}")

    csv_output = Path(args.csv_output)
    json_output = Path(args.json_output)
    md_output = Path(args.md_output)
    csv_output.parent.mkdir(parents=True, exist_ok=True)
    json_output.parent.mkdir(parents=True, exist_ok=True)
    md_output.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = list(summary_rows[0].keys()) if summary_rows else []
    with csv_output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)

    json_output.write_text(
        json.dumps(
            {
                "status": "PASS" if not gate_failures else "FAIL",
                "raw_paths": [str(p) for p in raw_paths],
                "thresholds": {
                    "max_expert_fall_rate": args.max_expert_fall_rate,
                    "min_expert_length": args.min_expert_length,
                    "min_expert_return_margin": args.min_expert_return_margin,
                    "min_expert_return_ratio": args.min_expert_return_ratio,
                    "min_episodes_per_empirical_expert": args.min_episodes_per_empirical_expert,
                },
                "task_refs": task_refs,
                "gate_failures": gate_failures,
                "summary": summary_rows,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    lines = [
        "# MJLab-QS Empirical Quality Audit",
        "",
        f"- status: {'PASS' if not gate_failures else 'FAIL'}",
        f"- raw_shards: {len(raw_paths)}",
        f"- episodes: {len(rows)}",
        f"- csv: `{csv_output}`",
        f"- json: `{json_output}`",
        "",
        "## Gate Failures",
        "",
    ]
    if gate_failures:
        lines.extend(f"- {failure}" for failure in gate_failures)
    else:
        lines.append("- none")
    lines.extend(
        [
            "",
            "## Summary",
            "",
            "| task | collector_bin | empirical_bin | n | return_mean | length_mean | fall_rate | nan_reward | nan_action |",
            "|---|---|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in summary_rows:
        lines.append(
            f"| {row['task']} | {row['collector_quality_bin']} | {row['empirical_quality_bin']} | "
            f"{row['n']} | {float(row['return_mean']):.3f} | {float(row['episode_length_mean']):.1f} | "
            f"{float(row['fall_rate_mean']):.3f} | {row['nan_reward_count']} | {row['nan_policy_action_count']} |"
        )
    md_output.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"status: {'PASS' if not gate_failures else 'FAIL'}")
    print(f"saved csv: {csv_output}")
    print(f"saved json: {json_output}")
    print(f"saved md: {md_output}")
    if gate_failures and not args.allow_fail:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
