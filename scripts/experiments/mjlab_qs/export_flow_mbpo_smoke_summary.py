#!/usr/bin/env python3
"""Export Flow-MBPO v0 smoke diagnostics from saved summary JSON files."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


FIELDS = [
    "output_dir",
    "wm_method",
    "wm_model_count",
    "uncertainty_defined",
    "seed",
    "horizon",
    "num_starts",
    "transitions",
    "reward_mean",
    "reward_std",
    "reward_p90",
    "reward_min",
    "reward_max",
    "next_state_delta_l2_mean",
    "next_state_delta_l2_p90",
    "next_state_delta_l2_max",
    "next_state_uncertainty_mean",
    "next_state_uncertainty_p90",
    "next_state_uncertainty_max",
    "reward_uncertainty_mean",
    "reward_uncertainty_p90",
    "reward_uncertainty_max",
    "done_probability_mean",
    "done_probability_p90",
    "done_probability_max",
    "action_l2_mean",
    "action_l2_p90",
    "action_l2_max",
    "predicted_done_fraction",
    "wall_clock_seconds",
    "git_sha",
    "policy_checkpoint",
    "wm_checkpoints",
    "summary_path",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, default=Path("scripts/outputs/mjlab_qs/flow_mbpo_v0_smoke"))
    p.add_argument("--summary", type=Path, action="append", default=[])
    p.add_argument("--output-csv", type=Path, required=True)
    p.add_argument("--output-md", type=Path, required=True)
    p.add_argument("--require-complete", action="store_true")
    return p.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def metric(summary: dict[str, Any], group: str, key: str) -> str:
    values = summary.get(group)
    if not isinstance(values, dict):
        return ""
    value = values.get(key)
    return "" if value is None else str(value)


def scalar(summary: dict[str, Any], key: str) -> str:
    value = summary.get(key)
    if value is None:
        return ""
    if isinstance(value, bool):
        return str(value).lower()
    return str(value)


def record_from_summary(path: Path, root: Path) -> dict[str, str]:
    summary = load_json(path)
    wm_checkpoints = summary.get("wm_checkpoints") or []
    if not isinstance(wm_checkpoints, list):
        wm_checkpoints = [str(wm_checkpoints)]
    try:
        output_dir = str(path.parent.relative_to(root))
    except ValueError:
        output_dir = str(path.parent)
    return {
        "output_dir": output_dir,
        "wm_method": scalar(summary, "wm_method"),
        "wm_model_count": scalar(summary, "wm_model_count"),
        "uncertainty_defined": scalar(summary, "uncertainty_defined"),
        "seed": scalar(summary, "seed"),
        "horizon": scalar(summary, "horizon"),
        "num_starts": scalar(summary, "num_starts"),
        "transitions": scalar(summary, "transitions"),
        "reward_mean": metric(summary, "synthetic_reward", "mean"),
        "reward_std": metric(summary, "synthetic_reward", "std"),
        "reward_p90": metric(summary, "synthetic_reward", "p90"),
        "reward_min": metric(summary, "synthetic_reward", "min"),
        "reward_max": metric(summary, "synthetic_reward", "max"),
        "next_state_delta_l2_mean": metric(summary, "next_state_delta_l2", "mean"),
        "next_state_delta_l2_p90": metric(summary, "next_state_delta_l2", "p90"),
        "next_state_delta_l2_max": metric(summary, "next_state_delta_l2", "max"),
        "next_state_uncertainty_mean": metric(summary, "next_state_uncertainty", "mean"),
        "next_state_uncertainty_p90": metric(summary, "next_state_uncertainty", "p90"),
        "next_state_uncertainty_max": metric(summary, "next_state_uncertainty", "max"),
        "reward_uncertainty_mean": metric(summary, "reward_uncertainty", "mean"),
        "reward_uncertainty_p90": metric(summary, "reward_uncertainty", "p90"),
        "reward_uncertainty_max": metric(summary, "reward_uncertainty", "max"),
        "done_probability_mean": metric(summary, "done_probability", "mean"),
        "done_probability_p90": metric(summary, "done_probability", "p90"),
        "done_probability_max": metric(summary, "done_probability", "max"),
        "action_l2_mean": metric(summary, "action_l2", "mean"),
        "action_l2_p90": metric(summary, "action_l2", "p90"),
        "action_l2_max": metric(summary, "action_l2", "max"),
        "predicted_done_fraction": scalar(summary, "predicted_done_fraction"),
        "wall_clock_seconds": scalar(summary, "wall_clock_seconds"),
        "git_sha": scalar(summary, "git_sha"),
        "policy_checkpoint": scalar(summary, "policy_checkpoint"),
        "wm_checkpoints": ";".join(str(item) for item in wm_checkpoints),
        "summary_path": str(path),
    }


def collect_records(root: Path, summaries: list[Path]) -> list[dict[str, str]]:
    resolved_root = root.resolve()
    paths = list(summaries)
    if root.exists():
        paths.extend(sorted(root.glob("**/summary.json")))
    unique_paths = sorted({path.resolve() for path in paths})
    return [record_from_summary(path, resolved_root) for path in unique_paths if path.exists()]


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def fmt(value: str) -> str:
    if value == "":
        return ""
    try:
        return f"{float(value):.6g}"
    except ValueError:
        return value


def write_md(path: Path, rows: list[dict[str, str]]) -> None:
    columns = [
        "output_dir",
        "wm_method",
        "horizon",
        "transitions",
        "reward_mean",
        "next_state_delta_l2_mean",
        "next_state_uncertainty_mean",
        "reward_uncertainty_mean",
        "done_probability_mean",
        "action_l2_mean",
        "predicted_done_fraction",
        "git_sha",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(fmt(row[column]) for column in columns) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    rows = collect_records(args.root, args.summary)
    if args.require_complete and not rows:
        raise SystemExit(f"No Flow-MBPO smoke summaries found under {args.root}")
    write_csv(args.output_csv, rows)
    write_md(args.output_md, rows)
    print(f"wrote {len(rows)} Flow-MBPO smoke rows to {args.output_csv} and {args.output_md}")


if __name__ == "__main__":
    main()
