#!/usr/bin/env python3
"""Run one Flow-MBPO replay quality manifest row."""

from __future__ import annotations

import argparse
import csv
import subprocess
from pathlib import Path


def present(row: dict[str, str], key: str) -> bool:
    return bool(str(row.get(key, "")).strip())


def check_paths(row: dict[str, str]) -> None:
    errors: list[str] = []
    for key in ["synthetic_replay", "dataset", "metadata", "normalization"]:
        path = Path(row.get(key, ""))
        if not str(path) or not path.exists():
            errors.append(f"{key} missing or does not exist: {path}")
    if errors:
        raise SystemExit("Replay quality row input validation failed:\n" + "\n".join(errors))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", required=True)
    p.add_argument("--row-index", type=int, required=True)
    p.add_argument("--python-bin", default="python")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    with open(args.manifest, newline="", encoding="utf-8") as f:
        row = list(csv.DictReader(f))[args.row_index]
    check_paths(row)

    cmd = [
        args.python_bin,
        "scripts/experiments/mjlab_qs/analyze_flow_mbpo_replay_quality.py",
        "--synthetic-replay",
        row["synthetic_replay"],
        "--dataset",
        row["dataset"],
        "--metadata",
        row["metadata"],
        "--normalization",
        row["normalization"],
        "--output-json",
        row["output_json"],
        "--output-md",
        row["output_md"],
        "--device",
        row.get("device") or args.device,
    ]
    for option, key in (
        ("--split", "split"),
        ("--quality-filter", "quality_filter"),
        ("--support-max-rows", "support_max_rows"),
        ("--support-probe-rows", "support_probe_rows"),
        ("--distance-batch-size", "distance_batch_size"),
        ("--state-weight", "state_weight"),
        ("--command-weight", "command_weight"),
        ("--action-weight", "action_weight"),
        ("--high-reward-quantile", "high_reward_quantile"),
        ("--high-distance-quantile", "high_distance_quantile"),
        ("--seed", "seed"),
    ):
        if present(row, key):
            cmd.extend([option, row[key]])

    if args.dry_run:
        print(" ".join(cmd), flush=True)
        return
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
