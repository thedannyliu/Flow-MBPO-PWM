#!/usr/bin/env python3
"""Run one Flow-MBPO AWR diagnostic manifest row."""

from __future__ import annotations

import argparse
import csv
import subprocess
from pathlib import Path


def present(row: dict[str, str], key: str) -> bool:
    return bool(str(row.get(key, "")).strip())


def split_list(value: str) -> list[str]:
    return [item.strip() for item in value.split("|") if item.strip()]


def check_paths(row: dict[str, str]) -> None:
    keys = ["dataset", "metadata", "normalization", "bc_checkpoint", "synthetic_replay"]
    errors: list[str] = []
    for key in keys:
        path = Path(row.get(key, ""))
        if not str(path) or not path.exists():
            errors.append(f"{key} missing or does not exist: {path}")
    for path_text in split_list(row.get("policy_checkpoints", "")):
        if not Path(path_text).exists():
            errors.append(f"policy checkpoint does not exist: {path_text}")
    if errors:
        raise SystemExit("AWR diagnostic row input validation failed:\n" + "\n".join(errors))


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

    policy_checkpoints = split_list(row["policy_checkpoints"])
    labels = split_list(row.get("policy_labels", ""))
    if labels and len(labels) != len(policy_checkpoints):
        raise SystemExit("policy_labels must be empty or have one label per policy checkpoint")

    cmd = [
        args.python_bin,
        "scripts/experiments/mjlab_qs/analyze_flow_mbpo_awr_diagnostics.py",
        "--dataset",
        row["dataset"],
        "--metadata",
        row["metadata"],
        "--normalization",
        row["normalization"],
        "--bc-checkpoint",
        row["bc_checkpoint"],
        "--synthetic-replay",
        row["synthetic_replay"],
        "--output-json",
        row["output_json"],
        "--output-md",
        row["output_md"],
        "--device",
        row.get("device") or args.device,
    ]
    for checkpoint in policy_checkpoints:
        cmd.extend(["--policy-checkpoint", checkpoint])
    for label in labels:
        cmd.extend(["--policy-label", label])
    for option, key in (
        ("--split", "split"),
        ("--quality-filter", "quality_filter"),
        ("--num-real", "num_real"),
        ("--num-synthetic", "num_synthetic"),
        ("--adv-temperature", "adv_temperature"),
        ("--weight-clip", "weight_clip"),
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
