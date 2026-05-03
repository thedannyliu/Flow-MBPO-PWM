#!/usr/bin/env python3
"""Run one MJLab-native collector manifest row."""

from __future__ import annotations

import argparse
import csv
import subprocess


def as_bool(raw: str) -> bool:
    return raw.strip().lower() in {"1", "true", "yes", "y"}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", required=True)
    p.add_argument("--row-index", type=int, required=True)
    p.add_argument("--python-bin", default="python")
    args = p.parse_args()

    with open(args.manifest, newline="", encoding="utf-8") as f:
        row = list(csv.DictReader(f))[args.row_index]

    cmd = [
        args.python_bin,
        "scripts/experiments/mjlab_qs/run_mjlab_native_collector.py",
        "--task-id",
        row["task_id"],
        "--method",
        row["method"],
        "--seed",
        row["seed"],
        "--output-dir",
        row["output_dir"],
        "--num-envs",
        row["num_envs"],
        "--max-iterations",
        row["max_iterations"],
        "--save-interval",
        row["save_interval"],
        "--logger",
        row["logger"],
        "--wandb-project",
        row["wandb_project"],
        "--run-name",
        row["run_name"],
    ]
    if as_bool(row.get("resume", "")):
        cmd.append("--resume")
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
