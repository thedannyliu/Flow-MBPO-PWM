#!/usr/bin/env python3
"""Run one MJLab-native collection manifest row."""

from __future__ import annotations

import argparse
import csv
import subprocess


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", required=True)
    p.add_argument("--row-index", type=int, required=True)
    p.add_argument("--python-bin", default="python")
    p.add_argument("--device", default="cuda:0")
    args = p.parse_args()
    with open(args.manifest, newline="", encoding="utf-8") as f:
        row = list(csv.DictReader(f))[args.row_index]
    cmd = [
        args.python_bin,
        "scripts/experiments/mjlab_qs/collect_mjlab_qs_native_episodes.py",
        "--task-id",
        row["task_id"],
        "--method",
        row["method"],
        "--checkpoint",
        row.get("checkpoint", ""),
        "--output",
        row["output"],
        "--metadata-output",
        row["metadata_output"],
        "--device",
        args.device,
        "--seed",
        row["seed"],
        "--num-envs",
        row["num_envs"],
        "--episodes",
        row["episodes"],
        "--episode-length",
        row["episode_length"],
        "--quality-bin",
        row["quality_bin"],
        "--collector-id",
        row["collector_id"],
        "--collector-mode",
        row["collector_mode"],
        "--teacher-blend",
        row["teacher_blend"],
        "--action-noise-std",
        row["action_noise_std"],
        "--command-dim",
        row["command_dim"],
        "--command-position",
        row["command_position"],
    ]
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
