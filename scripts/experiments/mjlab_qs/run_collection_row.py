#!/usr/bin/env python3
"""Run one MJLab-QS collection manifest row."""

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
        "scripts/experiments/mjlab_qs/collect_mjlab_qs_episodes.py",
        "--env-config", row["env_config"],
        "--output", row["output"],
        "--metadata-output", row["metadata_output"],
        "--device", args.device,
        "--seed", row["seed"],
        "--num-envs", row["num_envs"],
        "--episodes", row["episodes"],
        "--episode-length", row["episode_length"],
        "--collector-mode", row["collector_mode"],
        "--collector-id", row["collector_id"],
        "--quality-bin", row["quality_bin"],
        "--teacher-blend", row["teacher_blend"],
        "--action-noise-std", row["action_noise_std"],
        "--command-dim", row["command_dim"],
        "--command-position", row["command_position"],
        "--strict-task-resolution",
        "--disable-domain-randomization",
    ]
    if row["collector_checkpoint"]:
        cmd += ["--collector-alg-config", row["collector_alg_config"], "--collector-checkpoint", row["collector_checkpoint"], "--teacher-deterministic"]
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
