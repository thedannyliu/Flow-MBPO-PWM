#!/usr/bin/env python3
"""Submit Phase 1 dataset-variant collection jobs from a CSV manifest."""

from __future__ import annotations

import argparse
import csv
import subprocess
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = Path(args.manifest)
    script = manifest.parent.parent / "submit_phase1_dataset_job.sh"

    with manifest.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))

    for row in rows:
        cmd = [
            "bash",
            str(script),
            "--env-config",
            row["env_config"],
            "--output",
            row["output"],
            "--metadata-output",
            row["metadata_output"],
            "--gpu-type",
            row["gpu_type"],
            "--time",
            row["time_limit"],
            "--mem",
            row["memory"],
            "--cpus",
            row["cpus"],
            "--python-bin",
            row["python_bin"],
            "--conda-env",
            row["conda_env"],
            "--num-envs",
            row["num_envs"],
            "--target-episodes",
            row["target_episodes"],
            "--episode-length",
            row["episode_length"],
            "--window-length",
            row["window_length"],
            "--window-stride",
            row["window_stride"],
            "--max-windows",
            row["max_windows"],
            "--action-mode",
            row["action_mode"],
            "--seed",
            row["seed"],
            "--wandb-project",
            row["wandb_project"],
            "--wandb-group",
            row["wandb_group"],
            "--wandb-name",
            f"{row['task_key']}_{row['dataset_variant']}_seed{row['seed']}",
            "--wandb-tags",
            row["wandb_tags"],
            "--job-name",
            row["job_name"],
        ]
        if row.get("teacher_alg_config"):
            cmd.extend(["--teacher-alg-config", row["teacher_alg_config"]])
        if row.get("teacher_checkpoint"):
            cmd.extend(["--teacher-checkpoint", row["teacher_checkpoint"]])
        if str(row.get("teacher_deterministic", "0")) == "1":
            cmd.append("--teacher-deterministic")
        cmd.extend(["--mixed-teacher-prob", row.get("mixed_teacher_prob", "0.5")])
        if str(row.get("disable_wandb", "0")) == "1":
            cmd.append("--disable-wandb")

        subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
