#!/usr/bin/env python3
"""Run one MJLab-native collector rollout-video manifest row."""

from __future__ import annotations

import argparse
import csv
import fcntl
import subprocess
from pathlib import Path


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", required=True)
    p.add_argument("--row-index", type=int, required=True)
    p.add_argument("--python-bin", default="python")
    p.add_argument("--device", default="cuda:0")
    args = p.parse_args()

    with open(args.manifest, newline="", encoding="utf-8") as f:
        row = list(csv.DictReader(f))[args.row_index]

    output_dir = Path(row["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    lock_path = output_dir / ".native_collector_rollout.lock"
    with lock_path.open("w", encoding="utf-8") as lock_file:
        try:
            fcntl.flock(lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            print(f"native collector rollout already running; skipping {output_dir}", flush=True)
            return
        if (output_dir / "summary.json").exists() and (output_dir / "rollout.mp4").exists():
            print(f"native collector rollout already complete; skipping {output_dir}", flush=True)
            return
        cmd = [
            args.python_bin,
            "scripts/experiments/mjlab_qs/render_native_collector_rollout.py",
            "--task-id",
            row["task_id"],
            "--method",
            row["method"],
            "--checkpoint",
            row.get("checkpoint", ""),
            "--output-dir",
            row["output_dir"],
            "--device",
            args.device,
            "--seed",
            row.get("seed", "0"),
            "--episodes",
            row.get("episodes", "3"),
            "--episode-length",
            row.get("episode_length", "1000"),
            "--collector-id",
            row.get("collector_id", ""),
            "--collector-mode",
            row.get("collector_mode", "checkpoint"),
            "--teacher-blend",
            row.get("teacher_blend", "1.0"),
            "--action-noise-std",
            row.get("action_noise_std", "0.0"),
            "--video-fps",
            row.get("video_fps", "30"),
            "--wandb-project",
            row.get("wandb_project", "flow-mbpo-mjlab-collector-baselines"),
            "--wandb-group",
            row.get("wandb_group", row.get("stage", "native_collector_rollouts")),
            "--wandb-name",
            row.get("wandb_name", ""),
        ]
        if row.get("disable_wandb", "").lower() in {"1", "true", "yes"}:
            cmd.append("--disable-wandb")
        subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
