#!/usr/bin/env python3
"""Run one completed MJLab-QS policy rollout-video manifest row."""

from __future__ import annotations

import argparse
import csv
import fcntl
import subprocess
from pathlib import Path


def output_dir(row: dict[str, str]) -> Path:
    return (
        Path("scripts/outputs/mjlab_qs/policy_rollouts")
        / row["stage"]
        / row.get("task_key", "task_unknown")
        / row["wm_method"]
        / row.get("policy_type", "mlp")
        / row.get("online_profile", "offline")
        / row["compute_profile"]
        / f"seed_{row['seed']}"
    )


def policy_dir(row: dict[str, str]) -> Path:
    return (
        Path("scripts/outputs/mjlab_qs/policy_extraction")
        / row["stage"]
        / row.get("task_key", "task_unknown")
        / row["wm_method"]
        / row.get("policy_type", "mlp")
        / row.get("online_profile", "offline")
        / row["compute_profile"]
        / f"seed_{row['seed']}"
    )


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", required=True)
    p.add_argument("--row-index", type=int, required=True)
    p.add_argument("--python-bin", default="python")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--rollout-episodes", type=int, default=3)
    p.add_argument("--max-steps", type=int, default=300)
    p.add_argument("--video-fps", type=int, default=30)
    args = p.parse_args()

    with open(args.manifest, newline="", encoding="utf-8") as f:
        row = list(csv.DictReader(f))[args.row_index]

    src = policy_dir(row) / "final_policy_extraction.pt"
    if not src.exists():
        print(f"policy checkpoint missing; skipping {src}", flush=True)
        return

    out = output_dir(row)
    out.mkdir(parents=True, exist_ok=True)
    complete_paths = [out / "summary.json", out / "rollout.mp4"]
    lock_path = out / ".policy_rollout.lock"
    cmd = [
        args.python_bin,
        "scripts/experiments/mjlab_qs/render_policy_rollout.py",
        "--policy-checkpoint",
        str(src),
        "--output-dir",
        str(out),
        "--device",
        args.device,
        "--rollout-episodes",
        str(args.rollout_episodes),
        "--max-steps",
        str(args.max_steps),
        "--video-fps",
        str(args.video_fps),
        "--wandb-project",
        row.get("wandb_project", "flow-mbpo-mjlab-offline-pwm-policy-extraction"),
        "--wandb-group",
        f"{row.get('wandb_group', row['stage'])}_rollouts",
        "--wandb-name",
        (
            f"{row['stage']}_{row.get('task_key', 'task_unknown')}_{row['wm_method']}_"
            f"{row.get('policy_type', 'mlp')}_{row.get('online_profile', 'offline')}_"
            f"{row['compute_profile']}_seed{row['seed']}_rollout"
        ),
    ]
    if row.get("disable_wandb", "").lower() in {"1", "true", "yes"}:
        cmd.append("--disable-wandb")

    with lock_path.open("w", encoding="utf-8") as lock_file:
        try:
            fcntl.flock(lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            print(f"policy rollout already running; skipping {out}", flush=True)
            return
        if all(path.exists() for path in complete_paths):
            print(f"policy rollout already complete; skipping {out}", flush=True)
            return
        subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
