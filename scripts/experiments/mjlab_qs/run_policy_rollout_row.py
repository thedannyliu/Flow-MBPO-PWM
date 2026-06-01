#!/usr/bin/env python3
"""Run one completed MJLab-QS policy rollout-video manifest row."""

from __future__ import annotations

import argparse
import csv
import fcntl
import subprocess
from pathlib import Path

import torch


def output_dir(row: dict[str, str]) -> Path:
    return (
        Path("scripts/outputs/mjlab_qs/policy_rollouts")
        / (row.get("rollout_stage") or row["stage"])
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
        / (row.get("policy_stage") or row["stage"])
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
    p.add_argument("--checkpoint-kinds", default="final,best")
    args = p.parse_args()

    with open(args.manifest, newline="", encoding="utf-8") as f:
        row = list(csv.DictReader(f))[args.row_index]

    rollout_episodes = int(row.get("rollout_episodes") or args.rollout_episodes)
    max_steps = int(row.get("rollout_max_steps") or args.max_steps)
    video_fps = int(row.get("video_fps") or args.video_fps)
    action_ramp_steps = int(row.get("action_ramp_steps") or 0)
    checkpoint_kinds_arg = row.get("checkpoint_kinds") or args.checkpoint_kinds
    rollout_stage = row.get("rollout_stage") or row["stage"]
    baseline_return = row.get("rollout_baseline_return") or row.get("baseline_return")
    baseline_length = row.get("rollout_baseline_length") or row.get("baseline_length")
    baseline_fall = row.get("rollout_baseline_fall") or row.get("baseline_fall")

    out = output_dir(row)
    out.mkdir(parents=True, exist_ok=True)
    lock_path = out / ".policy_rollout.lock"
    checkpoint_kinds = {value.strip() for value in checkpoint_kinds_arg.split(",") if value.strip()}

    def checkpoint_specs() -> list[tuple[str, Path, Path]]:
        specs: list[tuple[str, Path, Path]] = []
        base = policy_dir(row)
        if "final" in checkpoint_kinds:
            specs.append(("final", base / "final_policy_extraction.pt", out))
        if "best" in checkpoint_kinds:
            best = base / "best_policy_extraction.pt"
            if best.exists():
                ckpt = torch.load(best, map_location="cpu", weights_only=False)
                if ckpt.get("is_true_best_snapshot") is True:
                    specs.append(("best", best, out / "best"))
                else:
                    print(f"best checkpoint is legacy/non-snapshot; skipping {best}", flush=True)
        return specs

    with lock_path.open("w", encoding="utf-8") as lock_file:
        try:
            fcntl.flock(lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            print(f"policy rollout already running; skipping {out}", flush=True)
            return
        for kind, src, render_out in checkpoint_specs():
            if not src.exists():
                print(f"{kind} policy checkpoint missing; skipping {src}", flush=True)
                continue
            complete_paths = [render_out / "summary.json", render_out / "rollout.mp4"]
            if all(path.exists() for path in complete_paths):
                print(f"{kind} policy rollout already complete; skipping {render_out}", flush=True)
                continue
            wandb_name = (
                f"{rollout_stage}_{row.get('task_key', 'task_unknown')}_{row['wm_method']}_"
                f"{row.get('policy_type', 'mlp')}_{row.get('online_profile', 'offline')}_"
                f"{row['compute_profile']}_seed{row['seed']}_{kind}_rollout"
            )
            cmd = [
                args.python_bin,
                "scripts/experiments/mjlab_qs/render_policy_rollout.py",
                "--policy-checkpoint",
                str(src),
                "--output-dir",
                str(render_out),
                "--checkpoint-kind",
                kind,
                "--device",
                args.device,
                "--rollout-episodes",
                str(rollout_episodes),
                "--max-steps",
                str(max_steps),
                "--video-fps",
                str(video_fps),
                "--action-ramp-steps",
                str(action_ramp_steps),
                "--wandb-project",
                row.get("wandb_project", "flow-mbpo-mjlab-offline-pwm-policy-extraction"),
                "--wandb-group",
                f"{row.get('wandb_group', row['stage'])}_rollouts",
                "--wandb-name",
                wandb_name,
            ]
            if row.get("disable_wandb", "").lower() in {"1", "true", "yes"}:
                cmd.append("--disable-wandb")
            if baseline_return and baseline_length and baseline_fall:
                cmd.extend(
                    [
                        "--baseline-return",
                        baseline_return,
                        "--baseline-length",
                        baseline_length,
                        "--baseline-fall",
                        baseline_fall,
                    ]
                )
            subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
