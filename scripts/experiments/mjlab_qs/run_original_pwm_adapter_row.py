#!/usr/bin/env python3
"""Run one original-PWM-adapter manifest row."""

from __future__ import annotations

import argparse
import csv
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

    task_key = row.get("task_key", "task_unknown")
    stage = row["stage"]
    out = (
        Path("scripts/outputs/mjlab_qs/original_pwm_adapter")
        / stage
        / task_key
        / row.get("profile", "original_pwm_adapter")
        / f"seed_{row['seed']}"
    )
    out.mkdir(parents=True, exist_ok=True)

    cmd = [
        args.python_bin,
        "scripts/experiments/mjlab_qs/run_original_pwm_adapter.py",
        "--dataset",
        row["dataset"],
        "--metadata",
        row["metadata"],
        "--normalization",
        row["normalization"],
        "--seed",
        row["seed"],
        "--device",
        args.device,
        "--output-dir",
        str(out),
        "--task-id",
        row.get("task_id", "Mjlab-Velocity-Flat-Unitree-G1"),
        "--pretrain-iters",
        row.get("pretrain_iters", "50000"),
        "--policy-iters",
        row.get("policy_iters", "15000"),
        "--wm-batch-size",
        row.get("wm_batch_size", "256"),
        "--policy-batch-size",
        row.get("policy_batch_size", "64"),
        "--horizon",
        row.get("horizon", "16"),
        "--gamma",
        row.get("gamma", "0.99"),
        "--lam",
        row.get("lam", "0.95"),
        "--actor-lr",
        row.get("actor_lr", "5e-4"),
        "--critic-lr",
        row.get("critic_lr", "5e-4"),
        "--model-lr",
        row.get("model_lr", "3e-4"),
        "--critic-iterations",
        row.get("critic_iterations", "8"),
        "--critic-batches",
        row.get("critic_batches", "4"),
        "--num-critics",
        row.get("num_critics", "3"),
        "--latent-dim",
        row.get("latent_dim", "512"),
        "--eval-every",
        row.get("eval_every", "1000"),
        "--pretrain-log-every",
        row.get("pretrain_log_every", "1000"),
        "--eval-episodes",
        row.get("eval_episodes", "40"),
        "--eval-num-envs",
        row.get("eval_num_envs", "16"),
        "--episode-length",
        row.get("episode_length", "1000"),
        "--obs-mode",
        row.get("obs_mode", "normalized"),
        "--reward-mode",
        row.get("reward_mode", "normalized"),
        "--wandb-project",
        row.get("wandb_project", "flow-mbpo-mjlab-original-pwm-adapter"),
        "--wandb-group",
        row.get("wandb_group", stage),
        "--wandb-name",
        f"{stage}_{task_key}_{row.get('profile', 'original_pwm_adapter')}_seed{row['seed']}",
    ]
    if row.get("rew_rms", "false").lower() in {"0", "false", "no"}:
        cmd.append("--no-rew-rms")
    else:
        cmd.append("--rew-rms")
    if row.get("ret_rms", "true").lower() in {"0", "false", "no"}:
        cmd.append("--no-ret-rms")
    if row.get("skip_real_eval", "").lower() in {"1", "true", "yes"}:
        cmd.append("--skip-real-eval")
    if row.get("disable_wandb", "").lower() in {"1", "true", "yes"}:
        cmd.append("--disable-wandb")

    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
