#!/usr/bin/env python3
"""Run one MJLab-QS offline PWM policy-extraction manifest row."""

from __future__ import annotations

import argparse
import csv
import fcntl
import subprocess
from pathlib import Path


def row_value(row: dict[str, str], key: str, default: str) -> str:
    value = row.get(key, "")
    return value if value != "" else default


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
        Path("scripts/outputs/mjlab_qs/policy_extraction")
        / stage
        / task_key
        / row["wm_method"]
        / row.get("policy_type", "mlp")
        / row.get("online_profile", "offline")
        / row["compute_profile"]
        / f"seed_{row['seed']}"
    )
    out.mkdir(parents=True, exist_ok=True)
    lock_path = out / ".policy_extraction.lock"
    complete_paths = [
        out / "summary.json",
        out / "eval_summary.json",
        out / "final_policy_extraction.pt",
    ]

    cmd = [
        args.python_bin,
        "scripts/experiments/mjlab_qs/run_offline_pwm_policy_extraction.py",
        "--dataset",
        row["dataset"],
        "--metadata",
        row["metadata"],
        "--normalization",
        row["normalization"],
        "--wm-checkpoint",
        row["wm_checkpoint"],
        "--wm-method",
        row["wm_method"],
        "--policy-type",
        row.get("policy_type", "mlp"),
        "--seed",
        row["seed"],
        "--device",
        args.device,
        "--output-dir",
        str(out),
        "--task-id",
        row.get("task_id", "Mjlab-Velocity-Flat-Unitree-G1"),
        "--policy-iters",
        row_value(row, "policy_iters", "15000"),
        "--batch-size",
        row_value(row, "batch_size", "64"),
        "--horizon",
        row_value(row, "horizon", "16"),
        "--gamma",
        row_value(row, "gamma", "0.99"),
        "--lam",
        row_value(row, "lam", "0.95"),
        "--actor-lr",
        row_value(row, "actor_lr", "5e-4"),
        "--critic-lr",
        row_value(row, "critic_lr", "5e-4"),
        "--actor-units",
        row_value(row, "actor_units", "400,200,100"),
        "--critic-units",
        row_value(row, "critic_units", "400,200"),
        "--num-critics",
        row_value(row, "num_critics", "3"),
        "--critic-iterations",
        row_value(row, "critic_iterations", "8"),
        "--critic-batches",
        row_value(row, "critic_batches", "4"),
        "--actor-grad-norm",
        row_value(row, "actor_grad_norm", "1.0"),
        "--critic-grad-norm",
        row_value(row, "critic_grad_norm", "100.0"),
        "--eval-every",
        row_value(row, "eval_every", "1000"),
        "--eval-episodes",
        row_value(row, "eval_episodes", "40"),
        "--eval-num-envs",
        row_value(row, "eval_num_envs", "16"),
        "--episode-length",
        row_value(row, "episode_length", "1000"),
        "--action-l2",
        row_value(row, "action_l2", "1e-4"),
        "--policy-bc-reg",
        row_value(row, "policy_bc_reg", "0.0"),
        "--bc-warmstart-iters",
        row_value(row, "bc_warmstart_iters", "0"),
        "--bc-lr",
        row_value(row, "bc_lr", row_value(row, "actor_lr", "5e-4")),
        "--bc-batch-size",
        row_value(row, "bc_batch_size", "256"),
        "--bc-eval-every",
        row_value(row, "bc_eval_every", "1000"),
        "--bc-quality-filter",
        row_value(row, "bc_quality_filter", ""),
        "--policy-quality-filter",
        row_value(row, "policy_quality_filter", ""),
        "--flow-policy-substeps",
        row_value(row, "flow_policy_substeps", "2"),
        "--flow-policy-integrator",
        row_value(row, "flow_policy_integrator", "heun"),
        "--online-finetune-rounds",
        row_value(row, "online_finetune_rounds", "0"),
        "--online-collect-windows",
        row_value(row, "online_collect_windows", "256"),
        "--online-wm-iters",
        row_value(row, "online_wm_iters", "1000"),
        "--online-policy-iters",
        row_value(row, "online_policy_iters", "3000"),
        "--online-wm-lr",
        row_value(row, "online_wm_lr", "3e-4"),
        "--wandb-project",
        row_value(row, "wandb_project", "flow-mbpo-mjlab-offline-pwm-policy-extraction"),
        "--wandb-group",
        row.get("wandb_group", stage),
        "--wandb-name",
        (
            f"{stage}_{task_key}_{row['wm_method']}_{row.get('policy_type', 'mlp')}_"
            f"{row.get('online_profile', 'offline')}_{row['compute_profile']}_seed{row['seed']}"
        ),
    ]
    if row.get("ret_rms", "true").lower() in {"0", "false", "no"}:
        cmd.append("--no-ret-rms")
    if row.get("skip_real_eval", "").lower() in {"1", "true", "yes"}:
        cmd.append("--skip-real-eval")
    if row.get("disable_wandb", "").lower() in {"1", "true", "yes"}:
        cmd.append("--disable-wandb")

    with lock_path.open("w", encoding="utf-8") as lock_file:
        try:
            fcntl.flock(lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            print(f"policy extraction already running; skipping {out}", flush=True)
            return
        if all(path.exists() for path in complete_paths):
            print(f"policy extraction already complete; skipping {out}", flush=True)
            return
        subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
