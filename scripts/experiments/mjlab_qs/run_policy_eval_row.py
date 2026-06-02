#!/usr/bin/env python3
"""Run one completed MJLab-QS policy eval manifest row."""

from __future__ import annotations

import argparse
import csv
import fcntl
import subprocess
from pathlib import Path

import torch


def output_dir(row: dict[str, str]) -> Path:
    if row.get("policy_checkpoint") and row.get("eval_output_dir"):
        return Path(row["eval_output_dir"])
    return (
        Path("scripts/outputs/mjlab_qs/policy_evals")
        / (row.get("eval_stage") or row["stage"])
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
    p.add_argument("--eval-episodes", type=int, default=40)
    p.add_argument("--eval-num-envs", type=int, default=16)
    p.add_argument("--max-steps", type=int, default=1000)
    p.add_argument("--checkpoint-kinds", default="final,best")
    args = p.parse_args()

    with open(args.manifest, newline="", encoding="utf-8") as f:
        row = list(csv.DictReader(f))[args.row_index]

    eval_episodes = int(row.get("eval_episodes") or args.eval_episodes)
    eval_num_envs = int(row.get("eval_num_envs") or args.eval_num_envs)
    max_steps = int(row.get("eval_max_steps") or args.max_steps)
    action_ramp_steps = int(row.get("action_ramp_steps") or 0)
    checkpoint_kinds_arg = row.get("checkpoint_kinds") or args.checkpoint_kinds
    eval_stage = row.get("eval_stage") or row["stage"]
    baseline_return = row.get("eval_baseline_return") or row.get("baseline_return")
    baseline_length = row.get("eval_baseline_length") or row.get("baseline_length")
    baseline_fall = row.get("eval_baseline_fall") or row.get("baseline_fall")
    notes = row.get("notes", "")
    direct_checkpoint = row.get("policy_checkpoint")

    out = output_dir(row)
    out.mkdir(parents=True, exist_ok=True)
    lock_path = out / ".policy_eval.lock"
    checkpoint_kinds = {value.strip() for value in checkpoint_kinds_arg.split(",") if value.strip()}

    def checkpoint_specs() -> list[tuple[str, Path, Path]]:
        specs: list[tuple[str, Path, Path]] = []
        if direct_checkpoint:
            kind = row.get("checkpoint_kind") or row.get("candidate") or "policy"
            specs.append((kind, Path(direct_checkpoint), out))
            return specs
        base = policy_dir(row)
        if "final" in checkpoint_kinds:
            specs.append(("final", base / "final_policy_extraction.pt", out / "final"))
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
            print(f"policy eval already running; skipping {out}", flush=True)
            return
        for kind, src, eval_out in checkpoint_specs():
            if not src.exists():
                print(f"{kind} policy checkpoint missing; skipping {src}", flush=True)
                continue
            complete_paths = [eval_out / "summary.json", eval_out / "eval_episodes.csv"]
            if all(path.exists() for path in complete_paths):
                print(f"{kind} policy eval already complete; skipping {eval_out}", flush=True)
                continue
            if direct_checkpoint:
                wandb_name = row.get("wandb_name") or f"{eval_stage}_{kind}_eval"
                wandb_group = row.get("wandb_group") or f"{row['stage']}_eval"
            else:
                wandb_name = (
                    f"{eval_stage}_{row.get('task_key', 'task_unknown')}_{row['wm_method']}_"
                    f"{row.get('policy_type', 'mlp')}_{row.get('online_profile', 'offline')}_"
                    f"{row['compute_profile']}_seed{row['seed']}_{kind}_eval"
                )
                wandb_group = f"{row.get('wandb_group') or row['stage']}_eval"
            cmd = [
                args.python_bin,
                "scripts/experiments/mjlab_qs/eval_policy_checkpoint.py",
                "--policy-checkpoint",
                str(src),
                "--output-dir",
                str(eval_out),
                "--checkpoint-kind",
                kind,
                "--device",
                args.device,
                "--eval-episodes",
                str(eval_episodes),
                "--eval-num-envs",
                str(eval_num_envs),
                "--max-steps",
                str(max_steps),
                "--action-ramp-steps",
                str(action_ramp_steps),
                "--wandb-project",
                row.get("wandb_project") or "flow-mbpo-mjlab-policy-eval",
                "--wandb-group",
                wandb_group,
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
            if notes:
                cmd.extend(["--notes", notes])
            for option, key in (
                ("--checkpoint-format", "checkpoint_format"),
                ("--dataset", "dataset"),
                ("--metadata", "metadata"),
                ("--normalization", "normalization"),
                ("--task-id", "task_id"),
                ("--seed", "seed"),
                ("--command-dim", "command_dim"),
                ("--command-position", "command_position"),
                ("--obs-mode", "obs_mode"),
            ):
                if row.get(key):
                    cmd.extend([option, row[key]])
            subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
