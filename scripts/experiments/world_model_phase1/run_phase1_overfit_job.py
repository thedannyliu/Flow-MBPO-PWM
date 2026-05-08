#!/usr/bin/env python3
"""Run one row from a Phase 1 WM overfit manifest."""

from __future__ import annotations

import argparse
import csv
import subprocess
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--row-index", type=int, required=True)
    parser.add_argument("--python-bin", default="python")
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = Path(args.manifest)
    with manifest.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    row = rows[args.row_index]
    dataset_key = row.get("dataset_key")
    if not dataset_key:
        dataset_key = Path(row["dataset"]).stem

    out_dir = (
        Path(args.project_root)
        / "scripts"
        / "outputs"
        / "world_model_phase1"
        / row["stage"]
        / row["task_key"]
        / dataset_key
        / row["method_key"]
        / row.get("profile", "default")
        / f"seed_{row['seed']}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        args.python_bin,
        "scripts/experiments/world_model_phase1/run_wm_overfit.py",
        "--dataset",
        row["dataset"],
        "--metadata",
        row["dataset_metadata"],
        "--alg-config",
        row["alg_config"],
        "--seed",
        row["seed"],
        "--device",
        args.device,
        "--output-dir",
        str(out_dir),
        "--train-iters",
        row["train_iters"],
        "--batch-size",
        row["batch_size"],
        "--eval-every",
        row["eval_every"],
        "--log-every",
        row["log_every"],
        "--split-seed",
        row["split_seed"],
        "--wandb-project",
        row["wandb_project"],
        "--wandb-group",
        row["wandb_group"],
        "--wandb-job-type",
        row["wandb_job_type"],
        "--wandb-name",
        f"{row['task_key']}_{dataset_key}_{row['method_key']}_{row.get('profile', 'default')}_seed{row['seed']}",
        "--wandb-tags",
        f"phase1,{row['stage']},{row['task_key']},{dataset_key},{row['method_key']},{row.get('profile', 'default')},seed_{row['seed']}",
    ]
    frozen_encoder_checkpoint = str(row.get("frozen_encoder_checkpoint", "")).strip()
    if frozen_encoder_checkpoint:
        cmd.extend(["--frozen-encoder-checkpoint", frozen_encoder_checkpoint])
    disable_wandb = str(row.get("disable_wandb", "")).strip().lower()
    if disable_wandb in {"1", "true", "yes", "y"}:
        cmd.append("--disable-wandb")
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
