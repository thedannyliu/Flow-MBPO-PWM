#!/usr/bin/env python3
"""Run one MJLab-QS Flow train-loss-match manifest row."""

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
    seed = row["seed"]
    stage = row["stage"]
    out = Path("scripts/outputs/mjlab_qs/results") / stage / task_key / "flow_train_loss_match" / f"seed_{seed}"
    out.mkdir(parents=True, exist_ok=True)

    cmd = [
        args.python_bin,
        "scripts/experiments/mjlab_qs/run_phaseA_flow_train_match.py",
        "--dataset", row["dataset"],
        "--metadata", row["metadata"],
        "--normalization", row["normalization"],
        "--seed", seed,
        "--device", args.device,
        "--output-dir", str(out),
        "--mlp-train-iters", row.get("mlp_train_iters", "50000"),
        "--flow-max-iters", row.get("flow_max_iters", "300000"),
        "--batch-size", row.get("batch_size", "256"),
        "--eval-every", row.get("eval_every", "5000"),
        "--match-tolerance", row.get("match_tolerance", "0.05"),
        "--wandb-project", row.get("wandb_project", "flow-mbpo-mjlab-phaseA-train-loss-match"),
        "--wandb-group", row.get("wandb_group", f"{stage}_{task_key}"),
        "--wandb-name", f"{stage}_{task_key}_flow_train_loss_match_seed{seed}",
    ]
    if row.get("flow_lr"):
        cmd.extend(["--flow-lr", row["flow_lr"]])
    if row.get("disable_wandb", "").lower() in {"1", "true", "yes"}:
        cmd.append("--disable-wandb")
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
