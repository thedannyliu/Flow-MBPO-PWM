#!/usr/bin/env python3
"""Run one MJLab-QS A2.5 training manifest row."""

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
    out = Path("scripts/outputs/mjlab_qs/results") / row["stage"] / task_key / row["method"] / f"seed_{row['seed']}"
    out.mkdir(parents=True, exist_ok=True)
    cmd = [
        args.python_bin,
        "scripts/experiments/mjlab_qs/run_phaseA_wm_feasibility.py",
        "--dataset", row["dataset"],
        "--metadata", row["metadata"],
        "--normalization", row["normalization"],
        "--method", row["method"],
        "--seed", row["seed"],
        "--device", args.device,
        "--output-dir", str(out),
        "--train-iters", row["train_iters"],
        "--base-pretrain-iters", row["base_pretrain_iters"],
        "--batch-size", row["batch_size"],
        "--eval-batch-size", row.get("eval_batch_size", "1024"),
        "--eval-every", row["eval_every"],
        "--hidden", row.get("hidden", "512"),
        "--flow-substeps", row.get("flow_substeps", "4"),
        "--chunk-size", row.get("chunk_size", "3"),
        "--done-loss-weight", row.get("done_loss_weight", "0.1"),
        "--sigreg-weight", row.get("sigreg_weight", "0.0"),
        "--sigreg-projections", row.get("sigreg_projections", "128"),
        "--sigreg-knots", row.get("sigreg_knots", "8"),
        "--sigreg-bandwidth", row.get("sigreg_bandwidth", "1.0"),
        "--wandb-project", row["wandb_project"],
        "--wandb-group", row["wandb_group"],
        "--wandb-name", f"{row['stage']}_{task_key}_{row['method']}_seed{row['seed']}",
    ]
    if row.get("disable_wandb", "").lower() in {"1", "true", "yes"}:
        cmd.append("--disable-wandb")
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
