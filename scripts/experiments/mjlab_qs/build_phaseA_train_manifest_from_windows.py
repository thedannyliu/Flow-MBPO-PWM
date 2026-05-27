#!/usr/bin/env python3
"""Build Phase-A training manifests from existing MJLab-QS window datasets."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import List


def parse_csv_list(raw: str) -> List[str]:
    return [x.strip() for x in raw.split(",") if x.strip()]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--stage", required=True)
    p.add_argument("--dataset-stage", default="", help="Window dataset stage. Defaults to --stage.")
    p.add_argument("--output", required=True)
    p.add_argument("--root", default="scripts/outputs/mjlab_qs")
    p.add_argument("--tasks", default="velocity_flat_unitree_go1,velocity_flat_unitree_g1")
    p.add_argument("--methods", default="mlp_ref,flow_ref,residual_flow_frozen_mlp")
    p.add_argument("--seeds", default="0,1,2")
    p.add_argument("--train-iters", type=int, default=50000)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--eval-every", type=int, default=5000)
    p.add_argument("--sigreg-weight", type=float, default=0.0)
    p.add_argument("--sigreg-projections", type=int, default=128)
    p.add_argument("--sigreg-knots", type=int, default=8)
    p.add_argument("--sigreg-bandwidth", type=float, default=1.0)
    p.add_argument("--wandb-project", default="flow-mbpo-mjlab-phaseA-wm-feasibility")
    p.add_argument("--disable-wandb", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(args.root)
    dataset_stage = args.dataset_stage or args.stage
    tasks = parse_csv_list(args.tasks)
    methods = parse_csv_list(args.methods)
    seeds = [int(x) for x in parse_csv_list(args.seeds)]
    if not tasks or not methods or not seeds:
        raise RuntimeError("tasks, methods, and seeds must be non-empty")

    rows = []
    for task in tasks:
        dataset = root / "windows" / dataset_stage / task / "d_qs_core_h16.pt"
        metadata = dataset.with_suffix(".json")
        norm = dataset.with_name(dataset.stem + "_normalization.json")
        missing = [p for p in (dataset, metadata, norm) if not p.exists()]
        if missing:
            raise FileNotFoundError(
                f"Missing window artifacts for task={task}: "
                + ", ".join(str(p) for p in missing)
            )
        for method in methods:
            for seed in seeds:
                rows.append(
                    {
                        "stage": args.stage,
                        "task_key": task,
                        "method": method,
                        "seed": str(seed),
                        "dataset": str(dataset),
                        "metadata": str(metadata),
                        "normalization": str(norm),
                        "train_iters": str(args.train_iters),
                        "base_pretrain_iters": str(
                            args.train_iters if method == "residual_flow_frozen_mlp" else 0
                        ),
                        "batch_size": str(args.batch_size),
                        "eval_every": str(args.eval_every),
                        "sigreg_weight": str(args.sigreg_weight),
                        "sigreg_projections": str(args.sigreg_projections),
                        "sigreg_knots": str(args.sigreg_knots),
                        "sigreg_bandwidth": str(args.sigreg_bandwidth),
                        "wandb_project": args.wandb_project,
                        "wandb_group": f"{args.stage}_{task}",
                        "disable_wandb": str(bool(args.disable_wandb)).lower(),
                    }
                )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} training rows to {output}")


if __name__ == "__main__":
    main()
