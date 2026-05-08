#!/usr/bin/env python3
"""Build smoke/formal manifests for Phase 1 WM overfit experiments."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--dataset-metadata", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--task-key", default="velocity_flat_unitree_go1")
    parser.add_argument("--wandb-project", default="flow-mbpo-phase1-wm-overfit")
    parser.add_argument("--train-iters", type=int, default=2000)
    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--split-seed", type=int, default=0)
    parser.add_argument("--mode", choices=["smoke", "formal"], default="formal")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    seeds = [0] if args.mode == "smoke" else [0, 1, 2]
    methods = [
        {
            "method_key": "mlpwm_mlppolicy",
            "method_label": "MLP WM",
            "alg_config": "scripts/cfg/alg/pwm_5M_baseline_pwmorig.yaml",
        },
        {
            "method_key": "flowwm_mlppolicy",
            "method_label": "Flow WM",
            "alg_config": "scripts/cfg/alg/pwm_5M_flow_v2_substeps4.yaml",
        },
    ]

    for method in methods:
        for seed in seeds:
            rows.append(
                {
                    "phase": "phase1",
                    "stage": f"phase1_wm_overfit_{args.mode}",
                    "task_key": args.task_key,
                    "method_key": method["method_key"],
                    "method_label": method["method_label"],
                    "seed": seed,
                    "dataset": args.dataset,
                    "dataset_metadata": args.dataset_metadata,
                    "alg_config": method["alg_config"],
                    "train_iters": args.train_iters if args.mode == "formal" else min(args.train_iters, 100),
                    "batch_size": args.batch_size,
                    "eval_every": args.eval_every if args.mode == "formal" else min(args.eval_every, 20),
                    "log_every": args.log_every if args.mode == "formal" else min(args.log_every, 10),
                    "split_seed": args.split_seed,
                    "wandb_project": args.wandb_project,
                    "wandb_group": f"phase1_{args.mode}_{args.task_key}",
                    "wandb_job_type": args.mode,
                }
            )

    with output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} rows to {output}")


if __name__ == "__main__":
    main()
