#!/usr/bin/env python3
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
    parser.add_argument("--wandb-project", default="flow-mbpo-phase16-wm-capacity-sidecar")
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

    seeds = [0] if args.mode == "smoke" else [0, 1, 2]
    train_iters = args.train_iters if args.mode == "formal" else min(args.train_iters, 120)
    eval_every = args.eval_every if args.mode == "formal" else min(args.eval_every, 20)
    log_every = args.log_every if args.mode == "formal" else min(args.log_every, 10)
    batch_size = args.batch_size if args.mode == "formal" else min(args.batch_size, 32)

    profiles = [
        {
            "method_key": "mlpwm_mlppolicy",
            "method_label": "MLP WM Ref",
            "profile": "mlp_ref",
            "alg_config": "scripts/cfg/alg/pwm_5M_baseline_pwmorig.yaml",
        },
        {
            "method_key": "mlpwm_mlppolicy",
            "method_label": "Ensemble MLP WM",
            "profile": "mlp_ensemble5",
            "alg_config": "scripts/cfg/alg/pwm_5M_baseline_ensemble5.yaml",
        },
        {
            "method_key": "flowwm_mlppolicy",
            "method_label": "Flow WM Ref",
            "profile": "flow_ref_uniform_heun4",
            "alg_config": "scripts/cfg/alg/pwm_5M_flow_v2_substeps4.yaml",
        },
        {
            "method_key": "flowwm_mlppolicy",
            "method_label": "Flow WM Wide1024",
            "profile": "flow_capacity_wide1024",
            "alg_config": "scripts/cfg/alg/pwm_5M_flow_v2_substeps4_wide1024.yaml",
        },
        {
            "method_key": "flowwm_mlppolicy",
            "method_label": "Flow Path Cosine",
            "profile": "flow_path_cosine",
            "alg_config": "scripts/cfg/alg/pwm_5M_flow_v2_substeps4_path_cosine.yaml",
        },
        {
            "method_key": "flowwm_mlppolicy",
            "method_label": "Flow Target EndpointResidual",
            "profile": "flow_target_endpointres",
            "alg_config": "scripts/cfg/alg/pwm_5M_flow_v2_substeps4_target_endpointres.yaml",
        },
    ]

    stage = f"phase16_capacity_sidecar_{args.mode}"
    rows = []
    for prof in profiles:
        for seed in seeds:
            rows.append(
                {
                    "phase": "phase16",
                    "stage": stage,
                    "task_key": args.task_key,
                    "method_key": prof["method_key"],
                    "method_label": prof["method_label"],
                    "profile": prof["profile"],
                    "seed": seed,
                    "dataset": args.dataset,
                    "dataset_metadata": args.dataset_metadata,
                    "alg_config": prof["alg_config"],
                    "train_iters": train_iters,
                    "batch_size": batch_size,
                    "eval_every": eval_every,
                    "log_every": log_every,
                    "split_seed": args.split_seed,
                    "wandb_project": args.wandb_project,
                    "wandb_group": f"phase16_{args.mode}_{args.task_key}",
                    "wandb_job_type": args.mode,
                    "disable_wandb": str(args.mode == "smoke").lower(),
                }
            )

    with output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} rows to {output}")


if __name__ == "__main__":
    main()
