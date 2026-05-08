#!/usr/bin/env python3
"""Build a targeted Phase 1 diagnostic manifest for Flow WM variants."""

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
    parser.add_argument("--wandb-project", default="flow-mbpo-phase1-wm-diagnostics")
    parser.add_argument("--mode", choices=["smoke", "formal"], default="formal")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    seeds = [0] if args.mode == "smoke" else [0, 1, 2]
    stage = f"phase1_flow_diagnostics_{args.mode}"
    train_iters_default = 2000 if args.mode == "formal" else 200
    eval_every_default = 100 if args.mode == "formal" else 20
    log_every_default = 50 if args.mode == "formal" else 10

    variants = [
        {
            "method_key": "flowwm_mlppolicy",
            "method_label": "Flow WM Ref",
            "profile": "flow_ref_uniform_heun4",
            "alg_config": "scripts/cfg/alg/pwm_5M_flow_v2_substeps4.yaml",
            "train_iters": train_iters_default,
        },
        {
            "method_key": "flowwm_mlppolicy",
            "method_label": "Flow WM Midpoint",
            "profile": "flow_midpoint_heun4",
            "alg_config": "scripts/cfg/alg/pwm_5M_flow_v2_substeps4_midpoint.yaml",
            "train_iters": train_iters_default,
        },
        {
            "method_key": "flowwm_mlppolicy",
            "method_label": "Flow WM UShaped",
            "profile": "flow_ushaped_heun4",
            "alg_config": "scripts/cfg/alg/pwm_5M_flow_v2_substeps4_ushaped.yaml",
            "train_iters": train_iters_default,
        },
        {
            "method_key": "flowwm_mlppolicy",
            "method_label": "Flow WM Endpoint",
            "profile": "flow_endpoint1_heun4",
            "alg_config": "scripts/cfg/alg/pwm_5M_flow_v2_substeps4_endpoint1.yaml",
            "train_iters": train_iters_default,
        },
        {
            "method_key": "flowwm_mlppolicy",
            "method_label": "Flow WM DynHeavy",
            "profile": "flow_dyn2_rew0p5_heun4",
            "alg_config": "scripts/cfg/alg/pwm_5M_flow_v2_substeps4_dyn2_rew0p5.yaml",
            "train_iters": train_iters_default,
        },
        {
            "method_key": "flowwm_mlppolicy",
            "method_label": "Flow WM Euler8",
            "profile": "flow_uniform_euler8",
            "alg_config": "scripts/cfg/alg/pwm_5M_flow_v2_substeps8_euler_aligned.yaml",
            "train_iters": train_iters_default,
        },
        {
            "method_key": "flowwm_mlppolicy",
            "method_label": "Flow WM MoreCompute",
            "profile": "flow_ref_uniform_heun4_long",
            "alg_config": "scripts/cfg/alg/pwm_5M_flow_v2_substeps4.yaml",
            "train_iters": 10000 if args.mode == "formal" else 500,
        },
    ]

    rows = []
    for variant in variants:
        for seed in seeds:
            rows.append(
                {
                    "phase": "phase1",
                    "stage": stage,
                    "task_key": args.task_key,
                    "method_key": variant["method_key"],
                    "method_label": variant["method_label"],
                    "profile": variant["profile"],
                    "seed": seed,
                    "dataset": args.dataset,
                    "dataset_metadata": args.dataset_metadata,
                    "alg_config": variant["alg_config"],
                    "train_iters": variant["train_iters"],
                    "batch_size": 128 if args.mode == "formal" else 32,
                    "eval_every": eval_every_default,
                    "log_every": log_every_default,
                    "split_seed": 0,
                    "wandb_project": args.wandb_project,
                    "wandb_group": f"{stage}_{args.task_key}",
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
