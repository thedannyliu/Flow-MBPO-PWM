#!/usr/bin/env python3
"""Build smoke/formal manifests for the Phase 1 objective-ablation batch."""

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
    parser.add_argument("--wandb-project", default="flow-mbpo-phase1-wm-objective-ablations")
    parser.add_argument("--mode", choices=["smoke", "formal"], default="formal")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    seeds = [0] if args.mode == "smoke" else [0, 1, 2]
    stage = f"phase1_objective_ablations_{args.mode}"
    disable_wandb = args.mode == "smoke"
    train_iters = 200 if args.mode == "smoke" else 2000
    eval_every = 20 if args.mode == "smoke" else 100
    log_every = 10 if args.mode == "smoke" else 50
    batch_size = 32 if args.mode == "smoke" else 128

    variants = [
        {
            "method_key": "mlpwm_mlppolicy",
            "method_label": "MLP WM Ref",
            "profile": "mlp_ref",
            "alg_config": "scripts/cfg/alg/pwm_5M_baseline_pwmorig.yaml",
        },
        {
            "method_key": "flowwm_mlppolicy",
            "method_label": "Flow WM Ref",
            "profile": "flow_ref_uniform_heun4",
            "alg_config": "scripts/cfg/alg/pwm_5M_flow_v2_substeps4.yaml",
        },
        {
            "method_key": "flowwm_mlppolicy",
            "method_label": "Flow Path Smoothstep",
            "profile": "flow_path_smoothstep",
            "alg_config": "scripts/cfg/alg/pwm_5M_flow_v2_substeps4_path_smoothstep.yaml",
        },
        {
            "method_key": "flowwm_mlppolicy",
            "method_label": "Flow Path Quadratic",
            "profile": "flow_path_quadratic",
            "alg_config": "scripts/cfg/alg/pwm_5M_flow_v2_substeps4_path_quadratic.yaml",
        },
        {
            "method_key": "flowwm_mlppolicy",
            "method_label": "Flow Rollout Consistency 2",
            "profile": "flow_rollout_consistency_2step",
            "alg_config": "scripts/cfg/alg/pwm_5M_flow_v2_substeps4_rolloutcons2.yaml",
        },
        {
            "method_key": "flowwm_mlppolicy",
            "method_label": "Flow Rollout Consistency 4",
            "profile": "flow_rollout_consistency_4step",
            "alg_config": "scripts/cfg/alg/pwm_5M_flow_v2_substeps4_rolloutcons4.yaml",
        },
        {
            "method_key": "flowwm_mlppolicy",
            "method_label": "Flow Target Scale 0.5",
            "profile": "flow_target_scale0p5",
            "alg_config": "scripts/cfg/alg/pwm_5M_flow_v2_substeps4_targetscale0p5.yaml",
        },
        {
            "method_key": "flowwm_mlppolicy",
            "method_label": "Flow Target Normalized",
            "profile": "flow_target_normalized",
            "alg_config": "scripts/cfg/alg/pwm_5M_flow_v2_substeps4_targetnormalized.yaml",
        },
        {
            "method_key": "flowwm_mlppolicy",
            "method_label": "Flow Tau Midpoint",
            "profile": "flow_tau_midpoint",
            "alg_config": "scripts/cfg/alg/pwm_5M_flow_v2_substeps4_midpoint.yaml",
        },
        {
            "method_key": "flowwm_mlppolicy",
            "method_label": "Flow Tau UShaped",
            "profile": "flow_tau_ushaped",
            "alg_config": "scripts/cfg/alg/pwm_5M_flow_v2_substeps4_ushaped.yaml",
        },
    ]

    rows = []
    for variant in variants:
        for seed in seeds:
            rows.append(
                {
                    "phase": "phase1b",
                    "stage": stage,
                    "task_key": args.task_key,
                    "method_key": variant["method_key"],
                    "method_label": variant["method_label"],
                    "profile": variant["profile"],
                    "seed": seed,
                    "dataset": args.dataset,
                    "dataset_metadata": args.dataset_metadata,
                    "alg_config": variant["alg_config"],
                    "train_iters": train_iters,
                    "batch_size": batch_size,
                    "eval_every": eval_every,
                    "log_every": log_every,
                    "split_seed": 0,
                    "wandb_project": args.wandb_project,
                    "wandb_group": f"{stage}_{args.task_key}",
                    "wandb_job_type": args.mode,
                    "disable_wandb": str(disable_wandb).lower(),
                }
            )

    with output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} rows to {output}")


if __name__ == "__main__":
    main()
