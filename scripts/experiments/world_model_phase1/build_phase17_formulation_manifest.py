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
    parser.add_argument("--dataset-key", default="phase1_velocity_flat_unitree_go1_random_seed0")
    parser.add_argument("--wandb-project", default="flow-mbpo-phase17-wm-formulation-and-dataset-reruns")
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
        ("mlpwm_mlppolicy", "MLP WM Ref", "mlp_ref", "scripts/cfg/alg/pwm_5M_baseline_pwmorig.yaml"),
        ("mlpwm_mlppolicy", "Probabilistic MLP WM", "mlp_probabilistic", "scripts/cfg/alg/pwm_5M_baseline_probabilistic.yaml"),
        ("mlpwm_mlppolicy", "Ensemble MLP WM", "mlp_ensemble5", "scripts/cfg/alg/pwm_5M_baseline_ensemble5.yaml"),
        ("flowwm_mlppolicy", "Flow WM Ref", "flow_ref_uniform_heun4", "scripts/cfg/alg/pwm_5M_flow_v2_substeps4.yaml"),
        ("flowwm_mlppolicy", "Flow Target EndpointResidual", "flow_target_endpointres", "scripts/cfg/alg/pwm_5M_flow_v2_substeps4_target_endpointres.yaml"),
        ("flowwm_mlppolicy", "Flow Path Cosine", "flow_path_cosine", "scripts/cfg/alg/pwm_5M_flow_v2_substeps4_path_cosine.yaml"),
        ("flowwm_mlppolicy", "Flow WM Wide1024", "flow_capacity_wide1024", "scripts/cfg/alg/pwm_5M_flow_v2_substeps4_wide1024.yaml"),
        ("flowwm_mlppolicy", "Flow EndpointRes + AuxNext", "flow_endpointres_auxnext", "scripts/cfg/alg/pwm_5M_flow_v2_substeps4_endpointres_auxnext.yaml"),
        ("flowwm_mlppolicy", "Flow Anchored Consistency", "flow_anchoredcons4", "scripts/cfg/alg/pwm_5M_flow_v2_substeps4_anchoredcons4.yaml"),
        ("flowwm_mlppolicy", "Flow Path EndpointSigmoid", "flow_path_endpointsigmoid", "scripts/cfg/alg/pwm_5M_flow_v2_substeps4_path_endpointsigmoid.yaml"),
        ("flowwm_mlppolicy", "Flow Target DirMag", "flow_target_dirmag", "scripts/cfg/alg/pwm_5M_flow_v2_substeps4_target_dirmag.yaml"),
        ("flowwm_mlppolicy", "Flow Latent WhitenReg", "flow_latentwhiten_reg", "scripts/cfg/alg/pwm_5M_flow_v2_substeps4_latentwhiten_reg.yaml"),
        ("flowwm_mlppolicy", "Ensemble Flow WM", "flow_ensemble3", "scripts/cfg/alg/pwm_5M_flow_v2_substeps4_ensemble3.yaml"),
        ("flowwm_mlppolicy", "Flow Cycle Consistency", "flow_cyclecons1", "scripts/cfg/alg/pwm_5M_flow_v2_substeps4_cyclecons1.yaml"),
        ("flowwm_mlppolicy", "Flow Stochastic Interpolant", "flow_stochasticinterp", "scripts/cfg/alg/pwm_5M_flow_v2_substeps4_stochasticinterp.yaml"),
        ("flowwm_mlppolicy", "Flow OT Barycenter", "flow_otbarycenter", "scripts/cfg/alg/pwm_5M_flow_v2_substeps4_otbarycenter.yaml"),
    ]

    stage = f"phase17_formulation_{args.mode}"
    rows = []
    for method_key, method_label, profile, alg_config in profiles:
        for seed in seeds:
            rows.append(
                {
                    "phase": "phase17",
                    "stage": stage,
                    "task_key": args.task_key,
                    "dataset_key": args.dataset_key,
                    "method_key": method_key,
                    "method_label": method_label,
                    "profile": profile,
                    "seed": seed,
                    "dataset": args.dataset,
                    "dataset_metadata": args.dataset_metadata,
                    "alg_config": alg_config,
                    "train_iters": train_iters,
                    "batch_size": batch_size,
                    "eval_every": eval_every,
                    "log_every": log_every,
                    "split_seed": args.split_seed,
                    "wandb_project": args.wandb_project,
                    "wandb_group": f"phase17_formulation_{args.task_key}_{args.dataset_key}",
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
