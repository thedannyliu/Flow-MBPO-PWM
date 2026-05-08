#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-summary-csv", required=True)
    parser.add_argument("--output", required=True)
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
    ]

    datasets = list(csv.DictReader(open(args.dataset_summary_csv, newline="", encoding="utf-8")))
    stage = f"phase17_dataset_rerun_{args.mode}"
    rows = []
    for ds in datasets:
        dataset_key = ds["task_key"]
        if dataset_key.endswith("_teacher"):
            task_key = dataset_key[: -len("_teacher")]
        elif dataset_key.endswith("_mixed"):
            task_key = dataset_key[: -len("_mixed")]
        elif dataset_key.endswith("_random"):
            task_key = dataset_key[: -len("_random")]
        else:
            task_key = dataset_key
        dataset = f"scripts/outputs/world_model_phase1/datasets/phase1_{dataset_key}_seed0.pt"
        dataset_metadata = f"scripts/outputs/world_model_phase1/datasets/phase1_{dataset_key}_seed0.json"
        for method_key, method_label, profile, alg_config in profiles:
            for seed in seeds:
                rows.append(
                    {
                        "phase": "phase17",
                        "stage": stage,
                        "task_key": task_key,
                        "dataset_key": dataset_key,
                        "method_key": method_key,
                        "method_label": method_label,
                        "profile": profile,
                        "seed": seed,
                        "dataset": dataset,
                        "dataset_metadata": dataset_metadata,
                        "alg_config": alg_config,
                        "train_iters": train_iters,
                        "batch_size": batch_size,
                        "eval_every": eval_every,
                        "log_every": log_every,
                        "split_seed": args.split_seed,
                        "wandb_project": args.wandb_project,
                        "wandb_group": f"phase17_dataset_rerun_{task_key}_{dataset_key}",
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
