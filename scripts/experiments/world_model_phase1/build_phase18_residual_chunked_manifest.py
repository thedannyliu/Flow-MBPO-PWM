#!/usr/bin/env python3
"""Build Phase 1.8 residual/chunked WM smoke and formal manifests."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


DATASET_KEYS = {
    "velocity_flat_unitree_go1_teacher",
    "velocity_flat_unitree_g1_teacher",
}

ALL_PROFILES = [
    {
        "method_key": "mlpwm_mlppolicy",
        "method_label": "MLP WM Ref",
        "profile": "mlp_ref",
        "alg_config": "scripts/cfg/alg/pwm_5M_baseline_pwmorig.yaml",
        "is_new_codepath": False,
    },
    {
        "method_key": "mlpwm_mlppolicy",
        "method_label": "Ensemble MLP WM x5",
        "profile": "mlp_ensemble5",
        "alg_config": "scripts/cfg/alg/pwm_5M_baseline_ensemble5.yaml",
        "is_new_codepath": False,
    },
    {
        "method_key": "flowwm_mlppolicy",
        "method_label": "Flow WM Ref",
        "profile": "flow_ref_uniform_heun4",
        "alg_config": "scripts/cfg/alg/pwm_5M_flow_v2_substeps4.yaml",
        "is_new_codepath": False,
    },
    {
        "method_key": "flowwm_mlppolicy",
        "method_label": "Flow Target EndpointResidual",
        "profile": "flow_target_endpointres",
        "alg_config": "scripts/cfg/alg/pwm_5M_flow_v2_substeps4_target_endpointres.yaml",
        "is_new_codepath": False,
    },
    {
        "method_key": "residualflow_mlppolicy",
        "method_label": "Residual Flow Frozen MLP",
        "profile": "residual_flow_frozen_mlp",
        "alg_config": "scripts/cfg/alg/pwm_5M_residual_flow_frozen_mlp.yaml",
        "is_new_codepath": True,
    },
    {
        "method_key": "residualflow_mlppolicy",
        "method_label": "Residual Flow Joint",
        "profile": "residual_flow_joint",
        "alg_config": "scripts/cfg/alg/pwm_5M_residual_flow_joint.yaml",
        "is_new_codepath": True,
    },
    {
        "method_key": "chunkwm_mlppolicy",
        "method_label": "Chunk2 MLP WM",
        "profile": "chunk2_mlp",
        "alg_config": "scripts/cfg/alg/pwm_5M_chunk2_mlp.yaml",
        "is_new_codepath": True,
    },
    {
        "method_key": "chunkresidualflow_mlppolicy",
        "method_label": "Chunk2 Residual Flow WM",
        "profile": "chunk2_residual_flow",
        "alg_config": "scripts/cfg/alg/pwm_5M_chunk2_residual_flow.yaml",
        "is_new_codepath": True,
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-summary-csv", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--wandb-project", default="flow-mbpo-phase18-residual-chunked-wm")
    parser.add_argument("--train-iters", type=int, default=2000)
    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--split-seed", type=int, default=0)
    parser.add_argument("--mode", choices=["smoke", "formal"], default="formal")
    return parser.parse_args()


def task_from_dataset_key(dataset_key: str) -> str:
    for suffix in ("_teacher", "_mixed", "_random"):
        if dataset_key.endswith(suffix):
            return dataset_key[: -len(suffix)]
    return dataset_key


def main() -> None:
    args = parse_args()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    seeds = [0] if args.mode == "smoke" else [0, 1, 2]
    train_iters = args.train_iters if args.mode == "formal" else min(args.train_iters, 120)
    eval_every = args.eval_every if args.mode == "formal" else min(args.eval_every, 20)
    log_every = args.log_every if args.mode == "formal" else min(args.log_every, 10)
    batch_size = args.batch_size if args.mode == "formal" else min(args.batch_size, 32)
    stage = f"phase18_residual_chunked_{args.mode}"

    with open(args.dataset_summary_csv, newline="", encoding="utf-8") as f:
        datasets = [row for row in csv.DictReader(f) if row["task_key"] in DATASET_KEYS]
    if not datasets:
        raise RuntimeError(f"No Phase 1.8 datasets found in {args.dataset_summary_csv}")

    profiles = [p for p in ALL_PROFILES if args.mode == "formal" or p["is_new_codepath"]]
    rows = []
    for ds in sorted(datasets, key=lambda x: x["task_key"]):
        dataset_key = ds["task_key"]
        task_key = task_from_dataset_key(dataset_key)
        if args.mode == "smoke":
            dataset = f"scripts/outputs/world_model_phase1/datasets/phase1_smoke_{dataset_key}_seed0.pt"
            metadata = f"scripts/outputs/world_model_phase1/datasets/phase1_smoke_{dataset_key}_seed0.json"
        else:
            dataset = f"scripts/outputs/world_model_phase1/datasets/phase1_{dataset_key}_seed0.pt"
            metadata = f"scripts/outputs/world_model_phase1/datasets/phase1_{dataset_key}_seed0.json"
        for prof in profiles:
            for seed in seeds:
                rows.append(
                    {
                        "phase": "phase18",
                        "stage": stage,
                        "task_key": task_key,
                        "dataset_key": dataset_key,
                        "method_key": prof["method_key"],
                        "method_label": prof["method_label"],
                        "profile": prof["profile"],
                        "seed": seed,
                        "dataset": dataset,
                        "dataset_metadata": metadata,
                        "alg_config": prof["alg_config"],
                        "train_iters": train_iters,
                        "batch_size": batch_size,
                        "eval_every": eval_every,
                        "log_every": log_every,
                        "split_seed": args.split_seed,
                        "wandb_project": args.wandb_project,
                        "wandb_group": f"{stage}_{task_key}_{dataset_key}",
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
