#!/usr/bin/env python3
"""Build Phase 1.9 advanced WM smoke and formal manifests."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


DATASET_KEYS = {
    "velocity_flat_unitree_go1_teacher",
    "velocity_flat_unitree_g1_teacher",
}

PROFILES = [
    ("mlpwm_mlppolicy", "MLP WM Ref", "mlp_ref", "scripts/cfg/alg/pwm_5M_baseline_pwmorig.yaml", False),
    ("chunkwm_mlppolicy", "Chunk2 MLP WM", "chunk2_mlp", "scripts/cfg/alg/pwm_5M_chunk2_mlp.yaml", False),
    ("residualflow_mlppolicy", "Residual Flow Frozen MLP", "residual_flow_frozen_mlp", "scripts/cfg/alg/pwm_5M_residual_flow_frozen_mlp.yaml", False),
    ("transformerwm_mlppolicy", "Latent Transformer One-Step", "latent_transformer_onestep", "scripts/cfg/alg/pwm_5M_latent_transformer_onestep.yaml", True),
    ("transformerwm_mlppolicy", "Chunk2 Latent Transformer", "chunk2_latent_transformer", "scripts/cfg/alg/pwm_5M_chunk2_latent_transformer.yaml", True),
    ("mlpwm_mlppolicy", "MLP WM RolloutCons2", "mlp_rolloutcons2", "scripts/cfg/alg/pwm_5M_baseline_rolloutcons2.yaml", True),
    ("chunkwm_mlppolicy", "Chunk2 MLP RolloutCons2", "chunk2_mlp_rolloutcons2", "scripts/cfg/alg/pwm_5M_chunk2_mlp_rolloutcons2.yaml", True),
    ("residualflow_mlppolicy", "Residual Flow Frozen MLP RolloutCons2", "residual_flow_frozen_mlp_rolloutcons2", "scripts/cfg/alg/pwm_5M_residual_flow_frozen_mlp_rolloutcons2.yaml", True),
    ("gatedwm_mlppolicy", "Gated Residual MLP x2", "gated_residual_mlp2", "scripts/cfg/alg/pwm_5M_gated_residual_mlp2.yaml", True),
    ("gatedwm_mlppolicy", "Gated Residual Flow x2", "gated_residual_flow2", "scripts/cfg/alg/pwm_5M_gated_residual_flow2.yaml", True),
    ("latentactionwm_mlppolicy", "Chunk2 Latent Action MLP", "chunk2_latent_action_mlp", "scripts/cfg/alg/pwm_5M_chunk2_latent_action_mlp.yaml", True),
    ("latentactionwm_mlppolicy", "Chunk2 Latent Action Transformer", "chunk2_latent_action_transformer", "scripts/cfg/alg/pwm_5M_chunk2_latent_action_transformer.yaml", True),
    ("latentactionwm_mlppolicy", "Chunk2 Latent Action Residual Flow", "chunk2_latent_action_residual_flow", "scripts/cfg/alg/pwm_5M_chunk2_latent_action_residual_flow.yaml", True),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-summary-csv", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--wandb-project", default="flow-mbpo-phase19-transformer-rollout-wm")
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
    stage = f"phase19_advanced_wm_{args.mode}"

    with open(args.dataset_summary_csv, newline="", encoding="utf-8") as f:
        datasets = [row for row in csv.DictReader(f) if row["task_key"] in DATASET_KEYS]
    if not datasets:
        raise RuntimeError(f"No Phase 1.9 datasets found in {args.dataset_summary_csv}")

    profiles = PROFILES if args.mode == "formal" else [p for p in PROFILES if p[4]]
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
        for method_key, method_label, profile, alg_config, _ in profiles:
            for seed in seeds:
                rows.append(
                    {
                        "phase": "phase19",
                        "stage": stage,
                        "task_key": task_key,
                        "dataset_key": dataset_key,
                        "method_key": method_key,
                        "method_label": method_label,
                        "profile": profile,
                        "seed": seed,
                        "dataset": dataset,
                        "dataset_metadata": metadata,
                        "alg_config": alg_config,
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
