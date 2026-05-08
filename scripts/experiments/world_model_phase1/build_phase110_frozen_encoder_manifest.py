#!/usr/bin/env python3
"""Build Phase 1.10 frozen-encoder fixed-latent sanity-check manifests."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


DATASET_KEYS = [
    "velocity_flat_unitree_g1_teacher",
    "velocity_flat_unitree_go1_teacher",
]

PROFILES = [
    (
        "mlpwm_mlppolicy",
        "MLP WM Ref Fixed Encoder",
        "mlp_ref_fixedenc",
        "scripts/cfg/alg/pwm_5M_baseline_pwmorig.yaml",
    ),
    (
        "flowwm_mlppolicy",
        "Flow WM Ref Fixed Encoder",
        "flow_ref_uniform_heun4_fixedenc",
        "scripts/cfg/alg/pwm_5M_flow_v2_substeps4.yaml",
    ),
    (
        "flowwm_mlppolicy",
        "Flow Target EndpointResidual Fixed Encoder",
        "flow_target_endpointres_fixedenc",
        "scripts/cfg/alg/pwm_5M_flow_v2_substeps4_target_endpointres.yaml",
    ),
    (
        "residualflow_mlppolicy",
        "Residual Flow Joint Fixed Encoder",
        "residual_flow_joint_fixedenc",
        "scripts/cfg/alg/pwm_5M_residual_flow_joint.yaml",
    ),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--wandb-project", default="flow-mbpo-phase110-frozen-encoder-wm")
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


def frozen_encoder_checkpoint(task_key: str, dataset_key: str) -> str:
    return (
        "scripts/outputs/world_model_phase1/phase18_residual_chunked_formal/"
        f"{task_key}/{dataset_key}/mlpwm_mlppolicy/mlp_ref/seed_0/best_world_model.pt"
    )


def main() -> None:
    args = parse_args()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    seeds = [0] if args.mode == "smoke" else [0, 1, 2]
    train_iters = args.train_iters if args.mode == "formal" else min(args.train_iters, 120)
    eval_every = args.eval_every if args.mode == "formal" else min(args.eval_every, 20)
    log_every = args.log_every if args.mode == "formal" else min(args.log_every, 10)
    batch_size = args.batch_size if args.mode == "formal" else min(args.batch_size, 32)
    stage = f"phase110_frozen_encoder_{args.mode}"

    rows = []
    for dataset_key in DATASET_KEYS:
        task_key = task_from_dataset_key(dataset_key)
        if args.mode == "smoke":
            dataset = f"scripts/outputs/world_model_phase1/datasets/phase1_smoke_{dataset_key}_seed0.pt"
            metadata = f"scripts/outputs/world_model_phase1/datasets/phase1_smoke_{dataset_key}_seed0.json"
        else:
            dataset = f"scripts/outputs/world_model_phase1/datasets/phase1_{dataset_key}_seed0.pt"
            metadata = f"scripts/outputs/world_model_phase1/datasets/phase1_{dataset_key}_seed0.json"

        encoder_ckpt = frozen_encoder_checkpoint(task_key, dataset_key)
        if not Path(encoder_ckpt).exists():
            raise FileNotFoundError(f"Missing frozen encoder checkpoint: {encoder_ckpt}")
        if not Path(dataset).exists():
            raise FileNotFoundError(f"Missing dataset: {dataset}")
        if not Path(metadata).exists():
            raise FileNotFoundError(f"Missing metadata: {metadata}")

        for method_key, method_label, profile, alg_config in PROFILES:
            for seed in seeds:
                rows.append(
                    {
                        "phase": "phase110",
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
                        "frozen_encoder_checkpoint": encoder_ckpt,
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
