#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-summary-csv", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--wandb-project", default="flow-mbpo-phase2-diffusion-sidecar")
    parser.add_argument("--mode", choices=["smoke", "formal"], default="formal")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    seeds = [0] if args.mode == "smoke" else [0, 1, 2]
    budgets = [250] if args.mode == "smoke" else [250, 1000, 2000]
    batch_size = 32 if args.mode == "smoke" else 128
    eval_every = 20 if args.mode == "smoke" else 100
    log_every = 10 if args.mode == "smoke" else 50

    wanted = {"velocity_flat_unitree_go1_teacher", "velocity_flat_unitree_g1_teacher"}
    datasets = [row for row in csv.DictReader(open(args.dataset_summary_csv, newline="", encoding="utf-8")) if row["task_key"] in wanted]

    rows = []
    stage = f"phase2_diffusion_sidecar_{args.mode}"
    for ds in datasets:
        dataset_key = ds["task_key"]
        task_key = dataset_key[: -len("_teacher")]
        dataset = f"scripts/outputs/world_model_phase1/datasets/phase1_{dataset_key}_seed0.pt"
        dataset_metadata = f"scripts/outputs/world_model_phase1/datasets/phase1_{dataset_key}_seed0.json"
        for budget in budgets:
            budget_profile = f"diffusion_wm_ref_it{budget:04d}"
            for seed in seeds:
                rows.append(
                    {
                        "phase": "phase2",
                        "stage": stage,
                        "task_key": task_key,
                        "dataset_key": dataset_key,
                        "method_key": "diffwm_mlppolicy",
                        "method_label": "Diffusion WM",
                        "profile": budget_profile,
                        "seed": seed,
                        "dataset": dataset,
                        "dataset_metadata": dataset_metadata,
                        "alg_config": "scripts/cfg/alg/pwm_5M_baseline_diffusion8.yaml",
                        "train_iters": budget,
                        "batch_size": batch_size,
                        "eval_every": min(eval_every, budget),
                        "log_every": min(log_every, budget),
                        "split_seed": 0,
                        "wandb_project": args.wandb_project,
                        "wandb_group": f"{stage}_{task_key}_{dataset_key}_it{budget:04d}",
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
