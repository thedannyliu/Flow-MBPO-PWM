#!/usr/bin/env python3
"""Export Phase 1.9 advanced WM results."""

from __future__ import annotations

import argparse
import csv
import glob
import json
from collections import defaultdict
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def mean(xs):
    return sum(xs) / len(xs) if xs else ""


def metric(summary: dict, split: str, key: str):
    return summary.get(split, {}).get(key, 0.0)


def main() -> None:
    args = parse_args()
    pattern = f"scripts/outputs/world_model_phase1/{args.stage}/*/*/*/*/seed_*/phase1_summary.json"
    rows = defaultdict(list)
    for path in sorted(glob.glob(pattern)):
        d = json.load(open(path))
        parts = Path(path).parts
        task_key = parts[-6]
        dataset_key = parts[-5]
        method_key = parts[-4]
        profile = parts[-3]
        seed = parts[-2].replace("seed_", "")
        rows[(task_key, dataset_key, method_key, profile)].append((seed, d))

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "stage",
        "task_key",
        "dataset_key",
        "method_key",
        "profile",
        "seeds_completed",
        "n_eval_completed",
        "train_wm_loss_mean",
        "val_wm_loss_mean",
        "train_one_step_dyn_loss_mean",
        "val_one_step_dyn_loss_mean",
        "train_rollout_dyn_loss_mean",
        "val_rollout_dyn_loss_mean",
        "train_one_step_reward_loss_mean",
        "val_one_step_reward_loss_mean",
        "train_rollout_reward_loss_mean",
        "val_rollout_reward_loss_mean",
        "train_base_rollout_dyn_loss_mean",
        "val_base_rollout_dyn_loss_mean",
        "train_residual_contribution_mean",
        "val_residual_contribution_mean",
        "train_chunk_endpoint_dyn_loss_mean",
        "val_chunk_endpoint_dyn_loss_mean",
        "train_chunk_rollout_dyn_loss_mean",
        "val_chunk_rollout_dyn_loss_mean",
        "train_gate_entropy_mean",
        "val_gate_entropy_mean",
        "train_gate_usage_max_mean",
        "val_gate_usage_max_mean",
        "train_latent_action_norm_mean",
        "val_latent_action_norm_mean",
        "best_val_rollout_dyn_loss_mean",
        "elapsed_seconds_mean",
    ]
    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for (task_key, dataset_key, method_key, profile), items in sorted(rows.items()):
            seeds = sorted(seed for seed, _ in items)
            ds = [d for _, d in items]
            writer.writerow(
                {
                    "stage": args.stage,
                    "task_key": task_key,
                    "dataset_key": dataset_key,
                    "method_key": method_key,
                    "profile": profile,
                    "seeds_completed": ";".join(seeds),
                    "n_eval_completed": len(ds),
                    "train_wm_loss_mean": mean([metric(x, "final_train", "wm_loss") for x in ds]),
                    "val_wm_loss_mean": mean([metric(x, "final_val", "wm_loss") for x in ds]),
                    "train_one_step_dyn_loss_mean": mean([metric(x, "final_train", "one_step_dyn_loss") for x in ds]),
                    "val_one_step_dyn_loss_mean": mean([metric(x, "final_val", "one_step_dyn_loss") for x in ds]),
                    "train_rollout_dyn_loss_mean": mean([metric(x, "final_train", "rollout_dyn_loss") for x in ds]),
                    "val_rollout_dyn_loss_mean": mean([metric(x, "final_val", "rollout_dyn_loss") for x in ds]),
                    "train_one_step_reward_loss_mean": mean([metric(x, "final_train", "one_step_reward_loss") for x in ds]),
                    "val_one_step_reward_loss_mean": mean([metric(x, "final_val", "one_step_reward_loss") for x in ds]),
                    "train_rollout_reward_loss_mean": mean([metric(x, "final_train", "rollout_reward_loss") for x in ds]),
                    "val_rollout_reward_loss_mean": mean([metric(x, "final_val", "rollout_reward_loss") for x in ds]),
                    "train_base_rollout_dyn_loss_mean": mean([metric(x, "final_train", "base_rollout_dyn_loss") for x in ds]),
                    "val_base_rollout_dyn_loss_mean": mean([metric(x, "final_val", "base_rollout_dyn_loss") for x in ds]),
                    "train_residual_contribution_mean": mean([metric(x, "final_train", "residual_contribution") for x in ds]),
                    "val_residual_contribution_mean": mean([metric(x, "final_val", "residual_contribution") for x in ds]),
                    "train_chunk_endpoint_dyn_loss_mean": mean([metric(x, "final_train", "chunk_endpoint_dyn_loss") for x in ds]),
                    "val_chunk_endpoint_dyn_loss_mean": mean([metric(x, "final_val", "chunk_endpoint_dyn_loss") for x in ds]),
                    "train_chunk_rollout_dyn_loss_mean": mean([metric(x, "final_train", "chunk_rollout_dyn_loss") for x in ds]),
                    "val_chunk_rollout_dyn_loss_mean": mean([metric(x, "final_val", "chunk_rollout_dyn_loss") for x in ds]),
                    "train_gate_entropy_mean": mean([metric(x, "final_train", "gate_entropy") for x in ds]),
                    "val_gate_entropy_mean": mean([metric(x, "final_val", "gate_entropy") for x in ds]),
                    "train_gate_usage_max_mean": mean([metric(x, "final_train", "gate_usage_max") for x in ds]),
                    "val_gate_usage_max_mean": mean([metric(x, "final_val", "gate_usage_max") for x in ds]),
                    "train_latent_action_norm_mean": mean([metric(x, "final_train", "latent_action_norm") for x in ds]),
                    "val_latent_action_norm_mean": mean([metric(x, "final_val", "latent_action_norm") for x in ds]),
                    "best_val_rollout_dyn_loss_mean": mean([x["best_val_rollout_dyn_loss"] for x in ds]),
                    "elapsed_seconds_mean": mean([x["elapsed_seconds"] for x in ds]),
                }
            )
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
