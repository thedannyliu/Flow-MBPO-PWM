#!/usr/bin/env python3
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
    return sum(xs) / len(xs)


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
        rows[(task_key, dataset_key, method_key, profile)].append(d)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "stage",
        "task_key",
        "dataset_key",
        "method_key",
        "profile",
        "n_eval_completed",
        "train_wm_loss_mean",
        "train_one_step_dyn_loss_mean",
        "train_rollout_dyn_loss_mean",
        "val_wm_loss_mean",
        "val_one_step_dyn_loss_mean",
        "val_rollout_dyn_loss_mean",
        "train_one_step_reward_loss_mean",
        "val_one_step_reward_loss_mean",
        "elapsed_seconds_mean",
    ]
    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for (task_key, dataset_key, method_key, profile), ds in sorted(rows.items()):
            writer.writerow(
                {
                    "stage": args.stage,
                    "task_key": task_key,
                    "dataset_key": dataset_key,
                    "method_key": method_key,
                    "profile": profile,
                    "n_eval_completed": len(ds),
                    "train_wm_loss_mean": mean([x["final_train"]["wm_loss"] for x in ds]),
                    "train_one_step_dyn_loss_mean": mean([x["final_train"]["one_step_dyn_loss"] for x in ds]),
                    "train_rollout_dyn_loss_mean": mean([x["final_train"]["rollout_dyn_loss"] for x in ds]),
                    "val_wm_loss_mean": mean([x["final_val"]["wm_loss"] for x in ds]),
                    "val_one_step_dyn_loss_mean": mean([x["final_val"]["one_step_dyn_loss"] for x in ds]),
                    "val_rollout_dyn_loss_mean": mean([x["final_val"]["rollout_dyn_loss"] for x in ds]),
                    "train_one_step_reward_loss_mean": mean([x["final_train"]["one_step_reward_loss"] for x in ds]),
                    "val_one_step_reward_loss_mean": mean([x["final_val"]["one_step_reward_loss"] for x in ds]),
                    "elapsed_seconds_mean": mean([x["elapsed_seconds"] for x in ds]),
                }
            )
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
