#!/usr/bin/env python3
"""Build Phase-A MJLab-QS collection and A2.5 training manifests."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


TASKS = {
    "velocity_flat_unitree_go1": {
        "env_config": "scripts/cfg/env/mjlab_velocity_flat_unitree_go1.yaml",
        "checkpoint": "scripts/outputs/single_task_online/confirm_mjlab_curriculum4_strict_default/mjlab/velocity_flat_unitree_go1/mlpwm_mlppolicy/seed_0/default/logs/best_policy.pt",
    },
    "velocity_flat_unitree_g1": {
        "env_config": "scripts/cfg/env/mjlab_velocity_flat_unitree_g1.yaml",
        "checkpoint": "scripts/outputs/single_task_online/confirm_mjlab_curriculum4_strict_default/mjlab/velocity_flat_unitree_g1/mlpwm_mlppolicy/seed_0/default/logs/best_policy.pt",
    },
}

QS_BINS = [
    ("random_smooth", "random_smooth", 1.0, 0.0, 0.0, 63),
    ("weak", "checkpoint_blend_random", 0.25, 0.20, 0.25, 125),
    ("medium", "checkpoint_blend_random", 0.60, 0.10, 0.60, 219),
    ("expert", "checkpoint", 1.0, 0.0, 1.0, 157),
    ("expert_noisy", "checkpoint_noisy", 1.0, 0.05, 1.0, 63),
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["a1_smoke", "a2_mini", "a25"], required=True)
    p.add_argument("--collection-output", required=True)
    p.add_argument("--train-output", required=True)
    p.add_argument("--root", default="scripts/outputs/mjlab_qs")
    p.add_argument("--python-bin", default="python")
    p.add_argument("--wandb-project", default="flow-mbpo-mjlab-phaseA-wm-feasibility")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(args.root)
    coll_rows = []
    if args.mode == "a1_smoke":
        bins = [("random_smooth", "random_smooth", 1.0, 0.0, 0.0, 4), ("expert", "checkpoint", 1.0, 0.0, 1.0, 4)]
        num_envs = 4
    elif args.mode == "a2_mini":
        bins = [("random_smooth", "random_smooth", 1.0, 0.0, 0.0, 50), ("expert", "checkpoint", 1.0, 0.0, 1.0, 50)]
        num_envs = 16
    else:
        bins = QS_BINS
        num_envs = 32
    for task, spec in TASKS.items():
        for qbin, cmode, blend, noise, _quality, episodes in bins:
            out = root / "raw" / args.mode / f"{task}_{qbin}_seed0.pt"
            coll_rows.append(
                {
                    "stage": args.mode,
                    "task_key": task,
                    "quality_bin": qbin,
                    "env_config": spec["env_config"],
                    "output": str(out),
                    "metadata_output": str(out.with_suffix(".json")),
                    "collector_mode": cmode,
                    "collector_id": f"{cmode}_{qbin}",
                    "collector_alg_config": "scripts/cfg/alg/pwm_5M_baseline_pwmorig.yaml",
                    "collector_checkpoint": spec["checkpoint"] if cmode != "random_smooth" else "",
                    "teacher_blend": blend,
                    "action_noise_std": noise,
                    "seed": 0,
                    "num_envs": num_envs,
                    "episodes": episodes,
                    "episode_length": 1000,
                    "command_dim": 3,
                    "command_position": "tail",
                }
            )
    Path(args.collection_output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.collection_output, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(coll_rows[0].keys()))
        w.writeheader()
        w.writerows(coll_rows)

    if args.mode == "a25":
        methods = ["mlp_ref", "flow_ref", "residual_flow_frozen_mlp"]
        train_iters = 50000
    else:
        methods = ["mlp_ref", "flow_ref"]
        train_iters = 200 if args.mode == "a1_smoke" else 1000
    train_rows = []
    seeds = [0, 1, 2] if args.mode == "a25" else [0]
    for task in TASKS:
        dataset = root / "windows" / args.mode / task / "d_qs_core_h16.pt"
        metadata = dataset.with_suffix(".json")
        norm = dataset.with_name(dataset.stem + "_normalization.json")
        for method in methods:
            for seed in seeds:
                train_rows.append(
                    {
                        "stage": args.mode,
                        "task_key": task,
                        "method": method,
                        "seed": seed,
                        "dataset": str(dataset),
                        "metadata": str(metadata),
                        "normalization": str(norm),
                        "train_iters": train_iters,
                        "base_pretrain_iters": train_iters if method == "residual_flow_frozen_mlp" else 0,
                        "batch_size": 256 if args.mode == "a25" else 64,
                        "eval_every": max(50, train_iters // 10),
                        "wandb_project": args.wandb_project,
                        "wandb_group": f"{args.mode}_{task}",
                        "disable_wandb": str(args.mode != "a25").lower(),
                    }
                )
    Path(args.train_output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.train_output, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(train_rows[0].keys()))
        w.writeheader()
        w.writerows(train_rows)
    print(f"wrote collection manifest {args.collection_output} ({len(coll_rows)} rows)")
    print(f"wrote training manifest {args.train_output} ({len(train_rows)} rows)")


if __name__ == "__main__":
    main()
