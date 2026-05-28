#!/usr/bin/env python3
"""Build MJLab-QS policy extraction manifests from trained WM checkpoints."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import List


TASK_IDS = {
    "velocity_flat_unitree_g1": "Mjlab-Velocity-Flat-Unitree-G1",
    "velocity_flat_unitree_go1": "Mjlab-Velocity-Flat-Unitree-Go1",
}
SAMPLING_MODES = ["quality_balanced", "uniform", "yaw_balanced"]


def csv_list(raw: str) -> List[str]:
    return [x.strip() for x in raw.split(",") if x.strip()]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--stage", required=True, help="Policy extraction stage name.")
    p.add_argument("--wm-stage", required=True, help="Stage containing WM training outputs.")
    p.add_argument("--dataset-stage", required=True, help="Stage containing MJLab-QS windows.")
    p.add_argument("--output", required=True)
    p.add_argument("--root", default="scripts/outputs/mjlab_qs")
    p.add_argument("--tasks", default="velocity_flat_unitree_g1")
    p.add_argument("--wm-methods", default="mlp_ref,flow_endpoint")
    p.add_argument("--policy-types", default="mlp,flow")
    p.add_argument("--seeds", default="0,1,2")
    p.add_argument("--policy-iters", type=int, default=50000)
    p.add_argument("--eval-every", type=int, default=2500)
    p.add_argument("--eval-episodes", type=int, default=40)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--actor-lr", default="")
    p.add_argument("--critic-lr", default="")
    p.add_argument("--bc-lr", default="")
    p.add_argument("--critic-iterations", default="")
    p.add_argument("--bc-warmstart-iters", type=int, default=0)
    p.add_argument("--policy-bc-reg", type=float, default=0.0)
    p.add_argument("--bc-quality-filter", default="")
    p.add_argument("--policy-quality-filter", default="")
    p.add_argument("--bc-sampling", choices=SAMPLING_MODES, default="quality_balanced")
    p.add_argument("--policy-sampling", choices=SAMPLING_MODES, default="quality_balanced")
    p.add_argument("--bc-action-rate-reg", type=float, default=0.0)
    p.add_argument("--online-finetune-rounds", type=int, default=0)
    p.add_argument("--compute-profile", default="")
    p.add_argument("--wandb-project", default="flow-mbpo-mjlab-pwm-flow-endpoint")
    p.add_argument("--skip-real-eval", action="store_true")
    p.add_argument("--disable-wandb", action="store_true")
    p.add_argument("--allow-missing", action="store_true")
    return p.parse_args()


def require(path: Path, allow_missing: bool) -> None:
    if path.exists() or allow_missing:
        return
    raise FileNotFoundError(str(path))


def main() -> None:
    args = parse_args()
    root = Path(args.root)
    tasks = csv_list(args.tasks)
    wm_methods = csv_list(args.wm_methods)
    policy_types = csv_list(args.policy_types)
    seeds = [int(x) for x in csv_list(args.seeds)]
    rows = []

    for task in tasks:
        if task not in TASK_IDS:
            raise RuntimeError(f"Unknown task key: {task}")
        dataset = root / "windows" / args.dataset_stage / task / "d_qs_core_h16.pt"
        metadata = dataset.with_suffix(".json")
        normalization = dataset.with_name(dataset.stem + "_normalization.json")
        for path in (dataset, metadata, normalization):
            require(path, args.allow_missing)
        for wm_method in wm_methods:
            for policy_type in policy_types:
                for seed in seeds:
                    wm_checkpoint = root / "results" / args.wm_stage / task / wm_method / f"seed_{seed}" / "best.pt"
                    require(wm_checkpoint, args.allow_missing)
                    compute_profile = args.compute_profile or f"policy{args.policy_iters // 1000}k"
                    rows.append(
                        {
                            "stage": args.stage,
                            "task_key": task,
                            "task_id": TASK_IDS[task],
                            "dataset": str(dataset),
                            "metadata": str(metadata),
                            "normalization": str(normalization),
                            "wm_checkpoint": str(wm_checkpoint),
                            "wm_method": wm_method,
                            "policy_type": policy_type,
                            "seed": str(seed),
                            "compute_profile": compute_profile,
                            "online_profile": "online" if args.online_finetune_rounds else "offline",
                            "policy_iters": str(args.policy_iters),
                            "batch_size": str(args.batch_size),
                            "actor_lr": args.actor_lr,
                            "critic_lr": args.critic_lr,
                            "bc_lr": args.bc_lr,
                            "critic_iterations": args.critic_iterations,
                            "eval_every": str(args.eval_every),
                            "eval_episodes": str(args.eval_episodes),
                            "bc_warmstart_iters": str(args.bc_warmstart_iters),
                            "policy_bc_reg": str(args.policy_bc_reg),
                            "bc_quality_filter": args.bc_quality_filter,
                            "policy_quality_filter": args.policy_quality_filter,
                            "bc_sampling": args.bc_sampling,
                            "policy_sampling": args.policy_sampling,
                            "bc_action_rate_reg": str(args.bc_action_rate_reg),
                            "online_finetune_rounds": str(args.online_finetune_rounds),
                            "wandb_project": args.wandb_project,
                            "wandb_group": f"{args.stage}_{task}",
                            "skip_real_eval": str(bool(args.skip_real_eval)).lower(),
                            "disable_wandb": str(bool(args.disable_wandb)).lower(),
                        }
                    )

    if not rows:
        raise RuntimeError("No manifest rows built.")
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} policy extraction rows to {output}")


if __name__ == "__main__":
    main()
