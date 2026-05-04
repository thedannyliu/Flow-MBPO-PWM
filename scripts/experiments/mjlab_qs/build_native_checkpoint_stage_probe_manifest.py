#!/usr/bin/env python3
"""Build quality-probe rows for intermediate native collector checkpoints."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List


TASK_IDS = {
    "velocity_flat_unitree_go1": "Mjlab-Velocity-Flat-Unitree-Go1",
    "velocity_flat_unitree_g1": "Mjlab-Velocity-Flat-Unitree-G1",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--task", required=True, choices=sorted(TASK_IDS))
    p.add_argument("--native-root", required=True)
    p.add_argument("--method", required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--checkpoints", required=True, help="Comma-separated checkpoint iteration ids.")
    p.add_argument("--output", required=True)
    p.add_argument("--stage", required=True)
    p.add_argument("--episodes", type=int, default=100)
    p.add_argument("--num-envs", type=int, default=16)
    p.add_argument("--include-random", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    task_id = TASK_IDS[args.task]
    run_dir = Path(args.native_root) / args.task / args.method / f"seed_{args.seed}"
    ckpt_ids = [int(x.strip()) for x in args.checkpoints.split(",") if x.strip()]
    if not ckpt_ids:
        raise RuntimeError("--checkpoints must be non-empty")

    rows: List[Dict[str, str]] = []
    if args.include_random:
        out = Path("scripts/outputs/mjlab_qs/raw") / args.stage / f"{args.task}_random_smooth_seed0.pt"
        rows.append(
            {
                "stage": args.stage,
                "task_key": args.task,
                "task_id": task_id,
                "method": "random_smooth",
                "checkpoint": "",
                "quality_bin": "random_smooth",
                "collector_mode": "random_smooth",
                "collector_id": "native_random_smooth_reference",
                "output": str(out),
                "metadata_output": str(out.with_suffix(".json")),
                "seed": "0",
                "num_envs": str(args.num_envs),
                "episodes": str(args.episodes),
                "episode_length": "1000",
                "teacher_blend": "1.0",
                "action_noise_std": "0.0",
                "command_dim": "3",
                "command_position": "tail",
            }
        )

    for ckpt_id in ckpt_ids:
        ckpt = run_dir / f"model_{ckpt_id}.pt"
        if not ckpt.exists():
            raise FileNotFoundError(f"Missing checkpoint: {ckpt}")
        out = Path("scripts/outputs/mjlab_qs/raw") / args.stage / f"{args.task}_{args.method}_seed{args.seed}_iter{ckpt_id}.pt"
        rows.append(
            {
                "stage": args.stage,
                "task_key": args.task,
                "task_id": task_id,
                "method": args.method,
                "checkpoint": str(ckpt),
                "quality_bin": "stage_candidate",
                "collector_mode": "checkpoint",
                "collector_id": f"native_{args.method}_seed{args.seed}_iter{ckpt_id}",
                "output": str(out),
                "metadata_output": str(out.with_suffix(".json")),
                "seed": str(args.seed),
                "num_envs": str(args.num_envs),
                "episodes": str(args.episodes),
                "episode_length": "1000",
                "teacher_blend": "1.0",
                "action_noise_std": "0.0",
                "command_dim": "3",
                "command_position": "tail",
            }
        )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} checkpoint-stage probe rows to {output}")


if __name__ == "__main__":
    main()
