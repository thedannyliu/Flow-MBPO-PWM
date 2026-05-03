#!/usr/bin/env python3
"""Build MJLab-native PPO/RSL-RL collector manifests.

These collectors are neutral data-generation policies for MJLab-QS. They are
not part of the Flow-vs-MLP method comparison.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List


@dataclass(frozen=True)
class TaskSpec:
    task_key: str
    task_id: str
    max_iterations: int


@dataclass(frozen=True)
class MethodSpec:
    name: str
    notes: str


TASKS = [
    TaskSpec(
        task_key="velocity_flat_unitree_go1",
        task_id="Mjlab-Velocity-Flat-Unitree-Go1",
        max_iterations=10000,
    ),
    TaskSpec(
        task_key="velocity_flat_unitree_g1",
        task_id="Mjlab-Velocity-Flat-Unitree-G1",
        max_iterations=30000,
    ),
]


METHODS = [
    MethodSpec(
        name="rslrl_ppo_default",
        notes="MJLab-native RSL-RL/PPO default config with only num_envs/save/logging controlled.",
    ),
    MethodSpec(
        name="rslrl_ppo_conservative",
        notes=(
            "MJLab-native RSL-RL/PPO with lower LR/KL/entropy, obs normalization, "
            "and exogenous perturbation events removed for canonical flat collection."
        ),
    ),
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--output", required=True)
    p.add_argument("--stage", default="mjlab_native_collector_v1")
    p.add_argument("--seeds", default="0,1,2")
    p.add_argument("--tasks", default="")
    p.add_argument("--methods", default="")
    p.add_argument("--num-envs", type=int, default=2048)
    p.add_argument("--max-iterations", type=int, default=-1)
    p.add_argument("--save-interval", type=int, default=250)
    p.add_argument("--logger", choices=["wandb", "tensorboard"], default="wandb")
    p.add_argument("--wandb-project", default="flow-mbpo-mjlab-native-collector")
    p.add_argument("--resume", action="store_true")
    return p.parse_args()


def parse_filter(raw: str) -> set[str] | None:
    if not raw:
        return None
    vals = {x.strip() for x in raw.split(",") if x.strip()}
    return vals or None


def select_specs(items: Iterable[Any], names: set[str] | None, attr: str) -> List[Any]:
    selected: List[Any] = []
    for item in items:
        if names is None or getattr(item, attr) in names:
            selected.append(item)
    return selected


def main() -> None:
    args = parse_args()
    seeds = [int(x.strip()) for x in args.seeds.split(",") if x.strip()]
    tasks = select_specs(TASKS, parse_filter(args.tasks), "task_key")
    methods = select_specs(METHODS, parse_filter(args.methods), "name")
    if not tasks:
        raise RuntimeError("No tasks selected.")
    if not methods:
        raise RuntimeError("No methods selected.")
    if not seeds:
        raise RuntimeError("No seeds selected.")

    rows: List[Dict[str, str]] = []
    for task in tasks:
        for method in methods:
            for seed in seeds:
                max_iterations = args.max_iterations if args.max_iterations > 0 else task.max_iterations
                run_key = f"{args.stage}_{task.task_key}_{method.name}_seed{seed}"
                output_dir = (
                    Path("scripts/outputs/mjlab_qs/native_collectors")
                    / args.stage
                    / task.task_key
                    / method.name
                    / f"seed_{seed}"
                )
                rows.append(
                    {
                        "run_key": run_key,
                        "stage": args.stage,
                        "task_key": task.task_key,
                        "task_id": task.task_id,
                        "method": method.name,
                        "seed": str(seed),
                        "num_envs": str(args.num_envs),
                        "max_iterations": str(max_iterations),
                        "save_interval": str(args.save_interval),
                        "logger": args.logger,
                        "wandb_project": args.wandb_project,
                        "wandb_group": args.stage,
                        "run_name": run_key,
                        "output_dir": str(output_dir),
                        "resume": "true" if args.resume else "false",
                        "notes": method.notes,
                    }
                )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "run_key",
        "stage",
        "task_key",
        "task_id",
        "method",
        "seed",
        "num_envs",
        "max_iterations",
        "save_interval",
        "logger",
        "wandb_project",
        "wandb_group",
        "run_name",
        "output_dir",
        "resume",
        "notes",
    ]
    with output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} rows to {output}")


if __name__ == "__main__":
    main()
