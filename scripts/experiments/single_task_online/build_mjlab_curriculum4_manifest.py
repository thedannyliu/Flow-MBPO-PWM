#!/usr/bin/env python3
"""Build a registry-aligned 4-task MJLab curriculum manifest.

This manifest is the canonical fair-comparison line for the currently installed
MJLab package. It only uses task IDs that are actually registered in the active
`mjlab` site-packages installation.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List


@dataclass(frozen=True)
class TaskSpec:
    task_key: str
    env: str
    complexity: str
    episode_length: int
    num_envs: int
    extra_overrides: List[str]
    notes: str


@dataclass(frozen=True)
class MethodSpec:
    method_key: str
    method_description: str
    alg: str


TASK_SPECS: List[TaskSpec] = [
    TaskSpec(
        task_key="velocity_flat_unitree_go1",
        env="mjlab_velocity_flat_unitree_go1",
        complexity="simple",
        episode_length=1000,
        num_envs=128,
        extra_overrides=["alg.horizon=7"],
        notes="Simple canonical MJLab task: flat-terrain quadruped velocity control.",
    ),
    TaskSpec(
        task_key="velocity_flat_unitree_g1",
        env="mjlab_velocity_flat_unitree_g1",
        complexity="medium",
        episode_length=1000,
        num_envs=128,
        extra_overrides=["alg.horizon=13"],
        notes="Medium canonical MJLab task: flat-terrain humanoid velocity control.",
    ),
    TaskSpec(
        task_key="velocity_rough_unitree_g1",
        env="mjlab_velocity_rough_unitree_g1",
        complexity="hard",
        episode_length=1000,
        num_envs=128,
        extra_overrides=["alg.horizon=13"],
        notes="Hard canonical MJLab task: rough-terrain humanoid velocity control.",
    ),
    TaskSpec(
        task_key="tracking_flat_unitree_g1",
        env="mjlab_tracking_flat_unitree_g1",
        complexity="very_hard",
        episode_length=1000,
        num_envs=128,
        extra_overrides=["alg.horizon=13"],
        notes="Very hard canonical MJLab task: humanoid motion tracking on flat terrain.",
    ),
]


METHOD_SPECS: List[MethodSpec] = [
    MethodSpec(
        method_key="mlpwm_mlppolicy",
        method_description="PWM baseline (MLP world model + MLP policy)",
        alg="pwm_5M_baseline_pwmorig",
    ),
    MethodSpec(
        method_key="flowwm_mlppolicy",
        method_description="Flow world model + MLP policy",
        alg="pwm_5M_flow_v2_substeps4",
    ),
    MethodSpec(
        method_key="mlpwm_flowpolicy",
        method_description="MLP world model + Flow policy",
        alg="pwm_5M_flowpolicy",
    ),
    MethodSpec(
        method_key="flowwm_flowpolicy",
        method_description="Flow world model + Flow policy",
        alg="pwm_5M_fullflow",
    ),
]


def parse_filter_set(raw: str | None) -> set[str] | None:
    if not raw:
        return None
    return {token.strip() for token in raw.split(",") if token.strip()}


def main() -> None:
    parser = argparse.ArgumentParser(description="Build canonical 4-task MJLab curriculum manifest.")
    parser.add_argument("--output", required=True)
    parser.add_argument("--stage", default="confirm_mjlab_curriculum4_strict_default")
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--tasks", default="")
    parser.add_argument("--methods", default="")
    parser.add_argument("--wandb-project", default="flow-mbpo-mjlab-strict-aligned")
    parser.add_argument("--max-epochs", type=int, default=15000)
    parser.add_argument("--eval-runs", type=int, default=40)
    parser.add_argument("--rollout-episodes", type=int, default=5)
    args = parser.parse_args()

    seeds = [int(token.strip()) for token in args.seeds.split(",") if token.strip()]
    task_filter = parse_filter_set(args.tasks)
    method_filter = parse_filter_set(args.methods)

    tasks = [task for task in TASK_SPECS if task_filter is None or task.task_key in task_filter]
    methods = [method for method in METHOD_SPECS if method_filter is None or method.method_key in method_filter]

    rows: List[Dict[str, str]] = []
    for task in tasks:
        for method in methods:
            for seed in seeds:
                run_key = f"{args.stage}_mjlab_{task.task_key}_{method.method_key}_s{seed}_default"
                overrides = [
                    "alg.save_interval=500",
                    "++alg.wandb_log_every_epoch=true",
                ]
                overrides.extend(task.extra_overrides)
                rows.append(
                    {
                        "run_key": run_key,
                        "stage": args.stage,
                        "suite": "mjlab",
                        "task_key": task.task_key,
                        "env": task.env,
                        "complexity": task.complexity,
                        "episode_length": str(task.episode_length),
                        "method_key": method.method_key,
                        "method_description": method.method_description,
                        "alg": method.alg,
                        "seed": str(seed),
                        "hparam_profile": "default",
                        "max_epochs": str(args.max_epochs),
                        "num_envs": str(task.num_envs),
                        "eval_runs": str(args.eval_runs),
                        "rollout_episodes": str(args.rollout_episodes),
                        "rollout_max_steps": str(task.episode_length),
                        "wandb_project": args.wandb_project,
                        "wandb_group": f"single_task_online_{args.stage}_mjlab",
                        "notes": (
                            "Canonical registry-aligned MJLab curriculum fair comparison. "
                            f"task={task.task_key}, method={method.method_key}, seeds=3. "
                            f"{task.notes}"
                        ),
                        "overrides": ";".join(overrides),
                    }
                )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "run_key",
        "stage",
        "suite",
        "task_key",
        "env",
        "complexity",
        "episode_length",
        "method_key",
        "method_description",
        "alg",
        "seed",
        "hparam_profile",
        "max_epochs",
        "num_envs",
        "eval_runs",
        "rollout_episodes",
        "rollout_max_steps",
        "wandb_project",
        "wandb_group",
        "notes",
        "overrides",
    ]

    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} rows to {output_path}")


if __name__ == "__main__":
    main()
