#!/usr/bin/env python3
"""Build MLP-anchor manifests matched to flow-ablation training budget."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List


@dataclass(frozen=True)
class TaskSpec:
    suite: str
    task_key: str
    env: str
    complexity: str
    episode_length: int
    num_envs: int


@dataclass(frozen=True)
class MethodSpec:
    method_key: str
    method_description: str
    alg: str


TASK_SPECS: List[TaskSpec] = [
    TaskSpec("gym", "hopper", "gym_hopper_mujoco", "low", 1000, 64),
    TaskSpec("gym", "ant", "gym_ant_mujoco", "medium", 1000, 64),
    TaskSpec("mjlab_proxy", "anymal", "mjlab_velocity_flat_unitree_go2", "medium", 1000, 128),
    TaskSpec("gym", "humanoid", "gym_humanoid_mujoco", "medium_high", 1000, 64),
    TaskSpec("mjlab_proxy", "snu_humanoid", "mjlab_velocity_flat_unitree_g1", "high", 1000, 128),
    TaskSpec(
        "mjlab",
        "leap_left_grasp_asymmetric",
        "mjlab_leap_left_grasp_asymmetric",
        "medium",
        500,
        128,
    ),
]

METHOD_SPECS: List[MethodSpec] = [
    MethodSpec(
        method_key="mlpwm_mlppolicy",
        method_description="MLP world model + MLP policy (anchor baseline)",
        alg="pwm_5M_baseline_pwmorig",
    ),
    MethodSpec(
        method_key="mlpwm_flowpolicy",
        method_description="MLP world model + Flow policy (anchor baseline)",
        alg="pwm_5M_flowpolicy",
    ),
]

TASK_EXTRA_OVERRIDES: Dict[str, List[str]] = {
    "hopper": ["alg.horizon=7"],
    "ant": ["alg.horizon=14"],
    "anymal": ["alg.horizon=7"],
    "humanoid": ["alg.horizon=13"],
    "snu_humanoid": ["alg.horizon=13"],
    "leap_left_grasp_asymmetric": ["alg.horizon=13"],
}


def parse_filter_set(raw: str | None) -> set[str] | None:
    if not raw:
        return None
    return {x.strip() for x in raw.split(",") if x.strip()}


def main() -> None:
    parser = argparse.ArgumentParser(description="Build MLP anchor manifest.")
    parser.add_argument("--output", required=True, help="Output CSV path")
    parser.add_argument("--seeds", default="0,1,2", help="Comma-separated seeds")
    parser.add_argument("--tasks", default="", help="Optional comma-separated task_key filter.")
    parser.add_argument("--max-epochs", type=int, default=8000)
    parser.add_argument("--eval-runs", type=int, default=20)
    parser.add_argument("--rollout-episodes", type=int, default=3)
    parser.add_argument("--wandb-project", default="flow-mbpo-formal-training")
    parser.add_argument("--stage", default="flow_anchor")
    args = parser.parse_args()

    seed_list = [int(x.strip()) for x in args.seeds.split(",") if x.strip()]
    task_filter = parse_filter_set(args.tasks)
    tasks = [t for t in TASK_SPECS if task_filter is None or t.task_key in task_filter]

    rows: List[Dict[str, str]] = []
    for task in tasks:
        for method in METHOD_SPECS:
            for seed in seed_list:
                run_key = (
                    f"{args.stage}_{task.suite}_{task.task_key}_{method.method_key}"
                    f"_s{seed}_anchor_default"
                )
                overrides: List[str] = [
                    "alg.save_interval=500",
                    "++alg.wandb_log_every_epoch=true",
                ]
                overrides.extend(TASK_EXTRA_OVERRIDES.get(task.task_key, []))
                rows.append(
                    {
                        "run_key": run_key,
                        "stage": args.stage,
                        "suite": task.suite,
                        "task_key": task.task_key,
                        "env": task.env,
                        "complexity": task.complexity,
                        "episode_length": str(task.episode_length),
                        "method_key": method.method_key,
                        "method_description": method.method_description,
                        "alg": method.alg,
                        "seed": str(seed),
                        "hparam_profile": "anchor_default",
                        "max_epochs": str(args.max_epochs),
                        "num_envs": str(task.num_envs),
                        "eval_runs": str(args.eval_runs),
                        "rollout_episodes": str(args.rollout_episodes),
                        "rollout_max_steps": str(task.episode_length),
                        "wandb_project": args.wandb_project,
                        "wandb_group": f"single_task_online_{args.stage}_{task.suite}",
                        "notes": (
                            "MLP anchor for flow-vs-mlp comparability. "
                            f"task={task.task_key}, method={method.method_key}, profile=anchor_default"
                        ),
                        "overrides": ";".join(overrides),
                    }
                )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
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
    with output.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} rows to {output}")


if __name__ == "__main__":
    main()
