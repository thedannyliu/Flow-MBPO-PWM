#!/usr/bin/env python3
"""Build balanced flow-ablation manifests for single-task online RL."""

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


@dataclass(frozen=True)
class FlowProfile:
    name: str
    extra_overrides: List[str]


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
        method_key="flowwm_mlppolicy",
        method_description="Flow world model + MLP policy",
        alg="pwm_5M_flow_v2_substeps4",
    ),
    MethodSpec(
        method_key="flowwm_flowpolicy",
        method_description="Flow world model + Flow policy",
        alg="pwm_5M_fullflow",
    ),
]

FLOW_PROFILES: List[FlowProfile] = [
    # Flow dynamics controls live under alg.* in Hydra config.
    FlowProfile("flow_s2_heun", ["alg.flow_integrator=heun", "alg.flow_substeps=2"]),
    FlowProfile("flow_s4_heun", ["alg.flow_integrator=heun", "alg.flow_substeps=4"]),
    FlowProfile("flow_s8_euler", ["alg.flow_integrator=euler", "alg.flow_substeps=8"]),
    FlowProfile("flow_lowlr", ["alg.actor_lr=3e-4", "alg.critic_lr=3e-4", "alg.model_lr=2e-4"]),
    FlowProfile("flow_strongreg", ["alg.actor_grad_norm=0.5", "alg.critic_grad_norm=50.0", "alg.wm_grad_norm=10.0"]),
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
    parser = argparse.ArgumentParser(description="Build balanced flow ablation manifest.")
    parser.add_argument("--output", required=True, help="Output CSV path")
    parser.add_argument("--seeds", default="0,1,2", help="Comma-separated seeds")
    parser.add_argument(
        "--tasks",
        default="",
        help="Optional comma-separated task_key filter (default: built-in balanced set).",
    )
    parser.add_argument(
        "--methods",
        default="",
        help="Optional comma-separated method_key filter (default: flowwm methods).",
    )
    parser.add_argument("--max-epochs", type=int, default=8000)
    parser.add_argument("--eval-runs", type=int, default=20)
    parser.add_argument("--rollout-episodes", type=int, default=3)
    parser.add_argument("--wandb-project", default="flow-mbpo-formal-training")
    parser.add_argument("--stage", default="flow_ablation")
    args = parser.parse_args()

    seed_list = [int(x.strip()) for x in args.seeds.split(",") if x.strip()]
    task_filter = parse_filter_set(args.tasks)
    method_filter = parse_filter_set(args.methods)

    tasks = [t for t in TASK_SPECS if task_filter is None or t.task_key in task_filter]
    methods = [m for m in METHOD_SPECS if method_filter is None or m.method_key in method_filter]

    rows: List[Dict[str, str]] = []
    for task in tasks:
        for method in methods:
            for profile in FLOW_PROFILES:
                for seed in seed_list:
                    run_key = (
                        f"{args.stage}_{task.suite}_{task.task_key}_{method.method_key}"
                        f"_s{seed}_{profile.name}"
                    )
                    overrides: List[str] = [
                        "alg.save_interval=500",
                        "++alg.wandb_log_every_epoch=true",
                    ]
                    overrides.extend(TASK_EXTRA_OVERRIDES.get(task.task_key, []))
                    overrides.extend(profile.extra_overrides)
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
                            "hparam_profile": profile.name,
                            "max_epochs": str(args.max_epochs),
                            "num_envs": str(task.num_envs),
                            "eval_runs": str(args.eval_runs),
                            "rollout_episodes": str(args.rollout_episodes),
                            "rollout_max_steps": str(task.episode_length),
                            "wandb_project": args.wandb_project,
                            "wandb_group": f"single_task_online_{args.stage}_{task.suite}",
                            "notes": (
                                "Flow ablation on PACE-ICE. "
                                f"task={task.task_key}, method={method.method_key}, profile={profile.name}"
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
