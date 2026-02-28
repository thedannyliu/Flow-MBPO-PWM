#!/usr/bin/env python3
"""Build CSV manifests for single-task online RL sweeps on PACE-ICE."""

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
    num_envs_by_stage: Dict[str, int]


@dataclass(frozen=True)
class MethodSpec:
    method_key: str
    alg: str
    description: str
    extra_overrides: List[str]


TASK_SPECS: List[TaskSpec] = [
    TaskSpec("gym", "hopper", "gym_hopper_mujoco", "low", 1000, {"smoke": 16, "pilot": 32, "confirm": 64}),
    TaskSpec("gym", "ant", "gym_ant_mujoco", "medium", 1000, {"smoke": 16, "pilot": 32, "confirm": 64}),
    # Proxy for quadruped locomotion while keeping GPU-native simulation.
    TaskSpec("mjlab_proxy", "anymal", "mjlab_velocity_flat_unitree_go2", "medium", 1000, {"smoke": 32, "pilot": 64, "confirm": 128}),
    TaskSpec("gym", "humanoid", "gym_humanoid_mujoco", "medium_high", 1000, {"smoke": 16, "pilot": 32, "confirm": 64}),
    # Proxy for SNU humanoid-class morphology in mjlab stack.
    TaskSpec("mjlab_proxy", "snu_humanoid", "mjlab_velocity_flat_unitree_g1", "high", 1000, {"smoke": 32, "pilot": 64, "confirm": 128}),
    TaskSpec(
        "mjlab",
        "velocity_flat_unitree_go2",
        "mjlab_velocity_flat_unitree_go2",
        "medium",
        1000,
        {"smoke": 32, "pilot": 64, "confirm": 128},
    ),
    TaskSpec(
        "mjlab",
        "velocity_flat_unitree_g1",
        "mjlab_velocity_flat_unitree_g1",
        "medium",
        1000,
        {"smoke": 32, "pilot": 64, "confirm": 128},
    ),
    TaskSpec(
        "mjlab",
        "tracking_flat_unitree_g1",
        "mjlab_tracking_flat_unitree_g1",
        "medium_high",
        1000,
        {"smoke": 32, "pilot": 48, "confirm": 96},
    ),
    TaskSpec(
        "mjlab",
        "leap_left_handcube_rotate",
        "mjlab_leap_left_handcube_rotate",
        "high",
        500,
        {"smoke": 16, "pilot": 32, "confirm": 64},
    ),
]


METHOD_SPECS: List[MethodSpec] = [
    MethodSpec(
        method_key="mlpwm_mlppolicy",
        alg="pwm_5M_baseline_final",
        description="PWM baseline (MLP world model + MLP policy)",
        extra_overrides=[],
    ),
    MethodSpec(
        method_key="flowwm_mlppolicy",
        alg="pwm_5M_flow_v2_substeps4",
        description="Flow world model + MLP policy",
        extra_overrides=[],
    ),
    MethodSpec(
        method_key="mlpwm_flowpolicy",
        alg="pwm_5M_flowpolicy",
        description="MLP world model + Flow policy",
        extra_overrides=[],
    ),
    MethodSpec(
        method_key="flowwm_flowpolicy",
        alg="pwm_5M_fullflow",
        description="Flow world model + Flow policy",
        extra_overrides=[],
    ),
]


HPARAM_PROFILES: Dict[str, List[str]] = {
    "default": [],
    "lr_x05": ["alg.actor_lr=2.5e-4", "alg.critic_lr=2.5e-4", "alg.model_lr=1.5e-4"],
    "lr_x20": ["alg.actor_lr=1e-3", "alg.critic_lr=1e-3", "alg.model_lr=6e-4"],
}

TASK_EXTRA_OVERRIDES: Dict[str, List[str]] = {
    # Motion-imitation tracking can produce very short episodes in early training.
    # Keep rollout slices valid for replay sampling during smoke/pilot bring-up.
    "tracking_flat_unitree_g1": ["alg.horizon=1"],
}


STAGE_SPECS = {
    "smoke": {
        "seeds": [0],
        "hparam_profiles": ["default"],
        "max_epochs": 200,
        "eval_runs": 8,
        "rollout_episodes": 2,
        "stage_overrides": ["alg.save_interval=100", "++alg.wandb_log_every_epoch=true"],
    },
    "pilot": {
        "seeds": [0, 1, 2],
        "hparam_profiles": ["default", "lr_x05", "lr_x20"],
        "max_epochs": 3000,
        "eval_runs": 20,
        "rollout_episodes": 3,
        "stage_overrides": ["alg.save_interval=250", "++alg.wandb_log_every_epoch=true"],
    },
    "confirm": {
        "seeds": list(range(10)),
        "hparam_profiles": ["default"],
        "max_epochs": 15000,
        "eval_runs": 40,
        "rollout_episodes": 5,
        "stage_overrides": ["alg.save_interval=500", "++alg.wandb_log_every_epoch=true"],
    },
}


def parse_filter_set(raw: str | None) -> set[str] | None:
    if not raw:
        return None
    return {token.strip() for token in raw.split(",") if token.strip()}


def main() -> None:
    parser = argparse.ArgumentParser(description="Build sweep manifest CSV.")
    parser.add_argument("--stage", choices=sorted(STAGE_SPECS.keys()), required=True)
    parser.add_argument("--output", required=True, help="Output CSV path")
    parser.add_argument(
        "--tasks",
        default="",
        help="Comma-separated task_key filters. Empty means all tasks.",
    )
    parser.add_argument(
        "--methods",
        default="",
        help="Comma-separated method_key filters. Empty means all methods.",
    )
    parser.add_argument(
        "--wandb-project",
        default="flow-mbpo-single-task-online-v1",
        help="WandB project name for generated rows.",
    )
    args = parser.parse_args()

    stage_spec = STAGE_SPECS[args.stage]
    task_filter = parse_filter_set(args.tasks)
    method_filter = parse_filter_set(args.methods)

    selected_tasks = [task for task in TASK_SPECS if task_filter is None or task.task_key in task_filter]
    selected_methods = [
        method for method in METHOD_SPECS if method_filter is None or method.method_key in method_filter
    ]

    rows: List[Dict[str, str]] = []
    for task in selected_tasks:
        for method in selected_methods:
            for seed in stage_spec["seeds"]:
                for hparam_profile in stage_spec["hparam_profiles"]:
                    overrides = []
                    overrides.extend(stage_spec["stage_overrides"])
                    overrides.extend(method.extra_overrides)
                    overrides.extend(TASK_EXTRA_OVERRIDES.get(task.task_key, []))
                    overrides.extend(HPARAM_PROFILES[hparam_profile])

                    run_key = (
                        f"{args.stage}_{task.suite}_{task.task_key}_{method.method_key}"
                        f"_s{seed}_{hparam_profile}"
                    )
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
                            "method_description": method.description,
                            "alg": method.alg,
                            "seed": str(seed),
                            "hparam_profile": hparam_profile,
                            "max_epochs": str(stage_spec["max_epochs"]),
                            "num_envs": str(task.num_envs_by_stage[args.stage]),
                            "eval_runs": str(stage_spec["eval_runs"]),
                            "rollout_episodes": str(stage_spec["rollout_episodes"]),
                            "rollout_max_steps": str(task.episode_length),
                            "wandb_project": args.wandb_project,
                            "wandb_group": f"single_task_online_{args.stage}_{task.suite}",
                            "notes": (
                                "Single-task online RL from scratch on PACE-ICE. "
                                f"task={task.task_key}, method={method.method_key}, profile={hparam_profile}"
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

    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} rows to {output_path}")


if __name__ == "__main__":
    main()
