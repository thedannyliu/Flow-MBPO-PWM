#!/usr/bin/env python3
"""Build flow-WM advanced hyperparameter manifests (all tasks, 1+ seeds)."""

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
class ProfileSpec:
    name: str
    note: str
    extra_overrides: List[str]


TASK_SPECS: List[TaskSpec] = [
    TaskSpec("gym", "hopper", "gym_hopper_mujoco", "low", 1000, 64),
    TaskSpec("gym", "ant", "gym_ant_mujoco", "medium", 1000, 64),
    TaskSpec("mjlab_proxy", "anymal", "mjlab_velocity_flat_unitree_go2", "medium", 1000, 128),
    TaskSpec("gym", "humanoid", "gym_humanoid_mujoco", "medium_high", 1000, 64),
    TaskSpec("mjlab_proxy", "snu_humanoid", "mjlab_velocity_flat_unitree_g1", "high", 1000, 128),
    TaskSpec("mjlab", "leap_left_grasp_asymmetric", "mjlab_leap_left_grasp_asymmetric", "medium", 500, 128),
    TaskSpec("mjlab", "tracking_rough_unitree_g1", "mjlab_tracking_rough_unitree_g1", "medium_high", 1000, 96),
    TaskSpec("mjlab", "leap_left_inhand_pen_twirl", "mjlab_leap_left_inhand_pen_twirl", "high", 500, 64),
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


PROFILE_SPECS: List[ProfileSpec] = [
    ProfileSpec("flow_base", "Baseline flow WM config", []),
    ProfileSpec("flow_rewrms_on", "Enable reward RMS normalization", ["alg.rew_rms=true"]),
    ProfileSpec("flow_model_lr_2e4", "Lower model LR to 2e-4", ["alg.model_lr=2e-4"]),
    ProfileSpec("flow_model_lr_1e4", "Lower model LR to 1e-4", ["alg.model_lr=1e-4"]),
    ProfileSpec("flow_wmgn_10", "Tighter world-model grad clip", ["alg.wm_grad_norm=10.0"]),
    ProfileSpec("flow_wmgn_5", "Very tight world-model grad clip", ["alg.wm_grad_norm=5.0"]),
    ProfileSpec("flow_heun_s2", "Use Heun with 2 substeps", ["alg.flow_integrator=heun", "alg.flow_substeps=2"]),
    ProfileSpec("flow_euler_s8", "Use Euler with 8 substeps", ["alg.flow_integrator=euler", "alg.flow_substeps=8"]),
    ProfileSpec("flow_tau_midpoint", "Midpoint tau sampling for lower variance", ["alg.flow_tau_sampling=midpoint"]),
    ProfileSpec("flow_dyn2_rew1", "Dynamics loss weight 2x reward loss", ["alg.wm_dyn_loss_weight=2.0", "alg.wm_rew_loss_weight=1.0"]),
    ProfileSpec("flow_dyn1_rew2", "Reward loss weight 2x dynamics loss", ["alg.wm_dyn_loss_weight=1.0", "alg.wm_rew_loss_weight=2.0"]),
    ProfileSpec("flow_bootstrap_300", "Short WM bootstrap warmup (300 iters)", ["alg.wm_bootstrap_iterations=300"]),
    ProfileSpec("flow_bootstrap_2000", "Long WM bootstrap warmup (2000 iters)", ["alg.wm_bootstrap_iterations=2000"]),
    ProfileSpec("flow_hsched_fast", "Horizon schedule 1 -> base @ epoch 1500", ["alg.horizon_start=1", "alg.horizon_switch_epoch=1500"]),
    ProfileSpec("flow_hsched_slow", "Horizon schedule 1 -> base @ epoch 5000", ["alg.horizon_start=1", "alg.horizon_switch_epoch=5000"]),
]


TASK_BASE_OVERRIDES: Dict[str, List[str]] = {
    "hopper": ["alg.horizon=7"],
    "ant": ["alg.horizon=14"],
    "anymal": ["alg.horizon=7"],
    "humanoid": ["alg.horizon=13"],
    "snu_humanoid": ["alg.horizon=13"],
    "leap_left_grasp_asymmetric": ["alg.horizon=13"],
    "leap_left_inhand_pen_twirl": ["alg.horizon=13"],
    "tracking_rough_unitree_g1": ["alg.horizon=1"],
}


def parse_filter_set(raw: str | None) -> set[str] | None:
    if not raw:
        return None
    return {x.strip() for x in raw.split(",") if x.strip()}


def main() -> None:
    parser = argparse.ArgumentParser(description="Build advanced flow-WM hyperparameter manifest.")
    parser.add_argument("--output", required=True)
    parser.add_argument("--seeds", default="0", help="Comma-separated seeds (default: 0)")
    parser.add_argument("--tasks", default="", help="Optional comma-separated task_key filter")
    parser.add_argument("--methods", default="", help="Optional comma-separated method_key filter")
    parser.add_argument("--profiles", default="", help="Optional comma-separated profile filter")
    parser.add_argument("--max-epochs", type=int, default=8000)
    parser.add_argument("--eval-runs", type=int, default=20)
    parser.add_argument("--rollout-episodes", type=int, default=3)
    parser.add_argument("--wandb-project", default="flow-mbpo-formal-training")
    parser.add_argument("--stage", default="flowwm_hparam_explore")
    args = parser.parse_args()

    seed_list = [int(x.strip()) for x in args.seeds.split(",") if x.strip()]
    task_filter = parse_filter_set(args.tasks)
    method_filter = parse_filter_set(args.methods)
    profile_filter = parse_filter_set(args.profiles)

    tasks = [t for t in TASK_SPECS if task_filter is None or t.task_key in task_filter]
    methods = [m for m in METHOD_SPECS if method_filter is None or m.method_key in method_filter]
    profiles = [p for p in PROFILE_SPECS if profile_filter is None or p.name in profile_filter]

    rows: List[Dict[str, str]] = []
    for task in tasks:
        for method in methods:
            for profile in profiles:
                for seed in seed_list:
                    run_key = (
                        f"{args.stage}_{task.suite}_{task.task_key}_{method.method_key}"
                        f"_s{seed}_{profile.name}"
                    )
                    overrides: List[str] = [
                        "alg.save_interval=500",
                        "++alg.wandb_log_every_epoch=true",
                    ]
                    overrides.extend(TASK_BASE_OVERRIDES.get(task.task_key, []))
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
                                "Flow WM advanced hyperparameter sweep. "
                                f"task={task.task_key}, method={method.method_key}, "
                                f"profile={profile.name}, rationale={profile.note}"
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
