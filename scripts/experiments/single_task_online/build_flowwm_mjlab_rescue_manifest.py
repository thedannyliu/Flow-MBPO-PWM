#!/usr/bin/env python3
"""Build a focused MJLab Flow-WM rescue manifest.

This sweep is intentionally separate from fair 2x2 confirmation runs:
- fair confirm answers whether aligned defaults are competitive
- rescue sweep answers whether a small set of targeted Flow-WM changes helps
"""

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
    extra_overrides: List[str]


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
    TaskSpec(
        suite="mjlab",
        task_key="leap_left_grasp_asymmetric",
        env="mjlab_leap_left_grasp_asymmetric",
        complexity="medium",
        episode_length=500,
        num_envs=128,
        extra_overrides=["alg.horizon=13"],
    ),
    TaskSpec(
        suite="mjlab",
        task_key="leap_left_inhand_pen_twirl",
        env="mjlab_leap_left_inhand_pen_twirl",
        complexity="high",
        episode_length=500,
        num_envs=64,
        extra_overrides=["alg.horizon=13"],
    ),
    TaskSpec(
        suite="mjlab",
        task_key="tracking_rough_unitree_g1",
        env="mjlab_tracking_rough_unitree_g1",
        complexity="high",
        episode_length=1000,
        num_envs=96,
        extra_overrides=["alg.horizon=1"],
    ),
    TaskSpec(
        suite="mjlab",
        task_key="velocity_flat_unitree_g1",
        env="mjlab_velocity_flat_unitree_g1",
        complexity="medium_high",
        episode_length=1000,
        num_envs=128,
        extra_overrides=["alg.horizon=13"],
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


PROFILE_SPECS: List[ProfileSpec] = [
    ProfileSpec("flow_base", "Baseline flow WM config", []),
    ProfileSpec("flow_rewrms_on", "Enable reward RMS normalization", ["alg.rew_rms=true"]),
    ProfileSpec(
        "flow_dyn2_rew1",
        "Bias world-model training toward dynamics fitting",
        ["alg.wm_dyn_loss_weight=2.0", "alg.wm_rew_loss_weight=1.0"],
    ),
    ProfileSpec(
        "flow_bootstrap_300",
        "Shorten bootstrap warmup to improve early online usability",
        ["alg.wm_bootstrap_iterations=300"],
    ),
    ProfileSpec(
        "flow_hsched_slow",
        "Use a short-to-long horizon schedule to stabilize early model rollouts",
        ["alg.horizon_start=1", "alg.horizon_switch_epoch=5000"],
    ),
]


def parse_filter_set(raw: str | None) -> set[str] | None:
    if not raw:
        return None
    return {token.strip() for token in raw.split(",") if token.strip()}


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a focused MJLab Flow-WM rescue manifest.")
    parser.add_argument("--output", required=True)
    parser.add_argument("--seeds", default="0,1,2", help="Comma-separated seeds (default: 0,1,2)")
    parser.add_argument("--tasks", default="", help="Optional comma-separated task filter")
    parser.add_argument("--methods", default="", help="Optional comma-separated method filter")
    parser.add_argument("--profiles", default="", help="Optional comma-separated profile filter")
    parser.add_argument("--max-epochs", type=int, default=8000)
    parser.add_argument("--eval-runs", type=int, default=20)
    parser.add_argument("--rollout-episodes", type=int, default=3)
    parser.add_argument("--wandb-project", default="flow-mbpo-formal-training")
    parser.add_argument("--stage", default="flowwm_mjlab_rescue_focus")
    args = parser.parse_args()

    seeds = [int(token.strip()) for token in args.seeds.split(",") if token.strip()]
    task_filter = parse_filter_set(args.tasks)
    method_filter = parse_filter_set(args.methods)
    profile_filter = parse_filter_set(args.profiles)

    tasks = [task for task in TASK_SPECS if task_filter is None or task.task_key in task_filter]
    methods = [method for method in METHOD_SPECS if method_filter is None or method.method_key in method_filter]
    profiles = [profile for profile in PROFILE_SPECS if profile_filter is None or profile.name in profile_filter]

    rows: List[Dict[str, str]] = []
    for task in tasks:
        for method in methods:
            for profile in profiles:
                for seed in seeds:
                    run_key = (
                        f"{args.stage}_{task.suite}_{task.task_key}_{method.method_key}"
                        f"_s{seed}_{profile.name}"
                    )
                    overrides = [
                        "alg.save_interval=500",
                        "++alg.wandb_log_every_epoch=true",
                    ]
                    overrides.extend(task.extra_overrides)
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
                                "Focused MJLab Flow-WM rescue sweep. "
                                f"task={task.task_key}, method={method.method_key}, "
                                f"profile={profile.name}, rationale={profile.note}"
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
