#!/usr/bin/env python3
"""Build neutral MJLab collector retraining manifests.

These runs are a prerequisite for canonical MJLab-QS data collection. They are
not Flow-vs-MLP method comparisons: every row uses an MLP policy and an MLP
world model. Multiple MLP baseline profiles are included only to find a
collector that satisfies empirical return/length/fall-rate quality gates.
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
    horizon: int
    num_envs: int
    episode_length: int = 1000


@dataclass(frozen=True)
class ProfileSpec:
    name: str
    alg: str
    max_epochs: int
    overrides: List[str]
    notes: str


TASKS = [
    TaskSpec(
        task_key="velocity_flat_unitree_go1",
        env="mjlab_velocity_flat_unitree_go1",
        complexity="simple",
        horizon=7,
        num_envs=128,
    ),
    TaskSpec(
        task_key="velocity_flat_unitree_g1",
        env="mjlab_velocity_flat_unitree_g1",
        complexity="medium",
        horizon=13,
        num_envs=128,
    ),
]


PROFILES = [
    ProfileSpec(
        name="pwmorig_long",
        alg="pwm_5M_baseline_pwmorig",
        max_epochs=50000,
        overrides=[
            "alg.save_interval=1000",
            "++alg.wandb_log_every_epoch=true",
            "env.config.mjlab_env_kwargs.domain_randomization=false",
        ],
        notes="Original PWM-aligned MLP baseline, extended collector training budget.",
    ),
    ProfileSpec(
        name="baseline_final_rewrms_long",
        alg="pwm_5M_baseline_final",
        max_epochs=50000,
        overrides=[
            "alg.save_interval=1000",
            "++alg.wandb_log_every_epoch=true",
            "env.config.mjlab_env_kwargs.domain_randomization=false",
        ],
        notes="MLP baseline with reward RMS enabled; still neutral MLP policy/WM.",
    ),
    ProfileSpec(
        name="large48m_long",
        alg="pwm_48M",
        max_epochs=50000,
        overrides=[
            "alg.save_interval=2500",
            "++alg.wandb_log_every_epoch=true",
            "env.config.mjlab_env_kwargs.domain_randomization=false",
        ],
        notes="Larger MLP WM capacity profile to maximize chance of stable expert collector.",
    ),
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--output", required=True)
    p.add_argument("--stage", default="mjlab_qs_collector_retrain_v1")
    p.add_argument("--seeds", default="0,1,2")
    p.add_argument("--tasks", default="")
    p.add_argument("--profiles", default="")
    p.add_argument("--wandb-project", default="flow-mbpo-mjlab-qs-collector-retrain")
    p.add_argument("--eval-runs", type=int, default=80)
    p.add_argument("--rollout-episodes", type=int, default=5)
    return p.parse_args()


def parse_filter(raw: str) -> set[str] | None:
    if not raw:
        return None
    vals = {x.strip() for x in raw.split(",") if x.strip()}
    return vals or None


def main() -> None:
    args = parse_args()
    seeds = [int(x.strip()) for x in args.seeds.split(",") if x.strip()]
    task_filter = parse_filter(args.tasks)
    profile_filter = parse_filter(args.profiles)

    tasks = [t for t in TASKS if task_filter is None or t.task_key in task_filter]
    profiles = [p for p in PROFILES if profile_filter is None or p.name in profile_filter]
    rows: List[Dict[str, str]] = []
    for task in tasks:
        for profile in profiles:
            for seed in seeds:
                run_key = f"{args.stage}_mjlab_{task.task_key}_mlpwm_mlppolicy_s{seed}_{profile.name}"
                overrides = list(profile.overrides)
                overrides.append(f"alg.horizon={task.horizon}")
                rows.append(
                    {
                        "run_key": run_key,
                        "stage": args.stage,
                        "suite": "mjlab",
                        "task_key": task.task_key,
                        "env": task.env,
                        "complexity": task.complexity,
                        "episode_length": str(task.episode_length),
                        "method_key": "mlpwm_mlppolicy",
                        "method_description": "Neutral MLP/PWM collector candidate",
                        "alg": profile.alg,
                        "seed": str(seed),
                        "hparam_profile": profile.name,
                        "max_epochs": str(profile.max_epochs),
                        "num_envs": str(task.num_envs),
                        "eval_runs": str(args.eval_runs),
                        "rollout_episodes": str(args.rollout_episodes),
                        "rollout_max_steps": str(task.episode_length),
                        "wandb_project": args.wandb_project,
                        "wandb_group": f"collector_retrain_{args.stage}",
                        "notes": (
                            "Neutral collector retraining for MJLab-QS. "
                            "Use only after empirical return/length/fall-rate audit passes. "
                            f"profile={profile.name}. {profile.notes}"
                        ),
                        "overrides": ";".join(overrides),
                    }
                )

    if not rows:
        raise RuntimeError("No rows generated.")
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
    with output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} collector retraining rows to {output}")


if __name__ == "__main__":
    main()
