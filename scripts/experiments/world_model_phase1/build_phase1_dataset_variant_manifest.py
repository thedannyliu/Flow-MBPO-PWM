#!/usr/bin/env python3
"""Build dataset-collection manifests for Phase 1 dataset variants."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--mode", choices=["smoke", "formal"], default="formal")
    parser.add_argument("--wandb-project", default="flow-mbpo-phase1-dataset-variants")
    parser.add_argument("--teacher-alg-config", default="scripts/cfg/alg/pwm_5M_baseline_pwmorig.yaml")
    parser.add_argument(
        "--teacher-checkpoint-go1",
        default=(
            "scripts/outputs/single_task_online/confirm_mjlab_curriculum4_strict_default/"
            "mjlab/velocity_flat_unitree_go1/mlpwm_mlppolicy/seed_0/default/logs/best_policy.pt"
        ),
    )
    parser.add_argument(
        "--teacher-checkpoint-g1",
        default=(
            "scripts/outputs/single_task_online/confirm_mjlab_curriculum4_strict_default/"
            "mjlab/velocity_flat_unitree_g1/mlpwm_mlppolicy/seed_0/default/logs/best_policy.pt"
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    root = "scripts/outputs/world_model_phase1/datasets"
    common = {
        "gpu_type": "H200",
        "time_limit": "02:00:00",
        "memory": "64G",
        "cpus": 8,
        "python_bin": "/storage/ice1/2/9/eliu354/conda_envs/flow-mbpo/bin/python",
        "conda_env": "",
        "wandb_project": args.wandb_project,
        "wandb_group": "phase1_dataset_variants",
        "wandb_tags": "phase1,dataset_collection,mjlab,world_model,dataset_variant",
    }

    if args.mode == "smoke":
        target_episodes = 8
        num_envs = 8
        max_windows = 64
        rows = [
            {
                **common,
                "task_key": "velocity_flat_unitree_go1",
                "env_config": "scripts/cfg/env/mjlab_velocity_flat_unitree_go1.yaml",
                "dataset_variant": "teacher",
                "action_mode": "teacher_policy",
                "teacher_alg_config": args.teacher_alg_config,
                "teacher_checkpoint": args.teacher_checkpoint_go1,
                "teacher_deterministic": 1,
                "mixed_teacher_prob": 0.5,
                "seed": 0,
                "output": f"{root}/phase1_smoke_velocity_flat_unitree_go1_teacher_seed0.pt",
                "metadata_output": f"{root}/phase1_smoke_velocity_flat_unitree_go1_teacher_seed0.json",
                "num_envs": num_envs,
                "target_episodes": target_episodes,
                "episode_length": 128,
                "window_length": 8,
                "window_stride": 1,
                "max_windows": max_windows,
                "disable_wandb": 1,
                "job_name": "phase1_dataset_smoke_go1_teacher",
            },
            {
                **common,
                "task_key": "velocity_flat_unitree_go1",
                "env_config": "scripts/cfg/env/mjlab_velocity_flat_unitree_go1.yaml",
                "dataset_variant": "mixed",
                "action_mode": "mixed_episode",
                "teacher_alg_config": args.teacher_alg_config,
                "teacher_checkpoint": args.teacher_checkpoint_go1,
                "teacher_deterministic": 1,
                "mixed_teacher_prob": 0.5,
                "seed": 0,
                "output": f"{root}/phase1_smoke_velocity_flat_unitree_go1_mixed_seed0.pt",
                "metadata_output": f"{root}/phase1_smoke_velocity_flat_unitree_go1_mixed_seed0.json",
                "num_envs": num_envs,
                "target_episodes": target_episodes,
                "episode_length": 128,
                "window_length": 8,
                "window_stride": 1,
                "max_windows": max_windows,
                "disable_wandb": 1,
                "job_name": "phase1_dataset_smoke_go1_mixed",
            },
            {
                **common,
                "task_key": "velocity_flat_unitree_g1",
                "env_config": "scripts/cfg/env/mjlab_velocity_flat_unitree_g1.yaml",
                "dataset_variant": "teacher",
                "action_mode": "teacher_policy",
                "teacher_alg_config": args.teacher_alg_config,
                "teacher_checkpoint": args.teacher_checkpoint_g1,
                "teacher_deterministic": 1,
                "mixed_teacher_prob": 0.5,
                "seed": 0,
                "output": f"{root}/phase1_smoke_velocity_flat_unitree_g1_teacher_seed0.pt",
                "metadata_output": f"{root}/phase1_smoke_velocity_flat_unitree_g1_teacher_seed0.json",
                "num_envs": num_envs,
                "target_episodes": target_episodes,
                "episode_length": 128,
                "window_length": 8,
                "window_stride": 1,
                "max_windows": max_windows,
                "disable_wandb": 1,
                "job_name": "phase1_dataset_smoke_g1_teacher",
            },
        ]
    else:
        target_episodes = 48
        num_envs = 16
        max_windows = 256
        rows = [
            {
                **common,
                "task_key": "velocity_flat_unitree_go1",
                "env_config": "scripts/cfg/env/mjlab_velocity_flat_unitree_go1.yaml",
                "dataset_variant": "teacher",
                "action_mode": "teacher_policy",
                "teacher_alg_config": args.teacher_alg_config,
                "teacher_checkpoint": args.teacher_checkpoint_go1,
                "teacher_deterministic": 1,
                "mixed_teacher_prob": 0.5,
                "seed": 0,
                "output": f"{root}/phase1_velocity_flat_unitree_go1_teacher_seed0.pt",
                "metadata_output": f"{root}/phase1_velocity_flat_unitree_go1_teacher_seed0.json",
                "num_envs": num_envs,
                "target_episodes": target_episodes,
                "episode_length": 128,
                "window_length": 8,
                "window_stride": 1,
                "max_windows": max_windows,
                "disable_wandb": 0,
                "job_name": "phase1_dataset_go1_teacher",
            },
            {
                **common,
                "task_key": "velocity_flat_unitree_go1",
                "env_config": "scripts/cfg/env/mjlab_velocity_flat_unitree_go1.yaml",
                "dataset_variant": "mixed",
                "action_mode": "mixed_episode",
                "teacher_alg_config": args.teacher_alg_config,
                "teacher_checkpoint": args.teacher_checkpoint_go1,
                "teacher_deterministic": 1,
                "mixed_teacher_prob": 0.5,
                "seed": 0,
                "output": f"{root}/phase1_velocity_flat_unitree_go1_mixed_seed0.pt",
                "metadata_output": f"{root}/phase1_velocity_flat_unitree_go1_mixed_seed0.json",
                "num_envs": num_envs,
                "target_episodes": target_episodes,
                "episode_length": 128,
                "window_length": 8,
                "window_stride": 1,
                "max_windows": max_windows,
                "disable_wandb": 0,
                "job_name": "phase1_dataset_go1_mixed",
            },
            {
                **common,
                "task_key": "velocity_flat_unitree_g1",
                "env_config": "scripts/cfg/env/mjlab_velocity_flat_unitree_g1.yaml",
                "dataset_variant": "random",
                "action_mode": "random_uniform",
                "teacher_alg_config": "",
                "teacher_checkpoint": "",
                "teacher_deterministic": 0,
                "mixed_teacher_prob": 0.5,
                "seed": 0,
                "output": f"{root}/phase1_velocity_flat_unitree_g1_random_seed0.pt",
                "metadata_output": f"{root}/phase1_velocity_flat_unitree_g1_random_seed0.json",
                "num_envs": num_envs,
                "target_episodes": target_episodes,
                "episode_length": 128,
                "window_length": 8,
                "window_stride": 1,
                "max_windows": max_windows,
                "disable_wandb": 0,
                "job_name": "phase1_dataset_g1_random",
            },
            {
                **common,
                "task_key": "velocity_flat_unitree_g1",
                "env_config": "scripts/cfg/env/mjlab_velocity_flat_unitree_g1.yaml",
                "dataset_variant": "teacher",
                "action_mode": "teacher_policy",
                "teacher_alg_config": args.teacher_alg_config,
                "teacher_checkpoint": args.teacher_checkpoint_g1,
                "teacher_deterministic": 1,
                "mixed_teacher_prob": 0.5,
                "seed": 0,
                "output": f"{root}/phase1_velocity_flat_unitree_g1_teacher_seed0.pt",
                "metadata_output": f"{root}/phase1_velocity_flat_unitree_g1_teacher_seed0.json",
                "num_envs": num_envs,
                "target_episodes": target_episodes,
                "episode_length": 128,
                "window_length": 8,
                "window_stride": 1,
                "max_windows": max_windows,
                "disable_wandb": 0,
                "job_name": "phase1_dataset_g1_teacher",
            },
            {
                **common,
                "task_key": "velocity_flat_unitree_g1",
                "env_config": "scripts/cfg/env/mjlab_velocity_flat_unitree_g1.yaml",
                "dataset_variant": "mixed",
                "action_mode": "mixed_episode",
                "teacher_alg_config": args.teacher_alg_config,
                "teacher_checkpoint": args.teacher_checkpoint_g1,
                "teacher_deterministic": 1,
                "mixed_teacher_prob": 0.5,
                "seed": 0,
                "output": f"{root}/phase1_velocity_flat_unitree_g1_mixed_seed0.pt",
                "metadata_output": f"{root}/phase1_velocity_flat_unitree_g1_mixed_seed0.json",
                "num_envs": num_envs,
                "target_episodes": target_episodes,
                "episode_length": 128,
                "window_length": 8,
                "window_stride": 1,
                "max_windows": max_windows,
                "disable_wandb": 0,
                "job_name": "phase1_dataset_g1_mixed",
            },
        ]

    with output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} rows to {output}")


if __name__ == "__main__":
    main()
