#!/usr/bin/env python3
"""Build corrected quality-probe collection manifests for MJLab-QS.

This is not a training manifest. It collects a small random/expert-candidate
pool with the fixed raw episode schema, then the empirical quality audit decides
whether a formal QS dataset can be built.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


TASKS = {
    "velocity_flat_unitree_go1": {
        "env_config": "scripts/cfg/env/mjlab_velocity_flat_unitree_go1.yaml",
        # Best available existing Go1 checkpoint by eval return, but still
        # expected to be challenged by the empirical expert gate.
        "checkpoint": "scripts/outputs/single_task_online/confirm_mjlab_curriculum4_strict_default_h100shadow/mjlab/velocity_flat_unitree_go1/mlpwm_mlppolicy/seed_2/default/logs/best_policy.pt",
        "alg_config": "scripts/cfg/alg/pwm_5M_baseline_pwmorig.yaml",
    },
    "velocity_flat_unitree_g1": {
        "env_config": "scripts/cfg/env/mjlab_velocity_flat_unitree_g1.yaml",
        # Best available existing G1 checkpoint by long-horizon eval length.
        "checkpoint": "scripts/outputs/single_task_online/confirm_mjlab_curriculum4_strict_default/mjlab/velocity_flat_unitree_g1/flowwm_mlppolicy/seed_0/default/logs/best_policy.pt",
        "alg_config": "scripts/cfg/alg/pwm_5M_flow_v2_substeps4.yaml",
    },
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--mode", required=True, help="Output mode suffix, e.g. quality_probe_h100.")
    p.add_argument("--output", required=True)
    p.add_argument("--root", default="scripts/outputs/mjlab_qs")
    p.add_argument("--episodes-per-source", type=int, default=100)
    p.add_argument("--num-envs", type=int, default=16)
    p.add_argument("--episode-length", type=int, default=1000)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(args.root)
    rows = []
    for task_key, spec in TASKS.items():
        for qbin, collector_mode, checkpoint, alg_config, noise in [
            ("random_smooth", "random_smooth", "", "", 0.0),
            ("expert", "checkpoint", spec["checkpoint"], spec["alg_config"], 0.0),
        ]:
            out = root / "raw" / args.mode / f"{task_key}_{qbin}_seed0.pt"
            rows.append(
                {
                    "stage": args.mode,
                    "task_key": task_key,
                    "quality_bin": qbin,
                    "env_config": spec["env_config"],
                    "output": str(out),
                    "metadata_output": str(out.with_suffix(".json")),
                    "collector_mode": collector_mode,
                    "collector_id": f"{collector_mode}_{qbin}",
                    "collector_alg_config": alg_config,
                    "collector_checkpoint": checkpoint,
                    "teacher_blend": 1.0,
                    "action_noise_std": noise,
                    "seed": 0,
                    "num_envs": args.num_envs,
                    "episodes": args.episodes_per_source,
                    "episode_length": args.episode_length,
                    "command_dim": 3,
                    "command_position": "tail",
                }
            )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote quality-probe collection manifest {output} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
