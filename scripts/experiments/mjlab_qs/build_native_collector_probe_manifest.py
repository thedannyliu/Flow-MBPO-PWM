#!/usr/bin/env python3
"""Build quality-probe manifest for completed MJLab-native collectors."""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path
from typing import Dict, List


TASK_IDS = {
    "velocity_flat_unitree_go1": "Mjlab-Velocity-Flat-Unitree-Go1",
    "velocity_flat_unitree_g1": "Mjlab-Velocity-Flat-Unitree-G1",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--native-root", default="scripts/outputs/mjlab_qs/native_collectors/mjlab_native_collector_v1")
    p.add_argument("--output", required=True)
    p.add_argument("--stage", default="mjlab_native_quality_probe_v1")
    p.add_argument("--episodes", type=int, default=100)
    p.add_argument("--num-envs", type=int, default=16)
    p.add_argument("--include-random", action="store_true")
    return p.parse_args()


def last_checkpoint(run_dir: Path) -> Path | None:
    ckpts = []
    for path in run_dir.glob("model_*.pt"):
        match = re.match(r"model_(\d+)\.pt", path.name)
        if match:
            ckpts.append((int(match.group(1)), path))
    if not ckpts:
        return None
    return sorted(ckpts)[-1][1]


def main() -> None:
    args = parse_args()
    root = Path(args.native_root)
    rows: List[Dict[str, str]] = []
    for task_key, task_id in TASK_IDS.items():
        if args.include_random:
            out = Path("scripts/outputs/mjlab_qs/raw") / args.stage / f"{task_key}_random_smooth_seed0.pt"
            rows.append(
                {
                    "stage": args.stage,
                    "task_key": task_key,
                    "task_id": task_id,
                    "method": "random_smooth",
                    "checkpoint": "",
                    "quality_bin": "random_smooth",
                    "collector_mode": "random_smooth",
                    "collector_id": "native_random_smooth_reference",
                    "output": str(out),
                    "metadata_output": str(out.with_suffix(".json")),
                    "seed": "0",
                    "num_envs": str(args.num_envs),
                    "episodes": str(args.episodes),
                    "episode_length": "1000",
                    "teacher_blend": "1.0",
                    "action_noise_std": "0.0",
                    "command_dim": "3",
                    "command_position": "tail",
                }
            )
        for run_dir in sorted((root / task_key).glob("rslrl_ppo_*/*")):
            if not run_dir.is_dir() or not run_dir.name.startswith("seed_"):
                continue
            method = run_dir.parent.name
            seed = run_dir.name.replace("seed_", "")
            ckpt = last_checkpoint(run_dir)
            if ckpt is None:
                continue
            out = Path("scripts/outputs/mjlab_qs/raw") / args.stage / f"{task_key}_{method}_seed{seed}.pt"
            rows.append(
                {
                    "stage": args.stage,
                    "task_key": task_key,
                    "task_id": task_id,
                    "method": method,
                    "checkpoint": str(ckpt),
                    "quality_bin": "expert",
                    "collector_mode": "checkpoint",
                    "collector_id": f"native_{method}_seed{seed}",
                    "output": str(out),
                    "metadata_output": str(out.with_suffix(".json")),
                    "seed": seed,
                    "num_envs": str(args.num_envs),
                    "episodes": str(args.episodes),
                    "episode_length": "1000",
                    "teacher_blend": "1.0",
                    "action_noise_std": "0.0",
                    "command_dim": "3",
                    "command_position": "tail",
                }
            )
    if not rows:
        raise RuntimeError(f"No native collector rows found under {root}")
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} native probe rows to {output}")


if __name__ == "__main__":
    main()
