#!/usr/bin/env python3
"""Build formal MJLab-QS collection manifest from native probe ranking."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List


TASK_IDS = {
    "Mjlab-Velocity-Flat-Unitree-Go1": "velocity_flat_unitree_go1",
    "Mjlab-Velocity-Flat-Unitree-G1": "velocity_flat_unitree_g1",
}

QS_BINS = [
    ("random_smooth", "random_smooth", 1.0, 0.0, 63),
    ("weak", "checkpoint_blend_random", 0.25, 0.20, 125),
    ("medium", "checkpoint_blend_random", 0.60, 0.10, 219),
    ("expert", "checkpoint", 1.0, 0.0, 157),
    ("expert_noisy", "checkpoint_noisy", 1.0, 0.05, 63),
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--ranking", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--stage", default="a25_native_qs")
    p.add_argument(
        "--tasks",
        default="",
        help=(
            "Optional comma-separated task keys or task IDs to include. "
            "Default: all canonical tasks."
        ),
    )
    p.add_argument("--num-envs", type=int, default=32)
    p.add_argument("--episodes-scale", type=float, default=1.0)
    return p.parse_args()


def as_bool(raw: object) -> bool:
    return str(raw).strip().lower() in {"1", "true", "yes", "y"}


def main() -> None:
    args = parse_args()
    selected_task_ids = set(TASK_IDS)
    if args.tasks.strip():
        selected_task_ids = set()
        requested = {x.strip() for x in args.tasks.split(",") if x.strip()}
        for task_id, task_key in TASK_IDS.items():
            if task_id in requested or task_key in requested:
                selected_task_ids.add(task_id)
        unknown = requested - selected_task_ids - {TASK_IDS[x] for x in selected_task_ids}
        if unknown:
            raise RuntimeError(f"Unknown requested tasks: {sorted(unknown)}")
    if not selected_task_ids:
        raise RuntimeError("No tasks selected.")

    with open(args.ranking, newline="", encoding="utf-8") as f:
        ranking = list(csv.DictReader(f))
    selected: Dict[str, Dict[str, str]] = {}
    for row in ranking:
        task_id = row["task_id"]
        if task_id not in selected_task_ids:
            continue
        if not as_bool(row.get("expert_gate_pass")):
            continue
        selected.setdefault(task_id, row)
    missing = sorted(selected_task_ids - set(selected))
    if missing:
        raise RuntimeError(f"No expert-gate-passing native collector for tasks: {missing}")

    rows: List[Dict[str, str]] = []
    for task_id, task_key in TASK_IDS.items():
        if task_id not in selected_task_ids:
            continue
        best = selected[task_id]
        for qbin, mode, blend, noise, episodes in QS_BINS:
            out = Path("scripts/outputs/mjlab_qs/raw") / args.stage / f"{task_key}_{qbin}_seed0.pt"
            method = "random_smooth" if mode == "random_smooth" else str(best["method"])
            checkpoint = "" if mode == "random_smooth" else str(best["checkpoint"])
            collector_id = (
                "native_random_smooth_reference"
                if mode == "random_smooth"
                else f"{best['collector_id']}_{qbin}"
            )
            rows.append(
                {
                    "stage": args.stage,
                    "task_key": task_key,
                    "task_id": task_id,
                    "method": method,
                    "checkpoint": checkpoint,
                    "quality_bin": qbin,
                    "collector_mode": mode,
                    "collector_id": collector_id,
                    "output": str(out),
                    "metadata_output": str(out.with_suffix(".json")),
                    "seed": "0",
                    "num_envs": str(args.num_envs),
                    "episodes": str(max(1, int(round(episodes * args.episodes_scale)))),
                    "episode_length": "1000",
                    "teacher_blend": str(blend),
                    "action_noise_std": str(noise),
                    "command_dim": "3",
                    "command_position": "tail",
                }
            )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} formal native QS rows to {output}")
    for task_id, row in selected.items():
        print(f"selected {task_id}: {row['collector_id']} return={row['return_mean']} len={row['episode_length_mean']}")


if __name__ == "__main__":
    main()
