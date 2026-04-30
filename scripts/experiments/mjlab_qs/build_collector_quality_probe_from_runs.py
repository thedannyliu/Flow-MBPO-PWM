#!/usr/bin/env python3
"""Build quality-probe collection manifest from retrained collector runs."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List


TASK_TO_ENV = {
    "velocity_flat_unitree_go1": "scripts/cfg/env/mjlab_velocity_flat_unitree_go1.yaml",
    "velocity_flat_unitree_g1": "scripts/cfg/env/mjlab_velocity_flat_unitree_g1.yaml",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--collector-stage", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--probe-mode", required=True)
    p.add_argument("--root", default="scripts/outputs")
    p.add_argument("--episodes-per-source", type=int, default=100)
    p.add_argument("--num-envs", type=int, default=16)
    p.add_argument("--top-k-per-task", type=int, default=3)
    p.add_argument("--min-eval-length", type=float, default=100.0)
    p.add_argument("--include-random", action="store_true")
    return p.parse_args()


def alg_for_profile(profile: str) -> str:
    if profile == "large48m_long":
        return "scripts/cfg/alg/pwm_48M.yaml"
    if profile == "baseline_final_rewrms_long":
        return "scripts/cfg/alg/pwm_5M_baseline_final.yaml"
    return "scripts/cfg/alg/pwm_5M_baseline_pwmorig.yaml"


def collect_candidates(root: Path, stage: str) -> Dict[str, List[Dict[str, object]]]:
    base = root / "single_task_online" / stage / "mjlab"
    out: Dict[str, List[Dict[str, object]]] = {}
    for eval_summary in sorted(base.glob("*/mlpwm_mlppolicy/seed_*/*/eval/eval_summary.json")):
        parts = eval_summary.parts
        task_key = parts[-6]
        seed = parts[-4].replace("seed_", "")
        profile = parts[-3]
        try:
            payload = json.loads(eval_summary.read_text(encoding="utf-8"))
        except Exception:
            continue
        ckpt = eval_summary.parent.parent / "logs" / "best_policy.pt"
        if not ckpt.exists():
            ckpt = eval_summary.parent.parent / "logs" / "final_policy.pt"
        if not ckpt.exists():
            continue
        out.setdefault(task_key, []).append(
            {
                "task_key": task_key,
                "seed": seed,
                "profile": profile,
                "checkpoint": str(ckpt),
                "alg_config": alg_for_profile(profile),
                "return_mean": float(payload.get("return_mean", float("-inf"))),
                "episode_length_mean": float(payload.get("episode_length_mean", 0.0)),
            }
        )
    return out


def main() -> None:
    args = parse_args()
    root = Path(args.root)
    candidates = collect_candidates(root, args.collector_stage)
    rows: List[Dict[str, str]] = []
    for task_key, task_candidates in sorted(candidates.items()):
        ranked = sorted(
            [c for c in task_candidates if float(c["episode_length_mean"]) >= args.min_eval_length],
            key=lambda c: (float(c["episode_length_mean"]), float(c["return_mean"])),
            reverse=True,
        )[: args.top_k_per_task]
        if args.include_random and task_key in TASK_TO_ENV:
            out = Path("scripts/outputs/mjlab_qs/raw") / args.probe_mode / f"{task_key}_random_smooth_seed0.pt"
            rows.append(
                {
                    "stage": args.probe_mode,
                    "task_key": task_key,
                    "quality_bin": "random_smooth",
                    "env_config": TASK_TO_ENV[task_key],
                    "output": str(out),
                    "metadata_output": str(out.with_suffix(".json")),
                    "collector_mode": "random_smooth",
                    "collector_id": "random_smooth_reference",
                    "collector_alg_config": "",
                    "collector_checkpoint": "",
                    "teacher_blend": "1.0",
                    "action_noise_std": "0.0",
                    "seed": "0",
                    "num_envs": str(args.num_envs),
                    "episodes": str(args.episodes_per_source),
                    "episode_length": "1000",
                    "command_dim": "3",
                    "command_position": "tail",
                }
            )
        for idx, cand in enumerate(ranked):
            out = Path("scripts/outputs/mjlab_qs/raw") / args.probe_mode / (
                f"{task_key}_expert_candidate{idx}_{cand['profile']}_seed{cand['seed']}.pt"
            )
            rows.append(
                {
                    "stage": args.probe_mode,
                    "task_key": task_key,
                    "quality_bin": "expert",
                    "env_config": TASK_TO_ENV[task_key],
                    "output": str(out),
                    "metadata_output": str(out.with_suffix(".json")),
                    "collector_mode": "checkpoint",
                    "collector_id": f"checkpoint_{cand['profile']}_seed{cand['seed']}",
                    "collector_alg_config": str(cand["alg_config"]),
                    "collector_checkpoint": str(cand["checkpoint"]),
                    "teacher_blend": "1.0",
                    "action_noise_std": "0.0",
                    "seed": "0",
                    "num_envs": str(args.num_envs),
                    "episodes": str(args.episodes_per_source),
                    "episode_length": "1000",
                    "command_dim": "3",
                    "command_position": "tail",
                }
            )

    if not rows:
        raise RuntimeError(f"No collector candidates found for stage {args.collector_stage}")
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} quality-probe rows to {output}")


if __name__ == "__main__":
    main()
