#!/usr/bin/env python3
"""Build QS collection rows from empirical training-stage checkpoint roles."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List


TASK_IDS = {
    "Mjlab-Velocity-Flat-Unitree-Go1": "velocity_flat_unitree_go1",
    "Mjlab-Velocity-Flat-Unitree-G1": "velocity_flat_unitree_g1",
}

QS_EPISODES = {
    "random_smooth": 63,
    "weak": 125,
    "medium": 219,
    "expert": 157,
    "expert_noisy": 63,
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--ranking", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--stage", required=True)
    p.add_argument("--tasks", default="velocity_flat_unitree_g1")
    p.add_argument(
        "--roles",
        default="random_smooth,weak,medium,expert,expert_noisy",
        help=(
            "Comma-separated QS buckets to collect. Supported: random_smooth, "
            "weak, medium, expert, expert_noisy."
        ),
    )
    p.add_argument("--num-envs", type=int, default=32)
    p.add_argument("--episodes-scale", type=float, default=1.0)
    p.add_argument(
        "--source-overrides",
        default="",
        help="Comma-separated role=collector_id overrides for weak,medium,expert source selection.",
    )
    p.add_argument(
        "--episode-overrides",
        default="",
        help="Comma-separated qbin=episodes overrides, e.g. expert=1024,expert_noisy=256.",
    )
    return p.parse_args()


def selected_task_ids(raw: str) -> List[str]:
    requested = [x.strip() for x in raw.split(",") if x.strip()]
    out: List[str] = []
    for item in requested:
        if item in TASK_IDS:
            out.append(item)
            continue
        matches = [task_id for task_id, task_key in TASK_IDS.items() if task_key == item]
        if matches:
            out.extend(matches)
            continue
        raise RuntimeError(f"Unknown task: {item}")
    if not out:
        raise RuntimeError("No tasks selected.")
    return out


def choose(rows: List[Dict[str, str]], role: str) -> Dict[str, str]:
    candidates = [r for r in rows if r.get("empirical_role") == role and r.get("checkpoint")]
    if not candidates:
        raise RuntimeError(f"No checkpoint-stage candidate for empirical_role={role}")
    candidates.sort(
        key=lambda r: (
            float(r.get("quality_score", "nan")),
            float(r.get("episode_length_mean", "nan")),
            float(r.get("return_mean", "nan")),
        ),
        reverse=True,
    )
    return candidates[0]


def parse_keyvals(raw: str) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for item in [x.strip() for x in raw.split(",") if x.strip()]:
        if "=" not in item:
            raise RuntimeError(f"Expected key=value override, got: {item}")
        key, value = item.split("=", 1)
        out[key.strip()] = value.strip()
    return out


def choose_override(rows: List[Dict[str, str]], role: str, collector_id: str) -> Dict[str, str]:
    candidates = [r for r in rows if r.get("collector_id") == collector_id and r.get("checkpoint")]
    if not candidates:
        raise RuntimeError(f"No checkpoint-stage candidate for {role} override collector_id={collector_id}")
    return candidates[0]


def parse_roles(raw: str) -> List[str]:
    roles = [x.strip() for x in raw.split(",") if x.strip()]
    if not roles:
        raise RuntimeError("--roles must be non-empty")
    unknown = sorted(set(roles) - set(QS_EPISODES))
    if unknown:
        raise RuntimeError(f"Unknown QS roles: {unknown}")
    if "expert_noisy" in roles and "expert" not in roles:
        raise RuntimeError("expert_noisy requires an expert source role")
    return roles


def make_row(
    stage: str,
    task_id: str,
    qbin: str,
    source: Dict[str, str] | None,
    num_envs: int,
    episodes_scale: float,
    episode_overrides: Dict[str, int],
) -> Dict[str, str]:
    task_key = TASK_IDS[task_id]
    out = Path("scripts/outputs/mjlab_qs/raw") / stage / f"{task_key}_{qbin}_seed0.pt"
    episodes_n = episode_overrides.get(qbin, max(1, int(round(QS_EPISODES[qbin] * episodes_scale))))
    episodes = str(episodes_n)
    if qbin == "random_smooth":
        return {
            "stage": stage,
            "task_key": task_key,
            "task_id": task_id,
            "method": "random_smooth",
            "checkpoint": "",
            "quality_bin": qbin,
            "collector_mode": "random_smooth",
            "collector_id": "native_random_smooth_reference",
            "output": str(out),
            "metadata_output": str(out.with_suffix(".json")),
            "seed": "0",
            "num_envs": str(num_envs),
            "episodes": episodes,
            "episode_length": "1000",
            "teacher_blend": "1.0",
            "action_noise_std": "0.0",
            "command_dim": "3",
            "command_position": "tail",
        }
    assert source is not None
    mode = "checkpoint_noisy" if qbin == "expert_noisy" else "checkpoint"
    noise = "0.05" if qbin == "expert_noisy" else "0.0"
    return {
        "stage": stage,
        "task_key": task_key,
        "task_id": task_id,
        "method": source["method"],
        "checkpoint": source["checkpoint"],
        "quality_bin": qbin,
        "collector_mode": mode,
        "collector_id": f"{source['collector_id']}_{qbin}",
        "output": str(out),
        "metadata_output": str(out.with_suffix(".json")),
        "seed": str(source.get("seed", "0")),
        "num_envs": str(num_envs),
        "episodes": episodes,
        "episode_length": "1000",
        "teacher_blend": "1.0",
        "action_noise_std": noise,
        "command_dim": "3",
        "command_position": "tail",
    }


def main() -> None:
    args = parse_args()
    task_ids = selected_task_ids(args.tasks)
    roles = parse_roles(args.roles)
    source_overrides = parse_keyvals(args.source_overrides)
    unknown_sources = sorted(set(source_overrides) - {"weak", "medium", "expert"})
    if unknown_sources:
        raise RuntimeError(f"Unknown source override roles: {unknown_sources}")
    episode_overrides_raw = parse_keyvals(args.episode_overrides)
    unknown_episode_bins = sorted(set(episode_overrides_raw) - set(QS_EPISODES))
    if unknown_episode_bins:
        raise RuntimeError(f"Unknown episode override bins: {unknown_episode_bins}")
    episode_overrides = {k: int(v) for k, v in episode_overrides_raw.items()}
    with open(args.ranking, newline="", encoding="utf-8") as f:
        ranking = list(csv.DictReader(f))
    rows: List[Dict[str, str]] = []
    for task_id in task_ids:
        task_rows = [r for r in ranking if r.get("task_id") == task_id]
        if not task_rows:
            raise RuntimeError(f"No ranking rows for task {task_id}")
        sources: Dict[str, Dict[str, str]] = {}
        for role in ("weak", "medium", "expert"):
            if role in roles or (role == "expert" and "expert_noisy" in roles):
                if role in source_overrides:
                    sources[role] = choose_override(task_rows, role, source_overrides[role])
                else:
                    sources[role] = choose(task_rows, role)
        for role in roles:
            if role == "random_smooth":
                rows.append(make_row(args.stage, task_id, role, None, args.num_envs, args.episodes_scale, episode_overrides))
            elif role == "expert_noisy":
                rows.append(make_row(args.stage, task_id, role, sources["expert"], args.num_envs, args.episodes_scale, episode_overrides))
            else:
                rows.append(make_row(args.stage, task_id, role, sources[role], args.num_envs, args.episodes_scale, episode_overrides))

        selected_bits = []
        for role, source in sorted(sources.items()):
            selected_bits.append(
                f"{role}={source['collector_id']}({source['return_mean']}, fall={source['fall_rate_mean']})"
            )
        print(f"selected {TASK_IDS[task_id]} roles={','.join(roles)}: " + "; ".join(selected_bits))

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} QS collection rows to {output}")


if __name__ == "__main__":
    main()
