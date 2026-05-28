#!/usr/bin/env python3
"""Export collector/BC/PWM rollout comparisons from saved rollout summaries."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--policy-root", default="scripts/outputs/mjlab_qs/policy_rollouts")
    p.add_argument("--collector-root", default="scripts/outputs/mjlab_qs/native_collector_rollouts")
    p.add_argument("--output-csv", required=True)
    p.add_argument("--output-md", required=True)
    return p.parse_args()


def fnum(value: Any) -> float | None:
    if value is None or value == "":
        return None
    return float(value)


def mean(values: list[float | None]) -> float | None:
    clean = [v for v in values if v is not None]
    return statistics.mean(clean) if clean else None


def std(values: list[float | None]) -> float | None:
    clean = [v for v in values if v is not None]
    return statistics.pstdev(clean) if len(clean) > 1 else 0.0 if len(clean) == 1 else None


def fmt(value: float | None, digits: int = 4) -> str:
    return "" if value is None else f"{value:.{digits}f}"


def fall_rate_from_rollout_csv(summary_path: Path) -> float | None:
    rollout_csv = summary_path.with_name("rollout_summary.csv")
    if not rollout_csv.exists():
        return None
    with rollout_csv.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows or "terminated" not in rows[0]:
        return None
    return sum(float(row["terminated"]) for row in rows) / len(rows)


def classify_policy_stage(stage: str, compute_profile: str) -> str:
    if "bc_only" in stage or "policy0k" in compute_profile:
        return "bc_only"
    if "bcwarm" in stage:
        return "bc_warm_pwm"
    if "pwm_flow_policy2x2" in stage:
        return "pwm_2x2"
    return "learned_policy"


def load_policy_records(root: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for path in sorted(root.glob("**/summary.json")):
        rel = path.relative_to(root)
        parts = rel.parts
        if len(parts) < 8:
            continue
        stage, task, wm, policy, online_profile, compute_profile = parts[:6]
        seed = next((part for part in parts if part.startswith("seed_")), "")
        checkpoint_kind = "best" if "best" in parts else "final"
        data = json.loads(path.read_text(encoding="utf-8"))
        records.append(
            {
                "family": classify_policy_stage(stage, compute_profile),
                "stage": stage,
                "task": task,
                "variant": f"{wm}+{policy}",
                "profile": compute_profile,
                "seed": seed.replace("seed_", ""),
                "checkpoint_kind": data.get("checkpoint_kind") or checkpoint_kind,
                "return_mean": fnum(data.get("return_mean")),
                "return_std": fnum(data.get("return_std")),
                "episode_length_mean": fnum(data.get("episode_length_mean")),
                "fall_rate_mean": fall_rate_from_rollout_csv(path),
                "num_episodes": int(data.get("num_episodes") or 0),
                "num_frames": int(data.get("num_frames") or 0),
                "video": data.get("video", ""),
                "summary_path": str(path),
            }
        )
    return records


def load_collector_records(root: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for path in sorted(root.glob("**/summary.json")):
        rel = path.relative_to(root)
        parts = rel.parts
        if len(parts) < 4:
            continue
        stage, task, variant = parts[:3]
        seed = next((part for part in parts if part.startswith("seed_")), "")
        data = json.loads(path.read_text(encoding="utf-8"))
        records.append(
            {
                "family": "collector_reference",
                "stage": stage,
                "task": task,
                "variant": variant,
                "profile": data.get("collector_mode", ""),
                "seed": seed.replace("seed_", ""),
                "checkpoint_kind": "collector",
                "return_mean": fnum(data.get("return_mean")),
                "return_std": fnum(data.get("return_std")),
                "episode_length_mean": fnum(data.get("episode_length_mean")),
                "fall_rate_mean": fnum(data.get("fall_rate_mean")),
                "num_episodes": int(data.get("num_episodes") or 0),
                "num_frames": int(data.get("num_frames") or 0),
                "video": data.get("video", ""),
                "summary_path": str(path),
            }
        )
    return records


def aggregate(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, str, str, str], list[dict[str, Any]]] = {}
    for record in records:
        key = (
            record["family"],
            record["stage"],
            record["variant"],
            record["profile"],
            record["checkpoint_kind"],
        )
        groups.setdefault(key, []).append(record)
    rows = []
    for (family, stage, variant, profile, checkpoint_kind), group in sorted(groups.items()):
        rows.append(
            {
                "family": family,
                "stage": stage,
                "variant": variant,
                "profile": profile,
                "checkpoint_kind": checkpoint_kind,
                "n": len(group),
                "return_mean": mean([row["return_mean"] for row in group]),
                "return_std_across_rows": std([row["return_mean"] for row in group]),
                "episode_length_mean": mean([row["episode_length_mean"] for row in group]),
                "fall_rate_mean": mean([row["fall_rate_mean"] for row in group]),
                "num_frames_total": sum(int(row["num_frames"] or 0) for row in group),
                "videos": ";".join(row["video"] for row in group if row["video"]),
            }
        )
    return rows


def choose_expert_return(rows: list[dict[str, Any]]) -> float | None:
    candidates = [
        row["return_mean"]
        for row in rows
        if row["family"] == "collector_reference"
        and "expert" in row["variant"]
        and "noisy" not in row["variant"]
        and row["return_mean"] is not None
    ]
    return max(candidates) if candidates else None


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "family",
        "stage",
        "variant",
        "profile",
        "checkpoint_kind",
        "n",
        "return_mean",
        "return_std_across_rows",
        "return_gap_to_expert",
        "episode_length_mean",
        "fall_rate_mean",
        "num_frames_total",
        "videos",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def write_markdown(path: Path, rows: list[dict[str, Any]], expert_return: float | None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    top = sorted(rows, key=lambda row: (row["return_mean"] is None, -(row["return_mean"] or -1e9)))
    lines = [
        "# MJLab QS Rollout Comparison",
        "",
        f"Expert-return reference: {fmt(expert_return)}",
        "",
        "| Family | Stage | Variant | Checkpoint | n | Return | Gap to Expert | Length | Fall |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in top:
        lines.append(
            "| "
            + " | ".join(
                [
                    row["family"],
                    row["stage"],
                    row["variant"],
                    row["checkpoint_kind"],
                    str(row["n"]),
                    fmt(row["return_mean"]),
                    fmt(row["return_gap_to_expert"]),
                    fmt(row["episode_length_mean"], 2),
                    fmt(row["fall_rate_mean"], 3),
                ]
            )
            + " |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    records = load_collector_records(Path(args.collector_root)) + load_policy_records(Path(args.policy_root))
    rows = aggregate(records)
    expert_return = choose_expert_return(rows)
    for row in rows:
        row["return_gap_to_expert"] = None if expert_return is None or row["return_mean"] is None else row["return_mean"] - expert_return
    rows.sort(key=lambda row: (row["return_mean"] is None, -(row["return_mean"] or -1e9)))
    write_csv(Path(args.output_csv), rows)
    write_markdown(Path(args.output_md), rows, expert_return)
    print(f"wrote {len(rows)} aggregate rows to {args.output_csv}")
    print(f"wrote markdown report to {args.output_md}")


if __name__ == "__main__":
    main()
