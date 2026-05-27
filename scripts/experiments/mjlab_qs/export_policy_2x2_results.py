#!/usr/bin/env python3
"""Export completed MJLab-QS PWM/Flow 2x2 policy results."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any


def read_manifest(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def load_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {}


def output_dir(row: dict[str, str]) -> Path:
    return (
        Path("scripts/outputs/mjlab_qs/policy_extraction")
        / row["stage"]
        / row.get("task_key", "task_unknown")
        / row["wm_method"]
        / row.get("policy_type", "mlp")
        / row.get("online_profile", "offline")
        / row["compute_profile"]
        / f"seed_{row['seed']}"
    )


def numeric(data: dict[str, Any], *keys: str) -> str:
    for key in keys:
        value = data.get(key)
        if isinstance(value, (int, float)):
            return str(value)
    return ""


def row_records(manifest: Path) -> list[dict[str, str]]:
    records = []
    for idx, row in enumerate(read_manifest(manifest)):
        out = output_dir(row)
        summary_path = out / "summary.json"
        eval_path = out / "eval_summary.json"
        final_path = out / "final_policy_extraction.pt"
        summary = load_json(summary_path)
        eval_summary = load_json(eval_path)
        done = summary_path.exists() and eval_path.exists() and final_path.exists()
        records.append(
            {
                "row": str(idx),
                "stage": row["stage"],
                "task_key": row.get("task_key", ""),
                "wm_method": row["wm_method"],
                "policy_type": row.get("policy_type", "mlp"),
                "seed": row["seed"],
                "status": "done" if done else "missing",
                "eval_return_mean": numeric(eval_summary, "return_mean")
                or numeric(summary, "eval/return_mean"),
                "eval_return_std": numeric(eval_summary, "return_std")
                or numeric(summary, "eval/return_std"),
                "eval_episode_length_mean": numeric(eval_summary, "episode_length_mean")
                or numeric(summary, "eval/episode_length_mean"),
                "best_imagined_return": numeric(summary, "best_imagined_return"),
                "best_iter": numeric(summary, "best_iter"),
                "wall_clock_seconds": numeric(summary, "wall_clock_seconds"),
                "summary": str(summary_path) if summary_path.exists() else "",
                "eval_summary": str(eval_path) if eval_path.exists() else "",
                "final_policy": str(final_path) if final_path.exists() else "",
            }
        )
    return records


def mean_std(values: list[float]) -> tuple[str, str]:
    if not values:
        return "", ""
    mean = sum(values) / len(values)
    if len(values) == 1:
        return str(mean), ""
    var = sum((value - mean) ** 2 for value in values) / (len(values) - 1)
    return str(mean), str(math.sqrt(var))


def aggregate_records(records: list[dict[str, str]]) -> list[dict[str, str]]:
    groups: dict[tuple[str, str], list[dict[str, str]]] = {}
    for record in records:
        groups.setdefault((record["wm_method"], record["policy_type"]), []).append(record)

    aggregates = []
    for (wm_method, policy_type), group in sorted(groups.items()):
        completed = [record for record in group if record["status"] == "done"]
        return_values = [
            float(record["eval_return_mean"])
            for record in completed
            if record["eval_return_mean"] != ""
        ]
        length_values = [
            float(record["eval_episode_length_mean"])
            for record in completed
            if record["eval_episode_length_mean"] != ""
        ]
        imagined_values = [
            float(record["best_imagined_return"])
            for record in completed
            if record["best_imagined_return"] != ""
        ]
        return_mean, return_std = mean_std(return_values)
        length_mean, length_std = mean_std(length_values)
        imagined_mean, imagined_std = mean_std(imagined_values)
        aggregates.append(
            {
                "wm_method": wm_method,
                "policy_type": policy_type,
                "completed": str(len(completed)),
                "expected": str(len(group)),
                "eval_return_mean": return_mean,
                "eval_return_std_across_seeds": return_std,
                "eval_episode_length_mean": length_mean,
                "eval_episode_length_std_across_seeds": length_std,
                "best_imagined_return_mean": imagined_mean,
                "best_imagined_return_std_across_seeds": imagined_std,
            }
        )
    return aggregates


def write_csv(path: Path, rows: list[dict[str, str]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--rows-output", type=Path, required=True)
    parser.add_argument("--aggregate-output", type=Path, required=True)
    parser.add_argument("--require-complete", action="store_true")
    args = parser.parse_args()

    rows = row_records(args.manifest)
    missing = [row for row in rows if row["status"] != "done"]
    if args.require_complete and missing:
        raise SystemExit(f"Missing {len(missing)} rows; first missing row={missing[0]['row']}")

    row_fields = [
        "row",
        "stage",
        "task_key",
        "wm_method",
        "policy_type",
        "seed",
        "status",
        "eval_return_mean",
        "eval_return_std",
        "eval_episode_length_mean",
        "best_imagined_return",
        "best_iter",
        "wall_clock_seconds",
        "summary",
        "eval_summary",
        "final_policy",
    ]
    aggregate_fields = [
        "wm_method",
        "policy_type",
        "completed",
        "expected",
        "eval_return_mean",
        "eval_return_std_across_seeds",
        "eval_episode_length_mean",
        "eval_episode_length_std_across_seeds",
        "best_imagined_return_mean",
        "best_imagined_return_std_across_seeds",
    ]
    write_csv(args.rows_output, rows, row_fields)
    aggregates = aggregate_records(rows)
    write_csv(args.aggregate_output, aggregates, aggregate_fields)
    print(
        f"wrote {len(rows)} rows to {args.rows_output} and "
        f"{len(aggregates)} aggregates to {args.aggregate_output}"
    )


if __name__ == "__main__":
    main()
