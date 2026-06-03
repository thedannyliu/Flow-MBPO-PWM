#!/usr/bin/env python3
"""Export MJLab-QS world-model ablation summaries from training manifests."""

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
        Path("scripts/outputs/mjlab_qs/results")
        / row["stage"]
        / row.get("task_key", "task_unknown")
        / row["method"]
        / f"seed_{row['seed']}"
    )


def variant_name(row: dict[str, str]) -> str:
    weight = row.get("sigreg_weight", "")
    if weight and weight not in {"0", "0.0", "0.00"}:
        return f"{row['method']}_sigreg{weight}"
    return row["method"]


def numeric(data: dict[str, Any], key: str) -> str:
    value = data.get(key)
    if isinstance(value, (int, float)):
        return str(value)
    return ""


def row_records(manifests: list[Path]) -> list[dict[str, str]]:
    records = []
    for manifest in manifests:
        for row in read_manifest(manifest):
            out = output_dir(row)
            summary_path = out / "summary.json"
            best_path = out / "best.pt"
            summary = load_json(summary_path)
            records.append(
                {
                    "stage": row["stage"],
                    "task_key": row.get("task_key", ""),
                    "variant": variant_name(row),
                    "method": row["method"],
                    "seed": row["seed"],
                    "sigreg_weight": row.get("sigreg_weight", "0"),
                    "status": "done" if summary_path.exists() else "partial" if best_path.exists() else "missing",
                    "best_val_rollout_dyn_mse_H16": numeric(summary, "best_val_rollout_dyn_mse_H16"),
                    "test_rollout_dyn_mse_H16": numeric(summary, "test/rollout_dyn_mse_H16"),
                    "test_reward_mse": numeric(summary, "test/reward_mse"),
                    "test_one_step_dyn_mse": numeric(summary, "test/one_step_dyn_mse"),
                    "best_iter": numeric(summary, "best_iter"),
                    "wall_clock_seconds": numeric(summary, "wall_clock_seconds"),
                    "summary": str(summary_path) if summary_path.exists() else "",
                    "best": str(best_path) if best_path.exists() else "",
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
    groups: dict[str, list[dict[str, str]]] = {}
    for record in records:
        groups.setdefault(record["variant"], []).append(record)

    aggregates = []
    for variant, group in sorted(groups.items()):
        completed = [record for record in group if record["status"] == "done"]
        h16 = [
            float(record["test_rollout_dyn_mse_H16"])
            for record in completed
            if record["test_rollout_dyn_mse_H16"]
        ]
        reward = [
            float(record["test_reward_mse"])
            for record in completed
            if record["test_reward_mse"]
        ]
        val = [
            float(record["best_val_rollout_dyn_mse_H16"])
            for record in completed
            if record["best_val_rollout_dyn_mse_H16"]
        ]
        h16_mean, h16_std = mean_std(h16)
        reward_mean, reward_std = mean_std(reward)
        val_mean, val_std = mean_std(val)
        aggregates.append(
            {
                "variant": variant,
                "completed": str(len(completed)),
                "expected": str(len(group)),
                "test_rollout_dyn_mse_H16_mean": h16_mean,
                "test_rollout_dyn_mse_H16_std": h16_std,
                "best_val_rollout_dyn_mse_H16_mean": val_mean,
                "best_val_rollout_dyn_mse_H16_std": val_std,
                "test_reward_mse_mean": reward_mean,
                "test_reward_mse_std": reward_std,
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
    parser.add_argument("--manifest", type=Path, action="append", required=True)
    parser.add_argument("--rows-output", type=Path, required=True)
    parser.add_argument("--aggregate-output", type=Path, required=True)
    parser.add_argument("--require-complete", action="store_true")
    args = parser.parse_args()

    rows = row_records(args.manifest)
    missing = [row for row in rows if row["status"] != "done"]
    if args.require_complete and missing:
        raise SystemExit(
            f"Missing {len(missing)} rows; first missing variant={missing[0]['variant']} "
            f"seed={missing[0]['seed']}"
        )

    row_fields = [
        "stage",
        "task_key",
        "variant",
        "method",
        "seed",
        "sigreg_weight",
        "status",
        "best_val_rollout_dyn_mse_H16",
        "test_rollout_dyn_mse_H16",
        "test_reward_mse",
        "test_one_step_dyn_mse",
        "best_iter",
        "wall_clock_seconds",
        "summary",
        "best",
    ]
    aggregate_fields = [
        "variant",
        "completed",
        "expected",
        "test_rollout_dyn_mse_H16_mean",
        "test_rollout_dyn_mse_H16_std",
        "best_val_rollout_dyn_mse_H16_mean",
        "best_val_rollout_dyn_mse_H16_std",
        "test_reward_mse_mean",
        "test_reward_mse_std",
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
