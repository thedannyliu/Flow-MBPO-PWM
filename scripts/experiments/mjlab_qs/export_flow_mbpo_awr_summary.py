#!/usr/bin/env python3
"""Export Flow-MBPO AWR update diagnostics from manifest rows and summaries."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


FIELDS = [
    "status",
    "stage",
    "seed",
    "output_dir",
    "critic_enabled",
    "conservative_q_weight",
    "critic_actor_weight",
    "critic_random_actions",
    "critic_ood_action_source",
    "critic_action_noise_std",
    "critic_cql_temperature",
    "update_iters",
    "last_iter",
    "awr_loss",
    "critic_loss",
    "critic_bellman_loss",
    "critic_cql_loss",
    "critic_cql_gap_mean",
    "critic_q_data_mean",
    "critic_q_actor_mean",
    "critic_q_random_mean",
    "critic_q_random_max",
    "synthetic_reward_mean",
    "synthetic_done_fraction",
    "best_real_return",
    "best_real_score",
    "best_is_true_snapshot",
    "early_stop_iter",
    "early_stop_reason",
    "wandb_run_id",
    "wandb_run_url",
    "notes",
    "summary_path",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", type=Path, action="append", default=[])
    p.add_argument("--summary", type=Path, action="append", default=[])
    p.add_argument("--root", type=Path, action="append", default=[])
    p.add_argument("--output-csv", type=Path, required=True)
    p.add_argument("--output-md", type=Path, required=True)
    p.add_argument("--require-complete", action="store_true")
    return p.parse_args()


def scalar(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return str(value).lower()
    return str(value)


def nested(summary: dict[str, Any], group: str, key: str) -> str:
    value = summary.get(group)
    if not isinstance(value, dict):
        return ""
    return scalar(value.get(key))


def last_metric(summary: dict[str, Any], key: str) -> str:
    metrics = summary.get("last_metrics")
    if not isinstance(metrics, dict):
        return ""
    return scalar(metrics.get(key))


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def row_output_dir(row: dict[str, str]) -> Path:
    return Path(row.get("output_dir") or row.get("awr_output_dir") or "")


def base_record_from_manifest(row: dict[str, str]) -> dict[str, str]:
    return {
        "status": "missing",
        "stage": row.get("stage", ""),
        "seed": row.get("seed", ""),
        "output_dir": str(row_output_dir(row)),
        "critic_enabled": "",
        "conservative_q_weight": row.get("conservative_q_weight", ""),
        "critic_actor_weight": row.get("critic_actor_weight", ""),
        "critic_random_actions": row.get("critic_random_actions", ""),
        "critic_ood_action_source": row.get("critic_ood_action_source", ""),
        "critic_action_noise_std": row.get("critic_action_noise_std", ""),
        "critic_cql_temperature": row.get("critic_cql_temperature", ""),
        "update_iters": row.get("update_iters", ""),
        "last_iter": "",
        "awr_loss": "",
        "critic_loss": "",
        "critic_bellman_loss": "",
        "critic_cql_loss": "",
        "critic_cql_gap_mean": "",
        "critic_q_data_mean": "",
        "critic_q_actor_mean": "",
        "critic_q_random_mean": "",
        "critic_q_random_max": "",
        "synthetic_reward_mean": "",
        "synthetic_done_fraction": "",
        "best_real_return": "",
        "best_real_score": "",
        "best_is_true_snapshot": "",
        "early_stop_iter": "",
        "early_stop_reason": "",
        "wandb_run_id": "",
        "wandb_run_url": "",
        "notes": row.get("notes", ""),
        "summary_path": "",
    }


def record_from_summary(path: Path, manifest_row: dict[str, str] | None = None) -> dict[str, str]:
    summary = read_json(path)
    base = base_record_from_manifest(manifest_row or {})
    output_dir = scalar(summary.get("output_dir")) or str(path.parent)
    base.update(
        {
            "status": "complete",
            "stage": base["stage"] or scalar(summary.get("stage")),
            "seed": scalar(summary.get("seed")) or base["seed"],
            "output_dir": output_dir,
            "critic_enabled": scalar(summary.get("critic_enabled")),
            "conservative_q_weight": scalar(summary.get("conservative_q_weight")) or base["conservative_q_weight"],
            "critic_actor_weight": scalar(summary.get("critic_actor_weight")) or base["critic_actor_weight"],
            "critic_random_actions": scalar(summary.get("critic_random_actions")) or base["critic_random_actions"],
            "critic_ood_action_source": scalar(summary.get("critic_ood_action_source")) or base["critic_ood_action_source"],
            "critic_action_noise_std": scalar(summary.get("critic_action_noise_std"))
            or base["critic_action_noise_std"],
            "critic_cql_temperature": scalar(summary.get("critic_cql_temperature")) or base["critic_cql_temperature"],
            "update_iters": scalar(summary.get("update_iters")) or base["update_iters"],
            "last_iter": last_metric(summary, "awr/iter"),
            "awr_loss": last_metric(summary, "awr/loss"),
            "critic_loss": last_metric(summary, "critic/loss"),
            "critic_bellman_loss": last_metric(summary, "critic/bellman_loss"),
            "critic_cql_loss": last_metric(summary, "critic/cql_loss"),
            "critic_cql_gap_mean": last_metric(summary, "critic/cql_gap_mean"),
            "critic_q_data_mean": last_metric(summary, "critic/q_data_mean"),
            "critic_q_actor_mean": last_metric(summary, "critic/q_actor_mean"),
            "critic_q_random_mean": last_metric(summary, "critic/q_random_mean"),
            "critic_q_random_max": last_metric(summary, "critic/q_random_max"),
            "synthetic_reward_mean": nested(summary, "synthetic_reward_conservative", "mean"),
            "synthetic_done_fraction": scalar(summary.get("synthetic_done_fraction")),
            "best_real_return": scalar(summary.get("best_real_return")),
            "best_real_score": scalar(summary.get("best_real_score")),
            "best_is_true_snapshot": scalar(summary.get("best_is_true_snapshot")),
            "early_stop_iter": scalar(summary.get("early_stop_iter")),
            "early_stop_reason": scalar(summary.get("early_stop_reason")),
            "wandb_run_id": scalar(summary.get("wandb_run_id")),
            "wandb_run_url": scalar(summary.get("wandb_run_url")),
            "notes": scalar(summary.get("notes")) or base["notes"],
            "summary_path": str(path),
        }
    )
    return base


def records_from_manifest(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    records: list[dict[str, str]] = []
    for row in rows:
        summary_path = row_output_dir(row) / "summary.json"
        if summary_path.exists():
            records.append(record_from_summary(summary_path, row))
        else:
            records.append(base_record_from_manifest(row))
    return records


def collect_records(args: argparse.Namespace) -> list[dict[str, str]]:
    records: list[dict[str, str]] = []
    for manifest in args.manifest:
        records.extend(records_from_manifest(manifest))
    seen = {record["summary_path"] for record in records if record["summary_path"]}
    for root in args.root:
        for path in sorted(root.glob("**/summary.json")):
            resolved = str(path)
            if resolved not in seen:
                records.append(record_from_summary(path))
                seen.add(resolved)
    for path in args.summary:
        resolved = str(path)
        if path.exists() and resolved not in seen:
            records.append(record_from_summary(path))
            seen.add(resolved)
    return records


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def fmt(value: str) -> str:
    if value == "":
        return ""
    try:
        return f"{float(value):.6g}"
    except ValueError:
        return value.replace("|", "\\|")


def write_md(path: Path, rows: list[dict[str, str]]) -> None:
    columns = [
        "status",
        "critic_ood_action_source",
        "critic_action_noise_std",
        "critic_cql_temperature",
        "last_iter",
        "critic_cql_gap_mean",
        "critic_q_random_mean",
        "critic_q_random_max",
        "critic_bellman_loss",
        "best_real_return",
        "notes",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(fmt(row[column]) for column in columns) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    rows = collect_records(args)
    complete_count = sum(1 for row in rows if row["status"] == "complete")
    if args.require_complete and complete_count != len(rows):
        raise SystemExit(f"Only {complete_count}/{len(rows)} AWR summaries are complete")
    write_csv(args.output_csv, rows)
    write_md(args.output_md, rows)
    print(json.dumps({"rows": len(rows), "complete": complete_count, "output_csv": str(args.output_csv)}, sort_keys=True))


if __name__ == "__main__":
    main()
