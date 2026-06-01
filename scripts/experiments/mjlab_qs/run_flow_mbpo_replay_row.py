#!/usr/bin/env python3
"""Run one Flow-MBPO synthetic-replay manifest row."""

from __future__ import annotations

import argparse
import csv
import subprocess
from pathlib import Path


def truthy(value: str | None) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes"}


def present(row: dict[str, str], key: str) -> bool:
    return bool(str(row.get(key, "")).strip())


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", required=True)
    p.add_argument("--row-index", type=int, required=True)
    p.add_argument("--python-bin", default="python")
    p.add_argument("--device", default="cuda:0")
    args = p.parse_args()

    with open(args.manifest, newline="", encoding="utf-8") as f:
        row = list(csv.DictReader(f))[args.row_index]

    output_dir = Path(row.get("output_dir") or row.get("replay_output_dir") or "")
    if not str(output_dir):
        stage = row.get("stage") or "flow_mbpo_replay"
        seed = row.get("seed") or "0"
        output_dir = Path("scripts/outputs/mjlab_qs/flow_mbpo_v0_replay") / stage / f"seed_{seed}"

    cmd = [
        args.python_bin,
        "scripts/experiments/mjlab_qs/prepare_flow_mbpo_v0_synthetic_replay.py",
        "--synthetic-buffer",
        row["synthetic_buffer"],
        "--output-dir",
        str(output_dir),
        "--lambda-uncertainty",
        row.get("lambda_uncertainty") or "0.0",
        "--lambda-fall",
        row.get("lambda_fall") or "0.0",
        "--uncertainty-quantile-termination",
        row.get("uncertainty_quantile_termination") or "0.0",
        "--done-threshold",
        row.get("done_threshold") or "0.5",
        "--fall-threshold",
        row.get("fall_threshold") or "0.5",
        "--max-transitions",
        row.get("max_transitions") or "0",
        "--seed",
        row.get("seed") or "0",
    ]

    optional_value_args = [
        "support_dataset",
        "support_metadata",
        "support_normalization",
        "support_risk_penalty_weight",
        "support_max_rows",
        "support_probe_rows",
        "support_threshold",
        "support_threshold_quantile",
        "support_distance_batch_size",
        "support_split",
        "support_quality_filter",
        "support_state_weight",
        "support_command_weight",
        "support_action_weight",
        "wandb_project",
        "wandb_group",
        "wandb_name",
        "notes",
    ]
    for key in optional_value_args:
        if present(row, key):
            cmd.extend([f"--{key.replace('_', '-')}", row[key]])
    if truthy(row.get("support_risk_termination")):
        cmd.append("--support-risk-termination")
    if truthy(row.get("truncate_rollouts_after_done")):
        cmd.append("--truncate-rollouts-after-done")
    if truthy(row.get("enable_wandb")):
        cmd.append("--enable-wandb")

    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
