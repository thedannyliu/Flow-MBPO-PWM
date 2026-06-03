#!/usr/bin/env python3
"""Run one Flow-MBPO synthetic-buffer manifest row."""

from __future__ import annotations

import argparse
import csv
import subprocess
from pathlib import Path


def truthy(value: str | None) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes"}


def present(row: dict[str, str], key: str) -> bool:
    return bool(str(row.get(key, "")).strip())


def split_list(value: str) -> list[str]:
    normalized = value.replace("|", ";")
    return [item.strip() for item in normalized.split(";") if item.strip()]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", required=True)
    p.add_argument("--row-index", type=int, required=True)
    p.add_argument("--python-bin", default="python")
    p.add_argument("--device", default="cuda:0")
    args = p.parse_args()

    with open(args.manifest, newline="", encoding="utf-8") as f:
        row = list(csv.DictReader(f))[args.row_index]

    output_dir = Path(row.get("output_dir") or row.get("smoke_output_dir") or "")
    if not str(output_dir):
        stage = row.get("stage") or "flow_mbpo_smoke"
        seed = row.get("seed") or "0"
        output_dir = Path("scripts/outputs/mjlab_qs/flow_mbpo_v0_smoke") / stage / f"seed_{seed}"

    wm_checkpoints = split_list(row.get("wm_checkpoints") or row.get("wm_checkpoint") or "")
    if not wm_checkpoints:
        raise ValueError("Flow-MBPO smoke rows require wm_checkpoint or wm_checkpoints")

    cmd = [
        args.python_bin,
        "scripts/experiments/mjlab_qs/run_flow_mbpo_v0_smoke.py",
        "--dataset",
        row["dataset"],
        "--metadata",
        row["metadata"],
        "--normalization",
        row["normalization"],
        "--policy-checkpoint",
        row["policy_checkpoint"],
        "--output-dir",
        str(output_dir),
        "--device",
        row.get("device") or args.device,
        "--seed",
        row.get("seed") or "0",
        "--num-starts",
        row.get("num_starts") or "256",
        "--horizon",
        row.get("horizon") or "1",
        "--split",
        row.get("split") or "train",
        "--quality-filter",
        row.get("quality_filter") or "expert,expert_noisy",
    ]
    for checkpoint in wm_checkpoints:
        cmd.extend(["--wm-checkpoint", checkpoint])

    optional_value_args = [
        "support_max_rows",
        "support_probe_rows",
        "support_threshold",
        "support_threshold_quantile",
        "support_distance_batch_size",
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
    if truthy(row.get("enable_wandb")):
        cmd.append("--enable-wandb")

    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
