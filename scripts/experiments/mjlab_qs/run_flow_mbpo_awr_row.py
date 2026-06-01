#!/usr/bin/env python3
"""Run one Flow-MBPO AWR update manifest row."""

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

    output_dir = Path(row.get("output_dir") or row.get("awr_output_dir") or "")
    if not str(output_dir):
        stage = row.get("stage") or "flow_mbpo_awr"
        seed = row.get("seed") or "0"
        output_dir = Path("scripts/outputs/mjlab_qs/flow_mbpo_v0_awr") / stage / f"seed_{seed}"

    cmd = [
        args.python_bin,
        "scripts/experiments/mjlab_qs/run_flow_mbpo_v0_awr_update.py",
        "--dataset",
        row["dataset"],
        "--metadata",
        row["metadata"],
        "--normalization",
        row["normalization"],
        "--policy-checkpoint",
        row["policy_checkpoint"],
        "--synthetic-replay",
        row["synthetic_replay"],
        "--output-dir",
        str(output_dir),
        "--device",
        row.get("device") or args.device,
        "--seed",
        row.get("seed") or "0",
    ]

    optional_value_args = [
        "update_iters",
        "real_batch_size",
        "synthetic_batch_size",
        "actor_lr",
        "adv_temperature",
        "weight_clip",
        "bc_anchor_weight",
        "action_deviation_weight",
        "support_action_penalty_weight",
        "support_max_rows",
        "support_probe_rows",
        "support_threshold",
        "support_threshold_quantile",
        "support_state_weight",
        "support_command_weight",
        "support_action_weight",
        "support_risk_features",
        "support_risk_penalty_weight",
        "support_risk_batch_size",
        "support_risk_min_distance",
        "support_risk_top_quantile",
        "conservative_q_weight",
        "critic_actor_weight",
        "critic_lr",
        "critic_hidden",
        "critic_gamma",
        "critic_tau",
        "critic_random_actions",
        "critic_ood_action_source",
        "critic_action_noise_std",
        "critic_cql_temperature",
        "grad_norm",
        "split",
        "quality_filter",
        "log_every",
        "real_eval_every",
        "real_eval_episodes",
        "real_eval_num_envs",
        "real_eval_selection_metric",
        "real_eval_length_weight",
        "real_eval_fall_penalty",
        "real_eval_stop_score_below",
        "real_eval_early_stop_patience",
        "real_eval_min_delta",
        "real_eval_baseline_return",
        "real_eval_baseline_length",
        "real_eval_baseline_fall",
        "episode_length",
        "task_id",
        "command_dim",
        "command_position",
        "wandb_project",
        "wandb_group",
        "wandb_name",
        "notes",
    ]
    for key in optional_value_args:
        if present(row, key):
            cmd.extend([f"--{key.replace('_', '-')}", row[key]])
    if truthy(row.get("enable_wandb")):
        cmd.append("--enable-wandb")

    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
