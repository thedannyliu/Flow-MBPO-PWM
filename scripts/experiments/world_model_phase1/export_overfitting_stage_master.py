#!/usr/bin/env python3
from __future__ import annotations

import csv
import glob
import json
from collections import defaultdict
from pathlib import Path


ROOT = Path("scripts/outputs/world_model_phase1")
DEFAULT_DATASET_KEY = "velocity_flat_unitree_go1_random_initial"


def mean(xs: list[float]) -> float:
    return sum(xs) / len(xs)


def infer_profile_from_alg_config(alg_config: str, method_key: str) -> str:
    name = Path(alg_config).stem
    mapping = {
        "pwm_5M_baseline_pwmorig": "mlp_ref",
        "pwm_5M_flow_v2_substeps4": "flow_ref_uniform_heun4",
    }
    if name in mapping:
        return mapping[name]
    if method_key == "mlpwm_mlppolicy":
        return "mlp_ref"
    if method_key == "flowwm_mlppolicy":
        return "flow_ref_uniform_heun4"
    return name


def aggregate_phase1_initial() -> list[dict[str, object]]:
    pattern = ROOT / "phase1_wm_overfit_formal" / "*" / "*" / "seed_*" / "phase1_summary.json"
    rows: defaultdict[tuple[str, str, str, str], list[dict[str, object]]] = defaultdict(list)
    for path in sorted(glob.glob(str(pattern))):
        d = json.load(open(path))
        parts = Path(path).parts
        task_key = parts[-4]
        method_key = parts[-3]
        profile = infer_profile_from_alg_config(d["alg_config"], method_key)
        rows[("phase1_initial_overfit", task_key, DEFAULT_DATASET_KEY, method_key, profile)].append(d)

    out: list[dict[str, object]] = []
    for (stage, task_key, dataset_key, method_key, profile), ds in sorted(rows.items()):
        out.append(
            {
                "record_type": "model_eval",
                "stage_family": "phase1_initial_overfit",
                "stage": stage,
                "task_key": task_key,
                "dataset_key": dataset_key,
                "method_key": method_key,
                "profile": profile,
                "n_eval_completed": len(ds),
                "train_wm_loss_mean": mean([x["final_train"]["wm_loss"] for x in ds]),
                "train_one_step_dyn_loss_mean": mean([x["final_train"]["one_step_dyn_loss"] for x in ds]),
                "train_rollout_dyn_loss_mean": mean([x["final_train"]["rollout_dyn_loss"] for x in ds]),
                "val_wm_loss_mean": mean([x["final_val"]["wm_loss"] for x in ds]),
                "val_one_step_dyn_loss_mean": mean([x["final_val"]["one_step_dyn_loss"] for x in ds]),
                "val_rollout_dyn_loss_mean": mean([x["final_val"]["rollout_dyn_loss"] for x in ds]),
                "train_one_step_reward_loss_mean": mean([x["final_train"]["one_step_reward_loss"] for x in ds]),
                "val_one_step_reward_loss_mean": mean([x["final_val"]["one_step_reward_loss"] for x in ds]),
                "elapsed_seconds_mean": mean([x["elapsed_seconds"] for x in ds]),
                "env_config": "",
                "action_mode": "",
                "task_id_requested": "",
                "task_id_resolved": "",
                "strict_task_id_match": "",
                "teacher_checkpoint": "",
                "target_episodes": "",
                "collected_episodes": "",
                "num_windows": "",
                "obs_dim": "",
                "act_dim": "",
                "mixed_teacher_prob": "",
                "count_random_uniform": "",
                "count_teacher_policy": "",
                "count_mixed_teacher": "",
                "count_mixed_random": "",
            }
        )
    return out


def aggregate_phase16() -> list[dict[str, object]]:
    pattern = ROOT / "phase16_capacity_sidecar_formal" / "*" / "*" / "*" / "seed_*" / "phase1_summary.json"
    rows: defaultdict[tuple[str, str, str, str, str], list[dict[str, object]]] = defaultdict(list)
    for path in sorted(glob.glob(str(pattern))):
        d = json.load(open(path))
        parts = Path(path).parts
        task_key = parts[-5]
        method_key = parts[-4]
        profile = parts[-3]
        rows[("phase16_capacity_sidecar_formal", task_key, DEFAULT_DATASET_KEY, method_key, profile)].append(d)

    out: list[dict[str, object]] = []
    for (stage, task_key, dataset_key, method_key, profile), ds in sorted(rows.items()):
        out.append(
            {
                "record_type": "model_eval",
                "stage_family": "phase16_capacity_sidecar",
                "stage": stage,
                "task_key": task_key,
                "dataset_key": dataset_key,
                "method_key": method_key,
                "profile": profile,
                "n_eval_completed": len(ds),
                "train_wm_loss_mean": mean([x["final_train"]["wm_loss"] for x in ds]),
                "train_one_step_dyn_loss_mean": mean([x["final_train"]["one_step_dyn_loss"] for x in ds]),
                "train_rollout_dyn_loss_mean": mean([x["final_train"]["rollout_dyn_loss"] for x in ds]),
                "val_wm_loss_mean": mean([x["final_val"]["wm_loss"] for x in ds]),
                "val_one_step_dyn_loss_mean": mean([x["final_val"]["one_step_dyn_loss"] for x in ds]),
                "val_rollout_dyn_loss_mean": mean([x["final_val"]["rollout_dyn_loss"] for x in ds]),
                "train_one_step_reward_loss_mean": mean([x["final_train"]["one_step_reward_loss"] for x in ds]),
                "val_one_step_reward_loss_mean": mean([x["final_val"]["one_step_reward_loss"] for x in ds]),
                "elapsed_seconds_mean": mean([x["elapsed_seconds"] for x in ds]),
                "env_config": "",
                "action_mode": "",
                "task_id_requested": "",
                "task_id_resolved": "",
                "strict_task_id_match": "",
                "teacher_checkpoint": "",
                "target_episodes": "",
                "collected_episodes": "",
                "num_windows": "",
                "obs_dim": "",
                "act_dim": "",
                "mixed_teacher_prob": "",
                "count_random_uniform": "",
                "count_teacher_policy": "",
                "count_mixed_teacher": "",
                "count_mixed_random": "",
            }
        )
    return out


def load_existing_model_csv(path: str, stage_family: str, default_dataset_key: str = DEFAULT_DATASET_KEY) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows.append(
                {
                    "record_type": "model_eval",
                    "stage_family": stage_family,
                    "stage": row.get("stage", ""),
                    "task_key": row.get("task_key", ""),
                    "dataset_key": row.get("dataset_key", "") or default_dataset_key,
                    "method_key": row.get("method_key", ""),
                    "profile": row.get("profile", ""),
                    "n_eval_completed": row.get("n_eval_completed", ""),
                    "train_wm_loss_mean": row.get("train_wm_loss_mean", ""),
                    "train_one_step_dyn_loss_mean": row.get("train_one_step_dyn_loss_mean", ""),
                    "train_rollout_dyn_loss_mean": row.get("train_rollout_dyn_loss_mean", ""),
                    "val_wm_loss_mean": row.get("val_wm_loss_mean", ""),
                    "val_one_step_dyn_loss_mean": row.get("val_one_step_dyn_loss_mean", ""),
                    "val_rollout_dyn_loss_mean": row.get("val_rollout_dyn_loss_mean", ""),
                    "train_one_step_reward_loss_mean": row.get("train_one_step_reward_loss_mean", ""),
                    "val_one_step_reward_loss_mean": row.get("val_one_step_reward_loss_mean", ""),
                    "elapsed_seconds_mean": row.get("elapsed_seconds_mean", ""),
                    "env_config": "",
                    "action_mode": "",
                    "task_id_requested": "",
                    "task_id_resolved": "",
                    "strict_task_id_match": "",
                    "teacher_checkpoint": "",
                    "target_episodes": "",
                    "collected_episodes": "",
                    "num_windows": "",
                    "obs_dim": "",
                    "act_dim": "",
                    "mixed_teacher_prob": "",
                    "count_random_uniform": "",
                    "count_teacher_policy": "",
                    "count_mixed_teacher": "",
                    "count_mixed_random": "",
                }
            )
    return rows


def load_dataset_summary_csv(path: str) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows.append(
                {
                    "record_type": "dataset_collection",
                    "stage_family": "phase1_dataset_collection",
                    "stage": "phase1_dataset_collection",
                    "task_key": row.get("task_key", ""),
                    "dataset_key": row.get("task_key", ""),
                    "method_key": "",
                    "profile": "",
                    "n_eval_completed": "",
                    "train_wm_loss_mean": "",
                    "train_one_step_dyn_loss_mean": "",
                    "train_rollout_dyn_loss_mean": "",
                    "val_wm_loss_mean": "",
                    "val_one_step_dyn_loss_mean": "",
                    "val_rollout_dyn_loss_mean": "",
                    "train_one_step_reward_loss_mean": "",
                    "val_one_step_reward_loss_mean": "",
                    "elapsed_seconds_mean": "",
                    "env_config": row.get("env_config", ""),
                    "action_mode": row.get("action_mode", ""),
                    "task_id_requested": row.get("task_id_requested", ""),
                    "task_id_resolved": row.get("task_id_resolved", ""),
                    "strict_task_id_match": row.get("strict_task_id_match", ""),
                    "teacher_checkpoint": row.get("teacher_checkpoint", ""),
                    "target_episodes": row.get("target_episodes", ""),
                    "collected_episodes": row.get("collected_episodes", ""),
                    "num_windows": row.get("num_windows", ""),
                    "obs_dim": row.get("obs_dim", ""),
                    "act_dim": row.get("act_dim", ""),
                    "mixed_teacher_prob": row.get("mixed_teacher_prob", ""),
                    "count_random_uniform": row.get("count_random_uniform", ""),
                    "count_teacher_policy": row.get("count_teacher_policy", ""),
                    "count_mixed_teacher": row.get("count_mixed_teacher", ""),
                    "count_mixed_random": row.get("count_mixed_random", ""),
                }
            )
    return rows


def main() -> None:
    rows: list[dict[str, object]] = []
    rows.extend(aggregate_phase1_initial())
    rows.extend(
        load_existing_model_csv(
            "docs/phase1_objective_ablations_formal_results_20260402.csv",
            stage_family="phase1_objective_ablation",
        )
    )
    rows.extend(
        load_existing_model_csv(
            "docs/phase15_formulation_ablations_formal_results_20260402.csv",
            stage_family="phase15_formulation",
        )
    )
    rows.extend(aggregate_phase16())
    rows.extend(
        load_existing_model_csv(
            "docs/phase17_combined_results_20260402.csv",
            stage_family="phase17",
            default_dataset_key="",
        )
    )
    rows.extend(load_dataset_summary_csv("docs/phase1_dataset_variants_formal_summary_20260402.csv"))

    fieldnames = [
        "record_type",
        "stage_family",
        "stage",
        "task_key",
        "dataset_key",
        "method_key",
        "profile",
        "n_eval_completed",
        "train_wm_loss_mean",
        "train_one_step_dyn_loss_mean",
        "train_rollout_dyn_loss_mean",
        "val_wm_loss_mean",
        "val_one_step_dyn_loss_mean",
        "val_rollout_dyn_loss_mean",
        "train_one_step_reward_loss_mean",
        "val_one_step_reward_loss_mean",
        "elapsed_seconds_mean",
        "env_config",
        "action_mode",
        "task_id_requested",
        "task_id_resolved",
        "strict_task_id_match",
        "teacher_checkpoint",
        "target_episodes",
        "collected_episodes",
        "num_windows",
        "obs_dim",
        "act_dim",
        "mixed_teacher_prob",
        "count_random_uniform",
        "count_teacher_policy",
        "count_mixed_teacher",
        "count_mixed_random",
    ]
    out = Path("docs/overfitting_stage_master_results_20260403.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in sorted(rows, key=lambda r: (r["record_type"], r["stage_family"], r["stage"], r["task_key"], r["dataset_key"], r["method_key"], r["profile"])):
            writer.writerow(row)
    print(f"Wrote {out} with {len(rows)} rows")


if __name__ == "__main__":
    main()
