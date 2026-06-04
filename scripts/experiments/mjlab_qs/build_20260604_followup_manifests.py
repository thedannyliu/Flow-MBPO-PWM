#!/usr/bin/env python3
"""Build 2026-06-04 MJLab follow-up manifests."""

from __future__ import annotations

import csv
from pathlib import Path


ROOT = Path("scripts/outputs/mjlab_qs")
MANIFEST_DIR = Path("scripts/experiments/mjlab_qs/manifests")
TASK_KEY = "velocity_flat_unitree_g1"
TASK_ID = "Mjlab-Velocity-Flat-Unitree-G1"
DATASET_STAGE = "rerun_a25_native_qs_g1stage4_expertboost_20260527"
DATASET = ROOT / "windows" / DATASET_STAGE / TASK_KEY / "d_qs_core_h16.pt"
METADATA = ROOT / "windows" / DATASET_STAGE / TASK_KEY / "d_qs_core_h16.json"
NORMALIZATION = ROOT / "windows" / DATASET_STAGE / TASK_KEY / "d_qs_core_h16_normalization.json"
BC_ROOT = ROOT / "policy_extraction" / "rerun_g1_bc_expert_uniform_mlp50k_20260528" / TASK_KEY
WM_STAGE = ROOT / "results" / DATASET_STAGE / TASK_KEY
H1_REPLAY = ROOT / "flow_mbpo_v0_replay" / "flow_endpoint_ensemble_seed0_h1_unc0p5_q0p90_truncate_check" / "synthetic_replay.pt"


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    if not rows:
        raise RuntimeError(f"no rows for {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} rows: {path}")


def policy_checkpoint(seed: int) -> Path:
    return BC_ROOT / "mlp_ref" / "mlp" / "offline" / "bc50k_expert_uniform_policy0k" / f"seed_{seed}" / "final_policy_extraction.pt"


def awr_base(stage: str, seed: int, output_dir: Path, notes: str) -> dict[str, str]:
    return {
        "stage": stage,
        "task_key": TASK_KEY,
        "task_id": TASK_ID,
        "dataset": str(DATASET),
        "metadata": str(METADATA),
        "normalization": str(NORMALIZATION),
        "policy_checkpoint": str(policy_checkpoint(seed)),
        "synthetic_replay": str(H1_REPLAY),
        "output_dir": str(output_dir),
        "seed": str(seed),
        "update_iters": "500",
        "real_batch_size": "224",
        "synthetic_batch_size": "32",
        "actor_lr": "1e-5",
        "advantage_source": "reward",
        "adv_temperature": "1.0",
        "weight_clip": "20.0",
        "bc_anchor_weight": "1.0",
        "action_deviation_weight": "0.0",
        "support_action_penalty_weight": "0.0",
        "conservative_q_weight": "0.0",
        "critic_actor_weight": "0.0",
        "grad_norm": "1.0",
        "split": "train",
        "quality_filter": "expert,expert_noisy",
        "real_quality_mixture": "",
        "real_require_no_fall_window": "false",
        "real_require_no_done_window": "false",
        "log_every": "50",
        "real_eval_every": "250",
        "real_eval_episodes": "8",
        "real_eval_num_envs": "16",
        "real_eval_selection_metric": "return_length_fall",
        "real_eval_length_weight": "0.01",
        "real_eval_fall_penalty": "100.0",
        "real_eval_stop_score_below": "-1000000000.0",
        "real_eval_early_stop_patience": "0",
        "real_eval_min_delta": "0.0",
        "real_eval_baseline_return": "45.8491",
        "real_eval_baseline_length": "594.97",
        "real_eval_baseline_fall": "0.625",
        "episode_length": "1000",
        "enable_wandb": "false",
        "wandb_project": "flow-mbpo-mjlab-followup-20260604",
        "wandb_group": stage,
        "wandb_name": output_dir.name,
        "notes": notes,
    }


def build_endpoint_h1_ablation() -> Path:
    stage = "flow_mbpo_endpoint_h1_multiseed_ablation_fix1_20260604"
    out_root = ROOT / stage / "h200"
    rows: list[dict[str, str]] = []
    configs = [
        ("r224_s32_anchor1_iter500", "224", "32"),
        ("r192_s64_anchor1_iter500", "192", "64"),
        ("r248_s8_anchor1_iter500", "248", "8"),
    ]
    for seed in [0, 1, 2]:
        for label, real_bs, synth_bs in configs:
            row = awr_base(
                stage,
                seed,
                out_root / f"endpoint_h1_{label}_s{seed}",
                "Endpoint H1 Flow-MBPO AWR multiseed/ratio ablation; synthetic replay source is flow_endpoint WM seed0 H1; fix1 disables premature real-eval early stop.",
            )
            row["real_batch_size"] = real_bs
            row["synthetic_batch_size"] = synth_bs
            row["wandb_name"] = f"endpoint_h1_{label}_s{seed}_h200"
            rows.append(row)
    path = MANIFEST_DIR / "flow_mbpo_endpoint_h1_multiseed_ablation_fix1_h200_20260604.csv"
    write_csv(path, rows)
    return path


def build_data_distribution_awr() -> Path:
    stage = "flow_mbpo_data_distribution_awr_fix1_20260604"
    out_root = ROOT / stage / "h100"
    rows: list[dict[str, str]] = []
    configs = [
        {
            "label": "mixed_uniform_windows",
            "quality_filter": "expert,expert_noisy,medium,random_smooth",
            "mixture": "",
            "no_fall": "false",
            "no_done": "false",
        },
        {
            "label": "expert_only",
            "quality_filter": "expert",
            "mixture": "",
            "no_fall": "false",
            "no_done": "false",
        },
        {
            "label": "expert50_medium50",
            "quality_filter": "expert,medium",
            "mixture": "expert:0.5,medium:0.5",
            "no_fall": "false",
            "no_done": "false",
        },
        {
            "label": "expert50_noisy50",
            "quality_filter": "expert,expert_noisy",
            "mixture": "expert:0.5,expert_noisy:0.5",
            "no_fall": "false",
            "no_done": "false",
        },
        {
            "label": "nofall_nodone_success100_proxy",
            "quality_filter": "expert,expert_noisy,medium",
            "mixture": "",
            "no_fall": "true",
            "no_done": "true",
        },
    ]
    for seed in [0, 1]:
        for cfg in configs:
            row = awr_base(
                stage,
                seed,
                out_root / f"{cfg['label']}_s{seed}",
                "Offline dataset-distribution Flow-MBPO-style AWR extraction on fixed endpoint H1 replay; fix1 disables premature real-eval early stop.",
            )
            row["quality_filter"] = cfg["quality_filter"]
            row["real_quality_mixture"] = cfg["mixture"]
            row["real_require_no_fall_window"] = cfg["no_fall"]
            row["real_require_no_done_window"] = cfg["no_done"]
            row["wandb_name"] = f"{cfg['label']}_s{seed}_h100"
            rows.append(row)
    path = MANIFEST_DIR / "flow_mbpo_data_distribution_awr_fix1_h100_20260604.csv"
    write_csv(path, rows)
    return path


def build_2x2_all_expert() -> Path:
    stage = "rerun_g1_bcwarm_pwm_bcreg10_2x2_allexpert_20260604"
    rows: list[dict[str, str]] = []
    for seed in [0, 1, 2]:
        for wm_method in ["mlp_ref", "flow_endpoint"]:
            for policy_type in ["mlp", "flow"]:
                rows.append(
                    {
                        "stage": stage,
                        "task_key": TASK_KEY,
                        "task_id": TASK_ID,
                        "dataset": str(DATASET),
                        "metadata": str(METADATA),
                        "normalization": str(NORMALIZATION),
                        "wm_checkpoint": str(WM_STAGE / wm_method / f"seed_{seed}" / "best.pt"),
                        "wm_method": wm_method,
                        "policy_type": policy_type,
                        "seed": str(seed),
                        "compute_profile": "bc50k_policy2k_bcreg10_allexpert",
                        "online_profile": "offline",
                        "policy_iters": "2000",
                        "batch_size": "64",
                        "actor_lr": "",
                        "critic_lr": "",
                        "bc_lr": "",
                        "critic_iterations": "",
                        "eval_every": "500",
                        "eval_episodes": "16",
                        "eval_num_envs": "16",
                        "episode_length": "1000",
                        "bc_warmstart_iters": "50000",
                        "policy_bc_reg": "10.0",
                        "bc_quality_filter": "expert",
                        "bc_quality_window_action_norm_max": "",
                        "bc_quality_loss_weights": "",
                        "bc_yaw_abs_loss_weights": "",
                        "policy_quality_filter": "expert",
                        "bc_sampling": "quality_balanced",
                        "policy_sampling": "quality_balanced",
                        "bc_action_rate_reg": "0.0",
                        "online_finetune_rounds": "0",
                        "wandb_project": "flow-mbpo-mjlab-bcwarm-pwm-flow-2x2-20260604",
                        "wandb_group": f"{stage}_{TASK_KEY}",
                        "skip_real_eval": "false",
                        "disable_wandb": "true",
                    }
                )
    path = MANIFEST_DIR / "rerun_g1_bcwarm_pwm_bcreg10_2x2_allexpert_a100_20260604.csv"
    write_csv(path, rows)
    return path


def main() -> None:
    build_endpoint_h1_ablation()
    build_data_distribution_awr()
    build_2x2_all_expert()


if __name__ == "__main__":
    main()
