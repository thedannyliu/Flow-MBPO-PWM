#!/usr/bin/env python3
"""Build Flow-MBPO H1 root-cause manifests for 2026-06-04."""

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
EXACT_H1_REPLAY = ROOT / "flow_mbpo_v0_replay" / "flow_endpoint_ensemble_seed0_h1_unc0p5_q0p90" / "synthetic_replay.pt"
TRUNCATE_H1_REPLAY = (
    ROOT / "flow_mbpo_v0_replay" / "flow_endpoint_ensemble_seed0_h1_unc0p5_q0p90_truncate_check" / "synthetic_replay.pt"
)
STRONGEST_AWR = ROOT / "flow_mbpo_v0_awr" / "flow_endpoint_seed0_h1_unc0p5_q0p90_cons_r224_s32_anchor1_iter500_s0"
FIX1_ROOT = ROOT / "flow_mbpo_endpoint_h1_multiseed_ablation_fix1_20260604" / "h200"

EVAL_ROOT = ROOT / "flow_mbpo_h1_rootcause_checkpoint_eval_20260604" / "h100"
AWR_RATIO_ROOT = ROOT / "flow_mbpo_h1_exact_replay_realonly_ratio_20260604" / "h200"
AWR_DIAG_ROOT = ROOT / "flow_mbpo_h1_awr_diagnostics_20260604" / "h100"
REPLAY_QUALITY_ROOT = ROOT / "flow_mbpo_h1_replay_quality_20260604" / "l40s"

BASELINE_RETURN = "45.8491"
BASELINE_LENGTH = "594.97"
BASELINE_FALL = "0.625"


def require_path(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(path)
    return path


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
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} rows: {path}")


def bc_checkpoint(seed: int) -> Path:
    return require_path(
        BC_ROOT
        / "mlp_ref"
        / "mlp"
        / "offline"
        / "bc50k_expert_uniform_policy0k"
        / f"seed_{seed}"
        / "final_policy_extraction.pt"
    )


def eval_row(candidate: str, checkpoint: Path, kind: str, seed: int, notes: str) -> dict[str, str]:
    require_path(checkpoint)
    return {
        "stage": "flow_mbpo_h1_rootcause_checkpoint_eval_20260604",
        "task_key": TASK_KEY,
        "task_id": TASK_ID,
        "candidate": candidate,
        "policy_checkpoint": str(checkpoint),
        "checkpoint_kind": kind,
        "eval_output_dir": str(EVAL_ROOT / candidate),
        "eval_episodes": "40",
        "eval_num_envs": "16",
        "eval_max_steps": "1000",
        "eval_baseline_return": BASELINE_RETURN,
        "eval_baseline_length": BASELINE_LENGTH,
        "eval_baseline_fall": BASELINE_FALL,
        "dataset": str(DATASET),
        "metadata": str(METADATA),
        "normalization": str(NORMALIZATION),
        "seed": str(seed),
        "command_dim": "3",
        "command_position": "tail",
        "obs_mode": "normalized",
        "disable_wandb": "true",
        "wandb_project": "flow-mbpo-mjlab-h1-rootcause-20260604",
        "wandb_group": "checkpoint_eval",
        "wandb_name": f"{candidate}_eval40",
        "notes": notes,
    }


def build_checkpoint_eval_manifest() -> Path:
    rows: list[dict[str, str]] = []
    for seed in [0, 1, 2]:
        rows.append(
            eval_row(
                f"bc_seed{seed}_initial",
                bc_checkpoint(seed),
                "bc_initial",
                seed,
                "Initial BC checkpoint direct real-env eval for AWR damage baseline.",
            )
        )
    rows.extend(
        [
            eval_row(
                "strongest_exact_h1_final",
                STRONGEST_AWR / "final_policy_extraction.pt",
                "final",
                0,
                "Strongest historical Flow-MBPO H1 final checkpoint; prior eval40 return was 60.8721.",
            ),
            eval_row(
                "strongest_exact_h1_best_real_eval",
                STRONGEST_AWR / "best_policy_extraction.pt",
                "best_real_eval",
                0,
                "Strongest historical Flow-MBPO H1 best-real-eval checkpoint; prior eval40 return was 46.1720.",
            ),
        ]
    )

    configs = ["r224_s32_anchor1_iter500", "r192_s64_anchor1_iter500", "r248_s8_anchor1_iter500"]
    for seed in [0, 1, 2]:
        for cfg in configs:
            base = FIX1_ROOT / f"endpoint_h1_{cfg}_s{seed}"
            rows.extend(
                [
                    eval_row(
                        f"fix1_{cfg}_s{seed}_iter250",
                        base / "real_eval_snapshots" / "iter_000250_policy_extraction.pt",
                        "iter250",
                        seed,
                        "Fix1 H1 AWR snapshot after 250 updates; used to diagnose whether AWR progressively damages BC.",
                    ),
                    eval_row(
                        f"fix1_{cfg}_s{seed}_iter500",
                        base / "real_eval_snapshots" / "iter_000500_policy_extraction.pt",
                        "iter500",
                        seed,
                        "Fix1 H1 AWR snapshot after 500 updates; used to diagnose whether AWR progressively damages BC.",
                    ),
                    eval_row(
                        f"fix1_{cfg}_s{seed}_final",
                        base / "final_policy_extraction.pt",
                        "final",
                        seed,
                        "Fix1 H1 AWR final checkpoint direct real-env eval.",
                    ),
                    eval_row(
                        f"fix1_{cfg}_s{seed}_best_real_eval",
                        base / "best_policy_extraction.pt",
                        "best_real_eval",
                        seed,
                        "Fix1 H1 AWR best-real-eval checkpoint direct real-env eval.",
                    ),
                ]
            )
    path = MANIFEST_DIR / "flow_mbpo_h1_rootcause_checkpoint_eval_h100_20260604.csv"
    write_csv(path, rows)
    return path


def awr_base(stage: str, seed: int, output_dir: Path, real_bs: int, synth_bs: int) -> dict[str, str]:
    return {
        "stage": stage,
        "task_key": TASK_KEY,
        "task_id": TASK_ID,
        "dataset": str(DATASET),
        "metadata": str(METADATA),
        "normalization": str(NORMALIZATION),
        "policy_checkpoint": str(bc_checkpoint(seed)),
        "synthetic_replay": str(require_path(EXACT_H1_REPLAY)),
        "output_dir": str(output_dir),
        "seed": str(seed),
        "update_iters": "500",
        "real_batch_size": str(real_bs),
        "synthetic_batch_size": str(synth_bs),
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
        "real_eval_selection_metric": "return",
        "real_eval_length_weight": "0.01",
        "real_eval_fall_penalty": "100.0",
        "real_eval_stop_score_below": "-1000000000.0",
        "real_eval_early_stop_patience": "0",
        "real_eval_min_delta": "0.0",
        "real_eval_baseline_return": BASELINE_RETURN,
        "real_eval_baseline_length": BASELINE_LENGTH,
        "real_eval_baseline_fall": BASELINE_FALL,
        "episode_length": "1000",
        "enable_wandb": "false",
        "wandb_project": "flow-mbpo-mjlab-h1-rootcause-20260604",
        "wandb_group": stage,
        "wandb_name": output_dir.name,
        "notes": "Exact historical H1 replay real-only/synthetic-ratio AWR root-cause sweep; selection metric matches strongest-era default return.",
    }


def build_awr_ratio_manifest() -> Path:
    stage = "flow_mbpo_h1_exact_replay_realonly_ratio_20260604"
    ratios = [(256, 0), (248, 8), (224, 32), (192, 64)]
    rows: list[dict[str, str]] = []
    for seed in [0, 1, 2]:
        for real_bs, synth_bs in ratios:
            label = f"exact_h1_r{real_bs}_s{synth_bs}_anchor1_iter500_seed{seed}"
            rows.append(awr_base(stage, seed, AWR_RATIO_ROOT / label, real_bs, synth_bs))
    path = MANIFEST_DIR / "flow_mbpo_h1_exact_replay_realonly_ratio_h200_20260604.csv"
    write_csv(path, rows)
    return path


def diagnostic_row(case: str, bc_seed: int, replay: Path, policy_paths: list[tuple[str, Path]], notes: str) -> dict[str, str]:
    for _, path in policy_paths:
        require_path(path)
    return {
        "stage": "flow_mbpo_h1_awr_diagnostics_20260604",
        "case": case,
        "dataset": str(DATASET),
        "metadata": str(METADATA),
        "normalization": str(NORMALIZATION),
        "bc_checkpoint": str(bc_checkpoint(bc_seed)),
        "synthetic_replay": str(require_path(replay)),
        "policy_checkpoints": "|".join(str(path) for _, path in policy_paths),
        "policy_labels": "|".join(label for label, _ in policy_paths),
        "output_json": str(AWR_DIAG_ROOT / case / "diagnostics.json"),
        "output_md": str(AWR_DIAG_ROOT / case / "diagnostics.md"),
        "split": "train",
        "quality_filter": "expert,expert_noisy",
        "num_real": "8192",
        "num_synthetic": "4096",
        "adv_temperature": "1.0",
        "weight_clip": "20.0",
        "seed": str(bc_seed),
        "notes": notes,
    }


def build_awr_diagnostics_manifest() -> Path:
    rows: list[dict[str, str]] = []
    rows.append(
        diagnostic_row(
            "strongest_exact_h1_seed0",
            0,
            EXACT_H1_REPLAY,
            [
                ("strongest_final", STRONGEST_AWR / "final_policy_extraction.pt"),
                ("strongest_best_real_eval", STRONGEST_AWR / "best_policy_extraction.pt"),
            ],
            "Action drift and BC-error diagnostics for the historical strongest exact-H1 AWR run.",
        )
    )
    configs = ["r224_s32_anchor1_iter500", "r192_s64_anchor1_iter500", "r248_s8_anchor1_iter500"]
    for seed in [0, 1, 2]:
        for cfg in configs:
            base = FIX1_ROOT / f"endpoint_h1_{cfg}_s{seed}"
            rows.append(
                diagnostic_row(
                    f"fix1_{cfg}_s{seed}",
                    seed,
                    TRUNCATE_H1_REPLAY,
                    [
                        ("iter250", base / "real_eval_snapshots" / "iter_000250_policy_extraction.pt"),
                        ("iter500", base / "real_eval_snapshots" / "iter_000500_policy_extraction.pt"),
                        ("final", base / "final_policy_extraction.pt"),
                        ("best_real_eval", base / "best_policy_extraction.pt"),
                    ],
                    "Action drift and BC-error diagnostics for completed fix1 H1 AWR run.",
                )
            )
    path = MANIFEST_DIR / "flow_mbpo_h1_awr_diagnostics_h100_20260604.csv"
    write_csv(path, rows)
    return path


def replay_quality_row(case: str, replay: Path, notes: str) -> dict[str, str]:
    return {
        "stage": "flow_mbpo_h1_replay_quality_20260604",
        "case": case,
        "synthetic_replay": str(require_path(replay)),
        "dataset": str(DATASET),
        "metadata": str(METADATA),
        "normalization": str(NORMALIZATION),
        "output_json": str(REPLAY_QUALITY_ROOT / case / "quality.json"),
        "output_md": str(REPLAY_QUALITY_ROOT / case / "quality.md"),
        "split": "train",
        "quality_filter": "expert,expert_noisy",
        "support_max_rows": "20000",
        "support_probe_rows": "4096",
        "distance_batch_size": "256",
        "state_weight": "1.0",
        "command_weight": "1.0",
        "action_weight": "1.0",
        "high_reward_quantile": "0.90",
        "high_distance_quantile": "0.90",
        "seed": "0",
        "notes": notes,
    }


def build_replay_quality_manifest() -> Path:
    rows = [
        replay_quality_row("exact_h1_strongest_replay", EXACT_H1_REPLAY, "Historical exact H1 replay used by the 60.87 final checkpoint."),
        replay_quality_row(
            "truncate_check_h1_replay",
            TRUNCATE_H1_REPLAY,
            "Post-truncation-check H1 replay used by the later fix1 sweep.",
        ),
    ]
    path = MANIFEST_DIR / "flow_mbpo_h1_replay_quality_l40s_20260604.csv"
    write_csv(path, rows)
    return path


def main() -> None:
    for path in [DATASET, METADATA, NORMALIZATION, EXACT_H1_REPLAY, TRUNCATE_H1_REPLAY]:
        require_path(path)
    build_checkpoint_eval_manifest()
    build_awr_ratio_manifest()
    build_awr_diagnostics_manifest()
    build_replay_quality_manifest()


if __name__ == "__main__":
    main()
