#!/usr/bin/env python3
"""Build follow-up eval manifests for Flow-MBPO H1 evidence gaps."""

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
STRONGEST_AWR = ROOT / "flow_mbpo_v0_awr" / "flow_endpoint_seed0_h1_unc0p5_q0p90_cons_r224_s32_anchor1_iter500_s0"
FIX1_ROOT = ROOT / "flow_mbpo_endpoint_h1_multiseed_ablation_fix1_20260604" / "h200"
EXACT_RATIO_ROOT = ROOT / "flow_mbpo_h1_exact_replay_realonly_ratio_20260604" / "h200"

ROBUST_EVAL_ROOT = ROOT / "flow_mbpo_h1_strongest_robust_eval_20260604" / "h100"
RATIO_EVAL_ROOT = ROOT / "flow_mbpo_h1_exact_ratio_checkpoint_eval_20260604" / "h100"

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


def eval_row(
    *,
    stage: str,
    candidate: str,
    checkpoint: Path,
    checkpoint_kind: str,
    output_dir: Path,
    seed: int,
    notes: str,
) -> dict[str, str]:
    require_path(checkpoint)
    return {
        "stage": stage,
        "task_key": TASK_KEY,
        "task_id": TASK_ID,
        "candidate": candidate,
        "policy_checkpoint": str(checkpoint),
        "checkpoint_kind": checkpoint_kind,
        "eval_output_dir": str(output_dir),
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
        "wandb_project": "flow-mbpo-mjlab-h1-followup-eval-20260604",
        "wandb_group": stage,
        "wandb_name": f"{candidate}_evalseed{seed}_eval40",
        "notes": notes,
    }


def build_strongest_robust_eval() -> Path:
    stage = "flow_mbpo_h1_strongest_robust_eval_20260604"
    candidates = [
        (
            "bc_seed0_initial",
            bc_checkpoint(0),
            "bc_initial",
            "BC seed0 baseline repeated direct eval for strongest-H1 robustness comparison.",
        ),
        (
            "strongest_exact_h1_final",
            STRONGEST_AWR / "final_policy_extraction.pt",
            "final",
            "Historical strongest exact-H1 final checkpoint repeated direct eval.",
        ),
        (
            "strongest_exact_h1_best_real_eval",
            STRONGEST_AWR / "best_policy_extraction.pt",
            "best_real_eval",
            "Historical strongest exact-H1 best-real-eval checkpoint repeated direct eval.",
        ),
        (
            "fix1_r224_s32_s0_iter500",
            FIX1_ROOT / "endpoint_h1_r224_s32_anchor1_iter500_s0" / "real_eval_snapshots" / "iter_000500_policy_extraction.pt",
            "iter500",
            "Fix1 r224/s32 seed0 iter500 checkpoint repeated direct eval.",
        ),
    ]
    rows: list[dict[str, str]] = []
    for seed in range(5):
        for candidate, checkpoint, kind, notes in candidates:
            rows.append(
                eval_row(
                    stage=stage,
                    candidate=f"{candidate}_evalseed{seed}",
                    checkpoint=checkpoint,
                    checkpoint_kind=kind,
                    output_dir=ROBUST_EVAL_ROOT / f"{candidate}_evalseed{seed}",
                    seed=seed,
                    notes=notes,
                )
            )
    path = MANIFEST_DIR / "flow_mbpo_h1_strongest_robust_eval_h100_20260604.csv"
    write_csv(path, rows)
    return path


def exact_ratio_checkpoint_specs(run_dir: Path) -> list[tuple[str, Path]]:
    return [
        ("iter250", run_dir / "real_eval_snapshots" / "iter_000250_policy_extraction.pt"),
        ("iter500", run_dir / "real_eval_snapshots" / "iter_000500_policy_extraction.pt"),
        ("final", run_dir / "final_policy_extraction.pt"),
        ("best_real_eval", run_dir / "best_policy_extraction.pt"),
    ]


def build_exact_ratio_checkpoint_eval() -> Path:
    stage = "flow_mbpo_h1_exact_ratio_checkpoint_eval_20260604"
    rows: list[dict[str, str]] = []
    for run_dir in sorted(EXACT_RATIO_ROOT.glob("exact_h1_r*_anchor1_iter500_seed*")):
        if not run_dir.is_dir():
            continue
        label = run_dir.name
        seed = int(label.rsplit("seed", 1)[1])
        for kind, checkpoint in exact_ratio_checkpoint_specs(run_dir):
            rows.append(
                eval_row(
                    stage=stage,
                    candidate=f"{label}_{kind}",
                    checkpoint=checkpoint,
                    checkpoint_kind=kind,
                    output_dir=RATIO_EVAL_ROOT / f"{label}_{kind}",
                    seed=seed,
                    notes="Formal eval40 for exact-replay real/synthetic-ratio AWR checkpoint.",
                )
            )
    path = MANIFEST_DIR / "flow_mbpo_h1_exact_ratio_checkpoint_eval_h100_20260604.csv"
    write_csv(path, rows)
    return path


def main() -> None:
    for path in [DATASET, METADATA, NORMALIZATION]:
        require_path(path)
    build_strongest_robust_eval()
    build_exact_ratio_checkpoint_eval()


if __name__ == "__main__":
    main()
