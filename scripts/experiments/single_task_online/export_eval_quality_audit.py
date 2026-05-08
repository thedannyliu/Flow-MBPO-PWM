#!/usr/bin/env python3
"""Export single-task-online eval rows with trust/issue annotations."""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any


FAIR_ALIGNED_STAGES = {
    "confirm_fair_small",
    "confirm_fair_small_mjlab3",
    "confirm_fair_small_mjlab_extra3",
    "confirm_fair_small_mjlab6_strict",
    "confirm_mjlab_curriculum4_strict_default",
    "confirm_fair_small_tracking_rough_strict",
    "confirm_tracking_rough_strict",
    "confirm_fair_small_mjlab6_strict_h100shadow",
    "confirm_mjlab_curriculum4_strict_default_h100shadow",
    "confirm_tracking_rough_strict_h100shadow",
}

STRICT_EXPLORATORY_STAGES = {
    "flowwm_mjlab3_tracking_rough_strict",
    "flowwm_mjlab3_tracking_rough_strict_h100shadow",
}

EXPLORATORY_STAGES = {
    "flow_ablation",
    "flow_ablation_deep",
    "flow_rms_align",
    "flowwm_hparam_explore",
    "flowwm_hsched_bugfix_must",
    "flowwm_hsched_bugfix_recommended",
    "flowwm_mjlab_rescue_focus",
    "flowwm_mjlab_rescue_focus_h100shadow",
    "flowwm_mjlab3_fair_sweep",
    "flow_anchor",
}

HSCHED_PROFILES = {
    "flow_hsched_fast",
    "flow_hsched_slow",
}

OUTPUT_COLUMNS = [
    "run_key",
    "source_manifests",
    "stage",
    "suite",
    "task_key",
    "env",
    "complexity",
    "episode_length",
    "method_key",
    "method_description",
    "alg",
    "seed",
    "hparam_profile",
    "overrides",
    "notes",
    "output_dir",
    "requested_task_id",
    "resolved_task_id",
    "return_mean",
    "return_iqm",
    "discounted_return_mean",
    "episode_length_mean",
    "episode_length_std",
    "success_rate",
    "has_eval_summary",
    "has_episode_metrics_csv",
    "has_rollout_gif",
    "has_rollout_mp4",
    "stage_alignment_status",
    "baseline_alignment_status",
    "task_resolution_status",
    "known_bug_status",
    "metric_caveat_status",
    "trust_tier",
    "issue_flags",
    "issue_count",
    "use_for_final_fair_comparison",
    "use_for_exploratory_analysis",
    "issue_summary",
]


@dataclass
class EvalRecord:
    row: dict[str, str]
    source_manifests: list[str]
    output_dir: Path
    metrics: dict[str, Any]
    has_eval_summary: bool
    has_episode_metrics_csv: bool
    has_rollout_gif: bool
    has_rollout_mp4: bool


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest-dir",
        type=Path,
        default=Path("scripts/experiments/single_task_online/manifests"),
    )
    parser.add_argument(
        "--outputs-root",
        type=Path,
        default=Path("scripts/outputs/single_task_online"),
    )
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--output-legend-md", type=Path, required=True)
    return parser.parse_args()


def parse_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        val = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(val) or math.isinf(val):
        return None
    return val


def load_manifest_rows(manifest_dir: Path) -> tuple[dict[str, dict[str, str]], dict[str, list[str]]]:
    rows_by_key: dict[str, dict[str, str]] = {}
    sources_by_key: dict[str, list[str]] = {}
    for manifest in sorted(manifest_dir.glob("*.csv")):
        with manifest.open(newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                run_key = row.get("run_key", "")
                if not run_key:
                    continue
                if run_key not in rows_by_key:
                    rows_by_key[run_key] = row
                    sources_by_key[run_key] = [manifest.name]
                else:
                    sources_by_key[run_key].append(manifest.name)
    return rows_by_key, sources_by_key


def build_records(
    rows_by_key: dict[str, dict[str, str]],
    sources_by_key: dict[str, list[str]],
    outputs_root: Path,
) -> list[EvalRecord]:
    records: list[EvalRecord] = []
    for run_key, row in rows_by_key.items():
        output_dir = (
            outputs_root
            / row["stage"]
            / row["suite"]
            / row["task_key"]
            / row["method_key"]
            / f"seed_{row['seed']}"
            / row["hparam_profile"]
        )
        eval_dir = output_dir / "eval"
        summary_path = eval_dir / "eval_summary.json"
        has_eval_summary = summary_path.exists()
        if not has_eval_summary:
            continue
        metrics: dict[str, Any] = {}
        with summary_path.open(encoding="utf-8") as f:
            metrics = json.load(f)
        records.append(
            EvalRecord(
                row=row,
                source_manifests=sources_by_key.get(run_key, []),
                output_dir=output_dir,
                metrics=metrics,
                has_eval_summary=True,
                has_episode_metrics_csv=(eval_dir / "episode_metrics.csv").exists(),
                has_rollout_gif=(eval_dir / "rollout.gif").exists(),
                has_rollout_mp4=(eval_dir / "rollout.mp4").exists(),
            )
        )
    records.sort(
        key=lambda r: (
            r.row.get("stage", ""),
            r.row.get("suite", ""),
            r.row.get("task_key", ""),
            r.row.get("method_key", ""),
            r.row.get("hparam_profile", ""),
            int(r.row.get("seed", "0")),
        )
    )
    return records


def stage_alignment_status(rec: EvalRecord) -> str:
    stage = rec.row["stage"]
    if stage in FAIR_ALIGNED_STAGES:
        return "aligned_fair_stage"
    if stage in STRICT_EXPLORATORY_STAGES:
        return "strict_task_fixed_exploratory_stage"
    if stage in EXPLORATORY_STAGES:
        return "exploratory_stage"
    return "historical_stage"


def baseline_alignment_status(rec: EvalRecord) -> str:
    method_key = rec.row["method_key"]
    alg = rec.row.get("alg", "")
    if method_key != "mlpwm_mlppolicy":
        return "not_applicable"
    if alg == "pwm_5M_baseline_pwmorig":
        return "aligned_pwmorig"
    if alg == "pwm_5M_baseline_final":
        return "non_aligned_baseline_final"
    if alg == "pwm_5M":
        return "ambiguous_local_pwm5M"
    return f"other_{alg or 'unknown'}"


def task_resolution_status(rec: EvalRecord) -> str:
    requested = str(rec.metrics.get("requested_task_id", "")).strip()
    resolved = str(rec.metrics.get("resolved_task_id", "")).strip()
    task_key = rec.row["task_key"]
    stage = rec.row["stage"]
    if requested and resolved:
        return "exact_match" if requested == resolved else "fallback_mismatch"
    if task_key == "tracking_rough_unitree_g1" and "strict" not in stage:
        return "known_historical_tracking_rough_contaminated"
    return "not_tracked"


def known_bug_status(rec: EvalRecord) -> str:
    stage = rec.row["stage"]
    profile = rec.row.get("hparam_profile", "")
    if stage == "flowwm_hparam_explore" and profile in HSCHED_PROFILES:
        return "pre_fix_hsched_path"
    if stage in {"flowwm_hsched_bugfix_must", "flowwm_hsched_bugfix_recommended"}:
        return "hsched_bugfix_rerun"
    return "none"


def metric_caveat_status(rec: EvalRecord) -> str:
    ep_cap = parse_float(rec.row.get("episode_length"))
    ep_mean = parse_float(rec.metrics.get("episode_length_mean"))
    ep_std = parse_float(rec.metrics.get("episode_length_std"))
    if ep_cap is None or ep_mean is None:
        return "none"
    if abs(ep_mean - ep_cap) <= 1e-6 and (ep_std is None or abs(ep_std) <= 1e-6):
        return "episode_length_saturated_at_cap"
    return "none"


def annotate(rec: EvalRecord) -> dict[str, Any]:
    stage_status = stage_alignment_status(rec)
    baseline_status = baseline_alignment_status(rec)
    resolution_status = task_resolution_status(rec)
    bug_status = known_bug_status(rec)
    caveat_status = metric_caveat_status(rec)

    flags: list[str] = []

    if baseline_status in {"non_aligned_baseline_final", "ambiguous_local_pwm5M"}:
        flags.append("baseline_not_original_pwm_aligned")
    if stage_status == "historical_stage":
        flags.append("historical_stage_not_for_final_fair_table")
    if resolution_status in {"fallback_mismatch", "known_historical_tracking_rough_contaminated"}:
        flags.append("task_resolution_contaminated")
    if bug_status == "pre_fix_hsched_path":
        flags.append("pre_fix_hsched_bug_path")
    if caveat_status == "episode_length_saturated_at_cap":
        flags.append("episode_length_metric_saturated")
    if not rec.has_rollout_gif and not rec.has_rollout_mp4:
        flags.append("no_rollout_video_artifact")

    if "task_resolution_contaminated" in flags:
        trust_tier = "contaminated_do_not_use"
        use_final = False
        use_exploratory = False
    elif "pre_fix_hsched_bug_path" in flags:
        trust_tier = "provisional_bug_path"
        use_final = False
        use_exploratory = True
    elif stage_status == "aligned_fair_stage":
        if caveat_status == "episode_length_saturated_at_cap":
            trust_tier = "final_fair_use_with_metric_caveat"
        else:
            trust_tier = "final_fair_use"
        use_final = True
        use_exploratory = True
    elif stage_status in {"strict_task_fixed_exploratory_stage", "exploratory_stage"}:
        trust_tier = "exploratory_only"
        use_final = False
        use_exploratory = True
    else:
        trust_tier = "historical_non_aligned"
        use_final = False
        use_exploratory = True

    if trust_tier == "historical_non_aligned" and "baseline_not_original_pwm_aligned" not in flags:
        flags.append("comparison_context_not_original_pwm_aligned")

    summary_parts: list[str] = []
    if trust_tier == "final_fair_use":
        summary_parts.append("Aligned fair-comparison row.")
    elif trust_tier == "final_fair_use_with_metric_caveat":
        summary_parts.append("Aligned fair-comparison row, but episode_length is saturated at the task cap.")
    elif trust_tier == "exploratory_only":
        summary_parts.append("Useful for exploratory analysis only, not for the final fair-comparison table.")
    elif trust_tier == "historical_non_aligned":
        summary_parts.append("Historical row; do not use as original-PWM-aligned evidence.")
    elif trust_tier == "provisional_bug_path":
        summary_parts.append("Completed on a known pre-fix hsched code path; keep as provisional only.")
    elif trust_tier == "contaminated_do_not_use":
        summary_parts.append("Contaminated by task-resolution mismatch; exclude from final analysis until rerun.")

    if baseline_status == "non_aligned_baseline_final":
        summary_parts.append("MLP baseline used pwm_5M_baseline_final instead of pwm_5M_baseline_pwmorig.")
    elif baseline_status == "ambiguous_local_pwm5M":
        summary_parts.append("MLP baseline used repo-local pwm_5M, which was historically drifted from original PWM.")

    if resolution_status == "fallback_mismatch":
        summary_parts.append("Eval summary records requested/resolved MJLab task mismatch.")
    elif resolution_status == "known_historical_tracking_rough_contaminated":
        summary_parts.append("Historical tracking_rough rows are flagged contaminated due to pre-fix silent fallback behavior.")

    if bug_status == "pre_fix_hsched_path":
        summary_parts.append("This row belongs to the hsched sweep before the replay-sampling bugfix.")

    if caveat_status == "episode_length_saturated_at_cap":
        summary_parts.append("episode_length_mean is not discriminative for this row because all eval episodes hit the configured cap.")

    return {
        "stage_alignment_status": stage_status,
        "baseline_alignment_status": baseline_status,
        "task_resolution_status": resolution_status,
        "known_bug_status": bug_status,
        "metric_caveat_status": caveat_status,
        "trust_tier": trust_tier,
        "issue_flags": ";".join(flags),
        "issue_count": len(flags),
        "use_for_final_fair_comparison": use_final,
        "use_for_exploratory_analysis": use_exploratory,
        "issue_summary": " ".join(summary_parts),
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_legend(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = """# Eval Quality Audit Legend

## trust_tier

- `final_fair_use`: safe to use in the final original-PWM-aligned comparison table
- `final_fair_use_with_metric_caveat`: safe for final comparison, but one metric is saturated and should be interpreted carefully
- `exploratory_only`: valid as an exploratory/ablation result, not for the final fair-comparison table
- `historical_non_aligned`: historical result from a non-aligned comparison context
- `provisional_bug_path`: result came from a known bug-affected code path and should be treated as provisional
- `contaminated_do_not_use`: do not use until rerun

## main issue flags

- `baseline_not_original_pwm_aligned`: MLP baseline config did not use `pwm_5M_baseline_pwmorig`
- `comparison_context_not_original_pwm_aligned`: row belongs to a historical stage that should not be treated as a final fair-comparison table
- `task_resolution_contaminated`: requested and resolved MJLab task do not match, or the row is a known pre-fix contaminated `tracking_rough` run
- `pre_fix_hsched_bug_path`: row belongs to the hsched sweep before the replay-sampling bugfix
- `episode_length_metric_saturated`: `episode_length_mean` hit the configured task cap and is not discriminative
- `no_rollout_video_artifact`: no saved gif/mp4 artifact was found
"""
    path.write_text(text, encoding="utf-8")


def main() -> None:
    args = parse_args()
    rows_by_key, sources_by_key = load_manifest_rows(args.manifest_dir)
    records = build_records(rows_by_key, sources_by_key, args.outputs_root)

    output_rows: list[dict[str, Any]] = []
    for rec in records:
        anno = annotate(rec)
        output_rows.append(
            {
                "run_key": rec.row["run_key"],
                "source_manifests": ";".join(rec.source_manifests),
                "stage": rec.row.get("stage", ""),
                "suite": rec.row.get("suite", ""),
                "task_key": rec.row.get("task_key", ""),
                "env": rec.row.get("env", ""),
                "complexity": rec.row.get("complexity", ""),
                "episode_length": rec.row.get("episode_length", ""),
                "method_key": rec.row.get("method_key", ""),
                "method_description": rec.row.get("method_description", ""),
                "alg": rec.row.get("alg", ""),
                "seed": rec.row.get("seed", ""),
                "hparam_profile": rec.row.get("hparam_profile", ""),
                "overrides": rec.row.get("overrides", ""),
                "notes": rec.row.get("notes", ""),
                "output_dir": str(rec.output_dir.resolve()),
                "requested_task_id": rec.metrics.get("requested_task_id", ""),
                "resolved_task_id": rec.metrics.get("resolved_task_id", ""),
                "return_mean": rec.metrics.get("return_mean", ""),
                "return_iqm": rec.metrics.get("return_iqm", ""),
                "discounted_return_mean": rec.metrics.get("discounted_return_mean", ""),
                "episode_length_mean": rec.metrics.get("episode_length_mean", ""),
                "episode_length_std": rec.metrics.get("episode_length_std", ""),
                "success_rate": rec.metrics.get("success_rate", ""),
                "has_eval_summary": rec.has_eval_summary,
                "has_episode_metrics_csv": rec.has_episode_metrics_csv,
                "has_rollout_gif": rec.has_rollout_gif,
                "has_rollout_mp4": rec.has_rollout_mp4,
                **anno,
            }
        )

    write_csv(args.output_csv, output_rows)
    write_legend(args.output_legend_md)
    print(f"exported_rows={len(output_rows)}")
    print(f"output_csv={args.output_csv}")
    print(f"output_legend_md={args.output_legend_md}")


if __name__ == "__main__":
    main()
