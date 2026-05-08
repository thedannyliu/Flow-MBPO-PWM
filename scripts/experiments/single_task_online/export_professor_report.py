#!/usr/bin/env python3
"""Export professor-facing single-task-online evaluation reports.

Outputs:
1) Full eval CSV (one row per evaluated run).
2) Per-task high-level CSV (mean/std + seed coverage + hparam combo).
3) English insights markdown.
4) English unfinished-experiments + parameter-design markdown.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

INCLUDE_STAGES = {
    "confirm",
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
    "flow_ablation",
    "flow_ablation_deep",
    "flow_anchor",
    "flow_rms_align",
    "flowwm_hsched_bugfix_must",
    "flowwm_hsched_bugfix_recommended",
    "flowwm_mjlab_rescue_focus",
    "flowwm_mjlab_rescue_focus_h100shadow",
    "flowwm_mjlab3_fair_sweep",
    "flowwm_mjlab3_tracking_rough_strict_h100shadow",
    "flowwm_hparam_explore",
}

METRIC_KEYS = [
    "return_mean",
    "return_std",
    "return_iqm",
    "return_median",
    "discounted_return_mean",
    "episode_length_mean",
    "episode_length_std",
    "success_rate",
    "return_min",
    "return_max",
    "return_p25",
    "return_p75",
    "num_episodes",
    "num_games_requested",
    "num_envs_eval",
]

FULL_EVAL_COLUMNS = [
    "run_key",
    "source_manifest",
    "stage",
    "suite",
    "task_key",
    "env",
    "complexity",
    "episode_length",
    "method_key",
    "method_description",
    "wm_family",
    "policy_family",
    "alg",
    "seed",
    "hparam_profile",
    "max_epochs",
    "num_envs",
    "eval_runs",
    "rollout_episodes",
    "rollout_max_steps",
    "wandb_project",
    "wandb_group",
    "notes",
    "overrides",
    "output_dir",
    "has_final_policy",
    "has_best_policy",
    "has_eval_summary",
    "has_episode_metrics_csv",
    "has_rollout_steps_csv",
    "has_rollout_summary_csv",
    "has_rollout_gif",
    "has_rollout_mp4",
    "status",
    "checkpoint",
    "env_target",
] + METRIC_KEYS

TASK_SUMMARY_COLUMNS = [
    "stage",
    "suite",
    "task_key",
    "method_key",
    "method_description",
    "wm_family",
    "policy_family",
    "hparam_profile",
    "alg",
    "overrides",
    "notes",
    "n_planned",
    "n_train_completed",
    "n_eval_completed",
    "eval_completion_rate",
    "seed_count_planned",
    "seed_count_eval",
    "seeds_planned",
    "seeds_eval",
    "return_mean_avg",
    "return_mean_std",
    "return_iqm_avg",
    "return_iqm_std",
    "discounted_return_mean_avg",
    "discounted_return_mean_std",
    "episode_length_mean_avg",
    "episode_length_mean_std",
    "success_rate_avg",
    "success_rate_std",
]


@dataclass
class RunRecord:
    manifest_row: dict[str, str]
    source_manifest: str
    output_dir: Path
    has_final_policy: bool
    has_best_policy: bool
    has_eval_summary: bool
    has_episode_metrics_csv: bool
    has_rollout_steps_csv: bool
    has_rollout_summary_csv: bool
    has_rollout_gif: bool
    has_rollout_mp4: bool
    metrics: dict[str, Any]


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
    parser.add_argument("--output-full-csv", type=Path, required=True)
    parser.add_argument("--output-task-csv", type=Path, required=True)
    parser.add_argument("--output-insights-md", type=Path, required=True)
    parser.add_argument("--output-unfinished-md", type=Path, required=True)
    return parser.parse_args()


def parse_float(x: Any) -> float | None:
    if x is None:
        return None
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    if math.isnan(v) or math.isinf(v):
        return None
    return v


def mean_std(values: list[float]) -> tuple[float | None, float | None]:
    if not values:
        return None, None
    if len(values) == 1:
        return values[0], 0.0
    return statistics.fmean(values), statistics.stdev(values)


def fmt(v: float | None, digits: int = 3) -> str:
    if v is None:
        return "nan"
    return f"{v:.{digits}f}"


def method_family(method_key: str) -> tuple[str, str]:
    wm_family = "flowwm" if method_key.startswith("flowwm_") else "mlpwm"
    policy_family = "flowpolicy" if method_key.endswith("_flowpolicy") else "mlppolicy"
    return wm_family, policy_family


def status_from_flags(has_final: bool, has_eval: bool) -> str:
    if has_final and has_eval:
        return "completed_eval"
    if has_final and not has_eval:
        return "train_done_no_eval"
    if (not has_final) and has_eval:
        return "eval_no_final"
    return "pending_or_failed"


def load_manifest_rows(manifest_dir: Path) -> tuple[dict[str, dict[str, str]], dict[str, list[str]]]:
    rows_by_key: dict[str, dict[str, str]] = {}
    sources_by_key: dict[str, list[str]] = defaultdict(list)
    for manifest in sorted(manifest_dir.glob("*.csv")):
        with manifest.open(newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row.get("stage") not in INCLUDE_STAGES:
                    continue
                run_key = row["run_key"]
                sources_by_key[run_key].append(manifest.name)
                if run_key not in rows_by_key:
                    rows_by_key[run_key] = row
    return rows_by_key, sources_by_key


def enrich_record(row: dict[str, str], source_manifest: str, outputs_root: Path) -> RunRecord:
    output_dir = (
        outputs_root
        / row["stage"]
        / row["suite"]
        / row["task_key"]
        / row["method_key"]
        / f"seed_{row['seed']}"
        / row["hparam_profile"]
    )
    logs_dir = output_dir / "logs"
    eval_dir = output_dir / "eval"

    has_final_policy = (logs_dir / "final_policy.pt").exists()
    has_best_policy = (logs_dir / "best_policy.pt").exists()
    has_eval_summary = (eval_dir / "eval_summary.json").exists()
    has_episode_metrics_csv = (eval_dir / "episode_metrics.csv").exists()
    has_rollout_steps_csv = (eval_dir / "rollout_steps.csv").exists()
    has_rollout_summary_csv = (eval_dir / "rollout_summary.csv").exists()
    has_rollout_gif = (eval_dir / "rollout.gif").exists()
    has_rollout_mp4 = (eval_dir / "rollout.mp4").exists()

    metrics: dict[str, Any] = {}
    if has_eval_summary:
        with (eval_dir / "eval_summary.json").open() as f:
            metrics = json.load(f)

    return RunRecord(
        manifest_row=row,
        source_manifest=source_manifest,
        output_dir=output_dir,
        has_final_policy=has_final_policy,
        has_best_policy=has_best_policy,
        has_eval_summary=has_eval_summary,
        has_episode_metrics_csv=has_episode_metrics_csv,
        has_rollout_steps_csv=has_rollout_steps_csv,
        has_rollout_summary_csv=has_rollout_summary_csv,
        has_rollout_gif=has_rollout_gif,
        has_rollout_mp4=has_rollout_mp4,
        metrics=metrics,
    )


def build_full_eval_rows(records: list[RunRecord]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for rec in records:
        if not rec.has_eval_summary:
            continue
        row = dict(rec.manifest_row)
        wm_family, policy_family = method_family(row["method_key"])
        status = status_from_flags(rec.has_final_policy, rec.has_eval_summary)
        out: dict[str, Any] = {
            "run_key": row["run_key"],
            "source_manifest": rec.source_manifest,
            "stage": row["stage"],
            "suite": row["suite"],
            "task_key": row["task_key"],
            "env": row.get("env", ""),
            "complexity": row.get("complexity", ""),
            "episode_length": row.get("episode_length", ""),
            "method_key": row["method_key"],
            "method_description": row.get("method_description", ""),
            "wm_family": wm_family,
            "policy_family": policy_family,
            "alg": row.get("alg", ""),
            "seed": row.get("seed", ""),
            "hparam_profile": row.get("hparam_profile", ""),
            "max_epochs": row.get("max_epochs", ""),
            "num_envs": row.get("num_envs", ""),
            "eval_runs": row.get("eval_runs", ""),
            "rollout_episodes": row.get("rollout_episodes", ""),
            "rollout_max_steps": row.get("rollout_max_steps", ""),
            "wandb_project": row.get("wandb_project", ""),
            "wandb_group": row.get("wandb_group", ""),
            "notes": row.get("notes", ""),
            "overrides": row.get("overrides", ""),
            "output_dir": str(rec.output_dir.resolve()),
            "has_final_policy": rec.has_final_policy,
            "has_best_policy": rec.has_best_policy,
            "has_eval_summary": rec.has_eval_summary,
            "has_episode_metrics_csv": rec.has_episode_metrics_csv,
            "has_rollout_steps_csv": rec.has_rollout_steps_csv,
            "has_rollout_summary_csv": rec.has_rollout_summary_csv,
            "has_rollout_gif": rec.has_rollout_gif,
            "has_rollout_mp4": rec.has_rollout_mp4,
            "status": status,
            "checkpoint": rec.metrics.get("checkpoint", ""),
            "env_target": rec.metrics.get("env_target", ""),
        }
        for k in METRIC_KEYS:
            out[k] = rec.metrics.get(k)
        rows.append(out)
    rows.sort(key=lambda r: (r["stage"], r["suite"], r["task_key"], r["method_key"], r["hparam_profile"], int(r["seed"])))
    return rows


def build_task_summary(records: list[RunRecord]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, str, str, str], list[RunRecord]] = defaultdict(list)
    for rec in records:
        row = rec.manifest_row
        key = (row["stage"], row["suite"], row["task_key"], row["method_key"], row["hparam_profile"])
        groups[key].append(rec)

    summary_rows: list[dict[str, Any]] = []
    for key, recs in groups.items():
        stage, suite, task_key, method_key, hparam_profile = key
        template = recs[0].manifest_row
        wm_family, policy_family = method_family(method_key)

        seeds_planned = sorted({int(r.manifest_row["seed"]) for r in recs})
        eval_recs = [r for r in recs if r.has_eval_summary]
        train_recs = [r for r in recs if r.has_final_policy]
        seeds_eval = sorted({int(r.manifest_row["seed"]) for r in eval_recs})

        def collect(metric: str) -> list[float]:
            vals = []
            for r in eval_recs:
                v = parse_float(r.metrics.get(metric))
                if v is not None:
                    vals.append(v)
            return vals

        rm_mean, rm_std = mean_std(collect("return_mean"))
        iqm_mean, iqm_std = mean_std(collect("return_iqm"))
        dr_mean, dr_std = mean_std(collect("discounted_return_mean"))
        ep_mean, ep_std = mean_std(collect("episode_length_mean"))
        sr_mean, sr_std = mean_std(collect("success_rate"))

        summary_rows.append(
            {
                "stage": stage,
                "suite": suite,
                "task_key": task_key,
                "method_key": method_key,
                "method_description": template.get("method_description", ""),
                "wm_family": wm_family,
                "policy_family": policy_family,
                "hparam_profile": hparam_profile,
                "alg": template.get("alg", ""),
                "overrides": template.get("overrides", ""),
                "notes": template.get("notes", ""),
                "n_planned": len(recs),
                "n_train_completed": len(train_recs),
                "n_eval_completed": len(eval_recs),
                "eval_completion_rate": len(eval_recs) / len(recs) if recs else 0.0,
                "seed_count_planned": len(seeds_planned),
                "seed_count_eval": len(seeds_eval),
                "seeds_planned": ",".join(str(s) for s in seeds_planned),
                "seeds_eval": ",".join(str(s) for s in seeds_eval),
                "return_mean_avg": rm_mean,
                "return_mean_std": rm_std,
                "return_iqm_avg": iqm_mean,
                "return_iqm_std": iqm_std,
                "discounted_return_mean_avg": dr_mean,
                "discounted_return_mean_std": dr_std,
                "episode_length_mean_avg": ep_mean,
                "episode_length_mean_std": ep_std,
                "success_rate_avg": sr_mean,
                "success_rate_std": sr_std,
            }
        )

    summary_rows.sort(key=lambda r: (r["stage"], r["suite"], r["task_key"], r["method_key"], r["hparam_profile"]))
    return summary_rows


def write_csv(path: Path, rows: list[dict[str, Any]], columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def render_stage_coverage(records: list[RunRecord]) -> list[str]:
    stage_counts: dict[str, Counter[str]] = defaultdict(Counter)
    for rec in records:
        stage = rec.manifest_row["stage"]
        st = status_from_flags(rec.has_final_policy, rec.has_eval_summary)
        stage_counts[stage][st] += 1

    lines = [
        "| Stage | Planned | Completed (train+eval) | Train-only | Pending/Failed |",
        "|---|---:|---:|---:|---:|",
    ]
    for stage in sorted(stage_counts):
        c = stage_counts[stage]
        planned = sum(c.values())
        lines.append(
            f"| {stage} | {planned} | {c['completed_eval']} | {c['train_done_no_eval']} | {c['pending_or_failed'] + c['eval_no_final']} |"
        )
    return lines


def render_confirm_method_table(records: list[RunRecord]) -> list[str]:
    # Directly comparable setting: confirm stage, default profile, same budget.
    groups: dict[tuple[str, str], list[float]] = defaultdict(list)
    n_by_group: Counter[tuple[str, str]] = Counter()

    for rec in records:
        row = rec.manifest_row
        if row["stage"] != "confirm":
            continue
        if not rec.has_eval_summary:
            continue
        metric = parse_float(rec.metrics.get("return_mean"))
        if metric is None:
            continue
        key = (row["task_key"], row["method_key"])
        groups[key].append(metric)
        n_by_group[key] += 1

    lines = [
        "| Task | Method | Completed seeds | Return mean (mean +- std) |",
        "|---|---|---:|---:|",
    ]
    for task, method in sorted(groups):
        vals = groups[(task, method)]
        m, s = mean_std(vals)
        lines.append(f"| {task} | {method} | {n_by_group[(task, method)]} | {fmt(m)} +- {fmt(s)} |")
    return lines


def render_hopper_advanced_table(records: list[RunRecord]) -> list[str]:
    # flowwm_hparam_explore currently mostly hopper. Show early profile signal.
    groups: dict[tuple[str, str], list[float]] = defaultdict(list)
    for rec in records:
        row = rec.manifest_row
        if row["stage"] != "flowwm_hparam_explore":
            continue
        if row["task_key"] != "hopper":
            continue
        if not rec.has_eval_summary:
            continue
        rm = parse_float(rec.metrics.get("return_mean"))
        if rm is None:
            continue
        groups[(row["method_key"], row["hparam_profile"])].append(rm)

    ranked = []
    for (method, profile), vals in groups.items():
        m, s = mean_std(vals)
        ranked.append((method, profile, len(vals), m, s))
    ranked.sort(key=lambda x: (x[0], -(x[3] if x[3] is not None else -1e18)))

    lines = [
        "| Method | Profile | Completed seeds | Return mean (mean +- std) |",
        "|---|---|---:|---:|",
    ]
    for method, profile, n, m, s in ranked:
        lines.append(f"| {method} | {profile} | {n} | {fmt(m)} +- {fmt(s)} |")
    return lines


def summarize_confirm_gaps(records: list[RunRecord]) -> dict[str, list[str]]:
    per_task: dict[str, dict[str, float]] = defaultdict(dict)
    for rec in records:
        row = rec.manifest_row
        if row["stage"] != "confirm" or row.get("hparam_profile") != "default":
            continue
        if not rec.has_eval_summary:
            continue
        rm = parse_float(rec.metrics.get("return_mean"))
        if rm is None:
            continue
        per_task[row["task_key"]][row["method_key"]] = rm

    mlp_better: list[str] = []
    flow_better: list[str] = []
    near_parity: list[str] = []
    incomplete: list[str] = []

    for task, vals in sorted(per_task.items()):
        required = {
            "flowwm_flowpolicy",
            "flowwm_mlppolicy",
            "mlpwm_flowpolicy",
            "mlpwm_mlppolicy",
        }
        if not required.issubset(vals):
            incomplete.append(task)
            continue
        flow_mean = (vals["flowwm_flowpolicy"] + vals["flowwm_mlppolicy"]) / 2.0
        mlp_mean = (vals["mlpwm_flowpolicy"] + vals["mlpwm_mlppolicy"]) / 2.0
        gap = flow_mean - mlp_mean
        scale = max(abs(flow_mean), abs(mlp_mean), 1.0)
        rel_gap = abs(gap) / scale
        if rel_gap <= 0.10:
            near_parity.append(task)
        elif gap > 0:
            flow_better.append(task)
        else:
            mlp_better.append(task)

    return {
        "mlp_better": mlp_better,
        "flow_better": flow_better,
        "near_parity": near_parity,
        "incomplete": incomplete,
    }


def advanced_stage_summary(records: list[RunRecord]) -> dict[str, Any]:
    eval_recs = [
        r for r in records
        if r.manifest_row["stage"] == "flowwm_hparam_explore" and r.has_eval_summary
    ]
    task_counts = Counter(r.manifest_row["task_key"] for r in eval_recs)
    if not task_counts:
        return {
            "top_task": None,
            "top_count": 0,
            "total_eval": 0,
            "best_profiles": [],
        }

    top_task, top_count = task_counts.most_common(1)[0]
    task_recs = [r for r in eval_recs if r.manifest_row["task_key"] == top_task]
    per_method_profile: dict[tuple[str, str], list[float]] = defaultdict(list)
    for rec in task_recs:
        rm = parse_float(rec.metrics.get("return_mean"))
        if rm is None:
            continue
        key = (rec.manifest_row["method_key"], rec.manifest_row["hparam_profile"])
        per_method_profile[key].append(rm)

    best_profiles: list[tuple[str, str, float | None, float | None]] = []
    for method in sorted({m for m, _ in per_method_profile}):
        ranked: list[tuple[str, float | None, float | None]] = []
        for (m, profile), vals in per_method_profile.items():
            if m != method:
                continue
            mean_val, _ = mean_std(vals)
            ranked.append((profile, mean_val, len(vals)))
        ranked.sort(key=lambda item: -(item[1] if item[1] is not None else -1e18))
        if ranked:
            profile, mean_val, n = ranked[0]
            best_profiles.append((method, profile, mean_val, n))

    return {
        "top_task": top_task,
        "top_count": top_count,
        "total_eval": len(eval_recs),
        "best_profiles": best_profiles,
    }


def build_insights_md(
    records: list[RunRecord],
    full_rows: list[dict[str, Any]],
    task_rows: list[dict[str, Any]],
    sources_by_key: dict[str, list[str]],
    output_full_csv: Path,
    output_task_csv: Path,
    now_str: str,
) -> str:
    total_planned = len(records)
    status_counts = Counter(status_from_flags(r.has_final_policy, r.has_eval_summary) for r in records)

    eval_rows = [r for r in records if r.has_eval_summary]
    final_rows = [r for r in records if r.has_final_policy]
    gif_count = sum(1 for r in eval_rows if r.has_rollout_gif)
    mp4_count = sum(1 for r in eval_rows if r.has_rollout_mp4)

    method_perf: dict[str, list[float]] = defaultdict(list)
    for r in eval_rows:
        rm = parse_float(r.metrics.get("return_mean"))
        if rm is not None:
            method_perf[r.manifest_row["method_key"]].append(rm)

    duplicate_run_keys = sum(1 for manifests in sources_by_key.values() if len(manifests) > 1)
    confirm_gap_summary = summarize_confirm_gaps(records)
    advanced_summary = advanced_stage_summary(records)
    pending_counts = Counter(
        r.manifest_row["stage"]
        for r in records
        if status_from_flags(r.has_final_policy, r.has_eval_summary) != "completed_eval"
    )

    lines: list[str] = []
    lines.append("# Single-Task Online RL: Current Insights Snapshot")
    lines.append("")
    lines.append(f"Generated on: {now_str}")
    lines.append("")
    lines.append("## Data Scope")
    lines.append("")
    lines.append(f"- Included stages: {', '.join(sorted(INCLUDE_STAGES))}")
    lines.append(f"- Planned unique runs (deduplicated by run_key): {total_planned}")
    lines.append(f"- Evaluated runs (`eval_summary.json` present): {len(eval_rows)}")
    lines.append(
        f"- Completed train+eval runs: {status_counts['completed_eval']} / {total_planned} "
        f"({status_counts['completed_eval'] / total_planned:.1%})"
    )
    lines.append(f"- Duplicate run_key entries across retry/remaining manifests: {duplicate_run_keys}")
    lines.append(f"- Rollout artifacts among evaluated runs: GIF={gif_count}, MP4={mp4_count}")
    lines.append(f"- Full eval CSV: `{output_full_csv}`")
    lines.append(f"- Per-task high-level CSV: `{output_task_csv}`")
    lines.append("")
    lines.append("## Stage Coverage")
    lines.append("")
    lines.extend(render_stage_coverage(records))
    lines.append("")
    lines.append("## Key Performance Patterns")
    lines.append("")

    # Method-family aggregate signal (coarse, mixes tasks/scales; kept as trend only).
    for method in sorted(method_perf):
        m, s = mean_std(method_perf[method])
        lines.append(
            f"- Aggregate `return_mean` for `{method}` across all evaluated runs: {fmt(m)} +- {fmt(s)} "
            f"(n={len(method_perf[method])})."
        )
    lines.append("- Important: cross-task absolute returns are not directly comparable in magnitude; use per-task/matched-setting comparisons.")
    lines.append("")

    lines.append("## Matched Confirm-Stage Comparison (same budget/profile)")
    lines.append("")
    lines.extend(render_confirm_method_table(records))
    lines.append("")
    lines.append("Interpretation:")
    if confirm_gap_summary["mlp_better"]:
        lines.append(
            "- In matched confirm runs, MLP-world-model methods are stronger on: "
            + ", ".join(f"`{task}`" for task in confirm_gap_summary["mlp_better"])
            + "."
        )
    if confirm_gap_summary["flow_better"]:
        lines.append(
            "- Flow-world-model methods are stronger on: "
            + ", ".join(f"`{task}`" for task in confirm_gap_summary["flow_better"])
            + "."
        )
    if confirm_gap_summary["near_parity"]:
        lines.append(
            "- Near-parity tasks (flow vs MLP within about 10% relative gap): "
            + ", ".join(f"`{task}`" for task in confirm_gap_summary["near_parity"])
            + "."
        )
    if confirm_gap_summary["incomplete"]:
        lines.append(
            "- Confirm tasks with incomplete 4-method coverage: "
            + ", ".join(f"`{task}`" for task in confirm_gap_summary["incomplete"])
            + "."
        )
    lines.append("")

    lines.append("## Early Signal from Advanced Flow Hyperparameter Sweep")
    lines.append("")
    if advanced_summary["top_task"] is None:
        lines.append("No advanced-sweep eval rows are completed yet.")
    else:
        lines.append(
            f"Current completion is concentrated on `{advanced_summary['top_task']}` "
            f"({advanced_summary['top_count']}/{advanced_summary['total_eval']} evaluated advanced runs), "
            "so findings remain preliminary."
        )
        lines.append("")
        if advanced_summary["top_task"] == "hopper":
            lines.extend(render_hopper_advanced_table(records))
    lines.append("")
    if advanced_summary["best_profiles"]:
        lines.append(f"Preliminary takeaway on `{advanced_summary['top_task']}`:")
        for method, profile, mean_val, n in advanced_summary["best_profiles"]:
            lines.append(
                f"- Best currently evaluated profile for `{method}` is `{profile}` "
                f"with mean return `{fmt(mean_val)}` over `{int(n)}` seed(s)."
            )
    lines.append("")

    lines.append("## Training-Status Insight")
    lines.append("")
    train_only_count = status_counts["train_done_no_eval"]
    if train_only_count > 0:
        lines.append(
            f"- {train_only_count} run(s) have `final_policy.pt` but no `eval_summary.json`, "
            "indicating training likely completed but evaluation/export did not finish."
        )
    else:
        lines.append("- No train-only rows are currently visible; completed trainings are generally paired with evaluation outputs.")
    if pending_counts:
        stage_list = ", ".join(
            f"`{stage}` ({count})"
            for stage, count in pending_counts.most_common(3)
        )
        lines.append(f"- Remaining evidence gaps are concentrated in: {stage_list}.")
    else:
        lines.append("- No unfinished stages remain in the currently indexed manifests.")
    lines.append(
        f"- Local outputs include final checkpoints for {len(final_rows)} runs and eval summaries for {len(eval_rows)} runs, "
        "so evaluation completion is high once training reaches a final checkpoint."
    )
    lines.append("")

    lines.append("## Recommended Reading Order for Advisor")
    lines.append("")
    lines.append("1. Start with per-task summary CSV for compact statistical view (seed counts + mean/std + parameter profile).")
    lines.append("2. Use full eval CSV to drill down to per-seed evidence and exact run-level overrides.")
    lines.append("3. Use unfinished-experiments doc to see what remains and why specific parameter axes were selected.")
    lines.append("")

    return "\n".join(lines)


def build_unfinished_md(
    records: list[RunRecord],
    output_full_csv: Path,
    output_task_csv: Path,
    now_str: str,
) -> str:
    pending = [r for r in records if status_from_flags(r.has_final_policy, r.has_eval_summary) != "completed_eval"]

    # Aggregate unfinished by stage/task/method/profile.
    group_counts: Counter[tuple[str, str, str, str, str]] = Counter()
    for r in pending:
        row = r.manifest_row
        key = (row["stage"], row["task_key"], row["method_key"], row["hparam_profile"], row["suite"])
        group_counts[key] += 1

    # Rationale map from notes (mainly flowwm_hparam_explore).
    profile_rationale: dict[str, str] = {}
    for r in records:
        row = r.manifest_row
        profile = row["hparam_profile"]
        note = row.get("notes", "")
        marker = "rationale="
        if marker in note and profile not in profile_rationale:
            profile_rationale[profile] = note.split(marker, 1)[1].strip()

    # Original vs flow-default reference (from yaml configs).
    param_table = [
        ("World model type", "MLP WorldModel", "FlowWorldModel", "Core modeling change (baseline vs flow)."),
        ("model_lr", "3e-4", "3e-4 (sweep: 2e-4, 1e-4)", "Controls world-model update speed/stability."),
        ("rew_rms", "true", "false (sweep includes true)", "Reward normalization; can stabilize reward head scale."),
        ("wm_grad_norm", "20.0", "20.0 (sweep: 10.0, 5.0)", "Gradient clipping on WM; tighter clip may reduce instability."),
        ("flow_integrator", "N/A", "heun", "Flow ODE integration scheme (accuracy vs cost)."),
        ("flow_substeps", "N/A", "4", "Temporal integration granularity in flow dynamics."),
        ("flow_tau_sampling", "N/A", "uniform (sweep includes midpoint)", "Sampling of flow time variable; affects variance/bias."),
        ("wm_dyn_loss_weight : wm_rew_loss_weight", "Implicit single objective", "1:1 (sweep 2:1, 1:2)", "Balances dynamics fitting vs reward prediction."),
        ("wm_bootstrap_iterations", "N/A", "0 (sweep 300, 2000)", "Warm-start training length for world model."),
        ("horizon_start / horizon_switch_epoch", "fixed horizon", "0 / 0 (sweep 1->base)", "Short-to-long planning schedule; reduces early model-exploitation risk."),
    ]

    # Task-specific horizon (currently used in manifests).
    horizon_overrides = {
        "hopper": "7",
        "ant": "14",
        "anymal": "7",
        "humanoid": "13",
        "snu_humanoid": "13",
        "leap_left_grasp_asymmetric": "13",
        "leap_left_inhand_pen_twirl": "13",
        "tracking_rough_unitree_g1": "1",
    }

    lines: list[str] = []
    lines.append("# Unfinished Experiments and Hyperparameter Design Notes")
    lines.append("")
    lines.append(f"Generated on: {now_str}")
    lines.append("")
    lines.append("## What Is Still Unfinished")
    lines.append("")
    lines.append(f"- Unfinished runs (not yet `train+eval` complete): {len(pending)}")
    lines.append(f"- Full eval CSV: `{output_full_csv}`")
    lines.append(f"- Per-task summary CSV: `{output_task_csv}`")
    lines.append("")

    stage_pending = Counter(r.manifest_row["stage"] for r in pending)
    lines.append("### Pending Count by Stage")
    lines.append("")
    lines.append("| Stage | Pending runs |")
    lines.append("|---|---:|")
    for stage, cnt in sorted(stage_pending.items()):
        lines.append(f"| {stage} | {cnt} |")
    lines.append("")

    lines.append("### Largest Unfinished Buckets")
    lines.append("")
    lines.append("| Stage | Suite | Task | Method | Profile | Pending runs |")
    lines.append("|---|---|---|---|---|---:|")
    for (stage, task, method, profile, suite), cnt in sorted(group_counts.items(), key=lambda kv: (-kv[1], kv[0]))[:40]:
        lines.append(f"| {stage} | {suite} | {task} | {method} | {profile} | {cnt} |")
    lines.append("")

    lines.append("## Parameter Design and Meaning")
    lines.append("")
    lines.append("The flow-WM exploration is designed to isolate likely failure modes while preserving comparability to PWM baselines.")
    lines.append("")
    lines.append("### Original Baseline vs Flow Default")
    lines.append("")
    lines.append("| Parameter | Original baseline-aligned (pwm_5M_baseline_pwmorig) | Flow default (pwm_5M_flow_v2_substeps4) | Why it matters |")
    lines.append("|---|---|---|---|")
    for p, base, flow, why in param_table:
        lines.append(f"| {p} | {base} | {flow} | {why} |")
    lines.append("")

    lines.append("### Task-Specific Horizon Overrides Used in Current Single-Task Manifests")
    lines.append("")
    lines.append("| Task | Horizon override |")
    lines.append("|---|---:|")
    for task, hz in horizon_overrides.items():
        lines.append(f"| {task} | {hz} |")
    lines.append("")

    lines.append("### Advanced Flow Profile Rationale (from manifest notes)")
    lines.append("")
    lines.append("| Profile | Rationale |")
    lines.append("|---|---|")
    for profile in sorted(profile_rationale):
        lines.append(f"| {profile} | {profile_rationale[profile]} |")
    lines.append("")

    lines.append("## How to Read Remaining Work")
    lines.append("")
    if not pending:
        lines.append("- No unfinished runs remain in the currently indexed manifests.")
    else:
        if "flow_rms_align" in stage_pending:
            lines.append(
                f"- `flow_rms_align` still has {stage_pending['flow_rms_align']} unfinished run(s); "
                "RMS-on/off conclusions should remain provisional until they complete."
            )
        if "flowwm_hparam_explore" in stage_pending:
            lines.append(
                f"- `flowwm_hparam_explore` still has {stage_pending['flowwm_hparam_explore']} unfinished run(s); "
                "cross-task conclusions for advanced flow profiles are still incomplete."
            )
        if "confirm" in stage_pending:
            lines.append(
                f"- `confirm` still has {stage_pending['confirm']} unfinished run(s); "
                "refresh final method ranking after confirm fully closes."
            )
        train_only_count = sum(
            1 for r in records
            if status_from_flags(r.has_final_policy, r.has_eval_summary) == "train_done_no_eval"
        )
        if train_only_count > 0:
            lines.append(
                f"- {train_only_count} run(s) are train-complete but eval-missing; "
                "these are recoverable by rerunning evaluation only."
            )
    lines.append("")

    lines.append("## Reporting Guidance")
    lines.append("")
    lines.append("- For fair model comparisons, prioritize matched settings: same task, method family, epoch budget, and horizon override.")
    lines.append("- Use per-task summary rows with `seed_count_eval >= 3` for stable statements; treat `seed_count_eval = 1` as directional evidence only.")
    lines.append("- Keep advanced-profile findings labeled as preliminary until replicated on at least one additional seed and one additional task.")
    lines.append("")

    return "\n".join(lines)


def main() -> None:
    args = parse_args()

    rows_by_key, sources_by_key = load_manifest_rows(args.manifest_dir)
    records = [
        enrich_record(row, sources_by_key[run_key][0], args.outputs_root)
        for run_key, row in sorted(rows_by_key.items())
    ]

    full_rows = build_full_eval_rows(records)
    task_rows = build_task_summary(records)

    write_csv(args.output_full_csv, full_rows, FULL_EVAL_COLUMNS)
    write_csv(args.output_task_csv, task_rows, TASK_SUMMARY_COLUMNS)

    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S %Z")

    insights_md = build_insights_md(
        records=records,
        full_rows=full_rows,
        task_rows=task_rows,
        sources_by_key=sources_by_key,
        output_full_csv=args.output_full_csv,
        output_task_csv=args.output_task_csv,
        now_str=now_str,
    )
    args.output_insights_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_insights_md.write_text(insights_md)

    unfinished_md = build_unfinished_md(
        records=records,
        output_full_csv=args.output_full_csv,
        output_task_csv=args.output_task_csv,
        now_str=now_str,
    )
    args.output_unfinished_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_unfinished_md.write_text(unfinished_md)

    status_counts = Counter(status_from_flags(r.has_final_policy, r.has_eval_summary) for r in records)
    print(f"planned_unique_runs={len(records)}")
    print(f"completed_eval_runs={status_counts['completed_eval']}")
    print(f"eval_rows_exported={len(full_rows)}")
    print(f"task_summary_rows={len(task_rows)}")
    print(f"full_csv={args.output_full_csv}")
    print(f"task_csv={args.output_task_csv}")
    print(f"insights_md={args.output_insights_md}")
    print(f"unfinished_md={args.output_unfinished_md}")


if __name__ == "__main__":
    main()
