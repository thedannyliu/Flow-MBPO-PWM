#!/usr/bin/env python3
"""Rank Flow-MBPO candidate checkpoints using eval and rollout evidence."""

from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
from pathlib import Path
from typing import Any


METRIC_FIELDS = [
    "return_mean",
    "episode_length_mean",
    "fall_rate_mean",
    "timeout_rate_mean",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--eval-dir", required=True, help="Directory with candidate subdirs containing eval summary.json")
    p.add_argument("--rollout-dir", required=True, help="Directory with candidate subdirs containing rollout summary.json")
    p.add_argument("--baseline-eval-summary", required=True)
    p.add_argument("--baseline-rollout-summary", required=True)
    p.add_argument("--baseline-name", default="baseline")
    p.add_argument("--output-csv", required=True)
    p.add_argument("--output-md", required=True)
    return p.parse_args()


def git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_candidate_summaries(root: Path) -> dict[str, dict[str, Any]]:
    summaries: dict[str, dict[str, Any]] = {}
    if not root.exists():
        return summaries
    for path in sorted(root.glob("*/summary.json")):
        summaries[path.parent.name] = load_json(path)
    return summaries


def as_float(summary: dict[str, Any] | None, key: str) -> float:
    if summary is None or key not in summary:
        return math.nan
    try:
        return float(summary[key])
    except (TypeError, ValueError):
        return math.nan


def better_than_baseline(candidate: dict[str, float], prefix: str, baseline: dict[str, Any]) -> bool:
    return (
        candidate[f"{prefix}_return_mean"] > as_float(baseline, "return_mean")
        and candidate[f"{prefix}_episode_length_mean"] > as_float(baseline, "episode_length_mean")
        and candidate[f"{prefix}_fall_rate_mean"] < as_float(baseline, "fall_rate_mean")
    )


def summary_gate(summary: dict[str, Any] | None, computed: bool) -> tuple[bool, str]:
    if summary and summary.get("baseline_gate_configured") is True and "baseline_gate_pass" in summary:
        return bool(summary["baseline_gate_pass"]), "summary"
    return computed, "computed"


def row_for_candidate(
    name: str,
    eval_summary: dict[str, Any] | None,
    rollout_summary: dict[str, Any] | None,
    baseline_eval: dict[str, Any],
    baseline_rollout: dict[str, Any],
) -> dict[str, Any]:
    row: dict[str, Any] = {"candidate": name}
    for field in METRIC_FIELDS:
        row[f"eval_{field}"] = as_float(eval_summary, field)
        row[f"rollout_{field}"] = as_float(rollout_summary, field)
    row["eval_summary"] = eval_summary.get("command", "") if eval_summary else ""
    row["rollout_video"] = rollout_summary.get("video", "") if rollout_summary else ""
    computed_scalar_gate = better_than_baseline(row, "eval", baseline_eval)
    computed_video_gate = better_than_baseline(row, "rollout", baseline_rollout)
    row["scalar_gate_pass"], row["scalar_gate_source"] = summary_gate(eval_summary, computed_scalar_gate)
    row["video_gate_pass"], row["video_gate_source"] = summary_gate(rollout_summary, computed_video_gate)
    row["joint_gate_pass"] = row["scalar_gate_pass"] and row["video_gate_pass"]
    return row


def sort_key(row: dict[str, Any]) -> tuple[Any, ...]:
    return (
        row["joint_gate_pass"],
        row["scalar_gate_pass"],
        row["video_gate_pass"],
        row["eval_return_mean"],
        row["rollout_return_mean"],
        -row["eval_fall_rate_mean"],
        -row["rollout_fall_rate_mean"],
    )


def format_cell(value: Any) -> str:
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float):
        if math.isnan(value):
            return ""
        return f"{value:.4f}"
    return str(value)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "candidate",
        "scalar_gate_pass",
        "scalar_gate_source",
        "video_gate_pass",
        "video_gate_source",
        "joint_gate_pass",
        "eval_return_mean",
        "eval_episode_length_mean",
        "eval_fall_rate_mean",
        "eval_timeout_rate_mean",
        "rollout_return_mean",
        "rollout_episode_length_mean",
        "rollout_fall_rate_mean",
        "rollout_timeout_rate_mean",
        "rollout_video",
        "eval_summary",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows([{field: row.get(field, "") for field in fields} for row in rows])


def write_md(
    path: Path,
    rows: list[dict[str, Any]],
    baseline_name: str,
    baseline_eval: dict[str, Any],
    baseline_rollout: dict[str, Any],
) -> None:
    fields = [
        "candidate",
        "scalar_gate_pass",
        "scalar_gate_source",
        "video_gate_pass",
        "video_gate_source",
        "joint_gate_pass",
        "eval_return_mean",
        "eval_episode_length_mean",
        "eval_fall_rate_mean",
        "rollout_return_mean",
        "rollout_episode_length_mean",
        "rollout_fall_rate_mean",
    ]
    lines = [
        "# Flow-MBPO Candidate Evidence Ranking",
        "",
        f"Git SHA: `{git_sha()}`",
        "",
        f"Baseline: `{baseline_name}`",
        "",
        (
            "Eval baseline: "
            f"return `{as_float(baseline_eval, 'return_mean'):.4f}`, "
            f"length `{as_float(baseline_eval, 'episode_length_mean'):.2f}`, "
            f"fall `{as_float(baseline_eval, 'fall_rate_mean'):.3f}`"
        ),
        (
            "Rollout baseline: "
            f"return `{as_float(baseline_rollout, 'return_mean'):.4f}`, "
            f"length `{as_float(baseline_rollout, 'episode_length_mean'):.2f}`, "
            f"fall `{as_float(baseline_rollout, 'fall_rate_mean'):.3f}`"
        ),
        "",
        "| " + " | ".join(fields) + " |",
        "| " + " | ".join(["---"] * len(fields)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(format_cell(row.get(field, "")) for field in fields) + " |")
    lines.extend(
        [
            "",
            "A candidate passes only if return and episode length are above baseline and fall rate is below baseline.",
            "Gate source is `summary` when the candidate summary recorded baseline_gate_pass; otherwise it is recomputed from the baseline summaries supplied to this ranking command.",
            "Rows that pass only one gate remain diagnostic and are not policy-improvement claims.",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    eval_summaries = load_candidate_summaries(Path(args.eval_dir))
    rollout_summaries = load_candidate_summaries(Path(args.rollout_dir))
    baseline_eval = load_json(Path(args.baseline_eval_summary))
    baseline_rollout = load_json(Path(args.baseline_rollout_summary))
    names = sorted(set(eval_summaries) | set(rollout_summaries))
    rows = [
        row_for_candidate(name, eval_summaries.get(name), rollout_summaries.get(name), baseline_eval, baseline_rollout)
        for name in names
    ]
    rows.sort(key=sort_key, reverse=True)
    write_csv(Path(args.output_csv), rows)
    write_md(Path(args.output_md), rows, args.baseline_name, baseline_eval, baseline_rollout)
    print(json.dumps({"rows": len(rows), "output_csv": args.output_csv, "output_md": args.output_md}, sort_keys=True))


if __name__ == "__main__":
    main()
