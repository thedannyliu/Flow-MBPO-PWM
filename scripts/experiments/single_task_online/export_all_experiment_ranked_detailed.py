#!/usr/bin/env python3
"""Export a detailed ranked experiment summary for all evaluated blocks."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path


METHOD_LABELS = {
    "mlpwm_mlppolicy": "PWM",
    "flowwm_mlppolicy": "Flow WM",
    "mlpwm_flowpolicy": "Flow Policy",
    "flowwm_flowpolicy": "Full Flow",
}

TRUST_ORDER = {
    "final_fair_use": 0,
    "final_fair_use_with_metric_caveat": 1,
    "exploratory_only": 2,
    "historical_non_aligned": 3,
    "provisional_bug_path": 4,
    "contaminated_do_not_use": 5,
}


@dataclass
class MethodAudit:
    trust_tier: str
    flags: list[str]
    summaries: list[str]
    seeds: list[str]
    task_resolution_statuses: list[str]
    requested_resolved_pairs: list[str]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-summary-csv", type=Path, required=True)
    parser.add_argument("--audit-csv", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    return parser.parse_args()


def load_task_summary(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def load_method_audit(path: Path) -> dict[tuple[str, str, str, str, str], MethodAudit]:
    grouped_flags: dict[tuple[str, str, str, str, str], set[str]] = defaultdict(set)
    grouped_summaries: dict[tuple[str, str, str, str, str], list[str]] = defaultdict(list)
    grouped_trust: dict[tuple[str, str, str, str, str], list[str]] = defaultdict(list)
    grouped_seeds: dict[tuple[str, str, str, str, str], set[str]] = defaultdict(set)
    grouped_res_status: dict[tuple[str, str, str, str, str], set[str]] = defaultdict(set)
    grouped_pairs: dict[tuple[str, str, str, str, str], set[str]] = defaultdict(set)

    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            key = (
                row["stage"],
                row["suite"],
                row["task_key"],
                row["method_key"],
                row["hparam_profile"],
            )
            grouped_trust[key].append(row["trust_tier"].strip())
            grouped_seeds[key].add(row["seed"].strip())
            grouped_res_status[key].add(row["task_resolution_status"].strip())
            requested = row["requested_task_id"].strip()
            resolved = row["resolved_task_id"].strip()
            if requested or resolved:
                grouped_pairs[key].add(f"{requested or 'NA'} -> {resolved or 'NA'}")
            for flag in row["issue_flags"].split(";"):
                flag = flag.strip()
                if flag:
                    grouped_flags[key].add(flag)
            summary = row["issue_summary"].strip()
            if summary and summary not in grouped_summaries[key]:
                grouped_summaries[key].append(summary)

    result: dict[tuple[str, str, str, str, str], MethodAudit] = {}
    for key, trust_list in grouped_trust.items():
        worst_trust = sorted(trust_list, key=lambda t: TRUST_ORDER.get(t, 999))[-1]
        result[key] = MethodAudit(
            trust_tier=worst_trust,
            flags=sorted(grouped_flags[key]),
            summaries=grouped_summaries[key],
            seeds=sorted(grouped_seeds[key], key=lambda s: int(s) if s.isdigit() else s),
            task_resolution_statuses=sorted(grouped_res_status[key]),
            requested_resolved_pairs=sorted(grouped_pairs[key]),
        )
    return result


def method_sort_key(row: dict[str, str]) -> tuple[float, str]:
    try:
        value = float(row["return_mean_avg"])
    except ValueError:
        value = float("-inf")
    return (-value, row["method_key"])


def has_numeric_metrics(row: dict[str, str]) -> bool:
    for key in ("return_mean_avg", "return_mean_std", "episode_length_mean_avg", "n_eval_completed"):
        value = row.get(key, "").strip()
        if not value:
            return False
    return True


def coverage_label(rows: list[dict[str, str]]) -> str:
    return "complete_4_methods" if len(rows) >= 4 else f"partial_{len(rows)}_methods"


def summarize_block_flags(rows: list[dict[str, str]], audit: dict[tuple[str, str, str, str, str], MethodAudit]) -> tuple[str, list[str]]:
    trust_levels: list[str] = []
    flags: set[str] = set()
    for row in rows:
        key = (row["stage"], row["suite"], row["task_key"], row["method_key"], row["hparam_profile"])
        audit_row = audit.get(key)
        if not audit_row:
            continue
        trust_levels.append(audit_row.trust_tier)
        flags.update(audit_row.flags)
    if trust_levels:
        block_trust = sorted(trust_levels, key=lambda t: TRUST_ORDER.get(t, 999))[-1]
    else:
        block_trust = "unknown"
    return block_trust, sorted(flags)


def write_csv(rows: list[dict[str, str]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    task_rows = load_task_summary(args.task_summary_csv)
    audit = load_method_audit(args.audit_csv)

    grouped: dict[tuple[str, str, str], list[dict[str, str]]] = defaultdict(list)
    for row in task_rows:
        grouped[(row["stage"], row["suite"], row["task_key"])].append(row)

    output_rows: list[dict[str, str]] = []
    lines: list[str] = []
    lines.append("# All Experiment Ranked Summary")
    lines.append("")
    lines.append("Each block is grouped by `stage / suite / task` and ranked by `return_mean_avg`.")
    lines.append("Every method entry includes seed coverage, rough hyperparameter settings, task-resolution status, and anomaly flags.")
    lines.append("")

    for block_key in sorted(grouped):
        stage, suite, task_key = block_key
        rows = [row for row in grouped[block_key] if has_numeric_metrics(row)]
        if not rows:
            continue
        rows = sorted(rows, key=method_sort_key)
        block_trust, block_flags = summarize_block_flags(rows, audit)

        lines.append(f"## {task_key}")
        lines.append(f"- stage: `{stage}`")
        lines.append(f"- suite: `{suite}`")
        lines.append(f"- coverage: `{coverage_label(rows)}`")
        lines.append(f"- block_trust: `{block_trust}`")
        if block_flags:
            lines.append(f"- block_flags: `{'; '.join(block_flags)}`")

        for idx, row in enumerate(rows, start=1):
            method_key = row["method_key"]
            method_label = METHOD_LABELS.get(method_key, method_key)
            audit_key = (stage, suite, task_key, method_key, row["hparam_profile"])
            audit_row = audit.get(audit_key)

            seeds_eval = row.get("seeds_eval", "")
            trust = audit_row.trust_tier if audit_row else "unknown"
            task_resolution = ", ".join(audit_row.task_resolution_statuses) if audit_row else "unknown"
            fallback_pairs = "; ".join(audit_row.requested_resolved_pairs) if audit_row else ""
            anomalies = "; ".join(audit_row.flags) if audit_row and audit_row.flags else ""
            note = " ".join(audit_row.summaries) if audit_row and audit_row.summaries else ""

            lines.append(
                f"- {idx}. {method_label} (`{method_key}`): "
                f"return_mean_avg={float(row['return_mean_avg']):.3f}, "
                f"return_std={float(row['return_mean_std']):.3f}, "
                f"episode_length_mean_avg={float(row['episode_length_mean_avg']):.3f}, "
                f"n={int(float(row['n_eval_completed']))}"
            )
            lines.append(f"  seeds_eval: `{seeds_eval}`")
            lines.append(f"  profile: `{row['hparam_profile']}` | alg: `{row['alg']}`")
            lines.append(f"  overrides: `{row['overrides']}`")
            lines.append(f"  trust: `{trust}` | task_resolution: `{task_resolution}`")
            if fallback_pairs:
                lines.append(f"  requested_resolved_pairs: `{fallback_pairs}`")
            if anomalies:
                lines.append(f"  anomalies: `{anomalies}`")
            if note:
                lines.append(f"  note: {note}")

            output_rows.append(
                {
                    "stage": stage,
                    "suite": suite,
                    "task_key": task_key,
                    "rank": str(idx),
                    "method_key": method_key,
                    "method_label": method_label,
                    "coverage": coverage_label(rows),
                    "block_trust": block_trust,
                    "block_flags": ";".join(block_flags),
                    "return_mean_avg": row["return_mean_avg"],
                    "return_mean_std": row["return_mean_std"],
                    "episode_length_mean_avg": row["episode_length_mean_avg"],
                    "n_eval_completed": row["n_eval_completed"],
                    "seeds_eval": seeds_eval,
                    "hparam_profile": row["hparam_profile"],
                    "alg": row["alg"],
                    "overrides": row["overrides"],
                    "trust_tier": trust,
                    "task_resolution": task_resolution,
                    "requested_resolved_pairs": fallback_pairs,
                    "anomalies": anomalies,
                    "note": note,
                }
            )
        lines.append("")

    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.write_text("\n".join(lines), encoding="utf-8")
    write_csv(output_rows, args.output_csv)
    print(f"Wrote markdown to {args.output_md}")
    print(f"Wrote csv to {args.output_csv}")


if __name__ == "__main__":
    main()
