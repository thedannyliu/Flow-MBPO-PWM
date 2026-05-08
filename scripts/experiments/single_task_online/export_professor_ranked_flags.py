#!/usr/bin/env python3
"""Export professor-facing ranked summaries with trust/pipeline flags."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path


TRUST_ORDER = {
    "final_fair_use": 0,
    "final_fair_use_with_metric_caveat": 1,
    "exploratory_only": 2,
    "historical_non_aligned": 3,
    "provisional_bug_path": 4,
    "contaminated_do_not_use": 5,
}

METHOD_LABELS = {
    "mlpwm_mlppolicy": "PWM",
    "flowwm_mlppolicy": "Flow WM",
    "mlpwm_flowpolicy": "Flow Policy",
    "flowwm_flowpolicy": "Full Flow",
}


@dataclass
class GroupAudit:
    trust_tier: str
    issue_flags: list[str]
    issue_summary: str
    use_for_final_fair_comparison: bool
    use_for_exploratory_analysis: bool


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-summary-csv", type=Path, required=True)
    parser.add_argument("--audit-csv", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    return parser.parse_args()


def load_audit_groups(audit_csv: Path) -> dict[tuple[str, str, str, str, str], GroupAudit]:
    grouped_flags: dict[tuple[str, str, str, str, str], set[str]] = defaultdict(set)
    grouped_summaries: dict[tuple[str, str, str, str, str], list[str]] = defaultdict(list)
    grouped_trust: dict[tuple[str, str, str, str, str], list[str]] = defaultdict(list)
    grouped_final: dict[tuple[str, str, str, str, str], list[bool]] = defaultdict(list)
    grouped_expl: dict[tuple[str, str, str, str, str], list[bool]] = defaultdict(list)

    with audit_csv.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            key = (
                row["stage"],
                row["suite"],
                row["task_key"],
                row["method_key"],
                row["hparam_profile"],
            )
            trust = row["trust_tier"].strip()
            grouped_trust[key].append(trust)
            flags = [flag.strip() for flag in row["issue_flags"].split(";") if flag.strip()]
            grouped_flags[key].update(flags)
            summary = row["issue_summary"].strip()
            if summary:
                grouped_summaries[key].append(summary)
            grouped_final[key].append(row["use_for_final_fair_comparison"].strip() == "True")
            grouped_expl[key].append(row["use_for_exploratory_analysis"].strip() == "True")

    result: dict[tuple[str, str, str, str, str], GroupAudit] = {}
    for key, trust_list in grouped_trust.items():
        sorted_trust = sorted(trust_list, key=lambda t: TRUST_ORDER.get(t, 999))
        unique_summaries = []
        for summary in grouped_summaries[key]:
            if summary not in unique_summaries:
                unique_summaries.append(summary)
        result[key] = GroupAudit(
            trust_tier=sorted_trust[-1] if sorted_trust else "exploratory_only",
            issue_flags=sorted(grouped_flags[key]),
            issue_summary=" ".join(unique_summaries),
            use_for_final_fair_comparison=all(grouped_final[key]) if grouped_final[key] else False,
            use_for_exploratory_analysis=any(grouped_expl[key]) if grouped_expl[key] else False,
        )
    return result


def load_task_summary(task_summary_csv: Path) -> list[dict[str, str]]:
    with task_summary_csv.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def build_rows(rows: list[dict[str, str]], audit_groups: dict[tuple[str, str, str, str, str], GroupAudit]) -> list[dict[str, str]]:
    output_rows: list[dict[str, str]] = []
    for row in rows:
        key = (
            row["stage"],
            row["suite"],
            row["task_key"],
            row["method_key"],
            row["hparam_profile"],
        )
        audit = audit_groups.get(
            key,
            GroupAudit(
                trust_tier="exploratory_only",
                issue_flags=[],
                issue_summary="No audit match found.",
                use_for_final_fair_comparison=False,
                use_for_exploratory_analysis=True,
            ),
        )
        out = dict(row)
        out["method_label"] = METHOD_LABELS.get(row["method_key"], row["method_key"])
        out["trust_tier"] = audit.trust_tier
        out["issue_flags"] = ";".join(audit.issue_flags)
        out["issue_summary"] = audit.issue_summary
        out["use_for_final_fair_comparison"] = str(audit.use_for_final_fair_comparison)
        out["use_for_exploratory_analysis"] = str(audit.use_for_exploratory_analysis)
        output_rows.append(out)
    return output_rows


def write_csv(rows: list[dict[str, str]], output_csv: Path) -> None:
    preferred_prefix = [
        "stage",
        "suite",
        "task_key",
        "method_key",
        "method_label",
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
        "trust_tier",
        "issue_flags",
        "use_for_final_fair_comparison",
        "use_for_exploratory_analysis",
        "issue_summary",
    ]
    extra_fields = []
    for row in rows:
        for key in row:
            if key not in preferred_prefix and key not in extra_fields:
                extra_fields.append(key)
    fieldnames = [key for key in preferred_prefix if any(key in row for row in rows)] + extra_fields
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def trust_note(trust_tier: str) -> str:
    if trust_tier == "final_fair_use":
        return "safe for final fair comparison"
    if trust_tier == "final_fair_use_with_metric_caveat":
        return "safe for final fair comparison, but one metric is saturated/caveated"
    if trust_tier == "exploratory_only":
        return "use only for exploratory analysis"
    if trust_tier == "historical_non_aligned":
        return "historical result; pipeline/baseline not aligned"
    if trust_tier == "provisional_bug_path":
        return "provisional; known bug-affected code path"
    if trust_tier == "contaminated_do_not_use":
        return "do not use; contaminated"
    return trust_tier


def has_numeric_return(row: dict[str, str]) -> bool:
    value = row.get("return_mean_avg", "").strip()
    if not value:
        return False
    try:
        float(value)
    except ValueError:
        return False
    return True


def render_md(rows: list[dict[str, str]], output_md: Path) -> None:
    fair_rows = [row for row in rows if row["stage"] == "confirm_fair_small" and has_numeric_return(row)]
    fair_by_task: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in fair_rows:
        fair_by_task[row["task_key"]].append(row)

    all_by_stage_task: dict[tuple[str, str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        all_by_stage_task[(row["stage"], row["suite"], row["task_key"])].append(row)

    lines: list[str] = []
    lines.append("# Professor Ranked Summary With Flags")
    lines.append("")
    lines.append("## Recommendation")
    lines.append("")
    lines.append("The ranking format is usable, but only if each block is explicitly labeled with trust/pipeline status.")
    lines.append("For professor-facing reporting, the fair section should keep the ranking style, but contaminated or non-aligned tasks must be marked directly next to the ranking.")
    lines.append("")
    lines.append("## Fair Summary")
    lines.append("")

    for task in sorted(fair_by_task):
        task_rows = sorted(
            fair_by_task[task],
            key=lambda row: float(row["return_mean_avg"]),
            reverse=True,
        )
        trust_levels = {row["trust_tier"] for row in task_rows}
        task_trust = sorted(trust_levels, key=lambda t: TRUST_ORDER.get(t, 999))[-1]
        coverage = f"partial_{len(task_rows)}_methods" if len(task_rows) < 4 else "complete_4_methods"
        lines.append(f"## {task}")
        lines.append(f"- stage: `confirm_fair_small`")
        lines.append(f"- coverage: `{coverage}`")
        lines.append(f"- trust: `{task_trust}` ({trust_note(task_trust)})")
        task_flags = sorted({flag for row in task_rows for flag in row['issue_flags'].split(';') if flag})
        if task_flags:
            lines.append(f"- flags: `{'; '.join(task_flags)}`")
        for idx, row in enumerate(task_rows, start=1):
            lines.append(
                f"- {idx}. {row['method_label']} (`{row['method_key']}`): "
                f"return_mean_avg={float(row['return_mean_avg']):.3f}, "
                f"return_std={float(row['return_mean_std']):.3f}, "
                f"episode_length_mean_avg={float(row['episode_length_mean_avg']):.3f}, "
                f"n={int(float(row['n_eval_completed']))}, "
                f"trust=`{row['trust_tier']}`"
            )
        lines.append("")

    lines.append("## All Experiments")
    lines.append("")
    lines.append("Below, every stage/task block is grouped in the same ranked style, but each block is explicitly marked as fair, exploratory, historical, provisional, or contaminated.")
    lines.append("")

    for (stage, suite, task), group_rows in sorted(all_by_stage_task.items()):
        valid_rows = [row for row in group_rows if has_numeric_return(row)]
        if not valid_rows:
            continue
        sorted_rows = sorted(
            valid_rows,
            key=lambda row: float(row["return_mean_avg"]),
            reverse=True,
        )
        block_trust = sorted({row["trust_tier"] for row in valid_rows}, key=lambda t: TRUST_ORDER.get(t, 999))[-1]
        lines.append(f"### {stage} / {suite} / {task}")
        lines.append(f"- block_trust: `{block_trust}` ({trust_note(block_trust)})")
        block_flags = sorted({flag for row in valid_rows for flag in row['issue_flags'].split(';') if flag})
        if block_flags:
            lines.append(f"- block_flags: `{'; '.join(block_flags)}`")
        for idx, row in enumerate(sorted_rows, start=1):
            lines.append(
                f"- {idx}. {row['method_label']} (`{row['method_key']}`), profile=`{row['hparam_profile']}`: "
                f"return_mean_avg={float(row['return_mean_avg']):.3f}, "
                f"return_std={float(row['return_mean_std']):.3f}, "
                f"episode_length_mean_avg={float(row['episode_length_mean_avg']):.3f}, "
                f"n={int(float(row['n_eval_completed']))}, "
                f"trust=`{row['trust_tier']}`"
            )
            lines.append(f"  overrides: `{row['overrides']}`")
            if row["issue_summary"]:
                lines.append(f"  note: {row['issue_summary']}")
        lines.append("")

    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_md.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    audit_groups = load_audit_groups(args.audit_csv)
    task_summary_rows = load_task_summary(args.task_summary_csv)
    rows = build_rows(task_summary_rows, audit_groups)
    write_csv(rows, args.output_csv)
    render_md(rows, args.output_md)
    print(f"Wrote {len(rows)} rows to {args.output_csv}")
    print(f"Wrote markdown to {args.output_md}")


if __name__ == "__main__":
    main()
