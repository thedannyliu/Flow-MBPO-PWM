#!/usr/bin/env python3
"""Export confirm-phase completed eval CSV and markdown analysis snapshot."""

from __future__ import annotations

import argparse
import csv
import json
import re
import statistics
import subprocess
from collections import defaultdict
from pathlib import Path
from typing import Any


ROW_COLUMNS = [
    "row_index",
    "run_key",
    "suite",
    "task_key",
    "method_key",
    "seed",
    "alg",
    "num_envs",
    "eval_runs",
    "slurm_state",
    "slurm_elapsed",
    "slurm_maxrss",
    "output_dir",
    "has_final_policy",
    "has_best_policy",
    "has_eval_summary_json",
    "has_episode_metrics_csv",
    "has_rollout_steps_csv",
    "has_rollout_summary_csv",
    "has_rollout_mp4",
    "has_rollout_gif",
    "return_mean",
    "return_std",
    "return_iqm",
    "return_median",
    "discounted_return_mean",
    "episode_length_mean",
    "episode_length_std",
    "success_rate",
]

RUN_STATES = {"RUNNING", "COMPLETING"}
FAIL_STATES = {
    "FAILED",
    "OUT_OF_MEMORY",
    "TIMEOUT",
    "CANCELLED",
    "PREEMPTED",
    "NODE_FAIL",
    "BOOT_FAIL",
    "DEADLINE",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--outputs-root", type=Path, required=True)
    parser.add_argument("--slurm-job-id", required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--output-analysis-md", type=Path, required=True)
    return parser.parse_args()


def safe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if out != out:  # NaN
        return None
    return out


def avg(values: list[float]) -> float | None:
    return statistics.fmean(values) if values else None


def med(values: list[float]) -> float | None:
    return statistics.median(values) if values else None


def fmt_num(value: float | None, ndigits: int = 3) -> str:
    if value is None:
        return "nan"
    return f"{value:.{ndigits}f}"


def parse_sacct_row_states(job_id: str) -> dict[int, dict[str, str]]:
    cmd = [
        "sacct",
        "-j",
        job_id,
        "--format=JobID,State,Elapsed,MaxRSS",
        "--parsable2",
        "--noheader",
    ]
    result = subprocess.run(cmd, check=True, capture_output=True, text=True)
    row_pat = re.compile(rf"^{re.escape(job_id)}_(\d+)$")
    row_info: dict[int, dict[str, str]] = {}
    for raw_line in result.stdout.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        parts = line.split("|")
        if len(parts) < 4:
            continue
        job_name, state, elapsed, maxrss = parts[0], parts[1], parts[2], parts[3]
        match = row_pat.match(job_name)
        if not match:
            continue
        row_idx = int(match.group(1))
        row_info[row_idx] = {
            "state": state.strip(),
            "elapsed": elapsed.strip(),
            "maxrss": maxrss.strip(),
        }
    return row_info


def main() -> None:
    args = parse_args()

    with args.manifest.open(newline="") as fp:
        manifest_rows = list(csv.DictReader(fp))

    sacct_by_row = parse_sacct_row_states(args.slurm_job_id)
    enriched_rows: list[dict[str, Any]] = []

    for idx, row in enumerate(manifest_rows):
        suite = row["suite"]
        task = row["task_key"]
        method = row["method_key"]
        seed = row["seed"]
        profile = row["hparam_profile"]
        output_dir = (
            args.outputs_root / suite / task / method / f"seed_{seed}" / profile
        )
        logs_dir = output_dir / "logs"
        eval_dir = output_dir / "eval"

        has_final_policy = (logs_dir / "final_policy.pt").exists()
        has_best_policy = (logs_dir / "best_policy.pt").exists()
        has_eval_summary_json = (eval_dir / "eval_summary.json").exists()
        has_episode_metrics_csv = (eval_dir / "episode_metrics.csv").exists()
        has_rollout_steps_csv = (eval_dir / "rollout_steps.csv").exists()
        has_rollout_summary_csv = (eval_dir / "rollout_summary.csv").exists()
        has_rollout_mp4 = (eval_dir / "rollout.mp4").exists()
        has_rollout_gif = (eval_dir / "rollout.gif").exists()

        metrics = {}
        if has_eval_summary_json:
            with (eval_dir / "eval_summary.json").open() as fp:
                metrics = json.load(fp)

        state_info = sacct_by_row.get(
            idx, {"state": "PENDING", "elapsed": "", "maxrss": ""}
        )

        enriched_rows.append(
            {
                "row_index": idx,
                "run_key": row["run_key"],
                "suite": suite,
                "task_key": task,
                "method_key": method,
                "seed": row["seed"],
                "alg": row["alg"],
                "num_envs": row["num_envs"],
                "eval_runs": row["eval_runs"],
                "slurm_state": state_info["state"],
                "slurm_elapsed": state_info["elapsed"],
                "slurm_maxrss": state_info["maxrss"],
                "output_dir": str(output_dir.resolve()),
                "has_final_policy": has_final_policy,
                "has_best_policy": has_best_policy,
                "has_eval_summary_json": has_eval_summary_json,
                "has_episode_metrics_csv": has_episode_metrics_csv,
                "has_rollout_steps_csv": has_rollout_steps_csv,
                "has_rollout_summary_csv": has_rollout_summary_csv,
                "has_rollout_mp4": has_rollout_mp4,
                "has_rollout_gif": has_rollout_gif,
                "return_mean": metrics.get("return_mean"),
                "return_std": metrics.get("return_std"),
                "return_iqm": metrics.get("return_iqm"),
                "return_median": metrics.get("return_median"),
                "discounted_return_mean": metrics.get("discounted_return_mean"),
                "episode_length_mean": metrics.get("episode_length_mean"),
                "episode_length_std": metrics.get("episode_length_std"),
                "success_rate": metrics.get("success_rate"),
            }
        )

    completed_eval_rows = [
        row for row in enriched_rows if row["has_eval_summary_json"] and row["has_final_policy"]
    ]
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=ROW_COLUMNS)
        writer.writeheader()
        for row in completed_eval_rows:
            writer.writerow(row)

    task_counts: dict[str, dict[str, int]] = defaultdict(
        lambda: {"DONE": 0, "RUN": 0, "FAIL": 0, "PEND": 0}
    )
    method_counts: dict[str, dict[str, int]] = defaultdict(
        lambda: {"DONE": 0, "RUN": 0, "FAIL": 0, "PEND": 0}
    )
    row_state_counts = {"DONE": 0, "RUN": 0, "FAIL": 0, "PEND": 0}

    for row in enriched_rows:
        state = (row["slurm_state"] or "PENDING").upper()
        if state == "COMPLETED":
            bucket = "DONE"
        elif state in RUN_STATES:
            bucket = "RUN"
        elif state in FAIL_STATES:
            bucket = "FAIL"
        else:
            bucket = "PEND"
        row_state_counts[bucket] += 1
        task_counts[row["task_key"]][bucket] += 1
        method_counts[row["method_key"]][bucket] += 1

    eval_groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in completed_eval_rows:
        eval_groups[(row["task_key"], row["method_key"])].append(row)

    artifact_keys = [
        "has_final_policy",
        "has_best_policy",
        "has_eval_summary_json",
        "has_episode_metrics_csv",
        "has_rollout_steps_csv",
        "has_rollout_summary_csv",
        "has_rollout_mp4",
        "has_rollout_gif",
    ]
    artifact_counts = {
        key: sum(1 for row in completed_eval_rows if row[key]) for key in artifact_keys
    }

    lines: list[str] = []
    stamp = args.output_csv.stem.split("_", 3)[-1]
    lines.append(f"# Confirm Completed Eval Analysis ({stamp})")
    lines.append("")
    lines.append("## Snapshot")
    lines.append(f"- completed_runs_with_eval_csv: {len(completed_eval_rows)}")
    lines.append(
        "- row_state_counts: "
        f"DONE={row_state_counts['DONE']}, "
        f"RUN={row_state_counts['RUN']}, "
        f"FAIL={row_state_counts['FAIL']}, "
        f"PEND={row_state_counts['PEND']}"
    )
    lines.append("")
    lines.append("## Progress By Task")
    lines.append("task,DONE,RUN,FAIL,PEND")
    for task in sorted(task_counts):
        c = task_counts[task]
        lines.append(f"{task},{c['DONE']},{c['RUN']},{c['FAIL']},{c['PEND']}")
    lines.append("")
    lines.append("## Progress By Method")
    lines.append("method,DONE,RUN,FAIL,PEND")
    for method in sorted(method_counts):
        c = method_counts[method]
        lines.append(f"{method},{c['DONE']},{c['RUN']},{c['FAIL']},{c['PEND']}")
    lines.append("")
    lines.append("## Eval Aggregate (Completed Only)")
    lines.append(
        "task,method,n,return_mean_avg,return_mean_median,return_iqm_avg,"
        "episode_length_mean_avg,elapsed_hr_avg,elapsed_hr_median"
    )

    for (task, method), rows in sorted(eval_groups.items()):
        r_mean = [safe_float(r["return_mean"]) for r in rows]
        r_iqm = [safe_float(r["return_iqm"]) for r in rows]
        ep_len = [safe_float(r["episode_length_mean"]) for r in rows]
        elapsed_hr = []
        for r in rows:
            elapsed = r.get("slurm_elapsed") or ""
            try:
                hh, mm, ss = elapsed.split(":")
                elapsed_hr.append(int(hh) + int(mm) / 60.0 + int(ss) / 3600.0)
            except ValueError:
                pass
        r_mean_clean = [x for x in r_mean if x is not None]
        r_iqm_clean = [x for x in r_iqm if x is not None]
        ep_len_clean = [x for x in ep_len if x is not None]
        lines.append(
            ",".join(
                [
                    task,
                    method,
                    str(len(rows)),
                    fmt_num(avg(r_mean_clean), 3),
                    fmt_num(med(r_mean_clean), 3),
                    fmt_num(avg(r_iqm_clean), 3),
                    fmt_num(avg(ep_len_clean), 3),
                    fmt_num(avg(elapsed_hr), 3),
                    fmt_num(med(elapsed_hr), 3),
                ]
            )
        )

    lines.append("")
    lines.append("## Artifact Integrity (Completed Only)")
    total = len(completed_eval_rows)
    for key in artifact_keys:
        lines.append(f"- {key}: {artifact_counts[key]}/{total}")

    lines.append("")
    lines.append("## Key Insights")
    for method in sorted(method_counts):
        c = method_counts[method]
        launched = c["DONE"] + c["RUN"] + c["FAIL"]
        fail_ratio = c["FAIL"] / launched if launched else 0.0
        lines.append(
            f"- {method}: fail_ratio_on_launched={fail_ratio * 100:.1f}% "
            f"({c['FAIL']}/{launched})"
        )

    best_by_task: dict[str, tuple[str, float, int]] = {}
    for (task, method), rows in eval_groups.items():
        values = [safe_float(r["return_mean"]) for r in rows]
        values_clean = [x for x in values if x is not None]
        if not values_clean:
            continue
        mean_return = statistics.fmean(values_clean)
        candidate = (method, mean_return, len(values_clean))
        if task not in best_by_task or candidate[1] > best_by_task[task][1]:
            best_by_task[task] = candidate

    for task in sorted(best_by_task):
        method, best_mean, n = best_by_task[task]
        lines.append(
            f"- {task}: best_completed_mean_return={method} "
            f"({best_mean:.3f}, n={n})"
        )

    args.output_analysis_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_analysis_md.write_text("\n".join(lines) + "\n")

    print(f"Wrote CSV: {args.output_csv}")
    print(f"Wrote analysis: {args.output_analysis_md}")


if __name__ == "__main__":
    main()
