#!/usr/bin/env python3
"""Summarize the 2026-05-27 PWM/Flow rerun status from manifests and logs."""

from __future__ import annotations

import argparse
import csv
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any


JSON_LINE_RE = re.compile(r"^\{.*\}$")
WANDB_RUN_RE = re.compile(r"/runs/([A-Za-z0-9_-]+)")
FAILED_STATES = {
    "BOOT_FAIL",
    "CANCELLED",
    "DEADLINE",
    "FAILED",
    "NODE_FAIL",
    "OUT_OF_MEMORY",
    "PREEMPTED",
    "TIMEOUT",
}


def read_manifest(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def load_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def latest_iter(path: Path) -> tuple[str, str]:
    if not path.exists():
        return "", ""
    last_iter = ""
    last_return = ""
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not JSON_LINE_RE.match(line):
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if "iter" in payload:
            last_iter = str(payload["iter"])
        if "train/imagined_return" in payload:
            last_return = str(payload["train/imagined_return"])
    return last_iter, last_return


def wandb_run(path: Path) -> str:
    if not path.exists():
        return ""
    text = path.read_text(encoding="utf-8", errors="replace")
    matches = WANDB_RUN_RE.findall(text)
    if matches:
        return matches[-1]
    for line in text.splitlines():
        if "Run data is saved locally" in line and "run-" in line:
            return line.rsplit("-", 1)[-1].strip()
    return ""


def expand_array_suffix(suffix: str) -> list[int]:
    if not suffix.startswith("[") or not suffix.endswith("]"):
        return [int(suffix)]
    body = suffix[1:-1].split("%", 1)[0]
    indices: list[int] = []
    for chunk in body.split(","):
        if "-" in chunk:
            start, end = chunk.split("-", 1)
            indices.extend(range(int(start), int(end) + 1))
        else:
            indices.append(int(chunk))
    return indices


def sacct_state_map(job_id: str) -> dict[int, dict[str, str]]:
    if not job_id:
        return {}
    try:
        result = subprocess.run(
            [
                "sacct",
                "-j",
                job_id,
                "--format=JobID,State,QOS",
                "-P",
                "--noheader",
            ],
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError:
        return {}
    if result.returncode != 0:
        return {}
    states: dict[int, dict[str, str]] = {}
    prefix = f"{job_id}_"
    for line in result.stdout.splitlines():
        parts = line.split("|")
        if len(parts) < 3:
            continue
        raw_job, state, qos = parts[:3]
        if not raw_job.startswith(prefix) or "." in raw_job:
            continue
        suffix = raw_job[len(prefix) :]
        try:
            indices = expand_array_suffix(suffix)
        except ValueError:
            continue
        for idx in indices:
            states[idx] = {"slurm_state": state, "qos": qos}
    return states


def row_status(done: bool, partial: bool, slurm_state: str) -> str:
    base_state = slurm_state.split()[0] if slurm_state else ""
    if done:
        return "done"
    if base_state == "RUNNING":
        return "running"
    if base_state == "PENDING":
        return "pending"
    if base_state in FAILED_STATES:
        return "failed"
    if base_state == "COMPLETED":
        return "completed_missing"
    if partial:
        return "partial"
    return "missing"


def progress_fraction(latest: str, expected: str, done: bool) -> str:
    if done:
        return "1.0"
    if not latest or not expected:
        return ""
    try:
        denominator = float(expected)
        if denominator <= 0:
            return ""
        return str(float(latest) / denominator)
    except ValueError:
        return ""


def wm_output_dir(row: dict[str, str]) -> Path:
    return (
        Path("scripts/outputs/mjlab_qs/results")
        / row["stage"]
        / row.get("task_key", "task_unknown")
        / row["method"]
        / f"seed_{row['seed']}"
    )


def policy_output_dir(row: dict[str, str]) -> Path:
    return (
        Path("scripts/outputs/mjlab_qs/policy_extraction")
        / row["stage"]
        / row.get("task_key", "task_unknown")
        / row["wm_method"]
        / row.get("policy_type", "mlp")
        / row.get("online_profile", "offline")
        / row["compute_profile"]
        / f"seed_{row['seed']}"
    )


def wm_rows(manifest: Path, job_id: str, slurm: dict[int, dict[str, str]]) -> list[dict[str, str]]:
    rows = []
    for idx, row in enumerate(read_manifest(manifest)):
        out = wm_output_dir(row)
        summary = out / "summary.json"
        best = out / "best.pt"
        data = load_json(summary) if summary.exists() else {}
        err = Path(f"logs/slurm/mjlab_qs/train/mjqs_train_{job_id}_{idx}.err") if job_id else Path("")
        slurm_info = slurm.get(idx, {})
        status = row_status(summary.exists(), best.exists(), slurm_info.get("slurm_state", ""))
        rows.append(
            {
                "kind": "wm",
                "row": str(idx),
                "method": row["method"],
                "policy": "",
                "seed": row["seed"],
                "status": status,
                "slurm_state": slurm_info.get("slurm_state", ""),
                "qos": slurm_info.get("qos", ""),
                "expected_iters": row.get("train_iters", ""),
                "progress_fraction": progress_fraction("", row.get("train_iters", ""), summary.exists()),
                "wandb_project": row.get("wandb_project", ""),
                "disable_wandb": row.get("disable_wandb", ""),
                "summary": str(summary) if summary.exists() else "",
                "best": str(best) if best.exists() else "",
                "test_h16": str(data.get("test/rollout_dyn_mse_H16", "")),
                "eval_return_mean": "",
                "latest_iter": "",
                "imagined_return": "",
                "wandb_run": wandb_run(err),
            }
        )
    return rows


def policy_rows(manifest: Path, job_id: str, slurm: dict[int, dict[str, str]]) -> list[dict[str, str]]:
    rows = []
    for idx, row in enumerate(read_manifest(manifest)):
        out = policy_output_dir(row)
        summary = out / "summary.json"
        eval_summary = out / "eval_summary.json"
        final = out / "final_policy_extraction.pt"
        best = out / "best_policy_extraction.pt"
        summary_data = load_json(summary) if summary.exists() else {}
        eval_data = load_json(eval_summary) if eval_summary.exists() else {}
        stdout = Path(f"logs/slurm/mjlab_qs/policy_extract/mjqs_policy_extract_{job_id}_{idx}.out") if job_id else Path("")
        stderr = Path(f"logs/slurm/mjlab_qs/policy_extract/mjqs_policy_extract_{job_id}_{idx}.err") if job_id else Path("")
        iter_value, imagined_return = latest_iter(stdout)
        done = summary.exists() and eval_summary.exists() and final.exists()
        partial = best.exists() or bool(iter_value)
        slurm_info = slurm.get(idx, {})
        status = row_status(done, partial, slurm_info.get("slurm_state", ""))
        expected_iters = row.get("policy_iters", "")
        rows.append(
            {
                "kind": "policy",
                "row": str(idx),
                "method": row["wm_method"],
                "policy": row.get("policy_type", "mlp"),
                "seed": row["seed"],
                "status": status,
                "slurm_state": slurm_info.get("slurm_state", ""),
                "qos": slurm_info.get("qos", ""),
                "expected_iters": expected_iters,
                "progress_fraction": progress_fraction(iter_value, expected_iters, done),
                "wandb_project": row.get("wandb_project", ""),
                "disable_wandb": row.get("disable_wandb", ""),
                "summary": str(summary) if summary.exists() else "",
                "best": str(best) if best.exists() else "",
                "test_h16": "",
                "eval_return_mean": str(
                    eval_data.get("return_mean", summary_data.get("eval/return_mean", ""))
                ),
                "latest_iter": iter_value,
                "imagined_return": imagined_return,
                "wandb_run": wandb_run(stderr),
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--wm-manifest", type=Path)
    parser.add_argument("--policy-manifest", type=Path)
    parser.add_argument("--wm-job", default="")
    parser.add_argument("--policy-job", default="")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    rows: list[dict[str, str]] = []
    if args.wm_manifest:
        rows.extend(wm_rows(args.wm_manifest, args.wm_job, sacct_state_map(args.wm_job)))
    if args.policy_manifest:
        rows.extend(
            policy_rows(args.policy_manifest, args.policy_job, sacct_state_map(args.policy_job))
        )

    if not rows:
        raise SystemExit("No manifest supplied.")

    fields = [
        "kind",
        "row",
        "method",
        "policy",
        "seed",
        "status",
        "slurm_state",
        "qos",
        "expected_iters",
        "progress_fraction",
        "latest_iter",
        "imagined_return",
        "test_h16",
        "eval_return_mean",
        "wandb_project",
        "disable_wandb",
        "wandb_run",
        "summary",
        "best",
    ]
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        print(f"wrote {len(rows)} rows to {args.output}")
    else:
        writer = csv.DictWriter(sys.stdout, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
