#!/usr/bin/env python3
"""Summarize the 2026-05-27 PWM/Flow rerun status from manifests and logs."""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path
from typing import Any


JSON_LINE_RE = re.compile(r"^\{.*\}$")
WANDB_RUN_RE = re.compile(r"/runs/([A-Za-z0-9_-]+)")


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


def wm_rows(manifest: Path, job_id: str) -> list[dict[str, str]]:
    rows = []
    for idx, row in enumerate(read_manifest(manifest)):
        out = wm_output_dir(row)
        summary = out / "summary.json"
        best = out / "best.pt"
        data = load_json(summary) if summary.exists() else {}
        err = Path(f"logs/slurm/mjlab_qs/train/mjqs_train_{job_id}_{idx}.err") if job_id else Path("")
        status = "done" if summary.exists() else "partial" if best.exists() else "missing"
        rows.append(
            {
                "kind": "wm",
                "row": str(idx),
                "method": row["method"],
                "policy": "",
                "seed": row["seed"],
                "status": status,
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


def policy_rows(manifest: Path, job_id: str) -> list[dict[str, str]]:
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
        if summary.exists() and eval_summary.exists() and final.exists():
            status = "done"
        elif best.exists() or iter_value:
            status = "partial"
        else:
            status = "missing"
        rows.append(
            {
                "kind": "policy",
                "row": str(idx),
                "method": row["wm_method"],
                "policy": row.get("policy_type", "mlp"),
                "seed": row["seed"],
                "status": status,
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
        rows.extend(wm_rows(args.wm_manifest, args.wm_job))
    if args.policy_manifest:
        rows.extend(policy_rows(args.policy_manifest, args.policy_job))

    if not rows:
        raise SystemExit("No manifest supplied.")

    fields = [
        "kind",
        "row",
        "method",
        "policy",
        "seed",
        "status",
        "latest_iter",
        "imagined_return",
        "test_h16",
        "eval_return_mean",
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
