#!/usr/bin/env python3
"""Export NEWT/LeWM flow 2x2 Slurm metrics to CSV."""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path


ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")
FLOAT_RE = re.compile(r"[-+]?(?:\d{1,3}(?:,\d{3})+|\d+\.\d+|\d+)(?:[eE][-+]?\d+)?")
ARCH_RE = re.compile(r"architecture=(\w+)")


def clean(text: str) -> str:
    return ANSI_RE.sub("", text)


def read_manifest(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def first_float(text: str) -> str:
    match = FLOAT_RE.search(text)
    return match.group(0).replace(",", "") if match else ""


def parse_newt_log(path: Path) -> dict[str, str]:
    metrics: dict[str, str] = {}
    if not path.exists():
        return metrics
    for raw in path.read_text(errors="ignore").splitlines():
        line = clean(raw)
        if "[flow_2x2]" in line:
            arch_match = ARCH_RE.search(line)
            if "world-model architecture" in line:
                metrics["printed_wm_arch"] = arch_match.group(1) if arch_match else ""
            if "policy architecture" in line:
                metrics["printed_policy_arch"] = arch_match.group(1) if arch_match else ""
        if line.strip().startswith("eval"):
            metrics["initial_eval_return"] = first_float(line.split(" R:", 1)[-1]) if " R:" in line else ""
            metrics["initial_eval_success"] = first_float(line.split(" S:", 1)[-1]) if " S:" in line else ""
        if line.strip().startswith("train"):
            metrics["final_train_return"] = first_float(line.split(" R:", 1)[-1]) if " R:" in line else ""
            metrics["final_train_success"] = first_float(line.split(" S:", 1)[-1]) if " S:" in line else ""
            metrics["final_train_step"] = first_float(line.split(" I:", 1)[-1]) if " I:" in line else ""
        if "Training completed successfully" in line:
            metrics["completed_marker"] = "1"
    return metrics


def parse_lewm_log(path: Path) -> dict[str, str]:
    metrics: dict[str, str] = {}
    if not path.exists():
        return metrics
    for raw in path.read_text(errors="ignore").splitlines():
        line = clean(raw)
        if "|" not in line:
            continue
        parts = [p.strip() for p in line.strip().strip("|").split("|")]
        if len(parts) != 2:
            continue
        key, value = parts
        key = key.replace("/", "_")
        if key in {
            "fit_loss",
            "fit_pred_loss",
            "fit_sigreg_loss",
            "validate_loss",
            "validate_loss_epoch",
            "validate_pred_loss",
            "validate_pred_loss_epoch",
            "validate_sigreg_loss",
            "validate_sigreg_loss_epoch",
        }:
            metrics[key] = first_float(value)
    return metrics


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--kind", choices=["newt", "lewm"], required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--log-dir", type=Path, default=Path("logs/slurm/image_official"))
    parser.add_argument("--run-label", required=True)
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    rows = []
    for row in read_manifest(args.manifest):
        task_id = row["row"]
        log_path = args.log_dir / f"{args.run_label}_{args.job_id}_{task_id}.out"
        metrics = parse_newt_log(log_path) if args.kind == "newt" else parse_lewm_log(log_path)
        err_path = args.log_dir / f"{args.run_label}_{args.job_id}_{task_id}.err"
        status = "done" if metrics else "missing"
        rows.append(
            {
                **row,
                "status": status,
                "stdout": str(log_path),
                "stderr": str(err_path),
                **metrics,
            }
        )
    write_csv(args.output, rows)
    print(f"wrote {len(rows)} rows to {args.output}")


if __name__ == "__main__":
    main()
