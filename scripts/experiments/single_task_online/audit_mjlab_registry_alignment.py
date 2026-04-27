#!/usr/bin/env python3
"""Audit local MJLab env configs against the installed mjlab registry."""

from __future__ import annotations

import argparse
import csv
import importlib.util
import os
import re
from pathlib import Path
from typing import Dict, List, Tuple

import yaml


REPO_ROOT = Path(__file__).resolve().parents[3]
ENV_CFG_DIR = REPO_ROOT / "scripts" / "cfg" / "env"


def discover_registered_tasks() -> Tuple[List[str], Path]:
    spec = importlib.util.find_spec("mjlab")
    if spec is None or spec.origin is None:
        raise RuntimeError("Could not locate installed mjlab package.")
    mjlab_pkg = Path(spec.origin).resolve().parent
    task_files = sorted((mjlab_pkg / "tasks").glob("**/__init__.py"))
    registered: List[str] = []
    pattern = re.compile(r'task_id=\"([^\"]+)\"')
    for path in task_files:
        text = path.read_text(encoding="utf-8", errors="ignore")
        for match in pattern.findall(text):
            if match.startswith("Mjlab-"):
                registered.append(match)
    return sorted(set(registered)), mjlab_pkg


def classify(task_id: str, fallbacks: List[str], registered: set[str]) -> Tuple[str, str]:
    if task_id in registered:
        return "exact_resolve", task_id
    for fallback in fallbacks:
        if fallback in registered:
            return "fallback_only", fallback
    return "missing", ""


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit MJLab env configs against installed registry.")
    parser.add_argument("--csv-out", required=True)
    parser.add_argument("--md-out", required=True)
    args = parser.parse_args()

    registered, mjlab_pkg = discover_registered_tasks()
    registered_set = set(registered)

    rows: List[Dict[str, str]] = []
    for cfg_path in sorted(ENV_CFG_DIR.glob("mjlab_*.yaml")):
        data = yaml.safe_load(cfg_path.read_text())
        cfg = data.get("config", {})
        task_id = str(cfg.get("task_id", ""))
        fallbacks = list(cfg.get("task_id_fallbacks") or [])
        status, resolved = classify(task_id, fallbacks, registered_set)
        rows.append(
            {
                "env_cfg": cfg_path.name,
                "task_id": task_id,
                "strict_task_id_match": str(bool(cfg.get("strict_task_id_match", False))).lower(),
                "fallbacks": ",".join(fallbacks),
                "registry_status": status,
                "first_resolvable_task": resolved,
            }
        )

    csv_out = Path(args.csv_out)
    csv_out.parent.mkdir(parents=True, exist_ok=True)
    with csv_out.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "env_cfg",
                "task_id",
                "strict_task_id_match",
                "fallbacks",
                "registry_status",
                "first_resolvable_task",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    exact = [r for r in rows if r["registry_status"] == "exact_resolve"]
    fallback = [r for r in rows if r["registry_status"] == "fallback_only"]
    missing = [r for r in rows if r["registry_status"] == "missing"]

    md_lines = [
        "# MJLab Registry Alignment Audit",
        "",
        f"- installed mjlab package: `{mjlab_pkg}`",
        f"- total registered tasks discovered: `{len(registered)}`",
        f"- repo mjlab env configs audited: `{len(rows)}`",
        "",
        "## Summary",
        "",
        f"- exact_resolve: `{len(exact)}`",
        f"- fallback_only: `{len(fallback)}`",
        f"- missing: `{len(missing)}`",
        "",
        "## Exact Resolve",
        "",
    ]
    for row in exact:
        md_lines.append(f"- `{row['env_cfg']}` -> `{row['task_id']}`")
    md_lines.extend(["", "## Fallback Only", ""])
    for row in fallback:
        md_lines.append(
            f"- `{row['env_cfg']}` -> requested `{row['task_id']}`, first resolvable fallback `{row['first_resolvable_task']}`"
        )
    md_lines.extend(["", "## Missing", ""])
    for row in missing:
        md_lines.append(f"- `{row['env_cfg']}` -> requested `{row['task_id']}`")

    md_out = Path(args.md_out)
    md_out.parent.mkdir(parents=True, exist_ok=True)
    md_out.write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    print(f"Wrote CSV audit to {csv_out}")
    print(f"Wrote Markdown audit to {md_out}")


if __name__ == "__main__":
    main()
