#!/usr/bin/env python3
"""Build per-task MJLab-QS window datasets from a collection manifest."""

from __future__ import annotations

import argparse
import csv
import subprocess
from collections import defaultdict
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--collection-manifest", required=True)
    p.add_argument("--mode", required=True)
    p.add_argument("--root", default="scripts/outputs/mjlab_qs")
    p.add_argument("--python-bin", default="python")
    p.add_argument("--horizon", type=int, default=16)
    p.add_argument("--stride", type=int, default=4)
    p.add_argument("--min-train-episodes-per-bucket", type=int, default=50)
    p.add_argument("--min-valid-train-windows-per-bucket", type=int, default=10000)
    p.add_argument("--allow-preliminary", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    rows = list(csv.DictReader(open(args.collection_manifest, newline="", encoding="utf-8")))
    by_task = defaultdict(list)
    for row in rows:
        by_task[row["task_key"]].append(row["output"])
    if not by_task:
        raise RuntimeError(f"No rows in collection manifest: {args.collection_manifest}")

    for task_key, raw_paths in sorted(by_task.items()):
        out = Path(args.root) / "windows" / args.mode / task_key / "d_qs_core_h16.pt"
        out.parent.mkdir(parents=True, exist_ok=True)
        cmd = [
            args.python_bin,
            "scripts/experiments/mjlab_qs/build_mjlab_qs_windows.py",
            "--raw",
            *raw_paths,
            "--output",
            str(out),
            "--metadata-output",
            str(out.with_suffix(".json")),
            "--normalization-output",
            str(out.with_name(out.stem + "_normalization.json")),
            "--report-output",
            str(out.with_name(out.stem + "_report.md")),
            "--horizon",
            str(args.horizon),
            "--stride",
            str(args.stride),
            "--min-train-episodes-per-bucket",
            str(args.min_train_episodes_per_bucket),
            "--min-valid-train-windows-per-bucket",
            str(args.min_valid_train_windows_per_bucket),
        ]
        if args.allow_preliminary:
            cmd.append("--allow-preliminary")
        print(" ".join(cmd), flush=True)
        subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
