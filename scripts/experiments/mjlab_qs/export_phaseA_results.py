#!/usr/bin/env python3
"""Export MJLab-QS Phase-A summaries."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--results-root", default="scripts/outputs/mjlab_qs/results/a25")
    p.add_argument("--output", required=True)
    p.add_argument("--train-manifest", default=None)
    args = p.parse_args()
    rows = []
    root = Path(args.results_root)
    for path in root.glob("**/summary.json"):
        data = json.loads(path.read_text())
        rel = path.relative_to(root)
        parts = rel.parts
        if len(parts) >= 4:
            data.setdefault("task_key", parts[-4])
            data.setdefault("method", parts[-3])
            data.setdefault("seed", parts[-2].replace("seed_", ""))
        data["summary_path"] = str(path)
        rows.append(data)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise RuntimeError(f"No summaries under {args.results_root}")
    if args.train_manifest:
        expected = []
        with open(args.train_manifest, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                expected.append((row["task_key"], row["method"], str(row["seed"])))
        observed = {(str(r.get("task_key", "")), str(r.get("method", "")), str(r.get("seed", ""))) for r in rows}
        missing = [item for item in expected if item not in observed]
        if missing:
            raise RuntimeError(f"Missing {len(missing)} expected summaries: {missing[:20]}")
    keys = sorted(set().union(*(r.keys() for r in rows)))
    with open(args.output, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {len(rows)} rows to {args.output}")


if __name__ == "__main__":
    main()
