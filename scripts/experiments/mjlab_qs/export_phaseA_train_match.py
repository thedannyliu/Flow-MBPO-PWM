#!/usr/bin/env python3
"""Export MJLab-QS Flow train-loss-match summaries to CSV."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def flatten(prefix: str, value: Any, out: dict[str, Any]) -> None:
    if isinstance(value, dict):
        for k, v in value.items():
            flatten(f"{prefix}{k}.", v, out)
    else:
        out[prefix[:-1]] = value


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--results-root", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--manifest", default=None)
    args = p.parse_args()

    root = Path(args.results_root)
    rows = []
    for path in root.glob("**/train_match_summary.json"):
        raw = json.loads(path.read_text())
        row: dict[str, Any] = {}
        flatten("", raw, row)
        rel = path.relative_to(root)
        parts = rel.parts
        if len(parts) >= 4:
            row.setdefault("task_key", parts[-4])
            row.setdefault("method", parts[-3])
            row.setdefault("seed", parts[-2].replace("seed_", ""))
        row["summary_path"] = str(path)
        rows.append(row)

    if not rows:
        raise RuntimeError(f"No train_match_summary.json files found under {root}")

    if args.manifest:
        expected = []
        with open(args.manifest, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                expected.append((row["task_key"], str(row["seed"])))
        observed = {(str(r.get("task_key", "")), str(r.get("seed", ""))) for r in rows}
        missing = [item for item in expected if item not in observed]
        if missing:
            raise RuntimeError(f"Missing {len(missing)} expected train-match summaries: {missing[:20]}")

    keys = sorted(set().union(*(r.keys() for r in rows)))
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {len(rows)} rows to {out}")


if __name__ == "__main__":
    main()
