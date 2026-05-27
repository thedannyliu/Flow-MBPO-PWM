#!/usr/bin/env python3
"""Write a manifest containing selected rows from another manifest."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def parse_row_spec(spec: str) -> set[int]:
    rows: set[int] = set()
    if not spec:
        return rows
    for chunk in spec.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        if "-" in chunk:
            start, end = chunk.split("-", 1)
            rows.update(range(int(start), int(end) + 1))
        else:
            rows.add(int(chunk))
    return rows


def rows_from_status(path: Path, statuses: set[str]) -> set[int]:
    selected: set[int] = set()
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row.get("status", "") in statuses:
                selected.add(int(row["row"]))
    return selected


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rows", default="", help="Comma/range row ids, e.g. 0,2,5-7")
    parser.add_argument("--status-csv", type=Path)
    parser.add_argument(
        "--statuses",
        default="failed,completed_missing",
        help="Comma-separated statuses to select from --status-csv",
    )
    args = parser.parse_args()

    selected = parse_row_spec(args.rows)
    if args.status_csv:
        statuses = {value.strip() for value in args.statuses.split(",") if value.strip()}
        selected.update(rows_from_status(args.status_csv, statuses))
    if not selected:
        raise SystemExit("No rows selected.")

    with args.manifest.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        if fieldnames is None:
            raise SystemExit(f"Manifest has no header: {args.manifest}")
        rows = [row for idx, row in enumerate(reader) if idx in selected]

    if not rows:
        raise SystemExit(f"Selected row ids not present in manifest: {sorted(selected)}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} rows to {args.output}: {sorted(selected)}")


if __name__ == "__main__":
    main()
