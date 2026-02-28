#!/usr/bin/env python3
"""Split a single-task-online manifest into fixed PACE-ICE/Phoenix manifests.

Design constraints:
- A task is pinned to exactly one cluster.
- No cross-cluster resume/checkpoint sharing is required.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List


# High-risk / fragile tasks stay on ICE (H100/H200).
PACE_ICE_TASKS = {
    "humanoid",
    "snu_humanoid",
    "velocity_flat_unitree_g1",
    "tracking_flat_unitree_g1",
    "leap_left_handcube_rotate",
}

# Throughput-friendly tasks run on Phoenix (L40S).
PACE_PHOENIX_TASKS = {
    "hopper",
    "ant",
    "anymal",
    "velocity_flat_unitree_go2",
}


def _cluster_for_task(task_key: str) -> str:
    if task_key in PACE_ICE_TASKS:
        return "pace_ice"
    if task_key in PACE_PHOENIX_TASKS:
        return "pace_phoenix"
    raise KeyError(
        f"Task '{task_key}' is not assigned to any cluster. "
        "Update PACE_ICE_TASKS/PACE_PHOENIX_TASKS."
    )


def _write_rows(path: Path, fieldnames: List[str], rows: List[Dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Split one manifest into fixed PACE-ICE/Phoenix manifests."
    )
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument(
        "--ice-output",
        type=Path,
        default=None,
        help="Output path for PACE-ICE manifest. Default: <input>_pace_ice.csv",
    )
    parser.add_argument(
        "--phoenix-output",
        type=Path,
        default=None,
        help="Output path for PACE-Phoenix manifest. Default: <input>_pace_phoenix.csv",
    )
    args = parser.parse_args()

    manifest_path = args.manifest.resolve()
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    with manifest_path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
        if not rows:
            raise RuntimeError(f"Manifest has no rows: {manifest_path}")
        fieldnames = list(rows[0].keys())

    stem = manifest_path.stem
    default_ice = manifest_path.with_name(f"{stem}_pace_ice.csv")
    default_phoenix = manifest_path.with_name(f"{stem}_pace_phoenix.csv")
    ice_output = (args.ice_output or default_ice).resolve()
    phoenix_output = (args.phoenix_output or default_phoenix).resolve()

    ice_rows: List[Dict[str, str]] = []
    phoenix_rows: List[Dict[str, str]] = []
    for row in rows:
        task_key = row.get("task_key", "").strip()
        cluster = _cluster_for_task(task_key)
        if cluster == "pace_ice":
            ice_rows.append(row)
        else:
            phoenix_rows.append(row)

    _write_rows(ice_output, fieldnames, ice_rows)
    _write_rows(phoenix_output, fieldnames, phoenix_rows)

    print(f"Input rows: {len(rows)}")
    print(f"PACE-ICE rows: {len(ice_rows)} -> {ice_output}")
    print(f"PACE-Phoenix rows: {len(phoenix_rows)} -> {phoenix_output}")
    print("Task mapping:")
    print(f"  pace_ice: {sorted(PACE_ICE_TASKS)}")
    print(f"  pace_phoenix: {sorted(PACE_PHOENIX_TASKS)}")


if __name__ == "__main__":
    main()
