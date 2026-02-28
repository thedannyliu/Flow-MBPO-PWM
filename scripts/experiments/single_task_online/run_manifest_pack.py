#!/usr/bin/env python3
"""Run a slice of manifest rows in one Slurm job (single GPU, packed rows)."""

from __future__ import annotations

import argparse
import csv
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple


def _load_rows(manifest_path: Path) -> List[Dict[str, str]]:
    with manifest_path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _build_row_command(
    python_bin: str,
    project_root: Path,
    manifest_path: Path,
    row_index: int,
) -> List[str]:
    return [
        python_bin,
        "scripts/experiments/single_task_online/run_manifest_job.py",
        "--manifest",
        str(manifest_path),
        "--row-index",
        str(row_index),
        "--project-root",
        str(project_root),
        "--python-bin",
        python_bin,
    ]


def _selected_row_indices(total_rows: int, pack_index: int, pack_size: int) -> List[int]:
    start = pack_index * pack_size
    end = min(total_rows, start + pack_size)
    if start >= total_rows:
        return []
    return list(range(start, end))


def _poll_active(
    active: List[Tuple[int, subprocess.Popen]],
    done: List[Tuple[int, int]],
) -> List[Tuple[int, subprocess.Popen]]:
    still_active: List[Tuple[int, subprocess.Popen]] = []
    for row_index, proc in active:
        rc = proc.poll()
        if rc is None:
            still_active.append((row_index, proc))
        else:
            done.append((row_index, int(rc)))
            print(f"[pack] row_index={row_index} finished rc={rc}")
    return still_active


def _terminate_active(active: List[Tuple[int, subprocess.Popen]]) -> None:
    for row_index, proc in active:
        if proc.poll() is not None:
            continue
        print(f"[pack] terminating running row_index={row_index}")
        proc.terminate()
    deadline = time.time() + 20
    while time.time() < deadline:
        if all(proc.poll() is not None for _, proc in active):
            return
        time.sleep(0.5)
    for row_index, proc in active:
        if proc.poll() is None:
            print(f"[pack] killing stuck row_index={row_index}")
            proc.kill()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a packed subset of one single-task-online manifest in one Slurm job."
    )
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--pack-index", required=True, type=int)
    parser.add_argument("--pack-size", required=True, type=int)
    parser.add_argument("--max-parallel", default=2, type=int)
    parser.add_argument("--project-root", default=".", type=Path)
    parser.add_argument("--python-bin", default=sys.executable)
    parser.add_argument("--fail-fast", action="store_true")
    args = parser.parse_args()

    manifest_path = args.manifest.resolve()
    project_root = args.project_root.resolve()
    if args.pack_size <= 0:
        raise ValueError("--pack-size must be > 0")
    if args.max_parallel <= 0:
        raise ValueError("--max-parallel must be > 0")

    rows = _load_rows(manifest_path)
    total_rows = len(rows)
    row_indices = _selected_row_indices(total_rows, args.pack_index, args.pack_size)

    print(
        f"[pack] manifest={manifest_path} total_rows={total_rows} "
        f"pack_index={args.pack_index} pack_size={args.pack_size} "
        f"max_parallel={args.max_parallel} selected_rows={row_indices}"
    )
    print(
        "[pack] runtime env: "
        f"SLURM_JOB_ID={os.environ.get('SLURM_JOB_ID', '')} "
        f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '')}"
    )

    if not row_indices:
        print("[pack] no rows selected for this pack index; exiting cleanly.")
        return

    pending = list(row_indices)
    active: List[Tuple[int, subprocess.Popen]] = []
    done: List[Tuple[int, int]] = []

    while pending or active:
        while pending and len(active) < args.max_parallel:
            row_index = pending.pop(0)
            cmd = _build_row_command(
                python_bin=args.python_bin,
                project_root=project_root,
                manifest_path=manifest_path,
                row_index=row_index,
            )
            print(f"[pack] launching row_index={row_index}: {' '.join(shlex.quote(x) for x in cmd)}")
            proc = subprocess.Popen(cmd, cwd=str(project_root), env=os.environ.copy())
            active.append((row_index, proc))
            # Reduce start-time contention spikes (JIT + env init).
            time.sleep(1.0)

        time.sleep(2.0)
        active = _poll_active(active, done)

        if args.fail_fast and any(rc != 0 for _, rc in done):
            print("[pack] fail-fast triggered, stopping remaining rows.")
            _terminate_active(active)
            break

    done.sort(key=lambda x: x[0])
    failed = [(idx, rc) for idx, rc in done if rc != 0]
    succeeded = [(idx, rc) for idx, rc in done if rc == 0]

    print(f"[pack] summary: succeeded={len(succeeded)} failed={len(failed)}")
    if failed:
        print("[pack] failed rows:")
        for idx, rc in failed:
            print(f"  - row_index={idx}, rc={rc}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
