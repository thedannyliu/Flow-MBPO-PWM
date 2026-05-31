#!/usr/bin/env python3
"""Audit raw MJLab-QS done/fall labels without loading the full window tensor."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.experiments.mjlab_qs.build_mjlab_qs_windows import valid_starts  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--raw", nargs="+", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--horizon", type=int, default=16)
    p.add_argument("--stride", type=int, default=4)
    return p.parse_args()


def git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def git_branch() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--abbrev-ref", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def command_line() -> str:
    return " ".join([sys.executable, *sys.argv])


def safe_div(num: int | float, den: int | float) -> float:
    return float(num) / float(den) if den else 0.0


def audit_raw(path: Path, horizon: int, stride: int) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    episodes = payload.get("episodes", [])
    if not episodes:
        raise ValueError(f"{path} contains no episodes")
    quality = str(episodes[0].get("quality_bin", ""))
    task = str(episodes[0].get("task_id_resolved", ""))
    transition_count = 0
    done_transitions = 0
    termination_transitions = 0
    truncation_transitions = 0
    fall_episodes = 0
    timeout_episodes = 0
    valid_window_count = 0
    terminal_windows = 0
    fall_windows = 0
    truncation_windows = 0
    terminal_last_step_windows = 0
    fall_last_step_windows = 0
    truncation_last_step_windows = 0

    for ep in episodes:
        done = ep["done"].bool()
        termination = ep["termination"].bool()
        truncation = ep["truncation"].bool()
        trans_done = done[1:]
        trans_termination = termination[1:]
        trans_truncation = truncation[1:]
        transition_count += int(trans_done.numel())
        done_transitions += int(trans_done.sum().item())
        termination_transitions += int(trans_termination.sum().item())
        truncation_transitions += int(trans_truncation.sum().item())
        fall_episodes += int(bool(trans_termination.any().item()))
        timeout_episodes += int(bool(trans_truncation.any().item()))
        for start in valid_starts(done, horizon, stride):
            trans = slice(start, start + horizon)
            window_done = trans_done[trans]
            window_termination = trans_termination[trans]
            window_truncation = trans_truncation[trans]
            valid_window_count += 1
            terminal_windows += int(bool(window_done.any().item()))
            fall_windows += int(bool(window_termination.any().item()))
            truncation_windows += int(bool(window_truncation.any().item()))
            terminal_last_step_windows += int(bool(window_done[-1].item()))
            fall_last_step_windows += int(bool(window_termination[-1].item()))
            truncation_last_step_windows += int(bool(window_truncation[-1].item()))

    return {
        "raw_path": str(path),
        "quality": quality,
        "task": task,
        "episodes": len(episodes),
        "transitions": transition_count,
        "done_transition_count": done_transitions,
        "termination_transition_count": termination_transitions,
        "truncation_transition_count": truncation_transitions,
        "done_transition_rate": safe_div(done_transitions, transition_count),
        "termination_transition_rate": safe_div(termination_transitions, transition_count),
        "truncation_transition_rate": safe_div(truncation_transitions, transition_count),
        "fall_episode_count": fall_episodes,
        "timeout_episode_count": timeout_episodes,
        "fall_episode_rate": safe_div(fall_episodes, len(episodes)),
        "timeout_episode_rate": safe_div(timeout_episodes, len(episodes)),
        "valid_windows": valid_window_count,
        "terminal_window_count": terminal_windows,
        "fall_window_count": fall_windows,
        "truncation_window_count": truncation_windows,
        "terminal_window_rate": safe_div(terminal_windows, valid_window_count),
        "fall_window_rate": safe_div(fall_windows, valid_window_count),
        "truncation_window_rate": safe_div(truncation_windows, valid_window_count),
        "terminal_last_step_window_rate": safe_div(terminal_last_step_windows, valid_window_count),
        "fall_last_step_window_rate": safe_div(fall_last_step_windows, valid_window_count),
        "truncation_last_step_window_rate": safe_div(truncation_last_step_windows, valid_window_count),
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_md(path: Path, rows: list[dict[str, Any]], summary: dict[str, Any]) -> None:
    cols = [
        "quality",
        "episodes",
        "fall_episode_rate",
        "timeout_episode_rate",
        "valid_windows",
        "terminal_window_rate",
        "fall_window_rate",
        "terminal_last_step_window_rate",
        "fall_last_step_window_rate",
    ]
    lines = [
        "# MJLab-QS Done/Fall Label Audit",
        "",
        f"- horizon: `{summary['horizon']}`",
        f"- stride: `{summary['stride']}`",
        f"- git_sha: `{summary['git_sha']}`",
        "",
        "| " + " | ".join(cols) + " |",
        "| " + " | ".join("---" for _ in cols) + " |",
    ]
    for row in rows:
        cells = []
        for col in cols:
            val = row.get(col, "")
            cells.append(f"{val:.6g}" if isinstance(val, float) else str(val))
        lines.append("| " + " | ".join(cells) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = [audit_raw(Path(raw), int(args.horizon), int(args.stride)) for raw in args.raw]
    summary = {
        "git_sha": git_sha(),
        "git_branch": git_branch(),
        "command": command_line(),
        "raw_paths": [str(p) for p in args.raw],
        "output_dir": str(output_dir),
        "horizon": int(args.horizon),
        "stride": int(args.stride),
    }
    (output_dir / "done_fall_label_audit.json").write_text(json.dumps({"summary": summary, "rows": rows}, indent=2), encoding="utf-8")
    write_csv(output_dir / "done_fall_label_audit.csv", rows)
    write_md(output_dir / "done_fall_label_audit.md", rows, summary)
    print(json.dumps({"summary": summary, "rows": rows}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
