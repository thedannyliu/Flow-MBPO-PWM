#!/usr/bin/env python3
"""Audit MJLab-QS window datasets by split and quality bin."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import torch


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True)
    p.add_argument("--metadata", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--split", default="train", help="Comma-separated split names or ids, or 'all'.")
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


def split_ids_from_arg(raw: str, split_id_map: dict[str, int]) -> set[int]:
    if raw.strip().lower() == "all":
        return set(split_id_map.values())
    out: set[int] = set()
    for item in [x.strip() for x in raw.split(",") if x.strip()]:
        if item in split_id_map:
            out.add(int(split_id_map[item]))
        else:
            out.add(int(item))
    if not out:
        raise ValueError("--split selected no split ids")
    return out


def safe_quantiles(x: torch.Tensor) -> dict[str, float]:
    if x.numel() == 0:
        return {"p10": float("nan"), "p50": float("nan"), "p90": float("nan")}
    q = torch.quantile(x.float(), torch.tensor([0.10, 0.50, 0.90], device=x.device))
    return {"p10": float(q[0].item()), "p50": float(q[1].item()), "p90": float(q[2].item())}


def quality_stats(data: dict[str, Any], indices: torch.Tensor, quality_name: str, split_name: str) -> dict[str, Any]:
    command = data["command"][indices].float()
    action = data["policy_action"][indices].float()
    reward = data["reward"][indices].float()
    done = data["done"][indices].bool()
    termination = data["termination"][indices].bool()
    truncation = data["truncation"][indices].bool()
    window_reward_sum = reward.sum(dim=1)
    action_norm = action.pow(2).mean(dim=-1).sqrt()
    action_rate = action[:, 1:] - action[:, :-1] if action.shape[1] > 1 else torch.empty(0)
    action_rate_norm = action_rate.pow(2).mean(dim=-1).sqrt() if action_rate.numel() else torch.empty(0)
    window_action_norm_mean = action_norm.mean(dim=1)
    window_action_rate_norm_mean = action_rate_norm.mean(dim=1) if action_rate_norm.numel() else torch.empty(0)
    terminal_window = done.any(dim=1)
    fall_window = termination.any(dim=1)
    trunc_window = truncation.any(dim=1)
    yaw_abs_max = command[:, :, 2].abs().amax(dim=1) if command.shape[-1] >= 3 else torch.empty(0)

    row: dict[str, Any] = {
        "quality": quality_name,
        "split": split_name,
        "windows": int(indices.numel()),
        "terminal_window_rate": float(terminal_window.float().mean().item()),
        "fall_window_rate": float(fall_window.float().mean().item()),
        "truncation_window_rate": float(trunc_window.float().mean().item()),
        "window_reward_sum_mean": float(window_reward_sum.mean().item()),
        "window_reward_sum_std": float(window_reward_sum.std(unbiased=False).item()),
        "action_norm_mean": float(action_norm.mean().item()),
        "action_norm_std": float(action_norm.std(unbiased=False).item()),
        "action_rate_norm_mean": float(action_rate_norm.mean().item()) if action_rate_norm.numel() else float("nan"),
    }
    row.update({f"window_reward_sum_{k}": v for k, v in safe_quantiles(window_reward_sum).items()})
    row.update({f"window_action_norm_mean_{k}": v for k, v in safe_quantiles(window_action_norm_mean).items()})
    row.update({f"window_action_rate_norm_mean_{k}": v for k, v in safe_quantiles(window_action_rate_norm_mean).items()})
    if command.shape[-1] >= 1:
        row["command_0_mean"] = float(command[:, :, 0].mean().item())
        row["command_0_abs_max_mean"] = float(command[:, :, 0].abs().amax(dim=1).mean().item())
    if command.shape[-1] >= 2:
        row["command_1_mean"] = float(command[:, :, 1].mean().item())
        row["command_1_abs_max_mean"] = float(command[:, :, 1].abs().amax(dim=1).mean().item())
    if command.shape[-1] >= 3:
        row["command_2_mean"] = float(command[:, :, 2].mean().item())
        row["command_2_abs_max_mean"] = float(yaw_abs_max.mean().item())
        row.update({f"command_2_abs_max_{k}": v for k, v in safe_quantiles(yaw_abs_max).items()})
    return row


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def format_float(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def write_markdown(path: Path, rows: list[dict[str, Any]], summary: dict[str, Any]) -> None:
    cols = [
        "quality",
        "split",
        "windows",
        "terminal_window_rate",
        "fall_window_rate",
        "truncation_window_rate",
        "window_reward_sum_mean",
        "action_norm_mean",
        "window_action_norm_mean_p90",
        "action_rate_norm_mean",
        "window_action_rate_norm_mean_p90",
        "command_2_abs_max_mean",
    ]
    lines = [
        "# MJLab-QS Window Audit",
        "",
        f"- dataset: `{summary['dataset']}`",
        f"- split_ids: `{summary['selected_split_ids']}`",
        f"- git_sha: `{summary['git_sha']}`",
        "",
        "| " + " | ".join(cols) + " |",
        "| " + " | ".join("---" for _ in cols) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(format_float(row.get(col, "")) for col in cols) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata = json.loads(Path(args.metadata).read_text(encoding="utf-8"))
    split_id_map = {str(k): int(v) for k, v in metadata.get("split_id_map", {"train": 0, "val": 1, "test": 2}).items()}
    inverse_split = {v: k for k, v in split_id_map.items()}
    quality_id_map = {str(k): int(v) for k, v in metadata["quality_id_map"].items()}
    inverse_quality = {v: k for k, v in quality_id_map.items()}
    selected_split_ids = split_ids_from_arg(args.split, split_id_map)

    data = torch.load(args.dataset, map_location="cpu", weights_only=False)
    split_ids = data["split_id"].long()
    quality_ids = data["quality_bin_id"].long()
    rows: list[dict[str, Any]] = []
    for split_id in sorted(selected_split_ids):
        split_mask = split_ids == split_id
        for quality_id in sorted(torch.unique(quality_ids[split_mask]).tolist()):
            mask = split_mask & (quality_ids == int(quality_id))
            indices = mask.nonzero(as_tuple=False).squeeze(-1)
            if indices.numel() == 0:
                continue
            rows.append(quality_stats(data, indices, inverse_quality.get(int(quality_id), str(quality_id)), inverse_split.get(split_id, str(split_id))))

    summary = {
        "dataset": args.dataset,
        "metadata": args.metadata,
        "selected_split_ids": sorted(selected_split_ids),
        "num_rows": len(rows),
        "git_sha": git_sha(),
        "git_branch": git_branch(),
        "command": command_line(),
    }
    (output_dir / "window_audit_summary.json").write_text(json.dumps({"summary": summary, "rows": rows}, indent=2), encoding="utf-8")
    write_csv(output_dir / "window_audit_by_quality.csv", rows)
    write_markdown(output_dir / "window_audit_by_quality.md", rows, summary)
    print(json.dumps(summary, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
