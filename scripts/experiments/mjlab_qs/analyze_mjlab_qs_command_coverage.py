#!/usr/bin/env python3
"""Analyze MJLab-QS command-conditioned window coverage."""

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
    p.add_argument("--split", default="train")
    p.add_argument("--qualities", default="expert,expert_noisy")
    p.add_argument("--yaw-edges", default="0.175,0.35,0.525")
    p.add_argument("--chunk-size", type=int, default=32768)
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


def parse_float_list(raw: str) -> list[float]:
    return [float(item.strip()) for item in raw.split(",") if item.strip()]


def parse_name_list(raw: str) -> list[str]:
    return [item.strip() for item in raw.split(",") if item.strip()]


def bin_labels(edges: list[float]) -> list[str]:
    labels = [f"[0,{edges[0]:g})"]
    labels.extend(f"[{lo:g},{hi:g})" for lo, hi in zip(edges[:-1], edges[1:]))
    labels.append(f"[{edges[-1]:g},inf)")
    return labels


def selected_split_ids(raw: str, split_id_map: dict[str, int]) -> set[int]:
    if raw == "all":
        return set(split_id_map.values())
    out: set[int] = set()
    for item in parse_name_list(raw):
        out.add(split_id_map[item] if item in split_id_map else int(item))
    return out


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


def fmt(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def write_markdown(path: Path, rows: list[dict[str, Any]], summary: dict[str, Any]) -> None:
    cols = [
        "quality",
        "yaw_bin",
        "windows",
        "quality_fraction",
        "reward_sum_mean",
        "action_norm_mean",
        "first_action_l2_mean",
        "action_rate_norm_mean",
        "positive_yaw_fraction",
    ]
    lines = [
        "# MJLab-QS Command Coverage",
        "",
        f"- dataset: `{summary['dataset']}`",
        f"- split: `{summary['split']}`",
        f"- qualities: `{summary['qualities']}`",
        f"- yaw_edges: `{summary['yaw_edges']}`",
        f"- git_sha: `{summary['git_sha']}`",
        "",
        "| " + " | ".join(cols) + " |",
        "| " + " | ".join("---" for _ in cols) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(fmt(row.get(col, "")) for col in cols) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    metadata = json.loads(Path(args.metadata).read_text(encoding="utf-8"))
    split_id_map = {str(k): int(v) for k, v in metadata.get("split_id_map", {"train": 0}).items()}
    quality_id_map = {str(k): int(v) for k, v in metadata["quality_id_map"].items()}
    inverse_quality = {v: k for k, v in quality_id_map.items()}
    split_ids = selected_split_ids(args.split, split_id_map)
    quality_names = parse_name_list(args.qualities)
    quality_ids = {quality_id_map[name] for name in quality_names}
    edges = parse_float_list(args.yaw_edges)
    edge_tensor = torch.tensor(edges, dtype=torch.float32)
    labels = bin_labels(edges)

    data = torch.load(args.dataset, map_location="cpu", weights_only=False, mmap=True)
    split = data["split_id"].long()
    quality = data["quality_bin_id"].long()
    mask = torch.zeros(split.shape, dtype=torch.bool)
    for split_id in split_ids:
        mask |= split == int(split_id)
    qmask = torch.zeros(quality.shape, dtype=torch.bool)
    for quality_id in quality_ids:
        qmask |= quality == int(quality_id)
    indices = (mask & qmask).nonzero(as_tuple=False).squeeze(-1)

    accum: dict[tuple[int, int], dict[str, float]] = {}
    for qid in quality_ids:
        for bid in range(len(labels)):
            accum[(qid, bid)] = {
                "windows": 0.0,
                "reward_sum": 0.0,
                "action_norm": 0.0,
                "first_action_l2": 0.0,
                "action_rate_norm": 0.0,
                "positive_yaw": 0.0,
            }

    for start in range(0, int(indices.numel()), args.chunk_size):
        ids = indices[start : start + args.chunk_size]
        command = data["command"][ids].float()
        action = data["policy_action"][ids].float()
        reward = data["reward"][ids].float()
        qids = quality[ids]
        yaw_seq = command[:, :, 2]
        yaw_abs_max = yaw_seq.abs().amax(dim=1)
        yaw_sign = yaw_seq.mean(dim=1) >= 0
        bin_ids = torch.bucketize(yaw_abs_max, edge_tensor)
        reward_sum = reward.sum(dim=1)
        action_l2 = action.pow(2).mean(dim=-1).sqrt()
        action_norm = action_l2.mean(dim=1)
        first_action_l2 = action_l2[:, 0]
        if action.shape[1] > 1:
            action_rate_norm = (action[:, 1:] - action[:, :-1]).pow(2).mean(dim=-1).sqrt().mean(dim=1)
        else:
            action_rate_norm = torch.zeros_like(action_norm)
        for qid in quality_ids:
            qsel = qids == int(qid)
            if not bool(qsel.any()):
                continue
            for bid in range(len(labels)):
                sel = qsel & (bin_ids == bid)
                count = int(sel.sum().item())
                if count == 0:
                    continue
                bucket = accum[(qid, bid)]
                bucket["windows"] += count
                bucket["reward_sum"] += float(reward_sum[sel].sum().item())
                bucket["action_norm"] += float(action_norm[sel].sum().item())
                bucket["first_action_l2"] += float(first_action_l2[sel].sum().item())
                bucket["action_rate_norm"] += float(action_rate_norm[sel].sum().item())
                bucket["positive_yaw"] += float(yaw_sign[sel].float().sum().item())

    quality_totals = {
        qid: sum(accum[(qid, bid)]["windows"] for bid in range(len(labels))) for qid in quality_ids
    }
    rows: list[dict[str, Any]] = []
    for qid in sorted(quality_ids):
        for bid, label in enumerate(labels):
            bucket = accum[(qid, bid)]
            count = int(bucket["windows"])
            if count == 0:
                continue
            rows.append(
                {
                    "quality": inverse_quality.get(qid, str(qid)),
                    "yaw_bin": label,
                    "windows": count,
                    "quality_fraction": count / max(1.0, quality_totals[qid]),
                    "reward_sum_mean": bucket["reward_sum"] / count,
                    "action_norm_mean": bucket["action_norm"] / count,
                    "first_action_l2_mean": bucket["first_action_l2"] / count,
                    "action_rate_norm_mean": bucket["action_rate_norm"] / count,
                    "positive_yaw_fraction": bucket["positive_yaw"] / count,
                }
            )

    summary = {
        "dataset": args.dataset,
        "metadata": args.metadata,
        "split": args.split,
        "qualities": quality_names,
        "yaw_edges": edges,
        "selected_windows": int(indices.numel()),
        "git_sha": git_sha(),
        "git_branch": git_branch(),
        "command": command_line(),
    }
    write_csv(out / "command_coverage_by_quality_yaw.csv", rows)
    write_markdown(out / "command_coverage_by_quality_yaw.md", rows, summary)
    (out / "command_coverage_summary.json").write_text(json.dumps({"summary": summary, "rows": rows}, indent=2), encoding="utf-8")
    print(json.dumps(summary, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
