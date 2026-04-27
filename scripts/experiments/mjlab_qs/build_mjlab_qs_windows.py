#!/usr/bin/env python3
"""Build MJLab-QS window dataset and normalization manifest from raw shards."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List

import torch


QUALITY_TO_ID = {
    "random_smooth": 0,
    "weak": 1,
    "medium": 2,
    "expert": 3,
    "expert_noisy": 4,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--metadata-output", default=None)
    parser.add_argument("--normalization-output", default=None)
    parser.add_argument("--report-output", default=None)
    parser.add_argument("--horizon", type=int, default=16)
    parser.add_argument("--stride", type=int, default=4)
    parser.add_argument("--split-seed", type=int, default=0)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--min-train-episodes-per-bucket", type=int, default=50)
    parser.add_argument("--min-valid-train-windows-per-bucket", type=int, default=10000)
    parser.add_argument("--allow-preliminary", action="store_true")
    return parser.parse_args()


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def split_group(num_eps: int, seed: int, train_ratio: float, val_ratio: float) -> List[str]:
    gen = torch.Generator().manual_seed(seed)
    perm = torch.randperm(num_eps, generator=gen).tolist()
    n_train = int(round(num_eps * train_ratio))
    n_val = int(round(num_eps * val_ratio))
    n_train = min(max(1, n_train), num_eps)
    n_val = min(max(1, n_val), max(0, num_eps - n_train))
    split = ["test"] * num_eps
    for idx in perm[:n_train]:
        split[idx] = "train"
    for idx in perm[n_train : n_train + n_val]:
        split[idx] = "val"
    return split


def stratified_splits(
    episodes: List[Dict[str, object]],
    seed: int,
    train_ratio: float,
    val_ratio: float,
) -> List[str]:
    groups = defaultdict(list)
    for idx, ep in enumerate(episodes):
        groups[(str(ep["task_id_resolved"]), str(ep["quality_bin"]))].append(idx)
    out = ["test"] * len(episodes)
    for group_id, indices in sorted(groups.items()):
        local = split_group(len(indices), seed + abs(hash(group_id)) % 1_000_000, train_ratio, val_ratio)
        for idx, split in zip(indices, local):
            out[idx] = split
    return out


def valid_starts(done: torch.Tensor, horizon: int, stride: int) -> List[int]:
    # Rows 1..T hold transition done flags. A window may include a terminal
    # transition but must not include any transition after a true done.
    trans_done = done[1:].bool()
    max_start = int(trans_done.shape[0]) - horizon
    starts: List[int] = []
    for start in range(0, max_start + 1, stride):
        flags = trans_done[start : start + horizon]
        done_idx = flags.nonzero(as_tuple=False)
        if done_idx.numel() > 0 and int(done_idx[0].item()) < horizon - 1:
            continue
        starts.append(start)
    return starts


def main() -> None:
    args = parse_args()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    metadata_output = Path(args.metadata_output) if args.metadata_output else output.with_suffix(".json")
    norm_output = Path(args.normalization_output) if args.normalization_output else output.with_name(output.stem + "_normalization.json")
    report_output = Path(args.report_output) if args.report_output else output.with_name(output.stem + "_report.md")

    episodes: List[Dict[str, object]] = []
    raw_paths = [Path(p) for p in args.raw]
    raw_hashes = {str(p): sha256(p) for p in raw_paths}
    for path in raw_paths:
        payload = torch.load(path, map_location="cpu", weights_only=False)
        episodes.extend(payload["episodes"])
    if not episodes:
        raise RuntimeError("No episodes loaded.")

    splits = stratified_splits(episodes, args.split_seed, args.train_ratio, args.val_ratio)
    windows = defaultdict(list)
    source_episode: List[int] = []
    source_start: List[int] = []
    split_ids: List[int] = []
    quality_ids: List[int] = []
    task_keys: List[str] = []
    split_to_id = {"train": 0, "val": 1, "test": 2}
    counts = Counter()
    valid_window_counts = Counter()
    train_episode_counts = Counter()

    for ep_idx, ep in enumerate(episodes):
        split = splits[ep_idx]
        qbin = str(ep["quality_bin"])
        task = str(ep["task_id_resolved"])
        key = (task, qbin, split)
        counts[key] += 1
        if split == "train":
            train_episode_counts[(task, qbin)] += 1
        starts = valid_starts(ep["done"], args.horizon, args.stride)
        valid_window_counts[(task, qbin, split)] += len(starts)
        for start in starts:
            end_obs = start + args.horizon + 1
            trans = slice(start + 1, start + args.horizon + 1)
            windows["phys_obs"].append(ep["phys_obs"][start:end_obs])
            windows["model_obs"].append(ep["model_obs"][start:end_obs])
            windows["command"].append(ep["command"][trans])
            windows["policy_action"].append(ep["policy_action"][trans])
            windows["env_action"].append(ep["env_action"][trans])
            windows["reward"].append(ep["reward"][trans])
            windows["done"].append(ep["done"][trans])
            windows["termination"].append(ep["termination"][trans])
            windows["truncation"].append(ep["truncation"][trans])
            source_episode.append(ep_idx)
            source_start.append(start)
            split_ids.append(split_to_id[split])
            quality_ids.append(QUALITY_TO_ID.get(qbin, -1))
            task_keys.append(task)

    if not windows["phys_obs"]:
        raise RuntimeError("No valid windows generated.")

    data = {k: torch.stack(v, dim=0) for k, v in windows.items()}
    data["source_episode"] = torch.tensor(source_episode, dtype=torch.long)
    data["source_start"] = torch.tensor(source_start, dtype=torch.long)
    data["split_id"] = torch.tensor(split_ids, dtype=torch.long)
    data["quality_bin_id"] = torch.tensor(quality_ids, dtype=torch.long)
    data["task_key"] = task_keys

    train_mask = data["split_id"] == 0
    norm = {
        "computed_from": "train_split_only",
        "dataset_output": str(output),
        "horizon": args.horizon,
        "stride": args.stride,
    }
    for key in ("phys_obs", "model_obs"):
        vals = data[key][train_mask].reshape(-1, data[key].shape[-1]).float()
        norm[f"{key}_mean"] = vals.mean(dim=0).tolist()
        norm[f"{key}_std"] = vals.std(dim=0).clamp_min(1e-6).tolist()
    cmd = data["command"][train_mask].reshape(-1, data["command"].shape[-1]).float()
    if cmd.shape[-1] > 0:
        norm["command_mean"] = cmd.mean(dim=0).tolist()
        norm["command_std"] = cmd.std(dim=0).clamp_min(1e-6).tolist()
    else:
        norm["command_mean"] = []
        norm["command_std"] = []
    rew = data["reward"][train_mask].reshape(-1, 1).float()
    norm["reward_mean"] = rew.mean(dim=0).tolist()
    norm["reward_std"] = rew.std(dim=0).clamp_min(1e-6).tolist()
    norm_output.write_text(json.dumps(norm, indent=2), encoding="utf-8")

    metadata = {
        "script": "build_mjlab_qs_windows.py",
        "raw_paths": [str(p) for p in raw_paths],
        "raw_sha256": raw_hashes,
        "num_episodes": len(episodes),
        "num_windows": int(data["phys_obs"].shape[0]),
        "horizon": args.horizon,
        "stride": args.stride,
        "split_seed": args.split_seed,
        "split_id_map": split_to_id,
        "quality_id_map": QUALITY_TO_ID,
        "phys_obs_dim": int(data["phys_obs"].shape[-1]),
        "model_obs_dim": int(data["model_obs"].shape[-1]),
        "command_dim": int(data["command"].shape[-1]),
        "act_dim": int(data["policy_action"].shape[-1]),
        "normalization_manifest": str(norm_output),
        "normalization_sha256": sha256(norm_output),
    }
    metadata_output.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    torch.save(data, output)

    failures = []
    for key, n in sorted(train_episode_counts.items()):
        if n < args.min_train_episodes_per_bucket:
            failures.append(f"{key}: train episodes {n} < {args.min_train_episodes_per_bucket}")
        w = valid_window_counts[(key[0], key[1], "train")]
        if w < args.min_valid_train_windows_per_bucket:
            failures.append(f"{key}: train windows {w} < {args.min_valid_train_windows_per_bucket}")
    lines = [
        "# MJLab-QS Dataset Report",
        "",
        f"- output: `{output}`",
        f"- num_episodes: {len(episodes)}",
        f"- num_windows: {int(data['phys_obs'].shape[0])}",
        f"- normalization_manifest: `{norm_output}`",
        "",
        "## Counts",
        "",
        "| task | quality_bin | split | episodes | valid_windows |",
        "|---|---|---:|---:|---:|",
    ]
    for (task, qbin, split), n in sorted(counts.items()):
        lines.append(f"| {task} | {qbin} | {split} | {n} | {valid_window_counts[(task, qbin, split)]} |")
    lines.extend(["", "## Gate", ""])
    if failures:
        lines.append("- status: FAIL")
        lines.extend(f"- {f}" for f in failures)
    else:
        lines.append("- status: PASS")
    report_output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    if failures and not args.allow_preliminary:
        raise RuntimeError("Minimum valid-window gate failed:\n" + "\n".join(failures))
    print(f"saved windows: {output}")
    print(f"saved metadata: {metadata_output}")
    print(f"saved normalization: {norm_output}")
    print(f"saved report: {report_output}")


if __name__ == "__main__":
    main()
