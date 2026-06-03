#!/usr/bin/env python3
"""Audit saved policy imitation error on MJLab-QS dataset slices."""

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
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from scripts.experiments.mjlab_qs.render_policy_rollout import build_actor  # noqa: E402
from scripts.experiments.mjlab_qs.run_phaseA_wm_feasibility import load_norm, norm  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--checkpoint",
        action="append",
        required=True,
        help="Policy checkpoint in label=path form. Can be repeated.",
    )
    p.add_argument("--dataset", default="", help="Override dataset path; default uses first checkpoint args.")
    p.add_argument("--metadata", default="", help="Override metadata path; default uses first checkpoint args.")
    p.add_argument("--normalization", default="", help="Override normalization path; default uses first checkpoint args.")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--split", default="train")
    p.add_argument("--qualities", default="expert,expert_noisy")
    p.add_argument("--yaw-edges", default="0.175,0.35,0.525")
    p.add_argument("--chunk-size", type=int, default=4096)
    p.add_argument("--max-windows", type=int, default=0)
    p.add_argument("--sample-seed", type=int, default=0)
    p.add_argument("--device", default="cpu")
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


def parse_checkpoint_specs(raw_specs: list[str]) -> list[tuple[str, Path]]:
    specs: list[tuple[str, Path]] = []
    for spec in raw_specs:
        if "=" not in spec:
            raise ValueError(f"checkpoint spec {spec!r} must have form label=path")
        label, path = spec.split("=", 1)
        label = label.strip()
        if not label:
            raise ValueError(f"checkpoint spec {spec!r} has empty label")
        specs.append((label, Path(path.strip())))
    return specs


def selected_ids(raw: str, id_map: dict[str, int], all_ids: set[int]) -> set[int]:
    if raw.strip().lower() == "all":
        return set(all_ids)
    out: set[int] = set()
    for item in parse_name_list(raw):
        out.add(id_map[item] if item in id_map else int(item))
    return out


def bin_labels(edges: list[float]) -> list[str]:
    if not edges:
        return ["all"]
    labels = [f"[0,{edges[0]:g})"]
    labels.extend(f"[{lo:g},{hi:g})" for lo, hi in zip(edges[:-1], edges[1:]))
    labels.append(f"[{edges[-1]:g},inf)")
    return labels


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def fmt(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def write_markdown(path: Path, rows: list[dict[str, Any]], summary: dict[str, Any]) -> None:
    cols = [
        "checkpoint",
        "quality",
        "yaw_bin",
        "windows",
        "action_mse_mean",
        "action_l2_error_mean",
        "target_action_l2_mean",
        "pred_action_l2_mean",
        "action_rate_l2_mean",
    ]
    lines = [
        "# MJLab-QS Policy BC Error Slices",
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


def add_to_bucket(bucket: dict[str, float], values: dict[str, torch.Tensor], sel: torch.Tensor) -> None:
    count = int(sel.sum().item())
    if count == 0:
        return
    bucket["windows"] += count
    for key, tensor in values.items():
        bucket[key] += float(tensor[sel].sum().item())


@torch.no_grad()
def audit_checkpoint(
    label: str,
    checkpoint_path: Path,
    data: dict[str, torch.Tensor],
    metadata: dict[str, Any],
    nrm: dict[str, torch.Tensor],
    indices: torch.Tensor,
    yaw_edges: list[float],
    device: torch.device,
    chunk_size: int,
) -> list[dict[str, Any]]:
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    ckpt_args = ckpt["args"]
    state_dim = int(data["phys_obs"].shape[-1])
    command_dim = int(data["command"].shape[-1])
    action_dim = int(data["policy_action"].shape[-1])
    actor = build_actor(ckpt, state_dim=state_dim, command_dim=command_dim, action_dim=action_dim, device=device)
    quality_inverse = {int(v): str(k) for k, v in metadata.get("quality_id_map", {}).items()}
    quality_ids = sorted(int(x) for x in torch.unique(data["quality_bin_id"][indices]).tolist())
    labels = bin_labels(yaw_edges)
    edge_tensor = torch.tensor(yaw_edges, dtype=torch.float32)
    accum: dict[tuple[int, int], dict[str, float]] = {}
    for qid in quality_ids:
        for bid in range(len(labels)):
            accum[(qid, bid)] = {
                "windows": 0.0,
                "action_mse": 0.0,
                "action_l2_error": 0.0,
                "target_action_l2": 0.0,
                "pred_action_l2": 0.0,
                "action_rate_l2": 0.0,
            }

    for start in range(0, int(indices.numel()), chunk_size):
        ids = indices[start : start + chunk_size]
        cmd = data["command"][ids].to(device).float()
        target = data["policy_action"][ids].to(device).float()
        bsz, horizon, _ = target.shape
        phys = data["phys_obs"][ids, :horizon].to(device).float()
        z = norm(phys, nrm["phys_obs_mean"], nrm["phys_obs_std"])
        c = cmd
        if c.shape[-1] and "command_mean" in nrm:
            c = norm(c, nrm["command_mean"], nrm["command_std"])
        pred = actor(
            z.reshape(bsz * horizon, state_dim),
            c.reshape(bsz * horizon, command_dim),
            deterministic=True,
        ).reshape(bsz, horizon, action_dim)
        diff = pred - target
        action_mse = diff.pow(2).mean(dim=(1, 2))
        action_l2_error = diff.pow(2).mean(dim=-1).sqrt().mean(dim=1)
        target_action_l2 = target.pow(2).mean(dim=-1).sqrt().mean(dim=1)
        pred_action_l2 = pred.pow(2).mean(dim=-1).sqrt().mean(dim=1)
        if horizon > 1:
            action_rate_l2 = (target[:, 1:] - target[:, :-1]).pow(2).mean(dim=-1).sqrt().mean(dim=1)
        else:
            action_rate_l2 = torch.zeros_like(target_action_l2)
        yaw_abs_max = data["command"][ids, :, 2].float().abs().amax(dim=1)
        bin_ids = torch.bucketize(yaw_abs_max, edge_tensor)
        qids = data["quality_bin_id"][ids].long()
        values = {
            "action_mse": action_mse.cpu(),
            "action_l2_error": action_l2_error.cpu(),
            "target_action_l2": target_action_l2.cpu(),
            "pred_action_l2": pred_action_l2.cpu(),
            "action_rate_l2": action_rate_l2.cpu(),
        }
        for qid in quality_ids:
            qsel = qids.cpu() == int(qid)
            if not bool(qsel.any()):
                continue
            for bid in range(len(labels)):
                sel = qsel & (bin_ids == bid)
                add_to_bucket(accum[(qid, bid)], values, sel)

    rows: list[dict[str, Any]] = []
    for qid in quality_ids:
        for bid, yaw_label in enumerate(labels):
            bucket = accum[(qid, bid)]
            count = int(bucket["windows"])
            if count == 0:
                continue
            rows.append(
                {
                    "checkpoint": label,
                    "checkpoint_path": str(checkpoint_path),
                    "quality": quality_inverse.get(qid, str(qid)),
                    "yaw_bin": yaw_label,
                    "windows": count,
                    "action_mse_mean": bucket["action_mse"] / count,
                    "action_l2_error_mean": bucket["action_l2_error"] / count,
                    "target_action_l2_mean": bucket["target_action_l2"] / count,
                    "pred_action_l2_mean": bucket["pred_action_l2"] / count,
                    "action_rate_l2_mean": bucket["action_rate_l2"] / count,
                }
            )
    return rows


def main() -> None:
    args = parse_args()
    specs = parse_checkpoint_specs(args.checkpoint)
    first_ckpt = torch.load(specs[0][1], map_location="cpu", weights_only=False)
    first_args = first_ckpt["args"]
    dataset_path = Path(args.dataset or first_args["dataset"])
    metadata_path = Path(args.metadata or first_args["metadata"])
    normalization_path = Path(args.normalization or first_args["normalization"])
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    data = torch.load(dataset_path, map_location="cpu", weights_only=False, mmap=True)
    device = torch.device(args.device)
    nrm = load_norm(normalization_path, device)
    split_map = {str(k): int(v) for k, v in metadata.get("split_id_map", {"train": 0}).items()}
    quality_map = {str(k): int(v) for k, v in metadata["quality_id_map"].items()}
    split_ids = selected_ids(args.split, split_map, set(split_map.values()))
    quality_ids = selected_ids(args.qualities, quality_map, set(quality_map.values()))
    mask = torch.zeros(data["split_id"].shape, dtype=torch.bool)
    for split_id in split_ids:
        mask |= data["split_id"].long() == int(split_id)
    qmask = torch.zeros(data["quality_bin_id"].shape, dtype=torch.bool)
    for quality_id in quality_ids:
        qmask |= data["quality_bin_id"].long() == int(quality_id)
    indices = (mask & qmask).nonzero(as_tuple=False).squeeze(-1)
    if args.max_windows > 0 and indices.numel() > args.max_windows:
        generator = torch.Generator().manual_seed(int(args.sample_seed))
        positions = torch.randperm(indices.numel(), generator=generator)[: args.max_windows]
        indices = indices[positions.sort().values]
    if indices.numel() == 0:
        raise SystemExit("selected zero windows")
    if args.max_windows > 0:
        data = {
            "phys_obs": data["phys_obs"][indices].clone(),
            "command": data["command"][indices].clone(),
            "policy_action": data["policy_action"][indices].clone(),
            "quality_bin_id": data["quality_bin_id"][indices].clone(),
        }
        indices = torch.arange(data["quality_bin_id"].shape[0])
    yaw_edges = parse_float_list(args.yaw_edges)
    rows: list[dict[str, Any]] = []
    for label, checkpoint_path in specs:
        rows.extend(
            audit_checkpoint(
                label,
                checkpoint_path,
                data,
                metadata,
                nrm,
                indices,
                yaw_edges,
                device,
                args.chunk_size,
            )
        )

    summary = {
        "dataset": str(dataset_path),
        "metadata": str(metadata_path),
        "normalization": str(normalization_path),
        "split": args.split,
        "qualities": args.qualities,
        "yaw_edges": args.yaw_edges,
        "selected_windows": int(indices.numel()),
        "max_windows": int(args.max_windows),
        "sample_seed": int(args.sample_seed),
        "checkpoints": [{"label": label, "path": str(path)} for label, path in specs],
        "git_sha": git_sha(),
        "git_branch": git_branch(),
        "command": command_line(),
    }
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    write_csv(out / "bc_error_slices.csv", rows)
    write_markdown(out / "bc_error_slices.md", rows, summary)
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({"output_dir": str(out), "rows": len(rows), **summary}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
