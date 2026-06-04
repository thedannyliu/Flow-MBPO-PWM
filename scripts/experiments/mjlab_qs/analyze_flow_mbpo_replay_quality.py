#!/usr/bin/env python3
"""Analyze Flow-MBPO synthetic replay quality against nearest real windows."""

from __future__ import annotations

import argparse
import json
import math
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


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--synthetic-replay", required=True)
    p.add_argument("--dataset", required=True)
    p.add_argument("--metadata", required=True)
    p.add_argument("--normalization", required=True)
    p.add_argument("--output-json", required=True)
    p.add_argument("--output-md", required=True)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--split", default="train", choices=["train", "val", "test"])
    p.add_argument("--quality-filter", default="expert,expert_noisy")
    p.add_argument("--support-max-rows", type=int, default=20000)
    p.add_argument("--support-probe-rows", type=int, default=4096)
    p.add_argument("--distance-batch-size", type=int, default=256)
    p.add_argument("--state-weight", type=float, default=1.0)
    p.add_argument("--command-weight", type=float, default=1.0)
    p.add_argument("--action-weight", type=float, default=1.0)
    p.add_argument("--high-reward-quantile", type=float, default=0.90)
    p.add_argument("--high-distance-quantile", type=float, default=0.90)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def command_line() -> str:
    return " ".join([sys.executable, *sys.argv])


def git_value(*args: str) -> str:
    try:
        return subprocess.check_output(["git", *args], cwd=PROJECT_ROOT, text=True).strip()
    except Exception:
        return "unknown"


def load_norm(path: Path) -> dict[str, torch.Tensor]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    return {key: torch.tensor(value, dtype=torch.float32) for key, value in raw.items() if isinstance(value, list)}


def norm(x: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    return (x.float() - mean) / std.clamp_min(1.0e-6)


def summarize_tensor(x: torch.Tensor) -> dict[str, float]:
    x = x.detach().float().reshape(-1).cpu()
    finite = x[torch.isfinite(x)]
    if finite.numel() == 0:
        return {
            "mean": math.nan,
            "std": math.nan,
            "min": math.nan,
            "p50": math.nan,
            "p90": math.nan,
            "p99": math.nan,
            "max": math.nan,
        }
    return {
        "mean": float(finite.mean().item()),
        "std": float(finite.std(unbiased=False).item()),
        "min": float(finite.min().item()),
        "p50": float(torch.quantile(finite, 0.50).item()),
        "p90": float(torch.quantile(finite, 0.90).item()),
        "p99": float(torch.quantile(finite, 0.99).item()),
        "max": float(finite.max().item()),
    }


def fraction(mask: torch.Tensor) -> float:
    mask = mask.detach().bool().reshape(-1)
    if mask.numel() == 0:
        return math.nan
    return float(mask.float().mean().item())


def isin(values: torch.Tensor, candidates: torch.Tensor) -> torch.Tensor:
    mask = torch.zeros(values.shape, dtype=torch.bool)
    for candidate in candidates.tolist():
        mask |= values == int(candidate)
    return mask


def select_real_indices(data: dict[str, torch.Tensor], metadata: dict[str, Any], args: argparse.Namespace) -> torch.Tensor:
    split_id = int(metadata["split_id_map"][args.split])
    quality_names = [item.strip() for item in args.quality_filter.split(",") if item.strip()]
    quality_ids = torch.tensor([int(metadata["quality_id_map"][name]) for name in quality_names], dtype=torch.long)
    mask = data["split_id"].long() == split_id
    mask = mask & isin(data["quality_bin_id"].long(), quality_ids)
    ids = mask.nonzero(as_tuple=False).reshape(-1)
    if ids.numel() == 0:
        raise ValueError(f"No rows match split={args.split!r}, quality_filter={quality_names!r}")
    return ids


def real_features(
    data: dict[str, torch.Tensor],
    indices: torch.Tensor,
    nrm: dict[str, torch.Tensor],
    args: argparse.Namespace,
) -> torch.Tensor:
    state = norm(data["phys_obs"][indices, 0], nrm["phys_obs_mean"], nrm["phys_obs_std"])
    command = data["command"][indices, 0].float()
    if command.shape[-1] and "command_mean" in nrm:
        command = norm(command, nrm["command_mean"], nrm["command_std"])
    action = data["policy_action"][indices, 0].float()
    return torch.cat(
        [
            state * float(args.state_weight),
            command * float(args.command_weight),
            action * float(args.action_weight),
        ],
        dim=-1,
    ).contiguous()


def synthetic_features(replay: dict[str, torch.Tensor], args: argparse.Namespace) -> torch.Tensor:
    return torch.cat(
        [
            replay["state"].float() * float(args.state_weight),
            replay["command"].float() * float(args.command_weight),
            replay["action"].float() * float(args.action_weight),
        ],
        dim=-1,
    ).contiguous()


def nearest_l2_with_index(
    query: torch.Tensor,
    support: torch.Tensor,
    batch_size: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    if query.shape[-1] != support.shape[-1]:
        raise ValueError(f"Feature dimension mismatch: query={query.shape[-1]}, support={support.shape[-1]}")
    denom = math.sqrt(float(query.shape[-1]))
    support_device = support.to(device)
    distances: list[torch.Tensor] = []
    indices: list[torch.Tensor] = []
    for start in range(0, int(query.shape[0]), int(batch_size)):
        chunk = query[start : start + int(batch_size)].to(device)
        dist = torch.cdist(chunk, support_device, p=2)
        min_dist, min_idx = dist.min(dim=1)
        distances.append((min_dist / denom).detach().cpu())
        indices.append(min_idx.detach().cpu())
    return torch.cat(distances, dim=0), torch.cat(indices, dim=0)


def optional_tensor(replay: dict[str, torch.Tensor], key: str, fallback: torch.Tensor) -> torch.Tensor:
    value = replay.get(key)
    if torch.is_tensor(value):
        return value
    return fallback


def slice_stats(mask: torch.Tensor, values: dict[str, torch.Tensor], bools: dict[str, torch.Tensor]) -> dict[str, Any]:
    mask = mask.detach().bool().reshape(-1).cpu()
    if mask.numel() == 0 or not bool(mask.any().item()):
        return {"count": 0, "fraction": 0.0}
    out: dict[str, Any] = {"count": int(mask.sum().item()), "fraction": float(mask.float().mean().item())}
    for key, value in values.items():
        out[key] = summarize_tensor(value[mask])
    for key, value in bools.items():
        out[f"{key}_fraction"] = fraction(value[mask])
    return out


def write_md(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# Flow-MBPO H1 Replay Quality",
        "",
        f"- git: `{summary['git_sha']}`",
        f"- replay: `{summary['synthetic_replay']}`",
        f"- real rows selected: `{summary['selected_real_rows']}`",
        f"- support rows: `{summary['support_rows']}`",
        f"- support threshold p{int(100 * summary['support_threshold_quantile'])}: `{summary['support_threshold']:.6g}`",
        f"- synthetic OOD fraction: `{summary['synthetic_ood_fraction']:.6g}`",
        "",
        "## Reward Alignment",
        "",
        f"- synthetic conservative reward mean/p90/max: `{summary['synthetic']['reward_conservative']['mean']:.6g}` / `{summary['synthetic']['reward_conservative']['p90']:.6g}` / `{summary['synthetic']['reward_conservative']['max']:.6g}`",
        f"- nearest-real reward0 mean/p90/max: `{summary['nearest_real']['reward0']['mean']:.6g}` / `{summary['nearest_real']['reward0']['p90']:.6g}` / `{summary['nearest_real']['reward0']['max']:.6g}`",
        f"- conservative minus nearest-real reward0 mean/p90/max: `{summary['reward_delta_conservative_minus_nearest_real']['mean']:.6g}` / `{summary['reward_delta_conservative_minus_nearest_real']['p90']:.6g}` / `{summary['reward_delta_conservative_minus_nearest_real']['max']:.6g}`",
        "",
        "## Done/Fall Proxy",
        "",
        f"- synthetic done/model/uncertainty fractions: `{summary['synthetic_done_fraction']:.6g}` / `{summary['synthetic_done_model_fraction']:.6g}` / `{summary['synthetic_done_uncertainty_fraction']:.6g}`",
        f"- nearest-real done-any / termination-any fractions: `{summary['nearest_real_done_any_fraction']:.6g}` / `{summary['nearest_real_termination_any_fraction']:.6g}`",
        "",
        "## Slices",
        "",
        "| slice | n | synth reward mean | nearest reward mean | reward delta mean | support dist mean | synth done | nearest term any |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name, row in summary["slices"].items():
        if int(row["count"]) == 0:
            lines.append(f"| {name} | 0 | nan | nan | nan | nan | nan | nan |")
            continue
        lines.append(
            "| "
            + " | ".join(
                [
                    name,
                    str(row["count"]),
                    f"{row['reward_conservative']['mean']:.6g}",
                    f"{row['nearest_real_reward0']['mean']:.6g}",
                    f"{row['reward_delta_conservative']['mean']:.6g}",
                    f"{row['support_distance']['mean']:.6g}",
                    f"{row['synthetic_done_fraction']:.6g}",
                    f"{row['nearest_real_termination_any_fraction']:.6g}",
                ]
            )
            + " |"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit(f"CUDA device requested ({args.device}) but torch.cuda.is_available() is false")

    replay = torch.load(args.synthetic_replay, map_location="cpu", weights_only=False)
    data = torch.load(args.dataset, map_location="cpu", weights_only=False)
    metadata = json.loads(Path(args.metadata).read_text(encoding="utf-8"))
    nrm = load_norm(Path(args.normalization))
    for key in ["state", "command", "action", "reward_conservative", "done"]:
        if key not in replay:
            raise ValueError(f"Synthetic replay is missing key {key!r}")

    selected = select_real_indices(data, metadata, args)
    generator = torch.Generator().manual_seed(int(args.seed))
    perm = selected[torch.randperm(selected.numel(), generator=generator)]
    support_n = min(int(args.support_max_rows), int(perm.numel()))
    probe_n = min(int(args.support_probe_rows), max(0, int(perm.numel()) - support_n))
    if support_n == 0 or probe_n == 0:
        raise ValueError("Need non-empty support and probe rows; lower support-max-rows if necessary")
    support_indices = perm[:support_n]
    probe_indices = perm[support_n : support_n + probe_n]

    support_feat = real_features(data, support_indices, nrm, args)
    probe_feat = real_features(data, probe_indices, nrm, args)
    synth_feat = synthetic_features(replay, args)
    real_probe_distance, _ = nearest_l2_with_index(probe_feat, support_feat, int(args.distance_batch_size), device)
    q = min(max(float(args.high_distance_quantile), 0.0), 1.0)
    support_threshold = torch.quantile(real_probe_distance[torch.isfinite(real_probe_distance)], q)
    support_distance, support_nearest_pos = nearest_l2_with_index(
        synth_feat, support_feat, int(args.distance_batch_size), device
    )
    nearest_indices = support_indices[support_nearest_pos]

    n = int(replay["reward_conservative"].shape[0])
    zeros = torch.zeros(n)
    reward_raw = optional_tensor(replay, "reward_raw", replay.get("reward", replay["reward_conservative"])).float().cpu()
    reward_conservative = replay["reward_conservative"].float().cpu()
    done = replay["done"].bool().cpu()
    done_model = optional_tensor(replay, "done_model", torch.zeros(n, dtype=torch.bool)).bool().cpu()
    done_uncertainty = optional_tensor(replay, "done_uncertainty", torch.zeros(n, dtype=torch.bool)).bool().cpu()
    uncertainty = optional_tensor(replay, "uncertainty", zeros).float().cpu()
    reward_uncertainty = optional_tensor(replay, "reward_uncertainty", zeros).float().cpu()
    next_state_uncertainty = optional_tensor(replay, "next_state_uncertainty", zeros).float().cpu()
    horizon_step = optional_tensor(replay, "horizon_step", torch.zeros(n, dtype=torch.long)).float().cpu()

    nearest_reward0 = data["reward"][nearest_indices, 0].float().cpu()
    nearest_return = torch.nan_to_num(data["reward"][nearest_indices].float().cpu(), nan=0.0).sum(dim=1)
    nearest_done0 = data["done"][nearest_indices, 0].bool().cpu()
    nearest_done_any = data["done"][nearest_indices].bool().cpu().any(dim=1)
    nearest_termination0 = data["termination"][nearest_indices, 0].bool().cpu()
    nearest_termination_any = data["termination"][nearest_indices].bool().cpu().any(dim=1)
    nearest_action = data["policy_action"][nearest_indices, 0].float().cpu()
    synth_action = replay["action"].float().cpu()
    action_delta = (synth_action - nearest_action).pow(2).mean(dim=-1).sqrt()
    reward_delta_raw = reward_raw - nearest_reward0
    reward_delta_cons = reward_conservative - nearest_reward0
    ood = support_distance > support_threshold
    high_reward = reward_conservative >= torch.quantile(
        reward_conservative[torch.isfinite(reward_conservative)], min(max(float(args.high_reward_quantile), 0.0), 1.0)
    )
    high_distance = support_distance >= torch.quantile(
        support_distance[torch.isfinite(support_distance)], min(max(float(args.high_distance_quantile), 0.0), 1.0)
    )

    values = {
        "reward_raw": reward_raw,
        "reward_conservative": reward_conservative,
        "nearest_real_reward0": nearest_reward0,
        "nearest_real_return_horizon": nearest_return,
        "reward_delta_raw": reward_delta_raw,
        "reward_delta_conservative": reward_delta_cons,
        "support_distance": support_distance,
        "uncertainty": uncertainty,
        "reward_uncertainty": reward_uncertainty,
        "next_state_uncertainty": next_state_uncertainty,
        "horizon_step": horizon_step,
        "action_delta_to_nearest_real": action_delta,
        "synthetic_action_l2": synth_action.pow(2).mean(dim=-1).sqrt(),
        "nearest_real_action_l2": nearest_action.pow(2).mean(dim=-1).sqrt(),
    }
    bools = {
        "synthetic_done": done,
        "synthetic_done_model": done_model,
        "synthetic_done_uncertainty": done_uncertainty,
        "nearest_real_done0": nearest_done0,
        "nearest_real_done_any": nearest_done_any,
        "nearest_real_termination0": nearest_termination0,
        "nearest_real_termination_any": nearest_termination_any,
        "ood_by_real_probe_threshold": ood,
    }
    slices = {
        "all": slice_stats(torch.ones(n, dtype=torch.bool), values, bools),
        "top_reward_decile": slice_stats(high_reward, values, bools),
        "top_support_distance_decile": slice_stats(high_distance, values, bools),
        "ood_by_real_probe_threshold": slice_stats(ood, values, bools),
        "synthetic_done": slice_stats(done, values, bools),
        "synthetic_not_done": slice_stats(~done, values, bools),
    }

    summary: dict[str, Any] = {
        "git_sha": git_value("rev-parse", "HEAD"),
        "git_branch": git_value("rev-parse", "--abbrev-ref", "HEAD"),
        "command": command_line(),
        "synthetic_replay": args.synthetic_replay,
        "dataset": args.dataset,
        "metadata": args.metadata,
        "normalization": args.normalization,
        "split": args.split,
        "quality_filter": args.quality_filter,
        "selected_real_rows": int(selected.numel()),
        "support_rows": int(support_indices.numel()),
        "support_probe_rows": int(probe_indices.numel()),
        "support_feature_dim": int(support_feat.shape[-1]),
        "support_threshold": float(support_threshold.item()),
        "support_threshold_quantile": float(args.high_distance_quantile),
        "state_weight": float(args.state_weight),
        "command_weight": float(args.command_weight),
        "action_weight": float(args.action_weight),
        "synthetic_transitions": n,
        "synthetic_ood_fraction": fraction(ood),
        "synthetic_done_fraction": fraction(done),
        "synthetic_done_model_fraction": fraction(done_model),
        "synthetic_done_uncertainty_fraction": fraction(done_uncertainty),
        "nearest_real_done0_fraction": fraction(nearest_done0),
        "nearest_real_done_any_fraction": fraction(nearest_done_any),
        "nearest_real_termination0_fraction": fraction(nearest_termination0),
        "nearest_real_termination_any_fraction": fraction(nearest_termination_any),
        "real_probe_distance": summarize_tensor(real_probe_distance),
        "support_distance": summarize_tensor(support_distance),
        "synthetic": {
            "reward_raw": summarize_tensor(reward_raw),
            "reward_conservative": summarize_tensor(reward_conservative),
            "uncertainty": summarize_tensor(uncertainty),
            "reward_uncertainty": summarize_tensor(reward_uncertainty),
            "next_state_uncertainty": summarize_tensor(next_state_uncertainty),
            "horizon_step": summarize_tensor(horizon_step),
            "action_l2": summarize_tensor(synth_action.pow(2).mean(dim=-1).sqrt()),
        },
        "nearest_real": {
            "reward0": summarize_tensor(nearest_reward0),
            "return_horizon": summarize_tensor(nearest_return),
            "action_l2": summarize_tensor(nearest_action.pow(2).mean(dim=-1).sqrt()),
        },
        "reward_delta_raw_minus_nearest_real": summarize_tensor(reward_delta_raw),
        "reward_delta_conservative_minus_nearest_real": summarize_tensor(reward_delta_cons),
        "action_delta_to_nearest_real": summarize_tensor(action_delta),
        "slices": slices,
    }

    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_md(Path(args.output_md), summary)
    print(json.dumps(summary, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
