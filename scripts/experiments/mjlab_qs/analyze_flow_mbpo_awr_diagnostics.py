#!/usr/bin/env python3
"""Diagnose Flow-MBPO AWR action drift and synthetic replay weighting."""

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

from scripts.experiments.mjlab_qs.render_policy_rollout import build_actor  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True)
    p.add_argument("--metadata", required=True)
    p.add_argument("--normalization", required=True)
    p.add_argument("--bc-checkpoint", required=True)
    p.add_argument("--policy-checkpoint", action="append", required=True)
    p.add_argument("--policy-label", action="append", default=[])
    p.add_argument("--synthetic-replay", required=True)
    p.add_argument("--output-json", required=True)
    p.add_argument("--output-md", required=True)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--split", default="train", choices=["train", "val", "test"])
    p.add_argument("--quality-filter", default="expert,expert_noisy")
    p.add_argument("--num-real", type=int, default=4096)
    p.add_argument("--num-synthetic", type=int, default=4096)
    p.add_argument("--adv-temperature", type=float, default=1.0)
    p.add_argument("--weight-clip", type=float, default=20.0)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def command_line() -> str:
    return " ".join([sys.executable, *sys.argv])


def git_value(*args: str) -> str:
    try:
        return subprocess.check_output(["git", *args], text=True).strip()
    except Exception:
        return "unknown"


def load_norm(path: Path, device: torch.device) -> dict[str, torch.Tensor]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    return {k: torch.tensor(v, dtype=torch.float32, device=device) for k, v in raw.items() if isinstance(v, list)}


def norm(x: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    return (x - mean) / std.clamp_min(1e-6)


def summarize_tensor(x: torch.Tensor) -> dict[str, float]:
    x = x.detach().float().reshape(-1).cpu()
    finite = x[torch.isfinite(x)]
    if finite.numel() == 0:
        return {"mean": math.nan, "std": math.nan, "min": math.nan, "p50": math.nan, "p90": math.nan, "p99": math.nan, "max": math.nan}
    return {
        "mean": float(finite.mean().item()),
        "std": float(finite.std(unbiased=False).item()),
        "min": float(finite.min().item()),
        "p50": float(torch.quantile(finite, 0.50).item()),
        "p90": float(torch.quantile(finite, 0.90).item()),
        "p99": float(torch.quantile(finite, 0.99).item()),
        "max": float(finite.max().item()),
    }


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


def sample_indices(n: int, count: int, seed: int) -> torch.Tensor:
    if count <= 0 or count >= n:
        return torch.arange(n)
    generator = torch.Generator().manual_seed(seed)
    return torch.randperm(n, generator=generator)[:count]


def load_actor(path: Path, state_dim: int, command_dim: int, action_dim: int, device: torch.device):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    return build_actor(ckpt, state_dim, command_dim, action_dim, device), ckpt


def advantage_weights(reward: torch.Tensor, temperature: float, weight_clip: float) -> torch.Tensor:
    centered = reward - reward.mean()
    scaled = centered / max(float(temperature), 1.0e-6)
    return torch.exp(scaled.clamp(min=-20.0, max=20.0)).clamp(max=float(weight_clip))


@torch.no_grad()
def actor_actions(actor, state: torch.Tensor, command: torch.Tensor, batch_size: int = 2048) -> torch.Tensor:
    pieces = []
    for start in range(0, state.shape[0], batch_size):
        z = state[start : start + batch_size]
        c = command[start : start + batch_size]
        pieces.append(actor(z, c, deterministic=True).clamp(-1.0, 1.0).detach().cpu())
    return torch.cat(pieces, dim=0)


def action_delta_metrics(reference: torch.Tensor, candidate: torch.Tensor, target: torch.Tensor | None = None) -> dict[str, Any]:
    delta = candidate - reference
    out: dict[str, Any] = {
        "delta_l2": summarize_tensor(delta.pow(2).mean(dim=-1).sqrt()),
        "delta_abs_max": summarize_tensor(delta.abs().max(dim=-1).values),
        "candidate_action_l2": summarize_tensor(candidate.pow(2).mean(dim=-1).sqrt()),
        "reference_action_l2": summarize_tensor(reference.pow(2).mean(dim=-1).sqrt()),
    }
    if target is not None:
        out["candidate_bc_mse_to_logged"] = float((candidate - target).pow(2).mean().item())
        out["reference_bc_mse_to_logged"] = float((reference - target).pow(2).mean().item())
    return out


def slice_stats(mask: torch.Tensor, values: dict[str, torch.Tensor]) -> dict[str, Any]:
    if mask.numel() == 0 or not bool(mask.any().item()):
        return {"count": 0}
    out: dict[str, Any] = {"count": int(mask.sum().item())}
    for key, value in values.items():
        out[key] = summarize_tensor(value[mask])
    return out


def write_md(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# Flow-MBPO AWR Diagnostics",
        "",
        f"- git: `{summary['git_sha']}`",
        f"- synthetic replay: `{summary['synthetic_replay']}`",
        f"- real rows sampled: `{summary['real_sample_count']}`",
        f"- synthetic rows sampled: `{summary['synthetic_sample_count']}`",
        "",
        "## Replay",
        "",
        f"- conservative reward mean: `{summary['synthetic']['reward_conservative']['mean']:.6g}`",
        f"- conservative reward p90/p99/max: `{summary['synthetic']['reward_conservative']['p90']:.6g}` / `{summary['synthetic']['reward_conservative']['p99']:.6g}` / `{summary['synthetic']['reward_conservative']['max']:.6g}`",
        f"- AWR weight mean/p90/max: `{summary['synthetic']['adv_weight']['mean']:.6g}` / `{summary['synthetic']['adv_weight']['p90']:.6g}` / `{summary['synthetic']['adv_weight']['max']:.6g}`",
        f"- done fraction: `{summary['synthetic']['done_fraction']:.6g}`",
        f"- top reward decile count: `{summary['synthetic_slices']['top_reward_decile']['count']}`",
        "",
        "## Action Drift",
        "",
        "| policy | real delta mean | real delta p90 | synth delta mean | synth delta p90 | real logged MSE |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for label, metrics in summary["policies"].items():
        real = metrics["real_action_delta"]["delta_l2"]
        synth = metrics["synthetic_action_delta"]["delta_l2"]
        real_mse = metrics["real_action_delta"].get("candidate_bc_mse_to_logged", math.nan)
        lines.append(
            f"| {label} | {real['mean']:.6g} | {real['p90']:.6g} | {synth['mean']:.6g} | {synth['p90']:.6g} | {real_mse:.6g} |"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit(f"CUDA device requested ({args.device}) but torch.cuda.is_available() is false")

    data = torch.load(args.dataset, map_location="cpu", weights_only=False)
    metadata = json.loads(Path(args.metadata).read_text(encoding="utf-8"))
    replay = torch.load(args.synthetic_replay, map_location="cpu", weights_only=False)
    nrm = load_norm(Path(args.normalization), device)

    state_dim = int(data["phys_obs"].shape[-1])
    action_dim = int(data["policy_action"].shape[-1])
    command_dim = int(data["command"].shape[-1])
    bc_actor, _ = load_actor(Path(args.bc_checkpoint), state_dim, command_dim, action_dim, device)
    policy_paths = [Path(path) for path in args.policy_checkpoint]
    labels = list(args.policy_label)
    if labels and len(labels) != len(policy_paths):
        raise ValueError("--policy-label must be omitted or have one entry per --policy-checkpoint")
    if not labels:
        labels = [path.stem for path in policy_paths]

    real_ids_all = select_real_indices(data, metadata, args)
    real_ids = real_ids_all[sample_indices(int(real_ids_all.numel()), int(args.num_real), int(args.seed))]
    real_state = norm(data["phys_obs"][real_ids, 0].to(device).float(), nrm["phys_obs_mean"], nrm["phys_obs_std"])
    real_command = data["command"][real_ids, 0].to(device).float()
    if real_command.shape[-1] and "command_mean" in nrm:
        real_command = norm(real_command, nrm["command_mean"], nrm["command_std"])
    real_target = data["policy_action"][real_ids, 0].float().cpu()

    synth_n = int(replay["reward_conservative"].shape[0])
    synth_ids = sample_indices(synth_n, int(args.num_synthetic), int(args.seed) + 17)
    synth_state = replay["state"][synth_ids].to(device).float()
    synth_command = replay["command"][synth_ids].to(device).float()
    synth_target = replay["action"][synth_ids].float().cpu()
    reward_conservative = replay["reward_conservative"][synth_ids].float().cpu()
    reward_raw = replay.get("reward_raw", replay["reward"])[synth_ids].float().cpu()
    done = replay["done"][synth_ids].bool().cpu()
    uncertainty = replay.get("uncertainty", torch.zeros_like(replay["reward_conservative"]))[synth_ids].float().cpu()
    horizon_step = replay.get("horizon_step", torch.zeros_like(replay["reward_conservative"], dtype=torch.long))[synth_ids].long().cpu()
    weights = advantage_weights(reward_conservative, args.adv_temperature, args.weight_clip)
    active_weights = weights * (~done).float()

    bc_real_action = actor_actions(bc_actor, real_state, real_command)
    bc_synth_action = actor_actions(bc_actor, synth_state, synth_command)

    summary: dict[str, Any] = {
        "git_sha": git_value("rev-parse", "HEAD"),
        "git_branch": git_value("rev-parse", "--abbrev-ref", "HEAD"),
        "command": command_line(),
        "dataset": args.dataset,
        "metadata": args.metadata,
        "normalization": args.normalization,
        "bc_checkpoint": args.bc_checkpoint,
        "synthetic_replay": args.synthetic_replay,
        "real_sample_count": int(real_ids.numel()),
        "synthetic_sample_count": int(synth_ids.numel()),
        "adv_temperature": float(args.adv_temperature),
        "weight_clip": float(args.weight_clip),
        "synthetic": {
            "reward_raw": summarize_tensor(reward_raw),
            "reward_conservative": summarize_tensor(reward_conservative),
            "uncertainty": summarize_tensor(uncertainty),
            "adv_weight": summarize_tensor(weights),
            "active_adv_weight": summarize_tensor(active_weights),
            "done_fraction": float(done.float().mean().item()),
            "horizon_step": summarize_tensor(horizon_step.float()),
        },
        "synthetic_slices": {},
        "policies": {},
    }
    top_reward = reward_conservative >= torch.quantile(reward_conservative, 0.90)
    top_weight = weights >= torch.quantile(weights, 0.90)
    done_mask = done
    values = {
        "reward_conservative": reward_conservative,
        "reward_raw": reward_raw,
        "uncertainty": uncertainty,
        "adv_weight": weights,
        "active_adv_weight": active_weights,
        "horizon_step": horizon_step.float(),
        "replay_action_l2": synth_target.pow(2).mean(dim=-1).sqrt(),
    }
    summary["synthetic_slices"]["top_reward_decile"] = slice_stats(top_reward, values)
    summary["synthetic_slices"]["top_weight_decile"] = slice_stats(top_weight, values)
    summary["synthetic_slices"]["done"] = slice_stats(done_mask, values)
    summary["synthetic_slices"]["not_done"] = slice_stats(~done_mask, values)

    for label, path in zip(labels, policy_paths):
        actor, ckpt = load_actor(path, state_dim, command_dim, action_dim, device)
        real_action = actor_actions(actor, real_state, real_command)
        synth_action = actor_actions(actor, synth_state, synth_command)
        synth_delta = synth_action - bc_synth_action
        policy_summary = {
            "checkpoint": str(path),
            "checkpoint_kind": ckpt.get("checkpoint_kind", ""),
            "is_true_best_snapshot": bool(ckpt.get("is_true_best_snapshot", False)),
            "real_action_delta": action_delta_metrics(bc_real_action, real_action, target=real_target),
            "synthetic_action_delta": action_delta_metrics(bc_synth_action, synth_action, target=synth_target),
            "top_reward_synthetic_delta_l2": summarize_tensor(synth_delta[top_reward].pow(2).mean(dim=-1).sqrt()),
            "top_weight_synthetic_delta_l2": summarize_tensor(synth_delta[top_weight].pow(2).mean(dim=-1).sqrt()),
            "done_synthetic_delta_l2": summarize_tensor(synth_delta[done_mask].pow(2).mean(dim=-1).sqrt()) if bool(done_mask.any().item()) else {},
            "not_done_synthetic_delta_l2": summarize_tensor(synth_delta[~done_mask].pow(2).mean(dim=-1).sqrt()) if bool((~done_mask).any().item()) else {},
        }
        summary["policies"][label] = policy_summary

    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_md(Path(args.output_md), summary)
    print(json.dumps(summary, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
