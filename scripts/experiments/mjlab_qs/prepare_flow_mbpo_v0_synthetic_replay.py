#!/usr/bin/env python3
"""Prepare conservative Flow-MBPO synthetic replay from a smoke buffer."""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from scripts.experiments.mjlab_qs.render_policy_rollout import command_line, git_branch, git_sha  # noqa: E402
from scripts.experiments.mjlab_qs.add_flow_mbpo_support_penalty import (  # noqa: E402
    load_norm,
    nearest_l2_per_dim,
    real_features,
    replay_features,
    select_indices,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--synthetic-buffer", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--lambda-uncertainty", type=float, default=0.0)
    p.add_argument("--lambda-fall", type=float, default=0.0)
    p.add_argument("--uncertainty-quantile-termination", type=float, default=0.0)
    p.add_argument("--done-threshold", type=float, default=0.5)
    p.add_argument("--fall-threshold", type=float, default=0.5)
    p.add_argument(
        "--support-risk-termination",
        action="store_true",
        help="Mark synthetic transitions outside a calibrated real-data support threshold as done.",
    )
    p.add_argument("--support-dataset", default="")
    p.add_argument("--support-metadata", default="")
    p.add_argument("--support-normalization", default="")
    p.add_argument("--support-risk-penalty-weight", type=float, default=0.0)
    p.add_argument("--support-max-rows", type=int, default=20000)
    p.add_argument("--support-probe-rows", type=int, default=4096)
    p.add_argument("--support-threshold", type=float, default=-1.0)
    p.add_argument("--support-threshold-quantile", type=float, default=0.90)
    p.add_argument("--support-distance-batch-size", type=int, default=256)
    p.add_argument("--support-split", default="train", choices=["train", "val", "test"])
    p.add_argument("--support-quality-filter", default="expert,expert_noisy")
    p.add_argument("--support-state-weight", type=float, default=1.0)
    p.add_argument("--support-command-weight", type=float, default=1.0)
    p.add_argument("--support-action-weight", type=float, default=1.0)
    p.add_argument(
        "--truncate-rollouts-after-done",
        action="store_true",
        help="For multi-step synthetic rollouts, mark all transitions after the first model/uncertainty done for a start as done.",
    )
    p.add_argument("--max-transitions", type=int, default=0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--notes", default="")
    p.add_argument("--enable-wandb", action="store_true")
    p.add_argument("--wandb-project", default="flow-mbpo-mjlab-flow-mbpo-v0-replay")
    p.add_argument("--wandb-group", default="")
    p.add_argument("--wandb-name", default="")
    return p.parse_args()


def summarize_tensor(x: torch.Tensor) -> dict[str, float]:
    x = x.detach().float().reshape(-1).cpu()
    finite = x[torch.isfinite(x)]
    if finite.numel() == 0:
        return {"mean": math.nan, "std": math.nan, "min": math.nan, "p90": math.nan, "max": math.nan}
    return {
        "mean": float(finite.mean().item()),
        "std": float(finite.std(unbiased=False).item()),
        "min": float(finite.min().item()),
        "p90": float(torch.quantile(finite, 0.90).item()),
        "max": float(finite.max().item()),
    }


def tensor_shapes(replay: dict[str, torch.Tensor]) -> dict[str, list[int]]:
    return {key: list(value.shape) for key, value in replay.items() if isinstance(value, torch.Tensor)}


def require_keys(buffer: dict[str, torch.Tensor]) -> None:
    required = [
        "state",
        "command",
        "action",
        "reward",
        "next_state",
        "next_state_uncertainty",
        "reward_uncertainty",
        "done",
    ]
    missing = [key for key in required if key not in buffer]
    if missing:
        raise ValueError(f"Synthetic buffer is missing required keys: {missing}")


def subset_buffer(buffer: dict[str, torch.Tensor], max_transitions: int, seed: int) -> dict[str, torch.Tensor]:
    if max_transitions <= 0:
        return buffer
    n = int(buffer["reward"].shape[0])
    if n <= max_transitions:
        return buffer
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(n, generator=generator)[:max_transitions]
    out: dict[str, torch.Tensor] = {}
    for key, value in buffer.items():
        out[key] = value[indices] if isinstance(value, torch.Tensor) and value.shape[:1] == (n,) else value
    return out


def rollout_post_done_mask(buffer: dict[str, torch.Tensor], done: torch.Tensor) -> torch.Tensor:
    if "start_index" not in buffer or "horizon_step" not in buffer:
        raise ValueError("Rollout truncation requires start_index and horizon_step in the synthetic buffer")
    start_index = buffer["start_index"].reshape(-1).cpu()
    horizon_step = buffer["horizon_step"].reshape(-1).cpu()
    done_cpu = done.reshape(-1).cpu().bool()
    if start_index.numel() != done_cpu.numel() or horizon_step.numel() != done_cpu.numel():
        raise ValueError("start_index, horizon_step, and done must have the same leading dimension")
    order = torch.argsort(start_index * (int(horizon_step.max().item()) + 1) + horizon_step)
    post_done = torch.zeros_like(done_cpu)
    active_start: int | None = None
    seen_done = False
    for raw_idx in order.tolist():
        sid = int(start_index[raw_idx].item())
        if active_start != sid:
            active_start = sid
            seen_done = False
        if seen_done:
            post_done[raw_idx] = True
        if bool(done_cpu[raw_idx].item()):
            seen_done = True
    return post_done.to(done.device)


def support_args(args: argparse.Namespace) -> SimpleNamespace:
    return SimpleNamespace(
        split=args.support_split,
        quality_filter=args.support_quality_filter,
        state_weight=float(args.support_state_weight),
        command_weight=float(args.support_command_weight),
        action_weight=float(args.support_action_weight),
    )


def support_risk_fields(replay: dict[str, torch.Tensor], args: argparse.Namespace) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    if not bool(args.support_risk_termination):
        reward = replay["reward_raw"].float()
        fields = {
            "support_risk_done": torch.zeros_like(reward, dtype=torch.bool),
            "support_risk_distance": torch.zeros_like(reward),
            "support_risk_threshold": torch.full_like(reward, float("inf")),
            "support_risk_penalty": torch.zeros_like(reward),
        }
        summary = {
            "support_risk_termination": False,
            "support_risk_done_fraction": 0.0,
            "support_risk_threshold": None,
            "support_risk_penalty_weight": float(args.support_risk_penalty_weight),
        }
        return fields, summary

    for key, value in [
        ("--support-dataset", args.support_dataset),
        ("--support-metadata", args.support_metadata),
        ("--support-normalization", args.support_normalization),
    ]:
        if not value:
            raise ValueError(f"{key} is required when --support-risk-termination is enabled")

    data = torch.load(args.support_dataset, map_location="cpu", weights_only=False)
    if not isinstance(data, dict):
        raise TypeError(f"{args.support_dataset} must contain a dict")
    metadata = json.loads(Path(args.support_metadata).read_text(encoding="utf-8"))
    nrm = load_norm(Path(args.support_normalization))
    sargs = support_args(args)
    selected = select_indices(data, metadata, sargs)
    generator = torch.Generator().manual_seed(int(args.seed))
    perm = selected[torch.randperm(selected.numel(), generator=generator)]
    support_n = min(int(args.support_max_rows), int(perm.numel()))
    probe_n = min(int(args.support_probe_rows), max(0, int(perm.numel()) - support_n))
    support_indices = perm[:support_n]
    probe_indices = perm[support_n : support_n + probe_n]
    if support_indices.numel() == 0:
        raise ValueError("Need a non-empty support set")
    if float(args.support_threshold) < 0.0 and probe_indices.numel() == 0:
        raise ValueError("Need non-empty support probe rows when --support-threshold is not set")

    support_feat = real_features(data, support_indices, nrm, sargs)
    synthetic_feat = replay_features(replay, sargs)
    if float(args.support_threshold) >= 0.0:
        threshold = torch.tensor(float(args.support_threshold), dtype=torch.float32)
        real_probe_distance = torch.empty(0, dtype=torch.float32)
    else:
        probe_feat = real_features(data, probe_indices, nrm, sargs)
        real_probe_distance = nearest_l2_per_dim(probe_feat, support_feat, int(args.support_distance_batch_size))
        q = min(max(float(args.support_threshold_quantile), 0.0), 1.0)
        threshold = torch.quantile(real_probe_distance[torch.isfinite(real_probe_distance)], q)
    distance = nearest_l2_per_dim(synthetic_feat, support_feat, int(args.support_distance_batch_size))
    penalty = F.relu(distance - threshold)
    fields = {
        "support_risk_done": distance > threshold,
        "support_risk_distance": distance,
        "support_risk_threshold": threshold.repeat(distance.shape[0]),
        "support_risk_penalty": penalty,
    }
    summary = {
        "support_risk_termination": True,
        "support_dataset": args.support_dataset,
        "support_metadata": args.support_metadata,
        "support_normalization": args.support_normalization,
        "support_split": args.support_split,
        "support_quality_filter": args.support_quality_filter,
        "support_rows": int(support_indices.numel()),
        "support_probe_rows": int(probe_indices.numel()),
        "support_feature_dim": int(support_feat.shape[-1]),
        "support_threshold": float(threshold.item()),
        "support_threshold_quantile": float(args.support_threshold_quantile),
        "support_state_weight": float(args.support_state_weight),
        "support_command_weight": float(args.support_command_weight),
        "support_action_weight": float(args.support_action_weight),
        "support_risk_penalty_weight": float(args.support_risk_penalty_weight),
        "support_real_probe_distance": summarize_tensor(real_probe_distance)
        if real_probe_distance.numel() > 0
        else None,
        "support_risk_distance": summarize_tensor(distance),
        "support_risk_penalty": summarize_tensor(penalty),
        "support_risk_done_fraction": float(fields["support_risk_done"].float().mean().item()),
    }
    return fields, summary


def prepare_replay(
    buffer: dict[str, torch.Tensor],
    args: argparse.Namespace,
    lambda_uncertainty: float,
    lambda_fall: float,
    termination_quantile: float,
    done_threshold: float,
    fall_threshold: float,
    truncate_rollouts_after_done: bool,
) -> dict[str, torch.Tensor]:
    reward = buffer["reward"].float()
    next_unc = buffer["next_state_uncertainty"].float()
    reward_unc = buffer["reward_uncertainty"].float()
    uncertainty = next_unc + reward_unc
    done_probability = buffer.get("done_probability", torch.zeros_like(reward)).float()
    fall_prob = buffer.get("fall_prob", buffer.get("fall_probability", done_probability)).float()
    done_model = buffer["done"].bool() | (done_probability >= float(done_threshold))
    fall_done = fall_prob >= float(fall_threshold)
    uncertainty_done = torch.zeros_like(done_model)
    threshold = torch.tensor(float("inf"))
    if termination_quantile > 0.0:
        q = min(max(float(termination_quantile), 0.0), 1.0)
        finite_uncertainty = uncertainty[torch.isfinite(uncertainty)]
        if finite_uncertainty.numel() > 0 and finite_uncertainty.max() > finite_uncertainty.min():
            threshold = torch.quantile(finite_uncertainty, q)
            uncertainty_done = uncertainty >= threshold
    replay = dict(buffer)
    replay["uncertainty"] = uncertainty
    replay["reward_raw"] = reward
    replay["done_probability"] = done_probability
    replay["fall_prob"] = fall_prob
    replay["reward_conservative_pre_support_risk"] = (
        reward - float(lambda_uncertainty) * uncertainty - float(lambda_fall) * fall_prob
    )
    replay["done_model"] = done_model
    replay["done_fall"] = fall_done
    replay["done_uncertainty"] = uncertainty_done
    support_fields, support_summary = support_risk_fields(replay, args)
    replay.update(support_fields)
    replay["reward_conservative"] = replay["reward_conservative_pre_support_risk"] - float(
        args.support_risk_penalty_weight
    ) * replay["support_risk_penalty"]
    done_combined = done_model | fall_done | uncertainty_done | replay["support_risk_done"].bool()
    if truncate_rollouts_after_done:
        post_done = rollout_post_done_mask(buffer, done_combined)
    else:
        post_done = torch.zeros_like(done_combined)
    replay["done_post_first_done"] = post_done
    replay["done"] = done_combined | post_done
    replay["uncertainty_termination_threshold"] = threshold.repeat(reward.shape[0])
    replay["done_threshold"] = torch.full_like(reward, float(done_threshold))
    replay["fall_threshold"] = torch.full_like(reward, float(fall_threshold))
    replay["_support_risk_summary"] = support_summary
    return replay


def main() -> None:
    args = parse_args()
    t0 = time.time()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    buffer = torch.load(args.synthetic_buffer, map_location="cpu", weights_only=False)
    if not isinstance(buffer, dict):
        raise TypeError(f"{args.synthetic_buffer} must contain a dict")
    require_keys(buffer)
    buffer = subset_buffer(buffer, args.max_transitions, args.seed)
    replay = prepare_replay(
        buffer,
        args,
        args.lambda_uncertainty,
        args.lambda_fall,
        args.uncertainty_quantile_termination,
        args.done_threshold,
        args.fall_threshold,
        args.truncate_rollouts_after_done,
    )

    replay_path = output_dir / "synthetic_replay.pt"
    support_risk_summary = replay.pop("_support_risk_summary")
    torch.save(replay, replay_path)
    input_metadata_path = Path(args.synthetic_buffer).with_name("synthetic_buffer_metadata.json")
    input_metadata: dict[str, Any] | None = None
    if input_metadata_path.exists():
        input_metadata = json.loads(input_metadata_path.read_text(encoding="utf-8"))
    done = replay["done"].float()
    summary: dict[str, Any] = {
        "git_sha": git_sha(),
        "git_branch": git_branch(),
        "command": command_line(),
        "notes": args.notes,
        "enable_wandb": bool(args.enable_wandb),
        "wandb_project": args.wandb_project,
        "wandb_group": args.wandb_group,
        "wandb_name": args.wandb_name,
        "synthetic_buffer": args.synthetic_buffer,
        "synthetic_buffer_metadata": str(input_metadata_path) if input_metadata_path.exists() else "",
        "output_dir": str(output_dir),
        "replay_path": str(replay_path),
        "replay_metadata_path": str(output_dir / "synthetic_replay_metadata.json"),
        "lambda_uncertainty": float(args.lambda_uncertainty),
        "lambda_fall": float(args.lambda_fall),
        "uncertainty_quantile_termination": float(args.uncertainty_quantile_termination),
        "done_threshold": float(args.done_threshold),
        "fall_threshold": float(args.fall_threshold),
        **support_risk_summary,
        "truncate_rollouts_after_done": bool(args.truncate_rollouts_after_done),
        "max_transitions": int(args.max_transitions),
        "seed": int(args.seed),
        "transitions": int(replay["reward"].shape[0]),
        "raw_reward": summarize_tensor(replay["reward_raw"]),
        "conservative_reward_pre_support_risk": summarize_tensor(replay["reward_conservative_pre_support_risk"]),
        "conservative_reward": summarize_tensor(replay["reward_conservative"]),
        "uncertainty": summarize_tensor(replay["uncertainty"]),
        "done_probability": summarize_tensor(replay["done_probability"]),
        "fall_prob": summarize_tensor(replay["fall_prob"]),
        "model_done_fraction": float(replay["done_model"].float().mean().item()),
        "fall_done_fraction": float(replay["done_fall"].float().mean().item()),
        "uncertainty_done_fraction": float(replay["done_uncertainty"].float().mean().item()),
        "support_risk_done_fraction": float(replay["support_risk_done"].float().mean().item()),
        "post_first_done_fraction": float(replay["done_post_first_done"].float().mean().item()),
        "done_fraction": float(done.mean().item()),
        "wall_clock_seconds": time.time() - t0,
    }
    replay_metadata = {
        "artifact": "synthetic_replay.pt",
        "artifact_path": str(replay_path),
        "git_sha": summary["git_sha"],
        "git_branch": summary["git_branch"],
        "command": summary["command"],
        "notes": args.notes,
        "synthetic_buffer": args.synthetic_buffer,
        "synthetic_buffer_metadata": str(input_metadata_path) if input_metadata_path.exists() else "",
        "synthetic_buffer_notes": input_metadata.get("notes", "") if input_metadata is not None else "",
        "lambda_uncertainty": float(args.lambda_uncertainty),
        "lambda_fall": float(args.lambda_fall),
        "uncertainty_quantile_termination": float(args.uncertainty_quantile_termination),
        "done_threshold": float(args.done_threshold),
        "fall_threshold": float(args.fall_threshold),
        "support_risk_termination": bool(support_risk_summary.get("support_risk_termination", False)),
        "support_risk_penalty_weight": float(args.support_risk_penalty_weight),
        "truncate_rollouts_after_done": bool(args.truncate_rollouts_after_done),
        "max_transitions": int(args.max_transitions),
        "seed": int(args.seed),
        "transitions": int(replay["reward"].shape[0]),
        "done_fraction": float(done.mean().item()),
        "tensor_shapes": tensor_shapes(replay),
    }
    (output_dir / "synthetic_replay_metadata.json").write_text(
        json.dumps(replay_metadata, indent=2),
        encoding="utf-8",
    )
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    if args.enable_wandb:
        import wandb

        wandb_init = getattr(wandb, "init", None)
        if wandb_init is None:
            raise RuntimeError("W&B logging requested but the imported wandb module has no init()")
        run = wandb_init(
            project=args.wandb_project,
            group=args.wandb_group or "flow_mbpo_v0_replay",
            name=args.wandb_name or f"seed{args.seed}_synthetic_replay",
            job_type="flow_mbpo_v0_synthetic_replay",
            config=summary,
        )
        run.log(
            {
                "replay/transitions": summary["transitions"],
                "replay/raw_reward_mean": summary["raw_reward"]["mean"],
                "replay/conservative_reward_mean": summary["conservative_reward"]["mean"],
                "replay/uncertainty_mean": summary["uncertainty"]["mean"],
                "replay/done_fraction": summary["done_fraction"],
                "replay/model_done_fraction": summary["model_done_fraction"],
                "replay/fall_done_fraction": summary["fall_done_fraction"],
                "replay/uncertainty_done_fraction": summary["uncertainty_done_fraction"],
                "replay/support_risk_done_fraction": summary["support_risk_done_fraction"],
                "replay/post_first_done_fraction": summary["post_first_done_fraction"],
            }
        )
        run.finish()
    print(json.dumps(summary, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
