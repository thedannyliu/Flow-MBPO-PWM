#!/usr/bin/env python3
"""Prepare conservative Flow-MBPO v0 synthetic replay from a smoke buffer."""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from scripts.experiments.mjlab_qs.render_policy_rollout import command_line, git_branch, git_sha  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--synthetic-buffer", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--lambda-uncertainty", type=float, default=0.0)
    p.add_argument("--uncertainty-quantile-termination", type=float, default=0.0)
    p.add_argument(
        "--truncate-rollouts-after-done",
        action="store_true",
        help="For multi-step synthetic rollouts, mark all transitions after the first model/uncertainty done for a start as done.",
    )
    p.add_argument("--max-transitions", type=int, default=0)
    p.add_argument("--seed", type=int, default=0)
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


def prepare_replay(
    buffer: dict[str, torch.Tensor],
    lambda_uncertainty: float,
    termination_quantile: float,
    truncate_rollouts_after_done: bool,
) -> dict[str, torch.Tensor]:
    reward = buffer["reward"].float()
    next_unc = buffer["next_state_uncertainty"].float()
    reward_unc = buffer["reward_uncertainty"].float()
    uncertainty = next_unc + reward_unc
    done = buffer["done"].bool()
    uncertainty_done = torch.zeros_like(done)
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
    replay["reward_conservative"] = reward - float(lambda_uncertainty) * uncertainty
    replay["done_model"] = done
    replay["done_uncertainty"] = uncertainty_done
    done_combined = done | uncertainty_done
    if truncate_rollouts_after_done:
        post_done = rollout_post_done_mask(buffer, done_combined)
    else:
        post_done = torch.zeros_like(done_combined)
    replay["done_post_first_done"] = post_done
    replay["done"] = done_combined | post_done
    replay["uncertainty_termination_threshold"] = threshold.repeat(reward.shape[0])
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
        args.lambda_uncertainty,
        args.uncertainty_quantile_termination,
        args.truncate_rollouts_after_done,
    )

    replay_path = output_dir / "synthetic_replay.pt"
    torch.save(replay, replay_path)
    done = replay["done"].float()
    summary: dict[str, Any] = {
        "git_sha": git_sha(),
        "git_branch": git_branch(),
        "command": command_line(),
        "synthetic_buffer": args.synthetic_buffer,
        "output_dir": str(output_dir),
        "replay_path": str(replay_path),
        "lambda_uncertainty": float(args.lambda_uncertainty),
        "uncertainty_quantile_termination": float(args.uncertainty_quantile_termination),
        "truncate_rollouts_after_done": bool(args.truncate_rollouts_after_done),
        "max_transitions": int(args.max_transitions),
        "seed": int(args.seed),
        "transitions": int(replay["reward"].shape[0]),
        "raw_reward": summarize_tensor(replay["reward_raw"]),
        "conservative_reward": summarize_tensor(replay["reward_conservative"]),
        "uncertainty": summarize_tensor(replay["uncertainty"]),
        "model_done_fraction": float(replay["done_model"].float().mean().item()),
        "uncertainty_done_fraction": float(replay["done_uncertainty"].float().mean().item()),
        "post_first_done_fraction": float(replay["done_post_first_done"].float().mean().item()),
        "done_fraction": float(done.mean().item()),
        "wall_clock_seconds": time.time() - t0,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
