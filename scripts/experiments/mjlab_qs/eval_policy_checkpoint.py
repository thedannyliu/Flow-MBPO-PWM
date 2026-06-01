#!/usr/bin/env python3
"""Evaluate a saved MJLab-QS policy checkpoint without rendering video."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any

import torch

os.environ.setdefault("MUJOCO_GL", "egl")
if os.environ.get("MUJOCO_GL") == "egl":
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    os.environ.setdefault("EGL_PLATFORM", "surfaceless")

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.experiments.mjlab_qs.collect_mjlab_qs_native_episodes import (  # noqa: E402
    split_obs,
    tensor_from_actor_obs,
)
from scripts.experiments.mjlab_qs.render_policy_rollout import (  # noqa: E402
    build_actor,
    command_line,
    git_branch,
    git_sha,
)
from scripts.experiments.mjlab_qs.run_offline_pwm_policy_extraction import (  # noqa: E402
    build_eval_env,
    load_data,
    norm,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--policy-checkpoint", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--checkpoint-kind", default="")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--eval-episodes", type=int, default=40)
    p.add_argument("--eval-num-envs", type=int, default=16)
    p.add_argument("--max-steps", type=int, default=1000)
    p.add_argument("--wandb-project", default="")
    p.add_argument("--wandb-group", default="")
    p.add_argument("--wandb-name", default="")
    p.add_argument("--disable-wandb", action="store_true")
    p.add_argument("--action-ramp-steps", type=int, default=0)
    p.add_argument("--baseline-return", type=float, default=None)
    p.add_argument("--baseline-length", type=float, default=None)
    p.add_argument("--baseline-fall", type=float, default=None)
    p.add_argument("--notes", default="")
    return p.parse_args()


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def apply_action_ramp(action: torch.Tensor, lengths: torch.Tensor, ramp_steps: int) -> tuple[torch.Tensor, torch.Tensor]:
    if ramp_steps <= 0:
        return action, torch.ones(action.shape[0], device=action.device, dtype=action.dtype)
    factor = ((lengths.to(action.dtype) + 1.0) / float(ramp_steps)).clamp(max=1.0)
    return action * factor.unsqueeze(-1), factor


def l2_mean_sqrt(value: torch.Tensor) -> float:
    if value.numel() == 0:
        return math.nan
    return float(value.float().pow(2).mean().sqrt().item())


def terminal_obs_summary(
    phys: torch.Tensor,
    cmd: torch.Tensor,
    z: torch.Tensor,
    action: torch.Tensor,
    raw_action: torch.Tensor,
    reward: torch.Tensor,
    idx: int,
    command_dim: int,
) -> dict[str, float]:
    phys_i = phys[idx].float()
    cmd_i = cmd[idx].float()
    return {
        "final_command_0": float(cmd_i[0].item()) if command_dim > 0 else math.nan,
        "final_command_1": float(cmd_i[1].item()) if command_dim > 1 else math.nan,
        "final_command_2": float(cmd_i[2].item()) if command_dim > 2 else math.nan,
        "final_obs_norm": l2_mean_sqrt(z[idx]),
        "final_action_l2": l2_mean_sqrt(action[idx]),
        "final_raw_action_l2": l2_mean_sqrt(raw_action[idx]),
        "final_prev_action_l2": l2_mean_sqrt(phys_i[67:96]) if phys_i.numel() >= 96 else math.nan,
        "final_base_lin_vel_l2": l2_mean_sqrt(phys_i[0:3]) if phys_i.numel() >= 3 else math.nan,
        "final_base_ang_vel_l2": l2_mean_sqrt(phys_i[3:6]) if phys_i.numel() >= 6 else math.nan,
        "final_projected_gravity_z": float(phys_i[8].item()) if phys_i.numel() >= 9 else math.nan,
        "final_reward": float(reward[idx].item()),
    }


def resolve_device(requested: str) -> torch.device:
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit(
            f"CUDA device requested ({requested}) but torch.cuda.is_available() is false. "
            "Check the Slurm GPU allocation and Python/conda environment."
        )
    return device


@torch.no_grad()
def collect_eval(actor, ckpt_args: dict[str, Any], nrm: dict[str, torch.Tensor], args: argparse.Namespace):
    device = torch.device(args.device)
    eval_args = argparse.Namespace(**ckpt_args)
    eval_args.device = args.device
    eval_args.eval_episodes = args.eval_episodes
    eval_args.eval_num_envs = args.eval_num_envs
    eval_args.episode_length = args.max_steps
    env, obs_td, obs_groups = build_eval_env(eval_args)
    returns = torch.zeros(args.eval_num_envs, device=device)
    lengths = torch.zeros(args.eval_num_envs, device=device)
    episode_rows: list[dict[str, Any]] = []
    command_dim = int(ckpt_args.get("command_dim", 3))
    command_position = ckpt_args.get("command_position", "tail")
    start_command = torch.zeros((args.eval_num_envs, command_dim), device=device)
    start_obs_norm = torch.zeros(args.eval_num_envs, device=device)
    start_action_l2 = torch.zeros(args.eval_num_envs, device=device)
    raw_start_action_l2 = torch.zeros(args.eval_num_envs, device=device)
    start_action_ramp_factor = torch.ones(args.eval_num_envs, device=device)
    try:
        while len(episode_rows) < args.eval_episodes:
            obs = tensor_from_actor_obs(obs_td, obs_groups)
            phys, cmd = split_obs(obs, command_dim, command_position)
            z = norm(phys.float(), nrm["phys_obs_mean"], nrm["phys_obs_std"])
            c = cmd.float()
            if c.shape[-1] and "command_mean" in nrm:
                c = norm(c, nrm["command_mean"], nrm["command_std"])
            raw_action = actor(z, c, deterministic=True).clamp(-1.0, 1.0)
            action, ramp_factor = apply_action_ramp(raw_action, lengths, int(args.action_ramp_steps))
            new_episode = lengths == 0
            if bool(new_episode.any().item()):
                start_command[new_episode] = cmd.float()[new_episode]
                start_obs_norm[new_episode] = z[new_episode].pow(2).mean(dim=-1).sqrt()
                start_action_l2[new_episode] = action[new_episode].pow(2).mean(dim=-1).sqrt()
                raw_start_action_l2[new_episode] = raw_action[new_episode].pow(2).mean(dim=-1).sqrt()
                start_action_ramp_factor[new_episode] = ramp_factor[new_episode]
            next_obs_td, reward, done, extras = env.step(action)
            reward = reward.to(device).float().reshape(-1)
            done = done.to(device).bool().reshape(-1)
            time_out = extras.get("time_outs", torch.zeros_like(done)).to(device).bool().reshape(-1)
            terminated = done & (~time_out)
            returns = returns + reward
            lengths = lengths + 1.0
            for idx in done.nonzero(as_tuple=False).reshape(-1).tolist():
                episode_rows.append(
                    {
                        "episode": len(episode_rows),
                        "env_slot": idx,
                        "return": float(returns[idx].item()),
                        "length": float(lengths[idx].item()),
                        "terminated": int(terminated[idx].item()),
                        "truncated": int(time_out[idx].item()),
                        "start_command_0": float(start_command[idx, 0].item()) if command_dim > 0 else math.nan,
                        "start_command_1": float(start_command[idx, 1].item()) if command_dim > 1 else math.nan,
                        "start_command_2": float(start_command[idx, 2].item()) if command_dim > 2 else math.nan,
                        "start_obs_norm": float(start_obs_norm[idx].item()),
                        "start_action_l2": float(start_action_l2[idx].item()),
                        "raw_start_action_l2": float(raw_start_action_l2[idx].item()),
                        "start_action_ramp_factor": float(start_action_ramp_factor[idx].item()),
                        **terminal_obs_summary(phys, cmd, z, action, raw_action, reward, idx, command_dim),
                    }
                )
                returns[idx] = 0.0
                lengths[idx] = 0.0
                if len(episode_rows) >= args.eval_episodes:
                    break
            obs_td = next_obs_td
    finally:
        env.close()
    return episode_rows[: args.eval_episodes]


def summarize(rows: list[dict[str, Any]]) -> dict[str, float]:
    returns = torch.tensor([row["return"] for row in rows], dtype=torch.float32)
    lengths = torch.tensor([row["length"] for row in rows], dtype=torch.float32)
    falls = torch.tensor([row["terminated"] for row in rows], dtype=torch.float32)
    timeouts = torch.tensor([row["truncated"] for row in rows], dtype=torch.float32)
    return {
        "return_mean": float(returns.mean().item()),
        "return_std": float(returns.std(unbiased=False).item()),
        "return_min": float(returns.min().item()),
        "return_p10": float(torch.quantile(returns, 0.10).item()),
        "return_median": float(torch.quantile(returns, 0.50).item()),
        "episode_length_mean": float(lengths.mean().item()),
        "episode_length_std": float(lengths.std(unbiased=False).item()),
        "episode_length_min": float(lengths.min().item()),
        "episode_length_p10": float(torch.quantile(lengths, 0.10).item()),
        "episode_length_median": float(torch.quantile(lengths, 0.50).item()),
        "fall_rate_mean": float(falls.mean().item()),
        "timeout_rate_mean": float(timeouts.mean().item()),
    }


def add_baseline_gate(summary: dict[str, Any], args: argparse.Namespace) -> None:
    ret = summary.get("return_mean")
    length = summary.get("episode_length_mean")
    fall = summary.get("fall_rate_mean")
    configured = args.baseline_return is not None and args.baseline_length is not None and args.baseline_fall is not None
    summary["baseline_gate_configured"] = bool(configured)
    if args.baseline_return is not None and ret is not None:
        baseline = float(args.baseline_return)
        summary["baseline_return"] = baseline
        summary["return_gap_to_baseline"] = float(ret) - baseline
        summary["return_gate_pass"] = bool(float(ret) >= baseline)
    if args.baseline_length is not None and length is not None:
        baseline = float(args.baseline_length)
        summary["baseline_length"] = baseline
        summary["length_gap_to_baseline"] = float(length) - baseline
        summary["length_gate_pass"] = bool(float(length) >= baseline)
    if args.baseline_fall is not None and fall is not None:
        baseline = float(args.baseline_fall)
        summary["baseline_fall"] = baseline
        summary["fall_gap_to_baseline"] = float(fall) - baseline
        summary["fall_gate_pass"] = bool(float(fall) < baseline)
    if configured:
        summary["baseline_gate_pass"] = bool(
            summary.get("return_gate_pass") and summary.get("length_gate_pass") and summary.get("fall_gate_pass")
        )


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    args.device = str(device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    ckpt = torch.load(args.policy_checkpoint, map_location="cpu", weights_only=False)
    ckpt_args = ckpt["args"]
    data, _metadata, nrm, _train_idx = load_data(argparse.Namespace(**ckpt_args), device)
    actor = build_actor(
        ckpt,
        state_dim=int(data["phys_obs"].shape[-1]),
        command_dim=int(data["command"].shape[-1]),
        action_dim=int(data["policy_action"].shape[-1]),
        device=device,
    )
    run = None
    if args.wandb_project and not args.disable_wandb:
        import wandb

        run = wandb.init(
            project=args.wandb_project,
            group=args.wandb_group or ckpt_args.get("wandb_group", ""),
            name=args.wandb_name or f"{ckpt_args.get('wm_method')}_{ckpt_args.get('policy_type')}_seed{ckpt_args.get('seed')}_eval",
            job_type="policy_real_eval",
            config={
                **ckpt_args,
                "policy_checkpoint": args.policy_checkpoint,
                "checkpoint_kind": args.checkpoint_kind or ckpt.get("checkpoint_kind", ""),
                "eval_episodes": args.eval_episodes,
                "eval_num_envs": args.eval_num_envs,
                "max_steps": args.max_steps,
                "action_ramp_steps": args.action_ramp_steps,
                "baseline_return": args.baseline_return,
                "baseline_length": args.baseline_length,
                "baseline_fall": args.baseline_fall,
                "notes": args.notes,
                "git_sha": git_sha(),
                "git_branch": git_branch(),
                "command": command_line(),
            },
        )
    t0 = time.time()
    rows = collect_eval(actor, ckpt_args, nrm, args)
    write_csv(
        output_dir / "eval_episodes.csv",
        rows,
        [
            "episode",
            "env_slot",
            "return",
            "length",
            "terminated",
            "truncated",
            "start_command_0",
            "start_command_1",
            "start_command_2",
            "start_obs_norm",
            "start_action_l2",
            "raw_start_action_l2",
            "start_action_ramp_factor",
            "final_command_0",
            "final_command_1",
            "final_command_2",
            "final_obs_norm",
            "final_action_l2",
            "final_raw_action_l2",
            "final_prev_action_l2",
            "final_base_lin_vel_l2",
            "final_base_ang_vel_l2",
            "final_projected_gravity_z",
            "final_reward",
        ],
    )
    summary = {
        "policy_checkpoint": args.policy_checkpoint,
        "checkpoint_kind": args.checkpoint_kind or ckpt.get("checkpoint_kind", ""),
        "eval_episodes": args.eval_episodes,
        "eval_num_envs": args.eval_num_envs,
        "max_steps": args.max_steps,
        "action_ramp_steps": args.action_ramp_steps,
        "terminal_obs_summary_fields": [
            "final_obs_norm",
            "final_action_l2",
            "final_prev_action_l2",
            "final_base_lin_vel_l2",
            "final_base_ang_vel_l2",
            "final_projected_gravity_z",
            "final_reward",
        ],
        "wall_clock_seconds": time.time() - t0,
        "git_sha": git_sha(),
        "git_branch": git_branch(),
        "command": command_line(),
        "notes": args.notes,
        "dataset": ckpt_args.get("dataset", ""),
        "metadata": ckpt_args.get("metadata", ""),
        "normalization": ckpt_args.get("normalization", ""),
        "seed": ckpt_args.get("seed", ""),
        "task_id": ckpt_args.get("task_id", ""),
        "wm_method": ckpt_args.get("wm_method", ""),
        "policy_type": ckpt_args.get("policy_type", ""),
        **summarize(rows),
    }
    add_baseline_gate(summary, args)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    if run is not None:
        run.log({f"eval/{key}": value for key, value in summary.items() if isinstance(value, (int, float))})
        run.summary.update(summary)
        run.finish()
    print(json.dumps(summary, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
