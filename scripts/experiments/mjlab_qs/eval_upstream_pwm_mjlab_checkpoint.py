#!/usr/bin/env python3
"""Evaluate a full upstream PWM checkpoint on the real MJLab environment."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf


os.environ.setdefault("MUJOCO_GL", "egl")
if os.environ.get("MUJOCO_GL") == "egl":
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    os.environ.setdefault("EGL_PLATFORM", "surfaceless")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--hydra-run-dir", required=True)
    p.add_argument("--policy-checkpoint", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--checkpoint-kind", default="")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--eval-episodes", type=int, default=8)
    p.add_argument("--eval-num-envs", type=int, default=16)
    p.add_argument("--max-steps", type=int, default=1000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--baseline-return", type=float, default=None)
    p.add_argument("--baseline-length", type=float, default=None)
    p.add_argument("--baseline-fall", type=float, default=None)
    p.add_argument("--wandb-project", default="")
    p.add_argument("--wandb-group", default="")
    p.add_argument("--wandb-name", default="")
    p.add_argument("--disable-wandb", action="store_true")
    p.add_argument("--notes", default="")
    return p.parse_args()


def git_sha() -> str:
    override = os.environ.get("FLOW_MBPO_SUBMIT_GIT_SHA", "").strip()
    if override:
        return override
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def git_branch() -> str:
    override = os.environ.get("FLOW_MBPO_SUBMIT_GIT_BRANCH", "").strip()
    if override:
        return override
    try:
        return subprocess.check_output(["git", "rev-parse", "--abbrev-ref", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def command_line() -> str:
    return " ".join([sys.executable, *sys.argv])


def resolve_device(requested: str) -> torch.device:
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit(
            f"CUDA device requested ({requested}) but torch.cuda.is_available() is false. "
            "Check the Slurm GPU allocation and Python runtime."
        )
    return device


def mean_std(values: list[float]) -> tuple[float, float]:
    if not values:
        return math.nan, math.nan
    tensor = torch.tensor(values, dtype=torch.float32)
    return float(tensor.mean().item()), float(tensor.std(unbiased=False).item())


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "episode",
        "env_slot",
        "return",
        "length",
        "terminated",
        "truncated",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    hydra_run_dir = Path(args.hydra_run_dir)
    cfg_path = hydra_run_dir / ".hydra" / "config.yaml"
    if not cfg_path.exists():
        raise SystemExit(f"Missing Hydra config: {cfg_path}")

    cfg = OmegaConf.load(cfg_path)
    cfg.general.seed = args.seed
    cfg.general.device = str(device)
    cfg.general.run_wandb = False
    cfg.env.config.num_envs = args.eval_num_envs
    cfg.env.config.device = str(device)
    cfg.env.config.seed = args.seed
    cfg.env.config.episode_length = args.max_steps
    cfg.env.config.no_grad = True

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    logdir = output_dir / "env_logs"
    logdir.mkdir(parents=True, exist_ok=True)

    env = instantiate(cfg.env.config, logdir=str(logdir))
    agent = instantiate(
        cfg.alg,
        env=env,
        obs_dim=env.num_obs,
        act_dim=env.num_actions,
        logdir=str(output_dir / "agent_logs"),
        log=False,
    )
    agent.load(args.policy_checkpoint, buffer=False)
    agent.actor.to(device).eval()
    agent.wm.to(device).eval()

    rows: list[dict[str, Any]] = []
    returns = torch.zeros(env.num_envs, dtype=torch.float32, device=device)
    lengths = torch.zeros(env.num_envs, dtype=torch.float32, device=device)
    obs = env.reset()
    try:
        with torch.no_grad():
            while len(rows) < args.eval_episodes:
                obs_in = obs
                if agent.obs_rms is not None:
                    obs_in = agent.obs_rms.normalize(obs_in)
                z = agent.wm.encode(obs_in, task=None)
                action = torch.tanh(agent.actor(z, deterministic=True)).clamp(-1.0, 1.0)
                obs, reward, done, info = env.step(action)
                reward = reward.to(device).float().reshape(-1)
                done = done.to(device).bool().reshape(-1)
                truncated = info.get("truncation", torch.zeros_like(done)).to(device).bool().reshape(-1)
                terminated = info.get("termination", done & (~truncated)).to(device).bool().reshape(-1)
                returns += reward
                lengths += 1.0
                for idx in done.nonzero(as_tuple=False).reshape(-1).tolist():
                    rows.append(
                        {
                            "episode": len(rows),
                            "env_slot": idx,
                            "return": float(returns[idx].item()),
                            "length": float(lengths[idx].item()),
                            "terminated": int(terminated[idx].item()),
                            "truncated": int(truncated[idx].item()),
                        }
                    )
                    returns[idx] = 0.0
                    lengths[idx] = 0.0
                    if len(rows) >= args.eval_episodes:
                        break
    finally:
        env.close()

    rows = rows[: args.eval_episodes]
    return_values = [float(row["return"]) for row in rows]
    length_values = [float(row["length"]) for row in rows]
    fall_values = [float(row["terminated"]) for row in rows]
    timeout_values = [float(row["truncated"]) for row in rows]
    return_mean, return_std = mean_std(return_values)
    length_mean, length_std = mean_std(length_values)
    fall_mean, _ = mean_std(fall_values)
    timeout_mean, _ = mean_std(timeout_values)
    baseline_gate_pass = None
    if args.baseline_return is not None and args.baseline_length is not None and args.baseline_fall is not None:
        baseline_gate_pass = (
            return_mean >= args.baseline_return
            and length_mean >= args.baseline_length
            and fall_mean < args.baseline_fall
        )

    summary = {
        "policy_checkpoint": str(Path(args.policy_checkpoint)),
        "checkpoint_kind": args.checkpoint_kind,
        "hydra_run_dir": str(hydra_run_dir),
        "task_id": cfg.env.config.task_id,
        "eval_episodes": args.eval_episodes,
        "eval_num_envs": args.eval_num_envs,
        "max_steps": args.max_steps,
        "seed": args.seed,
        "return_mean": return_mean,
        "return_std": return_std,
        "episode_length_mean": length_mean,
        "episode_length_std": length_std,
        "fall_rate_mean": fall_mean,
        "timeout_rate_mean": timeout_mean,
        "baseline_return": args.baseline_return,
        "baseline_length": args.baseline_length,
        "baseline_fall": args.baseline_fall,
        "baseline_gate_pass": baseline_gate_pass,
        "git_sha": git_sha(),
        "git_branch": git_branch(),
        "command": command_line(),
        "notes": args.notes,
    }
    run = None
    if args.wandb_project and not args.disable_wandb:
        import wandb

        run = wandb.init(
            project=args.wandb_project,
            group=args.wandb_group,
            name=args.wandb_name or f"upstream_pwm_mjlab_{args.checkpoint_kind}_eval{args.eval_episodes}",
            job_type="upstream_pwm_mjlab_eval",
            config=summary,
        )
        run.log(
            {
                "eval/return_mean": return_mean,
                "eval/episode_length_mean": length_mean,
                "eval/fall_rate_mean": fall_mean,
                "eval/timeout_rate_mean": timeout_mean,
                "eval/baseline_gate_pass": baseline_gate_pass,
            }
        )
        summary["wandb_url"] = run.url
        run.summary.update(summary)
    write_csv(output_dir / "eval_episodes.csv", rows)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    if run is not None:
        run.finish()
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
