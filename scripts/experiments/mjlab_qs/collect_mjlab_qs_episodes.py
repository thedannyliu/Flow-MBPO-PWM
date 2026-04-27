#!/usr/bin/env python3
"""Collect raw MJLab-QS episode shards.

This collector is intentionally stricter than the old Phase-1 fixed-window
collector. It stores raw episodes first, separates physical observations from
commands, and records both normalized policy actions and env-applied actions.
"""

from __future__ import annotations

import argparse
import json
import random
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf

from flow_mbpo_pwm.algorithms.pwm import PWM


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--metadata-output", default=None)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-envs", type=int, default=16)
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--episode-length", type=int, default=1000)
    parser.add_argument("--collector-mode", choices=["random_smooth", "checkpoint", "checkpoint_noisy", "checkpoint_blend_random"], default="random_smooth")
    parser.add_argument("--collector-id", default="")
    parser.add_argument("--quality-bin", required=True)
    parser.add_argument("--collector-alg-config", default=None)
    parser.add_argument("--collector-checkpoint", default=None)
    parser.add_argument("--teacher-deterministic", action="store_true")
    parser.add_argument("--action-noise-std", type=float, default=0.0)
    parser.add_argument("--teacher-blend", type=float, default=1.0, help="1.0 = pure checkpoint, 0.0 = smooth random")
    parser.add_argument("--random-smooth-alpha", type=float, default=0.8)
    parser.add_argument("--command-dim", type=int, default=3)
    parser.add_argument("--command-position", choices=["tail", "head", "none"], default="tail")
    parser.add_argument("--strict-task-resolution", action="store_true")
    parser.add_argument("--disable-domain-randomization", action="store_true")
    return parser.parse_args()


def git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def split_obs(obs: torch.Tensor, command_dim: int, command_position: str) -> Tuple[torch.Tensor, torch.Tensor]:
    if command_dim <= 0 or command_position == "none":
        command = torch.zeros(obs.shape[0], 0, device=obs.device, dtype=obs.dtype)
        return obs, command
    if obs.shape[-1] < command_dim:
        raise ValueError(f"obs_dim={obs.shape[-1]} < command_dim={command_dim}")
    if command_position == "tail":
        return obs[..., :-command_dim], obs[..., -command_dim:]
    if command_position == "head":
        return obs[..., command_dim:], obs[..., :command_dim]
    raise ValueError(command_position)


def initial_record(obs: torch.Tensor, action_dim: int, command_dim: int, command_position: str) -> Dict[str, torch.Tensor]:
    phys, cmd = split_obs(obs, command_dim, command_position)
    return {
        "env_obs": obs.detach().cpu(),
        "phys_obs": phys.detach().cpu(),
        "model_obs": torch.cat([phys, cmd], dim=-1).detach().cpu(),
        "command": cmd.detach().cpu(),
        "policy_action": torch.full((obs.shape[0], action_dim), torch.nan).cpu(),
        "env_action": torch.full((obs.shape[0], action_dim), torch.nan).cpu(),
        "reward": torch.full((obs.shape[0],), torch.nan).cpu(),
        "termination": torch.full((obs.shape[0],), False).cpu(),
        "truncation": torch.full((obs.shape[0],), False).cpu(),
        "done": torch.full((obs.shape[0],), False).cpu(),
    }


def step_record(obs_before_reset: torch.Tensor, action: torch.Tensor, reward: torch.Tensor, termination: torch.Tensor, truncation: torch.Tensor, command_dim: int, command_position: str) -> Dict[str, torch.Tensor]:
    phys, cmd = split_obs(obs_before_reset, command_dim, command_position)
    done = termination | truncation
    action_cpu = action.detach().cpu()
    return {
        "env_obs": obs_before_reset.detach().cpu(),
        "phys_obs": phys.detach().cpu(),
        "model_obs": torch.cat([phys, cmd], dim=-1).detach().cpu(),
        "command": cmd.detach().cpu(),
        "policy_action": action_cpu,
        "env_action": action_cpu.clone(),
        "reward": reward.detach().cpu().reshape(-1),
        "termination": termination.detach().cpu().bool().reshape(-1),
        "truncation": truncation.detach().cpu().bool().reshape(-1),
        "done": done.detach().cpu().bool().reshape(-1),
    }


def stack_episode(records: List[Dict[str, torch.Tensor]], env_idx: int, meta: Dict[str, object]) -> Dict[str, object]:
    out: Dict[str, object] = dict(meta)
    for key in records[0].keys():
        out[key] = torch.stack([r[key][env_idx] for r in records], dim=0)
    # Actions/rewards are NaN at row 0 by construction; actual transition arrays
    # are rows 1..T and line up with next observations.
    rewards = out["reward"][1:]
    dones = out["done"][1:]
    out["episode_return"] = float(torch.nan_to_num(rewards).sum().item())
    out["episode_length"] = int(rewards.shape[0])
    out["fall_rate_episode"] = float(out["termination"][1:].any().item())
    out["clip_fraction"] = float((out["policy_action"][1:].abs() >= 0.999).float().mean().item())
    return out


def build_agent(alg_config: str, checkpoint: str, obs_dim: int, act_dim: int, device: str, horizon: int, logdir: str) -> PWM:
    cfg = OmegaConf.load(alg_config)
    cfg.device = device
    cfg.horizon = horizon
    cfg.max_epochs = 0
    agent = instantiate(cfg, env=None, obs_dim=obs_dim, act_dim=act_dim, logdir=logdir, log=False)
    agent.load(checkpoint, resume_training=False)
    return agent


@torch.no_grad()
def checkpoint_action(agent: PWM, obs: torch.Tensor, deterministic: bool) -> torch.Tensor:
    x = obs
    if agent.obs_rms is not None:
        x = agent.obs_rms.normalize(x)
    z = agent.wm.encode(x, task=None)
    return torch.tanh(agent.actor(z, deterministic=deterministic))


def main() -> None:
    args = parse_args()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    metadata_output = Path(args.metadata_output) if args.metadata_output else output.with_suffix(".json")
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    device = torch.device(args.device)

    env_cfg = OmegaConf.load(args.env_config)
    env_cfg.config.device = args.device
    env_cfg.config.seed = args.seed
    env_cfg.config.num_envs = args.num_envs
    env_cfg.config.episode_length = args.episode_length
    if args.disable_domain_randomization:
        env_cfg.config.mjlab_env_kwargs = env_cfg.config.get("mjlab_env_kwargs", {})
        env_cfg.config.mjlab_env_kwargs["domain_randomization"] = False

    env = instantiate(env_cfg.config, logdir=str(output.parent))
    requested_task = str(env_cfg.config.task_id)
    resolved_task = str(getattr(env, "resolved_task_id", requested_task))
    if args.strict_task_resolution and requested_task != resolved_task:
        raise RuntimeError(f"Task fallback not allowed: requested={requested_task}, resolved={resolved_task}")

    obs = env.reset(grads=False)
    action_dim = int(env.num_actions)
    obs_dim = int(env.num_obs)
    collector: Optional[PWM] = None
    if args.collector_mode in {"checkpoint", "checkpoint_noisy", "checkpoint_blend_random"}:
        if not args.collector_alg_config or not args.collector_checkpoint:
            raise ValueError("checkpoint collector modes require --collector-alg-config and --collector-checkpoint")
        collector = build_agent(args.collector_alg_config, args.collector_checkpoint, obs_dim, action_dim, args.device, horizon=16, logdir=str(output.parent))

    prev_random = torch.zeros(args.num_envs, action_dim, device=device)
    in_progress: List[List[Dict[str, torch.Tensor]]] = []
    init = initial_record(obs, action_dim, args.command_dim, args.command_position)
    for _ in range(args.num_envs):
        in_progress.append([])
    for i in range(args.num_envs):
        in_progress[i].append({k: v[i : i + 1] for k, v in init.items()})

    completed: List[Dict[str, object]] = []
    while len(completed) < args.episodes:
        eps = torch.empty(args.num_envs, action_dim, device=device).uniform_(-1.0, 1.0)
        smooth_random = args.random_smooth_alpha * prev_random + (1.0 - args.random_smooth_alpha) * eps
        smooth_random = smooth_random.clamp(-1.0, 1.0)
        prev_random = smooth_random
        if args.collector_mode == "random_smooth":
            action = smooth_random
        else:
            assert collector is not None
            teacher = checkpoint_action(collector, obs, deterministic=args.teacher_deterministic)
            if args.collector_mode == "checkpoint_blend_random":
                action = args.teacher_blend * teacher + (1.0 - args.teacher_blend) * smooth_random
            else:
                action = teacher
            if args.action_noise_std > 0:
                action = action + torch.randn_like(action) * args.action_noise_std
            action = action.clamp(-1.0, 1.0)

        next_obs, reward, done, info = env.step(action)
        rec = step_record(info["obs_before_reset"], action, reward, info["termination"], info["truncation"], args.command_dim, args.command_position)
        for i in range(args.num_envs):
            in_progress[i].append({k: v[i : i + 1] for k, v in rec.items()})
        done_ids = done.nonzero(as_tuple=False).squeeze(-1).tolist()
        for i in done_ids:
            meta = {
                "task_id_requested": requested_task,
                "task_id_resolved": resolved_task,
                "quality_bin": args.quality_bin,
                "collector_id": args.collector_id or args.collector_mode,
                "collector_mode": args.collector_mode,
                "collector_seed": args.seed,
                "checkpoint_path": args.collector_checkpoint or "",
                "checkpoint_alg_config": args.collector_alg_config or "",
                "command_dim": args.command_dim,
                "command_position": args.command_position,
                "domain_randomization": False if args.disable_domain_randomization else None,
                "adapter_version": "mjlab_qs_v1",
            }
            completed.append(stack_episode(in_progress[i], 0, meta))
            init_i = initial_record(next_obs[i : i + 1], action_dim, args.command_dim, args.command_position)
            in_progress[i] = [init_i]
            if len(completed) >= args.episodes:
                break
        obs = next_obs

    payload = {
        "episodes": completed,
        "metadata": {
            "script": "collect_mjlab_qs_episodes.py",
            "env_config": args.env_config,
            "seed": args.seed,
            "num_envs": args.num_envs,
            "target_episodes": args.episodes,
            "collected_episodes": len(completed),
            "episode_length": args.episode_length,
            "obs_dim": obs_dim,
            "act_dim": action_dim,
            "task_id_requested": requested_task,
            "task_id_resolved": resolved_task,
            "quality_bin": args.quality_bin,
            "collector_mode": args.collector_mode,
            "collector_id": args.collector_id or args.collector_mode,
            "collector_checkpoint": args.collector_checkpoint or "",
            "collector_alg_config": args.collector_alg_config or "",
            "command_dim": args.command_dim,
            "command_position": args.command_position,
            "domain_randomization": False if args.disable_domain_randomization else None,
            "repo_git_sha": git_sha(),
        },
    }
    torch.save(payload, output)
    metadata_output.write_text(json.dumps(payload["metadata"], indent=2), encoding="utf-8")
    print(f"saved raw episodes: {output}")
    print(f"saved metadata: {metadata_output}")


if __name__ == "__main__":
    main()
