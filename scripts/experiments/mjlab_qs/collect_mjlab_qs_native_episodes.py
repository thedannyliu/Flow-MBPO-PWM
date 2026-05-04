#!/usr/bin/env python3
"""Collect MJLab-QS raw episodes with MJLab-native RSL-RL policies."""

from __future__ import annotations

import argparse
import json
import random
import subprocess
import sys
import types
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Tuple

import torch


def patch_mujoco_compatibility() -> None:
    try:
        import mujoco  # type: ignore

        enable_bits = getattr(mujoco, "mjtEnableBit", None)
        if enable_bits is not None and not hasattr(enable_bits, "mjENBL_MULTICCD"):
            setattr(enable_bits, "mjENBL_MULTICCD", 0)
    except Exception:
        return


def patch_headless_display_dependency() -> None:
    try:
        import IPython.display  # type: ignore  # noqa: F401

        return
    except Exception:
        pass
    ipython_mod = types.ModuleType("IPython")
    display_mod = types.ModuleType("IPython.display")

    class HTML:
        def __init__(self, data=None, *args, **kwargs):
            self.data = data

    def display(*args, **kwargs):
        return None

    display_mod.HTML = HTML
    display_mod.display = display
    ipython_mod.display = display_mod
    sys.modules.setdefault("IPython", ipython_mod)
    sys.modules.setdefault("IPython.display", display_mod)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--task-id", required=True)
    p.add_argument("--method", required=True, choices=["random_smooth", "rslrl_ppo_default", "rslrl_ppo_conservative"])
    p.add_argument("--checkpoint", default="")
    p.add_argument("--output", required=True)
    p.add_argument("--metadata-output", default="")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--num-envs", type=int, default=16)
    p.add_argument("--episodes", type=int, default=100)
    p.add_argument("--episode-length", type=int, default=1000)
    p.add_argument("--quality-bin", required=True)
    p.add_argument("--collector-id", default="")
    p.add_argument("--collector-mode", choices=["random_smooth", "checkpoint", "checkpoint_noisy", "checkpoint_blend_random"], default="checkpoint")
    p.add_argument("--teacher-blend", type=float, default=1.0)
    p.add_argument("--action-noise-std", type=float, default=0.0)
    p.add_argument("--random-smooth-alpha", type=float, default=0.8)
    p.add_argument("--command-dim", type=int, default=3)
    p.add_argument("--command-position", choices=["tail", "head", "none"], default="tail")
    p.add_argument("--drop-terminal-transition", action="store_true", default=True)
    return p.parse_args()


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


def tensor_from_actor_obs(obs_td, obs_groups: List[str]) -> torch.Tensor:
    parts = [obs_td[group] for group in obs_groups]
    return torch.cat(parts, dim=-1).float()


def make_record(
    obs: torch.Tensor,
    action: torch.Tensor,
    reward: torch.Tensor,
    termination: torch.Tensor,
    truncation: torch.Tensor,
    transition_valid: bool,
    command_dim: int,
    command_position: str,
) -> Dict[str, torch.Tensor]:
    phys, cmd = split_obs(obs, command_dim, command_position)
    done = termination | truncation
    return {
        "env_obs": obs.detach().cpu(),
        "phys_obs": phys.detach().cpu(),
        "model_obs": torch.cat([phys, cmd], dim=-1).detach().cpu(),
        "command": cmd.detach().cpu(),
        "policy_action": action.detach().cpu(),
        "env_action": action.detach().cpu().clone(),
        "reward": reward.detach().cpu().reshape(-1),
        "transition_valid": torch.full((obs.shape[0],), bool(transition_valid)).cpu(),
        "termination": termination.detach().cpu().bool().reshape(-1),
        "truncation": truncation.detach().cpu().bool().reshape(-1),
        "done": done.detach().cpu().bool().reshape(-1),
    }


def initial_record(obs: torch.Tensor, action_dim: int, command_dim: int, command_position: str) -> Dict[str, torch.Tensor]:
    zeros_a = torch.zeros((obs.shape[0], action_dim), dtype=obs.dtype, device=obs.device)
    zeros_r = torch.zeros((obs.shape[0],), dtype=obs.dtype, device=obs.device)
    zeros_b = torch.zeros((obs.shape[0],), dtype=torch.bool, device=obs.device)
    return make_record(obs, zeros_a, zeros_r, zeros_b, zeros_b, False, command_dim, command_position)


def stack_episode(records: List[Dict[str, torch.Tensor]], env_idx: int, meta: Dict[str, object]) -> Dict[str, object]:
    out: Dict[str, object] = dict(meta)
    for key in records[0].keys():
        out[key] = torch.stack([r[key][env_idx] for r in records], dim=0)
    rewards = out["reward"][1:]
    out["episode_return"] = float(torch.nan_to_num(rewards.float(), nan=0.0).sum().item())
    out["episode_length"] = int(rewards.shape[0])
    out["fall_rate_episode"] = float(bool(out.get("episode_terminated", False)))
    out["clip_fraction"] = float((out["policy_action"][1:].abs() >= 0.999).float().mean().item()) if rewards.numel() else 0.0
    return out


def apply_method(cfg, method: str) -> None:
    if method == "random_smooth":
        return
    if method == "rslrl_ppo_default":
        return
    if method == "rslrl_ppo_conservative":
        cfg.agent.algorithm.learning_rate = 3.0e-4
        cfg.agent.algorithm.desired_kl = 0.008
        cfg.agent.algorithm.entropy_coef = 0.005
        cfg.agent.actor.obs_normalization = True
        cfg.agent.critic.obs_normalization = True
        for event_name in ("push_robot", "foot_friction", "encoder_bias", "base_com"):
            cfg.env.events.pop(event_name, None)
        return
    raise ValueError(method)


def build_env_and_policy(args: argparse.Namespace):
    patch_mujoco_compatibility()
    patch_headless_display_dependency()
    import mjlab.tasks  # noqa: F401
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
    from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls

    env_cfg = load_env_cfg(args.task_id, play=True)
    agent_cfg = load_rl_cfg(args.task_id)
    cfg = types.SimpleNamespace(env=env_cfg, agent=agent_cfg)
    apply_method(cfg, args.method)
    env_cfg.scene.num_envs = int(args.num_envs)
    env_cfg.seed = int(args.seed)
    if hasattr(env_cfg, "episode_length_s") and hasattr(env_cfg, "sim") and hasattr(env_cfg.sim, "mujoco"):
        env_dt = float(env_cfg.sim.mujoco.timestep) * float(env_cfg.decimation)
        env_cfg.episode_length_s = float(args.episode_length) * env_dt

    env = ManagerBasedRlEnv(cfg=env_cfg, device=args.device, render_mode=None)
    wrapped = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    obs_td = wrapped.get_observations()
    obs_groups = list(agent_cfg.obs_groups["actor"])
    action_dim = int(wrapped.num_actions)
    policy = None
    if args.method != "random_smooth" and args.collector_mode != "random_smooth":
        if not args.checkpoint:
            raise ValueError("--checkpoint is required for native checkpoint collection")
        runner_cls = load_runner_cls(args.task_id) or MjlabOnPolicyRunner
        runner = runner_cls(wrapped, asdict(agent_cfg), device=args.device)
        runner.load(args.checkpoint, load_cfg={"actor": True}, strict=True, map_location=args.device)
        policy = runner.get_inference_policy(device=args.device)
    return wrapped, obs_td, obs_groups, action_dim, policy


@torch.no_grad()
def main() -> None:
    args = parse_args()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    metadata_output = Path(args.metadata_output) if args.metadata_output else output.with_suffix(".json")
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    env, obs_td, obs_groups, action_dim, policy = build_env_and_policy(args)
    device = torch.device(args.device)
    obs_tensor = tensor_from_actor_obs(obs_td, obs_groups)
    prev_random = torch.zeros(args.num_envs, action_dim, device=device)

    in_progress: List[List[Dict[str, torch.Tensor]]] = [[] for _ in range(args.num_envs)]
    init = initial_record(obs_tensor, action_dim, args.command_dim, args.command_position)
    for i in range(args.num_envs):
        in_progress[i].append({k: v[i : i + 1] for k, v in init.items()})

    completed: List[Dict[str, object]] = []
    while len(completed) < args.episodes:
        eps = torch.empty(args.num_envs, action_dim, device=device).uniform_(-1.0, 1.0)
        smooth_random = (args.random_smooth_alpha * prev_random + (1.0 - args.random_smooth_alpha) * eps).clamp(-1.0, 1.0)
        prev_random = smooth_random
        if args.collector_mode == "random_smooth" or args.method == "random_smooth":
            action = smooth_random
        else:
            assert policy is not None
            teacher = policy(obs_td).clamp(-1.0, 1.0)
            if args.collector_mode == "checkpoint_blend_random":
                action = (args.teacher_blend * teacher + (1.0 - args.teacher_blend) * smooth_random).clamp(-1.0, 1.0)
            else:
                action = teacher
            if args.action_noise_std > 0:
                action = (action + torch.randn_like(action) * args.action_noise_std).clamp(-1.0, 1.0)

        next_obs_td, reward, done, extras = env.step(action)
        next_obs_tensor = tensor_from_actor_obs(next_obs_td, obs_groups)
        termination = extras.get("time_outs", torch.zeros_like(done)).to(device).bool()
        truncation = extras.get("time_outs", torch.zeros_like(done)).to(device).bool()
        done_bool = done.to(device).bool()
        termination = done_bool & (~truncation)

        done_ids = done_bool.nonzero(as_tuple=False).squeeze(-1).tolist()
        not_done = (~done_bool).nonzero(as_tuple=False).squeeze(-1).tolist()

        if not_done:
            rec = make_record(
                next_obs_tensor[not_done],
                action[not_done],
                reward[not_done],
                torch.zeros(len(not_done), dtype=torch.bool, device=device),
                torch.zeros(len(not_done), dtype=torch.bool, device=device),
                True,
                args.command_dim,
                args.command_position,
            )
            for local_idx, env_idx in enumerate(not_done):
                in_progress[env_idx].append({k: v[local_idx : local_idx + 1] for k, v in rec.items()})

        for env_idx in done_ids:
            meta = {
                "task_id_requested": args.task_id,
                "task_id_resolved": args.task_id,
                "quality_bin": args.quality_bin,
                "collector_id": args.collector_id or args.method,
                "collector_mode": args.collector_mode,
                "collector_seed": args.seed,
                "checkpoint_path": args.checkpoint,
                "checkpoint_alg_config": f"mjlab_native::{args.method}",
                "command_dim": args.command_dim,
                "command_position": args.command_position,
                "domain_randomization": False,
                "adapter_version": "mjlab_qs_native_v1",
                "episode_terminated": bool(termination[env_idx].item()),
                "episode_truncated": bool(truncation[env_idx].item()),
                "dropped_terminal_transition": bool(args.drop_terminal_transition),
            }
            completed.append(stack_episode(in_progress[env_idx], 0, meta))
            init_i = initial_record(next_obs_tensor[env_idx : env_idx + 1], action_dim, args.command_dim, args.command_position)
            in_progress[env_idx] = [init_i]
            if len(completed) >= args.episodes:
                break
        obs_td = next_obs_td

    payload = {
        "episodes": completed,
        "metadata": {
            "script": "collect_mjlab_qs_native_episodes.py",
            "task_id_requested": args.task_id,
            "task_id_resolved": args.task_id,
            "method": args.method,
            "checkpoint": args.checkpoint,
            "seed": args.seed,
            "num_envs": args.num_envs,
            "target_episodes": args.episodes,
            "collected_episodes": len(completed),
            "episode_length": args.episode_length,
            "quality_bin": args.quality_bin,
            "collector_mode": args.collector_mode,
            "collector_id": args.collector_id or args.method,
            "command_dim": args.command_dim,
            "command_position": args.command_position,
            "repo_git_sha": git_sha(),
            "note": "Native collector drops terminal transition to avoid reset-state contamination.",
        },
    }
    torch.save(payload, output)
    metadata_output.write_text(json.dumps(payload["metadata"], indent=2), encoding="utf-8")
    env.close()
    print(f"saved native raw episodes: {output}")
    print(f"saved metadata: {metadata_output}")


if __name__ == "__main__":
    main()
