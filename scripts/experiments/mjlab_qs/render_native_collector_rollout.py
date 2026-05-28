#!/usr/bin/env python3
"""Render W&B-backed rollout videos for MJLab-native collector policies."""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import subprocess
import sys
import time
import types
from dataclasses import asdict
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

from scripts.experiments.mjlab_qs.collect_mjlab_qs_native_episodes import (
    apply_method,
    patch_headless_display_dependency,
    patch_mujoco_compatibility,
    tensor_from_actor_obs,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--task-id", required=True)
    p.add_argument("--method", required=True, choices=["random_smooth", "rslrl_ppo_default", "rslrl_ppo_conservative"])
    p.add_argument("--checkpoint", default="")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--episodes", type=int, default=3)
    p.add_argument("--episode-length", type=int, default=1000)
    p.add_argument("--collector-id", default="")
    p.add_argument("--collector-mode", choices=["random_smooth", "checkpoint", "checkpoint_noisy", "checkpoint_blend_random"], default="checkpoint")
    p.add_argument("--teacher-blend", type=float, default=1.0)
    p.add_argument("--action-noise-std", type=float, default=0.0)
    p.add_argument("--random-smooth-alpha", type=float, default=0.8)
    p.add_argument("--video-fps", type=int, default=30)
    p.add_argument("--require-mp4", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--wandb-project", default="")
    p.add_argument("--wandb-group", default="")
    p.add_argument("--wandb-name", default="")
    p.add_argument("--disable-wandb", action="store_true")
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
    env_cfg.scene.num_envs = 1
    env_cfg.seed = int(args.seed)
    if hasattr(env_cfg, "episode_length_s") and hasattr(env_cfg, "sim") and hasattr(env_cfg.sim, "mujoco"):
        env_dt = float(env_cfg.sim.mujoco.timestep) * float(env_cfg.decimation)
        env_cfg.episode_length_s = float(args.episode_length) * env_dt

    base_env = ManagerBasedRlEnv(cfg=env_cfg, device=args.device, render_mode="rgb_array")
    wrapped = RslRlVecEnvWrapper(base_env, clip_actions=agent_cfg.clip_actions)
    obs_td = wrapped.get_observations()
    obs_groups = list(agent_cfg.obs_groups["actor"])
    policy = None
    if args.method != "random_smooth" and args.collector_mode != "random_smooth":
        if not args.checkpoint:
            raise ValueError("--checkpoint is required for native checkpoint rollout")
        runner_cls = load_runner_cls(args.task_id) or MjlabOnPolicyRunner
        runner = runner_cls(wrapped, asdict(agent_cfg), device=args.device)
        runner.load(args.checkpoint, load_cfg={"actor": True}, strict=True, map_location=args.device)
        policy = runner.get_inference_policy(device=args.device)
    return wrapped, base_env, obs_td, obs_groups, int(wrapped.num_actions), policy


def frame_from_render(env, base_env):
    for candidate_env in (env, base_env):
        render_fn = getattr(candidate_env, "render", None)
        if not callable(render_fn):
            continue
        try:
            frame = render_fn(mode="rgb_array")
        except TypeError:
            frame = render_fn()
        if isinstance(frame, torch.Tensor):
            frame = frame.detach().cpu().numpy()
        if isinstance(frame, (list, tuple)) and frame:
            frame = frame[0]
        if isinstance(frame, dict):
            for key in ("rgb", "frame", "image"):
                if key in frame:
                    frame = frame[key]
                    break
        if hasattr(frame, "shape"):
            return frame
    return None


def write_video(frames: list[Any], path: Path, fps: int, require_mp4: bool) -> None:
    if not frames:
        raise RuntimeError("No render frames were captured; cannot write rollout video.")
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        import imageio.v2 as iio

        with iio.get_writer(str(path), fps=fps, format="FFMPEG") as writer:
            for frame in frames:
                writer.append_data(frame)
    except Exception as exc:
        if require_mp4:
            raise RuntimeError(f"MP4 export failed for {path}: {exc}") from exc
        import imageio.v2 as iio_v2

        iio_v2.mimsave(str(path.with_suffix(".gif")), frames, format="GIF", fps=fps)


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


@torch.no_grad()
def collect_rollout(args: argparse.Namespace):
    device = torch.device(args.device)
    env, base_env, obs_td, obs_groups, action_dim, policy = build_env_and_policy(args)
    prev_random = torch.zeros(1, action_dim, device=device)
    frames: list[Any] = []
    step_rows: list[dict[str, Any]] = []
    episode_rows: list[dict[str, Any]] = []
    ep_return = torch.zeros(1, device=device)
    ep_len = torch.zeros(1, dtype=torch.long, device=device)
    episodes_done = 0
    try:
        while episodes_done < args.episodes:
            frame = frame_from_render(env, base_env)
            if frame is not None:
                frames.append(frame)
            eps = torch.empty(1, action_dim, device=device).uniform_(-1.0, 1.0)
            smooth_random = (args.random_smooth_alpha * prev_random + (1.0 - args.random_smooth_alpha) * eps).clamp(-1.0, 1.0)
            prev_random = smooth_random
            if args.method == "random_smooth" or args.collector_mode == "random_smooth":
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
            reward = reward.to(device).float().reshape(-1)
            done = done.to(device).bool().reshape(-1)
            time_out = extras.get("time_outs", torch.zeros_like(done)).to(device).bool().reshape(-1)
            terminated = done & (~time_out)
            ep_return += reward[:1]
            ep_len += 1
            step_rows.append(
                {
                    "episode_slot": episodes_done,
                    "step": int(ep_len[0].item()),
                    "reward": float(reward[0].item()),
                    "action_l2": float(action[0].pow(2).mean().sqrt().item()),
                    "done": int(done[0].item()),
                    "terminated": int(terminated[0].item()),
                    "truncated": int(time_out[0].item()),
                }
            )
            if bool(done[0].item()) or int(ep_len[0].item()) >= args.episode_length:
                episode_rows.append(
                    {
                        "episode": episodes_done,
                        "return": float(ep_return[0].item()),
                        "length": int(ep_len[0].item()),
                        "terminated": int(terminated[0].item()),
                        "truncated": int(time_out[0].item()),
                    }
                )
                episodes_done += 1
                ep_return.zero_()
                ep_len.zero_()
            obs_td = next_obs_td
    finally:
        env.close()
    return frames, step_rows, episode_rows


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    run = None
    if args.wandb_project and not args.disable_wandb:
        import wandb

        run = wandb.init(
            project=args.wandb_project,
            group=args.wandb_group,
            name=args.wandb_name or f"{args.collector_id or args.method}_seed{args.seed}_rollout",
            job_type="native_collector_rollout_video",
            config={**vars(args), "git_sha": git_sha(), "git_branch": git_branch(), "command": command_line()},
        )
    t0 = time.time()
    frames, step_rows, episode_rows = collect_rollout(args)
    video_path = output_dir / "rollout.mp4"
    write_video(frames, video_path, args.video_fps, args.require_mp4)
    write_csv(output_dir / "rollout_steps.csv", step_rows, ["episode_slot", "step", "reward", "action_l2", "done", "terminated", "truncated"])
    write_csv(output_dir / "rollout_summary.csv", episode_rows, ["episode", "return", "length", "terminated", "truncated"])
    returns = torch.tensor([row["return"] for row in episode_rows], dtype=torch.float32)
    lengths = torch.tensor([row["length"] for row in episode_rows], dtype=torch.float32)
    terminated = torch.tensor([row["terminated"] for row in episode_rows], dtype=torch.float32)
    summary = {
        "task_id": args.task_id,
        "method": args.method,
        "collector_id": args.collector_id or args.method,
        "collector_mode": args.collector_mode,
        "checkpoint": args.checkpoint,
        "video": str(video_path),
        "num_frames": len(frames),
        "num_episodes": len(episode_rows),
        "return_mean": float(returns.mean().item()) if returns.numel() else None,
        "return_std": float(returns.std(unbiased=False).item()) if returns.numel() else None,
        "episode_length_mean": float(lengths.mean().item()) if lengths.numel() else None,
        "fall_rate_mean": float(terminated.mean().item()) if terminated.numel() else None,
        "wall_clock_seconds": time.time() - t0,
        "git_sha": git_sha(),
        "git_branch": git_branch(),
        "command": command_line(),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    if run is not None:
        import wandb

        run.log(
            {
                "rollout/return_mean": summary["return_mean"],
                "rollout/episode_length_mean": summary["episode_length_mean"],
                "rollout/fall_rate_mean": summary["fall_rate_mean"],
                "rollout/video": wandb.Video(str(video_path), fps=args.video_fps, format="mp4"),
            }
        )
        run.summary.update(summary)
        run.finish()
    print(json.dumps(summary, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
