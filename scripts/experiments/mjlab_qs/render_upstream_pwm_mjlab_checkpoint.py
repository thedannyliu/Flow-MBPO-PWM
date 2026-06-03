#!/usr/bin/env python3
"""Render rollout videos for a full upstream PWM checkpoint on MJLab."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import subprocess
import sys
import time
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
    p.add_argument("--rollout-episodes", type=int, default=10)
    p.add_argument("--max-steps", type=int, default=1000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--video-fps", type=int, default=30)
    p.add_argument("--require-mp4", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--baseline-return", type=float, default=None)
    p.add_argument("--baseline-length", type=float, default=None)
    p.add_argument("--baseline-fall", type=float, default=None)
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


def frame_from_render(env: Any) -> Any | None:
    render_fn = getattr(env, "render", None)
    if not callable(render_fn):
        return None
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
    return frame if hasattr(frame, "shape") else None


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


def add_baseline_gate(summary: dict[str, Any], args: argparse.Namespace) -> None:
    configured = args.baseline_return is not None and args.baseline_length is not None and args.baseline_fall is not None
    summary["baseline_gate_configured"] = bool(configured)
    if args.baseline_return is not None:
        summary["baseline_return"] = float(args.baseline_return)
        summary["return_gap_to_baseline"] = float(summary["return_mean"]) - float(args.baseline_return)
        summary["return_gate_pass"] = bool(float(summary["return_mean"]) >= float(args.baseline_return))
    if args.baseline_length is not None:
        summary["baseline_length"] = float(args.baseline_length)
        summary["length_gap_to_baseline"] = float(summary["episode_length_mean"]) - float(args.baseline_length)
        summary["length_gate_pass"] = bool(float(summary["episode_length_mean"]) >= float(args.baseline_length))
    if args.baseline_fall is not None:
        summary["baseline_fall"] = float(args.baseline_fall)
        summary["fall_gap_to_baseline"] = float(summary["fall_rate_mean"]) - float(args.baseline_fall)
        summary["fall_gate_pass"] = bool(float(summary["fall_rate_mean"]) < float(args.baseline_fall))
    if configured:
        summary["baseline_gate_pass"] = bool(
            summary.get("return_gate_pass") and summary.get("length_gate_pass") and summary.get("fall_gate_pass")
        )


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
    cfg.env.config.num_envs = 1
    cfg.env.config.device = str(device)
    cfg.env.config.seed = args.seed
    cfg.env.config.episode_length = args.max_steps
    cfg.env.config.no_grad = True
    if "mjlab_env_kwargs" not in cfg.env.config or cfg.env.config.mjlab_env_kwargs is None:
        cfg.env.config.mjlab_env_kwargs = {}
    cfg.env.config.mjlab_env_kwargs.render_mode = "rgb_array"

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    env = instantiate(cfg.env.config, logdir=str(output_dir / "env_logs"))
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

    frames: list[Any] = []
    step_rows: list[dict[str, Any]] = []
    episode_rows: list[dict[str, Any]] = []
    returns = torch.zeros(env.num_envs, dtype=torch.float32, device=device)
    lengths = torch.zeros(env.num_envs, dtype=torch.float32, device=device)
    obs = env.reset()
    t0 = time.time()
    try:
        with torch.no_grad():
            while len(episode_rows) < args.rollout_episodes:
                frame = frame_from_render(env)
                if frame is not None:
                    frames.append(frame)
                obs_in = obs
                if agent.obs_rms is not None:
                    obs_in = agent.obs_rms.normalize(obs_in)
                z = agent.wm.encode(obs_in, task=None)
                raw_action = torch.tanh(agent.actor(z, deterministic=True)).clamp(-1.0, 1.0)
                obs, reward, done, info = env.step(raw_action)
                reward = reward.to(device).float().reshape(-1)
                done = done.to(device).bool().reshape(-1)
                truncated = info.get("truncation", torch.zeros_like(done)).to(device).bool().reshape(-1)
                terminated = info.get("termination", done & (~truncated)).to(device).bool().reshape(-1)
                returns += reward
                lengths += 1.0
                step_rows.append(
                    {
                        "episode_slot": len(episode_rows),
                        "step": int(lengths[0].item()),
                        "reward": float(reward[0].item()),
                        "action_l2": float(raw_action[0].float().pow(2).mean().sqrt().item()),
                        "done": int(done[0].item()),
                        "terminated": int(terminated[0].item()),
                        "truncated": int(truncated[0].item()),
                    }
                )
                if bool(done[0].item()) or int(lengths[0].item()) >= args.max_steps:
                    episode_rows.append(
                        {
                            "episode": len(episode_rows),
                            "return": float(returns[0].item()),
                            "length": float(lengths[0].item()),
                            "terminated": int(terminated[0].item()),
                            "truncated": int(truncated[0].item()),
                        }
                    )
                    returns.zero_()
                    lengths.zero_()
    finally:
        env.close()

    video_path = output_dir / "rollout.mp4"
    write_video(frames, video_path, args.video_fps, args.require_mp4)
    write_csv(
        output_dir / "rollout_steps.csv",
        step_rows,
        ["episode_slot", "step", "reward", "action_l2", "done", "terminated", "truncated"],
    )
    write_csv(output_dir / "rollout_summary.csv", episode_rows, ["episode", "return", "length", "terminated", "truncated"])
    return_mean, return_std = mean_std([float(row["return"]) for row in episode_rows])
    length_mean, length_std = mean_std([float(row["length"]) for row in episode_rows])
    fall_mean, _ = mean_std([float(row["terminated"]) for row in episode_rows])
    timeout_mean, _ = mean_std([float(row["truncated"]) for row in episode_rows])
    summary = {
        "policy_checkpoint": str(Path(args.policy_checkpoint)),
        "checkpoint_kind": args.checkpoint_kind,
        "hydra_run_dir": str(hydra_run_dir),
        "task_id": cfg.env.config.task_id,
        "video": str(video_path),
        "num_frames": len(frames),
        "num_episodes": len(episode_rows),
        "rollout_episodes": args.rollout_episodes,
        "max_steps": args.max_steps,
        "seed": args.seed,
        "return_mean": return_mean,
        "return_std": return_std,
        "episode_length_mean": length_mean,
        "episode_length_std": length_std,
        "fall_rate_mean": fall_mean,
        "timeout_rate_mean": timeout_mean,
        "wall_clock_seconds": time.time() - t0,
        "git_sha": git_sha(),
        "git_branch": git_branch(),
        "command": command_line(),
        "notes": args.notes,
    }
    add_baseline_gate(summary, args)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
