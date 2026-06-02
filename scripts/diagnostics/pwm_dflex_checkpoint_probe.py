#!/usr/bin/env python3
"""Probe one-step parity between a PWM checkpoint world model and DFlex."""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
from contextlib import contextmanager

import hydra
import torch
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from omegaconf import OmegaConf, open_dict


@contextmanager
def pushd(path: pathlib.Path):
    old = pathlib.Path.cwd()
    try:
        import os

        os.chdir(path)
        yield
    finally:
        os.chdir(old)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-root", type=pathlib.Path, default=pathlib.Path("baselines/PWM"))
    parser.add_argument("--checkpoint", type=pathlib.Path, required=True)
    parser.add_argument("--env", default="dflex_hopper")
    parser.add_argument("--checkpoint-mode", choices=("full", "wm_only"), default="full")
    parser.add_argument("--policy", choices=("actor", "random", "zero"), default="actor")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-envs", type=int, default=64)
    parser.add_argument("--steps", type=int, default=128)
    parser.add_argument("--output", type=pathlib.Path)
    return parser.parse_args()


def corrcoef(x: torch.Tensor, y: torch.Tensor) -> float:
    x = x.float().flatten()
    y = y.float().flatten()
    finite = torch.isfinite(x) & torch.isfinite(y)
    x = x[finite]
    y = y[finite]
    if x.numel() < 2:
        return float("nan")
    x = x - x.mean()
    y = y - y.mean()
    denom = torch.linalg.norm(x) * torch.linalg.norm(y)
    if denom.item() == 0.0:
        return float("nan")
    return (x.dot(y) / denom).item()


def summarize(name: str, value: torch.Tensor) -> dict[str, float]:
    value = value.detach().float().flatten().cpu()
    finite = value[torch.isfinite(value)]
    if finite.numel() == 0:
        return {f"{name}_mean": float("nan"), f"{name}_std": float("nan")}
    return {
        f"{name}_mean": finite.mean().item(),
        f"{name}_std": finite.std(unbiased=False).item(),
        f"{name}_min": finite.min().item(),
        f"{name}_max": finite.max().item(),
    }


def load_full_checkpoint(agent, checkpoint: str) -> None:
    try:
        agent.load(checkpoint, with_buffer=False)
    except TypeError as exc:
        if "with_buffer" not in str(exc):
            raise
        agent.load(checkpoint, buffer=False)


def main() -> None:
    args = parse_args()
    repo_root = pathlib.Path.cwd()
    baseline_root = (repo_root / args.baseline_root).resolve()
    sys.path.insert(0, str(baseline_root / "src"))
    sys.path.insert(0, str(baseline_root / "scripts"))

    from pwm.utils.common import seeding

    seeding(args.seed, False)

    config_dir = str((baseline_root / "scripts" / "cfg").resolve())
    hydra.core.global_hydra.GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=config_dir, version_base="1.2"):
        cfg = compose(
            config_name="config.yaml",
            overrides=[
                f"env={args.env}",
                "alg=pwm",
                f"general.device={args.device}",
                f"general.seed={args.seed}",
                f"env.config.num_envs={args.num_envs}",
                "general.run_wandb=False",
            ],
        )

    with open_dict(cfg):
        cfg.env.config.no_grad = True

    cfg_full = OmegaConf.to_container(cfg, resolve=True)
    logdir = str((repo_root / "logs" / "diagnostics" / "pwm_dflex_checkpoint_probe").resolve())

    with pushd(baseline_root / "scripts"):
        env = instantiate(cfg.env.config, logdir=logdir)
        agent = instantiate(
            cfg.alg,
            env=env,
            obs_dim=env.num_obs,
            act_dim=env.num_actions,
            logdir=logdir,
            log=False,
        )

        checkpoint = str(args.checkpoint.resolve())
        if args.checkpoint_mode == "full":
            load_full_checkpoint(agent, checkpoint)
        else:
            agent.load_wm_from_policy_checkpoint(checkpoint)
        agent.wm_bootstrapped = True
        if hasattr(agent, "log_effective_checkpoint_state"):
            agent.log_effective_checkpoint_state(f"probe checkpoint_mode={args.checkpoint_mode}")

        obs = env.reset()
        pred_rews = []
        real_rews = []
        real_rews_norm = []
        actions_his = []
        term_his = []
        trunc_his = []

        for _ in range(args.steps):
            model_obs = agent.obs_rms.normalize(obs) if agent.obs_rms else obs
            z = agent.wm.encode(model_obs, task=None)
            if args.policy == "actor":
                actions = torch.tanh(agent.actor(z))
            elif args.policy == "random":
                actions = torch.empty(args.num_envs, env.num_actions, device=args.device).uniform_(-1.0, 1.0)
            else:
                actions = torch.zeros(args.num_envs, env.num_actions, device=args.device)

            _, pred_rew = agent.wm.step(z, actions, task=None)
            pred_rew = agent.wm.almost_two_hot_inv(pred_rew).squeeze()
            obs, real_rew, done, info = env.step(actions)

            if agent.rew_rms:
                real_norm = agent.rew_rms.normalize(real_rew.reshape(-1, 1)).squeeze()
            else:
                real_norm = real_rew

            pred_rews.append(pred_rew.detach())
            real_rews.append(real_rew.detach())
            real_rews_norm.append(real_norm.detach())
            actions_his.append(actions.detach())
            term_his.append(info["termination"].detach().float())
            trunc_his.append(info["truncation"].detach().float())

        pred = torch.cat([x.flatten() for x in pred_rews])
        real = torch.cat([x.flatten() for x in real_rews])
        real_norm = torch.cat([x.flatten() for x in real_rews_norm])
        actions = torch.cat(actions_his, dim=0)
        term = torch.cat(term_his)
        trunc = torch.cat(trunc_his)

    mse_norm = torch.mean((pred - real_norm) ** 2).item()
    mae_norm = torch.mean(torch.abs(pred - real_norm)).item()
    result = {
        "baseline_root": str(baseline_root),
        "checkpoint": str(args.checkpoint.resolve()),
        "env": args.env,
        "checkpoint_mode": args.checkpoint_mode,
        "policy": args.policy,
        "seed": args.seed,
        "num_envs": args.num_envs,
        "steps": args.steps,
        "config": cfg_full,
        "wm_vs_real_reward_corr_raw": corrcoef(pred, real),
        "wm_vs_real_reward_corr_normalized": corrcoef(pred, real_norm),
        "wm_vs_real_reward_mse_normalized": mse_norm,
        "wm_vs_real_reward_mae_normalized": mae_norm,
        "action_abs_gt_0_95_frac": (actions.abs() > 0.95).float().mean().item(),
        "termination_frac": term.mean().item(),
        "truncation_frac": trunc.mean().item(),
    }
    result.update(summarize("wm_reward", pred))
    result.update(summarize("real_reward", real))
    result.update(summarize("real_reward_normalized", real_norm))
    result.update(summarize("action", actions))

    text = json.dumps(result, indent=2, sort_keys=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
