#!/usr/bin/env python3
"""Probe why replay-driven original PWM collapses on MJLab-QS.

This diagnostic is intentionally read-only: it loads a completed original-PWM
adapter checkpoint, measures WM fit on held-out QS windows, measures how far
the learned policy moves from dataset actions, and records the actor/critic
signals that PWM used for imagined policy extraction.
"""

from __future__ import annotations

import argparse
import json
import random
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parents[3]
PWM_SRC = PROJECT_ROOT / "baselines" / "PWM" / "src"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(PWM_SRC) not in sys.path:
    sys.path.insert(0, str(PWM_SRC))

from scripts.experiments.mjlab_qs.run_original_pwm_adapter import (  # noqa: E402
    batch_windows,
    build_pwm_agent,
    load_data,
    pack_obs,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True)
    p.add_argument("--metadata", required=True)
    p.add_argument("--normalization", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--checkpoint-kind", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--max-batches", type=int, default=64)
    p.add_argument("--horizon", type=int, default=16)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--lam", type=float, default=0.95)
    p.add_argument("--obs-mode", choices=["normalized", "raw"], default="normalized")
    p.add_argument("--reward-mode", choices=["normalized", "raw"], default="normalized")
    p.add_argument("--ret-rms", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--rew-rms", action=argparse.BooleanOptionalAction, default=False)
    return p.parse_args()


def git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def load_summary(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {}


def corrcoef(x: torch.Tensor, y: torch.Tensor) -> float:
    x = x.detach().float().reshape(-1).cpu()
    y = y.detach().float().reshape(-1).cpu()
    mask = torch.isfinite(x) & torch.isfinite(y)
    x = x[mask]
    y = y[mask]
    if x.numel() < 2:
        return float("nan")
    x = x - x.mean()
    y = y - y.mean()
    denom = x.norm() * y.norm()
    if float(denom.item()) == 0.0:
        return float("nan")
    return float((x @ y / denom).item())


def mean(values: list[float]) -> float:
    if not values:
        return float("nan")
    return float(sum(values) / len(values))


def quantile(values: list[float], q: float) -> float:
    if not values:
        return float("nan")
    t = torch.tensor(values, dtype=torch.float32)
    return float(torch.quantile(t, q).item())


def sample_ids(idx: torch.Tensor, batch_size: int) -> torch.Tensor:
    return idx[torch.randint(0, idx.numel(), (min(batch_size, idx.numel()),))]


@torch.no_grad()
def wm_and_policy_probe(agent, data, idx, nrm, args: argparse.Namespace, device: torch.device) -> dict[str, Any]:
    reward_pred_all: list[torch.Tensor] = []
    reward_true_all: list[torch.Tensor] = []
    action_mse: list[float] = []
    action_l1: list[float] = []
    policy_action_norm: list[float] = []
    dataset_action_norm: list[float] = []
    action_saturation: list[float] = []
    dyn_mse_by_h = {h: [] for h in [1, 2, 4, 8, 16] if h <= args.horizon}
    reward_mse_by_h = {h: [] for h in [1, 2, 4, 8, 16] if h <= args.horizon}
    dataset_pred_return: list[float] = []
    policy_pred_return: list[float] = []
    critic_value_mean: list[float] = []
    critic_value_std: list[float] = []

    for _ in range(args.max_batches):
        ids = sample_ids(idx, args.batch_size)
        obs, act, rew = batch_windows(data, ids, device, nrm, args)
        z0 = agent.wm.encode(obs[0], task=None)
        z = z0
        disc = 1.0
        dataset_ret = torch.zeros(z0.shape[0], device=device)
        for t in range(args.horizon):
            pred_rew = agent.wm.almost_two_hot_inv(agent.wm.reward(z, act[t], task=None)).squeeze(-1)
            true_rew = rew[t].squeeze(-1)
            reward_pred_all.append(pred_rew.detach().cpu())
            reward_true_all.append(true_rew.detach().cpu())
            dataset_ret = dataset_ret + disc * pred_rew
            z = agent.wm.next(z, act[t], task=None)
            if (t + 1) in dyn_mse_by_h:
                target_z = agent.wm.encode(obs[t + 1], task=None)
                dyn_mse_by_h[t + 1].append(float(F.mse_loss(z, target_z).item()))
                reward_mse_by_h[t + 1].append(float(F.mse_loss(pred_rew, true_rew).item()))
            disc *= args.gamma
        dataset_pred_return.extend(dataset_ret.detach().cpu().tolist())

        flat_obs = obs[:-1].reshape(-1, obs.shape[-1])
        flat_dataset_action = act.reshape(-1, act.shape[-1])
        flat_z = agent.wm.encode(flat_obs, task=None)
        flat_raw_action = agent.actor(flat_z, deterministic=True)
        flat_policy_action = torch.tanh(flat_raw_action).clamp(-1.0, 1.0)
        action_mse.append(float(F.mse_loss(flat_policy_action, flat_dataset_action).item()))
        action_l1.append(float((flat_policy_action - flat_dataset_action).abs().mean().item()))
        policy_action_norm.append(float(flat_policy_action.pow(2).mean(dim=-1).sqrt().mean().item()))
        dataset_action_norm.append(float(flat_dataset_action.pow(2).mean(dim=-1).sqrt().mean().item()))
        action_saturation.append(float((flat_policy_action.abs() > 0.95).float().mean().item()))
        values = agent.critic(flat_z).squeeze(-1)
        critic_value_mean.append(float(values.mean().item()))
        critic_value_std.append(float(values.std(unbiased=False).item()))

        z = z0
        disc = 1.0
        policy_ret = torch.zeros(z0.shape[0], device=device)
        for _t in range(args.horizon):
            policy_action = torch.tanh(agent.actor(z, deterministic=True)).clamp(-1.0, 1.0)
            pred_rew = agent.wm.almost_two_hot_inv(agent.wm.reward(z, policy_action, task=None)).squeeze(-1)
            policy_ret = policy_ret + disc * pred_rew
            z = agent.wm.next(z, policy_action, task=None)
            disc *= args.gamma
        policy_pred_return.extend(policy_ret.detach().cpu().tolist())

    reward_pred = torch.cat(reward_pred_all)
    reward_true = torch.cat(reward_true_all)
    return {
        "wm_reward_pred_mean": float(reward_pred.mean().item()),
        "wm_reward_true_mean": float(reward_true.mean().item()),
        "wm_reward_mse": float(F.mse_loss(reward_pred, reward_true).item()),
        "wm_reward_mae": float((reward_pred - reward_true).abs().mean().item()),
        "wm_reward_corr": corrcoef(reward_pred, reward_true),
        "wm_multistep_dyn_mse": {str(k): mean(v) for k, v in dyn_mse_by_h.items()},
        "wm_multistep_reward_mse": {str(k): mean(v) for k, v in reward_mse_by_h.items()},
        "dataset_pred_return_mean": mean(dataset_pred_return),
        "dataset_pred_return_p10": quantile(dataset_pred_return, 0.10),
        "dataset_pred_return_p90": quantile(dataset_pred_return, 0.90),
        "policy_pred_return_mean": mean(policy_pred_return),
        "policy_pred_return_p10": quantile(policy_pred_return, 0.10),
        "policy_pred_return_p90": quantile(policy_pred_return, 0.90),
        "policy_minus_dataset_pred_return": mean(policy_pred_return) - mean(dataset_pred_return),
        "policy_vs_dataset_action_mse": mean(action_mse),
        "policy_vs_dataset_action_l1": mean(action_l1),
        "policy_action_norm_mean": mean(policy_action_norm),
        "dataset_action_norm_mean": mean(dataset_action_norm),
        "policy_action_saturation_frac": mean(action_saturation),
        "critic_value_mean": mean(critic_value_mean),
        "critic_value_std": mean(critic_value_std),
    }


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    adapter_args = SimpleNamespace(
        dataset=args.dataset,
        metadata=args.metadata,
        normalization=args.normalization,
        seed=args.seed,
        device=args.device,
        output_dir=str(output_dir),
        task_id="Mjlab-Velocity-Flat-Unitree-G1",
        pretrain_iters=0,
        policy_iters=1,
        wm_batch_size=args.batch_size,
        policy_batch_size=args.batch_size,
        horizon=args.horizon,
        gamma=args.gamma,
        lam=args.lam,
        actor_lr=5e-4,
        critic_lr=5e-4,
        model_lr=3e-4,
        critic_iterations=8,
        critic_batches=4,
        num_critics=3,
        latent_dim=512,
        eval_every=1000,
        pretrain_log_every=1000,
        eval_episodes=1,
        eval_num_envs=1,
        episode_length=1000,
        command_dim=3,
        command_position="tail",
        obs_mode=args.obs_mode,
        reward_mode=args.reward_mode,
        rew_rms=args.rew_rms,
        ret_rms=args.ret_rms,
        skip_real_eval=True,
        wandb_project="",
        wandb_group="",
        wandb_name="",
        disable_wandb=True,
    )
    data, metadata, nrm, train_idx, val_idx, test_idx = load_data(adapter_args, device)
    obs_dim = int(data["phys_obs"].shape[-1]) + int(data["command"].shape[-1])
    action_dim = int(data["policy_action"].shape[-1])
    agent = build_pwm_agent(adapter_args, obs_dim, action_dim)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    agent.actor.load_state_dict(checkpoint["actor"])
    agent.critic.load_state_dict(checkpoint["critic"])
    agent.wm.load_state_dict(checkpoint["world_model"])
    agent.actor.eval()
    agent.critic.eval()
    agent.wm.eval()
    agent.obs_rms = checkpoint.get("obs_rms")
    agent.rew_rms = checkpoint.get("rew_rms")
    agent.ret_rms = checkpoint.get("ret_rms")

    t0 = time.time()
    splits = {
        "train": train_idx,
        "val": val_idx,
        "test": test_idx,
    }
    split_metrics = {
        name: wm_and_policy_probe(agent, data, idx, nrm, args, device)
        for name, idx in splits.items()
        if idx.numel() > 0
    }
    prior_summary = load_summary(Path(args.checkpoint).parent / "summary.json")
    prior_eval = load_summary(Path(args.checkpoint).parent / "eval_summary.json")
    summary = {
        "script": "analyze_original_pwm_collapse.py",
        "git_sha": git_sha(),
        "checkpoint": args.checkpoint,
        "checkpoint_kind": args.checkpoint_kind,
        "dataset": args.dataset,
        "metadata": args.metadata,
        "normalization": args.normalization,
        "seed": args.seed,
        "horizon": args.horizon,
        "batch_size": args.batch_size,
        "max_batches": args.max_batches,
        "obs_mode": args.obs_mode,
        "reward_mode": args.reward_mode,
        "prior_best_imagined_return_proxy": prior_summary.get("best_imagined_return_proxy"),
        "prior_best_iter": prior_summary.get("best_iter"),
        "prior_eval_return_mean": prior_summary.get("eval/return_mean") or prior_eval.get("return_mean"),
        "prior_eval_episode_length_mean": prior_summary.get("eval/episode_length_mean")
        or prior_eval.get("episode_length_mean"),
        "split_metrics": split_metrics,
        "wall_clock_seconds": time.time() - t0,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(summary, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
