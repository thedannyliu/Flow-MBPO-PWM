#!/usr/bin/env python3
"""
Phase 1 world-model-only training script.

Trains a world model on a fixed offline dataset and logs pure prediction metrics:
- one-step dynamics loss
- one-step reward loss
- rollout dynamics loss
- rollout reward loss
- total world-model loss
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Dict, Tuple

import torch
import torch.nn.functional as F
import wandb
from hydra.utils import instantiate
from omegaconf import OmegaConf
from tensordict import TensorDict

from flow_mbpo_pwm.algorithms.pwm import PWM


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, help="Path to fixed-window dataset .pt")
    parser.add_argument("--metadata", default=None, help="Optional dataset metadata json path")
    parser.add_argument("--alg-config", required=True, help="Path to algorithm yaml")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--train-iters", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--eval-batch-size", type=int, default=256)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--split-seed", type=int, default=0)
    parser.add_argument("--wandb-project", default="flow-mbpo-phase1-wm-overfit")
    parser.add_argument("--wandb-entity", default=None)
    parser.add_argument("--wandb-group", default="phase1_overfit")
    parser.add_argument("--wandb-name", default=None)
    parser.add_argument("--wandb-job-type", default="train")
    parser.add_argument("--wandb-tags", default="phase1,world_model_only,overfit")
    parser.add_argument("--disable-wandb", action="store_true")
    return parser.parse_args()


def _load_dataset(path: Path) -> Dict[str, torch.Tensor]:
    data = torch.load(path, map_location="cpu", weights_only=False)
    required = {"obs", "action", "reward", "term", "source_episode", "source_start"}
    missing = required.difference(data.keys())
    if missing:
        raise KeyError(f"Dataset missing keys: {sorted(missing)}")
    return data


def _split_by_source_episode(
    source_episode: torch.Tensor,
    val_ratio: float,
    split_seed: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    unique_eps = torch.unique(source_episode).tolist()
    if len(unique_eps) < 2:
        raise RuntimeError("Need at least two source episodes to form train/val splits.")
    gen = torch.Generator().manual_seed(split_seed)
    perm = torch.tensor(unique_eps)[torch.randperm(len(unique_eps), generator=gen)]
    num_val = max(1, int(round(len(unique_eps) * val_ratio)))
    if num_val >= len(unique_eps):
        num_val = len(unique_eps) - 1
    val_eps = perm[:num_val]
    train_eps = perm[num_val:]
    train_mask = torch.isin(source_episode, train_eps)
    val_mask = torch.isin(source_episode, val_eps)
    return train_mask, val_mask


def _subset_as_tensordict(data: Dict[str, torch.Tensor], mask: torch.Tensor) -> TensorDict:
    return TensorDict(
        {
            "obs": data["obs"][mask].clone(),
            "action": data["action"][mask].clone(),
            "reward": data["reward"][mask].clone(),
            "term": data["term"][mask].clone(),
        },
        batch_size=(int(mask.sum().item()), int(data["obs"].shape[1])),
    )


def _prepare_batch(td: TensorDict, device: torch.device):
    obs = td["obs"].permute(1, 0, 2).to(device)
    act = td["action"].permute(1, 0, 2)[1:].to(device)
    rew = td["reward"].permute(1, 0).unsqueeze(-1)[1:].to(device)
    return obs, act, rew


def _sample_fixed_window_batch(agent: PWM, td: TensorDict, batch_size: int):
    idx = torch.randint(0, td.batch_size[0], (min(batch_size, td.batch_size[0]),))
    batch = td[idx]
    obs, act, rew = _prepare_batch(batch, agent.device)
    if agent.obs_rms:
        obs = agent.obs_rms.normalize(obs)
    if agent.rew_rms:
        rew = agent.rew_rms.normalize(rew)
    return obs, act, rew


def _reward_to_scalar(agent: PWM, rew_hat: torch.Tensor) -> torch.Tensor:
    if rew_hat.ndim > 0 and rew_hat.shape[-1] > 1:
        rew_hat = agent.wm.almost_two_hot_inv(rew_hat)
    if rew_hat.ndim > 1 and rew_hat.shape[-1] == 1:
        rew_hat = rew_hat.squeeze(-1)
    return rew_hat


def _chunk_loss(agent: PWM, obs: torch.Tensor, act: torch.Tensor, rew: torch.Tensor):
    chunk_size = int(getattr(agent.wm, "chunk_size", 1))
    if chunk_size <= 1:
        raise RuntimeError("Chunk loss requested for a non-chunked world model.")
    all_z = agent.wm.encode(obs, None)
    with torch.no_grad():
        target_z = agent.wm.encode(obs, None)
    dyn_loss = 0.0
    rew_loss = 0.0
    count = 0
    if hasattr(agent.wm, "predict_chunk_sequence"):
        preds = agent.wm.predict_chunk_sequence(all_z, act, rew, None)
        for start in range(preds.shape[0]):
            dyn_loss = dyn_loss + F.mse_loss(preds[start], target_z[start + chunk_size]) * agent.gamma**start
            count += 1
    else:
        for start in range(0, agent.horizon - chunk_size + 1):
            z0 = all_z[start]
            action_chunk = act[start : start + chunk_size]
            pred = agent.wm.next_chunk(z0, action_chunk, None)
            dyn_loss = dyn_loss + F.mse_loss(pred, target_z[start + chunk_size]) * agent.gamma**start
            count += 1
    reward_count = 0
    for start in range(0, agent.horizon - chunk_size + 1):
        rew_hat = agent.wm.reward(all_z[start], act[start], None)
        rew_hat = _reward_to_scalar(agent, rew_hat)
        rew_target = rew[start]
        if rew_target.ndim > 1 and rew_target.shape[-1] == 1:
            rew_target = rew_target.squeeze(-1)
        rew_loss = rew_loss + F.mse_loss(rew_hat, rew_target) * agent.gamma**start
        reward_count += 1
    dyn_loss = dyn_loss / max(1, count)
    rew_loss = rew_loss / max(1, reward_count)
    extra = _rollout_consistency_extra(agent, all_z, act, rew)
    return dyn_loss + rew_loss + extra, dyn_loss + extra.detach() * 0.0, rew_loss


def _sequence_loss(agent: PWM, obs: torch.Tensor, act: torch.Tensor, rew: torch.Tensor):
    all_z = agent.wm.encode(obs, None)
    with torch.no_grad():
        target_z = agent.wm.encode(obs[1:], None)
    preds = agent.wm.predict_next_sequence(all_z[:-1], act, rew, None)
    discount = (agent.gamma ** torch.arange(agent.horizon, device=obs.device)).view(agent.horizon, 1, 1)
    dyn_loss = ((preds - target_z) ** 2 * discount).mean()
    rew_hat = agent.wm.reward(all_z[:-1], act, None)
    rew_hat = _reward_to_scalar(agent, rew_hat)
    rew_target = rew
    if rew_target.ndim > 2 and rew_target.shape[-1] == 1:
        rew_target = rew_target.squeeze(-1)
    rew_loss = ((rew_hat - rew_target) ** 2).mean()
    extra = _rollout_consistency_extra(agent, all_z, act, rew)
    return dyn_loss + rew_loss + extra, dyn_loss + extra.detach() * 0.0, rew_loss


def _rollout_consistency_extra(agent: PWM, all_z: torch.Tensor, act: torch.Tensor, rew: torch.Tensor):
    weight = float(getattr(agent.wm, "rollout_consistency_weight", 0.0))
    reward_weight = float(getattr(agent.wm, "rollout_reward_consistency_weight", 0.0))
    if weight <= 0 and reward_weight <= 0:
        return torch.zeros((), device=all_z.device, dtype=all_z.dtype)
    steps = int(getattr(agent.wm, "rollout_consistency_steps", 0))
    steps = min(steps if steps > 0 else agent.horizon, agent.horizon)
    loss = torch.zeros((), device=all_z.device, dtype=all_z.dtype)
    count = 0
    for start in range(agent.horizon):
        z = all_z[start]
        max_k = min(steps, agent.horizon - start)
        for k in range(max_k):
            z_prev = z
            z = agent.wm.next(z, act[start + k], None)
            if weight > 0:
                loss = loss + weight * F.mse_loss(z, all_z[start + k + 1].detach()) * agent.gamma ** (start + k)
            if reward_weight > 0:
                rew_hat = _reward_to_scalar(agent, agent.wm.reward(z_prev, act[start + k], None))
                target = rew[start + k]
                if target.ndim > 1 and target.shape[-1] == 1:
                    target = target.squeeze(-1)
                loss = loss + reward_weight * F.mse_loss(rew_hat, target) * agent.gamma ** (start + k)
            count += 1
    return loss / max(1, count)


def _pretrain_residual_base(agent: PWM, train_td: TensorDict, args: argparse.Namespace) -> None:
    pretrain_iters = int(getattr(agent.wm, "base_pretrain_iters", 0))
    pretrain_iters = min(pretrain_iters, max(1, int(args.train_iters)))
    if pretrain_iters <= 0:
        return
    print(f"Pretraining residual-flow MLP base for {pretrain_iters} iters")
    params = list(agent.wm._encoder.parameters()) + list(agent.wm._dynamics.parameters()) + list(agent.wm._reward.parameters())
    opt = torch.optim.Adam(params, lr=agent.model_lr)
    for i in range(pretrain_iters):
        idx = torch.randint(0, train_td.batch_size[0], (min(args.batch_size, train_td.batch_size[0]),))
        batch = train_td[idx]
        obs, act, rew = _prepare_batch(batch, agent.device)
        if agent.obs_rms:
            obs = agent.obs_rms.normalize(obs)
        if agent.rew_rms:
            rew = agent.rew_rms.normalize(rew)
        with torch.no_grad():
            target = agent.wm.encode(obs[1:], None)
        z = agent.wm.encode(obs[0], None)
        dyn_loss = 0.0
        rew_loss = 0.0
        for t in range(agent.horizon):
            z_prev = z
            z = agent.wm.base_next(z, act[t], None)
            dyn_loss = dyn_loss + F.mse_loss(z, target[t]) * agent.gamma**t
            rew_hat = _reward_to_scalar(agent, agent.wm.reward(z_prev, act[t], None))
            rew_target = rew[t]
            if rew_target.ndim > 1 and rew_target.shape[-1] == 1:
                rew_target = rew_target.squeeze(-1)
            rew_loss = rew_loss + F.mse_loss(rew_hat, rew_target) * agent.gamma**t
        loss = (dyn_loss + rew_loss) / agent.horizon
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, agent.wm_grad_norm)
        opt.step()
        if i % max(1, pretrain_iters // 5) == 0:
            print(f"[base-pretrain {i}/{pretrain_iters}] loss={float(loss.detach()):.6f}")
    if getattr(agent.wm, "freeze_base_after_pretrain", False):
        agent.wm.freeze_base()
        print("Froze residual-flow encoder and MLP base after pretraining")


@torch.no_grad()
def evaluate_split(agent: PWM, td: TensorDict, batch_size: int) -> Dict[str, float]:
    if td.batch_size[0] == 0:
        raise RuntimeError("Empty split encountered during evaluation.")

    total = 0
    agg = {
        "wm_loss": 0.0,
        "one_step_dyn_loss": 0.0,
        "one_step_reward_loss": 0.0,
        "rollout_dyn_loss": 0.0,
        "rollout_reward_loss": 0.0,
        "base_rollout_dyn_loss": 0.0,
        "residual_norm": 0.0,
        "residual_contribution": 0.0,
        "chunk_endpoint_dyn_loss": 0.0,
        "chunk_rollout_dyn_loss": 0.0,
        "chunk_base_endpoint_dyn_loss": 0.0,
        "chunk_residual_norm": 0.0,
        "chunk_residual_contribution": 0.0,
        "gate_entropy": 0.0,
        "gate_usage_max": 0.0,
        "latent_action_norm": 0.0,
    }
    horizon = agent.horizon

    for start in range(0, td.batch_size[0], batch_size):
        end = min(start + batch_size, td.batch_size[0])
        batch = td[start:end]
        obs, act, rew = _prepare_batch(batch, agent.device)

        if agent.obs_rms:
            obs = agent.obs_rms.normalize(obs)
        if agent.rew_rms:
            rew = agent.rew_rms.normalize(rew)

        if getattr(agent.wm, "chunked_dynamics", False):
            loss, _, _ = _chunk_loss(agent, obs, act, rew)
        elif getattr(agent.wm, "sequence_dynamics", False):
            loss, _, _ = _sequence_loss(agent, obs, act, rew)
        else:
            loss, _, _ = agent.compute_wm_loss(obs, act, rew)
            loss = loss + _rollout_consistency_extra(agent, agent.wm.encode(obs, None), act, rew)

        with torch.no_grad():
            next_z_target = agent.wm.encode(obs[1:], None)
            z = agent.wm.encode(obs[0], None)
            dyn_losses = []
            rew_losses = []
            residual_norms = []
            residual_contributions = []
            base_dyn_losses = []

            for t in range(horizon):
                z_prev = z
                if getattr(agent.wm, "residual_flow_dynamics", False):
                    base_z = agent.wm.base_next(z_prev, act[t], None)
                    z = agent.wm.next(z_prev, act[t], None)
                    residual = z - base_z
                    base_err = F.mse_loss(base_z, next_z_target[t])
                    pred_err = F.mse_loss(z, next_z_target[t])
                    residual_norms.append(residual.norm(dim=-1).mean())
                    residual_contributions.append(base_err - pred_err)
                    base_dyn_losses.append(base_err)
                    if hasattr(agent.wm, "gate_probs"):
                        probs = agent.wm.gate_probs(z_prev, act[t]).clamp_min(1e-8)
                        agg["gate_entropy"] += float((-(probs * probs.log()).sum(dim=-1).mean()).item()) * (end - start) / horizon
                        agg["gate_usage_max"] += float(probs.mean(dim=0).max().item()) * (end - start) / horizon
                elif agent.use_flow_dynamics:
                    z = agent.wm.next(
                        z,
                        act[t],
                        None,
                        integrator=agent.flow_integrator,
                        substeps=agent.flow_substeps,
                    )
                elif getattr(agent.wm, "sequence_dynamics", False):
                    z = agent.wm.next(z, act[t], None)
                else:
                    z = agent.wm.next(z, act[t], None)
                dyn_losses.append(F.mse_loss(z, next_z_target[t]))

                rew_hat = agent.wm.reward(z_prev, act[t], None)
                rew_hat = _reward_to_scalar(agent, rew_hat)
                rew_target = rew[t]
                if rew_target.ndim > 1 and rew_target.shape[-1] == 1:
                    rew_target = rew_target.squeeze(-1)
                rew_losses.append(F.mse_loss(rew_hat, rew_target))

        batch_count = end - start
        total += batch_count
        agg["wm_loss"] += float(loss.item()) * batch_count
        agg["one_step_dyn_loss"] += float(dyn_losses[0].item()) * batch_count
        agg["one_step_reward_loss"] += float(rew_losses[0].item()) * batch_count
        agg["rollout_dyn_loss"] += float(torch.stack(dyn_losses).mean().item()) * batch_count
        agg["rollout_reward_loss"] += float(torch.stack(rew_losses).mean().item()) * batch_count
        if getattr(agent.wm, "residual_flow_dynamics", False):
            agg["residual_norm"] += float(torch.stack(residual_norms).mean().item()) * batch_count
            agg["residual_contribution"] += float(torch.stack(residual_contributions).mean().item()) * batch_count
            agg["base_rollout_dyn_loss"] += float(torch.stack(base_dyn_losses).mean().item()) * batch_count
        else:
            agg["base_rollout_dyn_loss"] += float(torch.stack(dyn_losses).mean().item()) * batch_count

        if getattr(agent.wm, "chunked_dynamics", False):
            chunk_size = int(getattr(agent.wm, "chunk_size", 1))
            all_targets = agent.wm.encode(obs, None)
            endpoint_losses = []
            chunk_base_losses = []
            chunk_residual_norms = []
            chunk_residual_contributions = []
            context_preds = None
            if hasattr(agent.wm, "predict_chunk_sequence"):
                context_preds = agent.wm.predict_chunk_sequence(all_targets, act, rew, None)
            for s in range(0, horizon - chunk_size + 1):
                if hasattr(agent.wm, "latent_action"):
                    latent_action = agent.wm.latent_action(act[s : s + chunk_size])
                    agg["latent_action_norm"] += float(latent_action.norm(dim=-1).mean().item()) * batch_count / max(1, horizon - chunk_size + 1)
                if context_preds is not None:
                    pred = context_preds[s]
                elif getattr(agent.wm, "chunked_residual_flow_dynamics", False):
                    base_pred = agent.wm.base_next_chunk(all_targets[s], act[s : s + chunk_size], None)
                    pred = agent.wm.next_chunk(all_targets[s], act[s : s + chunk_size], None)
                    residual = pred - base_pred
                    base_err = F.mse_loss(base_pred, all_targets[s + chunk_size])
                    pred_err = F.mse_loss(pred, all_targets[s + chunk_size])
                    chunk_base_losses.append(base_err)
                    chunk_residual_norms.append(residual.norm(dim=-1).mean())
                    chunk_residual_contributions.append(base_err - pred_err)
                elif hasattr(agent.wm, "next_chunk"):
                    pred = agent.wm.next_chunk(all_targets[s], act[s : s + chunk_size], None)
                else:
                    continue
                endpoint_losses.append(F.mse_loss(pred, all_targets[s + chunk_size]))
            if endpoint_losses:
                chunk_endpoint = torch.stack(endpoint_losses).mean()
                agg["chunk_endpoint_dyn_loss"] += float(chunk_endpoint.item()) * batch_count
                if chunk_base_losses:
                    agg["chunk_base_endpoint_dyn_loss"] += float(torch.stack(chunk_base_losses).mean().item()) * batch_count
                    agg["chunk_residual_norm"] += float(torch.stack(chunk_residual_norms).mean().item()) * batch_count
                    agg["chunk_residual_contribution"] += float(torch.stack(chunk_residual_contributions).mean().item()) * batch_count
                z_chunk = all_targets[0]
                chunk_rollout_losses = []
                for s in range(0, horizon - chunk_size + 1, chunk_size):
                    z_chunk = agent.wm.next_chunk(z_chunk, act[s : s + chunk_size], None)
                    chunk_rollout_losses.append(F.mse_loss(z_chunk, all_targets[s + chunk_size]))
                if chunk_rollout_losses:
                    agg["chunk_rollout_dyn_loss"] += float(torch.stack(chunk_rollout_losses).mean().item()) * batch_count

    return {k: v / total for k, v in agg.items()}


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset_path = Path(args.dataset)
    metadata_path = Path(args.metadata) if args.metadata else dataset_path.with_suffix(".json")
    data = _load_dataset(dataset_path)
    metadata = json.loads(metadata_path.read_text()) if metadata_path.exists() else {}

    train_mask, val_mask = _split_by_source_episode(
        data["source_episode"],
        val_ratio=args.val_ratio,
        split_seed=args.split_seed,
    )
    train_td = _subset_as_tensordict(data, train_mask)
    val_td = _subset_as_tensordict(data, val_mask)

    seq_len = int(train_td["obs"].shape[1])
    obs_dim = int(train_td["obs"].shape[-1])
    act_dim = int(train_td["action"].shape[-1])
    horizon = seq_len - 1

    alg_cfg = OmegaConf.load(args.alg_config)
    alg_cfg.horizon = horizon
    alg_cfg.max_epochs = 0
    alg_cfg.wm_batch_size = args.batch_size
    alg_cfg.device = args.device

    agent = instantiate(
        alg_cfg,
        env=None,
        obs_dim=obs_dim,
        act_dim=act_dim,
        logdir=str(output_dir),
        log=not args.disable_wandb,
    )

    if agent.obs_rms:
        obs = train_td["obs"].reshape(-1, obs_dim)
        obs = torch.nan_to_num(obs)
        agent.obs_rms.update(obs.to(agent.device))

    if agent.rew_rms:
        rew = train_td["reward"][:, 1:].reshape(-1, 1)
        rew = torch.nan_to_num(rew)
        agent.rew_rms.update(rew.to(agent.device))

    if getattr(agent.wm, "residual_flow_dynamics", False):
        _pretrain_residual_base(agent, train_td, args)

    wandb_run = None
    if not args.disable_wandb:
        tags = [t.strip() for t in args.wandb_tags.split(",") if t.strip()]
        tags.extend(
            [
                f"task_{metadata.get('task_id_requested', 'unknown')}",
                f"seed_{args.seed}",
                f"horizon_{horizon}",
                f"wm_{'flow' if agent.use_flow_dynamics else 'mlp'}",
            ]
        )
        wandb_run = wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            group=args.wandb_group,
            name=args.wandb_name
            or f"{metadata.get('task_id_requested', 'task')}_{Path(args.alg_config).stem}_seed{args.seed}",
            job_type=args.wandb_job_type,
            tags=tags,
            config={
                **vars(args),
                "dataset_metadata": metadata,
                "obs_dim": obs_dim,
                "act_dim": act_dim,
                "horizon": horizon,
                "train_windows": int(train_td.batch_size[0]),
                "val_windows": int(val_td.batch_size[0]),
            },
        )

    start_time = time.time()
    best_val = math.inf
    best_metrics = None

    for i in range(args.train_iters):
        obs, act, rew = _sample_fixed_window_batch(agent, train_td, args.batch_size)

        agent.wm_optimizer.zero_grad()
        if getattr(agent.wm, "chunked_dynamics", False):
            loss, dyn_loss, rew_loss = _chunk_loss(agent, obs, act, rew)
        elif getattr(agent.wm, "sequence_dynamics", False):
            loss, dyn_loss, rew_loss = _sequence_loss(agent, obs, act, rew)
        else:
            loss, dyn_loss, rew_loss = agent.compute_wm_loss(obs, act, rew)
            extra = _rollout_consistency_extra(agent, agent.wm.encode(obs, None), act, rew)
            if float(extra.detach()) != 0.0:
                loss = loss + extra
                dyn_loss = dyn_loss + extra.detach() * 0.0
        loss.backward()
        wm_grad_norm = torch.nn.utils.clip_grad_norm_(agent.wm.parameters(), agent.wm_grad_norm)
        agent.wm_optimizer.step()

        if i % args.eval_every == 0 or i == args.train_iters - 1:
            train_metrics = evaluate_split(agent, train_td, args.eval_batch_size)
            val_metrics = evaluate_split(agent, val_td, args.eval_batch_size)
            metrics = {
                "phase1/train_total_wm_loss": train_metrics["wm_loss"],
                "phase1/train_one_step_dyn_loss": train_metrics["one_step_dyn_loss"],
                "phase1/train_one_step_reward_loss": train_metrics["one_step_reward_loss"],
                "phase1/train_rollout_dyn_loss": train_metrics["rollout_dyn_loss"],
                "phase1/train_rollout_reward_loss": train_metrics["rollout_reward_loss"],
                "phase1/train_base_rollout_dyn_loss": train_metrics["base_rollout_dyn_loss"],
                "phase1/train_residual_norm": train_metrics["residual_norm"],
                "phase1/train_residual_contribution": train_metrics["residual_contribution"],
                "phase1/train_chunk_endpoint_dyn_loss": train_metrics["chunk_endpoint_dyn_loss"],
                "phase1/train_chunk_rollout_dyn_loss": train_metrics["chunk_rollout_dyn_loss"],
                "phase1/train_chunk_base_endpoint_dyn_loss": train_metrics["chunk_base_endpoint_dyn_loss"],
                "phase1/train_chunk_residual_norm": train_metrics["chunk_residual_norm"],
                "phase1/train_chunk_residual_contribution": train_metrics["chunk_residual_contribution"],
                "phase1/train_gate_entropy": train_metrics["gate_entropy"],
                "phase1/train_gate_usage_max": train_metrics["gate_usage_max"],
                "phase1/train_latent_action_norm": train_metrics["latent_action_norm"],
                "phase1/val_total_wm_loss": val_metrics["wm_loss"],
                "phase1/val_one_step_dyn_loss": val_metrics["one_step_dyn_loss"],
                "phase1/val_one_step_reward_loss": val_metrics["one_step_reward_loss"],
                "phase1/val_rollout_dyn_loss": val_metrics["rollout_dyn_loss"],
                "phase1/val_rollout_reward_loss": val_metrics["rollout_reward_loss"],
                "phase1/val_base_rollout_dyn_loss": val_metrics["base_rollout_dyn_loss"],
                "phase1/val_residual_norm": val_metrics["residual_norm"],
                "phase1/val_residual_contribution": val_metrics["residual_contribution"],
                "phase1/val_chunk_endpoint_dyn_loss": val_metrics["chunk_endpoint_dyn_loss"],
                "phase1/val_chunk_rollout_dyn_loss": val_metrics["chunk_rollout_dyn_loss"],
                "phase1/val_chunk_base_endpoint_dyn_loss": val_metrics["chunk_base_endpoint_dyn_loss"],
                "phase1/val_chunk_residual_norm": val_metrics["chunk_residual_norm"],
                "phase1/val_chunk_residual_contribution": val_metrics["chunk_residual_contribution"],
                "phase1/val_gate_entropy": val_metrics["gate_entropy"],
                "phase1/val_gate_usage_max": val_metrics["gate_usage_max"],
                "phase1/val_latent_action_norm": val_metrics["latent_action_norm"],
                "phase1/iter_train_total_wm_loss": float(loss.item()),
                "phase1/iter_train_dyn_loss": float(dyn_loss.detach()),
                "phase1/iter_train_reward_loss": float(rew_loss.detach()),
                "phase1/wm_grad_norm": float(wm_grad_norm.detach()),
                "phase1/elapsed_seconds": time.time() - start_time,
            }
            if wandb_run is not None:
                wandb.log(metrics, step=i)
            if val_metrics["rollout_dyn_loss"] < best_val:
                best_val = val_metrics["rollout_dyn_loss"]
                best_metrics = {
                    "iter": i,
                    **train_metrics,
                    **{f"val_{k}": v for k, v in val_metrics.items()},
                }
                agent.save("best_world_model")

        if i % args.log_every == 0:
            print(
                f"[{i}/{args.train_iters}] loss={loss.item():.6f} dyn={float(dyn_loss.detach()):.6f} "
                f"rew={float(rew_loss):.6f} grad={float(wm_grad_norm.detach()):.4f}"
            )

    agent.save("final_world_model")

    final_train = evaluate_split(agent, train_td, args.eval_batch_size)
    final_val = evaluate_split(agent, val_td, args.eval_batch_size)
    summary = {
        "dataset": str(dataset_path),
        "dataset_metadata": metadata,
        "alg_config": str(args.alg_config),
        "seed": int(args.seed),
        "horizon": int(horizon),
        "train_windows": int(train_td.batch_size[0]),
        "val_windows": int(val_td.batch_size[0]),
        "elapsed_seconds": time.time() - start_time,
        "best_val_rollout_dyn_loss": float(best_val),
        "best_metrics": best_metrics,
        "final_train": final_train,
        "final_val": final_val,
    }
    (output_dir / "phase1_summary.json").write_text(json.dumps(summary, indent=2))

    if wandb_run is not None:
        wandb_run.summary.update(
            {
                "phase1_final_train_wm_loss": final_train["wm_loss"],
                "phase1_final_val_wm_loss": final_val["wm_loss"],
                "phase1_final_train_one_step_dyn_loss": final_train["one_step_dyn_loss"],
                "phase1_final_val_one_step_dyn_loss": final_val["one_step_dyn_loss"],
                "phase1_final_train_rollout_dyn_loss": final_train["rollout_dyn_loss"],
                "phase1_final_val_rollout_dyn_loss": final_val["rollout_dyn_loss"],
                "phase1_final_train_rollout_reward_loss": final_train["rollout_reward_loss"],
                "phase1_final_val_rollout_reward_loss": final_val["rollout_reward_loss"],
                "phase1_final_train_base_rollout_dyn_loss": final_train["base_rollout_dyn_loss"],
                "phase1_final_val_base_rollout_dyn_loss": final_val["base_rollout_dyn_loss"],
                "phase1_final_train_residual_norm": final_train["residual_norm"],
                "phase1_final_val_residual_norm": final_val["residual_norm"],
                "phase1_final_train_residual_contribution": final_train["residual_contribution"],
                "phase1_final_val_residual_contribution": final_val["residual_contribution"],
                "phase1_final_train_chunk_endpoint_dyn_loss": final_train["chunk_endpoint_dyn_loss"],
                "phase1_final_val_chunk_endpoint_dyn_loss": final_val["chunk_endpoint_dyn_loss"],
                "phase1_final_train_chunk_rollout_dyn_loss": final_train["chunk_rollout_dyn_loss"],
                "phase1_final_val_chunk_rollout_dyn_loss": final_val["chunk_rollout_dyn_loss"],
                "phase1_best_val_rollout_dyn_loss": best_val,
            }
        )
        wandb_run.finish()

    print(f"Wrote summary to {output_dir / 'phase1_summary.json'}")


if __name__ == "__main__":
    main()
