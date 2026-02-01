"""Batched online trainer for GPU environments (MuJoCo Playground).

Adapts TDMPC2's OnlineTrainer for batched environments with auto-reset,
using BatchedEpisodeTracker to handle episode boundaries.
"""

from time import time
from typing import Optional

import numpy as np
import torch

from trainer.base import Trainer
from .episode_tracker import BatchedEpisodeTracker


class BatchedOnlineTrainer(Trainer):
    """Trainer for batched GPU environments (RSLRLBraxWrapper)."""

    def __init__(self, cfg, env, agent, buffer, logger):
        super().__init__(cfg, env, agent, buffer, logger)

        self.num_envs = env.num_envs
        self.tracker = BatchedEpisodeTracker(
            num_envs=self.num_envs,
            obs_dim=env.num_obs,
            act_dim=env.num_actions,
            device='cuda',
        )

        self._step = 0
        self._ep_idx = 0
        self._start_time = time()
        self._pretrain_done = False

        self._min_episodes_for_pretrain = cfg.seed_steps // cfg.episode_length

        self._total_episodes_collected = 0
        self._last_progress_step = 0

    def common_metrics(self):
        """Return a dictionary of current metrics."""
        elapsed_time = time() - self._start_time
        return dict(
            step=self._step,
            episode=self._ep_idx,
            elapsed_time=elapsed_time,
            steps_per_second=self._step / elapsed_time if elapsed_time > 0 else 0,
        )

    def eval(self):
        """Evaluate using batched environment with aggregation."""
        obs_dict = self.env.reset()
        obs = obs_dict["state"]

        ep_rewards = torch.zeros(self.num_envs, device='cuda')
        ep_lengths = torch.zeros(self.num_envs, dtype=torch.long, device='cuda')
        ep_done = torch.zeros(self.num_envs, dtype=torch.bool, device='cuda')
        t0_mask = torch.ones(self.num_envs, dtype=torch.bool, device='cuda')

        completed_rewards = []
        completed_lengths = []

        max_steps = self.cfg.episode_length * 2

        for step in range(max_steps):
            actions = self._act_batched(obs, t0_mask, eval_mode=True)

            obs_dict, rewards, dones, info = self.env.step(actions)
            obs = obs_dict["state"]

            active = ~ep_done
            ep_rewards[active] += rewards[active]
            ep_lengths[active] += 1

            newly_done = dones.bool() & ~ep_done
            if newly_done.any():
                for i in torch.where(newly_done)[0]:
                    completed_rewards.append(ep_rewards[i].item())
                    completed_lengths.append(ep_lengths[i].item())
                ep_done |= dones.bool()

            t0_mask = newly_done
            if len(completed_rewards) >= self.cfg.eval_episodes:
                break
            if ep_done.all():
                break

        return dict(
            episode_reward=np.mean(completed_rewards) if completed_rewards else 0.0,
            episode_length=np.mean(completed_lengths) if completed_lengths else 0,
        )

    def train(self):
        """Train with batched environment collection."""
        obs_dict = self.env.reset()
        obs = obs_dict["state"]  # (num_envs, obs_dim)

        self.tracker.init_episodes(obs)

        train_metrics = {}
        eval_next = True

        while self._step <= self.cfg.steps:
            if eval_next:
                eval_metrics = self.eval()
                eval_metrics.update(self.common_metrics())
                self.logger.log(eval_metrics, 'eval')
                eval_next = False

            if self._step > 0 and self._step % self.cfg.eval_freq < self.num_envs:
                eval_next = True

            if self.buffer.num_eps >= self._min_episodes_for_pretrain:
                t0_mask = self.tracker.get_t0_mask()
                actions = self._act_batched(obs, t0_mask, eval_mode=False)
            else:
                actions = self._random_actions()

            obs_dict, rewards, dones, info = self.env.step(actions)
            obs = obs_dict["state"]

            truncated = info.get("time_outs", torch.zeros_like(dones))
            terminated = dones.bool() & ~truncated.bool()

            self.tracker.add_step(obs, actions, rewards, dones.bool(), terminated.float())

            completed_eps = self.tracker.get_completed_episodes()
            for episode in completed_eps:
                self._ep_idx = self.buffer.add(episode)
                self._total_episodes_collected += 1

                ep_reward = episode['reward'][1:].nansum()
                ep_length = len(episode) - 1
                ep_terminated = episode['terminated'][-1].item()

                train_metrics.update(
                    episode_reward=ep_reward,
                    episode_length=ep_length,
                    episode_terminated=ep_terminated,
                )
                train_metrics.update(self.common_metrics())
                self.logger.log(train_metrics, 'train')

            if self._step - self._last_progress_step >= 1000 or (completed_eps and self._total_episodes_collected <= 5):
                print(f"  [progress] step={self._step:>5} | buffer_eps={self.buffer.num_eps} | "
                      f"total_collected={self._total_episodes_collected} | pretrain_done={self._pretrain_done}")
                self._last_progress_step = self._step

            if self.buffer.num_eps >= self._min_episodes_for_pretrain:
                is_pretrain_burst = not self._pretrain_done
                if is_pretrain_burst:
                    num_updates = self.cfg.seed_steps
                    print(f'\n  >>> PRETRAINING: {num_updates} updates on {self.buffer.num_eps} seed episodes...')
                else:
                    num_updates = 1

                for i in range(num_updates):
                    _train_metrics = self.agent.update(self.buffer)
                    # Log pretrain progress periodically
                    if is_pretrain_burst and ((i + 1) % 100 == 0 or i == num_updates - 1):
                        if 'total_loss' in _train_metrics:
                            print(f"      pretrain update {i+1}/{num_updates}: loss={_train_metrics['total_loss']:.4f}")

                train_metrics.update(_train_metrics)

                if is_pretrain_burst:
                    self._pretrain_done = True
                    print(f"  >>> PRETRAINING COMPLETE\n")

            self._step += self.num_envs

        self.logger.finish(self.agent)

    def _act_batched(
        self,
        obs: torch.Tensor,
        t0_mask: torch.Tensor,
        eval_mode: bool = False,
    ) -> torch.Tensor:
        """Get actions for all environments.

        Uses batched inference if available
        """
        if hasattr(self.agent, 'act_batched'):
            torch.compiler.cudagraph_mark_step_begin()
            return self.agent.act_batched(obs, t0_mask, eval_mode=eval_mode)
        else:
            actions = []
            for i in range(self.num_envs):
                torch.compiler.cudagraph_mark_step_begin()
                action = self.agent.act(obs[i], t0=bool(t0_mask[i]), eval_mode=eval_mode)
                actions.append(action)
            return torch.stack(actions).to(obs.device)

    def _random_actions(self) -> torch.Tensor:
        """Generate uniform random actions in [-1, 1]."""
        return torch.rand(self.num_envs, self.env.num_actions, device='cuda') * 2 - 1
