"""Batched inference extension for TDMPC2.
reshape (num_envs, num_samples) → (num_envs * num_samples) for forward passes.
"""

import torch
import torch.nn.functional as F
from common import math


class BatchedTDMPC2Mixin:
    """

    Usage:
        class BatchedTDMPC2(BatchedTDMPC2Mixin, TDMPC2):
            pass

        agent = BatchedTDMPC2(cfg)
        agent.init_batched(num_envs=64)
        actions = agent.act_batched(obs_batch, t0_mask)
    """

    def init_batched(self, num_envs: int):
        """Initialize per-environment buffers for batched inference.

        Args:
            num_envs: Number of parallel environments.
        """
        self.num_envs = num_envs

        # (num_envs, horizon, action_dim)
        self._prev_mean_batched = torch.zeros(
            num_envs, self.cfg.horizon, self.cfg.action_dim,
            device=self.device
        )

        self._z_buffer = None
        self._actions_buffer = None

    @torch.no_grad()
    def act_batched(
        self,
        obs: torch.Tensor,
        t0_mask: torch.Tensor,
        eval_mode: bool = False,
        task: torch.Tensor = None,
    ) -> torch.Tensor:
        """Select actions for multiple environments in parallel.

        Args:
            obs: Batched observations (num_envs, obs_dim).
            t0_mask: Boolean mask indicating episode starts (num_envs,).
            eval_mode: Whether to use deterministic action selection.
            task: Task indices for multi-task (num_envs,) or None.

        Returns:
            Batched actions (num_envs, action_dim).
        """
        obs = obs.to(self.device, non_blocking=True)

        if self.cfg.mpc:
            return self._plan_batched(obs, t0_mask, eval_mode, task)
        else:
            z = self.model.encode(obs, task)
            action, info = self.model.pi(z, task)
            if eval_mode:
                action = info["mean"]
            return action

    @torch.no_grad()
    def _plan_batched(
        self,
        obs: torch.Tensor,
        t0_mask: torch.Tensor,
        eval_mode: bool = False,
        task: torch.Tensor = None,
    ) -> torch.Tensor:
        """Batched MPPI planning.

        Args:
            obs: (num_envs, obs_dim)
            t0_mask: (num_envs,) boolean, True = start of episode
            eval_mode: deterministic action selection
            task: (num_envs,) task indices or None

        Returns:
            actions: (num_envs, action_dim)
        """
        num_envs = obs.shape[0]
        horizon = self.cfg.horizon
        num_samples = self.cfg.num_samples
        num_pi_trajs = self.cfg.num_pi_trajs
        num_elites = self.cfg.num_elites
        action_dim = self.cfg.action_dim

        z_init = self.model.encode(obs, task)
        latent_dim = z_init.shape[-1]

        # === Policy trajectory rollouts ===
        if num_pi_trajs > 0:
            z_pi = z_init.unsqueeze(1).expand(-1, num_pi_trajs, -1).reshape(-1, latent_dim)

            pi_actions = torch.empty(
                horizon, num_envs * num_pi_trajs, action_dim, device=self.device
            )

            for t in range(horizon - 1):
                pi_actions[t], _ = self.model.pi(z_pi, task)
                z_pi = self.model.next(z_pi, pi_actions[t], task)
            pi_actions[-1], _ = self.model.pi(z_pi, task)

            pi_actions = pi_actions.reshape(horizon, num_envs, num_pi_trajs, action_dim)
            pi_actions = pi_actions.permute(1, 0, 2, 3)

        # === Initialize MPPI parameters ===
        # per-env mean and std: (num_envs, horizon, action_dim)
        mean = torch.zeros(num_envs, horizon, action_dim, device=self.device)
        std = torch.full(
            (num_envs, horizon, action_dim), self.cfg.max_std,
            dtype=torch.float, device=self.device
        )

        warm_mask = ~t0_mask
        if warm_mask.any():
            mean[warm_mask, :-1] = self._prev_mean_batched[warm_mask, 1:]

        z = z_init.unsqueeze(1).expand(-1, num_samples, -1).reshape(-1, latent_dim)
        actions = torch.empty(
            num_envs, horizon, num_samples, action_dim, device=self.device
        )

        if num_pi_trajs > 0:
            actions[:, :, :num_pi_trajs] = pi_actions

        # === MPPI iterations ===
        for _ in range(self.cfg.iterations):
            num_random = num_samples - num_pi_trajs
            r = torch.randn(
                num_envs, horizon, num_random, action_dim, device=self.device
            )
            actions_sample = mean.unsqueeze(2) + std.unsqueeze(2) * r
            actions_sample = actions_sample.clamp(-1, 1)
            actions[:, :, num_pi_trajs:] = actions_sample

            actions_flat = actions.permute(1, 0, 2, 3).reshape(horizon, -1, action_dim)
            value = self._estimate_value_batched(z, actions_flat, task)
            value = value.reshape(num_envs, num_samples)

            elite_idxs = torch.topk(value, num_elites, dim=1).indices
            elite_value = torch.gather(value, 1, elite_idxs)

            idx_expanded = elite_idxs.unsqueeze(1).unsqueeze(-1).expand(-1, horizon, -1, action_dim)
            elite_actions = torch.gather(actions, 2, idx_expanded)

            max_value = elite_value.max(dim=1, keepdim=True).values  # (num_envs, 1)
            score = torch.exp(self.cfg.temperature * (elite_value - max_value))  # (num_envs, num_elites)
            score = score / (score.sum(dim=1, keepdim=True) + 1e-9)  # Normalize

            # Weighted mean: (num_envs, horizon, action_dim)
            # score: (num_envs, num_elites) → (num_envs, 1, num_elites, 1)
            score_expanded = score.unsqueeze(1).unsqueeze(-1)
            mean = (score_expanded * elite_actions).sum(dim=2)

            # Weighted std
            std = ((score_expanded * (elite_actions - mean.unsqueeze(2)) ** 2).sum(dim=2)).sqrt()
            std = std.clamp(self.cfg.min_std, self.cfg.max_std)

        # === Select final actions ===
        # Sample from final elite distribution
        # score: (num_envs, num_elites)
        rand_idx = math.gumbel_softmax_sample(score, dim=-1)  # (num_envs,)

        idx_final = rand_idx.unsqueeze(1).unsqueeze(-1).expand(-1, horizon, action_dim)
        selected_actions = torch.gather(elite_actions, 2, idx_final.unsqueeze(2)).squeeze(2)

        a = selected_actions[:, 0]
        std_0 = std[:, 0]

        if not eval_mode:
            a = a + std_0 * torch.randn_like(a)

        self._prev_mean_batched.copy_(mean)

        return a.clamp(-1, 1)

    @torch.no_grad()
    def _estimate_value_batched(
        self,
        z: torch.Tensor,
        actions: torch.Tensor,
        task: torch.Tensor,
    ) -> torch.Tensor:
        """Estimate trajectory values for batched states.

        Args:
            z: Flattened latent states (num_envs * num_samples, latent_dim)
            actions: Action sequences (horizon, num_envs * num_samples, action_dim)
            task: Task indices or None

        Returns:
            values: (num_envs * num_samples, 1)
        """
        G = torch.zeros(z.shape[0], 1, device=z.device)
        discount = 1.0
        termination = torch.zeros(z.shape[0], 1, device=z.device)

        for t in range(self.cfg.horizon):
            reward = math.two_hot_inv(self.model.reward(z, actions[t], task), self.cfg)
            z = self.model.next(z, actions[t], task)
            G = G + discount * (1 - termination) * reward

            discount_update = self.discount[task] if self.cfg.multitask else self.discount
            discount = discount * discount_update

            if self.cfg.episodic:
                termination = torch.clip(
                    termination + (self.model.termination(z, task) > 0.5).float(),
                    max=1.0
                )

        action_pi, _ = self.model.pi(z, task)
        terminal_value = self.model.Q(z, action_pi, task, return_type='avg')
        G = G + discount * (1 - termination) * terminal_value

        return G.nan_to_num(0)


def create_batched_tdmpc2(cfg, num_envs: int):
    """Factory function to create a batched TDMPC2 agent.

    Args:
        cfg: TDMPC2 configuration.
        num_envs: Number of parallel environments.

    Returns:
        TDMPC2 agent with batched inference capabilities.
    """
    from tdmpc2 import TDMPC2

    class BatchedTDMPC2(BatchedTDMPC2Mixin, TDMPC2):
        pass

    agent = BatchedTDMPC2(cfg)
    agent.init_batched(num_envs)

    return agent
