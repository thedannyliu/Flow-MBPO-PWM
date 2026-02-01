"""Batched episode tracking for GPU environments with auto-reset.
"""

import torch
from tensordict import TensorDict
from typing import List, Optional


class BatchedEpisodeTracker:
    """Tracks episodes across a batched environment"""

    def __init__(
        self,
        num_envs: int,
        obs_dim: int,
        act_dim: int,
        device: str = "cuda",
    ):
        self.num_envs = num_envs
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.device = device

        self.episodes: List[List[TensorDict]] = [[] for _ in range(num_envs)]
        self._completed_episodes: List[TensorDict] = []
        self._episode_steps = torch.zeros(num_envs, dtype=torch.long, device=device)

    def init_episodes(self, obs: torch.Tensor):
        """Initialize with first observations after reset.

        Args:
            obs: (num_envs, obs_dim) initial observations
        """
        self._episode_steps.zero_()

        nan_action = torch.full((self.act_dim,), float('nan'), device=self.device)
        nan_scalar = torch.tensor(float('nan'), device=self.device)

        for i in range(self.num_envs):
            self.episodes[i] = [self._make_td(
                obs[i], nan_action, nan_scalar, nan_scalar
            )]

    def add_step(
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
        reward: torch.Tensor,
        done: torch.Tensor,
        terminated: torch.Tensor,
    ):
        """Add batched step, handling auto-reset observation ambiguity.

        Args:
            obs: (num_envs, obs_dim)
            action: (num_envs, act_dim)
            reward: (num_envs,)
            done: (num_envs,) - episode ended (termination OR truncation)
            terminated: (num_envs,) - true termination (not truncation)
        """
        for i in range(self.num_envs):
            self.episodes[i].append(self._make_td(
                obs[i], action[i], reward[i], terminated[i]
            ))

            if done[i]:
                episode = torch.cat(self.episodes[i])
                self._completed_episodes.append(episode)

                # start new episode with reset obs (which is obs[i])
                nan_action = torch.full((self.act_dim,), float('nan'), device=self.device)
                nan_scalar = torch.tensor(float('nan'), device=self.device)
                self.episodes[i] = [self._make_td(
                    obs[i], nan_action, nan_scalar, nan_scalar
                )]
                self._episode_steps[i] = 0
            else:
                self._episode_steps[i] += 1

    def get_t0_mask(self) -> torch.Tensor:
        """Returns boolean mask indicating which envs are at episode start (t0=True)."""
        return self._episode_steps == 0

    def get_completed_episodes(self) -> List[TensorDict]:
        """Retrieve and clear completed episodes."""
        episodes = self._completed_episodes
        self._completed_episodes = []
        return episodes

    def _make_td(
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
        reward: torch.Tensor,
        terminated: torch.Tensor,
    ) -> TensorDict:
        """Create TensorDict for single transition."""
        return TensorDict({
            "obs": obs.unsqueeze(0).cpu(),
            "action": action.unsqueeze(0).cpu(),
            "reward": reward.unsqueeze(0).cpu(),
            "terminated": terminated.unsqueeze(0).cpu(),
        }, batch_size=(1,))
