"""Adapter to make TDMPC2's DMControl environment compatible with PWM's interface.

"""

import os
os.environ["MUJOCO_GL"] = "egl"

import torch
import numpy as np
from gym import spaces
from dm_control import suite
from dm_control.suite.wrappers import action_scale


class DMControlPWMAdapter:
    """Wraps a single DMControl environment for PWM compatibility.

    PWM expects:
    - num_envs: number of parallel environments (=1 for DMControl)
    - observation_space.shape[0] for obs_dim
    - action_space.shape[0] for act_dim
    - episode_length attribute
    - reset(grads=True) returning a tensor
    - step(actions) returning (obs, reward, done, info) where info contains
      termination, truncation, obs_before_reset, primal
    """

    def __init__(
        self,
        task: str,
        episode_length: int = 500,
        action_repeat: int = 2,
        seed: int = 42,
        device: str = "cuda",
    ):
        """Initialize DMControl environment.

        Args:
            task: Task name in format "domain-task" (e.g., "walker-stand").
            episode_length: Maximum episode length (after action repeat).
            action_repeat: Number of physics steps per action.
            seed: Random seed.
            device: Device for output tensors.
        """
        self.num_envs = 1
        self.episode_length = episode_length
        self.action_repeat = action_repeat
        self.device = torch.device(device)

        domain, task_name = task.replace('-', '_').split('_', 1)
        domain = dict(cup='ball_in_cup', pointmass='point_mass').get(domain, domain)

        self._env = suite.load(
            domain,
            task_name,
            task_kwargs={'random': seed},
            visualize_reward=False,
        )
        self._env = action_scale.Wrapper(self._env, minimum=-1., maximum=1.)

        obs_spec = self._env.observation_spec()
        action_spec = self._env.action_spec()

        obs_dim = sum(np.prod(v.shape) if v.shape else 1 for v in obs_spec.values())

        self.observation_space = spaces.Box(
            low=-float('inf'), high=float('inf'),
            shape=(int(obs_dim),)
        )
        self.action_space = spaces.Box(
            low=-1.0, high=1.0,
            shape=action_spec.shape
        )

        self._step_count = 0
        self._current_obs = None

    def _obs_to_tensor(self, obs_dict) -> torch.Tensor:
        """Convert dm_control observation dict to flat tensor."""
        arrays = []
        for v in obs_dict.values():
            arr = np.asarray(v, dtype=np.float32).flatten()
            arrays.append(arr)
        return torch.from_numpy(np.concatenate(arrays))

    def reset(self, grads: bool = False) -> torch.Tensor:
        """Reset environment.

        Args:
            grads: If True, return current observation without resetting (for gradient
                   reinitialization in differentiable physics). If False, do full reset.

        Returns:
            Observation tensor of shape (1, obs_dim).
        """
        if grads and self._current_obs is not None:
            return self._current_obs.clone()

        timestep = self._env.reset()
        self._step_count = 0
        obs = self._obs_to_tensor(timestep.observation)
        self._current_obs = obs.unsqueeze(0).to(self.device)
        return self._current_obs.clone()

    def step(self, actions: torch.Tensor):
        """Step environment.

        Args:
            actions: Action tensor of shape (1, act_dim).

        Returns:
            Tuple of (obs, reward, done, info) where:
            - obs: Observation tensor (1, obs_dim)
            - reward: Reward tensor (1,)
            - done: Done flags tensor (1,)
            - info: Dict with termination, truncation, obs_before_reset, primal
        """
        action = actions[0].detach().cpu().numpy()

        reward = 0.0
        for _ in range(self.action_repeat):
            timestep = self._env.step(action)
            reward += timestep.reward or 0.0

        obs = self._obs_to_tensor(timestep.observation)
        self._step_count += 1

        terminated = timestep.last() and timestep.discount == 0.0
        truncated = self._step_count >= self.episode_length
        done = terminated or truncated

        obs_before_reset = obs.clone()

        if done:
            timestep = self._env.reset()
            obs = self._obs_to_tensor(timestep.observation)
            self._step_count = 0

        obs = obs.unsqueeze(0).to(self.device)
        self._current_obs = obs
        reward_t = torch.tensor([reward], dtype=torch.float32, device=self.device)
        done_t = torch.tensor([done], dtype=torch.bool, device=self.device)

        info = {
            "termination": torch.tensor([terminated], dtype=torch.bool, device=self.device),
            "truncation": torch.tensor([truncated], dtype=torch.bool, device=self.device),
            "obs_before_reset": obs_before_reset.unsqueeze(0).to(self.device),
            "primal": torch.tensor([reward], dtype=torch.float32, device=self.device),
        }

        return obs, reward_t, done_t, info

    def close(self):
        """Close environment."""
        try:
            self._env.close()
        except Exception:
            pass


def create_dmcontrol_pwm_env(
    task: str = "walker-stand",
    episode_length: int = 500,
    action_repeat: int = 2,
    seed: int = 42,
    device: str = "cuda",
) -> DMControlPWMAdapter:
    """Factory function to create PWM-compatible DMControl environment.

    Args:
        task: Task name in format "domain-task" (e.g., "walker-stand").
        episode_length: Maximum episode length.
        action_repeat: Number of physics steps per action.
        seed: Random seed.
        device: Device for output tensors.

    Returns:
        DMControlPWMAdapter instance.
    """
    return DMControlPWMAdapter(
        task=task,
        episode_length=episode_length,
        action_repeat=action_repeat,
        seed=seed,
        device=device,
    )
