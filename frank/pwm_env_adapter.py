"""Adapter to make RSLRLBraxWrapper compatible with PWM's interface."""

import torch
from gym import spaces


class PWMEnvAdapter:
    """Wraps RSLRLBraxWrapper for PWM compatibility.

    PWM expects:
    - observation_space.shape[0] for obs_dim
    - action_space.shape[0] for act_dim
    - episode_length attribute
    - reset() returning a tensor (not TensorDict)
    - info["termination"], info["truncation"], info["obs_before_reset"], info["primal"]
    """

    def __init__(self, brax_wrapper):
        self._env = brax_wrapper
        self.num_envs = brax_wrapper.num_envs
        self.episode_length = brax_wrapper.max_episode_length
        self._current_obs = None

        self.observation_space = spaces.Box(
            low=-float('inf'), high=float('inf'),
            shape=(brax_wrapper.num_obs,)
        )
        self.action_space = spaces.Box(
            low=-1.0, high=1.0,
            shape=(brax_wrapper.num_actions,)
        )

    def reset(self, grads=False):
        """Reset environment.

        Args:
            grads: If True, return current observation without resetting.
                   This is used by PWM to reinitialize gradients between epochs
                   without actually resetting the environment state.
                   If False, perform a full environment reset.

        Returns:
            Observation tensor of shape (num_envs, obs_dim).
        """
        if grads and self._current_obs is not None:
            return self._current_obs.clone()

        obs_dict = self._env.reset()
        self._current_obs = obs_dict["state"]
        return self._current_obs.clone()

    def step(self, actions):
        """Step environment with PWM-compatible info dict.

        Args:
            actions: Action tensor of shape (num_envs, act_dim).

        Returns:
            Tuple of (obs, reward, done, info) where info contains
            termination, truncation, obs_before_reset, and primal.
        """
        obs_dict, reward, done, info = self._env.step(actions.detach())
        obs = obs_dict["state"]

        self._current_obs = obs

        pwm_info = {
            "termination": info.get("termination", torch.zeros_like(done)).bool(),
            "truncation": info.get("time_outs", torch.zeros_like(done)).bool(),
            "obs_before_reset": info.get("obs_before_reset", obs),
            "primal": info.get("primal", reward),
        }

        return obs, reward, done, pwm_info
