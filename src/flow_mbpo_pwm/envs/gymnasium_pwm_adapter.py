"""Adapter for Gymnasium vector envs to match PWM's expected API."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch


def _to_tensor(value: Any, device: torch.device) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value.to(device=device)
    if isinstance(value, np.ndarray):
        return torch.from_numpy(value).to(device=device)
    return torch.as_tensor(value, device=device)


class GymnasiumPWMAdapter:
    """
    Wrap Gymnasium vector envs and expose PWM-compatible reset/step API.

    Returns tensors on `device`, while stepping envs on CPU.
    """

    def __init__(
        self,
        env: Any,
        *,
        device: str = "cuda",
        episode_length: int = 1000,
    ):
        self._env = env
        self.device = torch.device(device)
        self.episode_length = int(episode_length)

        self.num_envs = int(getattr(env, "num_envs", 1))
        self.observation_space = getattr(env, "single_observation_space", env.observation_space)
        self.action_space = getattr(env, "single_action_space", env.action_space)
        self.num_obs = int(np.prod(self.observation_space.shape))
        self.num_actions = int(np.prod(self.action_space.shape))

        self._current_obs: Optional[torch.Tensor] = None
        self._steps = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)

        self.reset(grads=False)

    def _flatten_obs(self, obs: Any) -> torch.Tensor:
        obs_t = _to_tensor(obs, self.device).float()
        if obs_t.ndim == 1:
            obs_t = obs_t.unsqueeze(0)
        return obs_t.reshape(obs_t.shape[0], -1)

    def _extract_final_obs(
        self,
        info: Dict[str, Any],
        done: torch.Tensor,
        pre_step_obs: torch.Tensor,
        obs: torch.Tensor,
    ) -> torch.Tensor:
        obs_before_reset = obs.clone()
        if done.sum().item() == 0:
            return obs_before_reset

        final_obs = info.get("final_observation", None)
        final_mask = info.get("_final_observation", None)

        done_idx = done.nonzero(as_tuple=False).squeeze(-1)
        if final_obs is None:
            obs_before_reset[done_idx] = pre_step_obs[done_idx]
            return obs_before_reset

        try:
            if isinstance(final_obs, np.ndarray) and final_obs.dtype != object:
                final_t = self._flatten_obs(final_obs)
                if final_mask is not None:
                    mask_t = _to_tensor(final_mask, self.device).bool().reshape(-1)
                    masked_idx = mask_t.nonzero(as_tuple=False).squeeze(-1)
                    obs_before_reset[masked_idx] = final_t[masked_idx]
                else:
                    obs_before_reset[done_idx] = final_t[done_idx]
                return obs_before_reset

            # object arrays / lists path
            for idx in done_idx.tolist():
                maybe_obs = final_obs[idx]
                if maybe_obs is None:
                    obs_before_reset[idx] = pre_step_obs[idx]
                else:
                    obs_before_reset[idx] = self._flatten_obs(maybe_obs)[0]
            return obs_before_reset
        except Exception:
            obs_before_reset[done_idx] = pre_step_obs[done_idx]
            return obs_before_reset

    def reset(self, grads: bool = False) -> torch.Tensor:
        if grads and self._current_obs is not None:
            return self._current_obs.clone()

        reset_out = self._env.reset()
        if isinstance(reset_out, tuple) and len(reset_out) == 2:
            obs, _ = reset_out
        else:
            obs = reset_out

        obs_t = self._flatten_obs(obs)
        self._current_obs = obs_t
        self._steps = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        return obs_t.clone()

    def step(self, actions: torch.Tensor):
        if self._current_obs is None:
            self.reset(grads=False)

        pre_step_obs = self._current_obs.clone()
        actions_cpu = actions.detach().to("cpu").numpy()

        step_out = self._env.step(actions_cpu)
        if len(step_out) == 5:
            obs, reward, terminated, truncated, info = step_out
        else:
            obs, reward, done, info = step_out
            terminated = done
            truncated = np.zeros_like(done)

        obs_t = self._flatten_obs(obs)
        reward_t = _to_tensor(reward, self.device).float().reshape(-1)
        terminated_t = _to_tensor(terminated, self.device).bool().reshape(-1)
        truncated_t = _to_tensor(truncated, self.device).bool().reshape(-1)

        self._steps += 1
        if self.episode_length > 0:
            truncated_t = truncated_t | (self._steps >= self.episode_length)

        done_t = terminated_t | truncated_t
        done_idx = done_t.nonzero(as_tuple=False).squeeze(-1)
        if done_idx.numel() > 0:
            self._steps[done_idx] = 0

        info = info if isinstance(info, dict) else {}
        obs_before_reset = self._extract_final_obs(info, done_t, pre_step_obs, obs_t)

        self._current_obs = obs_t
        pwm_info = {
            "termination": terminated_t,
            "truncation": truncated_t,
            "obs_before_reset": obs_before_reset,
            "primal": reward_t,
        }
        return obs_t, reward_t, done_t, pwm_info

    def render(self):
        # For vector envs this is mainly used in rollout eval with num_envs=1.
        call_fn = getattr(self._env, "call", None)
        if callable(call_fn):
            try:
                frames = call_fn("render")
                if isinstance(frames, (list, tuple)) and len(frames) > 0:
                    return frames[0]
                return frames
            except Exception:
                pass

        render_fn = getattr(self._env, "render", None)
        if callable(render_fn):
            try:
                return render_fn()
            except Exception:
                return None
        return None

    def close(self):
        close_fn = getattr(self._env, "close", None)
        if callable(close_fn):
            close_fn()


def _resolve_env_id(env_id: str, env_id_fallbacks: Optional[Sequence[str]] = None) -> str:
    import gymnasium as gym

    candidates: List[str] = [env_id]
    if env_id_fallbacks:
        for candidate in env_id_fallbacks:
            if candidate and candidate not in candidates:
                candidates.append(candidate)

    last_err: Optional[Exception] = None
    for candidate in candidates:
        try:
            gym.spec(candidate)
            return candidate
        except Exception as exc:
            last_err = exc
            continue

    raise RuntimeError(f"Could not resolve Gymnasium env id from {candidates}. Last error: {last_err}")


def create_gymnasium_mujoco_pwm_env(
    env_id: str,
    env_id_fallbacks: Optional[Sequence[str]] = None,
    num_envs: int = 16,
    device: str = "cuda",
    seed: int = 42,
    episode_length: int = 1000,
    render_mode: Optional[str] = None,
    env_kwargs: Optional[Dict[str, Any]] = None,
    no_grad: bool = True,
    logdir: Optional[str] = None,
    **kwargs,
) -> GymnasiumPWMAdapter:
    del no_grad, logdir, kwargs

    import gymnasium as gym

    resolved_env_id = _resolve_env_id(env_id=env_id, env_id_fallbacks=env_id_fallbacks)
    env_kwargs = dict(env_kwargs or {})

    def make_env_fn(idx: int):
        def _thunk():
            env = gym.make(resolved_env_id, render_mode=render_mode, **env_kwargs)
            env.reset(seed=int(seed) + idx)
            return env

        return _thunk

    env_fns = [make_env_fn(i) for i in range(int(num_envs))]
    vec_env = gym.vector.SyncVectorEnv(env_fns)
    adapter = GymnasiumPWMAdapter(vec_env, device=device, episode_length=episode_length)
    print(
        "[GymnasiumPWMAdapter] initialized "
        f"(env_id={resolved_env_id}, requested_env_id={env_id}, "
        f"num_envs={adapter.num_envs}, obs_dim={adapter.num_obs}, act_dim={adapter.num_actions})"
    )
    return adapter
