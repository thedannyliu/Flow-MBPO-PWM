"""Adapter that makes mjlab environments compatible with PWM's expected API."""

from __future__ import annotations

import importlib
import inspect
import os
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

try:
    from gym import spaces
except Exception:  # pragma: no cover - gymnasium fallback for newer envs
    try:
        from gymnasium import spaces
    except Exception:  # pragma: no cover - minimal fallback for lightweight smoke tests
        class _FallbackBox:
            def __init__(self, low, high, shape, dtype):
                self.low = low
                self.high = high
                self.shape = shape
                self.dtype = dtype

        class _FallbackSpaces:
            Box = _FallbackBox

        spaces = _FallbackSpaces()


_TERMINAL_OBS_KEYS = (
    "obs_before_reset",
    "final_observation",
    "terminal_observation",
    "last_observation",
    "pre_reset_observation",
)

_DEFAULT_OBS_KEYS = ("state", "policy", "observation", "obs")
_DEFAULT_PRIMAL_KEYS = ("primal", "reward_unscaled", "raw_reward")


def _patch_mujoco_compatibility() -> None:
    """Patch minor MuJoCo enum drift used by some MJLab task configs.

    MJLab task files may reference `mujoco.mjtEnableBit.mjENBL_MULTICCD`.
    MuJoCo 3.8 exposes CCD toggles as disable bits instead, so the enable-bit
    name is absent. Treating the missing enable flag as zero preserves the
    default simulator behavior and lets the same MJLab configs load across
    MuJoCo minor versions.
    """
    try:
        import mujoco  # type: ignore

        enable_bits = getattr(mujoco, "mjtEnableBit", None)
        if enable_bits is not None and not hasattr(enable_bits, "mjENBL_MULTICCD"):
            setattr(enable_bits, "mjENBL_MULTICCD", 0)
    except Exception:
        # Environment construction will surface any real MuJoCo/MJLab error.
        return


@dataclass
class AdapterDiagnostics:
    """Lightweight diagnostics for debugging done/reset semantics."""

    step_calls: int = 0
    done_events: int = 0
    done_terminated: int = 0
    done_truncated: int = 0
    terminal_obs_from_info: int = 0
    terminal_obs_from_fallback: int = 0
    terminal_obs_equal_next_obs: int = 0
    missing_terminal_obs_warnings: int = 0


def _to_tensor(value: Any, device: torch.device) -> torch.Tensor:
    """Convert value to tensor on the target device without changing numeric meaning."""
    if isinstance(value, torch.Tensor):
        return value.to(device=device)
    if isinstance(value, np.ndarray):
        return torch.from_numpy(value).to(device=device)
    return torch.as_tensor(value, device=device)


def _is_dict_like(value: Any) -> bool:
    return isinstance(value, dict)


class MJLabPWMAdapter:
    """
    Wrap a Gymnasium-style vectorized mjlab env so PWM can consume it directly.

    PWM expects:
    - `reset(grads=True/False)`
    - `step(actions) -> (obs, reward, done, info)`
    - info keys: termination, truncation, obs_before_reset, primal
    """

    def __init__(
        self,
        mjlab_env: Any,
        *,
        device: str = "cuda",
        episode_length: int = 1000,
        obs_key: Optional[str] = "state",
        obs_key_candidates: Optional[Sequence[str]] = None,
        action_dim: Optional[int] = None,
        strict_terminal_obs: bool = True,
        expect_auto_reset: bool = True,
        fail_on_missing_terminal_obs: bool = False,
        warn_missing_terminal_obs_every: int = 100,
    ):
        self._env = mjlab_env
        self.device = torch.device(device)
        self.episode_length = int(episode_length)
        self.obs_key = obs_key
        self.obs_key_candidates = tuple(obs_key_candidates or _DEFAULT_OBS_KEYS)
        self.strict_terminal_obs = bool(strict_terminal_obs)
        self.expect_auto_reset = bool(expect_auto_reset)
        self.fail_on_missing_terminal_obs = bool(fail_on_missing_terminal_obs)
        self.warn_missing_terminal_obs_every = max(1, int(warn_missing_terminal_obs_every))

        self._current_obs: Optional[torch.Tensor] = None
        self._episode_steps: Optional[torch.Tensor] = None
        self._diagnostics = AdapterDiagnostics()

        # Initialize dimensions by doing one reset once.
        obs = self.reset(grads=False)
        self.num_envs = int(obs.shape[0])
        self.num_obs = int(obs.shape[1])
        self.num_actions = self._resolve_num_actions(action_dim=action_dim)

        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.num_obs,),
            dtype=np.float32,
        )
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(self.num_actions,),
            dtype=np.float32,
        )

        if self._episode_steps is None or self._episode_steps.shape[0] != self.num_envs:
            self._episode_steps = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)

    def _resolve_num_actions(self, action_dim: Optional[int] = None) -> int:
        if action_dim is not None:
            return int(action_dim)
        if hasattr(self._env, "num_actions"):
            return int(self._env.num_actions)
        if hasattr(self._env, "action_space"):
            shape = tuple(getattr(self._env.action_space, "shape", ()) or ())
            if len(shape) == 0:
                return 1
            return int(shape[-1])
        raise ValueError(
            "Unable to infer action dimension from mjlab env. "
            "Pass `action_dim` explicitly in create_mjlab_pwm_env(...)."
        )

    def _get_obs_from_dict(self, obs_dict: Dict[str, Any]) -> Any:
        if self.obs_key and self.obs_key in obs_dict:
            return obs_dict[self.obs_key]
        for key in self.obs_key_candidates:
            if key in obs_dict:
                return obs_dict[key]

        # Fallback: deterministic concatenation to keep behavior stable.
        keys = sorted(obs_dict.keys())
        parts: List[torch.Tensor] = []
        for key in keys:
            tensor = _to_tensor(obs_dict[key], self.device).float()
            if tensor.ndim == 1:
                tensor = tensor.unsqueeze(0)
            parts.append(tensor.reshape(tensor.shape[0], -1))
        if not parts:
            raise ValueError("Observation dict is empty; cannot extract policy observation.")
        return torch.cat(parts, dim=-1)

    def _flatten_obs(self, obs_like: Any) -> torch.Tensor:
        if _is_dict_like(obs_like):
            obs_like = self._get_obs_from_dict(obs_like)

        if isinstance(obs_like, (list, tuple)):
            # Supports list-of-env observations in per-env format.
            stacked: List[torch.Tensor] = []
            for item in obs_like:
                t = self._flatten_obs(item)
                if t.ndim == 2 and t.shape[0] == 1:
                    t = t.squeeze(0)
                stacked.append(t.reshape(-1))
            return torch.stack(stacked, dim=0).to(self.device).float()

        obs = _to_tensor(obs_like, self.device).float()
        if obs.ndim == 0:
            obs = obs.view(1, 1)
        elif obs.ndim == 1:
            obs = obs.unsqueeze(0)
        else:
            obs = obs.reshape(obs.shape[0], -1)
        return obs

    def _unpack_reset(self, reset_out: Any) -> Tuple[Any, Dict[str, Any]]:
        if isinstance(reset_out, tuple) and len(reset_out) == 2:
            obs_like, info = reset_out
            info = info if isinstance(info, dict) else {}
            return obs_like, info
        return reset_out, {}

    def _unpack_step(
        self, step_out: Any
    ) -> Tuple[Any, torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, Any]]:
        if not isinstance(step_out, tuple):
            raise TypeError(f"mjlab env.step(...) must return tuple, got {type(step_out)!r}")

        if len(step_out) == 5:
            obs_like, reward, terminated, truncated, info = step_out
        elif len(step_out) == 4:
            obs_like, reward, done, info = step_out
            done_t = _to_tensor(done, self.device).bool()
            terminated = done_t
            truncated = torch.zeros_like(done_t)
        else:
            raise ValueError(
                "Unsupported env.step return format. Expected 4 or 5 values, "
                f"but got {len(step_out)}."
            )

        reward_t = _to_tensor(reward, self.device).float().reshape(-1)
        terminated_t = _to_tensor(terminated, self.device).bool().reshape(-1)
        truncated_t = _to_tensor(truncated, self.device).bool().reshape(-1)
        info_d = info if isinstance(info, dict) else {}
        return obs_like, reward_t, terminated_t, truncated_t, info_d

    def _extract_terminal_obs_tensor(self, info: Dict[str, Any]) -> Optional[torch.Tensor]:
        for key in _TERMINAL_OBS_KEYS:
            if key not in info or info[key] is None:
                continue
            try:
                terminal_obs = self._flatten_obs(info[key])
                if (
                    self._current_obs is not None
                    and terminal_obs.shape[0] == self._current_obs.shape[0]
                    and terminal_obs.shape[1] == self._current_obs.shape[1]
                ):
                    return terminal_obs
            except Exception:
                continue
        return None

    def _get_primal(self, info: Dict[str, Any], reward: torch.Tensor) -> torch.Tensor:
        for key in _DEFAULT_PRIMAL_KEYS:
            if key in info and info[key] is not None:
                primal = _to_tensor(info[key], self.device).float().reshape(-1)
                if primal.shape == reward.shape:
                    return primal
        return reward

    def _warn_missing_terminal_obs(self, done_count: int):
        self._diagnostics.terminal_obs_from_fallback += done_count
        should_warn = (
            self._diagnostics.terminal_obs_from_fallback % self.warn_missing_terminal_obs_every
            == 0
        )
        if should_warn:
            self._diagnostics.missing_terminal_obs_warnings += 1
            print(
                "[MJLabPWMAdapter] warning: missing terminal observation in info. "
                "Falling back to pre-step observation for done transitions."
            )

    def reset(self, grads: bool = False) -> torch.Tensor:
        """
        Reset environment.

        If `grads=True`, return cached observation without touching env state.
        """
        if grads and self._current_obs is not None:
            return self._current_obs.clone()

        reset_out = self._env.reset()
        obs_like, _ = self._unpack_reset(reset_out)
        obs = self._flatten_obs(obs_like)
        self._current_obs = obs
        self._episode_steps = torch.zeros(obs.shape[0], dtype=torch.long, device=self.device)
        return obs.clone()

    def step(self, actions: torch.Tensor):
        """
        Step environment and return PWM-compatible fields.

        Returns:
            obs, reward, done, info where info includes:
            - termination
            - truncation
            - obs_before_reset
            - primal
        """
        if self._current_obs is None:
            _ = self.reset(grads=False)

        pre_step_obs = self._current_obs.clone()
        step_out = self._env.step(actions.detach())
        obs_like, reward, terminated, truncated, info = self._unpack_step(step_out)

        obs = self._flatten_obs(obs_like)
        if obs.shape[0] != reward.shape[0]:
            raise ValueError(
                "Batch size mismatch from env.step: "
                f"obs batch={obs.shape[0]}, reward batch={reward.shape[0]}"
            )

        if self._episode_steps is None or self._episode_steps.shape[0] != obs.shape[0]:
            self._episode_steps = torch.zeros(obs.shape[0], dtype=torch.long, device=self.device)
        self._episode_steps += 1

        # If truncated isn't provided correctly, infer using episode length.
        if truncated.numel() == 0:
            truncated = torch.zeros_like(terminated)
        if self.episode_length > 0:
            inferred_trunc = self._episode_steps >= self.episode_length
            truncated = truncated | inferred_trunc

        done = terminated | truncated
        done_count = int(done.sum().item())
        done_indices = done.nonzero(as_tuple=False).squeeze(-1)

        obs_before_reset = obs.clone()
        terminal_obs = self._extract_terminal_obs_tensor(info)
        if terminal_obs is not None and done_count > 0:
            obs_before_reset[done_indices] = terminal_obs[done_indices]
            self._diagnostics.terminal_obs_from_info += done_count
        elif done_count > 0:
            if self.fail_on_missing_terminal_obs:
                raise RuntimeError(
                    "Done transitions observed, but env info does not contain terminal "
                    "observations (e.g., final_observation/obs_before_reset)."
                )
            obs_before_reset[done_indices] = pre_step_obs[done_indices]
            self._warn_missing_terminal_obs(done_count)

        if self.expect_auto_reset and done_count > 0:
            equal_mask = torch.all(
                torch.isclose(
                    obs_before_reset[done_indices],
                    obs[done_indices],
                    atol=1e-6,
                    rtol=1e-5,
                ),
                dim=-1,
            )
            equal_count = int(equal_mask.sum().item())
            self._diagnostics.terminal_obs_equal_next_obs += equal_count
            if self.strict_terminal_obs and equal_count > 0:
                raise AssertionError(
                    f"Found {equal_count} done transitions where obs_before_reset equals "
                    "next obs under auto-reset assumption. This risks replay contamination."
                )

        # Reset per-env episode counters for done envs.
        if done_count > 0:
            self._episode_steps[done_indices] = 0

        self._current_obs = obs
        self._diagnostics.step_calls += 1
        self._diagnostics.done_events += done_count
        self._diagnostics.done_terminated += int(terminated.sum().item())
        self._diagnostics.done_truncated += int(truncated.sum().item())

        pwm_info = {
            "termination": terminated,
            "truncation": truncated,
            "obs_before_reset": obs_before_reset,
            "primal": self._get_primal(info, reward),
        }
        return obs, reward, done, pwm_info

    def get_diagnostics(self, reset: bool = False) -> Dict[str, float]:
        """Return adapter diagnostics in scalar form for logging/monitoring."""
        stats = asdict(self._diagnostics)
        done_events = max(1, stats["done_events"])
        stats["terminal_obs_info_ratio"] = stats["terminal_obs_from_info"] / done_events
        stats["terminal_obs_fallback_ratio"] = stats["terminal_obs_from_fallback"] / done_events
        stats["terminal_obs_equal_next_obs_ratio"] = (
            stats["terminal_obs_equal_next_obs"] / done_events
        )

        if reset:
            self._diagnostics = AdapterDiagnostics()
        return {k: float(v) for k, v in stats.items()}

    def close(self):
        close_fn = getattr(self._env, "close", None)
        if callable(close_fn):
            close_fn()

    def render(self, *args, **kwargs):
        """
        Forward render calls to the wrapped mjlab env when available.

        This is needed for evaluation-time rollout video capture.
        """
        render_fn = getattr(self._env, "render", None)
        if not callable(render_fn):
            return None
        try:
            return render_fn(*args, **kwargs)
        except TypeError:
            # Some envs expose render() without kwargs such as mode.
            try:
                return render_fn()
            except Exception:
                return None
        except Exception:
            return None


def _filter_kwargs_for_callable(fn: Any, kwargs: Dict[str, Any]) -> Dict[str, Any]:
    sig = inspect.signature(fn)
    has_var_kwargs = any(
        p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()
    )
    if has_var_kwargs:
        return kwargs
    return {k: v for k, v in kwargs.items() if k in sig.parameters}


def _resolve_mjlab_constructor():
    attempts = (
        ("mjlab", "make"),
        ("mjlab.envs", "make"),
        ("mjlab", "create_env"),
        ("mjlab.envs", "create_env"),
    )
    errors: List[str] = []
    for module_name, attr_name in attempts:
        try:
            module = importlib.import_module(module_name)
        except Exception as err:
            errors.append(f"{module_name}: {err}")
            continue
        if hasattr(module, attr_name):
            return getattr(module, attr_name), module_name, attr_name
    raise ImportError(
        "Could not find an mjlab environment factory. Tried: "
        + ", ".join(f"{m}.{a}" for m, a in attempts)
        + ". Import errors: "
        + "; ".join(errors)
    )


def _build_mjlab_env_from_registry(
    task_id: str,
    num_envs: int,
    device: str,
    seed: int,
    episode_length: int,
    render_mode: Optional[str] = None,
    motion_file: Optional[str] = None,
):
    """
    Build mjlab env via task registry (newer mjlab API).

    This path is used when `mjlab.make(...)`-style constructors are unavailable.
    """
    import mjlab.tasks  # noqa: F401 - ensures task registry is populated
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.tasks.registry import load_env_cfg

    env_cfg = load_env_cfg(task_id)
    # Match vectorization and seed to PWM/Hydra inputs.
    if hasattr(env_cfg, "scene") and hasattr(env_cfg.scene, "num_envs"):
        env_cfg.scene.num_envs = int(num_envs)
    if hasattr(env_cfg, "seed"):
        env_cfg.seed = int(seed)

    # Convert step-based episode length into seconds if config supports it.
    if (
        episode_length is not None
        and episode_length > 0
        and hasattr(env_cfg, "sim")
        and hasattr(env_cfg.sim, "mujoco")
        and hasattr(env_cfg.sim.mujoco, "timestep")
        and hasattr(env_cfg, "decimation")
        and hasattr(env_cfg, "episode_length_s")
    ):
        env_dt = float(env_cfg.sim.mujoco.timestep) * float(env_cfg.decimation)
        env_cfg.episode_length_s = float(episode_length) * env_dt

    commands = getattr(env_cfg, "commands", None)
    motion_cmd = commands.get("motion") if isinstance(commands, dict) else None
    if motion_cmd is not None and hasattr(motion_cmd, "motion_file"):
        motion_file_to_use = (motion_file or "").strip()
        if not motion_file_to_use:
            motion_file_to_use = os.environ.get("MJLAB_MOTION_FILE", "").strip()
        if motion_file_to_use and Path(motion_file_to_use).is_file():
            motion_cmd.motion_file = motion_file_to_use
            print(f"[MJLabPWMAdapter] using motion file override: {motion_file_to_use}")
        elif str(getattr(motion_cmd, "motion_file", "")).strip() == "":
            try:
                from mjlab.scripts.gcs import ensure_default_motion

                resolved_motion_file = ensure_default_motion()
                motion_cmd.motion_file = resolved_motion_file
                print(
                    "[MJLabPWMAdapter] resolved tracking motion file via "
                    f"mjlab default asset: {resolved_motion_file}"
                )
            except Exception as exc:
                raise RuntimeError(
                    "Tracking task requires a valid motion file. "
                    "Set env.config.mjlab_env_kwargs.motion_file or MJLAB_MOTION_FILE."
                ) from exc

    return ManagerBasedRlEnv(cfg=env_cfg, device=device, render_mode=render_mode)


def _build_mjlab_env(
    constructor: Any,
    task_id: str,
    base_kwargs: Dict[str, Any],
) -> Any:
    candidate_task_fields = ("task_id", "task", "name", "id")
    last_error: Optional[Exception] = None
    for task_field in candidate_task_fields:
        kwargs = dict(base_kwargs)
        kwargs[task_field] = task_id
        filtered = _filter_kwargs_for_callable(constructor, kwargs)
        try:
            return constructor(**filtered)
        except Exception as err:  # pragma: no cover - depends on external mjlab API
            last_error = err
            continue

    # Positional task fallback.
    filtered = _filter_kwargs_for_callable(constructor, base_kwargs)
    try:
        return constructor(task_id, **filtered)
    except Exception as err:  # pragma: no cover - depends on external mjlab API
        last_error = err

    raise RuntimeError(
        f"Unable to construct mjlab env for task '{task_id}'. "
        f"Last error: {last_error}"
    )


def create_mjlab_pwm_env(
    task_id: str,
    task_id_fallbacks: Optional[Sequence[str]] = None,
    strict_task_id_match: bool = False,
    num_envs: int = 64,
    device: str = "cuda",
    episode_length: int = 500,
    action_repeat: int = 1,
    seed: int = 42,
    no_grad: bool = True,
    obs_key: str = "state",
    obs_key_candidates: Optional[Sequence[str]] = None,
    strict_terminal_obs: bool = True,
    expect_auto_reset: bool = True,
    fail_on_missing_terminal_obs: bool = False,
    warn_missing_terminal_obs_every: int = 100,
    action_dim: Optional[int] = None,
    mjlab_env_kwargs: Optional[Dict[str, Any]] = None,
    logdir: Optional[str] = None,
    **kwargs,
) -> MJLabPWMAdapter:
    """
    Factory used by Hydra to create PWM-compatible mjlab environments.

    Note:
    - `no_grad` and `logdir` are accepted for Hydra compatibility.
    - additional kwargs are forwarded to the underlying mjlab constructor.
    """
    del no_grad, logdir
    _patch_mujoco_compatibility()

    ctor_kwargs: Dict[str, Any] = {
        "num_envs": num_envs,
        "device": device,
        "seed": seed,
        "episode_length": episode_length,
        "action_repeat": action_repeat,
    }
    if mjlab_env_kwargs:
        ctor_kwargs.update(mjlab_env_kwargs)
    if kwargs:
        ctor_kwargs.update(kwargs)

    candidate_task_ids: List[str] = [task_id]
    if task_id_fallbacks:
        candidate_task_ids.extend([tid for tid in task_id_fallbacks if tid and tid not in candidate_task_ids])

    module_name = "mjlab.tasks.registry"
    attr_name = "load_env_cfg+ManagerBasedRlEnv"
    env = None
    used_task_id = task_id
    last_error: Optional[Exception] = None
    candidate_errors: Dict[str, str] = {}

    try:
        constructor, module_name, attr_name = _resolve_mjlab_constructor()
        for candidate_task_id in candidate_task_ids:
            try:
                env = _build_mjlab_env(
                    constructor=constructor,
                    task_id=candidate_task_id,
                    base_kwargs=ctor_kwargs,
                )
                used_task_id = candidate_task_id
                break
            except Exception as err:
                last_error = err
                candidate_errors[candidate_task_id] = repr(err)
                continue
    except ImportError:
        for candidate_task_id in candidate_task_ids:
            try:
                env = _build_mjlab_env_from_registry(
                    task_id=candidate_task_id,
                    num_envs=num_envs,
                    device=device,
                    seed=seed,
                    episode_length=episode_length,
                    render_mode=ctor_kwargs.get("render_mode"),
                    motion_file=ctor_kwargs.get("motion_file"),
                )
                used_task_id = candidate_task_id
                break
            except Exception as err:
                last_error = err
                candidate_errors[candidate_task_id] = repr(err)
                continue

    if env is None:
        raise RuntimeError(
            f"Failed to create mjlab env for task candidates {candidate_task_ids}. "
            f"Last error: {last_error}"
        )

    if strict_task_id_match and used_task_id != task_id:
        raise RuntimeError(
            "MJLab task resolution fell back to a different task. "
            f"requested_task_id={task_id}, resolved_task_id={used_task_id}. "
            "Set strict_task_id_match=false only if this fallback is intentional."
        )

    # If constructor returns wrapped env tuple-like, keep only env object.
    if isinstance(env, tuple) and len(env) > 0:
        env = env[0]

    adapter = MJLabPWMAdapter(
        env,
        device=device,
        episode_length=episode_length,
        obs_key=obs_key,
        obs_key_candidates=obs_key_candidates,
        action_dim=action_dim,
        strict_terminal_obs=strict_terminal_obs,
        expect_auto_reset=expect_auto_reset,
        fail_on_missing_terminal_obs=fail_on_missing_terminal_obs,
        warn_missing_terminal_obs_every=warn_missing_terminal_obs_every,
    )
    adapter.requested_task_id = task_id
    adapter.resolved_task_id = used_task_id
    adapter.task_id_candidates = tuple(candidate_task_ids)
    adapter.task_id_resolution_errors = dict(candidate_errors)

    if used_task_id != task_id:
        print(
            "[MJLabPWMAdapter] task fallback engaged "
            f"(requested_task_id={task_id}, resolved_task_id={used_task_id}, "
            f"candidate_errors={candidate_errors})"
        )
    print(
        f"[MJLabPWMAdapter] initialized via {module_name}.{attr_name} "
        f"(task_id={used_task_id}, requested_task_id={task_id}, num_envs={adapter.num_envs}, "
        f"obs_dim={adapter.num_obs}, act_dim={adapter.num_actions})"
    )
    return adapter
