#!/usr/bin/env python3
"""Smoke test for MJLabPWMAdapter done/reset semantics."""

import os
import sys
import torch

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(os.path.dirname(_THIS_DIR))
sys.path.insert(0, os.path.join(_REPO_ROOT, "src"))

from flow_mbpo_pwm.envs.mjlab_pwm_adapter import MJLabPWMAdapter


class _DummyMJLabEnv:
    def __init__(self, include_terminal_obs: bool):
        self.include_terminal_obs = include_terminal_obs
        self.num_envs = 2
        self.num_actions = 3
        self._obs = torch.tensor(
            [[1.0, 0.0, 0.5, -1.0], [2.0, 0.0, 1.5, -2.0]], dtype=torch.float32
        )
        self._step = 0

    def reset(self):
        self._step = 0
        self._obs = torch.tensor(
            [[1.0, 0.0, 0.5, -1.0], [2.0, 0.0, 1.5, -2.0]], dtype=torch.float32
        )
        return {"state": self._obs.clone()}, {}

    def step(self, actions):
        del actions
        self._step += 1
        next_obs = self._obs + 0.25

        # Force done on env-0 to emulate auto-reset behavior.
        terminated = torch.tensor([self._step >= 1, False], dtype=torch.bool)
        truncated = torch.zeros_like(terminated)
        done = terminated | truncated

        final_obs = next_obs.clone()

        # Auto-reset env-0 observation in returned next_obs.
        if done[0]:
            next_obs[0] = torch.tensor([10.0, 10.0, 10.0, 10.0])

        self._obs = next_obs.clone()
        reward = torch.tensor([1.0, 0.5], dtype=torch.float32)
        info = {"primal": reward.clone()}
        if self.include_terminal_obs:
            info["final_observation"] = {"state": final_obs}
        return {"state": next_obs}, reward, terminated, truncated, info


def _assert_done_semantics(include_terminal_obs: bool):
    env = _DummyMJLabEnv(include_terminal_obs=include_terminal_obs)
    adapter = MJLabPWMAdapter(
        env,
        device="cpu",
        episode_length=10,
        obs_key="state",
        strict_terminal_obs=True,
        expect_auto_reset=True,
        fail_on_missing_terminal_obs=False,
    )

    _ = adapter.reset()
    obs, rew, done, info = adapter.step(torch.zeros((2, 3), dtype=torch.float32))
    del rew

    done_idx = done.nonzero(as_tuple=False).squeeze(-1)
    assert done_idx.numel() == 1, "Expected one done environment in smoke env"
    idx = int(done_idx[0].item())

    # Core risk check: terminal obs must not match auto-reset obs on done transitions.
    assert not torch.allclose(
        info["obs_before_reset"][idx], obs[idx]
    ), "obs_before_reset contamination detected"

    diag = adapter.get_diagnostics(reset=False)
    if include_terminal_obs:
        assert diag["terminal_obs_from_info"] >= 1, "expected terminal obs from info"
    else:
        assert (
            diag["terminal_obs_from_fallback"] >= 1
        ), "expected fallback terminal obs path"


def main():
    _assert_done_semantics(include_terminal_obs=True)
    _assert_done_semantics(include_terminal_obs=False)
    print("MJLabPWMAdapter smoke semantics: PASS")


if __name__ == "__main__":
    main()
