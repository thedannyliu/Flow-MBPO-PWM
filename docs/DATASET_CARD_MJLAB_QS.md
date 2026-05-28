# Dataset Card: MJLab QS Velocity Flat Unitree G1

## Dataset

Current state-space QS windows are under:

- `scripts/outputs/mjlab_qs/windows/<dataset_stage>/velocity_flat_unitree_g1/d_qs_core_h16.pt`
- `scripts/outputs/mjlab_qs/windows/<dataset_stage>/velocity_flat_unitree_g1/d_qs_core_h16.json`
- `scripts/outputs/mjlab_qs/windows/<dataset_stage>/velocity_flat_unitree_g1/d_qs_core_h16_normalization.json`

The policy-extraction manifest records the exact dataset stage and paths for each formal run.

## Collector Sources

Current baseline evidence uses native RSL-RL PPO conservative collector checkpoints for `Mjlab-Velocity-Flat-Unitree-G1` plus a `random_smooth` reference policy.

Known rollout anchors:

| Bin | Return | Length | Fall rate | Notes |
|---|---:|---:|---:|---|
| expert | `82.6090` | `1000.00` | `0.000` | Stable full-horizon collector rollout. |
| expert_noisy | `80.3525` | `1000.00` | `0.000` | Stable full-horizon rollout with action noise. |
| medium | `49.1935` | `653.33` | `0.667` | Useful but unstable baseline. |
| random_smooth | `0.4857` | `75.33` | `1.000` | Reference failure behavior. |

## Quality Filters

Policy extraction supports separate filters:

- `--bc-quality-filter`, used for BC warm start;
- `--policy-quality-filter`, used for PWM imagined policy sampling.

The current strongest BC baseline used `expert,expert_noisy`, with `282887` train windows and `250559` BC train windows in the recorded run.

## Required Audits Before New Claims

- Verify no NaN reward or action values with `scripts/experiments/mjlab_qs/audit_mjlab_qs_quality.py`.
- Record action dimension, observation split, command position, and normalization path in W&B.
- Confirm termination/fall stats distinguish true termination from time-limit truncation.
- Do not mix datasets or collectors without a new ledger row and updated comparison report.
