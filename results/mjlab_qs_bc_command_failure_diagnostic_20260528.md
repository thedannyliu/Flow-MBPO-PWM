# MJLab QS BC Command Failure Diagnostic - 2026-05-28

## Scope

Offline analysis of existing 40-episode, 1000-step init diagnostic eval CSVs:

- expert+noisy BC:
  `scripts/outputs/mjlab_qs/policy_evals/rerun_g1_bc_initdiag_eval40_long1000_20260528/velocity_flat_unitree_g1/mlp_ref/mlp/offline/bc50k_expert_uniform_policy0k/seed_*/final/eval_episodes.csv`
- expert-only BC:
  `scripts/outputs/mjlab_qs/policy_evals/rerun_g1_bc_initdiag_eval40_long1000_20260528/velocity_flat_unitree_g1/mlp_ref/mlp/offline/bc50k_expertonly_uniform_policy0k/seed_*/final/eval_episodes.csv`

No new environment rollout was launched for this diagnostic.

## Aggregate Results

| Variant | Episodes | Return | Length | Fall | Timeout |
|---|---:|---:|---:|---:|---:|
| expert+noisy BC | 120 | `45.7831` | `589.43` | `0.667` | `0.333` |
| expert-only BC | 120 | `30.6292` | `412.38` | `0.800` | `0.200` |

## Correlation With Episode Length

| Variant | `start_command_0` | `start_command_1` | `start_command_2` | `start_action_l2` |
|---|---:|---:|---:|---:|
| expert+noisy BC | `0.093` | `-0.177` | `-0.319` | `-0.331` |
| expert-only BC | `0.119` | `-0.012` | `-0.333` | `-0.316` |

## Expert+Noisy BC Bins

Yaw bins use tertiles of `abs(start_command_2)` with cuts `0.2615` and `0.5190`.

| Bin | Episodes | Length | Fall | Return |
|---|---:|---:|---:|---:|
| low abs yaw | 41 | `675.3` | `0.659` | `55.19` |
| mid abs yaw | 40 | `650.2` | `0.525` | `50.35` |
| high abs yaw | 39 | `436.8` | `0.821` | `31.20` |
| negative yaw | 48 | `678.9` | `0.625` | `52.52` |
| positive yaw | 72 | `529.8` | `0.694` | `41.29` |

First-action bins use tertiles of `start_action_l2` with cuts `0.1598` and `0.1732`.

| Bin | Episodes | Length | Fall |
|---|---:|---:|---:|
| low first action | 41 | `688.9` | `0.634` |
| mid first action | 40 | `650.3` | `0.550` |
| high first action | 39 | `422.4` | `0.821` |

## Interpretation

The repeated failure pattern is high yaw-command and high first-action sensitivity, not a constant-reset-state issue. Expert-only BC has the same pattern and worse aggregate robustness, so removing noisy expert data is not a fix. The next BC intervention should target command-conditioned yaw/recovery coverage or action ramping at rollout start before returning to PWM-style imagined policy improvement.
