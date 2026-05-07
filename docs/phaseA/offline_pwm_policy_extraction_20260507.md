# Offline PWM Policy Extraction Follow-Up - 2026-05-07

## Purpose

This run is the next step after the MJLab-QS offline world-model feasibility restart. The goal is to move from fixed-data world-model pretraining to a PWM-style policy extraction check:

1. collect/fix a real MJLab offline dataset,
2. pretrain a world model on that fixed data,
3. freeze the pretrained world model,
4. train an actor-critic policy only through differentiable imagined rollouts,
5. evaluate the extracted policy back in the real MJLab environment.

This is not yet the full online Dreamer-style loop. It is the offline/frozen-WM policy extraction stage. The online loop would additionally return the improved policy to the real environment, append new real data to the buffer, and fine-tune the world model and actor-critic iteratively.

## Original PWM Code Checked

The original PWM implementation was checked under `baselines/PWM` before submitting this batch:

- `baselines/PWM/scripts/cfg/alg/pwm.yaml`
- `baselines/PWM/scripts/cfg/alg/pwm_5M.yaml`
- `baselines/PWM/src/pwm/algorithms/pwm.py`
- `baselines/PWM/src/pwm/models/actor.py`
- `baselines/PWM/src/pwm/models/critic.py`

The original PWM loop uses:

- world model pretraining on replay subsequences,
- differentiable imagined rollout inside the learned world model,
- stochastic actor update through the world model return,
- critic ensemble trained with imagined TD(lambda)-style targets,
- real environment evaluation after policy extraction.

## PWM-Aligned Hyperparameters Used Here

The new policy-extraction runner uses the original PWM actor/critic style where compatible with our fixed state-space WM checkpoints:

| Component | Setting |
|---|---|
| Actor | `ActorStochasticMLP` |
| Actor hidden units | `[400, 200, 100]` |
| Actor activation | Mish, inherited from PWM model class |
| Actor init log std | `-1.0` |
| Actor min log std | `-1.427` |
| Critic | `CriticMLP` ensemble |
| Number of critics | `3` |
| Critic hidden units | `[400, 200]` |
| Horizon | `16` |
| Batch size | `64` |
| Discount | `gamma = 0.99` |
| TD(lambda) | `lambda = 0.95` |
| Actor LR | `5e-4` |
| Critic LR | `5e-4` |
| Critic iterations | `8` |
| Critic batches | `4` |
| Actor grad norm | `1.0` |
| Critic grad norm | `100.0` |
| Return RMS | enabled |

## Important Implementation Differences From Original PWM

This runner is PWM-aligned, but it is not a byte-for-byte reuse of `baselines/PWM` because the current MJLab-QS world-model checkpoints are not original PWM checkpoints.

Controlled differences:

- `z_t := normalized phys_obs_t`; there is no learned encoder `E_phi` in this specific comparison. This is intentional to avoid cross-architecture latent comparability confounds.
- Dynamics are trained and rolled out in fixed normalized physical state space.
- Reward is scalar normalized MSE in the current state-space runner, not the original PWM two-hot/discrete reward head.
- Commands are separated from physical dynamics: the transition model predicts physical state, while reward/policy/value condition on command when available.
- The current stage freezes the pretrained WM during policy extraction. Full online WM fine-tuning is not part of this batch.

Therefore the current experiment matches the model-based RL/PWM diagram at the pipeline level, but not every original PWM internal implementation detail.

## Dataset And Checkpoints

Task:

- `velocity_flat_unitree_g1`

Offline dataset:

- `scripts/outputs/mjlab_qs/windows/a25_native_qs_g1stage4/velocity_flat_unitree_g1/d_qs_core_h16.pt`

World-model checkpoints used for policy extraction:

| Label | Path | Status |
|---|---|---|
| `mlp50` | `scripts/outputs/mjlab_qs/results/offline_pretrain_g1_mlp_ref_50k_20260506/velocity_flat_unitree_g1/mlp_ref/seed_*/best.pt` | complete |
| `flow50` | `scripts/outputs/mjlab_qs/results/offline_pretrain_g1_flow_endpoint_equal50k_20260506/velocity_flat_unitree_g1/flow_endpoint/seed_*/best.pt` | complete |
| `flow75` | `scripts/outputs/mjlab_qs/results/offline_pretrain_g1_flow_endpoint_1p5_75k_20260506/velocity_flat_unitree_g1/flow_endpoint/seed_*/best.pt` | complete |
| `mlp75` | `scripts/outputs/mjlab_qs/results/offline_pretrain_g1_mlp_ref_1p5_75k_20260507/velocity_flat_unitree_g1/mlp_ref/seed_*/best.pt` | submitted, pending/running |

## Is 50k Offline Pretraining Enough?

For equal-update comparison, yes: 50k is sufficient for both MLP and Flow endpoint to reach the same loss regime on this G1 QS dataset.

At 50k, Flow endpoint is not showing the old pure-flow failure mode. It fits the fixed data competitively and often slightly better on validation/test rollout metrics. Flow still costs roughly 2x wall-clock per 50k update budget because each transition uses ODE/endpoint integration.

Pretrain H16 rollout metrics from Slurm logs:

| Method | Seed | Best Iter | Train H16 at Best Val | Val H16 Best | Final Iter | Final Train H16 | Final Val H16 | Wall Clock s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| MLP 50k | 0 | 45000 | 0.013673 | 0.036206 | 50000 | 0.014020 | 0.036824 | 1025.3 |
| MLP 50k | 1 | 50000 | 0.012685 | 0.034618 | 50000 | 0.012685 | 0.034618 | 1114.0 |
| MLP 50k | 2 | 50000 | 0.012598 | 0.036123 | 50000 | 0.012598 | 0.036123 | 1095.9 |
| Flow endpoint 50k | 0 | 45000 | 0.013864 | 0.035220 | 50000 | 0.015610 | 0.036640 | 2051.2 |
| Flow endpoint 50k | 1 | 50000 | 0.011671 | 0.032761 | 50000 | 0.011671 | 0.032761 | 2079.2 |
| Flow endpoint 50k | 2 | 50000 | 0.013657 | 0.033782 | 50000 | 0.013657 | 0.033782 | 2151.0 |
| Flow endpoint 75k | 0 | 70000 | 0.010072 | 0.032209 | 75000 | 0.010381 | 0.033180 | 3104.2 |
| Flow endpoint 75k | 1 | 70000 | 0.010267 | 0.031754 | 75000 | 0.013704 | 0.035789 | 3105.1 |
| Flow endpoint 75k | 2 | 70000 | 0.010294 | 0.030990 | 75000 | 0.010630 | 0.031133 | 3224.6 |

Interpretation:

- MLP 50k and Flow endpoint 50k are the fair equal-update comparison.
- Flow endpoint 50k already matches MLP train H16 rollout loss scale and has lower validation H16 rollout loss in all three seeds.
- Flow endpoint 75k is the 1.5x-compute diagnostic requested to test whether Flow can push training loss lower.
- Flow endpoint 75k should not be used as the fair 1.5x claim until the MLP 75k control finishes.

## Submitted Jobs

### MLP 75k Controls

These are needed to make the 1.5x comparison fair instead of comparing Flow75 only against MLP50.

| GPU | Job ID | Manifest | Time / Mem |
|---|---:|---|---|
| H100 | `5268448` | `scripts/outputs/mjlab_qs/manifests/offline_pretrain_g1_mlp_ref_1p5_75k_20260507_seed0_h100.csv` | 8h / 128G |
| H200 | `5268447` | `scripts/outputs/mjlab_qs/manifests/offline_pretrain_g1_mlp_ref_1p5_75k_20260507_seed1_h200.csv` | 8h / 128G |
| L40S | `5268446` | `scripts/outputs/mjlab_qs/manifests/offline_pretrain_g1_mlp_ref_1p5_75k_20260507_seed2_l40s.csv` | 10h / 128G |

### Offline PWM Policy Extraction

W&B project:

- `flow-mbpo-mjlab-offline-pwm-final-eval`

Group:

- `offline_pwm_g1_policy_extract_20260507`

Submitted with `16h / 450G` after cancelling earlier shorter duplicate submissions.

| GPU | Job ID | Manifest | Rows |
|---|---:|---|---:|
| H100 | `5268466` | `scripts/outputs/mjlab_qs/manifests/offline_pwm_g1_policy_extract_20260507_seed0_h100.csv` | 3 |
| H200 | `5268467` | `scripts/outputs/mjlab_qs/manifests/offline_pwm_g1_policy_extract_20260507_seed1_h200.csv` | 3 |
| L40S | `5268468` | `scripts/outputs/mjlab_qs/manifests/offline_pwm_g1_policy_extract_20260507_seed2_l40s.csv` | 3 |

Each shard contains MLP50, Flow50, and Flow75 for the same seed. MLP75 policy extraction will be submitted after MLP75 WM checkpoints finish.

## How To Interpret The Current Runs

Primary fair comparison now:

- `MLP50 + MLP policy extraction`
- `Flow endpoint 50k + MLP policy extraction`

Diagnostic compute-push comparison now:

- `Flow endpoint 75k + MLP policy extraction`

Required later for fair 1.5x comparison:

- `MLP75 + MLP policy extraction`
- compare against `Flow75 + MLP policy extraction`

If Flow50 has better WM validation metrics but worse final real-environment return, the bottleneck is likely not fixed-data dynamics fitting alone. It would point toward actor-gradient quality, reward/value learning, rollout exploitation, or mismatch between offline starts and real closed-loop control.

If Flow50 and Flow75 both improve final policy return over MLP50, then endpoint-supervised Flow WM becomes a serious candidate for the PWM pipeline, not just WM-only feasibility.
