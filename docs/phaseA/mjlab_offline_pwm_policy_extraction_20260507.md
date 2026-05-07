# MJLab Offline PWM Policy Extraction: MLP WM vs Flow Endpoint WM

Date: 2026-05-07
Branch: dev/mjlab

## Purpose

This stage continues the corrected MJLab-QS experiment through the PWM pipeline:

1. collect fixed real MJLab data,
2. train / pretrain a world model on the fixed offline buffer,
3. freeze the pretrained world model,
4. train an actor-critic policy by differentiable imagined rollouts through the world model,
5. evaluate the extracted policy in the real MJLab environment.

The goal is not an online Dreamer-style loop. This is the first downstream control check after offline WM pretraining. It asks whether the world model that fits fixed G1 data better also gives a better PWM policy after FoG policy extraction.

## Alignment With Original PWM

I checked the original PWM code under `baselines/PWM` and the local adapted PWM implementation.

Original PWM config source:

- `baselines/PWM/scripts/cfg/alg/pwm.yaml`
- `baselines/PWM/scripts/cfg/alg/pwm_5M.yaml`
- `baselines/PWM/src/pwm/algorithms/pwm.py`
- `baselines/PWM/src/pwm/models/actor.py`
- `baselines/PWM/src/pwm/models/critic.py`

Original PWM policy-side hyperparameters:

| Component | Original PWM value | This experiment |
|---|---:|---:|
| Actor | stochastic MLP | stochastic MLP |
| Actor hidden layers | `[400, 200, 100]` | `[400, 200, 100]` |
| Actor activation | Mish | Mish |
| Actor init logstd | `-1.0` | `-1.0` |
| Actor min logstd | `-1.427` | `-1.427` |
| Critic | MLP ensemble | MLP ensemble |
| Number of critics | `3` | `3` |
| Critic hidden layers | `[400, 200]` | `[400, 200]` |
| Critic activation | Mish | Mish |
| Horizon | `H=16` | `H=16` |
| Policy batch size | `64` | `64` |
| Actor LR | `5e-4` | `5e-4` |
| Critic LR | `5e-4` | `5e-4` |
| Discount | `gamma=0.99` | `gamma=0.99` |
| TD lambda | `lambda=0.95` | `lambda=0.95` |
| Critic minibatch split | `4` | `4` |
| Critic iterations | `8` | `8` |
| Actor grad clip | `1.0` | `1.0` |
| Critic grad clip | `100.0` | `100.0` |
| Return RMS | enabled | enabled |

Important caveat: this runner uses the corrected MJLab-QS state-space WM checkpoints from `run_phaseA_wm_feasibility.py`, where `z_t` is normalized physical observation rather than the original learned encoder latent `E_phi(s_t)`. This is intentional for the current fairness restart because it removes cross-architecture encoder confounds. Therefore, this is PWM-style frozen-WM policy extraction, not a byte-for-byte execution of `baselines/PWM/src/pwm/algorithms/pwm.py`.

## What Is Different From Original PWM

The differences are explicit:

1. The world model latent is fixed normalized physical state, not a learned SimNorm encoder latent.
2. The reward model is scalar MSE-trained in the MJLab-QS WM runner, not original PWM's two-hot reward CE.
3. The world model is frozen during policy extraction. This corresponds to PWM's policy extraction ablation and avoids online WM fine-tuning confounds.
4. The policy extraction runner samples start states from fixed offline windows rather than stepping the real environment during training.
5. Real MJLab is only used after policy training for final evaluation.

These differences are acceptable for the current experiment because the scientific question is controlled: given the same fixed offline data and same policy-side PWM hyperparameters, does MLP WM or endpoint-supervised Flow WM produce a better extracted policy?

## Data And WM Inputs

Dataset:

- Task: `velocity_flat_unitree_g1`
- MJLab task id: `Mjlab-Velocity-Flat-Unitree-G1`
- Dataset file: `scripts/outputs/mjlab_qs/windows/a25_native_qs_g1stage4/velocity_flat_unitree_g1/d_qs_core_h16.pt`
- Metadata: `scripts/outputs/mjlab_qs/windows/a25_native_qs_g1stage4/velocity_flat_unitree_g1/d_qs_core_h16.json`
- Normalization: `scripts/outputs/mjlab_qs/windows/a25_native_qs_g1stage4/velocity_flat_unitree_g1/d_qs_core_h16_normalization.json`

The state is split as:

```text
z_t = normalize(phys_obs_t)
c_t = normalize(command_t)
a_t = normalized policy action in [-1, 1]
```

Dynamics model:

```text
z_{t+1} = F_phi(z_t, a_t)
```

Reward model:

```text
r_hat_t = R_phi(z_t, a_t, c_t)
```

The command is not included in the physical dynamics input because it is an external task command. It is included in reward and policy/critic conditioning.

## Policy Extraction Objective

For each batch of offline start windows, sample:

```text
z_0, c_{0:H-1} ~ D_real
```

Then imagine through the frozen WM:

```text
a_h ~ pi_theta(. | z_h, c_h)
r_hat_h = R_phi(z_h, a_h, c_h)
z_{h+1} = F_phi(z_h, a_h)
```

Actor objective:

```text
maximize_theta E[ sum_{h=0}^{H-1} gamma^h r_hat_h + gamma^H V_psi(z_H, c_{H-1}) ]
```

Critic objective uses TD(lambda) targets over the imagined rollout, following PWM's critic update structure.

Real MJLab receives no gradient. Policy gradients flow only through the frozen world model.

## Submitted Comparisons

Each run trains an MLP policy and critic using the same policy-side PWM hyperparameters.

| WM | WM pretrain budget | Policy budget | Seeds |
|---|---:|---:|---:|
| MLP WM | 50k | 15k | 0, 1, 2 |
| Flow endpoint WM | 50k | 15k | 0, 1, 2 |
| Flow endpoint WM | 75k | 15k | 0, 1, 2 |

W&B project:

```text
flow-mbpo-mjlab-offline-pwm-policy-extraction
```

Manifest:

```text
scripts/outputs/mjlab_qs/manifests/offline_pwm_policy_extract_g1_20260507.csv
```

Seed shards:

```text
scripts/outputs/mjlab_qs/manifests/offline_pwm_policy_extract_g1_20260507_seed0_h100.csv
scripts/outputs/mjlab_qs/manifests/offline_pwm_policy_extract_g1_20260507_seed1_h200.csv
scripts/outputs/mjlab_qs/manifests/offline_pwm_policy_extract_g1_20260507_seed2_l40s.csv
```

## Final Evaluation Metrics

Each extracted policy is evaluated in real MJLab after policy extraction:

- `eval/return_mean`
- `eval/return_std`
- `eval/episode_length_mean`
- `eval/episode_length_std`
- `eval/num_episodes`
- `eval/resolved_task_id`

Training metrics logged during imagined policy extraction:

- `train/imagined_return`
- `train/actor_loss`
- `train/critic_loss`
- `train/action_norm`
- `train/actor_std`
- `train/actor_grad_norm`
- `train/critic_grad_norm`
- `train/ret_rms_mean`
- `train/ret_rms_var`

## Interpretation Rule

This experiment can support a downstream statement only if real-env eval improves. A lower WM rollout error alone is insufficient. The result should be read as:

```text
Offline WM fit -> frozen-WM PWM-style policy extraction -> real MJLab eval
```

not as an online Dreamer loop and not as a full 2x2 Flow-policy experiment.

## Launch Log

### 2026-05-07

Local smoke:

```text
python run_offline_pwm_policy_extraction.py \
  --wm-method mlp_ref \
  --policy-iters 2 \
  --batch-size 4 \
  --skip-real-eval \
  --disable-wandb
```

Smoke result:

- Checkpoint loading succeeded.
- Frozen WM imagined rollout succeeded.
- Stochastic actor update succeeded.
- 3-critic TD(lambda) critic update succeeded.
- Summary writing succeeded.
- Real env eval was intentionally skipped.

First submission attempt:

```text
5268427 H200
5268428 H100
5268429 L40S
```

This attempt was canceled/replaced because the submit wrapper used system `python`, which did not have torch installed. L40S rows 0 and 1 failed immediately with `ModuleNotFoundError: No module named 'torch'`.

Corrected submission uses the explicit environment Python:

```text
/storage/ice1/2/9/eliu354/conda_envs/flow-mbpo/bin/python
```

Corrected formal jobs:

```text
5268438 H100 seed0 shard: MLP 50k, Flow endpoint 50k, Flow endpoint 75k
5268439 H200 seed1 shard: MLP 50k, Flow endpoint 50k, Flow endpoint 75k
5268440 L40S seed2 shard: MLP 50k, Flow endpoint 50k, Flow endpoint 75k
```

Current queue state at submission check:

```text
5268438_[0-2%1] PENDING H100
5268439_[0-2%1] PENDING H200
5268440_[0-2%1] PENDING L40S
```
