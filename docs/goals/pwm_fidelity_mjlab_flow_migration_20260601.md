# PWM Fidelity, MJLab Transfer, and Flow Migration

Date: 2026-06-01
Branch at setup: `mjlab-qs-rollout-policy-improvement`
Repository SHA at setup: `d372003b7293e94823d20661cc8e282aaecc52a9`

## Objective

Determine whether PWM failure on MJLab QS is caused by an implementation/fidelity bug or by true transfer/protocol mismatch. Only after original PWM parity and faithful MJLab PWM are documented should Flow replacements be tested.

The enforced order is:

1. Phase 0: document original PWM setup, commands, metrics, checkpoints, and deviations.
2. Phase 1: verify original-environment PWM parity before MJLab.
3. Phase 2: add focused bug/fidelity checks and fix confirmed bugs.
4. Phase 3: port faithful PWM to MJLab with adapters only.
5. Phase 4: test Flow replacements one variable at a time.

## Source Status

Requested source path:

```bash
ls -la /baselines /baselines/PWM
```

Result: `/baselines` and `/baselines/PWM` do not exist on this machine.

Available local original clone:

```bash
git -C baselines/PWM rev-parse HEAD
git -C baselines/PWM remote -v
git ls-remote https://github.com/imgeorgiev/PWM.git HEAD refs/heads/main
```

Result:

```text
baselines/PWM HEAD: 9816252019ad8ca9a4393bceacf8a4dde711a749
origin: https://github.com/imgeorgiev/PWM.git
official main HEAD: 9816252019ad8ca9a4393bceacf8a4dde711a749
```

Decision: use `baselines/PWM` as the clean original source. It matches official `main`, so no fresh clone is needed yet. The missing absolute `/baselines/PWM` path is a documented path deviation, not a code deviation.

Paper source:

- arXiv: `https://arxiv.org/abs/2407.02466`
- Current arXiv record inspected on 2026-06-01: v3, last revised 2025-02-24.
- Paper title on v3: `PWM: Policy Learning with Large World Models`.

## Original PWM Entrypoints

Original repo entrypoints:

- `baselines/PWM/scripts/train_dflex.py`
  - Single-task DFlex experiments.
  - Loads pretrained DFlex world models through `general.checkpoint`.
  - Can pretrain DFlex world models from DFlex-format offline data through `general.pretrain` and `general.pretrain_steps`.
  - Uses `baselines/PWM/scripts/cfg/config.yaml`.

- `baselines/PWM/scripts/train_multitask.py`
  - TD-MPC2 MT30/MT80 task policy extraction.
  - Loads TD-MPC2/PWM world model with `general.checkpoint`.
  - Loads TD-MPC2 offline `.pt` trajectories from `general.data_dir`.
  - Samples H-step sequences through `pwm.utils.buffer.Buffer`.
  - Uses `baselines/PWM/scripts/cfg/config_mt30.yaml` or `config_mt80.yaml`.

There is no upstream entrypoint that directly consumes MJLab QS windows. Any MJLab run before Phase 3 is therefore an adapter experiment, not original PWM parity.

## Supported Original Environments

DFlex config files in `baselines/PWM/scripts/cfg/env/`:

- `dflex_ant.yaml`
- `dflex_anymal.yaml`
- `dflex_cartpole.yaml`
- `dflex_doublependulum.yaml`
- `dflex_hopper.yaml`
- `dflex_humanoid.yaml`
- `dflex_snu_humanoid.yaml`

Paper locomotion tasks and dimensions from Appendix D:

| Task | State dim | Action dim |
| --- | ---: | ---: |
| Hopper | 11 | 3 |
| Ant | 37 | 8 |
| Anymal | 49 | 12 |
| Humanoid | 76 | 21 |
| SNU Humanoid | not listed in extracted text | 152 |

MT30/MT80 tasks are inherited from TD-MPC2 through `baselines/PWM/external/tdmpc2`.

## Original PWM Settings to Preserve

From `baselines/PWM/scripts/cfg/alg/pwm.yaml` for DFlex:

```text
actor hidden: [400, 200, 100]
critic hidden: [400, 200]
latent_dim: 512
world model hidden: [512, 512]
encoder hidden: [256]
actor_lr: 5e-4
critic_lr: 5e-4
model_lr: 3e-4
lr_schedule: linear
horizon: 16
gamma: 0.99
lambda: 0.95
num_critics: 3
critic_iterations: 8
critic_batches: 4
actor_grad_norm: 1.0
critic_grad_norm: 100.0
wm_batch_size: 256
wm_iterations: 8
wm_grad_norm: 20.0
obs_rms: false
rew_rms: true
ret_rms: true
```

From `baselines/PWM/scripts/cfg/alg/pwm_5M.yaml` and larger multitask configs:

```text
reward bins: 101
reward vmin/vmax: -10.0 / 10.0
multitask reward RMS: false
multitask return RMS: true
horizon: 16
buffer batch size: 512
policy epochs in config_mt30/config_mt80: 10000
eval frequency: 200
eval runs: 10
```

From paper Appendix C:

- Reward model uses two-hot binned reward classification in SymLog space for PWM world-model training.
- Actor loss backpropagates first-order gradients through H-step learned-world-model rollouts and bootstraps terminal value.
- Critic is trained with TD(lambda).
- Critic data from one rollout is split into 4 minibatches and trained for 8 iterations.
- Actor minimum policy standard deviation is kept around 0.24 through `min_logstd: -1.427`.
- Encoder and dynamics outputs use SimNorm.
- Paper Table 1 lists 48M world model settings with latent dimension 768, WM batch size 1024, SimNorm dimension 8, and task encoding dimension 96. The checked 5M local config uses latent dimension 512 and smaller hidden layers.

## Expected Original Metrics

Paper Appendix D reports asymptotic PPO-normalized 50% IQM across 10 seeds:

| Task | PWM normalized reward |
| --- | ---: |
| Hopper | 1.20 +/- 0.29 |
| Ant | 1.46 +/- 0.31 |
| Anymal | 1.16 +/- 0.24 |
| Humanoid | 1.19 +/- 0.025 |
| SNU Humanoid | 1.36 +/- 0.56 |

Local `baselines/PWM/results/data/pwm_*.csv` final and best raw reward summaries across 5 logged runs:

| Task | Final mean | Final std | Best mean | Best std |
| --- | ---: | ---: | ---: | ---: |
| Hopper | 5649.179 | 53.354 | 5712.018 | 7.139 |
| Ant | 9546.091 | 1042.779 | 9618.120 | 1047.619 |
| Anymal | 14073.010 | 1174.408 | 14643.619 | 575.076 |
| Humanoid | 8773.130 | 730.103 | 9415.270 | 500.460 |
| SNU Humanoid | 5850.687 | 298.912 | 5905.503 | 358.727 |

Parity target for Phase 1 is not bitwise determinism. It is a paper-like learning curve and final reward on an original supported DFlex task using the closest available original config/checkpoint.

## Local Pretrained Assets and Data

Local full-size DFlex assets:

```text
scripts/assets/pwm_hf/dflex/pretrained/PWM_AntEnv.pt       22M
scripts/assets/pwm_hf/dflex/pretrained/PWM_AnymalEnv.pt    22M
scripts/assets/pwm_hf/dflex/pretrained/PWM_HopperEnv.pt    22M
scripts/assets/pwm_hf/dflex/pretrained/PWM_HumanoidEnv.pt  22M
scripts/assets/pwm_hf/dflex/pretrained/PWM_SNUHumanoidEnv.pt 24M

scripts/assets/pwm_hf/dflex/data/ep_data_AntEnv.pt         2.9G
scripts/assets/pwm_hf/dflex/data/ep_data_AnymalEnv.pt      2.4G
scripts/assets/pwm_hf/dflex/data/ep_data_HopperEnv.pt      3.7G
scripts/assets/pwm_hf/dflex/data/ep_data_HumanoidEnv.pt    7.5G
scripts/assets/pwm_hf/dflex/data/ep_data_SNUHumanoidEnv.pt 7.9G
```

Other paths:

- `scripts/assets/pwm_pretrained/PWM_HopperEnv.pt` is a 22M duplicate Hopper checkpoint.
- `hf_pwm_repo/` contains some Git LFS pointer files and one full `mt30_48M_4900000.pt` file. Treat `scripts/assets/pwm_hf/` as the preferred local asset root unless verified otherwise.

## Environment Status

Conda env:

```text
pwm: /storage/home/hcoda1/9/eliu354/r-agarg35-0/envs/pwm
```

Current shell Python is CPU-only:

```text
torch 2.8.0+cpu
cuda_available False
cuda_device_count 0
```

Implication: Phase 1 smoke/formal parity should run under Slurm GPU allocation or inside the `pwm` conda env on a GPU node. Do not treat a CPU-only login-shell failure as PWM parity evidence.

## Phase 1 Candidate Commands

W&B-disabled Hopper smoke with pretrained WM:

```bash
cd baselines/PWM/scripts
conda activate pwm
python train_dflex.py \
  env=dflex_hopper \
  alg=pwm \
  general.run_wandb=False \
  general.checkpoint=/storage/project/r-agarg35-0/eliu354/projects/Flow-MBPO-PWM/scripts/assets/pwm_hf/dflex/pretrained/PWM_HopperEnv.pt \
  alg.max_epochs=10 \
  alg.save_interval=10 \
  general.eval_runs=1 \
  general.logdir=logs/phase1_hopper_smoke_20260601
```

Formal single-seed Hopper parity, W&B enabled, `embers` QOS:

```bash
sbatch \
  --job-name=pwm_phase1_hopper_seed0 \
  --account=gts-agarg35 \
  --partition=ice-gpu \
  --qos=embers \
  --gres=gpu:h100:1 \
  --nodes=1 \
  --ntasks=1 \
  --cpus-per-task=8 \
  --mem=128G \
  --time=04:00:00 \
  --output=logs/slurm/pwm_phase1/pwm_phase1_hopper_seed0_%j.out \
  --error=logs/slurm/pwm_phase1/pwm_phase1_hopper_seed0_%j.err \
  --wrap='cd /storage/project/r-agarg35-0/eliu354/projects/Flow-MBPO-PWM/baselines/PWM/scripts && source ~/.bashrc && conda activate pwm && python train_dflex.py env=dflex_hopper alg=pwm general.seed=0 general.run_wandb=True wandb.project=flow-mbpo-pwm-fidelity wandb.group=phase1_original_hopper_20260601 wandb.notes="Phase 1 original PWM parity. repo_sha=d372003b7293e94823d20661cc8e282aaecc52a9 pwm_sha=9816252019ad8ca9a4393bceacf8a4dde711a749 env=dflex_hopper checkpoint=scripts/assets/pwm_hf/dflex/pretrained/PWM_HopperEnv.pt qos=embers" general.checkpoint=/storage/project/r-agarg35-0/eliu354/projects/Flow-MBPO-PWM/scripts/assets/pwm_hf/dflex/pretrained/PWM_HopperEnv.pt general.eval_runs=12 general.logdir=logs/phase1_hopper_seed0_20260601'
```

The formal run must record:

- repo SHA and PWM SHA;
- full command;
- config overrides;
- env and checkpoint path;
- seed;
- Slurm job ID and log paths;
- W&B run ID/URL;
- final checkpoint paths;
- notes on any deviation from original paper conditions.

## Required Stop Rules

- If original DFlex parity is not paper-like, stop and debug Phase 1. Do not port to MJLab and do not test Flow.
- If Phase 1 passes and faithful MJLab PWM fails with real eval/video evidence, the likely diagnosis is transfer/protocol/fall-signal mismatch.
- If faithful MJLab PWM beats the current PWM-style runner, the current implementation was nonfaithful or buggy.
- Flow replacement is blocked until Phases 1-3 have committed commands, logs, checkpoints, and eval/video evidence.

## Phase 2 Audit Checklist

Add only minimal tests/logging around confirmed risk points:

- action normalization, denormalization, tanh, clamp, and joint order;
- observation/state normalization;
- command/task conditioning;
- last-action fields;
- reward sign, scale, bins, and SymLog/two-hot behavior where applicable;
- done/reset/termination/truncation/timeout masks;
- value bootstrap masking;
- checkpoint load/save/eval parity;
- expert or BC actions rolled through WM versus real dataset segments;
- actor gradient norm;
- action drift from BC/expert;
- action saturation;
- support/OOD distance of actor-generated states/actions.

## Phase 3 MJLab Evidence Requirements

Faithful PWM on MJLab must use MJLab QS dataset/env wrappers only as adapters first. It must preserve original PWM update logic and settings unless a documented IO mismatch forces a boundary adaptation.

Formal MJLab claims require:

- W&B-enabled formal run on `embers`;
- real MJLab eval for final actor and true-best actor;
- 40-episode return, episode length, and fall rate;
- 10-episode, 1000-step MP4 videos logged to W&B;
- comparison against expert, expert-noisy, medium, random/reference, and best BC.

## Phase 4 Flow A/B Gate

Only after Phase 1-3 results are documented:

| Row | World model | Policy/update |
| --- | --- | --- |
| 1 | original PWM | original PWM |
| 2 | Flow WM | original PWM |
| 3 | original PWM | Flow policy, only if policy architecture is suspected |
| 4 | Flow WM | Flow policy, only after rows 1-3 |

Keep horizon, actor-critic training, data, and eval fixed across rows. If imagined return improves but real MJLab rollout or fall rate worsens, treat it as model exploitation and pivot back to pessimistic short-rollout Flow-MBPO.

## Deviations and Open Items

- Absolute `/baselines/PWM` is missing; repository-local `baselines/PWM` is current with official `main`.
- No Phase 1 smoke has been run yet in this document.
- Current shell is CPU-only; use Slurm GPU or a GPU node for real parity.
- Need to verify DFlex import/runtime under the `pwm` conda env before submitting formal parity.
- Need to capture exact W&B run IDs and Slurm logs once Phase 1 jobs run.
