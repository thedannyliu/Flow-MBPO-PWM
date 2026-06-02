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

Previous parity attempts used this env:

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

### Locked Original PWM CUDA 11.8 Env

Created on 2026-06-01 after the first Hopper parity run failed far below the local reference result. The purpose is to remove runtime drift before diagnosing PWM algorithm fidelity.

Path:

```text
/storage/project/r-agarg35-0/eliu354/envs/pwm_orig_locked4
```

Tracked spec:

```text
docs/envs/pwm_original_locked_cu118_20260601.yaml
```

Exact conda create command used:

```bash
conda create -y -p /storage/project/r-agarg35-0/eliu354/envs/pwm_orig_locked4 \
  python=3.10.14 \
  'pytorch::pytorch=2.3.1=py3.10_cuda11.8_cudnn8.7.0_0' \
  'pytorch::torchvision=0.18.1=py310_cu118' \
  pytorch-cuda=11.8 \
  cuda-toolkit=11.8 cuda-version=11.8 cuda-nvcc=11.8.89 \
  mkl=2023.1.0 intel-openmp=2023.1.0 blas=1.0=mkl numpy=1.26 \
  pandas=2.2 matplotlib=3.8 seaborn=0.13 glew=2.1.0 pip \
  -c pytorch -c nvidia/label/cuda-11.8.0 -c defaults
```

Required runtime exports for all checks and Slurm jobs:

```bash
export ENV=/storage/project/r-agarg35-0/eliu354/envs/pwm_orig_locked4
export PYTHONNOUSERSITE=1
export PATH=$ENV/bin:$PATH
export CUDA_HOME=$ENV
export LD_LIBRARY_PATH=$ENV/lib:${LD_LIBRARY_PATH:-}
export MAX_JOBS=8
```

Reason: the first pip install saw packages from `~/.local`. All verification and training must use `PYTHONNOUSERSITE=1` so W&B, Hydra, TorchRL, and DFlex come from the locked env only.

Key installed versions after repair:

```text
python 3.10.14
pytorch 2.3.1 py3.10_cuda11.8_cudnn8.7.0_0
torchvision 0.18.1 py310_cu118
pytorch-cuda 11.8
cuda-toolkit 11.8.0
cuda-version 11.8
cuda-nvcc 11.8.89
mkl 2023.1.0
intel-openmp 2023.1.0
gcc_linux-64 11.2.0
gxx_linux-64 11.2.0
numpy 1.26.4
tensordict 0.4.0
torchrl 0.4.0
hydra-core 1.2.0
omegaconf 2.2.3
wandb 0.12.21
gym 0.23.1
ninja 1.11.1.4
ipython 8.24.0
setuptools 70.3.0
dflex 0.0.1 from DiffRL commit bb59db5cf65e63740787bf22f91bae3103b30d19
```

Login-node import status with user site disabled:

```text
torch 2.3.1, torch.version.cuda 11.8, cuda_available False on login node
tensordict 0.4.0
torchrl 0.4.0
hydra 1.2.0
omegaconf 2.2.3
wandb 0.12.21
gym 0.23.1
```

Repairs needed before DFlex import:

- `setuptools==82.0.1` did not satisfy W&B 0.12's `pkg_resources` import path; pinned `setuptools==70.3.0`.
- `hydra-core` and `omegaconf` initially resolved to 1.4/2.4 dev builds; pinned to 1.2.0/2.2.3 to match `version_base="1.2"`.
- `gym` was absent from the original environment YAML but required by PWM and DFlex; installed `gym==0.23.1`.
- `ninja` must be on `PATH`; installing the Python package alone is insufficient when calling the env Python directly without `PATH=$ENV/bin:$PATH`.

DFlex CPU-side extension rebuild check:

```bash
PYTHONNOUSERSITE=1 \
PATH=/storage/project/r-agarg35-0/eliu354/envs/pwm_orig_locked4/bin:$PATH \
CUDA_HOME=/storage/project/r-agarg35-0/eliu354/envs/pwm_orig_locked4 \
LD_LIBRARY_PATH=/storage/project/r-agarg35-0/eliu354/envs/pwm_orig_locked4/lib:${LD_LIBRARY_PATH:-} \
MAX_JOBS=4 \
/storage/project/r-agarg35-0/eliu354/envs/pwm_orig_locked4/bin/python - <<'PY'
import shutil, torch
print("ninja", shutil.which("ninja"))
print("torch", torch.__version__, torch.version.cuda, torch.cuda.is_available())
import dflex
print("dflex import ok", getattr(dflex, "__file__", None))
PY
```

Result: passed on login node as CPU-only rebuild. DFlex compiled `kernels.so` with the env toolchain after `PATH` was fixed.

DFlex CUDA rebuild probe:

```text
High-tier attempts, in requested priority order:
  9383647 H200 embers: pending due Priority; canceled before runtime
  9383692 H100 embers: pending due Priority; canceled before runtime
  9383693 A100 embers: pending due Resources; canceled before runtime
  9383697 L40S embers: pending due Priority; canceled before runtime

Compiler/runtime debug attempts:
  9383700 RTX6000 embers: failed; CUDA 11.8 nvcc picked GCC 12 from cluster environment
  9383705 RTX6000 embers: canceled; stale DFlex extension lock
  9383712 RTX6000 embers: failed; conda compiler wrapper could not execute cc1plus
  9383715 RTX6000 embers: failed; inherited cluster CPATH/LIBRARY_PATH broke C++ headers
  9383721 RTX6000 embers: failed; system GCC path still lacked explicit C++ include path

Passing CUDA rebuild:
  Slurm job: 9383730
  GPU/QOS: Quadro RTX 6000, embers
  Output: logs/pwm_original_parity/locked_env_20260601/dflex_cuda_probe_9383730.out
  Error: logs/pwm_original_parity/locked_env_20260601/dflex_cuda_probe_9383730.err
  Result: imported torch 2.3.1+cu118 with CUDA available, rebuilt DFlex kernels, imported dflex, and instantiated HopperEnv(num_envs=2, device="cuda:0")
```

The passing probe used `/usr/bin/gcc` and `/usr/bin/g++` 11.5 as CUDA host compilers with explicit C++ include paths. The conda `gcc_linux-64`/`gxx_linux-64` 11.2 packages remain documented in the env spec, but they were not the final successful CUDA host compiler path on Slurm.

Required Slurm compiler exports for DFlex CUDA rebuilds:

```bash
unset C_INCLUDE_PATH CPLUS_INCLUDE_PATH LIBRARY_PATH GCC_EXEC_PREFIX
export ENV=/storage/project/r-agarg35-0/eliu354/envs/pwm_orig_locked4
export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1
export PATH=$ENV/bin:$PATH
export CUDA_HOME=$ENV
export CUDACXX=$ENV/bin/nvcc
export CC=/usr/bin/gcc
export CXX=/usr/bin/g++
export CUDAHOSTCXX=$CXX
export CPATH=/usr/include/c++/11:/usr/include/c++/11/x86_64-redhat-linux:/usr/lib/gcc/x86_64-redhat-linux/11/include:/usr/include
export LD_LIBRARY_PATH=$ENV/lib:${LD_LIBRARY_PATH:-}
export MAX_JOBS=4
```

Formal H100/H200 parity jobs should delete generated DFlex kernel artifacts before starting so the extension rebuilds for the actual allocated GPU architecture rather than reusing the RTX6000 probe artifact.

## Phase 1 Candidate Commands

W&B-disabled Hopper smoke with pretrained WM:

```bash
cd /storage/project/r-agarg35-0/eliu354/projects/Flow-MBPO-PWM/baselines/PWM/scripts
python train_dflex.py \
  env=dflex_hopper alg=pwm \
  general.run_wandb=False general.seed=0 \
  general.checkpoint_mode=wm_only \
  general.checkpoint=/storage/project/r-agarg35-0/eliu354/projects/Flow-MBPO-PWM/scripts/assets/pwm_hf/dflex/pretrained/PWM_HopperEnv.pt \
  alg.max_epochs=70 alg.save_interval=70 general.eval_runs=1 \
  general.logdir=logs/phase1_hopper_smoke_locked_20260601
```

Smoke attempts:

```text
9383739: failed before training because IPython was missing; fixed by installing ipython==8.24.0
9383751: ran training but alg.max_epochs=2 ended before a 1000-step Hopper episode completed, so final buffer save saw an uninitialized replay buffer
9383756: passed on RTX6000/embers with W&B disabled

Passing smoke logs:
  stdout: logs/pwm_original_parity/locked_env_20260601/hopper_smoke_9383756.out
  stderr: logs/pwm_original_parity/locked_env_20260601/hopper_smoke_9383756.err

Passing smoke result:
  checkpoint_mode: wm_only
  env: dflex_hopper
  seed: 0
  epochs: 70
  final eval loss: -14.26
  final eval length: 34
  interpretation: runtime gate only; not a paper parity result
```

Formal single-seed Hopper parity, W&B enabled, `embers` QOS:

```bash
sbatch --parsable \
  --job-name=pwm_hopper_formal_locked_s0 \
  --account=gts-agarg35 \
  --partition=gpu-h200 \
  --gres=gpu:h200:1 \
  --nodes=1 \
  --ntasks=1 \
  --cpus-per-task=8 \
  --mem=128G \
  --time=04:00:00 \
  --qos=embers \
  --output=logs/pwm_original_parity/locked_env_20260601/hopper_formal_locked_s0_%j.out \
  --error=logs/pwm_original_parity/locked_env_20260601/hopper_formal_locked_s0_%j.err \
  --wrap='export locked-env variables above; delete DFlex kernel artifacts; print metadata; run train_dflex.py'
```

Locked formal submission:

```text
Slurm job: 9383776
Final status: failed before training
GPU/QOS: H200, embers
Main repo SHA at submit: 6fbbaf6d44946fe26e71e2162663bd5f7ddadeee
PWM repo SHA at submit: c7ed70a01916eee9a5b1ebaa356365b852c19418
Env path: /storage/project/r-agarg35-0/eliu354/envs/pwm_orig_locked4
Checkpoint: /storage/project/r-agarg35-0/eliu354/projects/Flow-MBPO-PWM/scripts/assets/pwm_hf/dflex/pretrained/PWM_HopperEnv.pt
Checkpoint mode: wm_only
Seed: 0
stdout: logs/pwm_original_parity/locked_env_20260601/hopper_formal_locked_s0_9383776.out
stderr: logs/pwm_original_parity/locked_env_20260601/hopper_formal_locked_s0_9383776.err
W&B project: flow-mbpo-pwm-fidelity
W&B group: phase1_original_hopper_locked_20260601
W&B run: https://wandb.ai/danny010324/flow-mbpo-pwm-fidelity/runs/0djv5v2d
Failure: DFlex printed `Using cached kernels`, then failed with `ImportError("No module named 'kernels'")`.
Cause: the wrapper deleted `.so/.o/build.ninja` artifacts but left `adjoint.gen`; DFlex cache validation only compares generated C++ source against `adjoint.gen`, so it skipped rebuild while the compiled module was absent.
```

Train command inside the Slurm wrapper:

```bash
/storage/project/r-agarg35-0/eliu354/envs/pwm_orig_locked4/bin/python train_dflex.py \
  env=dflex_hopper alg=pwm \
  general.seed=0 \
  general.run_wandb=True \
  general.checkpoint_mode=wm_only \
  general.checkpoint=/storage/project/r-agarg35-0/eliu354/projects/Flow-MBPO-PWM/scripts/assets/pwm_hf/dflex/pretrained/PWM_HopperEnv.pt \
  general.eval_runs=12 \
  general.logdir=logs/phase1_hopper_formal_locked_s0_20260601 \
  wandb.project=flow-mbpo-pwm-fidelity \
  wandb.group=phase1_original_hopper_locked_20260601 \
  +wandb.notes=phase1_original_pwm_hopper_locked_seed0_main_6fbbaf6d44946fe26e71e2162663bd5f7ddadeee_pwm_c7ed70a01916eee9a5b1ebaa356365b852c19418_env_pwm_orig_locked4_torch231_cu118_torchrl040_tensordict040_checkpoint_pwm_hopper_wm_only_qos_embers_partition_gpu_h200_rebuild_dflex
```

Corrected multi-GPU formal submissions:

```text
Strategy:
  Each job copies the locked env's `dflex/` package into `${TMPDIR}/dflex_sandbox_${SLURM_JOB_ID}`,
  removes that sandbox's `dflex/kernels` directory, prepends the sandbox to PYTHONPATH, and rebuilds
  DFlex kernels inside the job-local copy. This allows H200/H100/A100/L40S jobs to run concurrently
  without sharing or deleting one global env kernel cache.

Common settings:
  Main repo SHA at submit: 95527e7b1f8b9f5c852f6695e33ee6e9280d6b11
  PWM repo SHA at submit: c7ed70a01916eee9a5b1ebaa356365b852c19418
  Env path: /storage/project/r-agarg35-0/eliu354/envs/pwm_orig_locked4
  Checkpoint: /storage/project/r-agarg35-0/eliu354/projects/Flow-MBPO-PWM/scripts/assets/pwm_hf/dflex/pretrained/PWM_HopperEnv.pt
  Checkpoint mode: wm_only
  Seed: 0
  QOS: embers
  W&B project: flow-mbpo-pwm-fidelity
  W&B group: phase1_original_hopper_locked_20260601

Jobs:
  9383814 h200 gpu-h200: stdout logs/pwm_original_parity/locked_env_20260601/pwm_hopper_locked_h200_s0_9383814.out, stderr logs/pwm_original_parity/locked_env_20260601/pwm_hopper_locked_h200_s0_9383814.err
  9383815 h100 gpu-h100: stdout logs/pwm_original_parity/locked_env_20260601/pwm_hopper_locked_h100_s0_9383815.out, stderr logs/pwm_original_parity/locked_env_20260601/pwm_hopper_locked_h100_s0_9383815.err
  9383816 a100 gpu-a100: stdout logs/pwm_original_parity/locked_env_20260601/pwm_hopper_locked_a100_s0_9383816.out, stderr logs/pwm_original_parity/locked_env_20260601/pwm_hopper_locked_a100_s0_9383816.err
  9383817 l40s gpu-l40s: stdout logs/pwm_original_parity/locked_env_20260601/pwm_hopper_locked_l40s_s0_9383817.out, stderr logs/pwm_original_parity/locked_env_20260601/pwm_hopper_locked_l40s_s0_9383817.err
```

User-requested concurrency reduction:

```text
Time: 2026-06-01 evening EDT
Action: canceled 9383815, 9383816, and 9383817; kept only H200 job 9383814 running.
Reason: user requested only the highest-priority GPU run remain active after confirming that simultaneous
H200/H100/L40S sandbox rebuilds all worked and A100 was still pending.
Observed after cancellation:
  9383814 h200 RUNNING on atl1-1-03-020-18-0
  9383815 h100 CANCELLED by user
  9383816 a100 CANCELLED by user
  9383817 l40s CANCELLED by user
```

Prevent-repeat notes from the locked-env debug:

- Do not delete only `kernels.so`, `.o`, or `build.ninja` in a shared DFlex install. DFlex may still trust `adjoint.gen`, print `Using cached kernels`, skip rebuild, and then fail importing the missing compiled module.
- For formal GPU jobs, prefer a job-local DFlex sandbox copied from the locked env into `${TMPDIR}`. Delete the sandbox's whole `dflex/kernels` directory before import, put the sandbox first in `PYTHONPATH`, and rebuild there.
- Keep the locked compiler/runtime exports together: `PYTHONNOUSERSITE=1`, locked env `PATH`, `CUDA_HOME`, `CUDACXX`, `/usr/bin/gcc`, `/usr/bin/g++`, `CUDAHOSTCXX`, clean `C_INCLUDE_PATH`, `CPLUS_INCLUDE_PATH`, `LIBRARY_PATH`, and `GCC_EXEC_PREFIX`, plus the explicit GCC 11 include `CPATH`.
- H200/H100/L40S sandbox rebuilds succeeded independently. The earlier H200 failure was a DFlex cache invalidation error, not evidence that the locked torch/cu118 stack or H200 DFlex build was unusable.
- Continue Phase 1 with only job 9383814. Do not claim original PWM parity until the H200 run finishes, final and true-best checkpoints are present, and both actors pass a separate real-environment eval.

Follow-up queued jobs after the 2026-06-01 handoff:

```text
Current rule:
  Jobs may be queued in parallel, but interpretation remains Phase 1 -> Phase 2 -> Phase 3.
  All original PWM/DFlex jobs below use the locked env /storage/project/r-agarg35-0/eliu354/envs/pwm_orig_locked4
  and a job-local DFlex sandbox.

Phase 1, second original-task sanity:
  9384321 Ant smoke, H100/embers, W&B disabled, env=dflex_ant, checkpoint=PWM_AntEnv.pt,
          checkpoint_mode=wm_only, alg.max_epochs=80, eval_runs=4.
  9384344 Ant formal, H100/embers, W&B enabled, dependency=afterok:9384321,
          env=dflex_ant, checkpoint=PWM_AntEnv.pt, checkpoint_mode=wm_only, eval_runs=12.

Phase 1, Hopper final/best true DFlex eval and reload/cache isolation:
  9384354 final_policy true eval, H100/embers, dependency=afterok:9383814,
          checkpoint=baselines/PWM/scripts/outputs/2026-06-01/20-27-50/logs/phase1_hopper_formal_locked_h200_s0_20260601/final_policy.pt,
          output=eval_results/pwm_phase1_hopper_locked_h200/final_true_eval_20260601.
  9384355 best_policy true eval, H100/embers, dependency=afterok:9383814,
          checkpoint=baselines/PWM/scripts/outputs/2026-06-01/20-27-50/logs/phase1_hopper_formal_locked_h200_s0_20260601/best_policy.pt,
          output=eval_results/pwm_phase1_hopper_locked_h200/best_true_eval_20260601.

Phase 2, minimal WM-vs-real probe:
  9384374 final_policy WM-vs-real reward probe, H100/embers, dependency=afterok:9384354:9384355,
          output=logs/diagnostics/pwm_dflex_checkpoint_probe/hopper_h200_final_actor_256_20260601.json.
  9384375 best_policy WM-vs-real reward probe, H100/embers, dependency=afterok:9384354:9384355,
          output=logs/diagnostics/pwm_dflex_checkpoint_probe/hopper_h200_best_actor_256_20260601.json.

Phase 3, faithful MJLab adapter smoke queued but gated behind Phase 2:
  9384400 original_pwm_adapter smoke, H100/embers, dependency=afterok:9384374:9384375,
          manifest=scripts/experiments/mjlab_qs/manifests/original_pwm_adapter_phase3_smoke_20260601.csv,
          dataset=scripts/outputs/mjlab_qs/windows/rerun_a25_native_qs_g1stage4_expertboost_20260527/velocity_flat_unitree_g1/d_qs_core_h16.pt,
          W&B disabled, skip_real_eval=true, pretrain_iters=2, policy_iters=2.

Phase 3, faithful MJLab adapter formal queued but gated behind the smoke:
  9384485 original_pwm_adapter formal, H200/embers, dependency=afterok:9384400,
          manifest=scripts/experiments/mjlab_qs/manifests/original_pwm_adapter_phase3_formal_h200_seed0_20260601.csv,
          dataset=scripts/outputs/mjlab_qs/windows/rerun_a25_native_qs_g1stage4_expertboost_20260527/velocity_flat_unitree_g1/d_qs_core_h16.pt,
          W&B enabled, skip_real_eval=false, pretrain_iters=50000, policy_iters=15000,
          eval_episodes=40, eval_num_envs=16.

Rejected submission attempts before 9384485:
  H200 with cpus=12 failed because PACE enforces max CPU:GPU ratio 8:1 for the H200 node class.
  H200 with time=12:00:00 failed under embers with QOSMaxWallDurationPerJobLimit.
  Accepted settings: cpus=8, time=04:00:00, mem=192G.

Do not use 9384400 as a claim about MJLab performance. It is only a runtime smoke
for the original PWM algorithm adapter using existing MJLab QS IO.
Do not use 9384485 alone as a full MJLab claim either: the current adapter formal
does 40-episode real eval and saves best/final extraction policies, but separate
1000-step MP4/W&B video jobs are still required before final MJLab claims.
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
- Current shell is CPU-only; use Slurm GPU or a GPU node for real parity.
- Need to verify DFlex import/runtime under the `pwm` conda env before submitting formal parity.
- Need to capture exact W&B run IDs and Slurm logs once Phase 1 jobs run.

## Phase 1 Smoke Submission

Submitted W&B-disabled original PWM Hopper smoke:

```text
submitted_at: 2026-06-01
slurm_job_id: 9379551
job_name: pwm_p1_hopper_smoke
repo_sha: 6474b855457b0e726ea8359178af6a4e4fbf53bb
pwm_sha: 9816252019ad8ca9a4393bceacf8a4dde711a749
partition: gpu-h100
qos: embers
account: gts-agarg35
gres: gpu:h100:1
time_limit: 00:30:00
wandb: disabled
seed: 0
env: dflex_hopper
config: baselines/PWM/scripts/cfg/config.yaml + env=dflex_hopper + alg=pwm
checkpoint: scripts/assets/pwm_hf/dflex/pretrained/PWM_HopperEnv.pt
slurm_stdout: logs/slurm/pwm_phase1/pwm_p1_hopper_smoke_9379551.out
slurm_stderr: logs/slurm/pwm_phase1/pwm_p1_hopper_smoke_9379551.err
status_after_submit: PENDING (Priority)
```

Exact submitted command:

```bash
mkdir -p logs/slurm/pwm_phase1
sbatch \
  --job-name=pwm_p1_hopper_smoke \
  --account=gts-agarg35 \
  --partition=gpu-h100 \
  --qos=embers \
  --gres=gpu:h100:1 \
  --nodes=1 \
  --ntasks=1 \
  --cpus-per-task=8 \
  --mem=96G \
  --time=00:30:00 \
  --output=logs/slurm/pwm_phase1/pwm_p1_hopper_smoke_%j.out \
  --error=logs/slurm/pwm_phase1/pwm_p1_hopper_smoke_%j.err \
  --wrap='cd /storage/project/r-agarg35-0/eliu354/projects/Flow-MBPO-PWM && source ~/.bashrc && conda activate pwm && export WANDB_MODE=disabled && export PYTHONPATH=/storage/project/r-agarg35-0/eliu354/projects/Flow-MBPO-PWM/baselines/PWM/src:/storage/project/r-agarg35-0/eliu354/projects/Flow-MBPO-PWM/baselines/PWM/external/tdmpc2:$PYTHONPATH && cd baselines/PWM/scripts && python train_dflex.py env=dflex_hopper alg=pwm general.run_wandb=False general.seed=0 general.checkpoint=/storage/project/r-agarg35-0/eliu354/projects/Flow-MBPO-PWM/scripts/assets/pwm_hf/dflex/pretrained/PWM_HopperEnv.pt alg.max_epochs=10 alg.save_interval=10 general.eval_runs=1 general.logdir=logs/phase1_hopper_smoke_20260601'
```

Outcome: job `9379551` stayed pending with reason `Priority` and was canceled before running to avoid duplicate smoke jobs when an RTX6000 slot was available.

## Phase 1 Smoke Debug Log

The original-code smoke attempts exposed two runtime issues before a valid smoke passed:

| Job | Status | Evidence | Interpretation |
| --- | --- | --- | --- |
| `9379606` | failed | `torch.load` rejected `RunningMeanStd` because PyTorch 2.10 defaults toward `weights_only=True` | environment compatibility issue, not an algorithm deviation |
| `9379640` | failed | checkpoint loaded, but `alg.max_epochs=10` ended before a 1000-step Hopper episode completed; final buffer save saw no initialized replay buffer | smoke too short for original DFlex episode horizon |
| `9379646` | failed after training/save | 80-epoch smoke initialized replay buffer and saved policies, then final eval failed because `PWM.eval()` expected 3 values from `WorldModel.step()` while DFlex scalar-reward WM returns 2 | upstream/local original eval bug under DFlex scalar-reward path |

Compatibility settings retained for subsequent runs:

```text
TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
HYDRA_FULL_ERROR=1
PYTHONPATH=<repo>/baselines/PWM/src:<repo>/baselines/PWM/external/tdmpc2:$PYTHONPATH
```

Local baseline fix:

```text
baseline_repo: baselines/PWM
upstream_base_sha: 9816252019ad8ca9a4393bceacf8a4dde711a749
local_fix_sha: cb7b1689afbfd4b5662e43cdbae2c360e52d56a1
commit: Fix DFlex PWM eval world-model step unpack
file: baselines/PWM/src/pwm/algorithms/pwm.py
change: in PWM.eval(), unpack `z, rew = self.wm.step(...)` and apply `self.wm.almost_two_hot_inv(rew).squeeze()`, matching the training path.
```

This is a documented deviation from pristine upstream. It is limited to making original DFlex eval match the existing DFlex training/world-model API; it does not change architecture, horizon, losses, optimizers, actor/critic sizes, reward model, or update logic.

Passing W&B-disabled smoke:

```text
slurm_job_id: 9379661
job_name: pwm_p1_hopper_smoke_evalfix
repo_sha: 027058ca8cf73dea19113ebe6a3ba74628c86847
baseline_pwm_sha: cb7b1689afbfd4b5662e43cdbae2c360e52d56a1
upstream_pwm_sha: 9816252019ad8ca9a4393bceacf8a4dde711a749
partition: gpu-rtx6000
qos: embers
account: gts-agarg35
gres: gpu:rtx_6000:1
wandb: disabled
seed: 0
env: dflex_hopper
config: baselines/PWM/scripts/cfg/config.yaml + env=dflex_hopper + alg=pwm
checkpoint: scripts/assets/pwm_hf/dflex/pretrained/PWM_HopperEnv.pt
max_epochs: 80
eval_runs: 1
slurm_stdout: logs/slurm/pwm_phase1/pwm_p1_hopper_smoke_evalfix_9379661.out
slurm_stderr: logs/slurm/pwm_phase1/pwm_p1_hopper_smoke_evalfix_9379661.err
output_dir: baselines/PWM/scripts/outputs/2026-06-01/15-32-34/logs/phase1_hopper_smoke_20260601_evalfix
artifacts: init_policy.pt, best_policy.pt, final_policy.pt
slurm_state: COMPLETED
exit_code: 0:0
final_eval: mean episode loss = -377.16, mean discounted loss = -37.71, mean episode length = 1000.00
```

The smoke is not a parity result because it only ran 80 epochs. It is a runtime gate proving the original DFlex Hopper path can load the checkpoint, run original PWM actor/critic/WM updates, save final and best policies, and execute final eval with W&B disabled.

## Phase 1 Formal Submission

First formal submission:

```text
submitted_at: 2026-06-01
slurm_job_id: 9379689
job_name: pwm_p1_hopper_formal_s0
repo_sha: 027058ca8cf73dea19113ebe6a3ba74628c86847
baseline_pwm_sha: cb7b1689afbfd4b5662e43cdbae2c360e52d56a1
upstream_pwm_sha: 9816252019ad8ca9a4393bceacf8a4dde711a749
partition: gpu-rtx6000
qos: embers
account: gts-agarg35
gres: gpu:rtx_6000:1
wandb_project: flow-mbpo-pwm-fidelity
wandb_group: phase1_original_hopper_20260601
seed: 0
env: dflex_hopper
config: baselines/PWM/scripts/cfg/config.yaml + env=dflex_hopper + alg=pwm
checkpoint: scripts/assets/pwm_hf/dflex/pretrained/PWM_HopperEnv.pt
eval_runs: 12
slurm_stdout: logs/slurm/pwm_phase1/pwm_p1_hopper_formal_s0_9379689.out
slurm_stderr: logs/slurm/pwm_phase1/pwm_p1_hopper_formal_s0_9379689.err
status: FAILED before training
exit_code: 1:0
failure: Hydra override parse error from long W&B notes string containing unescaped `=` tokens.
```

Exact submitted command:

```bash
mkdir -p logs/slurm/pwm_phase1
sbatch \
  --job-name=pwm_p1_hopper_formal_s0 \
  --account=gts-agarg35 \
  --partition=gpu-rtx6000 \
  --qos=embers \
  --gres=gpu:rtx_6000:1 \
  --nodes=1 \
  --ntasks=1 \
  --cpus-per-task=6 \
  --mem=96G \
  --time=04:00:00 \
  --output=logs/slurm/pwm_phase1/pwm_p1_hopper_formal_s0_%j.out \
  --error=logs/slurm/pwm_phase1/pwm_p1_hopper_formal_s0_%j.err \
  --wrap='cd /storage/project/r-agarg35-0/eliu354/projects/Flow-MBPO-PWM && source ~/.bashrc && conda activate pwm && export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 && export HYDRA_FULL_ERROR=1 && export WANDB_DIR=/storage/project/r-agarg35-0/eliu354/projects/Flow-MBPO-PWM/baselines/PWM/scripts/wandb && mkdir -p "$WANDB_DIR" && export PYTHONPATH=/storage/project/r-agarg35-0/eliu354/projects/Flow-MBPO-PWM/baselines/PWM/src:/storage/project/r-agarg35-0/eliu354/projects/Flow-MBPO-PWM/baselines/PWM/external/tdmpc2:$PYTHONPATH && cd baselines/PWM/scripts && python train_dflex.py env=dflex_hopper alg=pwm general.run_wandb=True general.seed=0 general.checkpoint=/storage/project/r-agarg35-0/eliu354/projects/Flow-MBPO-PWM/scripts/assets/pwm_hf/dflex/pretrained/PWM_HopperEnv.pt general.eval_runs=12 general.logdir=logs/phase1_hopper_formal_seed0_20260601 wandb.project=flow-mbpo-pwm-fidelity wandb.group=phase1_original_hopper_20260601 +wandb.notes="Phase 1 original PWM Hopper formal seed. main_repo_sha=027058ca8cf73dea19113ebe6a3ba74628c86847 pwm_sha=cb7b1689afbfd4b5662e43cdbae2c360e52d56a1 upstream_pwm_sha=9816252019ad8ca9a4393bceacf8a4dde711a749 env=dflex_hopper alg=pwm config=baselines/PWM/scripts/cfg/config.yaml checkpoint=scripts/assets/pwm_hf/dflex/pretrained/PWM_HopperEnv.pt seed=0 qos=embers partition=gpu-rtx6000 smoke_job=9379661 compatibility=TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD plus local eval unpack fix"'
```

Corrected formal W&B seed on `embers`:

```text
submitted_at: 2026-06-01
slurm_job_id: 9379717
job_name: pwm_p1_hopper_formal_s0b
repo_sha: 027058ca8cf73dea19113ebe6a3ba74628c86847
baseline_pwm_sha: cb7b1689afbfd4b5662e43cdbae2c360e52d56a1
upstream_pwm_sha: 9816252019ad8ca9a4393bceacf8a4dde711a749
partition: gpu-rtx6000
qos: embers
account: gts-agarg35
gres: gpu:rtx_6000:1
wandb_project: flow-mbpo-pwm-fidelity
wandb_group: phase1_original_hopper_20260601
wandb_entity: danny010324
wandb_run_id: 3fzh44cb
wandb_url: https://wandb.ai/danny010324/flow-mbpo-pwm-fidelity/runs/3fzh44cb
seed: 0
env: dflex_hopper
config: baselines/PWM/scripts/cfg/config.yaml + env=dflex_hopper + alg=pwm
checkpoint: scripts/assets/pwm_hf/dflex/pretrained/PWM_HopperEnv.pt
eval_runs: 12
slurm_stdout: logs/slurm/pwm_phase1/pwm_p1_hopper_formal_s0b_9379717.out
slurm_stderr: logs/slurm/pwm_phase1/pwm_p1_hopper_formal_s0b_9379717.err
slurm_state: COMPLETED
exit_code: 0:0
elapsed: 02:40:39
start: 2026-06-01T15:35:24
end: 2026-06-01T18:16:03
output_dir: baselines/PWM/scripts/outputs/2026-06-01/15-35-34/logs/phase1_hopper_formal_seed0_20260601
```

Corrected command changed only the W&B notes override to a Hydra-safe value:

```text
+wandb.notes=phase1_original_pwm_hopper_formal_seed0_main_027058c_pwm_cb7b168_upstream_9816252_checkpoint_hopper_smoke_9379661_torch_force_no_weights_only_load_evalfix
```

Artifacts from job `9379717`:

```text
init_policy.pt
best_policy.pt
final_policy.pt
PWM_iter2500_rew1270.pt
PWM_iter5000_rew1272.pt
PWM_iter7500_rew1268.pt
PWM_iter10000_rew1268.pt
PWM_iter12500_rew1269.pt
```

Final training log evidence:

```text
best training-loop policy loss: -1276.41
best training-loop reward: 1276.41
last training line: [14937/15000] R:1271.43 T:0.0 H:13.9 S:15360000 pi_loss:-0.73 pi_grad:0.00/0.00 v_loss:0.01 wm_loss:0.00
final eval printed by training job: mean episode loss = -77.11, mean discounted loss = -13.36, mean episode length = 1000.00
```

Important interpretation detail: `PWM.eval()` in this DFlex path still computes episode loss from the learned world-model reward while stepping the real environment only for done signals. Therefore the printed final eval is original-entrypoint load/eval evidence, but it is not yet reconciled with a true real-env return metric.

## Phase 1 Checkpoint Eval

The first checkpoint eval submission failed before Python because `set -u` made `source ~/.bashrc` abort on an unset `BASHRCSOURCED` shell variable:

```text
slurm_job_id: 9382641
job_name: pwm_p1_hopper_eval_ckpts_s0
partition: gpu-rtx6000
qos: embers
status: FAILED
exit_code: 1:0
slurm_stdout: logs/slurm/pwm_phase1/pwm_p1_hopper_eval_ckpts_s0_9382641.out
slurm_stderr: logs/slurm/pwm_phase1/pwm_p1_hopper_eval_ckpts_s0_9382641.err
failure: /etc/bashrc: line 12: BASHRCSOURCED: unbound variable
```

Corrected W&B-enabled eval-only job:

```text
slurm_job_id: 9382643
job_name: pwm_p1_hopper_eval_ckpts_s0b
main_repo_sha: b829e8ccb7595881a4486a4dde9138361523dc8d
baseline_pwm_sha: cb7b1689afbfd4b5662e43cdbae2c360e52d56a1
partition: gpu-rtx6000
qos: embers
account: gts-agarg35
gres: gpu:rtx_6000:1
wandb_project: flow-mbpo-pwm-fidelity
wandb_group: phase1_original_hopper_eval_20260601
slurm_state: COMPLETED
exit_code: 0:0
elapsed: 00:01:24
start: 2026-06-01T18:22:09
end: 2026-06-01T18:23:33
slurm_stdout: logs/slurm/pwm_phase1/pwm_p1_hopper_eval_ckpts_s0b_9382643.out
slurm_stderr: logs/slurm/pwm_phase1/pwm_p1_hopper_eval_ckpts_s0b_9382643.err
```

Eval-only final actor:

```text
checkpoint: baselines/PWM/scripts/outputs/2026-06-01/15-35-34/logs/phase1_hopper_formal_seed0_20260601/final_policy.pt
wandb_run_id: elpnm8cc
wandb_url: https://wandb.ai/danny010324/flow-mbpo-pwm-fidelity/runs/elpnm8cc
command_core: python train_dflex.py env=dflex_hopper alg=pwm general.run_wandb=True general.train=False general.seed=0 general.checkpoint=<final_policy.pt> general.eval_runs=12 general.logdir=logs/phase1_hopper_eval_final_seed0_20260601
result: mean episode loss = -65.72, mean discounted loss = -13.15, mean episode length = 1000.00
```

Eval-only true-best actor:

```text
checkpoint: baselines/PWM/scripts/outputs/2026-06-01/15-35-34/logs/phase1_hopper_formal_seed0_20260601/best_policy.pt
wandb_run_id: 6jhmy3ap
wandb_url: https://wandb.ai/danny010324/flow-mbpo-pwm-fidelity/runs/6jhmy3ap
command_core: python train_dflex.py env=dflex_hopper alg=pwm general.run_wandb=True general.train=False general.seed=0 general.checkpoint=<best_policy.pt> general.eval_runs=12 general.logdir=logs/phase1_hopper_eval_best_seed0_20260601
result: mean episode loss = -109.38, mean discounted loss = -15.04, mean episode length = 1000.00
```

## Phase 1 Real-Env Reward Diagnostic

Because original `PWM.eval()` accumulates learned world-model reward in the DFlex path, a separate W&B-disabled real-environment evaluator was run before making further claims. The evaluator loads the same actor/world-model checkpoint, encodes the real DFlex observation, applies `tanh(actor(z))`, steps `dflex.envs.HopperEnv`, and accumulates the real `env.step()` reward.

Evaluator update:

```text
file: scripts/eval/eval_online_single_task.py
change: find `.hydra/config.yaml` by walking checkpoint parents and accept `--config-path` for checkpoints outside Hydra run dirs.
reason: original PWM stores policies below `outputs/.../logs/<logdir>/`, while pretrained assets live outside any Hydra output directory.
validation: python -m py_compile scripts/eval/eval_online_single_task.py
```

First real-env eval smoke failed before policy evaluation because the evaluator assumed the checkpoint parent was the Hydra run directory:

```text
slurm_job_id: 9382749
job_name: pwm_p1_hopper_realenv_eval_smoke
partition: gpu-rtx6000
qos: embers
status: FAILED
exit_code: 1:0
failure: missing Hydra config at checkpoint/logdir parent
slurm_stdout: logs/slurm/pwm_phase1/pwm_p1_hopper_realenv_eval_smoke_9382749.out
slurm_stderr: logs/slurm/pwm_phase1/pwm_p1_hopper_realenv_eval_smoke_9382749.err
```

Corrected W&B-disabled real-env eval smoke for final and true-best actors:

```text
slurm_job_id: 9382750
job_name: pwm_p1_hopper_realenv_eval_smoke2
main_repo_sha_at_submission: uncommitted evaluator diagnostic change on top of 2adeff0d11a8b8bb9c5cfd370bc453cb8caef496
baseline_pwm_sha: cb7b1689afbfd4b5662e43cdbae2c360e52d56a1
partition: gpu-rtx6000
qos: embers
account: gts-agarg35
gres: gpu:rtx_6000:1
wandb: disabled
env: dflex_hopper
config: baselines/PWM/scripts/outputs/2026-06-01/15-35-34/.hydra/config.yaml
num_games: 12
num_envs: 64
device: cuda:0
slurm_state: COMPLETED
exit_code: 0:0
elapsed: 00:00:25
start: 2026-06-01T18:32:35
end: 2026-06-01T18:33:00
slurm_stdout: logs/slurm/pwm_phase1/pwm_p1_hopper_realenv_eval_smoke2_9382750.out
slurm_stderr: logs/slurm/pwm_phase1/pwm_p1_hopper_realenv_eval_smoke2_9382750.err
```

Final actor real-env result:

```text
checkpoint: baselines/PWM/scripts/outputs/2026-06-01/15-35-34/logs/phase1_hopper_formal_seed0_20260601/final_policy.pt
output_dir: eval_results/pwm_phase1_hopper_realenv_final_smoke_20260601
return_mean: 1284.2498168945312
return_iqm: 1289.2972208658855
return_std: 36.90841641217852
return_min: 1213.7418212890625
return_max: 1342.860595703125
episode_length_mean: 1000.0
discounted_return_mean: 128.41964721679688
```

True-best actor real-env result:

```text
checkpoint: baselines/PWM/scripts/outputs/2026-06-01/15-35-34/logs/phase1_hopper_formal_seed0_20260601/best_policy.pt
output_dir: eval_results/pwm_phase1_hopper_realenv_best_smoke_20260601
return_mean: 1285.949198404948
return_iqm: 1282.025390625
return_std: 30.219568089728327
return_min: 1237.192138671875
return_max: 1338.37646484375
episode_length_mean: 1000.0
discounted_return_mean: 128.58977381388345
```

W&B-disabled real-env eval smoke for the raw local pretrained asset:

```text
slurm_job_id: 9382755
job_name: pwm_p1_hopper_asset_realenv_eval
partition: gpu-rtx6000
qos: embers
account: gts-agarg35
gres: gpu:rtx_6000:1
wandb: disabled
env: dflex_hopper
config: baselines/PWM/scripts/outputs/2026-06-01/15-35-34/.hydra/config.yaml
checkpoint: scripts/assets/pwm_hf/dflex/pretrained/PWM_HopperEnv.pt
num_games: 12
num_envs: 64
device: cuda:0
slurm_state: COMPLETED
exit_code: 0:0
elapsed: 00:00:15
start: 2026-06-01T18:34:09
end: 2026-06-01T18:34:23
slurm_stdout: logs/slurm/pwm_phase1/pwm_p1_hopper_asset_realenv_eval_9382755.out
slurm_stderr: logs/slurm/pwm_phase1/pwm_p1_hopper_asset_realenv_eval_9382755.err
output_dir: eval_results/pwm_phase1_hopper_realenv_asset_smoke_20260601
return_mean: 1220.9376831054688
return_iqm: 1226.5210367838542
return_std: 31.731999849854333
return_min: 1159.667236328125
return_max: 1259.6063232421875
episode_length_mean: 1000.0
discounted_return_mean: 122.0885378519694
```

Interpretation: the true real-env reward for the completed formal run matches the low training-loop plateau (`~1270`) rather than the local reference CSV scale (`~5650`). The raw `PWM_HopperEnv.pt` actor is also low in real-env reward, so this asset behaves like a bootstrap checkpoint for world-model/policy training rather than a paper-level final Hopper policy.

## Phase 1 Gate Decision

Original PWM parity is not established. The completed Hopper run is far below the local original Hopper reference scale recorded above (`final mean 5649.179`, `best mean 5712.018`). The real-env reward diagnostic confirms final and best actors are both around `1285` return, so the Phase 1 failure is not only an artifact of original `PWM.eval()` accumulating learned world-model reward.

Stop rule applied: do not port to MJLab and do not test Flow replacements yet.

Next Phase 1/2 debug target: determine why the original Hopper setup plateaus near `1285` when the local reference curve reaches `~5650`. The highest-priority checks are checkpoint provenance (`PWM_HopperEnv.pt` as bootstrap asset versus paper final policy), reward scaling/sign conventions, optimizer/resume behavior when loading `general.checkpoint`, and whether the local `baselines/PWM/results/data` reference CSV was produced by the same entrypoint/config/checkpoint path.

## Phase 1/2 Fidelity Debug: Environment Drift

After confirming that the resolved Hydra config matched the original Hopper/PWM config, the next audit target was the runtime dependency stack. This found a major fidelity violation: the active `pwm` conda environment is not the original PWM environment described by `baselines/PWM/environment.yaml`.

Original environment requirements:

```text
python: 3.10
pytorch: 2.3
torchvision: 0.18
pytorch-cuda: 11.8
cuda-toolkit: 11.8
tensordict: 0.4.*
torchrl: 0.4.*
dflex: git+https://github.com/imgeorgiev/DiffRL.git/#subdirectory=dflex
```

Observed active `pwm` environment:

```text
python: 3.10.19
torch import: 2.10.0+cu128
torch.version.cuda: 12.8
conda list torch: 2.9.0 pypi_0
tensordict: 0.11.0
torchrl: 0.11.0
pytorch-cuda: 11.8
cuda-version: 12.8
cuda-nvcc: 12.4.131
dflex direct_url commit: bb59db5cf65e63740787bf22f91bae3103b30d19
```

This is not a small patch-level drift. It changes PyTorch by multiple major/minor releases, TorchRL/TensorDict from `0.4` to `0.11`, and the CUDA toolchain from the intended `11.8` stack to a mixed `11.8/12.8` stack. The current formal and A/B parity runs therefore do not prove original PWM algorithm failure; they prove failure under a nonfaithful dependency environment.

Observed symptoms consistent with this drift:

```text
formal full-checkpoint Hopper run:
  final/true-best real-env return: ~1285
  reference CSV final scale: ~5650

full vs wm_only A/B at ~1.2M steps:
  full: ~1266 return
  wm_only: ~1265 return
  reference CSV around 1.024M steps: ~3227 mean

DFlex one-step probe job:
  slurm_job_id: 9382890
  status: FAILED
  failure: DFlex attempted to rebuild CUDA kernels on a new node and failed in CUDA headers
  reason: active env uses CUDA 12.8-era torch/toolchain despite original CUDA 11.8 requirement
```

Important checkpoint-load finding from the same audit:

```text
full checkpoint load:
  loads actor, critic, world_model, RMS, and optimizer states
  effective actor LR becomes 0.002 from checkpoint actor_opt
  config actor_lr remains 0.0005

wm_only checkpoint load:
  loads world_model and RMS only
  keeps actor LR at config value 0.0005
```

The full-load LR override is a real fidelity risk and was added as an explicit diagnostic mode, but the `wm_only` A/B also plateaus far below reference. Therefore the LR override is not sufficient to explain the parity failure.

Current diagnosis: Phase 1 is blocked by environment fidelity, not by MJLab transfer and not yet by Flow/PWM architecture. The next valid parity run must use a clean locked original stack, or at minimum:

```text
torch==2.3.x with CUDA 11.8
torchvision==0.18.x
tensordict==0.4.x
torchrl==0.4.x
dflex from imgeorgiev/DiffRL commit bb59db5cf65e63740787bf22f91bae3103b30d19
DFlex kernels rebuilt successfully under the same stack
```

No MJLab or Flow conclusions should be made from the current `pwm` env results.

Follow-up environment reconstruction attempt:

```text
command:
  conda create -y -p /storage/project/r-agarg35-0/eliu354/envs/pwm_orig23 \
    python=3.10 pytorch=2.3 torchvision=0.18 pytorch-cuda=11.8 cuda-toolkit=11.8 \
    pandas=2.2 matplotlib=3.8 seaborn=0.13 glew=2.1.0 \
    -c pytorch -c nvidia -c defaults

conda solve result:
  python: 3.10.20
  pytorch: 2.3.1 py3.10_cuda11.8_cudnn8.7.0_0
  pytorch-cuda: 11.8
  torchvision: 0.18.1 py310_cu118
  numpy: 1.26.4
  mkl: 2025.0.0
  llvm-openmp: 14.0.6
  cuda-version: 13.3
  cuda-nvcc: 13.3.33
  libnvjitlink: 13.3.33

torch import result:
  ImportError: libtorch_cpu.so: undefined symbol: iJIT_NotifyEvent
```

This reconstruction attempt confirms that the upstream `environment.yaml` pins are too loose for the current conda channel state. Even when the requested `pytorch` package is the intended `2.3.1` CUDA 11.8 build, the solve can still pull incompatible Intel/MKL/OpenMP and CUDA compiler-side packages. The next parity attempt needs an explicit lock file or additional pins for `mkl`, OpenMP/Intel runtime, `cuda-version`, `cuda-nvcc`, and related CUDA libraries before installing TorchRL/TensorDict/DFlex.

## Phase 1 Locked-Env Rerun Result

This section supersedes the earlier non-locked Phase 1 gate decision above. The low `~1285` true real-env returns were reproduced only under the drifted `pwm` environment. After reconstructing a locked original PWM runtime and forcing a clean DFlex sandbox rebuild, Hopper parity passes.

Locked runtime:

```text
env: /storage/project/r-agarg35-0/eliu354/envs/pwm_orig_locked4
python: 3.10.14
torch: 2.3.1 py3.10_cuda11.8_cudnn8.7.0_0
torchvision: 0.18.1 py310_cu118
pytorch-cuda: 11.8
cuda-toolkit: 11.8.0
cuda-version: 11.8
cuda-nvcc: 11.8.89
tensordict: 0.4.0
torchrl: 0.4.0
mkl: 2023.1.0
intel-openmp: 2023.1.0
dflex: imgeorgiev/DiffRL commit bb59db5cf65e63740787bf22f91bae3103b30d19
DFlex rebuild: passed in per-job sandbox with /usr/bin/gcc and /usr/bin/g++ 11.5
```

Locked W&B-disabled smoke:

```text
slurm_job_id: 9383756
partition: gpu-rtx6000
qos: embers
status: COMPLETED
exit_code: 0:0
command_core: python train_dflex.py env=dflex_hopper alg=pwm general.run_wandb=False general.seed=0 general.checkpoint_mode=wm_only general.checkpoint=scripts/assets/pwm_hf/dflex/pretrained/PWM_HopperEnv.pt alg.max_epochs=70 alg.save_interval=70 general.eval_runs=1 general.logdir=logs/phase1_hopper_smoke_locked_20260601
result: passed runtime/import/checkpoint/load/train/save gate
```

Locked formal Hopper seed:

```text
slurm_job_id: 9383814
main_repo_sha_at_submission: 95527e779a66978e614b688f8a5b48847052ae81
baseline_pwm_sha: c7ed70a01916eee9a5b1ebaa356365b852c19418
partition: gpu-h200
qos: embers
wandb_project: flow-mbpo-pwm-fidelity
wandb_group: phase1_original_hopper_locked_20260601
wandb_run_id: mx94exb3
wandb_url: https://wandb.ai/danny010324/flow-mbpo-pwm-fidelity/runs/mx94exb3
slurm_state: PREEMPTED
batch_exit_code: 0:0
elapsed: 01:38:33
node: atl1-1-03-020-18-0
command_core: python train_dflex.py env=dflex_hopper alg=pwm general.seed=0 general.run_wandb=True general.checkpoint_mode=wm_only general.checkpoint=scripts/assets/pwm_hf/dflex/pretrained/PWM_HopperEnv.pt general.eval_runs=12 general.logdir=logs/phase1_hopper_formal_locked_h200_s0_20260601 wandb.project=flow-mbpo-pwm-fidelity wandb.group=phase1_original_hopper_locked_20260601
slurm_stdout: logs/pwm_original_parity/locked_env_20260601/pwm_hopper_locked_h200_s0_9383814.out
slurm_stderr: logs/pwm_original_parity/locked_env_20260601/pwm_hopper_locked_h200_s0_9383814.err
```

Training result:

```text
last_logged_epoch: 14997 / 15000
last_logged_training_R: 5636.69
best_logged_training_R: 5662.93 at epoch 14841
best_policy_loss_wandb_summary: -5662.93066
checkpoint_dir: baselines/PWM/scripts/outputs/2026-06-01/20-27-50/logs/phase1_hopper_formal_locked_h200_s0_20260601
final_actor: baselines/PWM/scripts/outputs/2026-06-01/20-27-50/logs/phase1_hopper_formal_locked_h200_s0_20260601/final_policy.pt
true_best_actor: baselines/PWM/scripts/outputs/2026-06-01/20-27-50/logs/phase1_hopper_formal_locked_h200_s0_20260601/best_policy.pt
```

The Slurm state is `PREEMPTED`, but the batch step exited `0:0`, saved both final and best checkpoints, synced W&B successfully, and ran the original entrypoint final eval. The original `PWM.eval()` output remained misleading for this DFlex path (`mean episode loss = 36.33`, `mean episode length = 55.47`), so the separate real-env evaluator below is the parity metric used for the gate.

Locked real-env evaluator:

```text
slurm_job_id: 9385255
job_name: pwm_hopper_locked_realenv_eval_rtx6000_fix2
partition: gpu-rtx6000
qos: embers
status: COMPLETED
exit_code: 0:0
elapsed: 00:02:03
node: atl1-1-02-005-31-0
main_repo_sha_at_eval: 846cad45a0ad97c8cc674b5ee46bab223a9cc033
baseline_pwm_sha: c7ed70a01916eee9a5b1ebaa356365b852c19418
config: baselines/PWM/scripts/outputs/2026-06-01/20-27-50/.hydra/config.yaml
num_games: 40
num_envs: 64
device: cuda:0
dflex_eval_sandbox: /tmp/dflex_eval_sandbox_9385255/dflex
slurm_stdout: logs/pwm_original_parity/locked_env_20260601/pwm_hopper_locked_realenv_eval_rtx6000_fix2_9385255.out
slurm_stderr: logs/pwm_original_parity/locked_env_20260601/pwm_hopper_locked_realenv_eval_rtx6000_fix2_9385255.err
```

Real-env final actor result:

```text
checkpoint: baselines/PWM/scripts/outputs/2026-06-01/20-27-50/logs/phase1_hopper_formal_locked_h200_s0_20260601/final_policy.pt
output_dir: eval_results/pwm_phase1_hopper_locked_rtx6000_realenv_final_20260601
wandb_run_id: pjkf9bi3
wandb_url: https://wandb.ai/danny010324/flow-mbpo-pwm-fidelity/runs/pjkf9bi3
return_mean: 5665.566882324219
return_iqm: 5665.514379882812
return_std: 9.985102211515205
return_min: 5638.166015625
return_max: 5687.3232421875
episode_length_mean: 1000.0
discounted_return_mean: 407.0071846008301
```

Real-env true-best actor result:

```text
checkpoint: baselines/PWM/scripts/outputs/2026-06-01/20-27-50/logs/phase1_hopper_formal_locked_h200_s0_20260601/best_policy.pt
output_dir: eval_results/pwm_phase1_hopper_locked_rtx6000_realenv_best_20260601
wandb_run_id: dvhvr1gb
wandb_url: https://wandb.ai/danny010324/flow-mbpo-pwm-fidelity/runs/dvhvr1gb
return_mean: 5670.390954589844
return_iqm: 5669.7645263671875
return_std: 8.165196214570399
return_min: 5651.94580078125
return_max: 5690.13134765625
episode_length_mean: 1000.0
discounted_return_mean: 407.8704292297363
```

Wrapper/debug notes:

```text
formal job 9383776 failed before training because the wrapper deleted DFlex shared objects but left adjoint.gen; DFlex then reported "Using cached kernels" and failed importing kernels.
formal multi-GPU jobs 9383815/9383817 were canceled after partial training; 9383816 was canceled while pending once the H200 path was sufficient.
real-env eval jobs 9385192 and 9385235 exposed wrapper bugs: source ~/.bashrc under set -u, then missing export SANDBOX. Fixed in job 9385255.
```

## Updated Phase 1 Gate Decision

Original PWM parity is established for Hopper under the locked original stack. The resolved config and algorithm settings were not the root cause of the earlier parity failure; the main failure was runtime fidelity:

```text
primary cause: drifted active pwm environment, especially torch 2.10/cu128 + torchrl/tensordict 0.11 + mixed CUDA toolchain instead of torch 2.3.1/cu118 + torchrl/tensordict 0.4
secondary operational cause: DFlex kernel cache/sandbox invalidation mistakes during the first locked formal attempts
not supported by current evidence: true PWM implementation failure on original Hopper
not yet tested: MJLab transfer mismatch under faithful original PWM
```

Next gate: proceed to Phase 3 faithful PWM-on-MJLab with adapters only. Do not start Flow replacement rows until the faithful MJLab PWM smoke/formal/eval/video run is documented.

## 2026-06-02 Experiment Progress Check

Phase 1 supplemental Ant sanity completed on the locked original PWM stack:

```text
smoke_job_id: 9384321
smoke_status: COMPLETED
smoke_exit_code: 0:0
smoke_partition: gpu-h100
smoke_qos: embers
smoke_command_core: python train_dflex.py env=dflex_ant alg=pwm general.run_wandb=False general.seed=0 general.checkpoint_mode=wm_only general.checkpoint=scripts/assets/pwm_hf/dflex/pretrained/PWM_AntEnv.pt alg.max_epochs=80 general.eval_runs=4
smoke_stdout: logs/pwm_original_parity/locked_env_20260601/pwm_ant_smoke_locked_h100_s0_9384321.out
smoke_stderr: logs/pwm_original_parity/locked_env_20260601/pwm_ant_smoke_locked_h100_s0_9384321.err

formal_job_id: 9384344
formal_status: COMPLETED
formal_exit_code: 0:0
formal_partition: gpu-h100
formal_qos: embers
formal_elapsed: 01:54:11
formal_command_core: python train_dflex.py env=dflex_ant alg=pwm general.run_wandb=True general.seed=0 general.checkpoint_mode=wm_only general.checkpoint=scripts/assets/pwm_hf/dflex/pretrained/PWM_AntEnv.pt general.eval_runs=12
formal_stdout: logs/pwm_original_parity/locked_env_20260601/pwm_ant_formal_locked_h100_s0_9384344.out
formal_stderr: logs/pwm_original_parity/locked_env_20260601/pwm_ant_formal_locked_h100_s0_9384344.err
formal_wandb_run_id: 2epbv7y0
formal_wandb_url: https://wandb.ai/danny010324/flow-mbpo-pwm-fidelity/runs/2epbv7y0
formal_wandb_summary_rewards: 7439.96777
formal_wandb_summary_best_policy_loss: -7465.2085
formal_final_actor: baselines/PWM/scripts/outputs/2026-06-01/23-06-24/logs/phase1_ant_formal_locked_h100_s0_20260601/final_policy.pt
formal_true_best_actor: baselines/PWM/scripts/outputs/2026-06-01/23-06-24/logs/phase1_ant_formal_locked_h100_s0_20260601/best_policy.pt
formal_original_eval_tail: mean episode loss = 57.30, mean discounted loss = 46.57, mean episode length = 59.50
```

Interpretation: Ant provides a second original-task training sanity check and shows the locked stack can run another supported DFlex task to completion with high imagined/training reward. As with Hopper, the original `PWM.eval()` tail is not used as the decisive parity metric because it reports short learned-model loss/length. A true DFlex eval for Ant final/best has been queued before treating Ant as a strict real-env parity point.

Stale dependency diagnosis:

```text
stale_jobs_canceled: 9384354, 9384355, 9384374, 9384375, 9384400, 9384485
root_cause: jobs 9384354/9384355 used dependency=afterok:9383814, but Slurm recorded 9383814 top-level state as PREEMPTED even though its batch step completed 0:0 and wrote/evaluated checkpoints.
effect: 9384354/9384355 became DependencyNeverSatisfied; Phase 2 probes and Phase 3 MJLab adapter jobs remained pending behind that failed dependency chain.
prevention: do not use top-level afterok dependencies on a PREEMPTED-but-successful training job. Gate subsequent jobs on a completed eval job or on recorded artifacts plus a fresh explicit submission.
```

Resubmitted queue after manual Hopper parity gate:

```text
ant_true_dflex_eval_job_id: 9387422
ant_true_dflex_eval_job_name: pwm_ant_locked_realenv_eval_h200
ant_true_dflex_eval_partition: gpu-h200
ant_true_dflex_eval_qos: embers
ant_true_dflex_eval_status_at_submission: PENDING Priority
ant_true_dflex_eval_checkpoints: final_policy.pt and best_policy.pt from job 9384344
ant_true_dflex_eval_outputs:
  eval_results/pwm_phase1_ant_locked_h200_realenv_final_20260602
  eval_results/pwm_phase1_ant_locked_h200_realenv_best_20260602

phase2_probe_job_id: 9387423
phase2_probe_job_name: pwm_hopper_locked_wmprobe_h100
phase2_probe_partition: gpu-h100
phase2_probe_qos: embers
phase2_probe_status_at_submission: PENDING Priority
phase2_probe_inputs: Hopper final_policy.pt and best_policy.pt from job 9383814
phase2_probe_outputs:
  eval_results/pwm_phase2_hopper_locked_probe_20260602/final_actor_wm_vs_real.json
  eval_results/pwm_phase2_hopper_locked_probe_20260602/best_actor_wm_vs_real.json

phase3_smoke_job_id: 9387424
phase3_smoke_dependency: afterok:9387423
phase3_smoke_manifest: scripts/experiments/mjlab_qs/manifests/original_pwm_adapter_phase3_smoke_20260601.csv
phase3_smoke_partition: gpu-h100
phase3_smoke_qos: embers
phase3_smoke_wandb: disabled

phase3_formal_job_id: 9387425
phase3_formal_dependency: afterok:9387424
phase3_formal_manifest: scripts/experiments/mjlab_qs/manifests/original_pwm_adapter_phase3_formal_h200_seed0_20260601.csv
phase3_formal_partition: gpu-h200
phase3_formal_qos: embers
phase3_formal_wandb: enabled
```

Environment note: Phase 1/2 original DFlex jobs continue to use the locked original stack at `/storage/project/r-agarg35-0/eliu354/envs/pwm_orig_locked4` plus fresh per-job DFlex sandboxes. Phase 3 MJLab adapter jobs use the project MJLab runtime (`conda activate pwm`) because the locked DFlex-only parity environment is not the MJLab runtime; this is an IO-adapter phase, not a byte-identical upstream DFlex run.

## 2026-06-02 No-Dependency Resubmission

User instruction update: submit the experiments without waiting on Slurm
dependencies. The old dependency-gated Phase 3 jobs were canceled and replaced
with direct submissions.

```text
canceled_dependency_jobs:
  9387424 original PWM MJLab adapter smoke, dependency=afterok:9387423
  9387425 original PWM MJLab adapter formal, dependency=afterok:9387424

direct_phase3_smoke_job_id: 9387896
direct_phase3_smoke_job_name: mjqs_original_pwm_adapter_H100
direct_phase3_smoke_partition: gpu-h100
direct_phase3_smoke_qos: embers
direct_phase3_smoke_dependency: none
direct_phase3_smoke_status_at_check: PENDING Priority
direct_phase3_smoke_manifest: scripts/experiments/mjlab_qs/manifests/original_pwm_adapter_phase3_smoke_20260601.csv

direct_phase3_formal_job_id: 9387895
direct_phase3_formal_job_name: mjqs_original_pwm_adapter_H200
direct_phase3_formal_partition: gpu-h200
direct_phase3_formal_qos: embers
direct_phase3_formal_dependency: none
direct_phase3_formal_status_at_check: PENDING Priority
direct_phase3_formal_manifest: scripts/experiments/mjlab_qs/manifests/original_pwm_adapter_phase3_formal_h200_seed0_20260601.csv
```

The first no-dependency Ant eval / Hopper probe submissions failed due to a
DFlex sandbox wrapper bug, not due to model behavior:

```text
failed_ant_eval_job_id: 9387422
failed_hopper_probe_job_id: 9387423
failure: cp baselines/PWM/dflex failed because this repo does not contain that path; PYTHONPATH then did not point at a valid sandboxed dflex package, so Python imported env site-packages dflex and hit stale cached kernels.
error_signature: ImportError("No module named 'kernels'") after "Using cached kernels"
prevention:
  copy /storage/project/r-agarg35-0/eliu354/envs/pwm_orig_locked4/lib/python3.10/site-packages/dflex into the per-job sandbox
  put the sandbox parent directory on PYTHONPATH, not the sandbox/dflex package directory
  remove sandbox dflex/kernels before import so DFlex rebuilds kernels in the fresh sandbox
```

Fixed no-dependency replacements:

```text
ant_true_dflex_eval_fix_job_id: 9387942
ant_true_dflex_eval_fix_job_name: pwm_ant_locked_realenv_eval_h200_fix
ant_true_dflex_eval_fix_partition: gpu-h200
ant_true_dflex_eval_fix_qos: embers
ant_true_dflex_eval_fix_dependency: none
ant_true_dflex_eval_fix_status_at_check: PENDING Priority

phase2_probe_fix_job_id: 9387949
phase2_probe_fix_job_name: pwm_hopper_locked_wmprobe_h100_fix
phase2_probe_fix_partition: gpu-h100
phase2_probe_fix_qos: embers
phase2_probe_fix_dependency: none
phase2_probe_fix_status_at_check: PENDING Priority
```

## 2026-06-02 Preflight Inventory And Gate Results

Durable preflight inventory:

```text
docs/git/preflight_inventory_pwm_flow_sigreg_20260602.md
```

Current Slurm gate status after preflight:

```text
9387896_0 MJLab original PWM adapter smoke: COMPLETED, exit 0:0, elapsed 00:00:18
9387895_0 MJLab original PWM adapter formal: COMPLETED, exit 0:0, elapsed 00:49:21
9387942 Ant true DFlex eval replacement: FAILED, exit 1:0, elapsed 00:00:26
9387949 Hopper WM-vs-real probe replacement: FAILED, exit 1:0, elapsed 00:00:26
seff: unavailable on this shell
```

MJLab formal faithful adapter result:

```text
job_id: 9387895_0
git_sha_from_wandb: 88a4ca5b30a224f0df72ca4994b2ae19a480bf2a
wandb_run: https://wandb.ai/danny010324/flow-mbpo-mjlab-original-pwm-adapter/runs/17tlyzo2
dataset: scripts/outputs/mjlab_qs/windows/rerun_a25_native_qs_g1stage4_expertboost_20260527/velocity_flat_unitree_g1/d_qs_core_h16.pt
metadata: scripts/outputs/mjlab_qs/windows/rerun_a25_native_qs_g1stage4_expertboost_20260527/velocity_flat_unitree_g1/d_qs_core_h16.json
normalization: scripts/outputs/mjlab_qs/windows/rerun_a25_native_qs_g1stage4_expertboost_20260527/velocity_flat_unitree_g1/d_qs_core_h16_normalization.json
final_checkpoint: scripts/outputs/mjlab_qs/original_pwm_adapter/original_pwm_adapter_phase3_formal_20260601/velocity_flat_unitree_g1/normobs_normrew/seed_0/final_policy_extraction.pt
best_checkpoint: scripts/outputs/mjlab_qs/original_pwm_adapter/original_pwm_adapter_phase3_formal_20260601/velocity_flat_unitree_g1/normobs_normrew/seed_0/best_policy_extraction.pt
summary: scripts/outputs/mjlab_qs/original_pwm_adapter/original_pwm_adapter_phase3_formal_20260601/velocity_flat_unitree_g1/normobs_normrew/seed_0/summary.json
eval_summary: scripts/outputs/mjlab_qs/original_pwm_adapter/original_pwm_adapter_phase3_formal_20260601/velocity_flat_unitree_g1/normobs_normrew/seed_0/eval_summary.json
eval_episodes: 40
return_mean: -0.8009850382804871
episode_length_mean: 44.45000076293945
fall_rate: not logged by this adapter eval
best_imagined_return_proxy: 12.491907119750977
best_iter: 14984
```

Interpretation: the faithful MJLab adapter now has a completed formal seed and
checkpoints, but the 40-episode real return and episode length are far below the
expert collector (`82.6090`, length `1000`, fall `0`) and the best aggregate BC
reference (about `45.8491`, length about `594.97`, fall about `0.625`). The
rising imagined-return proxy plus short real episodes is a collapse/exploitation
signal, not a performance claim. Required next evidence is final/best checkpoint
eval with fall metrics plus final/best 10-episode, 1000-step MP4/W&B videos.

Original DFlex supplemental gate failures:

```text
9387942 failure: DFlex kernel rebuild failed because x86_64-conda-linux-gnu-c++ could not execute cc1plus in the locked env compiler path.
9387949 failure: same missing cc1plus / DFlex kernel rebuild failure before HopperEnv instantiation.
usable_as_algorithm_evidence: no
next_action: repair compiler/kernel-cache path and resubmit Ant true eval plus Hopper WM-vs-real probe.
```

Submitted required final/best evidence jobs:

```text
eval40_job_id: 9388552
eval40_kind: policy_eval
eval40_manifest: scripts/experiments/mjlab_qs/manifests/original_pwm_adapter_phase3_eval40_final_best_20260602.csv
eval40_rows: final_policy_extraction.pt, best_policy_extraction.pt
eval40_partition: gpu-h200
eval40_qos: embers
eval40_dependency: none
eval40_wandb_project: flow-mbpo-mjlab-original-pwm-adapter-eval40
eval40_output_root: scripts/outputs/mjlab_qs/policy_evals/original_pwm_adapter_phase3_eval40_20260602/velocity_flat_unitree_g1/normobs_normrew/seed_0

rollout10_job_id: 9388553
rollout10_kind: policy_rollout
rollout10_manifest: scripts/experiments/mjlab_qs/manifests/original_pwm_adapter_phase3_rollout10_final_best_20260602.csv
rollout10_rows: final_policy_extraction.pt, best_policy_extraction.pt
rollout10_partition: gpu-h200
rollout10_qos: embers
rollout10_dependency: none
rollout10_wandb_project: flow-mbpo-mjlab-original-pwm-adapter-rollout1000
rollout10_output_root: scripts/outputs/mjlab_qs/policy_rollouts/original_pwm_adapter_phase3_rollout10_20260602/velocity_flat_unitree_g1/normobs_normrew/seed_0
```

Repaired original-DFlex gate replacement submissions:

```text
ant_true_dflex_eval_fix2_job_id: 9388605
ant_true_dflex_eval_fix2_job_name: pwm_ant_locked_realenv_eval_h200_fix2
ant_true_dflex_eval_fix2_partition: gpu-h200
ant_true_dflex_eval_fix2_qos: embers
ant_true_dflex_eval_fix2_dependency: none
ant_true_dflex_eval_fix2_inputs:
  baselines/PWM/scripts/outputs/2026-06-01/23-06-24/logs/phase1_ant_formal_locked_h100_s0_20260601/final_policy.pt
  baselines/PWM/scripts/outputs/2026-06-01/23-06-24/logs/phase1_ant_formal_locked_h100_s0_20260601/best_policy.pt
ant_true_dflex_eval_fix2_outputs:
  eval_results/pwm_phase1_ant_locked_h200_realenv_final_20260602
  eval_results/pwm_phase1_ant_locked_h200_realenv_best_20260602
ant_true_dflex_eval_fix2_wandb_project: flow-mbpo-pwm-fidelity

hopper_wmprobe_fix2_job_id: 9388606
hopper_wmprobe_fix2_job_name: pwm_hopper_locked_wmprobe_h100_fix2
hopper_wmprobe_fix2_partition: gpu-h100
hopper_wmprobe_fix2_qos: embers
hopper_wmprobe_fix2_dependency: none
hopper_wmprobe_fix2_inputs:
  baselines/PWM/scripts/outputs/2026-06-01/20-27-50/logs/phase1_hopper_formal_locked_h200_s0_20260601/final_policy.pt
  baselines/PWM/scripts/outputs/2026-06-01/20-27-50/logs/phase1_hopper_formal_locked_h200_s0_20260601/best_policy.pt
hopper_wmprobe_fix2_outputs:
  eval_results/pwm_phase2_hopper_locked_probe_20260602/final_actor_wm_vs_real.json
  eval_results/pwm_phase2_hopper_locked_probe_20260602/best_actor_wm_vs_real.json
```

Wrapper repair applied to both jobs: use the locked env and a fresh job-local
DFlex sandbox, set `CC=/usr/bin/gcc`, `CXX=/usr/bin/g++`,
`CUDAHOSTCXX=/usr/bin/g++`, and unset `C_INCLUDE_PATH`, `CPLUS_INCLUDE_PATH`,
`LIBRARY_PATH`, `GCC_EXEC_PREFIX`, and `COMPILER_PATH`. This matches the
successful locked DFlex rebuild path and avoids the conda compiler wrapper that
could not execute `cc1plus` in jobs `9387942` and `9387949`.

## 2026-06-02 Matched Existing Evidence Inventory

Durable matched-evidence inventory:

```text
docs/git/flow_pwm_matched_evidence_inventory_20260602.md
```

Scheduler status at this inventory check:

```text
9388552_[0-1] MJLab faithful original PWM final/best eval40: PENDING Priority
9388553_[0-1] MJLab faithful original PWM final/best rollout10 video: PENDING Priority
9388605 Ant locked DFlex final/best true eval repair: PENDING Priority
9388606 Hopper locked DFlex WM-vs-real probe repair: PENDING Priority
logs_for_these_jobs: none found yet
```

Current conservative table from existing artifacts:

| Row | Evidence | Return | Length | Fall | Interpretation |
| --- | --- | ---: | ---: | ---: | --- |
| Faithful original PWM adapter | Formal seed0 summary only; final/best fall/video jobs pending. | `-0.8010` | `44.45` | missing | Collapse-like, but incomplete gate. |
| Prior PWM-style runner, MLP WM + MLP policy | Old 2x2 rollout aggregate, seeds 0-2. | `-4.5700` | `66.33` | `1.000` | Failed. |
| Flow policy only | Old 2x2 rollout aggregate, seeds 0-2. | `-3.4818` | `66.78` | `1.000` | Failed. |
| Flow WM only | Old 2x2 rollout aggregate, seeds 0-2. | `-4.7720` | `60.33` | `1.000` | Failed. |
| Flow WM + Flow policy | Old 2x2 rollout aggregate, seeds 0-2. | `-3.5414` | `85.33` | `0.889` | Failed despite slightly longer episodes. |
| Flow-MBPO trajectory/chunk H3 | Seed0 eval40 final. | `48.7296` | `637.22` | `0.575` | Promising diagnostic only. |
| Flow-MBPO trajectory/chunk H3 | Seed0 rollout10 final. | `54.4904` | `694.00` | `0.400` | Matches BC fall but does not improve it. |
| Matched seed0 BC video | Seed0 rollout10 final. | `54.1283` | `688.40` | `0.400` | Current video/fall comparator. |
| Best aggregate BC | 40-episode eval aggregate. | about `45.8491` | about `594.97` | about `0.625` | Formal BC reference. |
| Expert collector | Collector rollout. | `82.6090` | `1000.00` | `0.000` | Target reference. |

Interpretation: existing Flow-MBPO rows are promising but still unverified as
policy improvement because strict matched video/fall evidence is not cleared and
the faithful original PWM comparator is missing final/best fall/video artifacts.
The old 2x2 PWM/Flow replacement rows are already failed diagnostics, not a
basis for expanding a one-variable R0-R4 matrix before the pending faithful PWM
evidence package lands.
