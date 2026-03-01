# Dual-Cluster Execution Plan (PACE-ICE + PACE-Phoenix)

Date: February 28, 2026

## Goal

Run large single-task online RL sweeps with stable throughput while keeping
high-risk tasks on the cluster where they were already validated.

## Cluster Characteristics

- **PACE-ICE**
  - GPUs: mostly `H100`, `H200`
  - stronger per-GPU performance
  - less stable queue/resource availability due to heavier student usage

- **PACE-Phoenix**
  - GPUs: mostly `L40S`
  - more stable queue behavior
  - lower per-GPU peak than H100/H200, but good for steady throughput

## Fixed Task-to-Cluster Assignment

Tasks are pinned to one cluster only (no cross-cluster resume/checkpoint merge).

### PACE-ICE (high-risk / fragile tasks)

- `humanoid`
- `snu_humanoid` (proxy: `mjlab_velocity_flat_unitree_g1`)
- `leap_left_grasp_asymmetric`
- `tracking_rough_unitree_g1`
- `leap_left_inhand_pen_twirl`

### PACE-Phoenix (throughput-oriented tasks)

- `hopper`
- `ant`
- `anymal` (proxy: `mjlab_velocity_flat_unitree_go2`)

## Why This Split

- ICE has stronger single-GPU performance (`H100/H200`) for fragile tasks that
  are more likely to fail due to rendering, motion assets, or unstable early dynamics.
- Phoenix is usually queue-stable and is better for predictable throughput runs on `L40S`.
- This split also keeps resume logic simple: one task family, one cluster, one checkpoint lineage.

## One-GPU Multi-Experiment Strategy (Packed Mode)

For ICE high-risk tasks, use packed mode only when per-run memory is small enough.

- Script:
  - `scripts/experiments/single_task_online/submit_manifest_packed_array.sh`
- Behavior:
  - one Slurm task requests one GPU
  - inside that task, `run_manifest_pack.py` can run multiple manifest rows concurrently
  - configurable via `--runs-per-gpu`

Recommended settings:

- Smoke / quick-check:
  - `runs_per_gpu=2`, `pack_size=2`
- Full pilot / confirm:
  - start from `runs_per_gpu=1`
  - only increase if memory headroom remains stable

## Submission Policy

All jobs remain **single-GPU per run**. Concurrency comes from job arrays.

- **PACE-ICE**
  - preferred GPU: `H200` (fallback `H100`)
  - conservative array concurrency due queue volatility
  - recommended array limit: `2-4` concurrent jobs
  - optional packed mode for smoke/quick-check (`2 runs/GPU`)

- **PACE-Phoenix**
  - GPU: `L40S`
  - aggressive steady concurrency
  - recommended array limit: `6-12` concurrent jobs
  - usually keep `1 run/GPU` for pilot+ to avoid long-tail contention

## Fairness Rule for Method Comparison

For each `task_key`, all method variants (`mlpwm_mlppolicy`, `flowwm_mlppolicy`)
must run on the same cluster.

## Operational Rule for Resume

Since resume/checkpoint sharing will not be done across clusters:

- Each task family always stays on its assigned cluster.
- Resume uses only local cluster outputs for that task.

## Commands

### 1) Split one manifest into ICE/Phoenix manifests

```bash
python scripts/experiments/single_task_online/split_manifest_by_cluster.py \
  --manifest scripts/experiments/single_task_online/manifests/pilot_required_seed0_default_v2_20260228.csv
```

### 2) Submit on PACE-ICE

```bash
STRICT_EVAL_VIDEO=1 ENABLE_ROLLOUT_VIDEO=1 RESUME_IF_EXISTS=1 \
bash scripts/experiments/single_task_online/submit_manifest_array.sh \
  --manifest scripts/experiments/single_task_online/manifests/pilot_required_seed0_default_v2_20260228_pace_ice.csv \
  --gpu-type H200 \
  --max-concurrent 4 \
  --cpus 8 \
  --time 24:00:00
```

### 2b) Submit packed rows on one ICE GPU (smoke/quick-check)

```bash
STRICT_EVAL_VIDEO=1 ENABLE_ROLLOUT_VIDEO=1 RESUME_IF_EXISTS=1 \
bash scripts/experiments/single_task_online/submit_manifest_packed_array.sh \
  --manifest scripts/experiments/single_task_online/manifests/smoke_required_all_v2_20260228_pace_ice.csv \
  --gpu-type H100 \
  --pack-size 2 \
  --runs-per-gpu 2 \
  --max-concurrent-jobs 2 \
  --cpus 16 \
  --time 08:00:00
```

### 3) Submit on PACE-Phoenix

```bash
STRICT_EVAL_VIDEO=1 ENABLE_ROLLOUT_VIDEO=1 RESUME_IF_EXISTS=1 \
bash scripts/experiments/single_task_online/submit_manifest_array.sh \
  --manifest scripts/experiments/single_task_online/manifests/pilot_required_seed0_default_v2_20260228_pace_phoenix.csv \
  --gpu-type L40S \
  --max-concurrent 8 \
  --cpus 8 \
  --time 24:00:00
```

## Monitoring Checklist

- Slurm:
  - `squeue -u $USER`
  - `tail -f logs/slurm/single_task_online/<stage>/*.out`
- W&B:
  - train and eval runs created
  - eval artifacts include rollout video (`rollout.mp4` or `rollout.gif`)
- Outputs:
  - `logs/best_policy.pt`, `logs/final_policy.pt` exist
  - no extra checkpoint files after successful completion
