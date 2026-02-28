# Single-Task Online RL (PACE-ICE + PACE-Phoenix)

This directory is the active pipeline for the single-task online RL comparison
(`MLP world model` vs `Flow world model`) with one GPU per run.

## Active Tasks

- `hopper`
- `ant`
- `anymal` (proxy via `mjlab_velocity_flat_unitree_go2`)
- `humanoid`
- `snu_humanoid` (proxy via `mjlab_velocity_flat_unitree_g1`)
- `velocity_flat_unitree_go2`
- `velocity_flat_unitree_g1`
- `tracking_flat_unitree_g1`
- `leap_left_handcube_rotate`

## Core Files

- `build_manifest.py`: build stage/task/method manifests
- `split_manifest_by_cluster.py`: split one manifest into fixed ICE/Phoenix manifests
- `submit_manifest_array.sh`: submit one manifest as Slurm array (single GPU per row)
- `submit_manifest_packed_array.sh`: submit packed rows (single GPU per Slurm task, up to N rows in parallel)
- `run_manifest_job.py`: row runner (`train -> eval -> rollout artifacts`)
- `run_manifest_pack.py`: pack runner used by packed submission mode
- `submit_profile_single.sh`: profile one row (`nsys` when available)

## Active Manifests

- `manifests/smoke_required_all_v2_20260228.csv`
- `manifests/pilot_required_seed0_default_v2_20260228.csv`
- `manifests/pilot_quickcheck_required_seed0_v2_20260228.csv`

## Workflow

1. Build or update a manifest:

```bash
python scripts/experiments/single_task_online/build_manifest.py \
  --stage smoke \
  --tasks hopper,ant \
  --methods mlpwm_mlppolicy,flowwm_mlppolicy \
  --output scripts/experiments/single_task_online/manifests/smoke_tmp.csv
```

2. Split to fixed cluster manifests (no cross-cluster task split):

```bash
python scripts/experiments/single_task_online/split_manifest_by_cluster.py \
  --manifest scripts/experiments/single_task_online/manifests/pilot_required_seed0_default_v2_20260228.csv
```

3. Submit on each cluster:

```bash
# PACE-ICE
scripts/experiments/single_task_online/submit_manifest_array.sh \
  --manifest scripts/experiments/single_task_online/manifests/pilot_required_seed0_default_v2_20260228_pace_ice.csv \
  --gpu-type H200 --max-concurrent 4 --cpus 8 --time 24:00:00

# PACE-Phoenix
scripts/experiments/single_task_online/submit_manifest_array.sh \
  --manifest scripts/experiments/single_task_online/manifests/pilot_required_seed0_default_v2_20260228_pace_phoenix.csv \
  --gpu-type L40S --max-concurrent 8 --cpus 8 --time 24:00:00
```

4. Monitor:

```bash
squeue -u "$USER"
tail -f logs/slurm/single_task_online/*/*.out
```

## Packed Mode (One GPU, Multiple Rows)

Use packed mode when a single row is light enough to share one GPU safely.
Recommended first use is smoke/pilot-quickcheck for high-risk ICE tasks.

```bash
scripts/experiments/single_task_online/submit_manifest_packed_array.sh \
  --manifest scripts/experiments/single_task_online/manifests/smoke_required_all_v2_20260228_pace_ice.csv \
  --gpu-type H100 \
  --pack-size 2 \
  --runs-per-gpu 2 \
  --max-concurrent-jobs 2 \
  --cpus 16 \
  --time 08:00:00
```

Practical defaults:
- `runs_per_gpu=2` for smoke/quickcheck
- `runs_per_gpu=1` for full pilot/confirm if memory pressure appears

## Runtime Guarantees

- Train/eval runs are WandB-logged with task/method/stage tags.
- Eval runs attempt rollout video (`.mp4`) and fallback to `.gif`.
- Resume is automatic if `latest/final/best` checkpoints exist.
- End-of-training checkpoint pruning keeps only `best_policy.*` and `final_policy.*`.
