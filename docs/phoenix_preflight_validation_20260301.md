# PACE-Phoenix Preflight Validation (2026-03-01)

## Scope
This preflight validates the single-task online RL production path on PACE-Phoenix (L40S):

- End-to-end `train -> eval -> rollout artifact -> W&B`.
- Auto-resume from existing checkpoint (`RESUME_IF_EXISTS=1`).
- Checkpoint pruning policy (keep only `best_policy.*` and `final_policy.*`).
- Packed execution (`run_manifest_pack.py`) with multiple rows on one GPU.

## Environment
- Cluster: `pace-phoenix` (login host: `login-phoenix-gnr-3.pace.gatech.edu`)
- GPU partition: `gpu-l40s`
- Slurm account: `gts-agarg35-ideas_l40s`
- Conda env: `pwm`
- W&B project (new): `flow-mbpo-phoenix-preflight-20260301`

## Preflight Manifests
Created under `scripts/experiments/single_task_online/manifests/`:

- `preflight_smoke_phoenix_20260301.csv` (3 rows: hopper/ant/anymal)
- `preflight_resume_phase1_phoenix_20260301.csv` (1 row)
- `preflight_resume_phase2_phoenix_20260301.csv` (same run_key, higher max_epochs)
- `preflight_pack_phoenix_20260301.csv` (2 rows for packed run)

## Submitted Jobs and Result

### 1) Smoke (E2E)
- Submit script: `submit_manifest_array.sh`
- Job ID: `4429372`
- Status: **PASS** (rows 0/1/2 all COMPLETED)
- Verified from logs:
  - `Completed training + evaluation` present for all rows.
  - W&B train/eval runs were created in project `flow-mbpo-phoenix-preflight-20260301`.
  - Eval artifacts created per run: `eval_summary.json`, `episode_metrics.csv`, `rollout_steps.csv`, `rollout_summary.csv`, `rollout.gif`.
  - Checkpoint directory contains only `best_policy.pt` and `final_policy.pt`.

### 2) Resume (phase1 -> phase2)
- Phase1 Job ID: `4429813`
- Phase2 Job ID: `4429909`
- Status: **PASS**
- Verified from phase2 log:
  - Resume message exists:
    - `Resuming run preflight_resume_gym_hopper_mlpwm_mlppolicy_s7_default from checkpoint=.../final_policy.pt`
  - Train command includes:
    - `general.resume_training=true`
    - `general.checkpoint=.../final_policy.pt`
  - Run completed with eval and artifacts.

### 3) Packed mode (2 rows on one GPU)
- Submit script: `submit_manifest_packed_array.sh`
- Job ID: `4429941`
- Status: **PASS**
- Verified from log:
  - Two rows launched concurrently on one GPU (`max_parallel=2`).
  - Both rows finished with `rc=0`.
  - Summary reports: `succeeded=2 failed=0`.
  - Both output dirs have `best_policy.pt`, `final_policy.pt`, eval metrics, and rollout GIF.

## Notes / Observed Warnings

1. MP4 backend is unavailable in current runtime.
   - Evaluation falls back to GIF successfully.
   - Message seen: `MP4 video export skipped ... install imageio[ffmpeg] or imageio[pyav]`.
2. EGL warnings are present in stderr (`/dev/dri/card* Permission denied`), but rendering and training still complete via current setup.

## Formal Training Launch Plan on Phoenix

Use Phoenix for throughput tasks only: `hopper`, `ant`, `anymal`.

### Option A: Start from existing Phoenix pilot manifest
```bash
STRICT_EVAL_VIDEO=1 ENABLE_ROLLOUT_VIDEO=1 RESUME_IF_EXISTS=1 \
  scripts/experiments/single_task_online/submit_manifest_array.sh \
  --manifest scripts/experiments/single_task_online/manifests/pilot_required_seed0_default_v2_20260228_pace_phoenix.csv \
  --gpu-type L40S --max-concurrent 6 --cpus 8 --mem 96G --time 24:00:00
```

### Option B: Regenerate Phoenix pilot sweep (if you want fresh project naming)
```bash
python scripts/experiments/single_task_online/build_manifest.py \
  --stage pilot \
  --tasks hopper,ant,anymal \
  --methods mlpwm_mlppolicy,flowwm_mlppolicy \
  --wandb-project flow-mbpo-single-task-online-pilot-phoenix-20260301 \
  --output scripts/experiments/single_task_online/manifests/pilot_phoenix_custom_20260301.csv

STRICT_EVAL_VIDEO=1 ENABLE_ROLLOUT_VIDEO=1 RESUME_IF_EXISTS=1 \
  scripts/experiments/single_task_online/submit_manifest_array.sh \
  --manifest scripts/experiments/single_task_online/manifests/pilot_phoenix_custom_20260301.csv \
  --gpu-type L40S --max-concurrent 6 --cpus 8 --mem 96G --time 24:00:00
```

## Remaining Validation Before Large Confirm Sweeps

- Install and validate MP4 backend if MP4 is required instead of GIF fallback.
- Run one longer-duration pilot row (>= 1000 epochs) as a stability check on Phoenix L40S.
- Confirm W&B artifact retention policy to avoid excessive storage growth during large sweeps.
