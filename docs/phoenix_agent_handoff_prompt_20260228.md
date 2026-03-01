# Handoff Prompt for the Other Server Coding Agent (PACE-Phoenix)

Use the following prompt as-is for the coding agent on the other server.

---

You are continuing work in `/storage/project/r-agarg35-0/eliu354/projects/Flow-MBPO-PWM`.

## Mission

Run the **PACE-Phoenix** portion of the single-task online RL campaign
(`single GPU per run`, mostly `L40S`) for:

- `hopper`
- `ant`
- `anymal` (proxy via `mjlab_velocity_flat_unitree_go2`)

Do **not** run tasks assigned to PACE-ICE.

## Critical Constraints

1. Do not split a task across clusters.
2. Do not rely on cross-cluster resume/checkpoint sharing.
3. Keep method comparisons fair:
   - `mlpwm_mlppolicy` and `flowwm_mlppolicy` for the same task must stay on Phoenix.
4. Keep strict eval video checks enabled:
   - `STRICT_EVAL_VIDEO=1`, `ENABLE_ROLLOUT_VIDEO=1`.
5. Keep resume enabled:
   - `RESUME_IF_EXISTS=1`.
6. Do not use packed mode for full pilot/confirm by default on Phoenix.

## Source of Truth Files

- Cluster split plan:
  - `docs/dual_cluster_execution_plan_ice_phoenix_20260228.md`
- Recent implementation changes:
  - `docs/single_task_online_recent_changes_20260228.md`
- Active pipeline:
  - `scripts/experiments/single_task_online/README.md`
  - `scripts/experiments/single_task_online/run_manifest_job.py`
  - `scripts/experiments/single_task_online/submit_manifest_array.sh`
  - `scripts/experiments/single_task_online/submit_manifest_packed_array.sh` (optional, smoke only)
  - `scripts/eval/eval_online_single_task.py`

## Required Commands

1. Build/split manifests (if needed):

```bash
python scripts/experiments/single_task_online/split_manifest_by_cluster.py \
  --manifest scripts/experiments/single_task_online/manifests/pilot_required_seed0_default_v2_20260228.csv
```

2. Submit Phoenix pilot runs:

```bash
STRICT_EVAL_VIDEO=1 ENABLE_ROLLOUT_VIDEO=1 RESUME_IF_EXISTS=1 \
bash scripts/experiments/single_task_online/submit_manifest_array.sh \
  --manifest scripts/experiments/single_task_online/manifests/pilot_required_seed0_default_v2_20260228_pace_phoenix.csv \
  --gpu-type L40S \
  --max-concurrent 8 \
  --cpus 8 \
  --time 24:00:00
```

## Validation Checklist

For each completed run:

1. `Completed training + evaluation` appears in Slurm out log.
2. `scripts/outputs/single_task_online/.../logs/` contains:
   - `best_policy.pt`
   - `final_policy.pt`
3. Eval output contains rollout media:
   - `rollout.mp4` or `rollout.gif`
4. W&B train + eval runs are both present and synced.

## Reporting Format

When done, report:

1. Slurm job IDs
2. Completed/failed counts per manifest
3. Failed rows with exact `run_key` and root cause
4. Whether retries were submitted and their job IDs

---
