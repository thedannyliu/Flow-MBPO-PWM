# Handoff Prompt for a Coding Agent on Another Server

Use this prompt verbatim with the coding agent on the other server.

---

You are continuing work in:
`/storage/project/r-agarg35-0/eliu354/projects/Flow-MBPO-PWM`

## Context You Must Assume Is Already Done

1. Legacy scripts were removed from `scripts/legacy/`.
2. Active single-task online RL manifests were regenerated.
3. The direct mjlab task set was replaced with:
   - `leap_left_grasp_asymmetric` -> `mjlab_leap_left_grasp_asymmetric`
   - `tracking_rough_unitree_g1` -> `mjlab_tracking_rough_unitree_g1`
   - `leap_left_inhand_pen_twirl` -> `mjlab_leap_left_inhand_pen_twirl`
4. Proxy tasks remain:
   - `anymal` -> `mjlab_velocity_flat_unitree_go2`
   - `snu_humanoid` -> `mjlab_velocity_flat_unitree_g1`

## Primary Goal

Continue single-task online RL execution and monitoring with the updated mjlab task panel, while preserving fair Flow-vs-MLP comparison and single-GPU-per-run constraints.

## Active Source-of-Truth Files

- Manifest builder:
  - `scripts/experiments/single_task_online/build_manifest.py`
- Cluster splitter:
  - `scripts/experiments/single_task_online/split_manifest_by_cluster.py`
- Runner:
  - `scripts/experiments/single_task_online/run_manifest_job.py`
- Submitters:
  - `scripts/experiments/single_task_online/submit_manifest_array.sh`
  - `scripts/experiments/single_task_online/submit_manifest_packed_array.sh`
- Eval:
  - `scripts/eval/eval_online_single_task.py`
- Updated env configs:
  - `scripts/cfg/env/mjlab_leap_left_grasp_asymmetric.yaml`
  - `scripts/cfg/env/mjlab_tracking_rough_unitree_g1.yaml`
  - `scripts/cfg/env/mjlab_leap_left_inhand_pen_twirl.yaml`
- Summary of latest changes:
  - `docs/recent_work_summary_20260301.md`

## Current Manifest Files (Already Regenerated)

- `scripts/experiments/single_task_online/manifests/smoke_required_all_v2_20260228.csv`
- `scripts/experiments/single_task_online/manifests/pilot_required_seed0_default_v2_20260228.csv`
- `scripts/experiments/single_task_online/manifests/pilot_quickcheck_required_seed0_v2_20260228.csv`

And split outputs:
- `*_pace_ice.csv`
- `*_pace_phoenix.csv`

## Cluster Assignment Rules (Do Not Violate)

- Keep each `task_key` on one cluster only.
- No cross-cluster checkpoint sharing.

PACE-ICE:
- `humanoid`
- `snu_humanoid`
- `leap_left_grasp_asymmetric`
- `tracking_rough_unitree_g1`
- `leap_left_inhand_pen_twirl`

PACE-Phoenix:
- `hopper`
- `ant`
- `anymal`

## Required Execution Pattern

1. Submit manifests with strict eval and rollout artifacts enabled:

```bash
STRICT_EVAL_VIDEO=1 ENABLE_ROLLOUT_VIDEO=1 RESUME_IF_EXISTS=1 \
bash scripts/experiments/single_task_online/submit_manifest_array.sh \
  --manifest <manifest_path> \
  --gpu-type <H100|H200|L40S> \
  --max-concurrent <N> \
  --cpus 8 \
  --time <HH:MM:SS>
```

2. Use packed mode only for smoke/quickcheck when memory headroom is confirmed.

## Validation Checklist Per Run

1. Training finishes without crash.
2. `best_policy.pt` and `final_policy.pt` exist.
3. Eval artifacts exist:
   - `eval_summary.json`
   - `episode_metrics.csv`
   - `rollout_steps.csv`
   - `rollout_summary.csv`
4. Rollout media exists (`rollout.mp4` or `rollout.gif`).
5. WandB train/eval runs are synced with correct task/method tags.

## Reporting Format Back to User

Return:

1. Submitted job IDs
2. Completed / failed counts by manifest
3. Failed `run_key` list with root causes
4. Retries submitted (if any) with new job IDs
5. Any mismatch between manifest task panel and observed runtime task IDs

---
