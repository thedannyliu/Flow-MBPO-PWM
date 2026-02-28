# Recent Work Summary (Single-Task Online RL)

Date: February 28, 2026

This document summarizes the recent implementation and operations changes for the
single-task online RL campaign across PACE-ICE and PACE-Phoenix.

## 1. Pipeline Stabilization

- Added robust resume/load behavior for PyTorch 2.6 checkpoint loading.
- Added end-of-training checkpoint pruning so only best/final checkpoints remain.
- Enforced headless eval defaults and strict rollout media checks.
- Added rollout video fallback (`mp4 -> gif`) and W&B eval artifact logging.

Primary files:
- `src/flow_mbpo_pwm/algorithms/pwm.py`
- `scripts/experiments/single_task_online/run_manifest_job.py`
- `scripts/experiments/single_task_online/submit_manifest_array.sh`
- `scripts/eval/eval_online_single_task.py`

## 2. Environment Adapter Updates

- Added Gymnasium MuJoCo adapter for Hopper/Ant/Humanoid single-task runs.
- Extended MJLab adapter with render passthrough and tracking motion-file resolution.
- Added local tracking motion artifact to remove external dependency during smoke/pilot.

Primary files:
- `src/flow_mbpo_pwm/envs/gymnasium_pwm_adapter.py`
- `src/flow_mbpo_pwm/envs/mjlab_pwm_adapter.py`
- `scripts/assets/motions/g1_tracking_dummy_motion.npz`
- `scripts/cfg/env/mjlab_tracking_flat_unitree_g1.yaml`

## 3. Manifest + Cluster Orchestration

- Active manifests were consolidated for smoke/pilot/quickcheck.
- Added deterministic split tool to pin each task to exactly one cluster.
- Added explicit dual-cluster plan document and handoff prompt.

Primary files:
- `scripts/experiments/single_task_online/build_manifest.py`
- `scripts/experiments/single_task_online/split_manifest_by_cluster.py`
- `docs/dual_cluster_execution_plan_ice_phoenix_20260228.md`
- `docs/phoenix_agent_handoff_prompt_20260228.md`

## 4. One-GPU Multi-Experiment Support

- Added packed execution mode to run multiple light manifest rows concurrently on one GPU.
- This is intended for smoke/quickcheck acceleration on ICE high-risk tasks.

Primary files:
- `scripts/experiments/single_task_online/run_manifest_pack.py`
- `scripts/experiments/single_task_online/submit_manifest_packed_array.sh`

## 5. Scripts Cleanup

- Kept active pipeline scripts in top-level `scripts/`.
- Moved historical scripts into `scripts/legacy/` to reduce operational noise.
- Added index documents for active/legacy script paths.

Primary files:
- `scripts/README.md`
- `scripts/legacy/README.md`

## 6. Current Server Allocation Policy

- PACE-ICE: high-risk tasks
  - `humanoid`, `snu_humanoid`, `velocity_flat_unitree_g1`,
    `tracking_flat_unitree_g1`, `leap_left_handcube_rotate`
- PACE-Phoenix: throughput-stable tasks
  - `hopper`, `ant`, `anymal`, `velocity_flat_unitree_go2`

Constraints:
- One task family is pinned to one cluster.
- No cross-cluster resume/checkpoint sharing.
- Method comparisons for the same task stay on the same cluster.

See: `docs/dual_cluster_execution_plan_ice_phoenix_20260228.md`.
