# Single-Task Online RL: Recent Changes (February 28, 2026)

This document summarizes the recent implementation changes made to stabilize
single-task online RL runs on PACE clusters.

## 1. Training Resume Reliability

- Fixed PyTorch 2.6 checkpoint loading compatibility in PWM:
  - `torch.load(..., weights_only=False)` is now used in resume/load paths.
- Resume metadata (`iter_count`, `step_count`, optimizer state, replay state)
  can now be restored without `UnpicklingError`.

Primary file:
- `src/flow_mbpo_pwm/algorithms/pwm.py`

## 2. Checkpoint Retention Policy

- Added end-of-training pruning to keep only:
  - `best_policy.pt`
  - `final_policy.pt`
  - optional `best_policy.buffer`, `final_policy.buffer`
- Intermediate checkpoint artifacts (for example `latest_checkpoint.pt`) are
  removed after successful completion.
- Resume remains supported for interrupted runs before pruning happens.

Primary file:
- `src/flow_mbpo_pwm/algorithms/pwm.py`

## 3. Headless Evaluation + Video Logging

- Enforced headless rendering defaults for Slurm nodes:
  - `MUJOCO_GL=egl`
  - `PYOPENGL_PLATFORM=egl`
  - `EGL_PLATFORM=surfaceless`
- Evaluation video export now supports fallback:
  - tries `rollout.mp4`
  - if unavailable, saves `rollout.gif`
- W&B eval logging now records available rollout media and includes it in
  evaluation artifacts.
- Added strict mode (`STRICT_EVAL_VIDEO=1`) validation in row runner.

Primary files:
- `scripts/experiments/single_task_online/submit_manifest_array.sh`
- `scripts/experiments/single_task_online/run_manifest_job.py`
- `scripts/eval/eval_online_single_task.py`

## 4. MJLab Adapter Improvements

- Added `render()` passthrough in `MJLabPWMAdapter` for rollout capture.
- Added tracking motion-file resolution:
  - uses `env.config.mjlab_env_kwargs.motion_file` when provided
  - falls back to `MJLAB_MOTION_FILE` env var
  - otherwise attempts `mjlab.scripts.gcs.ensure_default_motion()`
- Added explicit error message when tracking motion file cannot be resolved.

Primary file:
- `src/flow_mbpo_pwm/envs/mjlab_pwm_adapter.py`

## 5. Tracking Task Stabilization

- Added local tracking motion asset to remove external download dependency:
  - `scripts/assets/motions/g1_tracking_dummy_motion.npz`
- Wired tracking env config to use this local file by default.
- Added task-specific manifest override for tracking:
  - `alg.horizon=1`
  - avoids short-trajectory replay sampling failures during early training.

Primary files:
- `scripts/cfg/env/mjlab_tracking_flat_unitree_g1.yaml`
- `scripts/assets/motions/g1_tracking_dummy_motion.npz`
- `scripts/experiments/single_task_online/build_manifest.py`

## 6. Active Manifest Baseline

Current active baseline manifests:
- `scripts/experiments/single_task_online/manifests/smoke_required_all_v2_20260228.csv`
- `scripts/experiments/single_task_online/manifests/pilot_required_seed0_default_v2_20260228.csv`
- `scripts/experiments/single_task_online/manifests/pilot_quickcheck_required_seed0_v2_20260228.csv`

## 7. Cluster Split Utility

- Added a deterministic manifest splitter to pin each task to one cluster:
  - `scripts/experiments/single_task_online/split_manifest_by_cluster.py`
- This prevents cross-cluster checkpoint sharing and preserves fair comparisons.

## 8. Packed Single-GPU Execution

- Added packed execution support so one GPU can run multiple light rows in
  parallel (best for smoke/quick-check).
- New scripts:
  - `scripts/experiments/single_task_online/run_manifest_pack.py`
  - `scripts/experiments/single_task_online/submit_manifest_packed_array.sh`
- This enables ICE high-risk bring-up with better wall-clock utilization while
  still keeping one Slurm task per GPU.

## 9. Scripts Directory Cleanup

- Reduced top-level `scripts/` clutter by keeping active single-task online files
  in place and moving legacy scripts to `scripts/legacy/`.
- Added:
  - `scripts/README.md`
  - `scripts/legacy/README.md`
- Legacy scripts remain available for history/repro but are no longer mixed into
  the active execution path.
