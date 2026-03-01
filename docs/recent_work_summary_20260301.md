# Recent Work Summary (March 1, 2026)

## Scope

This update consolidates the active single-task online RL pipeline, replaces the mjlab task set with the new requested tasks, and refreshes the smoke/pilot manifests for immediate execution.

## Completed Changes

### 1) Scripts Cleanup (Risk-Based)

- Removed the entire legacy script archive under `scripts/legacy/` after line-by-line review.
- Removed cached Python artifacts under `scripts/**/__pycache__` and `scripts/**/*.pyc`.
- Updated script-level docs to reflect that the active pipeline no longer uses the legacy tree.

Result:
- 112 tracked legacy files removed.
- Active `scripts/` now keeps only train/eval/config/experiment components plus outputs.

### 2) MJLab Task Panel Replacement

Replaced the previous direct mjlab task options with:

- `leap_left_grasp_asymmetric` (Medium)  
  env: `mjlab_leap_left_grasp_asymmetric`  
  task_id: `Mjlab-Leap-Left-Grasp-Asymmetric`
- `tracking_rough_unitree_g1` (Med-High)  
  env: `mjlab_tracking_rough_unitree_g1`  
  task_id: `Mjlab-Tracking-Rough-Unitree-G1`
- `leap_left_inhand_pen_twirl` (High)  
  env: `mjlab_leap_left_inhand_pen_twirl`  
  task_id: `Mjlab-Leap-Left-InHand-Pen-Twirl`

Added new env config files:

- `scripts/cfg/env/mjlab_leap_left_grasp_asymmetric.yaml`
- `scripts/cfg/env/mjlab_tracking_rough_unitree_g1.yaml`
- `scripts/cfg/env/mjlab_leap_left_inhand_pen_twirl.yaml`

Proxy tasks remain unchanged for required benchmark coverage:

- `anymal` proxy -> `mjlab_velocity_flat_unitree_go2`
- `snu_humanoid` proxy -> `mjlab_velocity_flat_unitree_g1`

### 3) Manifest Pipeline Updates

Updated:

- `scripts/experiments/single_task_online/build_manifest.py`
  - replaced direct mjlab task specs with the three new tasks
  - moved imitation-specific horizon safeguard to `tracking_rough_unitree_g1`
- `scripts/experiments/single_task_online/split_manifest_by_cluster.py`
  - ICE assignment now includes new mjlab tasks
  - backward-compatible mapping for old task keys kept intentionally
- `scripts/experiments/single_task_online/README.md`
  - active task list now reflects the new mjlab panel

Regenerated manifests (same file names, new content):

- `smoke_required_all_v2_20260228.csv` (16 rows)
- `pilot_required_seed0_default_v2_20260228.csv` (16 rows)
- `pilot_quickcheck_required_seed0_v2_20260228.csv` (16 rows)
- and all corresponding `_pace_ice.csv` / `_pace_phoenix.csv` split files

Current split with new panel:

- PACE-ICE: 10 rows per manifest
- PACE-Phoenix: 6 rows per manifest

### 4) Documentation Sync

Updated docs to match current execution reality:

- `README.md` (project structure)
- `scripts/README.md` (active script scope)
- `docs/single_task_online_pace_ice_spec_en.md` (new mjlab tasks)
- `docs/dual_cluster_execution_plan_ice_phoenix_20260228.md` (cluster task assignment)
- `docs/phoenix_agent_handoff_prompt_20260228.md` (Phoenix task scope)

## Operational State

The repository is ready to continue smoke/pilot/confirm runs with:

- cleaned script surface area (legacy removed),
- updated mjlab task panel,
- refreshed manifests already split for ICE/Phoenix.

## Notes

- Existing output artifacts under `scripts/outputs/` were intentionally preserved.
- Existing untracked directory `baselines/PWM/` was not modified.
