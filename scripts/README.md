# Scripts Directory Guide

This repository now keeps the active single-task online RL workflow in a focused layout.

## Active Paths

- `scripts/train_dflex.py` / `scripts/train_online.py`
- `scripts/cfg/`
- `scripts/eval/eval_online_single_task.py`
- `scripts/experiments/single_task_online/`
- `scripts/assets/motions/g1_tracking_dummy_motion.npz`

## Legacy Paths

Historical multitask and legacy submission utilities were moved to:

- `scripts/legacy/top_level/`
- `scripts/legacy/task_pipelines/`
- `scripts/legacy/eval/`
- `scripts/legacy/mjlab/`

This keeps the active workflow clean while preserving old scripts for reference.
