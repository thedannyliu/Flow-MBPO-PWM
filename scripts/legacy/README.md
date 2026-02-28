# Legacy Scripts

This folder stores scripts that are not part of the active single-task online RL pipeline.

## Structure

- `top_level/`: old submit/install/fix/multitask entrypoints
- `task_pipelines/`: old task-specific pipelines (`ant`, `anymal`, `humanoid`, `mt30`)
- `eval/`: legacy evaluation/report generation scripts
- `mjlab/`: older mjlab smoke scripts

These files are kept for reproducibility/debug history and should not be used as the default path for current experiments.
