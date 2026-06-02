# Repository Guidelines

## Agent Behavior

Behavioral guidelines to reduce common LLM coding mistakes. Merge with project-specific instructions as needed.

**Tradeoff:** These guidelines bias toward caution over speed. For trivial tasks, use judgment.

## 1. Think Before Coding

**Don't assume. Don't hide confusion. Surface tradeoffs.**

Before implementing:
- State your assumptions explicitly. If uncertain, ask.
- If multiple interpretations exist, present them - don't pick silently.
- If a simpler approach exists, say so. Push back when warranted.
- If something is unclear, stop. Name what's confusing. Ask.

## 2. Simplicity First

**Minimum code that solves the problem. Nothing speculative.**

- No features beyond what was asked.
- No abstractions for single-use code.
- No "flexibility" or "configurability" that wasn't requested.
- No error handling for impossible scenarios.
- If you write 200 lines and it could be 50, rewrite it.

Ask yourself: "Would a senior engineer say this is overcomplicated?" If yes, simplify.

## 3. Surgical Changes

**Touch only what you must. Clean up only your own mess.**

When editing existing code:
- Don't "improve" adjacent code, comments, or formatting.
- Don't refactor things that aren't broken.
- Match existing style, even if you'd do it differently.
- If you notice unrelated dead code, mention it - don't delete it.

When your changes create orphans:
- Remove imports/variables/functions that YOUR changes made unused.
- Don't remove pre-existing dead code unless asked.

The test: Every changed line should trace directly to the user's request.

## 4. Goal-Driven Execution

**Define success criteria. Loop until verified.**

Transform tasks into verifiable goals:
- "Add validation" → "Write tests for invalid inputs, then make them pass"
- "Fix the bug" → "Write a test that reproduces it, then make it pass"
- "Refactor X" → "Ensure tests pass before and after"

For multi-step tasks, state a brief plan:
```
1. [Step] → verify: [check]
2. [Step] → verify: [check]
3. [Step] → verify: [check]
```

Strong success criteria let you loop independently. Weak criteria ("make it work") require constant clarification.

---

**These guidelines are working if:** fewer unnecessary changes in diffs, fewer rewrites due to overcomplication, and clarifying questions come before implementation rather than after mistakes.

## Project Structure & Module Organization

Core code lives in `src/flow_mbpo_pwm/`. Use `algorithms/` for PWM/SHAC training logic, `models/` for world models and policies, `envs/` for Gymnasium and MJLab adapters, and `utils/` for buffers, monitoring, Hydra helpers, and reproducibility tools. Active workflows are under `scripts/`: `scripts/experiments/single_task_online/` handles online single-task manifest runs, `scripts/experiments/mjlab_qs/` handles MJLab QS collection, quality gates, and policy extraction, and `scripts/experiments/world_model_phase1/` handles offline world-model probes. Configs are in `scripts/cfg/alg/` and `scripts/cfg/env/`. Keep generated data in ignored locations such as `logs/`, `outputs/`, `eval_results/`, `wandb/`, `hf_pwm_repo/`, and `scripts/assets/pwm_hf/`.

## Build, Test, and Development Commands

Create the development environment from the repository root:

```bash
conda env create -f environment.yaml
conda activate pwm
pip install -e .
```

Build a single-task smoke manifest and split it by cluster:

```bash
python scripts/experiments/single_task_online/build_manifest.py --stage smoke --output scripts/experiments/single_task_online/manifests/smoke_tmp.csv
python scripts/experiments/single_task_online/split_manifest_by_cluster.py --manifest scripts/experiments/single_task_online/manifests/smoke_tmp.csv
```

Submit Slurm arrays with `scripts/experiments/single_task_online/submit_manifest_array.sh --manifest <manifest.csv> --gpu-type H100`. For MJLab QS work, prefer the manifest builders and `submit_array.sh` in `scripts/experiments/mjlab_qs/`.

## Coding Style & Naming Conventions

Use Python 3.10+, 4-space indentation, snake_case functions and config keys, and descriptive module names. Keep Hydra/OmegaConf settings in YAML instead of hard-coding experiment constants. Match existing config naming, for example `pwm_5M_flow_v2_substeps4_*.yaml` for Flow WM variants and dated manifest names such as `offline_pwm_ablate_*_20260507_*.csv`. Avoid committing caches, checkpoints, W&B runs, raw local result dumps, or files already ignored by `.gitignore`.

## Testing Guidelines

There is no formal unit-test suite. Validate with the smallest workflow that exercises your change: import/package checks for library edits, manifest generation for experiment tooling, and one smoke row or Slurm array for training/eval changes. Record task, seed, manifest path, GPU type, checkpoint/output directory, and any W&B project in PR notes.

## Commit & Pull Request Guidelines

Recent history uses imperative subjects: `Add ...` for new utilities or experiment flows, `Document ...` and `Record ...` for status/results, `Fix ...` for behavior corrections, and `Update ...` for refreshed job records. Keep commits narrow and separate code from large generated artifacts. PRs should state the research or pipeline impact, list validation commands or smoke runs, link relevant docs, and call out cluster-resource, checkpoint, dataset, or W&B implications.

## Documentation Notes

Tracked docs currently emphasize the MJLab restart plan in `docs/plans/`; older raw logs and intermediate CSV summaries are intentionally local/ignored. Add durable protocol decisions to tracked docs, but keep transient monitoring snapshots out of git unless they support a reviewed result.

## Cluster & QOS Policy

Current runs should target PACE-Phoenix by default. Use the `embers` QOS for GPU jobs because it is not charged to the account, though it has lower priority. Do not submit with `inferno` unless the user explicitly approves it; `inferno` has normal priority but incurs account charges.

Multiple GPU jobs may be submitted at the same time when the experiment plan benefits from parallel runs and cluster capacity permits it. The default execution style for active research goals is broad submission: submit all useful smoke, diagnostic, eval, and formal jobs that can plausibly run with currently available inputs, instead of waiting for one phase to finish before submitting the next. Do not add Slurm dependencies just to preserve phase order. Add a dependency only when the downstream command literally needs a file that does not exist yet, such as a checkpoint or generated dataset.

When broad submissions reveal a bad config, wrong checkpoint, missing dataset, broken wrapper, or bad environment, cancel the affected jobs with `scancel`, record the job IDs and root cause, fix the issue, and resubmit replacement jobs. This is preferred over delaying the whole queue. Prefer higher-end GPUs first, in this order unless a script or dependency requires otherwise: H200, H100, A100, L40S, then lower-tier available GPUs.

Use English for git commit messages and durable documentation records, including experiment status notes, failure diagnoses, run tables, and follow-up instructions.
