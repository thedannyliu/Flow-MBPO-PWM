# MJLab QS Runbook

This runbook is for Velocity Flat Unitree G1 rollout-based evidence.

## Baseline Rollout Comparison

Regenerate the saved aggregate report after any completed rollout:

```bash
python scripts/experiments/mjlab_qs/export_rollout_comparison.py \
  --output-csv scripts/outputs/mjlab_qs/reports/rollout_comparison_20260528.csv \
  --output-md scripts/outputs/mjlab_qs/reports/rollout_comparison_20260528.md
```

The current reference report is `scripts/outputs/mjlab_qs/reports/rollout_comparison_20260528.csv`.

## Policy Extraction

Build policy manifests from existing world-model checkpoints:

```bash
python scripts/experiments/mjlab_qs/build_policy_extraction_manifest_from_wm.py \
  --stage <stage> \
  --wm-stage <wm_stage> \
  --dataset-stage <dataset_stage> \
  --output scripts/outputs/mjlab_qs/manifests/<stage>.csv \
  --tasks velocity_flat_unitree_g1 \
  --wm-methods mlp_ref,flow_endpoint \
  --policy-types mlp,flow \
  --seeds 0,1,2 \
  --wandb-project <wandb_project>
```

For formal runs, leave W&B enabled. Each run must record git SHA, branch, full command, dataset, normalization, seed, WM checkpoint, final checkpoint, and best checkpoint.

BC-only runs use the same runner with `--policy-iters 0`. Use `--bc-sampling quality_balanced` to balance selected quality bins per batch, or `--bc-sampling uniform` to preserve the filtered dataset proportions. Use `--bc-action-rate-reg <weight>` for an opt-in action-rate smoothness penalty during BC warm start.

Submit on PACE-Phoenix with `embers` QOS:

```bash
bash scripts/experiments/mjlab_qs/submit_array.sh \
  --kind policy_extract \
  --manifest scripts/outputs/mjlab_qs/manifests/<stage>.csv \
  --gpu-type H100 \
  --partition gpu-h100 \
  --qos embers \
  --max-concurrent 1 \
  --python-bin /storage/home/hcoda1/9/eliu354/r-agarg35-0/envs/pwm/bin/python
```

Do not use `inferno` without explicit user approval.

## Policy Rollout Rendering

After policy extraction, render final and true-best actors:

```bash
bash scripts/experiments/mjlab_qs/submit_array.sh \
  --kind policy_rollout \
  --manifest scripts/outputs/mjlab_qs/manifests/<stage>.csv \
  --gpu-type H100 \
  --partition gpu-h100 \
  --qos embers \
  --time 02:00:00 \
  --max-concurrent 1 \
  --python-bin /storage/home/hcoda1/9/eliu354/r-agarg35-0/envs/pwm/bin/python
```

The rollout runner skips legacy `best_policy_extraction.pt` files that are not true actor snapshots.

## Policy Checkpoint Eval

Use no-video checkpoint evals to get larger episode-count return, length, fall-rate, and timeout-rate estimates. These evals complement rollout MP4 evidence; they do not replace videos.

```bash
bash scripts/experiments/mjlab_qs/submit_array.sh \
  --kind policy_eval \
  --manifest scripts/experiments/mjlab_qs/manifests/<eval_stage>.csv \
  --gpu-type A100 \
  --partition gpu-a100 \
  --qos embers \
  --time 02:00:00 \
  --max-concurrent 1 \
  --python-bin /storage/home/hcoda1/9/eliu354/r-agarg35-0/envs/pwm/bin/python
```

Manifest rows may set `policy_stage`, `eval_stage`, `eval_episodes`, `eval_num_envs`, `eval_max_steps`, and `checkpoint_kinds`. Formal evals must keep W&B enabled.

For Flow-MBPO snapshot candidates, build direct-checkpoint eval and rollout manifests with:

```bash
python scripts/experiments/mjlab_qs/build_flow_mbpo_candidate_eval_plan.py \
  --awr-dir <flow_mbpo_awr_output_dir> \
  --eval-dir <candidate_eval_output_dir> \
  --rollout-dir <candidate_rollout_output_dir> \
  --output-csv <candidate_plan.csv> \
  --output-sh <candidate_plan.sh> \
  --output-eval-manifest scripts/experiments/mjlab_qs/manifests/<candidate_eval_stage>.csv \
  --output-rollout-manifest scripts/experiments/mjlab_qs/manifests/<candidate_rollout_stage>.csv \
  --wandb-project-eval <wandb_eval_project> \
  --wandb-project-rollout <wandb_rollout_project> \
  --wandb-group <candidate_group> \
  --notes "<dataset/config/seed/checkpoint decision notes>"
```

Submit those manifests through `submit_array.sh --kind policy_eval` and `--kind policy_rollout` so candidate snapshots still use the `embers` QOS guard, W&B row-runner path, and baseline gate fields. For formal candidate eval/render, add `--require-formal-metadata`; this rejects rows with W&B disabled or missing W&B project/group, notes, baseline gates, or direct-checkpoint output directories before `sbatch`.

## Current Experiment Order

1. Improve or debug expert-filtered BC/IL.
2. Verify world models with long-horizon, reward, termination, and expert-in-model diagnostics.
3. Only then rerun conservative BC-warmstarted PWM 2x2.
4. Add SigReg only after long-horizon behavior and real rollout are part of the comparison.
