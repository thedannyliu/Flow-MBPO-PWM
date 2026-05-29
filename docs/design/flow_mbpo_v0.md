# Flow-MBPO v0 Design

Date: 2026-05-29

## Goal

Build the smallest reproducible Flow-MBPO path that tests whether short Flow-world-model synthetic rollouts can improve or preserve Velocity Flat Unitree G1 behavior relative to the best BC baseline.

Success still requires real MJLab eval, rollout MP4/W&B videos, return, episode length, fall rate, and comparison to collector/reference/BC policies. Imagined return and world-model loss remain diagnostic only.

## Non-Goals

- Do not add more yaw weighting, reset weighting, action-ramp, or medium-mixing sweeps by default.
- Do not optimize actors by unconstrained long-horizon backprop through a learned model.
- Do not claim improvement without real rollout videos.

## v0 Components

1. Model interface
   - Wrap existing MLP and Flow world models behind a shared synthetic-transition API.
   - Required outputs per model step: next latent/state, reward, done/fall probability if available, uncertainty score.
   - Start with existing `WorldModel`, `EnsembleWorldModel`, `FlowWorldModel`, and `EnsembleFlowWorldModel` before adding new chunked models.

2. Synthetic rollout generator
   - Initialize from real dataset windows.
   - Use a frozen BC-warmstarted actor to choose actions.
   - Roll out only short horizons: `H in {1, 3, 5}` for v0.
   - Stop synthetic rollout early on predicted fall or high uncertainty.

3. Synthetic replay buffer
   - Store `(state, command, action, reward, next_state, done, uncertainty, source_model, horizon_step)`.
   - Keep real dataset transitions separate so synthetic:real ratios are explicit.

4. Policy update
   - Start from the strongest expert+noisy uniform BC checkpoint.
   - Use model-free updates on mixed real/synthetic batches.
   - Prefer AWAC/AWR for first v0 because logged expert actions and BC warm starts are already available.
   - SAC/PPO-style updates can follow after the buffer path is validated.

5. Conservatism
   - Apply `r_conservative = r_model - lambda_uncertainty * uncertainty`.
   - Sweep only a tiny set first: `lambda_uncertainty in {0, 0.5, 1.0}` and synthetic:real ratio `{0.25, 1.0}`.
   - Record uncertainty distribution and early-termination counts.

6. Evaluation
   - No policy claim before 40-episode real eval and rollout videos.
   - Save and evaluate final and true-best actors.
   - Select best actor by real-eval gate when available, not imagined return alone.

## First Implementation Slice

Implement a CPU/GPU smoke path with W&B disabled:

1. Load `d_qs_core_h16.pt`, normalization, and the current best BC checkpoint.
2. Load one trained MLP WM checkpoint and one Flow WM checkpoint if present.
3. Generate a small synthetic buffer from 256 real start states with horizon `H=1`.
4. Compute and save diagnostics only:
   - synthetic reward mean/std;
   - next-state delta norm;
   - uncertainty mean/p90/max;
   - predicted done/fall fraction;
   - action norm;
   - source dataset split/quality counts.
5. Do not update policy until the synthetic buffer diagnostics are bounded and reproducible.

Initial script:

```bash
python scripts/experiments/mjlab_qs/run_flow_mbpo_v0_smoke.py \
  --dataset scripts/outputs/mjlab_qs/windows/rerun_a25_native_qs_g1stage4_expertboost_20260527/velocity_flat_unitree_g1/d_qs_core_h16.pt \
  --metadata scripts/outputs/mjlab_qs/windows/rerun_a25_native_qs_g1stage4_expertboost_20260527/velocity_flat_unitree_g1/d_qs_core_h16.json \
  --normalization scripts/outputs/mjlab_qs/windows/rerun_a25_native_qs_g1stage4_expertboost_20260527/velocity_flat_unitree_g1/d_qs_core_h16_normalization.json \
  --policy-checkpoint scripts/outputs/mjlab_qs/policy_extraction/rerun_g1_bc_expert_uniform_mlp50k_20260528/velocity_flat_unitree_g1/mlp_ref/mlp/offline/bc50k_expert_uniform_policy0k/seed_0/final_policy_extraction.pt \
  --wm-checkpoint scripts/outputs/mjlab_qs/results/rerun_a25_native_qs_g1stage4_expertboost_20260527/velocity_flat_unitree_g1/mlp_ref/seed_0/best.pt \
  --wm-checkpoint scripts/outputs/mjlab_qs/results/rerun_a25_native_qs_g1stage4_expertboost_20260527/velocity_flat_unitree_g1/mlp_ref/seed_1/best.pt \
  --wm-checkpoint scripts/outputs/mjlab_qs/results/rerun_a25_native_qs_g1stage4_expertboost_20260527/velocity_flat_unitree_g1/mlp_ref/seed_2/best.pt \
  --output-dir scripts/outputs/mjlab_qs/flow_mbpo_v0_smoke/mlp_ref_ensemble_seed0_h1 \
  --device cuda:0 \
  --num-starts 256 \
  --horizon 1
```

## Formal v0 Run Gate

Run one formal seed on `embers` only after the smoke path passes:

- W&B enabled.
- Git SHA, command, dataset, checkpoint paths, config, seed, and notes logged.
- Final and true-best actor checkpoints written.
- 40-episode real eval complete.
- Rollout MP4/W&B videos rendered if scalar eval preserves or improves BC.

## Baselines

Use these current anchors:

- expert collector: return `82.6090`, length `1000.00`, fall `0.000`
- expert-noisy collector: return `80.3525`, length `1000.00`, fall `0.000`
- expert+noisy uniform BC 40-episode final: return `45.8491`, length `594.97`, fall `0.625`
- expert+noisy uniform BC 1000-step rollout final: return `41.8965`, length `547.78`, fall `0.667`

The v0 method must preserve or exceed the BC anchors before expanding to more seeds or methods.
