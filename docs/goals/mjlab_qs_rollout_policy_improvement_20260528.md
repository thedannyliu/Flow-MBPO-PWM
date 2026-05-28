# Goal Log: MJLab QS Rollout Policy Improvement

Date started: 2026-05-28
Branch: `mjlab-qs-rollout-policy-improvement`

## Scope

Continue the Velocity Flat Unitree G1 MJLab QS / PWM-Flow project with rollout-first evidence. Use prior completed work as evidence where it already satisfies the goal, especially `docs/phaseA/pwm_extension_submission_20260507.md` and `docs/plans/mjlab_pwm_flow_next_steps_20260528.md`.

## Current Evidence

- Collector/reference rollouts exist with W&B videos and MP4s. Expert collector return is `82.6090`, length is `1000.00`, and fall rate is `0.000`.
- Expert-noisy collector is similarly stable: return `80.3525`, length `1000.00`, fall rate `0.000`.
- Medium collector reaches return `49.1935`, length `653.33`, fall rate `0.667`.
- Expert-filtered BC is nontrivial but unstable: return `19.0827`, length `238.22`, fall rate `0.333`.
- Current PWM-style extracted policies remain collapsed. The least-bad conservative row is return `-1.0999`, length `54.67`, fall rate `1.000`.
- True best actor snapshots are now saved for new policy-extraction runs and rollout tooling evaluates final plus true-best checkpoints.

## Work This Goal

- Created this concise goal log so future turns have a single current-state checkpoint.
- Added required protocol docs: `docs/RUNBOOK.md`, `docs/EXPERIMENT_LEDGER.md`, `docs/DATASET_CARD_MJLAB_QS.md`, `docs/CLAIM_POLICY.md`, and `docs/DEBUG_TREE.md`.
- Tightened formal-run metadata logging for policy extraction and rollout renderers:
  - W&B config and local summaries now include git branch and command line.
  - Policy extraction summaries now include dataset, metadata, normalization, WM checkpoint, output directory, final checkpoint, and best checkpoint paths.
  - Policy rollout summaries now log fall rate directly and distinguish termination from time-limit truncation.
  - Rollout comparison export now prefers `fall_rate_mean` from summary JSON when available.
- Made BC-only/no-policy-update extraction rows write an evaluable true snapshot for `best_policy_extraction.pt`, with metadata explaining that best equals the current actor when no imagined-return update was run.
- Exposed `bc_sampling` and `policy_sampling` in policy extraction manifests. Existing behavior is `quality_balanced`; the next ablation tests `uniform` expert/expert_noisy BC sampling to check whether batch composition is part of the BC weakness.
- Prepared tracked manifest `scripts/experiments/mjlab_qs/manifests/rerun_g1_bc_expert_uniform_mlp50k_20260528.csv`.
- Submitted the uniform expert/expert_noisy BC ablation:
  - git SHA: `163a626`
  - Slurm job: `9236994`
  - partition/GPU/QOS: `gpu-h100` / `H100` / `embers`
  - W&B project: `flow-mbpo-mjlab-bc-expert-uniform-20260528`
  - array: `0-2%1`
  - status at submit check: pending
- Submitted A100 fallback for the same manifest and output paths:
  - Slurm job: `9237030`
  - partition/GPU/QOS: `gpu-a100` / `A100` / `embers`
  - reason: H100 array remained pending; per-row output locks prevent duplicate work
- Policy extraction completed on A100 fallback job `9237030`; unused H100 job `9236994` was cancelled.
  - seed0 eval: return `36.2848`, length `487.15`
  - seed1 eval: return `39.8943`, length `535.67`
  - seed2 eval: return `61.9562`, length `766.70`
  - all rows wrote final and best checkpoints
- Submitted rollout rendering for the same manifest:
  - Slurm job: `9237329`
  - partition/GPU/QOS: `gpu-a100` / `A100` / `embers`
  - expected artifacts: final and best rollout MP4/W&B videos for seeds 0-2
- Rollout job `9237329` completed on `embers`.
  - final aggregate: return `21.5481`, length `267.89`, fall `0.222`
  - best aggregate: return `21.5937`, length `267.89`, fall `0.222`
  - prior expert-filtered BC final aggregate: return `19.0827`, length `238.22`, fall `0.333`
  - interpretation: uniform sampling is a modest BC rollout improvement but still far below expert collector return `82.6090` and length `1000.00`
- Refreshed `scripts/outputs/mjlab_qs/reports/rollout_comparison_20260528.csv` and `.md`; they now contain `21` aggregate rows.
- Added tracked `results/master_policy_comparison.csv` with collector, BC, and representative PWM rows.
- Added opt-in BC action-rate regularization (`--bc-action-rate-reg`) and prepared tracked manifest `scripts/experiments/mjlab_qs/manifests/rerun_g1_bc_expert_uniform_smooth1e2_mlp50k_20260528.csv`.
  - purpose: test whether explicit action smoothness improves real rollout stability beyond the uniform-sampling BC gain
  - stage: `rerun_g1_bc_expert_uniform_smooth1e2_mlp50k_20260528`
  - W&B project: `flow-mbpo-mjlab-bc-expert-uniform-smooth-20260528`
  - settings: expert/expert_noisy BC, uniform sampling, `bc_action_rate_reg=0.01`, 50k BC steps, 3 seeds
- Submitted smooth uniform-BC extraction:
  - git SHA: `4d4488e`
  - Slurm job: `9237622`
  - partition/GPU/QOS: `gpu-a100` / `A100` / `embers`
  - array: `0-2%1`
  - row0 started running at submit check
- Smooth uniform-BC seed0 completed:
  - W&B run: `48rqu8f9`
  - eval return `49.1083`, length `624.10`
  - final and best checkpoints written
  - note: this is much better than uniform BC seed0 eval (`36.2848`, length `487.15`), but rollout video is still required
- Smooth uniform-BC extraction completed on Slurm job `9237622`:
  - seed0 W&B `48rqu8f9`: eval return `49.1083`, length `624.10`, git `b3a06cf`
  - seed1 W&B `yuoi3zdp`: eval return `35.3721`, length `478.23`, git `f71489b`
  - seed2 W&B `in0vwa34`: eval return `53.7020`, length `678.42`, git `f71489b`
  - all rows wrote final and best checkpoints
  - note: seed1/2 logged a docs-only git SHA after the seed0 note commit; code/config remained from smooth-BC support
- Submitted smooth uniform-BC rollout rendering:
  - Slurm job: `9237887`
  - partition/GPU/QOS: `gpu-a100` / `A100` / `embers`
  - expected artifacts: final and best rollout MP4/W&B videos for seeds 0-2
- Smooth uniform-BC rollout job `9237887` completed on `embers`.
  - final rollout per seed: seed0 return `27.9737`, length `300.00`, fall `0.000`; seed1 return `22.3277`, length `300.00`, fall `0.000`; seed2 return `14.3850`, length `203.67`, fall `0.667`
  - best rollout per seed: seed0 return `28.0072`, length `300.00`, fall `0.000`; seed1 return `22.3120`, length `300.00`, fall `0.000`; seed2 return `14.3807`, length `203.67`, fall `0.667`
  - final aggregate: return `21.5621`, length `267.89`, fall `0.222`
  - best aggregate: return `21.5666`, length `267.89`, fall `0.222`
  - W&B rollout runs: final `5f9et04a`, `lk9cr22p`, `2quaut5a`; best `o2fpcvmb`, `e4hj0jlj`, `rz1h2svk`
  - refreshed `scripts/outputs/mjlab_qs/reports/rollout_comparison_20260528.csv` and `.md`; they now contain `23` aggregate rows
  - interpretation: action-rate smoothness improved policy-extraction eval but did not materially improve 300-step rendered rollout over uniform BC (`21.5481` final, `21.5937` best)
- Added 1000-step rollout comparison support and submitted an aligned BC render sanity check:
  - code/manifest commit: `68a328f`
  - manifest: `scripts/experiments/mjlab_qs/manifests/rerun_g1_bc_longroll1000_uniform_vs_smooth_20260528.csv`
  - Slurm job: `9238133`
  - partition/GPU/QOS: `gpu-a100` / `A100` / `embers`
  - rows: uniform BC and smooth uniform BC, seeds 0-2, final and best checkpoints, `rollout_max_steps=1000`
  - purpose: test whether the apparent BC eval vs rendered-rollout mismatch is partly caused by the previous 300-step video cap

## Next Action

Monitor Slurm job `9238133`. If 1000-step rollouts recover the higher policy-eval returns or lengths, refresh the comparison report with a clear max-step label before making any BC/PWM claims. If they remain near the 300-step evidence, target environment/eval wrapper mismatch before resuming PWM sweeps.
