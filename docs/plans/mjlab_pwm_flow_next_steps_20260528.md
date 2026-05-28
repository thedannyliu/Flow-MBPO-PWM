# MJLab PWM/Flow Next Steps

Date: 2026-05-28

## Current Evidence

The MJLab QS dataset is usable and the collector/reference baselines are now
anchored with real-environment rollout videos. The expert collector reaches
return `82.61`, episode length `1000`, and fall rate `0.0`. Expert-noisy is
similar at return `80.35`. The expert-filtered BC baseline is much weaker but
nontrivial: rollout return `19.08`, length `238.22`, and fall rate `0.333`.

All current PWM-style extracted policies remain far below BC and collector
quality. The best conservative BC-warmstarted PWM row reaches only return
`-1.10`, length `54.67`, and fall rate `1.0`. This means imagined-return gains
are not yet translating into real locomotion.

## Next Work I Would Do

1. Make expert-filtered BC stronger before more PWM exploitation.
   - Run longer BC, balanced expert/expert-noisy sampling, and possibly medium
     plus expert curricula.
   - Success criterion: real rollout beats the current BC baseline and moves
     toward medium collector scale, not just lower action MSE.

2. Add a BC-preservation gate to PWM warmstarts.
   - Before accepting any imagined actor update, compare final and best actor
     against the BC checkpoint with real-env rollout.
   - Success criterion: PWM does not reduce rollout return/length below BC.

3. Rerun conservative PWM only after the BC gate is credible.
   - Compare `mlp_ref` vs `flow_endpoint` world models and MLP vs flow policies.
   - Keep final and true-best actor snapshots, rollout MP4s, summaries, and
     W&B videos for every row.

4. Continue SigReg only as a world-model diagnostic.
   - Test whether SigReg improves rollout dynamics/reward prediction, then
     require downstream real-env rollout before making any policy claim.

5. Maintain the unified comparison report.
   - Re-run `scripts/experiments/mjlab_qs/export_rollout_comparison.py` after
     every completed policy rollout.
   - Compare each learned policy against expert collector, expert-noisy,
     medium collector, random_smooth, and BC-only baselines.

## Constraints To Preserve

- All GPU jobs must use `embers` unless the user explicitly approves `inferno`.
- Every formal run must have W&B logging.
- Every policy claim must include real-env scalar eval, rollout MP4, W&B video,
  and comparison in the unified rollout report.
- Do not treat world-model loss or imagined return as sufficient evidence.

## What I Would Avoid For Now

- Do not move to NEWT/image-based tasks yet.
- Do not run large PWM sweeps until BC can reliably recover stronger behavior.
- Do not claim Flow/PWM improvement from scalar metrics without rollout videos.
