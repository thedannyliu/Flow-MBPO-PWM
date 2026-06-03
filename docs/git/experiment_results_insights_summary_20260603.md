# Experiment Results And Insights Summary - 2026-06-03

This is the durable summary of the current MJLab PWM / Flow-MBPO / image
world-model evidence. Raw logs and generated outputs remain outside git; this
file records the stable conclusions, key numbers, job IDs, and artifact paths.

## Data And Comparator Context

The active MJLab policy-extraction and Flow-MBPO rows use the Velocity Flat
Unitree G1 QS H16 composite dataset:

```text
dataset: scripts/outputs/mjlab_qs/windows/rerun_a25_native_qs_g1stage4_expertboost_20260527/velocity_flat_unitree_g1/d_qs_core_h16.pt
metadata: scripts/outputs/mjlab_qs/windows/rerun_a25_native_qs_g1stage4_expertboost_20260527/velocity_flat_unitree_g1/d_qs_core_h16.json
normalization: scripts/outputs/mjlab_qs/windows/rerun_a25_native_qs_g1stage4_expertboost_20260527/velocity_flat_unitree_g1/d_qs_core_h16_normalization.json
```

The metadata reports the mixed QS quality IDs used by the current composite
dataset: `random_smooth`, `medium`, `expert`, and `expert_noisy`.

Primary comparators:

| Comparator | Return | Length | Fall | Notes |
| --- | ---: | ---: | ---: | --- |
| BC eval40 aggregate | `45.8491` | `594.97` | `0.625` | Main scalar comparator |
| BC matched rollout10 seed0 | `54.1283` | `688.40` | `0.400` | Matched video gate comparator |
| Expert collector | `82.6090` | `1000.00` | `0.000` | Target reference |
| Expert-noisy collector | `80.3525` | `1000.00` | `0.000` | Stable noisy target |
| Medium collector | `49.1935` | `653.33` | `0.667` | Medium data-quality reference |

## PWM Findings

Full upstream PWM on MJLab is mechanically feasible but collapses in real eval.

Evidence:

- Full upstream path runs through `baselines/PWM/scripts/train_dflex.py` and
  upstream `pwm.algorithms.pwm.PWM`.
- H200 full-upstream eval40:
  - final: return `-1.7261`, length `39.525`, fall `1.000`
  - best: return `-1.5293`, length `50.650`, fall `1.000`
- H200 full-upstream rollout10:
  - final: return `-1.8182`, length `37.900`, fall `1.000`
  - best: return `-1.3551`, length `52.500`, fall `1.000`

The upstream PWM algorithm adapter and replay/dataset-driven PWM extraction also
collapse:

- Original adapter formal eval40: final `-0.8010`, best `-0.7778`, fall `1.000`.
- Original adapter rollout10: final `-0.5265`, best `-0.5218`, fall `1.000`.
- Replay/dataset original PWM adapter `9404376_0`:
  - WM test loss `0.005466`
  - best imagined return proxy `4.4287` at iter `2986`
  - real eval16 return `-1.0475`, length `43.75`
  - eval40 final/best: about `-0.948`, length about `43-44`

## PWM Collapse Diagnosis

The collapse probe jobs `9414357_[0-2]` completed and indicate that the basic
dataset-distribution world-model fit is not the main failure.

Key collapse-probe numbers:

- WM reward correlation: about `0.962-0.966`
- WM reward MSE: about `0.069-0.083`
- H16 dynamics MSE: about `0.0031-0.0032`
- Dataset predicted return: train/val about `-0.645`, test about `-1.632`
- Extracted final/best policy predicted return: about `28.5-28.7`
- Extracted policy action saturation: about `33.5%`
- Extracted policy vs dataset action MSE: about `0.71-0.72`
- Critic value mean after extraction: about `163`

Interpretation:

The world model predicts dataset rewards/dynamics reasonably well on the fixed
QS windows, but PWM policy extraction and TD/value optimization drive the policy
into out-of-distribution action/state regions. The critic becomes very
optimistic, imagined return rises sharply, and the real MJLab policy collapses.
The failure is therefore primarily policy/critic exploitation of the imagined
model, not a basic reward-fit failure.

## Flow-MBPO Versus MLP

Detailed comparison: `docs/git/flow_mbpo_vs_mlp_comparison_20260603.md`.

Strongest scalar Flow-MBPO row:

| Row | Return | Length | Fall | Comparator |
| --- | ---: | ---: | ---: | --- |
| Flow-MBPO H1 endpoint final eval40 | `60.8721` | `759.30` | `0.450` | BC eval40 |
| MLP BC eval40 aggregate | `45.8491` | `594.97` | `0.625` | baseline |

Improvement over BC eval40:

- Return: `+15.0230`, about `+32.8%`
- Length: `+164.33`, about `+27.6%`
- Fall: `-0.175`

Matched video gate is weaker:

- H1 endpoint final rollout10: `47.4617 / 625.60 / fall 0.500`, below matched
  BC video.
- H1 endpoint best rollout10: `55.5533 / 707.60 / fall 0.400`, slightly above
  matched BC return/length and tied fall.

Interpretation:

Flow-MBPO has real positive evidence versus BC on eval40, but the improvement is
not fully stable under matched rollout/video gates. It is promising, not a
complete robust win.

## Negative Flow-MBPO Diagnostics

The later conservative broad AWR/AWAC/support-truncation diagnostics are
negative and should not be counted as Flow-MBPO improvements.

Examples:

- Conservative broad Flow-MBPO AWR sweep:
  - best 8-episode row about `25.9699` return, length about `360`, fall `1.000`
  - below BC `45.8491 / 594.97 / 0.625`
- Support-truncation AWR:
  - q0.90 return `23.1614`, length `337.375`, fall `1.000`
  - q0.50 return `19.4189`, length `291.250`, fall `1.000`
- AWAC diagnostics:
  - mechanically stable across GPU nodes and replay families
  - all completed rows still fall at rate `1.000` and fail the BC gate

Interpretation:

Conservative penalties, support truncation, and AWAC weighting did not solve the
fall/OOD issue in the tested settings. These should guide debugging but not be
promoted to formal candidates.

## MJLab Flow Architecture 2x2

Seed0 2x2 policy-extraction result:

| WM | Policy | Return | Length | Best imagined return |
| --- | --- | ---: | ---: | ---: |
| `mlp_ref` | `mlp` | `2.5724` | `93.3125` | `2316.7583` |
| `mlp_ref` | `flow` | `-1.4830` | `66.7500` | `1482.8914` |
| `flow_endpoint` | `mlp` | `-3.9967` | `57.3125` | `396.2366` |
| `flow_endpoint` | `flow` | `-1.4542` | `65.3125` | `534.7295` |

Seed1/2 final aggregate:

```text
rows:      scripts/outputs/mjlab_qs/policy_extraction/rerun_g1_bcwarm_pwm_bcreg10_2x2_seeds1_2_20260603/rows.csv
aggregate: scripts/outputs/mjlab_qs/policy_extraction/rerun_g1_bcwarm_pwm_bcreg10_2x2_seeds1_2_20260603/aggregate.csv
jobs:      9414359_[0-7] H200 completed; 9414399_[0-7] H100 backup completed
```

| WM | Policy | n | Return mean | Length mean | Best imagined return mean |
| --- | --- | ---: | ---: | ---: | ---: |
| `mlp_ref` | `mlp` | 2 | `1.4459` | `112.9063` | `1331.1754` |
| `mlp_ref` | `flow` | 2 | `-3.2666` | `66.4063` | `1257.6684` |
| `flow_endpoint` | `mlp` | 2 | `-0.4530` | `75.5938` | `782.2411` |
| `flow_endpoint` | `flow` | 2 | `-1.5293` | `80.0000` | `1543.9488` |

Interpretation:

In the MJLab PWM-style policy-extraction 2x2, the current flow policy and flow
world-model replacements do not fix the collapse. The best seed1/2 average is
still `mlp_ref + mlp`, and all rows remain far below the BC/Flow-MBPO gates.

## NEWT And LeWorldModel Official Pipelines

Official NEWT own-task status:

- Earlier official smokes `9404504_[7-15]` and `9404505_[0-15]` completed.
- Moderate walker-run `9414358_[0-1]` completed:
  - seed0 final train return `25.644`
  - seed1 final train return `9.667`

Official LeWorldModel status:

- Official PushT eval `9404506_[0-5]` completed:
  - seed0 h2 `100%`
  - seed1 h2 `100%`
  - seed2 h2 `75%`
  - seed0 h5 `75%`
  - random seed0 h2 `25%`
  - random seed1 h2 `0%`
- LeWM train smoke was repaired:
  - `9411595_0/1` completed with exit `0:0`
  - H100 backup `9411693` was canceled
- Moderate PushT eval `9414360_[0-2]` completed:
  - `83.3%`, `66.7%`, `91.7%` success rates

Interpretation:

The official NEWT and LeWM pipelines are operational. These official runs do not
by themselves introduce Flow-MBPO architecture changes.

## NEWT And LeWorldModel Flow 2x2

Detailed record: `docs/git/image_flow_2x2_plan_20260603.md`.

NEWT flow 2x2:

```text
metrics: scripts/outputs/image_official/newt_flow_2x2_h200_20260603/metrics.csv
primary: 9415373_[0-7] H200 completed
backups: 9415378 and 9415411 were canceled after primary completion
```

Metric: final train return at 5k walker-run steps, averaged over seeds 0 and 1.

| WM | Policy | n | Return mean | Success mean |
| --- | --- | ---: | ---: | ---: |
| `mlp` | `mlp` | 2 | `16.754` | `0.017` |
| `mlp` | `flow` | 2 | `26.446` | `0.027` |
| `flow` | `mlp` | 2 | `19.320` | `0.019` |
| `flow` | `flow` | 2 | `12.331` | `0.013` |

Interpretation: in this short smoke, a flow policy head improved the NEWT
train metric when the WM stayed MLP. The current residual-flow WM replacement
did not improve the short-run metric.

LeWorldModel flow 2x2:

```text
metrics: scripts/outputs/image_official/lewm_flow_2x2_train_h200_20260603/metrics.csv
primary: 9415374_[0-7] H200 completed
backups: 9415379 and 9415412 were canceled after primary completion
```

Metric: one epoch, 8 train batches, 2 validation batches, averaged over seeds 0
and 1.

| Predictor | Action encoder | n | Fit loss | Fit pred loss | Val loss | Val pred loss |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `mlp` | `mlp` | 2 | `0.346582` | `0.070738` | `0.340797` | `0.041543` |
| `mlp` | `flow` | 2 | `0.345693` | `0.068504` | `0.337587` | `0.038300` |
| `flow` | `mlp` | 2 | `0.347257` | `0.072154` | `0.337663` | `0.039430` |
| `flow` | `flow` | 2 | `0.343569` | `0.069079` | `0.335017` | `0.036737` |

Interpretation: in this tiny LeWM train probe, flow+flow has the best validation
loss and validation prediction loss, but the margins are small and this is not
yet a PushT eval-success claim.

## Current High-Level Conclusions

1. Full PWM and replay/dataset PWM collapse on MJLab.
2. The collapse is now localized mainly to imagined policy extraction / critic
   extrapolation, not basic dataset-distribution WM reward fit.
3. Flow-MBPO has the strongest positive MJLab policy result so far, especially
   H1 endpoint final eval40, but matched video evidence is weaker.
4. Conservative AWR/AWAC/support-truncation variants did not improve the issue.
5. The MJLab flow architecture 2x2 does not show that flow WM/policy alone fixes
   PWM-style extraction.
6. Official NEWT and LeWM pipelines are operational.
7. NEWT/LeWM flow 2x2 probes now run and produce metrics; NEWT shows a short-run
   signal for flow policy, while LeWM shows a small validation-loss signal for
   flow+flow. Both need longer/eval-level confirmation before any performance
   claim.

## Git Record

Recent commits that created or recorded the durable evidence:

```text
7c9e5e0 Record image flow 2x2 results
c0419c5 Fix image flow architecture metric parsing
44ec8ca Record A100 image flow backups
76c7ba1 Fix image flow metrics step parsing
923e717 Record image flow 2x2 submissions
961353c Add NEWT and LeWM flow 2x2 probes
0d77756 Record PWM collapse diagnostics and follow-up submissions
3927996 Add H100 backups for collapse and 2x2 probes
52b4a5b Add PWM collapse probes and follow-up jobs
32ec068 Record LeWM train smoke repair success
45cd89a Fix LeWM train smoke submission
d26d791 Record Flow comparison and new research jobs
```

