# Experiment Results and Insights - 2026-06-04

## Current Completed Results

### Flow-MBPO H1 real-only vs synthetic ratio

Completed job: `9419845`.

All rows used the historical exact H1 replay:

`scripts/outputs/mjlab_qs/flow_mbpo_v0_replay/flow_endpoint_ensemble_seed0_h1_unc0p5_q0p90/synthetic_replay.pt`

Mean best in-training real eval return over seeds 0-2:

| real/synthetic batch | mean return | seed returns |
| --- | ---: | --- |
| `256/0` real-only | `22.75` | `21.38, 25.39, 21.47` |
| `248/8` | `21.52` | `27.87, 15.66, 21.02` |
| `224/32` | `20.70` | `23.54, 17.05, 21.51` |
| `192/64` | `19.13` | `21.24, 15.08, 21.07` |

Result: real-only is not good, and adding synthetic does not improve it. Higher synthetic ratio trends worse in this small sweep.

### Flow-MBPO H1 replay quality

Completed job: `9419848`.

Exact replay and `_truncate_check` replay are effectively identical on the measured diagnostics.

Key numbers:

- synthetic transitions: `256`
- synthetic OOD fraction by nearest-real probe threshold: `0.078`
- synthetic done fraction: `0.102`
- synthetic done model fraction: `0.0`
- synthetic done uncertainty fraction: `0.102`
- nearest-real done-any fraction: `0.0`
- nearest-real termination-any fraction: `0.0`
- synthetic conservative reward mean/p90/max: `0.112 / 1.724 / 2.691`
- nearest-real reward0 mean/p90/max: `0.083 / 0.103 / 0.117`
- conservative reward minus nearest-real reward0 mean/p90/max: `0.030 / 1.620 / 2.574`

Important slice:

- top synthetic reward decile has reward delta mean `+2.34` over nearest real reward, nearest-real termination-any `0.0`, synthetic done `0.0`.

Result: the replay contains high-reward synthetic transitions whose rewards are far above nearest real one-step rewards. This is a plausible reward-model artifact, although it is not directly explained by nearest-real falls.

### AWR damage/action-drift diagnostics

Completed job: `9419844`.

Across strongest and fix1 H1 AWR checkpoints:

- replay done fraction: `0.102`
- AWR synthetic weight mean/p90/max: `1.99 / 5.01 / 13.18`
- policy action drift from BC is tiny:
  - real delta mean: roughly `0.0021-0.0034`
  - synthetic delta mean: roughly `0.0019-0.0034`
  - logged-action BC MSE remains around `5e-4`

Result: the 500-step AWR update barely moves the actor in action space. The weak in-training returns are unlikely to be caused by a large policy drift from BC.

## Prior Completed Comparisons

### Historical strongest Flow-MBPO H1 vs MLP BC

Historical formal eval40:

- Flow-MBPO H1 final: return `60.87`, length `759.30`, fall `0.45`
- MLP BC baseline: return `45.85`, length `594.97`, fall `0.625`
- apparent lift: return `+32.8%`, length `+27.6%`, fall `-0.175`

But this is not yet robust: the same historical AWR run had weak in-training real eval (`best_real_return=17.10`), and the new direct eval retry is still pending.

### H1 multiseed/ratio follow-up with `_truncate_check`

Mean best in-training real eval return:

| real/synthetic batch | mean return |
| --- | ---: |
| `248/8` | `22.17` |
| `224/32` | `20.71` |
| `192/64` | `19.49` |

Result: repeated H1 AWR extraction did not reproduce the historical `60.87`.

### Data distribution sweep

Mean best in-training real eval return:

| data setting | mean return |
| --- | ---: |
| expert 50% + expert_noisy 50% | `18.71` |
| expert only | `17.31` |
| mixed uniform windows | `16.11` |
| no-fall/no-done success proxy | `13.66` |
| expert 50% + medium 50% | `10.38` |

Result: dataset distribution matters, but none of these AWR variants approach the BC baseline or the historical `60.87`.

## Direct Eval Retry Update

Checkpoint direct eval retry:

- failed job: `9419850`, invalid manifest fields `command_position=first`, `obs_mode=phys`
- fixed commit: `cf813bc`
- retry job: `9432225`
- current state during this update: running on H100 embers

Partial direct-eval outputs from `9432225`:

| checkpoint | return | length | fall |
| --- | ---: | ---: | ---: |
| BC seed0 initial | `47.02` | `604.1` | `0.650` |
| BC seed1 initial | `36.10` | `480.0` | `0.800` |
| BC seed2 initial | `40.10` | `513.3` | `0.700` |
| strongest exact-H1 final | `62.02` | `786.0` | `0.425` |
| strongest exact-H1 best-real-eval | `39.47` | `513.6` | `0.775` |
| fix1 r224/s32 seed0 iter250 | `46.36` | `601.2` | `0.675` |
| fix1 r224/s32 seed0 iter500 | `52.87` | `672.9` | `0.575` |
| fix1 r224/s32 seed0 final | `50.94` | `649.0` | `0.575` |
| fix1 r224/s32 seed0 best-real-eval | `56.98` | `727.8` | `0.400` |
| fix1 r192/s64 seed0 iter250 | `45.49` | `585.2` | `0.650` |

Interim interpretation: the historical strongest final direct eval is reproducible under the current runner (`62.02` vs historical `60.87`). The suspicious gap is now more specific: short in-training real eval can underrate checkpoints that look strong under formal eval40. This means the result is not simply a stale artifact, but robustness over eval seeds and across AWR seeds is still unresolved.

## Newly Submitted Follow-Up Eval

Submitted to close the remaining evidence gaps:

| job | manifest | purpose |
| --- | --- | --- |
| `9432431` | `scripts/experiments/mjlab_qs/manifests/flow_mbpo_h1_strongest_robust_eval_h100_20260604.csv` | repeated eval40 over eval seeds 0-4 for BC seed0, strongest final, strongest best, and fix1 r224/s32 seed0 iter500 |
| `9432430` | `scripts/experiments/mjlab_qs/manifests/flow_mbpo_h1_exact_ratio_checkpoint_eval_h100_20260604.csv` | formal eval40 for all exact-replay ratio sweep checkpoints: iter250, iter500, final, best |

Both jobs were submitted on H100 with QOS `embers`.

## Pending Result

The decisive remaining test is no longer whether the historical `60.87` can be reproduced once; it can. The remaining question is whether that improvement is stable across eval seeds and across exact-replay ratio/AWR seeds.

## Insights

1. The core offline idea is not validated yet: `offline dataset -> WM/replay -> AWR policy extraction -> real eval` remains weak in current MJLab runs.
2. The problem is not simply the `_truncate_check` replay path; exact replay and `_truncate_check` have the same measured quality stats, and exact replay ratio sweeps are still weak.
3. AWR itself may be too weak or badly aligned: real-only AWR is also poor, and action drift from BC is extremely small.
4. The synthetic reward model is suspicious: top H1 synthetic rewards are much higher than nearest real rewards, so AWR may be overweighting unrealistic high-reward transitions even though the actor movement is small.
5. The historical `60.87` is reproducible in the current runner, but still not proven robust. Current evidence now points to a checkpoint-selection/eval-protocol issue: 8-episode in-training eval can look weak while formal eval40 finds strong checkpoints.
