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

## Current Pending Result

Checkpoint direct eval retry:

- failed job: `9419850`, invalid manifest fields `command_position=first`, `obs_mode=phys`
- fixed commit: `cf813bc`
- retry job: `9432225`
- current state: pending on H100 embers, reason `Priority`

This is the decisive test for whether the historical `60.87` can be reproduced by direct eval under the current runner.

## Insights

1. The core offline idea is not validated yet: `offline dataset -> WM/replay -> AWR policy extraction -> real eval` remains weak in current MJLab runs.
2. The problem is not simply the `_truncate_check` replay path; exact replay and `_truncate_check` have the same measured quality stats, and exact replay ratio sweeps are still weak.
3. AWR itself may be too weak or badly aligned: real-only AWR is also poor, and action drift from BC is extremely small.
4. The synthetic reward model is suspicious: top H1 synthetic rewards are much higher than nearest real rewards, so AWR may be overweighting unrealistic high-reward transitions even though the actor movement is small.
5. The historical `60.87` should be treated as unconfirmed until `9432225` completes; current evidence leans toward artifact/eval variance/config-specific result rather than a robust improvement.

