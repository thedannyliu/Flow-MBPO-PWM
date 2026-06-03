# Flow-MBPO vs MLP Comparison - 2026-06-03

Scope: MJLab Velocity Flat Unitree G1, existing QS composite dataset evidence. MLP comparator is the BC MLP policy, not the failed old PWM-style MLP extraction row unless stated otherwise.

| Row | Protocol | Return | Length | Fall | Comparator | dReturn | dLength | dFall | Interpretation |
| --- | --- | ---: | ---: | ---: | --- | ---: | ---: | ---: | --- |
| MLP BC matched rollout10 seed0 | matched_video | 54.1283 | 688.4000 | 0.4000 |  |  |  |  | baseline comparator |
| Flow-MBPO H1 endpoint final eval40 | eval40 | 60.8721 | 759.3000 | 0.4500 | MLP BC aggregate eval40 | 15.0230 | 164.3300 | -0.1750 | beats comparator on return/length and does not increase fall |
| Flow-MBPO H1 endpoint best eval40 | eval40 | 46.1720 | 600.6000 | 0.7000 | MLP BC aggregate eval40 | 0.3229 | 5.6300 | 0.0750 | return/length improve but fall worsens or is mixed |
| Flow-MBPO H1 endpoint final rollout10 | matched_video | 47.4617 | 625.6000 | 0.5000 | MLP BC matched rollout10 seed0 | -6.6666 | -62.8000 | 0.1000 | does not clear comparator gate |
| Flow-MBPO H1 endpoint best rollout10 | matched_video | 55.5533 | 707.6000 | 0.4000 | MLP BC matched rollout10 seed0 | 1.4249 | 19.2000 | 0.0000 | beats comparator on return/length and does not increase fall |
| Flow-MBPO trajchunk H3 final eval40 | eval40 | 48.7296 | 637.2250 | 0.5750 | MLP BC aggregate eval40 | 2.8805 | 42.2550 | -0.0500 | beats comparator on return/length and does not increase fall |
| Flow-MBPO trajchunk H3 final rollout10 | matched_video | 54.4904 | 694.0000 | 0.4000 | MLP BC matched rollout10 seed0 | 0.3621 | 5.6000 | 0.0000 | beats comparator on return/length and does not increase fall |
| Flow-MBPO trajchunk H3 low-synth final eval40 | eval40 | 47.5960 | 612.0000 | 0.6000 | MLP BC aggregate eval40 | 1.7469 | 17.0300 | -0.0250 | beats comparator on return/length and does not increase fall |
| Flow-MBPO trajchunk H3 low-synth final rollout10 | matched_video | 55.4222 | 707.2000 | 0.4000 | MLP BC matched rollout10 seed0 | 1.2938 | 18.8000 | 0.0000 | beats comparator on return/length and does not increase fall |

Key takeaways:

- The strongest scalar result is Flow-MBPO H1 endpoint final eval40: return 60.8721 versus MLP BC 45.8491 (+15.0230, +32.77%), length 759.30 versus 594.97 (+164.33, +27.62%), fall 0.45 versus 0.625 (-0.175).
- The matched video gate is weaker: H1 final rollout10 underperforms MLP BC video, while H1 best rollout10 slightly improves return/length and ties fall.
- Trajectory/chunk H3 variants show similar partial signal: some eval/video return and length gains, but no robust fall improvement over matched MLP BC video.
- Recent conservative AWR/AWAC/support-truncation diagnostics are negative and should not be counted as Flow-MBPO improvements.
