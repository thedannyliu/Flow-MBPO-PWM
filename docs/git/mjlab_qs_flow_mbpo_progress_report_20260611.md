# MJLab QS / Flow-MBPO Progress Report

Generated: 2026-06-11 EDT

Scope: Velocity Flat Unitree G1 MJLab QS, PWM fidelity/adapter work, and the
state-observation Flow-MBPO track. This version is written as an advisor-facing
research memo: the emphasis is on what the experiments imply, not on listing
every completed run.

## Conclusion & Research Signals

- **Flow is useful only after changing how it is used.**
  Architecture-only swaps inside the PWM-style extraction path collapse, while
  short-horizon Flow-MBPO replay improves return and episode length over BC.
  The signal is therefore about using Flow as controlled synthetic data, not as
  an unconstrained imagined optimizer.

- **The main blocker is not only data quality or one-step WM fit.**
  BC/data ablations did not produce a stronger baseline, and PWM diagnostics
  show reasonable fixed-window reward/dynamics fit while real rollout still
  collapses. The failure looks like learned-model exploitation under policy
  optimization, especially around unsupported states.

- **The improvement is still incomplete because fall rate does not move.**
  The strongest Flow-MBPO rows improve return/length, but they do not reliably
  reduce fall rate. This suggests the model is finding higher-reward behavior
  before it learns the failure boundary that matters for humanoid locomotion.

- **Synthetic replay is not useful by volume alone.**
  In the H1 exact-replay sweep, increasing synthetic batch ratio made short
  real eval worse. The useful question is therefore not "more model data?", but
  "which synthetic transitions are safe and decision-relevant?"

- **Support/OOD distance is the clearest next signal.**
  q90 support distance strongly separates terminated episodes from timeout
  episodes, and the signal is mostly state/command OOD rather than action OOD.
  This points to support-aware generation, replay termination/reward penalties,
  or conservative Q over generated out-of-support states/actions.

- **Current direction: Pessimistic short-horizon Flow-MBPO.**
  Treat Flow as a short-horizon trajectory generator with calibrated pessimism,
  not as a drop-in replacement for the whole PWM-style imagined optimizer.

Advisor-level takeaway:

> We have evidence that Flow-generated short rollouts can improve return/length,
> but the research problem is now fall-boundary control. The next experiments
> should test whether support/OOD-aware pessimism converts the partial Flow-MBPO
> signal into a robust fall-rate improvement.

## Key Results

Only rows that change the research decision are shown here. Other comparisons
are summarized in the ablation sections.

| Row | Protocol | Return | Length | Fall | What it tells us |
| --- | --- | ---: | ---: | ---: | --- |
| Expert collector | rollout reference | `82.6090` | `1000.00` | `0.000` | Target is stable locomotion, not just higher return. |
| Best scalar BC | eval40, 1000-step | `45.8491` | `594.97` | `0.625` | BC is usable but still falls often; it is a floor, not a solution. |
| Matched BC video | roll10, 1000-step, seed0 | `54.1283` | `688.40` | `0.400` | Video fall is a separate robustness gate; compare claims within matched protocol. |
| PWM-style extraction | representative rollout | about `-1.10` | about `54.67` | `1.000` | Unconstrained imagined policy extraction is unsafe on MJLab. |
| Flow-MBPO H1 endpoint | final eval40 | `60.8721` | `759.30` | `0.450` | Strong scalar signal, but not robust enough to claim. |
| Flow-MBPO H3 trajectory/chunk | final roll10 | `54.4904` | `694.00` | `0.400` | Best current direction: return/length improve, fall only ties BC. |

![Key results ladder](../assets/mjlab_qs_progress_20260611/key_results_ladder.png)

Key read:

- Flow-MBPO is not failing like PWM; it produces meaningful locomotion.
- The remaining gap is specific: fall robustness, not generic learning.
- Any next method should be judged by whether it lowers fall, not only whether
  it raises return.

## Pipeline (Debug Map)

Algorithm: MJLab QS Flow-MBPO Training and Evaluation Pipeline

Input:
    Velocity Flat Unitree G1 MJLab QS data.

    Main dataset:
        - `d_qs_core_h16.pt`
        - quality IDs: `expert`, `expert_noisy`, `medium`, `random_smooth`
        - state observation
        - command / task conditioning
        - action
        - reward
        - done / termination / truncation labels -> Current QS shards have effectively no positive done/fall labels, so learned fall heads are not reliable yet.

    Reference policies:
        - expert collector -> Target: high return, full episode length, zero fall.
        - expert-noisy collector -> Stable noisy target.
        - medium collector -> Data-quality midpoint, not a target.
        - random_smooth -> Failure floor.

Model:
    Baseline policy:
        - MLP BC policy trained from QS windows.
        - Expert+noisy uniform BC is the current practical warm start. -> Use it as both the initialization and the minimum baseline.

    World model candidates:
        - MLP one-step world model.
        - Flow endpoint world model.
        - Flow residual / frozen-MLP residual variants.
        - Flow trajectory/chunk world model. -> Current strongest signal comes from short trajectory/chunk replay, not pure architecture replacement.

    Policy update modules:
        - AWR / AWAC-style weighted behavior update.
        - Optional conservative Q critic. -> Useful as targeted pessimism, not as a generic extra loss.
        - Optional action-deviation / BC-anchor losses. -> Simple actor-side safeguards did not solve the fall gate.

Training Pipeline:

Step 1: Establish Collector And BC Baselines
    Use existing MJLab collector checkpoints and QS windows. -> We do not recollect the main dataset in the current report.

    Evaluate:
        - expert / expert-noisy / medium / random references
        - BC variants from QS windows

    Result:
        - expert collector defines the target behavior.
        - expert+noisy uniform BC is the scalar baseline.
        - matched seed0 BC roll10 is the video baseline.

Step 2: Diagnose PWM-Style Policy Extraction
    Original path:
        dataset windows
        -> learned world model
        -> long-horizon imagined actor/critic optimization
        -> extracted policy

    Current finding:
        - fixed-window WM reward/dynamics diagnostics can look reasonable.
        - imagined return / critic value can rise.
        - real MJLab rollout collapses. -> The failure is policy/critic exploitation of unsupported model regions.

Step 3: Generate Short-Horizon Synthetic Replay
    Start from real QS states and roll out a BC-warmstarted policy through a WM ensemble.

    Synthetic rollout settings:
        - H=1 endpoint replay -> cheap root-cause / ratio diagnostic.
        - H=3 trajectory/chunk replay -> strongest return/length signal.
        - H=5 and broader variants -> diagnostic only until H=3 reduces fall.

    Replay stores:
        - state / command / action
        - model reward
        - conservative reward
        - next state
        - done / uncertainty / support-risk metadata

    Modification:
        Original PWM used the model mainly for imagined optimization.
        -> We use the model to generate explicit short synthetic replay, then update the policy with model-free-style losses.

Step 4: Policy Update From BC Warm Start
    Mix real QS transitions and synthetic transitions in AWR/AWAC/CQL-style updates.

    Current useful settings:
        - BC warm start.
        - low synthetic ratios are safer than broad synthetic mixing.
        - final and true-best checkpoints are saved.

    Negative signals:
        - more H1 synthetic replay did not improve short real eval.
        - action-deviation regularization hurt scalar eval and did not improve video fall.
        - broad conservative/AWAC/support-truncation rows are diagnostic, not wins.

Step 5: Add Pessimism / Support / OOD Control
    Because fall labels are missing, use calibrated support distance as a proxy.

    Current support signal:
        - q90 support distance separates terminated vs timeout real episodes.
        - the signal is mostly state/command OOD, not action OOD.

    Next modification:
        Current support penalties were mostly actor-side or post-hoc.
        -> Move support risk into synthetic rollout generation, replay termination/reward, or conservative Q over generated out-of-support states/actions.

Evaluation Pipeline:
    For every serious candidate:
        1. Save final checkpoint.
        2. Save true-best / snapshot checkpoints when real-eval selection is enabled.
        3. Run 40-episode real MJLab eval at max_steps=1000.
        4. Run 10-episode, 1000-step rollout MP4/W&B video.
        5. Compare against expert, expert-noisy, medium, scalar BC, and matched BC video.

    Claim gate:
        - return must match or exceed the relevant BC baseline.
        - episode length must match or exceed the relevant BC baseline.
        - fall rate must be strictly lower than the relevant BC baseline.
        - W&B run, git SHA, dataset/version, checkpoint paths, command, and notes must be recorded.

    Current outcome:
        Flow-MBPO H1/H3 can improve return and length.
        -> No general policy-improvement claim yet because fall rate has not improved over matched BC.

## Ablations & Signals

### 1. Architecture / Policy Path

Signal:

- PWM-style MLP-vs-Flow swaps collapse under policy extraction. This suggests
  the failure is not solved by replacing the architecture inside the same
  imagined optimizer.
- Flow-policy BC underperforms MLP BC (`43.3222 / 564.46 / fall 0.675` versus
  BC `45.8491 / 594.97 / fall 0.625`). The policy class alone is not the
  immediate bottleneck.
- Flow-MBPO H1/H3 are more informative because they test Flow as short-horizon
  synthetic data rather than as an unconstrained simulator.

![Flow video gate deltas](../assets/mjlab_qs_progress_20260611/flow_video_gate_deltas.png)

-> Interpretation: Flow should be evaluated as a distributional short-horizon
rollout model with pessimism. The next architecture comparison should be
support-aware Flow-MBPO vs matched non-Flow MBPO, not another broad PWM-style
2x2.

### 2. Checkpoint Selection / Training-Time Metrics

Signal:

- Short in-training real eval can be misleading.
- H1 strongest run had weak in-training real eval (`best_real_return=17.10`)
  but strong final eval40 (`60.87`).
- H3 trajectory/chunk had low in-training best real eval (`20.88`) while final
  eval40 reached `48.73`.
- Candidate snapshot search found scalar-pass checkpoints, but none passed the
  joint scalar+video gate because video fall tied or worsened.

![Runtime and selection signal](../assets/mjlab_qs_progress_20260611/runtime_and_selection_signal.png)

![Checkpoint gate signal](../assets/mjlab_qs_progress_20260611/checkpoint_gate_signal.png)

-> Interpretation: 8-episode in-training eval should be a reject filter, not
the selection criterion. Formal snapshot ranking must include eval40 and roll10
fall. Otherwise, we may discard useful checkpoints or promote unstable ones.

### 3. Compute / Runtime

Signal:

| Row | AWR train wall clock | Synthetic transitions | Signal |
| --- | ---: | ---: | --- |
| H1 endpoint `r224/s32` | `35.8s` | `256` | Cheap, useful for root-cause tests; strongest scalar surprise. |
| H3 trajectory `r224/s32` | `93.2s` | `768` | More expensive, but better trajectory signal. |
| H3 low-synth `r240/s16` | `72.7s` | `768` | Saves update cost but does not reduce fall. |
| H3 action-deviation | `72.7s` | `768` | Extra safeguard cost is not justified by current evidence. |

-> Interpretation: use H1 for cheap diagnostics and H3 only when testing a new
fall-risk mechanism. Do not spend formal W&B/video budget on actor-only
regularizers that already fail scalar or fall gates.

### 4. Dataset Composition

Signal:

- Expert+noisy uniform BC remains the best practical anchor.
- Expert-only is worse, likely because it lacks recovery/coverage.
- Medium data hurts when mixed naively; filtering/downweighting helps but still
  does not beat expert+noisy.
- Flow policy does not rescue BC.

![Dataset and policy ablation](../assets/mjlab_qs_progress_20260611/dataset_policy_ablation.png)

-> Interpretation: the next phase should not be another BC micro-sweep. If
medium or failure data is used, it should be targeted recovery/fall-boundary
data, not uniform BC data.

### 5. Real/Synthetic Ratio And Replay Quality

Signal from the H1 exact replay ratio sweep:

| Real/synthetic batch | Mean best 8ep real eval |
| --- | ---: |
| `256/0` | `22.75` |
| `248/8` | `21.52` |
| `224/32` | `20.70` |
| `192/64` | `19.13` |

![Synthetic ratio signal](../assets/mjlab_qs_progress_20260611/synthetic_ratio_signal.png)

Additional replay diagnostic:

- top synthetic reward decile is about `+2.34` above nearest real one-step
  reward.
- policy action drift from BC is tiny in AWR diagnostics.

-> Interpretation: the problem is not simply large actor drift. Synthetic
replay may contain attractive but unrealistic reward regions, and AWR can
overweight them even when the actor moves only slightly. Synthetic replay needs
quality control before ratio tuning.

### 6. Support / OOD / Fall-Risk

Signal:

- QS shards have effectively no positive done/fall labels.
- q90 support distance separates real terminated episodes from timeout episodes.
- The separation remains when action weight is removed, so the risk is mostly
  state/command OOD.

![Support distance fall proxy](../assets/mjlab_qs_progress_20260611/support_distance_fall_proxy.png)

-> Interpretation: support distance is currently the best available fall-risk
proxy. Actor-only support penalties are too indirect; the risk should affect
model rollout generation, replay termination/reward, or conservative Q.

## Problems & Future Directions

### Where We Are Stuck

- **Fall gate:** Flow-MBPO improves return/length before it improves fall.
- **Replay trust:** H1 replay has high-reward synthetic regions that look
  suspicious relative to nearest real transitions.
- **Selection noise:** short in-training eval and formal eval can disagree.
- **Missing labels:** current QS windows do not provide positive fall labels for
  direct fall-head calibration.
- **Objective weakness:** AWR actor movement can be tiny, so better weighting
  alone may not be enough to move away from failure modes.

### Directions Worth Advisor Feedback

1. **Support-aware generation as the next main experiment.**
   Stop, freeze, or penalize synthetic branches when calibrated q90 support risk
   rises during generation. This directly targets the observed failure signal.

2. **Conservative Q on generated out-of-support states/actions.**
   Use CQL only where the support proxy says the model is leaving the data
   manifold. This may be more targeted than global actor regularization.

3. **Matched non-Flow MBPO baseline.**
   Run an MLP MBPO row under the same short-rollout, replay, update, and gate
   protocol. This tells us whether the useful signal is Flow-specific or comes
   from the MBPO formulation.

4. **Fall-positive or near-fall data.**
   If feasible, collect or mine fall/recovery segments. This would let us move
   from support-distance proxy to calibrated fall-risk prediction.

5. **Gate-aware checkpoint ranking.**
   Rank snapshots by eval40 + roll10 fall, not short in-training return. This is
   necessary before scaling seeds because the current selection signal is noisy.

### Current Recommendation

Do not claim policy improvement yet. The next claim-worthy experiment should be:

```text
Flow trajectory/chunk H3
-> support-aware synthetic generation or support-targeted conservative Q
-> BC-warmstarted AWR/AWAC update
-> final + true-best eval40
-> final + true-best roll10 video
-> beat matched BC return/length and strictly reduce fall
```

If this fails, the advisor-level question becomes whether MJLab needs explicit
fall/recovery data before model-based policy improvement can be reliable.

## Appendix

### Primary Sources

- `docs/EXPERIMENT_LEDGER.md`
- `results/master_policy_comparison.csv`
- `docs/goals/mjlab_qs_rollout_policy_improvement_20260528.md`
- `docs/goals/flow_mbpo_top_conf_research_plan_20260531.md`
- `docs/git/experiment_results_insights_summary_20260603.md`
- `docs/git/experiment_results_insights_summary_20260604.md`
- `docs/design/flow_mbpo_v0.md`
- `docs/design/flow_mbpo_v1_pessimistic.md`

### Evidence Standard

A policy-improvement claim requires:

- final and true-best checkpoints;
- 40-episode real MJLab eval at `max_steps=1000`;
- 10-episode, 1000-step MP4/W&B rollout;
- return and length at least matching the relevant BC baseline;
- fall rate strictly lower than the relevant BC baseline;
- W&B run, git SHA, dataset/version, checkpoint paths, command, and notes.

### Figure Assets

Generated figures live under:

```text
docs/assets/mjlab_qs_progress_20260611/
```

Files:

- `key_results_ladder.png`
- `flow_video_gate_deltas.png`
- `runtime_and_selection_signal.png`
- `checkpoint_gate_signal.png`
- `dataset_policy_ablation.png`
- `synthetic_ratio_signal.png`
- `support_distance_fall_proxy.png`
