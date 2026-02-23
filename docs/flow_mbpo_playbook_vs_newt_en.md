# Flow-MBPO Training Tactics Integration and Fair Comparison Guide vs. Newt

> Purpose: Convert the 4 papers you specified into directly actionable tactics, experiment directions, and fair-comparison protocols, while accounting for your ongoing migration to `mjlab`.

---

## 1) Research Strategy (Define Winning Criteria First)

Your current practical objective can be split into three layers:

1. **Internal win**: Flow-MBPO stably and repeatedly outperforms your own PWM baseline.
2. **Mechanistic evidence**: Do not look at reward only; explain *why* and under what settings Flow works.
3. **External competitiveness**: Build a defensible fair comparison with Newt (`Learning Massively Multitask World Models for Continuous Control`).

---

## 2) Actionable Tactics Extracted from the Papers

## 2.1 Policy-side tactics (FPO / FPO++) you can adopt directly

| Tactic | From | Implementation suggestion for your project | Priority |
|---|---|---|---|
| `per-sample ratio` (not per-action) | FPO++ 2602.02481 | If you implement a policy-gradient version of flow policy, switch ratio clipping to sample-level | High |
| `ASPO` asymmetric trust region | FPO++ | Use PPO clipping for positive advantages; apply a more conservative penalty for negative advantages to suppress collapse | Medium-High |
| `zero-sampling` evaluation strategy | FPO++ | Train with random noise, but evaluate with `epsilon=0` to reduce deployment variance and latency | High |
| CFM/ratio clamp + gradient-preserving clamp | FPO++ Appendix | Add clamp logic when flow loss oscillates heavily to prevent NaN and exploding gradients | Medium-High |
| `Nmc` (tau, epsilon) sample count as an efficiency knob | FPO 2507.21053 | Start with a small fixed Nmc (e.g., 4/8) for efficient runs, then run Nmc ablations | Medium |
| Target parameterization (`epsilon-target` vs `velocity-target`) | FPO | Compare `epsilon-target` and `velocity-target` first to inspect scale stability | Medium |

Key insight:

- The core advantage of flow policy often appears in **under-conditioned / multimodal action** scenarios.
- If you only compare on fully-conditioned, low-uncertainty tasks, differences can be hard to observe.

## 2.2 World-model-side tactics (Newt + RWM-O) you can adopt directly

| Tactic | From | Implementation suggestion for your project | Priority |
|---|---|---|---|
| Demo-driven pretraining (pretrain all modules together) | Newt 2511.19584 | Do not pretrain encoder only; pretrain dynamics/reward/policy prior together | High |
| Fourfold demo usage (pretrain / constrained planning / oversample / BC regularization) | Newt | Even without a planner, keep 3 items: pretrain + oversample + actor BC regularization | High |
| 50/50 sampling between demo and online buffer | Newt | Add replay sampling ratio ablation directly (50/50, 30/70) | High |
| Reward/Value discretization (bins + CE) + log-space value | Newt | Your current reward path is mostly MSE; make reward head a formal ablation axis | Medium-High |
| Per-task gamma (based on episode length) | Newt | Avoid one global gamma that mismatches both short and long episodes in multitask settings | Medium-High |
| Ensemble epistemic uncertainty + reward penalty | RWM-O 2504.16680 | Add `r_tilde = r - lambda * u` on WM rollout reward to suppress hallucination exploitation | High |
| `lambda` penalty sweep (small is opportunistic, large is over-conservative) | RWM-O | Sweep `lambda in {0, 0.5, 1.0, 2.0}`; best is often in the middle | High |

Key insight:

- Your judgment is correct: **WM training details usually decide success more than policy architecture**.
- Flow policy can still work, but if WM is unstable, the actor will still receive poor gradients.

## 2.3 Most important "do-this-first" items for your project

| Priority Order | Item | Why |
|---|---|---|
| P0 | `WM uncertainty penalty` + `lambda` sweep | Directly suppresses model exploitation, often improves stability immediately |
| P0 | `demo oversampling` + `actor BC regularization` | Lowest engineering cost, strong effect in early learning |
| P1 | Reward head: MSE vs two-hot/CE | Avoid misattributing reward-pipeline differences to Flow |
| P1 | Flow policy per-sample ratio / zero-sampling | Improves flow-policy trainability and deployment stability |
| P2 | ASPO, CFM clamp, target parameterization | Further improve upper bound and stability |

---

## 3) Suggested Additional Experiment Tables (to complement your current spec)

You already have a good core table (`E1-E4`). Add three WM-focused tables:

## Table-WM1: Uncertainty and Conservatism

| Exp ID | Setting | Purpose | Expected |
|---|---|---|---|
| WM1-0 | Baseline (no uncertainty) | Reference group | More hallucination exploitation risk |
| WM1-1 | Ensemble uncertainty + `lambda=0.5` | Mildly conservative | Potential generalization gains |
| WM1-2 | Ensemble uncertainty + `lambda=1.0` | Moderately conservative | Often the best tradeoff |
| WM1-3 | Ensemble uncertainty + `lambda=2.0` | Strongly conservative | Might be overly conservative, return may drop |

## Table-WM2: Demo-usage strategy decomposition

| Exp ID | Pretrain | Oversample | BC reg | Purpose |
|---|---:|---:|---:|---|
| WM2-0 | X | X | X | Pure online reference |
| WM2-1 | O | X | X | Isolate pretraining effect |
| WM2-2 | O | O | X | Add data-distribution stabilization |
| WM2-3 | O | O | O | Newt-style full leverage |

## Table-WM3: Reward-pipeline alignment

| Exp ID | Reward head | Regression space | Purpose |
|---|---|---|---|
| WM3-0 | Scalar MSE | Raw reward | Current implementation |
| WM3-1 | Scalar MSE | Symlog/normalized | Low-cost stabilization |
| WM3-2 | Two-hot + CE | Binned/log-space | Align with Newt/PWM-like design |

---

## 4) How to make a "fair and defensible" comparison with Newt

## 4.1 State the baseline rule first

**If task suites differ (MMBench vs. your mjlab task suite), you cannot claim "overall superiority to Newt."**
You can only say:

- You outperform a Newt-style baseline on your defined mjlab benchmark, or
- You match/exceed Newt reported values on an MMBench subset.

## 4.2 Three-track comparison protocol (recommended to run all)

## Track A: Strict comparability (strongest evidence)

- Platform: Use official Newt code + MMBench directly.
- Setup: Match observation mode (state-only first), synchronized budget, same number of seeds, same aggregation.
- Goal: Align with Newt paper/CSV normalized score.

This is the only track where you can directly claim "better than Newt."

## Track B: Internal mjlab competition (engineering mainline)

- Platform: Everything in `mjlab`.
- Baselines: `MLP-WM + MLP-policy` (your PWM version) + `Newt-style trick ablations`.
- Goal: Make Flow-MBPO stably outperform baseline in mjlab first.

This solves your current iteration-speed bottleneck.

## Track C: Cross-platform transfer (high research value)

- Pretrain first on MMBench/TD-MPC2-like data (including demos), then transfer to mjlab tasks.
- Compare `from-scratch` vs `pretrain+finetune` on sample efficiency and wall-clock.

This ties your paper narrative to "generalist pretrain -> target adaptation."

## 4.3 Fields that must be locked for comparison

| Category | Must-align items |
|---|---|
| Data | Same demo source, same filtering rules, same train/eval split |
| Budget | Same env steps, same wall-clock cutoff, same update frequency |
| Model | Similar parameter scale (e.g., 5M/20M) and rollout budget (H, K, Kpol) |
| Eval | Same seeds, same eval episodes, same deterministic/stochastic protocol |
| Statistics | Same primary endpoint (AUC + final return) and same bootstrap/FDR |

---

## 5) Practical rollout plan considering mjlab

## 5.1 Minimal viable comparison (within 4 weeks)

1. Run `E1-E4` in `mjlab` on 3 representative tasks (low/mid/high dimension).
2. Add `WM1` (uncertainty penalty).
3. Add `WM2` (demo leverage).
4. Use 5 seeds per group for pilot; keep top 2 settings for 10-seed confirmatory runs.

## 5.2 "Competition-style" metric board you can use now

| Metric | Definition | Use |
|---|---|---|
| `Score@EqualSteps` | Normalized AUC at fixed env steps | Compare sample efficiency |
| `Score@EqualTime` | Final return at fixed wall-clock | Compare practical iteration efficiency |
| `Stability` | Collapse rate / NaN rate | Compare trainability |
| `Generalization` | Performance retention under OOD/noise/perturbation | Compare robustness |

---

## 6) Suggested paper narrative (avoid generic claims)

You can structure the main narrative as:

1. **Flow policy has already been shown feasible in the literature**, but in your setting the performance ceiling is primarily constrained by WM quality.
2. With Newt/RWM-O inspired training strategies (demo leverage + uncertainty-aware world model), both stability and peak performance of Flow-MBPO improve.
3. Under fair protocols (EqualSteps/EqualTime/EqualCompute), Flow-MBPO shows advantages in high-dimensional and long-horizon settings.

---

## 7) Next steps (pragmatic execution order)

1. Implement `WM uncertainty penalty` (`WM1`) and `demo oversampling` (`WM2`) in `mjlab` first.
2. In parallel, add flow-policy `zero-sampling` and per-sample ratio (if you use FPO-style updates).
3. Start Track B (mjlab mainline) for first stable results, then launch Track A (strict MMBench comparability against Newt).

---

## Sources

- Flow Matching Policy Gradients (arXiv:2507.21053): https://arxiv.org/abs/2507.21053
- Flow Policy Gradients for Robot Control (arXiv:2602.02481): https://arxiv.org/abs/2602.02481
- Learning Massively Multitask World Models for Continuous Control (arXiv:2511.19584): https://arxiv.org/abs/2511.19584
- Offline Robotic World Model Learning Robotic Policies without a Physics Simulator (arXiv:2504.16680): https://arxiv.org/abs/2504.16680
- Newt official repo README/examples: https://github.com/nicklashansen/newt
- Newt model card/checkpoints summary: https://huggingface.co/nicklashansen/newt
- MMBench dataset card: https://huggingface.co/datasets/nicklashansen/mmbench
- mjlab official repo/readme: https://github.com/mujocolab/mjlab
- mjlab docs home: https://mujocolab.github.io/mjlab/
- mjlab PyPI: https://pypi.org/project/mjlab/
