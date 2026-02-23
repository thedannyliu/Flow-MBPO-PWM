# Flow-MBPO @ mjlab Experiment and Ablation Spec (Execution-Ready)

> Version: `v1.0`  
> Frozen Date: `2026-02-18`  
> Purpose: Define exactly what experiments to run, why to run them, how to analyze statistics, and how to log results, so the rest of execution is just running and filling in tables.

---

## 0. Core Research Claims and Matching Limitations

We explicitly decompose the research narrative into testable hypotheses:

| Hypothesis ID | Research claim | Matching prior limitation (PWM/MLP) | What it means if supported |
|---|---|---|---|
| H1 | Flow WM outperforms MLP WM on high-dimensional tasks | MLP latent dynamics accumulates error faster under high-dimensional/complex dynamics | Flow dynamics is more stable for complex transitions |
| H2 | Flow Policy outperforms MLP Policy with a fixed WM | Gaussian/MLP policy has limited expressiveness | Flow policy offers more effective exploration or action manifold modeling |
| H3 | Flow gains are not only from "running longer/more compute" | Flow models are usually heavier per step | Gains remain under equal compute/time constraints |
| H4 | Flow is more stable for long horizons or accumulated model error | In PWM, surrogate gradients can degrade when H increases | Flow is less sensitive to rollout mismatch |
| H5 | Flow latent representation is more separable/useful | MLP latent may be insufficiently informative for downstream control | Latent probes and dynamics metrics support a mechanism-level explanation |

### Why run this table

- Convert the story from "seems better" into claims where each can be falsified.

### Insight after running

- The final paper narrative maps directly to H1-H5, avoiding a disconnected pile of results.

---

## 1. Overall Experiment Design (Layered Execution)

| Level | Goal | Seeds | Usage |
|---|---|---|---|
| L0 Smoke | Confirm runs complete, no API/NaN issues | 1-2 | Development only, no scientific conclusions |
| L1 Pilot | Observe trends, filter obviously bad settings | 3-5 | Decide whether to proceed to full statistical runs |
| L2 Confirmatory | Formal hypothesis testing | 10 (main) / 5 (secondary ablations) | Reported conclusions |

### Why run this table

- Separate engineering debugging from scientific conclusions, avoiding claims from smoke results.

### Insight after running

- Clear distinction between "runnable" settings and "statistically reliable" settings.

---

## 2. mjlab Task Panel Definition (Freeze First, Then Run)

First freeze a task panel that covers dimensions and difficulties; do not change tasks while running.

| Panel Slot | Dimension/difficulty goal | Selection rule | Actual task_id (to fill) | Obs Dim | Act Dim | Episode Len |
|---|---|---|---|---:|---:|---:|
| MJ-L1 | Low-dimensional control | obs/act in the lowest 30% of panel |  |  |  |  |
| MJ-L2 | Low-dimensional control | Different dynamics type from MJ-L1 |  |  |  |  |
| MJ-M1 | Mid-dimensional locomotion | obs/act near median range |  |  |  |  |
| MJ-M2 | Mid-dimensional locomotion | Different reward shaping from MJ-M1 |  |  |  |  |
| MJ-H1 | High-dimensional humanoid/complex | top 30% in obs/act dimensions |  |  |  |  |
| MJ-H2 | High-dimensional humanoid/complex | Different contact pattern from MJ-H1 |  |  |  |  |

### Why run this table

- H1/H5 are fundamentally about dimension and complexity; without a frozen panel, cherry-picking risk is high.

### Insight after running

- Enables cross-dimension trend analysis instead of accidental single-task outcomes.

---

## 3. Main Experiments: 2x2 Factorial Design (Mandatory)

Fix all settings except WM/Policy architecture and run the full 2x2 design.

| Exp ID | WM | Policy | Main mapped hypotheses | Seeds (L2) | Budget | Primary metrics |
|---|---|---|---|---:|---|---|
| E1 | MLP | MLP | Baseline | 10 | Fixed env steps + fixed wall-clock report | AUC, Final Return, Success |
| E2 | Flow | MLP | H1 | 10 | Same as E1 | Same + WM losses |
| E3 | MLP | Flow | H2 | 10 | Same as E1 | Same + action stats |
| E4 | Flow | Flow | H1+H2 interaction | 10 | Same as E1 | Same |

Run every `Exp ID` across all tasks `MJ-L1..MJ-H2`.

### Why run this table

- This is the core evidence: cleanly separates WM and Policy factors to avoid confounded interpretation.

### Insight after running

- `(E2-E1)` = pure Flow WM gain; `(E3-E1)` = pure Flow Policy gain; `(E4-E2)` tests interaction.

---

## 4. Limitation-Driven Ablations (Answer "Why it works")

## 4.1 Horizon / Rollout Stability (H4)

| Exp ID | Fixed setting | Sweep axis | Values | Seeds | Metrics | Expected observation |
|---|---|---|---|---:|---|---|
| A1 | Best default from E1/E2 | Horizon H | 4, 8, 16 | 5 | Reward, ESNR, collapse rate | Flow degrades more slowly at larger H |
| A2 | Flow-WM only | Substeps K | 2, 4, 8 | 5 | Reward vs time, WM loss | Find performance/compute tradeoff |
| A3 | Flow-WM only | tau sampling | uniform, midpoint | 5 | Stability and variance | Determine whether training signal is smoother |

### Why run this table

- Flow is often challenged as "just bigger/slower"; this block tests under which dynamics conditions it actually helps.

### Insight after running

- Identifies effective zones (e.g., long horizon) and transferable design rules.

## 4.2 Reward Pipeline Alignment (avoid attribution errors)

| Exp ID | Fixed setting | Sweep axis | Values | Seeds | Metrics | Purpose |
|---|---|---|---|---:|---|---|
| A4 | E1/E2 | Reward loss | MSE vs Two-hot/CE (if implemented) | 5 | Reward, WM reward loss | Rule out fake gains caused by reward-head mismatch |
| A5 | E2 | FM source state | rollout-based vs teacher-forced | 5 | ESNR, stability, final return | Validate correction-flow mechanism |

### Why run this table

- Prevent misattributing pipeline differences as Flow-architecture advantages.

### Insight after running

- Gives a cleaner answer on whether Flow dynamics itself contributes.

---

## 5. Mechanism Validation Experiments (support representation narrative, H5)

## 5.1 Latent Probe Suite

| Exp ID | Probe task | Training data source | Measure | Comparison groups |
|---|---|---|---|---|
| M1 | `z_t -> r_t` linear regression | replay samples | `R^2` / MSE | E1 vs E2 |
| M2 | `z_t -> done_t` classification | replay samples | AUC / F1 | E1 vs E2 |
| M3 | `z_t, a_t -> z_{t+1}` linear approximation error | replay samples | one-step error | E1 vs E2 |
| M4 | task ID / command prediction (multi-task) | multi-task buffer | accuracy / NMI | E1 vs E2 |

### Why run this table

- If the narrative is "Flow latent is richer," you need intermediate evidence beyond final reward.

### Insight after running

- If high-dimensional tasks show `E2>E1` and probe metrics also improve, representation-based explanation becomes stronger.

## 5.2 Jacobian / Sensitivity Metrics (optional but strong)

| Exp ID | Metric | Definition | Purpose |
|---|---|---|---|
| M5 | `||∂z_{t+1}/∂a_t||` distribution | local controllability proxy | Check how controllable latent state is w.r.t. action |
| M6 | `||∂V/∂z||` and actor grad SNR | gradient quality proxy | Link to FoG trainability |

### Why run this table

- Translate "Flow is more stable" into measurable gradient/sensitivity signals.

### Insight after running

- Explains why reward gains are large on some tasks and limited on others.

---

## 6. Robustness / OOD Experiments (H4/H5 extension)

| Exp ID | Shift type | Setting | Comparison groups | Seeds | Metrics |
|---|---|---|---|---:|---|
| R1 | Observation noise | add test-time obs noise (3 levels) | E1/E2/E3/E4 | 5 | degradation slope |
| R2 | Dynamics perturbation | small shifts in mass/friction/delay | E1/E2/E3/E4 | 5 | robustness gap |
| R3 | Partial observability | hide part of observation channels | E1/E2/E3/E4 | 5 | success rate and recovery ability |

### Why run this table

- If Flow only overfits the in-distribution setup, practical value is limited.

### Insight after running

- Verifies whether Flow gains persist under distribution shift.

---

## 7. mjlab Migration Benefit and Compute Fairness (H3)

## 7.1 Simulator and Throughput Calibration

| Exp ID | Goal | Setting | Output |
|---|---|---|---|
| S0 | API/data correctness | 2 seeds, 300 epochs | No NaN, replay buffer healthy, eval healthy |
| S1 | Throughput benchmark | compare env FPS under fixed config | `env_steps/s`, `train_steps/s` |
| S2 | Time breakdown | profiling | env.step% / WM update% / critic% |

### Why run this table

- Confirm whether slowdown truly comes from the simulator before deciding optimization focus.

### Insight after running

- If env share is small, replacing simulator alone will not speed up much; training hotspots must be optimized too.

## 7.2 Fair-comparison protocols (must follow)

Every primary result must be reported under all three protocols:

| Protocol | Control method | Usage |
|---|---|---|
| Equal-EnvSteps | fix interaction steps | compare sample efficiency |
| Equal-WallClock | fix training time | compare practical iteration efficiency |
| Equal-ComputeIndex | fix compute proxy mapped from `H/K/Kpol` | remove "more compute" confound |

### Why run this table

- Prevent reviewers from attributing gains to extra computation.

### Insight after running

- You can clearly state where Flow still holds under each fairness definition.

---

## 8. Statistical Rigor Protocol (fixed analysis plan)

## 8.1 Primary and secondary endpoints (pre-register)

| Level | Metric | Definition |
|---|---|---|
| Primary | `Normalized AUC` | reward-curve area under fixed step interval (task-normalized) |
| Primary | `Final Return` | average over the final evaluation window |
| Secondary | `Success Rate` | average task success |
| Secondary | Stability | collapse rate / NaN rate |
| Mechanistic | Probe / Jacobian metrics | see M1-M6 |

## 8.2 Test methods (recommended fixed choices)

| Question type | Test | Report contents |
|---|---|---|
| Same task, two-method comparison | Paired bootstrap CI (seed-paired) | effect size + 95% CI |
| Multi-task aggregation | Hierarchical bootstrap (task x seed) | overall effect + CI |
| Multiple comparisons | Benjamini-Hochberg FDR | q-value |
| Practical significance | minimum effect threshold `>=5%` relative gain | whether threshold is met |

## 8.3 Decision threshold (avoid p-hacking)

A hypothesis is considered "supported" only if all are satisfied:

1. At least one primary endpoint is significant (CI does not cross 0, and remains valid after FDR).
2. Practical effect threshold is met (`>=5%`).
3. Direction is consistent in at least 2 task bins (L/M/H).

---

## 9. Reproducibility and Execution Hygiene

| Category | Required items |
|---|---|
| Code | record git commit hash, dirty state, patch summary |
| Config | save Hydra resolved config + checksum |
| Env | save `pip freeze`/`uv.lock`, CUDA driver, GPU model |
| Seed | fix seed list (10 for main experiments) and share across methods |
| Runtime | record node ID, start/end times, rerun reason on failure |
| Artifact | standardize paths for checkpoints, eval csv, raw logs |
| Failure policy | OOM/hardware failures can be rerun; algorithmic NaN cannot be silently rerun and must be logged as failure |

---

## 10. Result-Filling Templates (fill directly after runs)

## 10.1 Run-level registration table

| Run ID | Exp ID | Task | Seed | Commit | Config Hash | Status | Final Return | AUC | Success | Runtime(h) | Node | Notes |
|---|---|---|---:|---|---|---|---:|---:|---:|---:|---|---|
|  |  |  |  |  |  |  |  |  |  |  |  |  |

## 10.2 Aggregated statistics table (per Exp ID x Task)

| Exp ID | Task | N | Mean Final Return | Std | Mean AUC | 95% CI vs Baseline | Effect Size(%) | Verdict |
|---|---|---:|---:|---:|---:|---|---:|---|
|  |  |  |  |  |  |  |  |  |

## 10.3 Hypothesis summary table (paper-ready)

| Hypothesis | Support level (Yes/Partial/No) | Main evidence (Exp IDs) | Counterexamples/limits | Next step |
|---|---|---|---|---|
| H1 |  |  |  |  |
| H2 |  |  |  |  |
| H3 |  |  |  |  |
| H4 |  |  |  |  |
| H5 |  |  |  |  |

---

## 11. Recommended Execution Order (you can run this directly)

1. `S0 -> S1 -> S2` (first ensure mjlab migration works and really speeds things up)
2. Run `E1-E4` on `MJ-L1/MJ-M1/MJ-H1` for L1 pilot (3-5 seeds)
3. If pilot passes, run `E1-E4` on full task panel for L2 confirmatory (10 seeds)
4. Run `A1-A5` (5 seeds) to identify effective and failure regions
5. Run `M1-M6` to complete mechanism evidence
6. Run `R1-R3` to complete robustness
7. Fill `10.1~10.3` and export main/appendix tables for the paper

---

## 12. Relationship to Existing Documents

- This document is the executable experiment spec;
- `docs/experiment_log.md` is the run-by-run logbook and result registry;
- `docs/master_plan.md` is the high-level research roadmap.

Recommendation: map every new experiment to an `Exp ID` before job submission.
