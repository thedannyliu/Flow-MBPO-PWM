Title: Flow-MBPO / PWM Master Plan — Flow World Models and Flow Policies

This document is the development reference for this repository: code should follow this plan, but we intentionally avoid over-specifying low-level engineering details (e.g., exact directory layouts or script names).

---

**0) Storyline and Research Questions**

- 0.1 2×2 grid and A/B/C stages
  - We organize settings along two axes:
    - Horizontal: **offline vs. online** training.
    - Vertical: **single-task vs. multi-task**.
  - On top of this grid we define three stages:
    - **A – Offline multi-task pretraining**:
      - Train a large multi-task world model (PWM-style or Flow-based) on offline datasets (e.g., MT30/MT80, RWM/Newt).
    - **B – Online multi-task post-training/fine-tuning**:
      - Starting from A or from scratch, interact with multiple tasks online and update the world model and policy.
    - **C – Online single-task specialization**:
      - Starting from A or B, adapt the world model and policy to a single task (e.g., dflex_ant, a specific Meta-World task).
  - Example paths:
    - `A → C`: pretrain offline, then specialize directly to a single task.
    - `B → C`: online multi-task training, then specialize.
    - `A → B → C`: full pipeline (pretrain → online multi-task → specialization).

- 0.2 Where Flow Matching can be inserted
  - **World model (WM)**:
    - Replace PWM’s MLP dynamics with a Flow-matching latent ODE, keeping the same high-level FoG loop.
    - This is our current Phase 1 / 1.5 focus: Flow WM vs. MLP WM as surrogate dynamics.
  - **Policy**:
    - Replace PWM’s Gaussian policy with a Flow-parameterized policy (ODE-based at first, with optional FM objective later).
    - This is Phase 2: Flow policy vs. Gaussian policy under the same FoG + WM framework.
  - Combined:
    - Baselines and variants:
      - `no-FM WM + no-FM policy` (PWM baseline).
      - `FM WM + no-FM policy` (Flow WM only).
      - `no-FM WM + FM policy` (Flow policy only).
      - `FM WM + FM policy` (both Flow WM and Flow policy).

- 0.3 Core research questions
  - **RQ1 – Flow WM as surrogate dynamics for FoG (C in the grid)**  
    Under fixed parameter and compute budgets, can Flow-based world models provide *better surrogate dynamics* for PWM-style FoG than MLP WMs?
    - Metrics:
      - ESNR of actor updates.
      - Smoothness proxies (Jacobian norms `∥∂F/∂z∥`, `∥∂F/∂a∥`).
      - Final reward / stability on contact-rich tasks (e.g., dflex_ant), under matched or normalized compute.
  - **RQ2 – Flow-based policy in FoG + WM**  
    In the PWM FoG framework (world model + actor–critic), does a Flow-parameterized policy (ODE-based) improve sample efficiency or final reward compared to a Gaussian policy, holding the world model fixed?
    - Phase 2 focuses on Flow-**architecture** policy with RL loss only (ODE policy), not a full conditional flow-matching objective.
    - A future extension (Phase 2.5) is to integrate an FPO-style FM objective for the policy explicitly.
  - **RQ3 – Where in A/B/C does Flow help most?**  
    Across stages A (offline MT pretrain), B (online MT), and C (online ST spec), and across paths `A→C`, `B→C`, `A→B→C`, in which stage/path does Flow (WM and/or policy) provide the largest marginal gain over PWM-style baselines?
    - Comparisons within the PWM repo use PWM baselines as reference; once stable, we will port selected variants to RWM/Newt.
  - We **do not** expect Flow to help uniformly:
    - On low-dimensional, already-smooth tasks, Flow’s extra compute may bring little or no benefit over MLP WMs/policies.
    - On extremely contact-heavy tasks, overly aggressive smoothing (large `K`, strong regularization) may *increase* optimality gaps by washing out critical contact events.
  - To test these failure modes, we will:
    - Include at least one “simpler” control task as a negative control, where Flow is not expected to win.
    - Treat Flow hyperparameters (`K`, `H`, regularization) as smoothing knobs and explicitly look for regimes where Flow hurts as well as helps.

- 0.4 Minimal viable path (MVP) for this repository
  - To keep the project tractable on single L40S GPUs, we define an MVP path:
    - **MVP-Phase 1** (dflex_ant @ 5M):
      - Compare `MLP WM + MLP policy` (PWM baseline) vs. `Flow WM + MLP policy` with `H=16`, `K=4`, Heun.
    - **MVP-Phase 1.5** (dflex_ant @ 5M):
      - Sweep a *small* grid: `H ∈ {8,16}` × regularization `{base,strong}` with `K=4`, Heun.
      - Select a canonical Flow WM config balancing ESNR, smoothness proxy, reward, and compute.
    - **MVP-Phase 2** (dflex_ant @ 5M, canonical Flow WM):
      - Compare `Flow WM + Gaussian policy` vs. `Flow WM + Flow ODE policy` with `K_policy = 2`.
      - Measure policy ESNR, learning speed, and final reward.
  - Multi-task experiments (Phase 3) and full A/B/C pipelines (Phase 4) are *stretch goals* for later iterations or follow-up work; they should not block the MVP.

---

**1) High-Level Goals and Hypotheses**

- Goals
  - Build a family of Flow-based model-based policy optimization algorithms on top of PWM’s first-order training, starting in this PWM-based repo and later porting to RWM/Newt. Here “MBPO” is meant in this broader sense, not the specific MBPO algorithm.
  - Systematically compare Flow vs. MLP in both the world model and the policy, under matched parameter and compute budgets.
  - Understand when Flow helps most: contact-rich locomotion, multi-task transfer, and offline→online adaptation.
- Working hypothesis
  - Flow-based world models and policies can provide *better-behaved surrogate dynamics* for first-order optimization than the baseline MLP WM, especially on contact-rich locomotion and multi-task transfer. The key bet is that continuous-time velocity fields and ODE integration allow us to *control the degree of smoothing*: keeping enough expressivity for contacts while inducing a smoother optimization landscape and higher ESNR than zeroth-order / PPO-style updates or a purely feedforward MLP dynamics. The target is to exceed, not just match, current SoTA under transparent compute and parameter reporting.
- Non-Goals
  - Do not redesign PWM’s high-level algorithm from scratch: we preserve the overall structure of short-horizon first-order optimization on a learned world model, with an actor–critic loop and TD(λ) critic.
  - We *do* allow substantial changes to the internal world model architecture and, in later phases, to the policy architecture itself. This is a new family of algorithms inspired by PWM, not just a minor variant.
  - Do not commit to a single Flow-policy architecture at this stage; Phase 2 keeps the actor interface compatible with PWM and leaves architectural choices flexible.
  - Do not fix minor engineering details (precise directory layout, helper script names) in this document.

---

**2) Core Invariants (All Phases)**

- First-order training
  - Policies are always trained via first-order gradients (FoG) through short-horizon rollouts in the learned world model, as in PWM.
  - The real environment is only used to collect data into a replay buffer; policy gradients flow through the world model, not the environment.
- Responsibility separation
  - Models implement latent transition and reward functions exposed via: `encode`, `next`, `reward`, `step`.
  - For MLP WMs, `next` is a single MLP call. For Flow WMs, `next` may internally perform multiple ODE substeps (macro step), but the *algorithm* still reasons in terms of horizon `H` over these macro steps.
  - Algorithms own rollout, loss definitions, and optimization loops (`pwm.algorithms.pwm.PWM`).
- No duck typing
  - Hydra configuration selects world model and actor types explicitly (`world_model_config._target_`, `actor_config._target_`).
  - No new `hasattr`-based branching in user code (the existing optimizer setup hack should remain isolated and not spread).
- Fair comparison
  - Parameter-count parity between MLP and Flow variants: world model (encoder + dynamics + reward) parameter counts differ by at most ~2% relative. This is a *soft* constraint to keep capacity comparable, not an iron rule that overrides numerical stability.
  - Compute-aware reporting: always log FLOP proxies (e.g. number of velocity evaluations via substeps `K` per actor update) and wall-clock times.
  - All comparisons must report both *raw performance* (reward / success) and *compute-normalized* performance (e.g., reward per 10⁹ FLOPs or per GPU-hour). Where possible, we will design ablations that approximately match compute between MLP and Flow.
  - **Baseline immutability**: The original PWM baseline code and configs (`pwm_5M_baseline_final.yaml`, `dflex_ant.yaml`) must NOT be modified except for environment-specific adaptations (e.g., device settings). All experimental variants (Flow-MBPO) must match the baseline's hyperparameters (`wm_batch_size=256`, `wm_buffer_size=1_000_000`, `num_envs=128`, `max_epochs=15_000`) to ensure fair comparison. The original reference is kept in `baselines/original_pwm/` (local only, .gitignore).
- Diagnostics
  - ESNR and gradient norms are primary diagnostics for optimization quality, following PWM.
  - We will additionally track simple *smoothness proxies* for the dynamics, such as norms of Jacobians `∂F/∂z`, `∂F/∂a` on representative batches, to relate performance and ESNR back to PWM’s “smooth surrogate dynamics” story. These Jacobian metrics will be computed:
    - At low frequency (e.g., every N training steps) in eval mode, not inside every gradient step.
    - Using vector–Jacobian products with random vectors (vJPs) rather than full Jacobians, to keep compute manageable on single L40S GPUs.
    - For MLP WMs, `F(z,a)` is the one-step dynamics MLP. For Flow WMs, `F(z,a)` is defined as the macro-step mapping `wm.next(z,a, ...)` that includes ODE integration; Jacobians are taken w.r.t. this macro map.
  - NaN/Inf detection and gradient clipping are mandatory for world model and policy training.

---

**3) Phase 1 – Flow-Matching World Model in PWM (Single-Task)**

Status: Implemented in the codebase; configs and experiments in progress.

**3.1 Objectives**

- Replace the PWM MLP dynamics with a conditional flow-matching dynamics in the world model, while:
  - Keeping encoder and reward architectures identical.
  - Leaving actor/critic objectives unchanged.
  - Preserving PWM’s short-horizon, first-order training loop.
- Establish clean 5M and 48M baselines on `dflex_ant`, under parameter parity and matched training budgets.

**3.2 Mathematical specification (world model)**

- Latent encoding and reward
  - `z_t = E_φ(o_t)`, `ŝ_r,t = R_φ(z_t, a_t)`.
  - For both MLP and Flow variants, encoder and reward share the same architecture and (up to small adjustments for Flow’s time channel) comparable parameter footprint.
- Baseline MLP dynamics
  - `z_{t+1} = F_φ(z_t, a_t)`.
  - Dynamics loss for horizon `H`:
    - `L_dyn = (1/H) Σ_{t=0}^{H-1} γ^t || z_pred_{t+1} − z_tgt_{t+1} ||^2`, with `z_tgt_{t+1} = E_φ(o_{t+1})` and `z_pred_{t+1}` obtained by auto-regressive latent rollout through `F_φ`.
- Flow-matching dynamics
  - Velocity field: `v_θ(z, a, τ)` with explicit time conditioning `τ ∈ [0, 1]`.
  - For each time step `t`:
    - `z_s = z_pred_t` is the *current latent prediction* from the ongoing model rollout (as in the current implementation), not a teacher-forced `E(o_t)`.
    - `z_tgt = E_φ(o_{t+1})` is the “teacher” latent for the next observation (stop-gradient).
    - Sample `τ ~ U[0,1]` or set `τ = 0.5`.
    - `z_τ = (1−τ) z_s + τ z_tgt`.
    - Rectified-flow target: `v* = z_tgt − z_s`.
    - Per-step loss: `ℓ_t = || v_θ(z_τ, a_t, τ) − v* ||^2`.
  - Aggregated dynamics loss:
    - `L_dyn = (1/H) Σ_{t=0}^{H-1} γ^t ℓ_t`.
- Reward loss (shared)
  - In the original PWM paper (Eq. 10), reward prediction is formulated as a discrete regression problem in symlog space with a cross-entropy loss over bins. In this repo, the *current implementation* uses a simpler squared-error loss between reward logits and (normalized) scalar rewards:
    - `L_rew = mean_t [ || R_φ(z_t, a_t) − r_t ||^2 · γ^t ]`,
      where `r_t` is (optionally) normalized by `rew_rms`.
  - This is a deliberate divergence from PWM’s CE loss; we keep this design choice for now for simplicity and to focus on dynamics/ESNR, but we will revisit a PWM-aligned symlog/two-hot CE formulation if the simpler loss becomes a bottleneck.
  - Total world-model loss: `L_wm = L_dyn + L_rew`.

**3.3 Training/inference mismatch and smoothing story**

- Our Flow WM training is *auto-regressive*: both MLP and Flow roll out their own latent predictions `z_pred_t` and compare them against teacher latents `E(o_{t+1})`.
- For Flow WM, the rectified-flow loss is applied at `z_s = z_pred_t`, not at teacher `E(o_t)`, which means:
  - When rollout quality degrades, the Flow loss is learning a velocity field on “model states” rather than purely on true latent states.
  - There is an inherent train–inference distribution mismatch, as in many sequence models; we do not enforce a “perfect smoothing” objective over the true latent trajectory.
- Conceptually:
  - We rely on **ODE integration (Heun/Euler + K)** and regularization to provide controllable smoothing of the latent dynamics used by FoG, not on an explicit “smoothest possible” training objective.
  - This matches PWM’s spirit (smooth surrogate dynamics for FoG), but via a different mechanism: Flow WM trades capacity and integration cost for a tunable smoothness, rather than strictly minimizing one-step prediction error.
Implementation mapping:

- `pwm.models.world_model.WorldModel` implements the MLP variant.
- `pwm.models.flow_world_model.FlowWorldModel` implements the Flow variant:
  - `_encoder` and `_reward` mirror the baseline.
  - `_velocity` implements `v_θ`.
- `pwm.utils.integrators.{euler_step, heun_step, compute_flow_matching_loss}` implement ODE integration and flow-matching loss helpers.
- `pwm.algorithms.pwm.PWM.compute_wm_loss`:
  - Computes teacher latents `next_z[t] = E_φ(o_{t+1})` under `torch.no_grad()`.
  - Runs an auto-regressive latent rollout via `wm.next(...)` to obtain `z_pred_t`.
  - For MLP WM: uses MSE on `z_pred_{t+1}` vs `next_z[t]`.
  - For Flow WM: uses `compute_flow_matching_loss(self.wm.velocity, z_pred_t, next_z[t], a_t, task, ...)` without running the ODE inside the loss—ODE integration is only used to produce `z_pred_{t+1}` for the next step.

**3.4 ODE integrator and substeps**

- Default integrator
  - Heun’s method (explicit trapezoidal / RK2).
  - For fixed sub-steps `K`, `dt = 1/K`, per sub-step:
    - `k1 = v_θ(z, a, t_k)`,
    - `z' = z + dt · k1`,
    - `k2 = v_θ(z', a, t_k + dt)`,
    - `z ← z + (dt/2) · (k1 + k2)`.
- Ablation integrator
  - Euler with the same `K`.
- Implementation
  - `FlowWorldModel.next(z, a, task, integrator, substeps)` dispatches to `heun_step` or `euler_step`, performing `substeps` velocity evaluations per macro step.
  - Actor training and evaluation use `PWM.use_flow_dynamics`, `flow_integrator`, and `flow_substeps` to choose the codepath; from the algorithm’s perspective this is still one step in an `H`-step rollout, but with a different per-step compute budget.

**3.5 Parameter parity**

- Let `P_base` be the parameter count of the baseline world model (encoder + dynamics + reward) and `P_flow` the Flow variant count.
- Constraint:
  - `|P_flow − P_base| / P_base ≤ 0.02`.
- Implementation guidelines:
  - Keep depth identical to PWM.
  - Adjust hidden widths in `_velocity` to compensate for the +1 time channel.
  - Log `(P_base, P_flow, gap%)` at initialization; warn if `gap% > 2%` but do not crash.
- In code:
  - `FlowWorldModel.total_params` exposes parameter counts.
  - Configs such as `scripts/cfg/alg/pwm_5M_flow_*.yaml` and `pwm_48M_flow_*.yaml` are tuned to stay within this parity window.

**3.6 Diagnostics and acceptance for Phase 1**

- Environments
  - Single-task `dflex_ant`, with 5M as the MVP scale and 48M reserved for post-MVP scaling experiments once configs are stable.
- Metrics per run
  - Episode reward.
  - World-model dynamics and reward losses.
  - ESNR for actor updates.
  - Actor/critic/wm gradient norms.
  - Wall-clock time and number of velocity evaluations (via `substeps`).
  - Offline diagnostics for canonical configs:
    - Multi-step rollout error (e.g., 10–20 step latent rollouts vs. encoded true latents).
    - Approximate model–environment reward gap for fixed policies (e.g., evaluating the same policy in the WM and in the real env).
- Acceptance
  - Flow WM is stable at both 5M and 48M for at least one configuration.
  - Under parameter parity, Flow WM matches or is within ~10% of PWM baseline reward on `dflex_ant`.
  - No persistent NaNs in world-model or actor gradients at chosen hyperparameters.

---

**4) Phase 1.5 – Flow WM Regularization and Horizon Ablations (Single-Task)**

Motivation: Feedback from PWM authors indicates that (i) stronger world-model regularization is crucial for good first-order gradients, and (ii) the best horizon `H` for Flow may differ from the MLP case. PWM chose `H=16` because it maximized ESNR for their model; other works like TDMPC2 succeed with much shorter horizons.

**4.1 Goals**

- Tune Flow WM hyperparameters specifically for gradient quality, not just prediction error:
  - Regularization strength.
  - Horizon length `H`.
  - Integrator substeps `K` and τ sampling.
- Identify a canonical Flow WM setting on `dflex_ant` that we will carry forward into Phases 2–4.

**4.2 Hyperparameter axes**

To keep the search tractable, we distinguish a *minimal grid* (must-run) from optional ablations.

- Minimal grid (must-run)
  - Horizon `H ∈ {8, 16}` (PWM’s default is 16; we test whether a shorter horizon improves ESNR for Flow).
  - Substeps `K = 4`, `flow_integrator = heun`, `flow_tau_sampling = uniform`.
  - World-model regularization:
    - Baseline PWM-style regularization (SimNorm, weight decay, etc.).
    - Stronger L2 weight decay on world-model parameters (e.g., `3e-4`).
- Optional ablations (if time/compute allow)
  - Additional horizons: `H = 3` (TDMPC2-like).
  - Substeps `K ∈ {2, 8}`.
  - `flow_integrator = euler`.
  - `flow_tau_sampling = midpoint`.
  - Additional Flow-specific regularizers (e.g., penalties on `||∂F/∂z||`, `||∂F/∂a||`, or small stochastic noise per ODE substep).

**4.3 Evaluation and selection**

- For each candidate configuration at 5M and 48M:
  - Measure:
    - Final reward and learning curves.
    - ESNR (actor) and gradient norms.
    - Stability (collapses, NaNs).
    - Wall-clock and memory usage.
- Select:
  - One or two canonical Flow WM configs that offer a good reward/ESNR vs. compute tradeoff.
  - These configs become the default Flow WM choices in Phases 2–4.

**4.4 Reward-loss alignment (PWM vs current implementation)**

- If Flow WM + MSE reward loss significantly underperforms PWM baselines on dflex_ant (especially at 5M), we will:
  - Implement PWM’s symlog + two-hot + cross-entropy reward training for the world model on dflex_ant 5M.
  - Re-evaluate Flow WM vs. MLP WM under this reward loss to disentangle “Flow dynamics vs. reward pipeline” effects.
- This makes reward loss a *formal ablation axis* rather than a permanent divergence.

**4.5 Optional ablation: teacher-forced vs rollout-based Flow matching**

- Our default Flow WM uses rollout-based Flow matching (`z_s = z_pred_t`). As an optional ablation, we will compare against a teacher-forced variant:
  - Teacher-forced FM: `z_s = E(o_t)`, `z_tgt = E(o_{t+1})`.
  - Rollout-FM (default): `z_s = z_pred_t`, `z_tgt = E(o_{t+1})`.
- Comparing these two variants on ESNR, reward, Jacobian proxies, and stability will help clarify whether learning “correction flows” on model states (rollout-FM) is actually beneficial relative to standard CFM-style training on data latents.

---

**5) Phase 2 – Flow-Based ODE Policy in PWM (Single-Task)**

In Phase 2 we introduce a Flow-parameterized ODE policy while reusing the PWM training loop and Flow WMs from Phases 1/1.5. This phase focuses on Flow *architecture* for the policy; a full conditional flow-matching policy objective (FPO-style) is left for future work (Phase 2.5).

**5.1 Objectives**

- Implement a Flow-based ODE policy with the same external interface as `ActorStochasticMLP`:
  - `pi(z, deterministic=False) → action`.
  - Optional: `log_probs(z, action)` for logging and ESNR computation.
- Compare Flow-based policy against PWM’s Gaussian MLP policy under identical single-task settings.
- Study where Flow helps most: as a world model, as a policy, or both.

**5.2 Design constraints**

- The actor remains a stateless module taking latent `z_t` as input; we do not change core logic in `PWM.compute_actor_loss` beyond swapping `actor_config._target_`.
- The policy is trained purely by first-order gradients through the world-model rollout, as in PWM:
  - No gradient-free MBPOPPO-style updates.
- The Flow policy implementation must:
  - Preserve bounded actions (e.g., by applying `tanh` at the end).
  - Be pluggable via Hydra (e.g., `actor_config._target_ = pwm.models.actor.FlowActor`) without changes to the algorithm code.

**5.3 Minimal Flow ODE policy specification (first implementation)**

To make Phase 2 actionable, we fix a concrete initial Flow policy design:

- Base sampling process
  - Sample base noise `ε ~ N(0, I)` in action space.
  - In practice we may use `ε ~ N(0, σ^2 I)` with a scalar or diagonal `σ` (learnable or treated as a hyperparameter) to tune exploration strength.
  - Initialize `a_0 = W_z z + b_z + ε`, where `W_z, b_z` are learned linear projections of the latent `z` (optional).
- Velocity field
  - Define a velocity field in action space: `u_θ(a, z, τ, e_task)`, implemented as an MLP taking `[a, z, τ, e_task]` (with `e_task` being a learned task embedding in multi-task settings; for single-task it can be omitted) and outputting a vector in action space.
- ODE integration for action sampling
  - Choose a small number of policy substeps `K_policy` (e.g., 2 or 4).
  - Integrate from `τ=0` to `τ=1` using Heun’s method:
    - For `k = 0..K_policy−1`, with `dt = 1/K_policy`:
      - `k1 = u_θ(a_k, z, τ_k)`, `a' = a_k + dt · k1`.
      - `k2 = u_θ(a', z, τ_k + dt)`.
      - `a_{k+1} = a_k + (dt/2) · (k1 + k2)`.
  - Output action: `a = tanh(a_K)`.
- Log-probabilities
  - PWM’s actor loss does not require log-probabilities (it uses deterministic first-order gradients through the world model), so the minimal Flow policy *does not need an exact `log_prob`*. We may:
    - Omit `log_probs` entirely in the first implementation, or
    - Provide approximate log-probabilities for diagnostics only, clearly marked as such.

This minimal specification ensures that Flow policy is implementable without solving the full change-of-variables / exact-density problem on day one. More sophisticated Flow policy variants (e.g., with exact log-densities or multi-modal structure) can be added later as Phase 2 extensions.

**5.4 Experimental matrix (single-task dflex_ant)**

Using canonical Flow WM configs from Phase 1.5, run the following combinations at 5M and 48M:

- WM / Policy combinations
  - `MLP WM + MLP policy` (PWM baseline).
  - `MLP WM + Flow policy`.
  - `Flow WM + MLP policy`.
  - `Flow WM + Flow policy`.
- Metrics
  - Same as Phase 1.5, plus:
    - Policy ESNR and gradient norms.
    - Simple statistics of the action distribution (e.g., per-dimension variance) for both Gaussian and Flow policies.
- Questions
  - Does a Flow-based ODE policy improve sample efficiency or final reward over the Gaussian MLP policy, holding the world model fixed?
  - Does combining Flow WM + Flow policy offer compounding gains, or is most benefit attributable to one side (WM vs policy)?

**5.5 Phase 2.5 – FPO-style Flow-Matching Policy (Future Work)**

- Once Phase 2 is stable, we may extend the policy side with a true conditional flow-matching objective (FPO-style):
  - Define a conditional flow-matching objective for the policy that matches a target action distribution (e.g., an existing Gaussian policy or an offline expert) using a rectified-flow loss in action space.
  - Integrate this objective with the FoG + WM framework, standing on FPO / PolicyFlow-style methods rather than reinventing them.
- This extension is not required for the MVP but is an important avenue for future work if Phase 2 shows promising gains from Flow-based policy architectures alone.

---

**6) Phase 3 – Multi-Task PWM with Flow (MT30 / MT80)**

In Phase 3 we move from single-task dflex to PWM’s multi-task Meta-World settings, still entirely within this PWM-based repo.

**6.1 Objectives**

- Extend Phases 1–2 to multi-task training in PWM:
  - Use the existing MT30/MT80 environment wrappers and buffers.
  - Evaluate Flow WM and Flow policies in multi-task regimes.
- Identify whether Flow helps more in:
  - Multi-task generalization (training on many tasks concurrently).
  - Per-task performance with a shared world model and/or policy.

**6.2 Setup**

- Environments
  - MT30 and MT80 task sets, as configured in `scripts/cfg/config_mt30.yaml` and `config_mt80.yaml`.
- World models
  - MLP WM (PWM baseline).
  - Flow WM with canonical hyperparameters from Phase 1.5, adapted to multi-task.
    - Task embeddings follow PWM’s current code: each discrete task ID is mapped to a learned embedding `e`, which is concatenated to the inputs of `encoder`, `dynamics` or `velocity`, and `reward` MLPs.
- Policies
  - MLP policy.
  - Flow policy from Phase 2.

**6.3 Experimental matrix (multi-task)**

- WM / Policy combinations
  - `MLP WM + MLP policy`.
  - `MLP WM + Flow policy`.
  - `Flow WM + MLP policy`.
  - `Flow WM + Flow policy`.
- Metrics
  - Per-task reward and success rates.
  - Aggregate multi-task reward (mean/median across tasks).
  - ESNR and gradient norms for actor and world model.
  - Wall-clock and effective sample usage.

**6.4 Acceptance for Phase 3**

- At least one Flow configuration (WM, policy, or both) improves:
  - Mean multi-task reward or success vs. PWM baseline, and/or
  - ESNR and training stability under comparable compute.
- The multi-task training pipeline is stable (no structural NaN issues) for all four WM/policy combinations.

**6.5 Implementation Details (December 2025)**

Configuration files (all aligned with original PWM baseline hyperparameters):

| Config | World Model | Policy | File |
|--------|-------------|--------|------|
| Baseline | MLP | MLP | `scripts/cfg/alg/pwm_48M_mt_baseline.yaml` |
| Flow Policy | MLP | Flow ODE | `scripts/cfg/alg/pwm_48M_mt_flowpolicy.yaml` |
| Full Flow | Flow | Flow ODE | `scripts/cfg/alg/pwm_48M_mt_fullflow.yaml` |

Baseline alignment (all configs use these values for fair comparison):
- `wm_batch_size: 256`
- `wm_buffer_size: 1_000_000`
- `wm_iterations: 8`
- `horizon: 16` (via config_mt30.yaml)
- `max_epochs: 15_000`

Data and checkpoints:
- MT30 data: Download from https://www.tdmpc2.com/dataset
- MT30 checkpoint: `checkpoints/mt30_48M_4900000.pt` (from HuggingFace imgeorgiev/pwm)
- MT80 data: Download from https://www.tdmpc2.com/dataset
- MT80 checkpoint: `checkpoints/mt80_48M_2700000.pt` (from HuggingFace imgeorgiev/pwm)

Slurm submission scripts:
- `scripts/mt30/submit_baseline.sh` - Run MT30 baseline experiments
- `scripts/mt30/submit_flowpolicy.sh` - Run MT30 Flow Policy experiments
- `scripts/mt30/submit_fullflow.sh` - Run MT30 Full Flow experiments
- `scripts/mt30/download_data.sh` - Download checkpoints and data instructions

Experiment priority order:
1. **Policy-only comparison** (most fair): baseline vs flowpolicy with same WM checkpoint
2. **Full Flow comparison**: After Policy-only is stable, run fullflow experiments
3. Flow WM pretraining: Would require training Flow WM on MT30 data

**6.6 Checkpoint Resume Support**

For long-running jobs (>16 hours), training can be resumed from checkpoints:

```bash
# Resume training from a saved checkpoint
python scripts/train_multitask.py -cn config_mt30 \
  alg=pwm_48M_mt_baseline \
  task=reach-v2 \
  general.resume_from=/path/to/model_2000.pt \
  general.data_dir=/path/to/mt30 \
  general.epochs=10000
```

- Checkpoints saved every `eval_freq=200` epochs as `model_{epoch}.pt`
- Resume loads full state: actor, critic, WM, optimizers, training progress
- Training continues from the saved epoch



**7) Phase 4 – A/B/C Pipeline in PWM (Offline and Online, Multi-Task and Single-Task)**

Phase 4 defines the full training and transfer pipeline within the PWM repo, structured into three stages:

- A: Offline multi-task pretraining.
- B: Online multi-task fine-tuning.
- C: Online single-task specialization.

We study different paths through these stages and compare Flow-based variants against original PWM baselines.

**7.1 Stages**

- Stage A – Offline multi-task pretraining
  - Train large multi-task world models (MLP and Flow) on offline datasets (MT30/MT80) as in PWM.
  - Optionally pretrain a shared multi-task policy on top of the world model (purely offline).
- Stage B – Online multi-task fine-tuning
  - Starting from A, allow additional environment interaction across multiple tasks.
  - Fine-tune both WM and policy using the PWM first-order training loop (no planning).
- Stage C – Online single-task specialization
  - Specialize to a single task (e.g., a chosen Meta-World task or dflex task) starting from an A or B checkpoint.
  - Fine-tune WM and policy on that single task using the same PWM-style first-order loop.

**7.2 Paths to compare (within PWM repo)**

For each path, compare:

- Baseline PWM
  - `MLP WM + MLP policy`, using PWM’s recommended hyperparameters.
- Flow variants
  - `Flow WM + MLP policy`.
  - `MLP WM + Flow policy`.
  - `Flow WM + Flow policy`.

Paths:

- `A → C`
  - Offline multi-task world-model pretraining, then direct single-task online specialization (no multi-task online phase).
- `A → B → C`
  - Offline pretraining, then online multi-task fine-tuning, then single-task specialization.
- `B → C`
  - Multi-task online training from scratch (or minimal pretraining), followed by single-task specialization.

**7.3 Metrics and comparisons**

- Final single-task reward and success after Stage C.
- Speed of adaptation: steps to reach a fixed reward threshold on the target single task.
- Retained multi-task performance after Stage B (catastrophic forgetting vs. specialization).
- ESNR, gradient norms, and wall-clock across stages and paths.

**7.4 Acceptance for Phase 4**

- At least one Flow-based A/B/C path demonstrates:
  - Better or faster single-task specialization than the PWM baseline path under comparable compute, and/or
  - More stable gradients (higher ESNR, fewer collapses) in multi-task regimes.

**7.5 Relation to 2×2 grid and RQ3**

- In the 2×2 offline/online × single/multi grid:
  - Stage A sits in the *offline multi-task* quadrant.
  - Stage B sits in the *online multi-task* quadrant.
  - Stage C sits in the *online single-task* quadrant.
- Paths `A→C`, `B→C`, and `A→B→C` thus compare how Flow WM/policies impact:
  - Direct specialization from offline pretraining (`A→C`).
  - Online multi-task adaptation plus specialization (`B→C`).
  - Full offline + online multi-task + specialization pipeline (`A→B→C`).
- These comparisons directly address **RQ3**: *in which stage(s) and path(s) does Flow provide the largest marginal gain over PWM baselines?*

---

**8) Beyond PWM – RWM / Newt Integration (Future Work)**

After Phases 1–4 are complete and we have clear results in PWM, we will port the most promising Flow configurations to RWM/Newt:

- Use RWM environments (e.g., walking) as the main testbed, with potential sim-to-real on selected tasks.
- Reuse the A/B/C staging:
  - A: Offline multi-task pretraining on RWM/Newt data.
  - B: Online multi-task fine-tuning in RWM environments.
  - C: Online single-task specialization.
- Compare:
  - MBPOPPO-style baselines vs. Flow-based MBPO/FPO, focusing on contact-rich locomotion and cross-embodiment transfer.

Exact RWM/Newt integration details will be specified in a separate document once PWM-based phases are stable.

---

**9) Summary of Metrics and Reporting**

Across all phases we will consistently log:

- Performance
  - Episode reward and success rate.
  - For multi-task experiments: per-task and aggregated statistics.
- World model
  - Dynamics and reward losses.
  - ESNR for actor updates and (where applicable) world-model gradients.
  - Parameter counts, FLOP proxies, and wall-clock times.
- Optimization
  - Gradient norms for actor, critic, and world model.
  - Frequency and handling of NaNs / Infs.
- Configuration
  - All key hyperparameters: horizon `H`, substeps `K`, integrator type, regularization strengths, and Flow-specific settings.

- Compute proxy
  - We approximate per-update compute using a simple proxy:
    - `compute_index = (#WM_forward_calls + #WM_backward_calls) + λ · (#actor_forward_calls + #actor_backward_calls)`,
      where λ is a constant reflecting that actor and world model costs are of similar order (we treat λ ≈ 1 for simplicity).
  - For Flow WMs and Flow policies, `#WM_forward_calls` and `#actor_forward_calls` scale with `K` and `K_policy` respectively, so this proxy captures the main compute differences even if it ignores precise FLOP counts.

**9.1 Practical compute guidelines (single L40S GPUs)**

- For dflex_ant @ 5M (single-task), we target individual training runs in the range of ~12–24 GPU hours on a single L40S.
- To avoid runaway compute:
  - Phase 1 and 1.5:
    - Prefer `H ∈ {8,16}`, `K=4`, Heun, Gaussian policy (no Flow policy yet).
  - Phase 2:
    - Start with `H=8`, `K=4`, `K_policy=2` for Flow ODE policy.
  - Avoid combinations like `H=16`, `K=8`, `K_policy=4` unless justified by prior results and explicitly marked as “high-cost ablations”.
- Multi-task Flow WM + Flow policy runs (Phase 3/4) are considered high-risk, high-cost experiments and should only be launched once single-task and small-scale multi-task results are stable.
- For MVP experiments, we assume **3 seeds per configuration** unless otherwise noted; total compute should be planned as `num_configs × 3 × runtime_per_run`.

---

**10) Experiment Operations, Naming, and Logging**

- **Clusters and resources**
  - All GPU training and eval runs must be submitted to Phoenix via Slurm with `--gres=gpu:L40S:1`, `--mem=384GB`, `-t 32:00:00`, and account `-A gts-agarg35-ideas_l40s`.
  - Activate `conda activate pwm`; set `PYTHONPATH=src` if not installed in editable mode.
- **Script organization (do not rename existing baselines)**
  - `scripts/` root currently holds single-task Slurm scripts and entrypoints:
    - Baselines: `submit_5M_baseline_l40s_final.sh`, `submit_48M_baseline_l40s.sh`.
    - Flow WM single-task: `submit_5M_flow_v{1,2,3}_l40s_final.sh`, `submit_48M_flow_v{1,2,3}_l40s.sh`.
    - Helpers: `submit_all_verified.sh`, `mt30.bash`, `mt80.bash`, `train_dflex.py`, `train_multitask.py`, `cfg/`.
  - New scripts should follow `scripts/phase{1|1p5|2|3}/submit_<env>_<config>_l40s.sh` (create subfolders per phase to keep clarity) and mirror the Slurm header used above. Keep original PWM baselines under `scripts/cfg/alg/original_pwm` untouched for reference.
- **Run naming conventions**
  - **W&B**: project `flow-mbpo-pwm`; groups `phase{1|1p5|2|3}_<env>_<config>`; run names `{config}_seed{seed}_H{H}_K{K}_Kpol{Kpol}` (omit unused fields). Example: `phase1_dflex_ant_flow_v2_seed42_H16_K4`.
  - **Slurm job name**: `pwm_{phase}_{env}_{config}_s{seed}` to align logs with W&B.
  - **Log files**: write to `logs/slurm/{phase}/{env}/train_{config}_seed{seed}_%j.{out,err}` to keep jobid traceable. For quick smoke tests on CPU, use `logs/smoke/` with `_cpu` suffix.
- **Artifacts and checkpoints**
  - Store checkpoints under `outputs/{phase}/{env}/{config}/seed{seed}/` with `latest.pt` and periodic `epochXXXX.pt`. Copy W&B run ID into a `run_id.txt` in the same folder.
  - Evaluation results (CSV/JSON) go to `results/{phase}/{env}/{config}/seed{seed}/` with a short README noting the exact checkpoint used.
- **Minimal sanity tests before GPU jobs**
  - CPU-only import check: `PYTHONPATH=src python -c "from pwm.models.world_model import WorldModel; from pwm.models.flow_world_model import FlowWorldModel; print('OK')"`.
  - Config dry-run (no env rollout): `PYTHONPATH=src python scripts/train_dflex.py alg=pwm_5M_baseline_final general.device=cpu general.num_envs=1 general.num_steps=4 eval_every=0 train_every=0` to ensure Hydra config resolves on the login node; keep under 1 minute.
- **Experiment registry (see new docs files)**
  - `docs/progress_log.md`: chronological development log (what changed, why, open issues, next steps).
  - `docs/experiment_log.md`: per-run registry with Slurm job ID, config, seed, phase, metrics (final reward, ESNR if available), W&B link, status (queued/running/succeeded/failed), and log paths. Update after each run submission and completion.
- **Required hygiene**
  - Comments and documentation in English only; convert any Chinese comments encountered.
  - Every change recorded via git with English commit messages; never revert user changes unintentionally.
  - Keep PWM baselines from `imgeorgiev/PWM` intact; stored in `baselines/original_pwm/`.

All new implementations and experiments in this repository should be traceable back to a specific phase and subsection of this document.

---

**12) Phase 3: MT30 Multitask Comparison**

- **Goal**: Compare MLP Policy (Baseline) vs Flow ODE Policy on MT30 multitask tasks using the pre-trained 48M PWM world model.
- **Tasks**: `reacher-easy`, `walker-stand`, `cheetah-run` (selected for initial comparison).
- **Setup**:
  - World Model: Pre-trained 48M MLP WM (PWM original checkpoint).
  - Policy variants:
    1. **Baseline**: MLP Policy (`ActorStochasticMLP`).
    2. **Flow Policy**: Flow ODE Policy (`ActorFlowODE`).
  - Compute: H100 GPUs (PACE ICE Cluster), 450GB RAM, 16h walltime.
- **Evaluation**: Compare final reward and planning-based evaluation across 3 seeds per task.
- **Status (Jan 2026)**: Phase 3 experiments running (Attempt 8).
  - **Collision Incident**: Attempt 7 (H100/H200) completed quickly (~27m) but log files for simultaneous array tasks collided.
  - **Partial Results**: Recovered `cheetah-run` (R~115) and `walker-stand` (R~901).
  - **Current Action**: Resubmitted full batch (Attempt 8) with fixed logging paths to ensure clean data for all seeds.
  - **Hardware Incident**: Flow Policy jobs (seeds for walker/cheetah) failed on bad H200 node (ERR!/700W).
  - **Status**: ✅ **COMPLETED** (Jan 04).
  - **Results**:
    - `reacher-easy`: Tie (~982).
    - `walker-stand`: Baseline wins (+14%).
    - `cheetah-run`: Tie (Both failed, R~100). Need hyperparameter tuning or more data.
  - **Storage**: Weights deleted to respect quota. Metrics preserved.
  - **Next Step**: Proceed to Phase 4 (Full Flow Model).

---

**13) Phase 4: Full Flow & Dynamic Training (Ongoing)**

- **Goal**: End-to-end training of "Flow World Model + Flow Policy" (from scratch).
- **Status**: 🚧 **IN PROGRESS** (Jan 04).
- **Infrastructure Improvements**:
  - **Resume Support**: Robust state restoration (optimizers, LRS, iter count) to handle 16h walltime.
  - **Best Model Saving**: Periodic evaluation-based `model_best.pt` saving.
  - **Detailed Logging**: `MT30-Detailed` WandB project with comprehensive diagnostics.
- **Experiments**:
  - `4011986`: Main MT30 Full Flow (3 tasks, 3 seeds).
  - `4011987`: Cheetah Debug (Horizon=30).
- **Hypothesis**: The combination of Flow WM and Flow Policy will stabilize gradients (better ESNR) and potentially solve `cheetah-run` with extended horizons.

---

**11) Codebase Structure**

As of 2025-12-26, the codebase has been flattened to remove the PWM git submodule for easier collaboration:

```
Flow-MBPO-PWM/
├── src/pwm/                     # Main source code
│   ├── algorithms/              # PWM, SHAC algorithms
│   ├── models/                  # WorldModel, FlowWorldModel, Actor, Critic
│   └── utils/                   # Buffer, integrators, monitoring, etc.
├── scripts/                     # Training and evaluation scripts
│   ├── cfg/                     # Hydra config files
│   │   ├── alg/                 # Algorithm configs (pwm_5M_*, pwm_48M_*, etc.)
│   │   └── env/                 # Environment configs (dflex_ant, etc.)
│   ├── train_dflex.py           # Main single-task training script
│   └── train_multitask.py       # Multi-task training script
├── baselines/
│   └── original_pwm/            # Clone of imgeorgiev/PWM for comparison
├── docs/                        # Documentation
│   ├── master_plan.md           # This document
│   ├── progress_log.md          # Development log
│   ├── experiment_log.md        # Experiment registry
│   └── baseline_comparison.md   # Comparison with original PWM
├── hf_pwm_repo/                 # HuggingFace pre-trained models
├── setup.py                     # Package installation
└── environment.yaml             # Conda environment
```

Key changes from previous structure:
- Removed PWM git submodule (was `PWM/` folder)
- Source code moved from `PWM/src/pwm/` to `src/pwm/`
- Scripts moved from `PWM/scripts/` to `scripts/`
- Original baseline preserved in `baselines/original_pwm/` for comparison

