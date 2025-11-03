Title: Flow-Matching World Model for PWM — Design and Evaluation Plan

Status: Draft (source of truth for upcoming changes)
Owners: PWM repo extension (flow-based dynamics)
Scope: Introduce a flow-matching variant of the world model dynamics (F) that plugs into PWM with minimal disruption; define a fair A/B protocol against the baseline MLP dynamics.

---

1) Goals and Hypotheses
- Goal: Replace the MLP “one-step next-latent” dynamics in PWM with a conditional, smooth flow-matching dynamics that improves optimization landscape for gradient-based policy learning (higher reward, faster convergence, better stability) while keeping the rest of PWM unchanged.
- Hypothesis H1 (Smoothness): A flow-based (rectified flow) dynamics produces smoother gradients during H-step unrolls, improving actor optimization (higher ESNR, fewer NaNs, smaller grad variance).
- Hypothesis H2 (Performance): Under equal compute and data, the flow-dynamics variant matches or outperforms the baseline in single-task and multi-task settings on reward and wall-clock.

Non-goals
- Do not change actor/critic architectures or policy training objective beyond what is necessary to integrate the new dynamics.
- Do not change reward modeling; keep identical reward head and target handling.
- Do not change datasets, evaluation environments, or scripts other than strictly required configuration toggles.

---

2) High-Level Approach
- Keep the encoder E_phi and reward head R_phi exactly as baseline.
- Replace the dynamics F_phi(z_t, a_t) with a conditional velocity field v_theta(z, a, t) integrated over t in [0,1] to produce z_{t+1}. This is a rectified flow/flow-matching formulation.
- Training dynamics with flow-matching (not MSE on next state): we minimize E_tau[ || v_theta(z_tau, a, tau) − (z_target − z_source) ||^2 ], where z_tau = (1−tau) z_source + tau z_target and z_target = E_phi(obs_{t+1}). This encourages a smooth field that transports z_source to z_target.
- Keep PWM’s truncated H-step rollout, value bootstrapping, gamma=0.99, and horizon H (e.g., 16) intact.

---

3) Interfaces and Contracts
- New class: FlowWorldModel implementing the same public API as the baseline WorldModel to preserve PWM integration:
  - encode(obs, task) -> z: latent encoding (identical as baseline)
  - next(z, a, task) -> z_next: integrate velocity field v_theta over t ∈ [0,1] using fixed-step Euler; same tensor shapes as baseline.
  - reward(z, a, task) -> r_hat: identical reward head as baseline (including two-hot utilities if num_bins>1).
  - step(z, a, task) -> (z_next, r_hat): wrapper calling next(...) and reward(...).
- Additional training helper in the model for clarity and locality of logic:
  - latent_rollout_and_dynamics_loss(z0, act_seq, next_z_seq, horizon, gamma, task=None) -> (zs, dynamics_loss)
    - Performs H-step rollout with the flow dynamics; returns predicted zs and the flow-matching loss summed across steps with gamma^t weighting and normalized by H.
- Backward-compatibility: The baseline WorldModel is unchanged. The algorithm will detect the presence of the above helper and use flow-matching; otherwise it falls back to MSE-to-next_z.

---

4) Mathematical Details
- Dynamics integration (per step):
  - Given z_t and action a_t, integrate v_theta over t ∈ [0,1] with K uniform Euler sub-steps:
    - dt = 1/K; for k = 0..K−1: z ← z + dt · v_theta(z, a_t, t_k), where t_k = (k+0.5)·dt
    - Final z is z_{t+1}.
- Flow-matching loss (rectified flow):
  - For each unrolled step t:
    - Source z_s = z_t (predicted), target z_tgt = E_phi(obs_{t+1}).
    - Sample tau ~ U[0,1] (or midpoint 0.5 for deterministic variant).
    - z_tau = (1−tau)·z_s + tau·z_tgt; v_star = z_tgt − z_s (constant drift).
    - L_dyn^t = || v_theta(z_tau, a_t, tau) − v_star ||^2.
    - Discounted aggregation: L_dyn = (1/H) · Σ_{t=0}^{H−1} γ^t · L_dyn^t.
- Reward loss: identical to baseline (two-hot or scalar) using zs[:-1] and act_seq.
- Total world model loss: L_wm = L_dyn + L_rew.

---

5) Config Additions (Hydra YAML)
- Provide a separate algorithm config to switch the WM target class without touching baseline configs, e.g.:
  - PWM/scripts/cfg/alg/pwm_48M_flow.yaml
    - world_model_config._target_: pwm.models.flow_world_model.FlowWorldModel
    - encoder_units/units/task_dim/num_bins/vmin/vmax/tasks/action_dims: copied from baseline for parity.
    - New hyperparameters:
      - integrate_steps: int (default 4) — Euler sub-steps K per environment step.
      - fm_tau_sampling: {uniform, midpoint} (default uniform).
- No changes to actor/critic configs.

---

6) Implementation Plan (File-by-File)
- PWM/src/pwm/models/flow_world_model.py (new)
  - Implement FlowWorldModel with modules:
    - _encoder: identical to baseline (mlp with LayerNorm+Mish and same final head).
    - _velocity: mlp on [z, a, (optional task emb), t] → R^{latent_dim} with same hidden sizes as baseline dynamics (units) to keep capacity parity. Input grows by +1 dimension due to t.
    - _reward: identical to baseline reward head and two-hot utilities if enabled.
  - Public methods: encode, next (Euler integration), reward, step, latent_rollout_and_dynamics_loss.
- PWM/src/pwm/algorithms/pwm.py (minimal, additive only)
  - In compute_wm_loss(...):
    - Compute targets next_z = wm.encode(obs[1:]).
    - If the WM exposes latent_rollout_and_dynamics_loss, call it to obtain (zs, dynamics_loss). Else use current baseline MSE rollout to next_z.
    - Compute reward_loss same as baseline; total = dynamics_loss + reward_loss.
  - WM optimizer: include parameters of _velocity when present; keep others identical.
- PWM/scripts/cfg/alg/pwm_48M_flow.yaml (new)
  - Mirror pwm_48M.yaml with only world_model_config._target_ and the new dynamics hyperparameters changed.

Note: The baseline files remain untouched other than the guarded path in compute_wm_loss and optimizer param-group extension. The baseline codepath is unchanged when using the original world model class.

---

7) Fairness and Parity Controls
- Data and splits: identical offline datasets per task.
- Encoder and reward head: identical architectures and initialization.
- Dynamics capacity: same hidden layers and widths; only +1 input for time t. This changes parameter count minimally. Record parameter counts for both models and report.
- Optimizer, learning rates, weight decay, gradient clipping, batch sizes, horizon H, gamma, number of iterations/epochs: identical.
- Seeds: fix seeds across runs.
- Compute: run both on the same GPU type; report wall-clock.
- Evaluation protocol: same eval frequency, same number of eval episodes, same success metrics.

---

8) Experiments and Metrics
- Single-task locomotion (dflex tasks): reward curves over iterations and wall-clock; time to thresholds; stability (NaN counts).
- Multi-task MT30/MT80: task-wise reward, success rate, average reward.
- Optimization diagnostics:
  - Actor grad norm, critic grad norm.
  - ESNR proxy: ratio of gradient signal to batch variance (implement simple logging via running stats if needed).
  - WM losses: dynamics (flow vs MSE) and reward loss.
  - Rollout stability: exploding/vanishing checks for latent norms.

---

9) Hyperparameters to Sweep (small)
- integrate_steps: {2, 4, 8} (trade-off between smoothness and compute).
- fm_tau_sampling: {uniform, midpoint}.
- horizon H: keep 16 for the primary comparison; optionally 10/20 as ablation.
- Finetune WM during policy extraction: {False, True (small lr)}.

---

10) Risks and Mitigations
- Risk: Extra compute from integration increases wall-clock. Mitigation: small K (e.g., 4) and measure; ensure apples-to-apples by reporting wall-clock.
- Risk: Flow loss underfits with small batches. Mitigation: maintain baseline batch size; consider mild increase only if necessary and report.
- Risk: Reward head mismatch due to two-hot handling. Mitigation: reuse exact baseline utilities and configurations.
- Risk: Numerical instability in integration. Mitigation: use midpoint Euler and clip latents if norms explode; log latent norms.

---

11) Rollout and Training Pseudocode (for compute_wm_loss)
- Baseline branch (unchanged):
  - z0 = wm.encode(obs[0]); next_z = wm.encode(obs[1:]);
  - for t in 0..H−1:
      z = wm.next(z, act[t]); dyn_loss += MSE(z, next_z[t]) * gamma^t
  - reward_loss = MSE(wm.reward(zs[:-1], act), rew) with discount window
  - total = (dyn_loss + reward_loss) / H
- Flow branch (when wm exposes latent_rollout_and_dynamics_loss):
  - zs, dyn_loss = wm.latent_rollout_and_dynamics_loss(z0, act, next_z, H, gamma)
  - reward_loss same as baseline; total = (dyn_loss + reward_loss) / H

---

12) Deliverables and Milestones
- D1: Code implementing FlowWorldModel and minimal PWM hooks as described.
- D2: Config file to enable flow model without touching baseline configs.
- D3: Parity report with parameter counts and training wall-clock for both variants on at least one task.
- D4: Full MT30/MT80 comparison plots and tables.

---

13) Exact Files to be Added/Modified
- Added:
  - PWM/src/pwm/models/flow_world_model.py
  - PWM/scripts/cfg/alg/pwm_48M_flow.yaml
- Modified (guarded changes only):
  - PWM/src/pwm/algorithms/pwm.py
    - compute_wm_loss: call model’s helper if present; else unchanged path.
    - wm optimizer: include _velocity params if present.

---

14) How to Run (once implemented)
- Baseline (unchanged):
  - python scripts/train_multitask.py -cn config_mt30 alg=pwm_48M task=<task> general.data_dir=<dir> general.checkpoint=<wm_ckpt>
- Flow variant:
  - python scripts/train_multitask.py -cn config_mt30 alg=pwm_48M_flow task=<task> general.data_dir=<dir> general.checkpoint=<wm_ckpt>
- For single-task dflex, mirror the same substitution in its config.

---

15) Acceptance Criteria
- The flow variant code compiles and runs on the same datasets and scripts as baseline, with no changes required to evaluation code.
- Under equal compute, flow variant achieves ≥ baseline reward on at least half of tested tasks; shows improved convergence speed on a subset; and exhibits lower gradient instability or better ESNR proxy.

