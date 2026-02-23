# Original PWM Pipeline (Code + Paper Aligned Pseudocode)

This document is based on:

1. `baselines/PWM` (full cloned repository, commit `9816252`).
2. `baselines/original_pwm` (local snapshot with env configs, figures, results).
3. Paper: `docs/source/PWM Policy Learning with Multi-Task World Models 2407.02466v3.pdf`.

## Scope and Naming

- `E_phi`: encoder.
- `F_phi`: latent dynamics.
- `R_phi`: reward model.
- `pi_theta`: stochastic actor.
- `V_psi`: critic ensemble (3 critics).
- `B`: replay buffer of trajectory slices with length `H+1`.
- `H`: short rollout horizon (default `16`).

## Entrypoints in Original PWM Codebase

- Single-task DFlex training: `baselines/PWM/scripts/train_dflex.py`.
- Multi-task policy extraction: `baselines/PWM/scripts/train_multitask.py`.
- Core algorithm: `baselines/PWM/src/pwm/algorithms/pwm.py`.
- World model: `baselines/PWM/src/pwm/models/world_model.py`.

## Unified Pipeline Pseudocode (What Actually Runs)

```text
PROCEDURE PWM_FULL_PIPELINE(mode, config):
    # mode in {single_task_dflex, multitask_offline_extraction}

    INITIALIZE PWM agent with:
        actor pi_theta, critic ensemble V_psi, world model (E_phi, F_phi, R_phi)
        optimizers: Adam(actor), Adam(critic), Adam(world_model)
        replay buffer B sampling (H+1)-step slices

    IF mode == single_task_dflex:
        # scripts/train_dflex.py
        MAKE vectorized DFlex env
        IF checkpoint provided:
            LOAD actor+critic+world_model (or at least world model)
            wm_bootstrapped <- True
        IF pretrain data provided:
            CALL PRETRAIN_WORLD_MODEL(paths, pretrain_steps)
        CALL ONLINE_TRAIN_WITH_ENV()
        FINAL eval: run policy in env, report episode metrics

    ELSE IF mode == multitask_offline_extraction:
        # scripts/train_multitask.py
        BUILD MT30/MT80 task list (from TD-MPC2 task set)
        BUILD env wrapper for evaluation
        SET world model task embedding dim:
            64 for MT30, 96 for MT80
        LOAD world model checkpoint via load_wm():
            keep world model keys, drop policy/Q keys from TD-MPC2 checkpoint
        LOAD offline .pt dataset chunks, filter episodes by task_id, fill B
        FOR iter = 0..epochs-1 (default 10k):
            SAMPLE (obs, act, rew) from B
            task_ids <- repeated task index
            CALL UPDATE_OFFLINE(obs, act, rew, task_ids, finetune_wm flag)
            periodic eval in real env
        SAVE model and CSV logs


PROCEDURE PRETRAIN_WORLD_MODEL(paths, N):
    FOR each dataset file in paths:
        LOAD TensorDict episodes
        UPDATE optional obs/reward RMS stats
        ADD episodes to replay buffer B

    FOR i = 1..N:
        SAMPLE (obs, act, rew) from B  # shapes: [H+1, batch, ...]
        NORMALIZE obs/rew if RMS is enabled
        (L_wm, L_dyn, L_rew) <- COMPUTE_WM_LOSS(obs, act, rew, task=None)
        BACKPROP world model optimizer step
    wm_bootstrapped <- True
    SAVE pretrained model


PROCEDURE ONLINE_TRAIN_WITH_ENV():
    RESET env and initialize per-env episode trackers
    FOR epoch = 1..max_epochs:
        IF buffer has no completed episodes yet:
            # bootstrap by collecting data only
            RUN compute_actor_loss() under no_grad to populate B
            CONTINUE

        UPDATE learning rates (linear schedule)

        # 1) Actor update
        actor_loss <- COMPUTE_ACTOR_LOSS(obs=None, task=None)  # includes imagined rollout
        GRAD STEP on actor with grad clipping

        # 2) Critic update
        BUILD TD targets from rollout buffers via COMPUTE_TARGET_VALUES() (TD-lambda or 1-step)
        SPLIT flattened rollout data into critic mini-batches
        REPEAT critic_iterations times:
            optimize MSE(critic(obs_latent), target_values)

        # 3) World model update
        IF wm_bootstrapped:
            wm_train_iters <- wm_iterations (default 8)
        ELSE:
            wm_train_iters <- env.episode_length (first heavy bootstrap pass)
            wm_bootstrapped <- True
        REPEAT wm_train_iters:
            SAMPLE (obs, act, rew) from B
            NORMALIZE obs/rew if enabled
            optimize COMPUTE_WM_LOSS(obs, act, rew)

        LOG metrics; periodically SAVE checkpoints


PROCEDURE UPDATE_OFFLINE(obs, act, rew, task_ids, finetune_wm):
    # Used by multitask script (no env interaction inside update loop)

    IF finetune_wm:
        optimize COMPUTE_WM_LOSS(obs, act, rew, task_ids)

    actor_loss <- COMPUTE_ACTOR_LOSS(obs0=obs[0], task=task_ids)
    GRAD STEP actor

    BUILD critic dataset from imagined rollout buffers
    RUN critic_iterations of critic updates
    RETURN metrics


PROCEDURE COMPUTE_ACTOR_LOSS(obs0, task):
    IF obs0 is None:
        obs0 <- env.reset(grads=True)
    z <- E_phi(obs0, task)

    FOR t = 0..H-1:
        a_t ~ pi_theta(z_t); a_t <- tanh(a_t)
        (z_{t+1}, r_logits_t) <- world_model.step(z_t, a_t, task)
        r_hat_t <- almost_two_hot_inv(r_logits_t)  # differentiable reward inversion

        IF env exists (single-task online mode):
            step real env with same action
            append real transitions to replay buffer B
            if done: reset latent by encoding reset observation
            update done masks / term masks

        store latent/reward/next-value buffers for critic targets

    actor objective in code is negative imagined return with terminal bootstrap:
        L_actor = mean( -sum_t gamma^t r_hat_t - bootstrap_term )
    optional return RMS normalization
    RETURN L_actor


PROCEDURE COMPUTE_WM_LOSS(obs, act, rew, task):
    # obs shape: [H+1, batch, obs_dim]
    next_z_target <- stopgrad( E_phi(obs[1:], task) )
    z <- E_phi(obs[0], task)

    L_dyn <- 0
    FOR t = 0..H-1:
        z <- F_phi(z, act[t], task)
        L_dyn += gamma^t * MSE(z, next_z_target[t])
        cache z_t

    rew_pred <- R_phi(z_t, act[t], task) over rollout
    L_rew <- mean( gamma^t * (rew_pred - rew)^2 )
    L_wm <- (L_dyn + L_rew) / H
    RETURN L_wm, L_dyn/H, L_rew/H
```

## Natural-Language Pipeline Summary (Major Steps + Substeps)

1. Data Preparation and Experiment Setup
(a) Choose training mode: `single_task_dflex` (online interaction) or `multitask_offline_extraction` (offline policy extraction per task).
(b) Load YAML config and initialize all hyperparameters: rollout horizon `H`, learning rates, `gamma`, `lam`, gradient clipping thresholds, and update iteration counts.
(c) Build model components: stochastic actor, 3-critic ensemble, and world model (`encoder`, `dynamics`, `reward`).
(d) Initialize replay buffer to sample fixed-length trajectory slices of `H+1` steps.
(e) Build task/environment metadata.
For single-task: use the selected DFlex environment config.
For multitask: build MT30/MT80 wrapper, align observation/action dimensions, and set `task_dim` (`64` for MT30, `96` for MT80).

2. Load Data and Optional Checkpoints
(a) Single-task path: data comes from environment interaction during training, with optional offline trajectory files for pretraining.
(b) Multitask path: load TD-MPC2-format offline `.pt` files from `general.data_dir`, filter episodes by the selected `task_id`, and add them to replay buffer.
(c) If a world-model checkpoint is provided, load it before optimization.
In multitask runs this is commonly a TD-MPC2 checkpoint mapped through `load_wm()` (policy/Q keys are ignored).

3. Optional World-Model Pretraining (Before Policy Learning)
(a) Pretraining is triggered when `general.pretrain` is provided.
(b) Fill replay buffer from offline episodes first.
(c) Repeat pretraining updates for the requested step budget:
sample `(obs, act, rew)` slices of shape `[H+1, batch, ...]`, compute world-model losses, backpropagate, and update world-model weights.
(d) Mark world model as bootstrapped and save pretrained checkpoint.

4. Main Training Loop (Policy + Critic + World Model)
(a) Start iterative training (`max_epochs` or configured multitask epochs).
(b) Actor update:
run `H`-step imagined rollout in latent space, compute actor objective as negative discounted return with terminal bootstrap value, then update actor by first-order gradients through world-model rollouts.
(c) Critic update:
build TD targets (default TD-lambda), split rollout buffer into minibatches, and run multiple critic optimization iterations.
(d) World-model update:
sample replay slices and optimize latent dynamics + reward prediction losses in the same training cycle (or optionally via `finetune_wm` in multitask offline updates).
(e) Apply learning-rate schedule and gradient clipping; log core diagnostics (losses, gradient norms, reward statistics).

5. Data Flow During Training (Critical for Correctness)
(a) In single-task online mode, actor actions are executed in the real DFlex simulator while imagined rollouts are used for FoG optimization.
(b) Completed episodes are appended to replay buffer, and done/termination/truncation states are handled to reset rollout state correctly.
(c) In multitask offline mode, updates sample only from offline replay data for that task (unless explicit online finetuning is enabled externally).
(d) This design keeps policy updates short-horizon and model-based while preserving a consistent replay interface for both modes.

6. Evaluation, Logging, and Final Outputs
(a) Run periodic evaluation in real environments using configured `eval_freq` and `eval_runs`.
(b) Record per-task metrics such as `episode_reward` and `episode_success`.
(c) For multitask analysis, also run planning-enabled evaluation when required and record `episode_reward_planning` / `episode_success_planning`.
(d) Save checkpoints (`best`, periodic, `final`) and export per-task CSV logs.
(e) Aggregate MT30/MT80 results into final tables/figures for cross-task comparison.

7. Final Validation Checklist for Paper Comparability
(a) Data regime matches intended setup (single-task online vs multitask offline extraction).
(b) Horizon and discount settings are correct (especially `H=16`, `gamma=0.99` where applicable).
(c) Update order is preserved: actor rollout/objective, critic TD update, then world-model update.
(d) Evaluation protocol and metrics format match reported results.
(e) Reward-model objective differences vs paper are explicitly documented before claiming exact reproduction.

## Important Paper-vs-Code Notes

1. Paper Eq. (10) describes reward training with symlog/two-hot cross-entropy.
2. Current `PWM.compute_wm_loss()` in `baselines/PWM/src/pwm/algorithms/pwm.py` uses squared error reward loss.
3. In single-task config `scripts/cfg/alg/pwm.yaml`, `num_bins` is unset, so reward is scalar regression.
4. In multi-task configs (`pwm_48M.yaml` etc.), `num_bins=101` is enabled and actor uses `almost_two_hot_inv()` during imagined rollouts.
5. The algorithm remains FoG actor + TD-lambda critic over short imagined world-model rollouts, matching the main PWM design.

## Cross-Check with Local Snapshot (`baselines/original_pwm`)

- `baselines/original_pwm/results/data/mt30_results.csv` has 30 task rows at `iteration=9999`.
- `baselines/original_pwm/results/data/mt80_results.csv` has 80 task rows at `iteration=9999`.
- `baselines/original_pwm/scripts/cfg/env/*.yaml` matches the DFlex environment family used by PWM single-task experiments.
