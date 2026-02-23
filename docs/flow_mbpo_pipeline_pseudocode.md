# Flow-MBPO Training Pipeline (Pseudo Code, Fair PWM Comparison)

This document summarizes the **actual current pipeline** in this repo, with emphasis on fair comparison against PWM.

## Scope

- Codebase: `Flow-MBPO-PWM`
- Main entrypoints:
  - `scripts/pretrain_multitask_wm.py` (offline world-model pretraining)
  - `scripts/train_multitask.py` (offline dataset policy training per task)
  - `scripts/train_dflex.py` (single-task online training with real env interaction)
- Core algorithm: `src/flow_mbpo_pwm/algorithms/pwm.py`

## 0) Shared Building Blocks (All Variants)

```text
AGENT = PWM(
    actor_config, critic_config, world_model_config,
    horizon H, gamma, lambda, learning_rates,
    use_flow_dynamics, flow_integrator, flow_substeps, flow_tau_sampling
)

World model choices:
  - MLP WM:  flow_mbpo_pwm.models.world_model.WorldModel
  - Flow WM: flow_mbpo_pwm.models.flow_world_model.FlowWorldModel

Actor choices:
  - MLP policy:  flow_mbpo_pwm.models.actor.ActorStochasticMLP
  - Flow policy: flow_mbpo_pwm.models.flow_actor.ActorFlowODE
```

## 1) Stage A: Offline WM Pretraining (Only WM is trained)

```text
PROCEDURE PRETRAIN_WM(config, wm_variant):
    # wm_variant in {MLP_WM, FLOW_WM}
    # e.g., alg=pwm_48M_mt_baseline or alg=pwm_48M_mt_flowwm

    infer task_set and task_dim from data_dir (MT30->64, MT80->96)
    load all *.pt TD-MPC2 files into replay buffer

    instantiate PWM agent with env=None
    # actor/critic are built but NOT updated in this stage

    FOR iter in [1..wm_pretrain_iters]:
        obs, act, rew, task = buffer.sample_with_task()
        task_ids = extract scalar task ids

        wm_loss, dyn_loss, rew_loss = compute_wm_loss(obs, act, rew, task_ids)
        optimizer_step(world_model_only)

        track best wm_loss
        save best checkpoint periodically

    save final checkpoint
```

Notes:
- This is implemented in `scripts/pretrain_multitask_wm.py`.
- Output checkpoint has `world_model` weights (plus optimizer/meta).
- There is currently **no standalone policy pretraining script**.

## 2) Stage B: Task-Specific Offline Policy Training (From Dataset)

```text
PROCEDURE TRAIN_MULTITASK_TASK(config, variant, checkpoint, finetune_wm):
    build MT30/MT80 task list, pick one target task
    make multitask eval env wrapper

    instantiate PWM agent (env=None)

    IF resume_from is set:
        load full checkpoint (actor+critic+wm+optimizers+progress)
    ELSE IF checkpoint is set:
        load_wm(checkpoint)  # WM-only load
        wm_bootstrapped = True

    load offline dataset chunks (*.pt)
    filter episodes by target task_id
    add filtered episodes to training buffer

    FOR epoch in [start_epoch..epochs]:
        update_lrs(epoch)
        obs, act, rew = buffer.sample()
        metrics = agent.update(obs, act, rew, task_ids, finetune_wm)

        every eval_freq:
            evaluate in real env
            save last and best model

    save final model and csv metrics
```

`agent.update(...)` logic (shared):

```text
IF finetune_wm == True:
    update world model with compute_wm_loss(...)

update actor using compute_actor_loss(...)
update critic using TD-lambda targets
return metrics
```

## 3) Variant Branches You Asked For

## 3.1 Pretrain Flow WM (Flow WM + MLP Policy)

```text
Stage A:
    pretrain WM with alg=pwm_48M_mt_flowwm  -> flowwm_*_best.pt

Stage B:
    train task policy with alg=pwm_48M_mt_flowwm
    load_wm(flowwm_*_best.pt)
    actor = MLP policy
    world model = Flow WM
    use_flow_dynamics = True
```

## 3.2 Flow Policy (Pretrained MLP WM + Flow Policy)

```text
Stage A:
    pretrain WM with alg=pwm_48M_mt_baseline -> mlpwm_*_best.pt

Stage B:
    train task policy with alg=pwm_48M_mt_flowpolicy
    load_wm(mlpwm_*_best.pt)
    actor = Flow ODE policy
    world model = MLP WM
    use_flow_dynamics = False
```

## 3.3 Both (Pretrained Flow WM + Flow Policy)

```text
Stage A:
    pretrain WM with alg=pwm_48M_mt_flowwm -> flowwm_*_best.pt

Stage B:
    train task policy with alg=pwm_48M_mt_fullflow
    load_wm(flowwm_*_best.pt)
    actor = Flow ODE policy
    world model = Flow WM
    use_flow_dynamics = True
```

## 4) Inner Losses (Pseudo Code)

```text
PROCEDURE COMPUTE_WM_LOSS(obs[H+1], act[H], rew[H], task):
    next_z_target[t] = stopgrad(encode(obs[t+1], task))
    z = encode(obs[0], task)

    IF use_flow_dynamics:
        FOR t in [0..H-1]:
            dynamics_loss += FlowMatchingLoss(
                velocity_fn=wm.velocity,
                z_start=z,
                z_target=next_z_target[t],
                action=act[t],
                tau_sampling in {uniform, midpoint},
                gamma_weight=gamma^t
            )
            z = wm.next(z, act[t], task, integrator, substeps)
    ELSE:
        FOR t in [0..H-1]:
            z = wm.next(z, act[t], task)
            dynamics_loss += gamma^t * MSE(z, next_z_target[t])

    reward_loss = mean_t( gamma^t * (wm.reward(z_t, act_t)-rew_t)^2 )
    total = (dynamics_loss + reward_loss) / H
    return total
```

```text
PROCEDURE COMPUTE_ACTOR_LOSS(obs0, task):
    z = encode(obs0, task)
    FOR t in [0..H-1]:
        a_t = tanh(actor(z))
        IF use_flow_dynamics:
            z, r_hat = wm.step(z, a_t, task, integrator, substeps)
        ELSE:
            z, r_hat = wm.step(z, a_t, task)
        collect rollout buffers for critic targets
    optimize negative imagined return (with bootstrap term)
```

## 5) Fair Comparison Checklist vs PWM

Use this checklist when comparing with PWM:

1. Keep shared hyperparameters fixed across variants (`horizon`, `gamma`, `lam`, LRs, `wm_batch_size`, `wm_iterations`, `wm_buffer_size`).
2. Use the same task set, dataset split, seeds, epochs, eval frequency, and eval runs.
3. For a clean 2x2 factorial, use matched WM checkpoints:
   - MLP-WM checkpoint for baseline/flow-policy branches.
   - Flow-WM checkpoint for flow-WM/full-flow branches.
4. Keep `finetune_wm` policy consistent across all branches in one comparison.
5. Change only intended factors:
   - Flow WM factor: `world_model_config` + `use_flow_dynamics`
   - Flow Policy factor: `actor_config`

## 6) Natural-Language Pipeline Summary (Major Steps + Substeps)

1. Define the Flow-MBPO Experimental Condition  
   (a) Choose one of the four 2x2 conditions: Baseline (`MLP WM + MLP policy`), Flow WM (`Flow WM + MLP policy`), Flow Policy (`MLP WM + Flow policy`), or Full Flow (`Flow WM + Flow policy`).  
   (b) Map the condition to the correct config: `pwm_48M_mt_baseline`, `pwm_48M_mt_flowwm`, `pwm_48M_mt_flowpolicy`, or `pwm_48M_mt_fullflow`.  
   (c) Fix shared controls for fairness: same tasks, seeds, epoch budget, evaluation frequency, and shared optimizer/replay settings across all conditions.  

2. Data Preparation (Flow-MBPO Multi-Task Path)  
   (a) Use TD-MPC2 offline multitask files (`*.pt`) from `general.data_dir`.  
   (b) Infer MT30 vs MT80 from the data path and set task metadata (`task_dim=64` for MT30, `task_dim=96` for MT80).  
   (c) Configure multitask world-model metadata (task list and action-dimension masks).  
   (d) Validate episode length consistency before training; stop if data shape and config do not match.  

3. Stage A: World-Model Pretraining (Offline WM-Only)  
   (a) Run `scripts/pretrain_multitask_wm.py` with `alg=pwm_48M_mt_baseline` (MLP-WM pretrain) or `alg=pwm_48M_mt_flowwm` (Flow-WM pretrain).  
   (b) Instantiate PWM with `env=None`; actor and critic exist but are not optimized in this stage.  
   (c) Load all offline episodes into replay buffer and sample `(obs, act, rew, task)` slices.  
   (d) Compute WM loss each iteration:  
   - Flow WM: flow-matching dynamics loss + reward loss,  
   - MLP WM: latent MSE dynamics loss + reward loss.  
   (e) Apply gradient clipping and update only WM parameters; log WM metrics and save `*_best.pt` and `*_last.pt`.  

4. Stage B: Task-Specific Policy Training with Pretrained WM  
   (a) Run `scripts/train_multitask.py` for a selected task (for example one MT30 task per job).  
   (b) Load initialization state:  
   - `resume_from`: restore full model/optimizer/training counters,  
   - `general.checkpoint`: load WM only through `load_wm(...)`.  
   (c) Filter offline dataset by target `task_id`, add only that task’s episodes to replay buffer, and abort if no data remains.  
   (d) Start epoch loop and call `agent.update(obs, act, rew, task_ids, finetune_wm)` after LR scheduling each epoch.  

5. Per-Epoch Optimization Flow (Core Flow-MBPO Mechanics)  
   (a) Actor update: roll out `H` latent steps through current WM and optimize actor by first-order gradients through imagined returns with bootstrap.  
   (b) Critic update: compute TD-lambda targets and run configured critic mini-batch iterations.  
   (c) WM update: run only when `finetune_wm=True`; otherwise keep WM frozen during policy learning.  
   (d) Apply clipping/schedules and log losses, grad norms, and reward-related diagnostics.  

6. Branch-Specific Flow Logic (What Actually Changes)  
   (a) Flow WM branch: world model is `FlowWorldModel`, `use_flow_dynamics=True`, and WM transition uses configured ODE integrator/substeps.  
   (b) Flow Policy branch: actor is `ActorFlowODE`, world model remains MLP, and `use_flow_dynamics=False` for WM rollout.  
   (c) Full Flow branch: both Flow actor and Flow world model are active; WM rollout uses flow integrator/substeps while actor samples actions via ODE policy.  
   (d) Baseline branch: both actor and WM are MLP-based under the same training scaffold.  

7. Data Flow Guarantees and Correctness Conditions  
   (a) In multitask offline training, updates sample only from offline replay for the selected task unless explicit WM finetuning behavior is enabled.  
   (b) In single-task online training (`train_dflex.py`), real environment transitions are appended to replay while optimization remains model-based and short-horizon.  
   (c) Done/termination/truncation handling must correctly reset rollout state to prevent corrupted targets and unstable gradients.  

8. Evaluation, Logging, and Output Artifacts  
   (a) Run periodic real-environment evaluation using configured `eval_freq` and `eval_runs`.  
   (b) Record task metrics (for example `episode_reward`, `episode_success`) and save `model_last`/`model_best` during training.  
   (c) At training end, save `model_final`, run final evaluation, and run planning-enabled evaluation when required (`episode_reward_planning`, `episode_success_planning`).  
   (d) Export run artifacts: checkpoints, per-task CSV metrics, and experiment logs for later aggregation/reporting.  

9. Final Validation Checklist for Professor-Level Comparability  
   (a) Confirm the condition-to-checkpoint mapping is correct: MLP-WM checkpoints for Baseline/Flow-Policy, Flow-WM checkpoints for Flow-WM/Full-Flow.  
   (b) Confirm non-target settings did not drift across conditions (data split, seeds, epochs, eval protocol, shared hyperparameters, and `finetune_wm` policy).  
   (c) Confirm update order is preserved in implementation: actor rollout/objective, critic TD update, then WM update when enabled.  
   (d) Confirm flow-specific knobs (`use_flow_dynamics`, `flow_integrator`, `flow_substeps`, actor type) match the intended branch.  
   (e) Document reward-loss formulation differences versus original PWM paper before claiming exact reproduction.  

## 7) File Mapping (Quick Reference)

- WM pretraining: `scripts/pretrain_multitask_wm.py`
- Task policy training: `scripts/train_multitask.py`
- Online single-task training: `scripts/train_dflex.py`
- Core update/train/loss code: `src/flow_mbpo_pwm/algorithms/pwm.py`
- MT factorial configs:
  - `scripts/cfg/alg/pwm_48M_mt_baseline.yaml`
  - `scripts/cfg/alg/pwm_48M_mt_flowwm.yaml`
  - `scripts/cfg/alg/pwm_48M_mt_flowpolicy.yaml`
  - `scripts/cfg/alg/pwm_48M_mt_fullflow.yaml`
