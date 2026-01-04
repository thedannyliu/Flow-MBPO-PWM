# MT30 Experiment Log

> Registry of all MT30 multitask experiments. Status: `QUEUED` → `RUNNING` → `COMPLETED` / `FAILED`

---

## Phase 3: Baseline vs Flow Policy Comparison (COMPLETED)

**Goal**: Compare MLP Baseline vs Flow Policy using pretrained World Model.
**Config**: `pwm_48M_mt_baseline` vs `pwm_48M_mt_flowpolicy`
**WM**: Pretrained (`mt30_48M_4900000.pt`), frozen (`finetune_wm=False`)
**Epochs**: 10,000

### Summary Results

| Task | Baseline (MLP) | Flow Policy (ODE) | Winner |
|------|---------------|-------------------|--------|
| **reacher-easy** | 982.30 ± 1.2 | 980.67 ± 3.4 | Tie (Solved) |
| **walker-stand** | 957.72 ± 27.5 | 839.78 ± 87.6 | Baseline (+14%) |
| **cheetah-run** | 112.48 ± 20.7 | 98.74 ± 20.0 | Tie (Both Low) |

### Detailed Results

| Config | Task | Seed | Reward | Plan Reward | Job ID |
|--------|------|------|--------|-------------|--------|
| Baseline | reacher-easy | 42 | 981.20 | 981.70 | 4011713_0 |
| Baseline | reacher-easy | 123 | 983.50 | 985.60 | 4011713_1 |
| Baseline | reacher-easy | 456 | 982.20 | 985.50 | 4011713_2 |
| Baseline | walker-stand | 42 | 972.32 | 943.97 | 4011713_3 |
| Baseline | walker-stand | 123 | 923.48 | 933.11 | 4011713_4 |
| Baseline | walker-stand | 456 | 977.35 | 978.33 | 4011713_5 |
| Baseline | cheetah-run | 42 | 93.69 | 88.13 | 4011713_6 |
| Baseline | cheetah-run | 123 | 108.80 | 81.67 | 4011713_7 |
| Baseline | cheetah-run | 456 | 134.97 | 114.09 | 4011713_8 |
| Flow Policy | reacher-easy | 42 | 976.70 | 982.90 | 4011714_0 |
| Flow Policy | reacher-easy | 123 | 983.40 | 986.00 | 4011714_1 |
| Flow Policy | reacher-easy | 456 | 981.90 | 983.90 | 4011714_2 |
| Flow Policy | walker-stand | 42 | 854.53 | 864.39 | 4011740_3 |
| Flow Policy | walker-stand | 123 | 744.92 | 796.67 | 4011714_4 |
| Flow Policy | walker-stand | 456 | 919.90 | 933.66 | 4011740_5 |
| Flow Policy | cheetah-run | 42 | 80.97 | 82.31 | 4011740_6 |
| Flow Policy | cheetah-run | 123 | 94.75 | 73.21 | 4011740_7 |
| Flow Policy | cheetah-run | 456 | 120.52 | 110.27 | 4011740_8 |

---

## Phase 4: Full Flow Model (RUNNING)

**Goal**: Train Flow World Model + Flow Policy from scratch.
**Config**: `pwm_48M_mt_fullflow`
**WM**: Trained from scratch (`finetune_wm=True`)
**Epochs**: 15,000
**Status**: 🟢 Resubmitted after storage cleanup.

| Job ID | Array | Tasks | Seeds | Status |
|--------|-------|-------|-------|--------|
| 4012433 | 0-8 | reacher-easy, walker-stand, cheetah-run | 42, 123, 456 | QUEUED |

---

## Phase 5: Flow Hyperparameter Tuning (RUNNING)

**Goal**: Tune Flow integration parameters for better performance.
**Config**: `pwm_48M_mt_fullflow` with variations
**Job ID**: 4012434 (Array 0-17)
**Status**: 🟢 Resubmitted after storage cleanup.

### Variations Being Tested

| Variation | Parameter Changes |
|-----------|-------------------|
| `high_precision_wm` | `flow_substeps=8` (WM) |
| `high_precision_policy` | `actor_config.flow_substeps=4` |
| `euler_fast` | `flow_integrator=euler` (both) |

### Partial Completed Results (Euler Fast Only)

| Task | Seed | Reward | Plan Reward |
|------|------|--------|-------------|
| walker-stand | 42 | 142.30 | 128.08 |
| walker-stand | 123 | 153.06 | 151.15 |

---

## Configuration Reference

| Config File | World Model | Policy | Description |
|-------------|-------------|--------|-------------|
| `pwm_48M_mt_baseline.yaml` | MLP | MLP | Original PWM baseline |
| `pwm_48M_mt_flowpolicy.yaml` | MLP | Flow ODE | Flow policy only |
| `pwm_48M_mt_fullflow.yaml` | Flow | Flow ODE | Full flow (WM + Policy) |

## Resource Settings

| Partition | GPU | Memory | Time | Account |
|-----------|-----|--------|------|---------|
| ice-gpu | H100/H200 | 450GB | 16h | coc |
| coc-gpu | L40S/A100 | 450GB | 16h | coc |

## Data & Checkpoints

| Item | Path |
|------|------|
| Pretrained WM | `checkpoints/multitask/mt30_48M_4900000.pt` |
| Training Data | `/home/hice1/eliu354/scratch/Data/tdmpc2/mt30/` |
