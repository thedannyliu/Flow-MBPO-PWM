# Flow-MBPO-PWM

Flow Matching Model-Based Policy Optimization based on PWM (Policy Learning with Large World Models).

## Overview

This repository implements Flow-based world models for first-order gradient policy optimization, building on the [PWM framework](https://github.com/imgeorgiev/PWM). We explore whether Flow Matching can provide better-behaved surrogate dynamics for model-based RL.

## Installation

```bash
# Clone the repository
git clone git@github.com:thedannyliu/Flow-MBPO-PWM.git
cd Flow-MBPO-PWM

# Create conda environment
conda env create -f environment.yaml
conda activate pwm

# Install package in editable mode
pip install -e .
```

## Quick Start

### Single-Task Training (dflex_ant)

**Baseline (MLP World Model):**
```bash
cd scripts
python train_dflex.py alg=pwm_5M_baseline_final
```

**Flow World Model (Heun, K=4):**
```bash
cd scripts
python train_dflex.py alg=pwm_5M_flow_v2_substeps4
```

### Slurm Submission (Georgia Tech PACE Phoenix)

```bash
# Baseline
sbatch scripts/submit_5M_baseline_l40s_final.sh

# Flow World Model
sbatch scripts/submit_5M_flow_v2_l40s_final.sh
```

## Configuration

Configs are in `scripts/cfg/`:

| Config | Description |
|--------|-------------|
| `alg/pwm_5M_baseline_final.yaml` | MLP world model baseline |
| `alg/pwm_5M_flow_v1_substeps2.yaml` | Flow WM, Heun, K=2 |
| `alg/pwm_5M_flow_v2_substeps4.yaml` | Flow WM, Heun, K=4 (recommended) |
| `alg/pwm_5M_flow_v3_substeps8_euler.yaml` | Flow WM, Euler, K=8 |
| `env/dflex_ant.yaml` | Ant locomotion environment |

### Key Hyperparameters (Aligned for Fair Comparison)

Both baseline and flow configs use:
- `wm_batch_size: 256`
- `wm_buffer_size: 1_000_000`
- `num_envs: 128`
- `max_epochs: 15_000`
- `horizon: 16`

Flow-specific parameters:
- `use_flow_dynamics: true/false`
- `flow_integrator: heun/euler`
- `flow_substeps: 2/4/8`

## Project Structure

```
Flow-MBPO-PWM/
├── src/pwm/                 # Source code
│   ├── algorithms/          # PWM training algorithm
│   ├── models/              # WorldModel, FlowWorldModel, Actor
│   └── utils/               # Helpers, integrators, monitoring
├── scripts/                 # Training scripts and configs
├── docs/                    # Documentation
│   ├── master_plan.md       # Research plan and phases
│   ├── progress_log.md      # Development log
│   └── experiment_log.md    # Experiment registry
└── environment.yaml         # Conda environment
```

## Documentation

- [Master Plan](docs/master_plan.md) - Research questions, phases, and methodology
- [Progress Log](docs/progress_log.md) - Development timeline
- [Experiment Log](docs/experiment_log.md) - Experiment results

## Citation

```bibtex
@misc{georgiev2024pwm,
    title={PWM: Policy Learning with Large World Models},
    author={Ignat Georgiev, Varun Giridha, Nicklas Hansen, and Animesh Garg},
    eprint={2407.02466},
    archivePrefix={arXiv},
    primaryClass={cs.LG},
    year={2024}
}
```

## License

MIT License
