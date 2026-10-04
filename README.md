# Flow-MBPO-PWM

Flow-matching world models for differentiable model-based policy optimization,
built on [PWM](https://github.com/imgeorgiev/PWM). The research question is whether
learned flow dynamics provide useful surrogate gradients for policy learning.

The repository includes MLP/flow world models, online PWM training, offline
world-model probes, and MJLab data collection and policy extraction workflows.
These are distinct experimental paths; an adapter smoke is not a full PWM run.

## Install

Use Python 3.10+. For the pinned DFlex training environment:

```bash
conda env create -f environment.yaml
conda activate pwm
python -m pip install -e . --no-deps
```

For library development in an existing compatible PyTorch environment,
`python -m pip install -e '.[dev]'` installs core dependencies and pytest.
Simulator stacks are optional: `.[simulation]` contains the broader MuJoCo/JAX
research dependencies. The uv setup expects the `mujoco_playground` submodule;
initialize it before using that environment. Do not mix it into the legacy
CUDA 11.8 DFlex environment without resolving version compatibility.

## First check: generate a manifest

This checks experiment configuration without a simulator, GPU or cluster job:

```bash
python scripts/experiments/single_task_online/build_manifest.py \
  --stage smoke --output /tmp/flow_smoke.csv
python scripts/experiments/single_task_online/split_manifest_by_cluster.py \
  --manifest /tmp/flow_smoke.csv
python -m pytest tests
```

Manifest generation and tensor unit tests do not validate learning performance.

## Choose a workflow

| Workflow | Entry points |
| --- | --- |
| Online single-task PWM | [Manifest workflow](scripts/experiments/single_task_online/README.md) |
| MJLab collection, quality gates, policy extraction | `scripts/experiments/mjlab_qs/` |
| Offline world-model probes | `scripts/experiments/world_model_phase1/` |
| Training algorithms and models | `src/flow_mbpo_pwm/{algorithms,models}/` |
| Hydra configs | `scripts/cfg/{alg,env}/` |

Flow settings include `use_flow_dynamics`, `flow_integrator` and
`flow_substeps`. Config names encode the variant; compare the resolved configs,
not only their filenames. Cluster submission is an explicit separate step.

## Results and reproducibility

Start with the [documentation index](docs/README.md). For any learning claim,
record task, seeds, dataset/checkpoint, resolved configuration, training budget
and evaluation protocol. Diagnostics, partial runs and proxy adapters must be
identified as such. No single headline result is established by the smoke above.

The [maintenance notes](docs/maintenance.md) explain removed operational retries
and consolidated exporters. Development history and experimental records remain
available in git.

## Upstream reference

Ignat Georgiev, Varun Giridha, Nicklas Hansen and Animesh Garg,
*PWM: Policy Learning with Large World Models*, arXiv:2407.02466 (2024).
Preserve upstream attribution and the terms of each dependency.
