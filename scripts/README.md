# Scripts

Run from the repository root after installing `flow_mbpo_pwm`.

| Directory / entry | Role |
| --- | --- |
| `train_dflex.py`, `train_online.py` | Training drivers |
| `cfg/` | Hydra algorithms and environments |
| `experiments/single_task_online/` | Online run manifests and submission |
| `experiments/mjlab_qs/` | Collection, data gates and policy extraction |
| `experiments/world_model_phase1/` | Offline world-model training, evaluation and summaries |
| Other `experiments/` directories | Named research protocols; inspect required artifacts first |

Build a manifest, inspect it, then submit it using that workflow's launcher.
Dated CSVs/configurations preserve historical experiments and are not defaults.
Keep generated outputs in ignored directories. See [maintenance notes](../docs/maintenance.md)
for retired scripts and renamed exporters.
