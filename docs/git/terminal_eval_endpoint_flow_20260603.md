# Terminal Evaluation And Endpoint Flow Follow-Up - 2026-06-03

## Scope

This note records the terminal-side evaluation follow-up requested after the NEWT/LeWM flow-matching ODE probes.

It covers:

- LeWM terminal evaluation for the current fm_ode predictor and prior residual-flow variants.
- Prior NEWT/LeWM metrics already exported from terminal logs.
- MJLab endpoint-flow policy extraction rows that already reached real eval.
- The existing Flow-MBPO endpoint H1 formal eval/video result.

## LeWM Terminal Eval Jobs

Added:

- `scripts/experiments/image_official/submit_lewm_terminal_eval_20260603.sh`
- `FlowMatchingJEPA` now implements `rollout`, `criterion`, and `get_cost`, matching the inference API used by upstream LeWM `eval.py`.

Local checks:

- `bash -n scripts/experiments/image_official/submit_lewm_terminal_eval_20260603.sh`
- `python -m py_compile scripts/experiments/image_official/flow_variants/lewm_flow_matching.py`
- `load_pretrained()` succeeded for:
  - `lewm_fm_ode_train_h200_fix1_20260603_pusht_predmlp_seed0/weights_epoch_1.pt`
  - `lewm_fm_ode_train_h200_fix1_20260603_pusht_predfm_ode_seed0/weights_epoch_1.pt`
  - `lewm_flow_2x2_train_h200_20260603_pusht_predflow_actionflow_seed0/weights_epoch_1.pt`

Submitted jobs:

| Job ID | GPU | Label | Rows |
| --- | --- | --- | --- |
| `9416205` | H200 | `lewm_terminal_eval_h200_20260603` | fm_ode MLP/fm_ode seeds 0/1 plus residual 2x2 MLP/MLP and Flow/Flow seeds 0/1 |
| `9416212` | H100 | `lewm_terminal_eval_h100_20260603` | same rows, backup |
| `9416214` | A100 | `lewm_terminal_eval_a100_20260603` | same rows, backup |

Status at first checks:

- H200 and H100 were pending due to priority.
- A100 was submitted as an additional backup with a distinct output root.

## MJLab Endpoint-Flow 2x2 Policy Extraction

Existing seed-0 endpoint-flow 2x2 had already completed:

- Job: `9404525_[0-3]`
- Manifest: `scripts/experiments/mjlab_qs/manifests/rerun_g1_bcwarm_pwm_bcreg10_2x2_seed0_20260603.csv`
- Summary CSV generated: `scripts/outputs/mjlab_qs/rerun_g1_bcwarm_pwm_bcreg10_2x2_seed0_20260603_summary.csv`

Seed-0 real eval return means:

| WM | Policy | Return |
| --- | --- | ---: |
| `mlp_ref` | `mlp` | `2.5724` |
| `mlp_ref` | `flow` | `-1.4830` |
| `flow_endpoint` | `mlp` | `-3.9967` |
| `flow_endpoint` | `flow` | `-1.4542` |

Existing seeds 1/2 endpoint-flow 2x2 had also already completed:

- H200 job: `9414359_[0-7]`
- H100 backup: `9414399_[0-7]`
- H200 summary CSV generated: `scripts/outputs/mjlab_qs/rerun_g1_bcwarm_pwm_bcreg10_2x2_seeds1_2_20260603_summary.csv`
- H100 summary CSV generated: `scripts/outputs/mjlab_qs/rerun_g1_bcwarm_pwm_bcreg10_2x2_seeds1_2_h100_20260603_summary.csv`

Seeds 1/2 H200 mean real eval return:

| WM | Policy | Mean Return |
| --- | --- | ---: |
| `mlp_ref` | `mlp` | `1.4459` |
| `mlp_ref` | `flow` | `-3.2666` |
| `flow_endpoint` | `mlp` | `-0.4530` |
| `flow_endpoint` | `flow` | `-1.5293` |

Seeds 1/2 H100 backup mean real eval return:

| WM | Policy | Mean Return |
| --- | --- | ---: |
| `mlp_ref` | `mlp` | `1.8103` |
| `mlp_ref` | `flow` | `-3.3095` |
| `flow_endpoint` | `mlp` | `-0.6136` |
| `flow_endpoint` | `flow` | `-1.4801` |

Interpretation:

- In this PWM-style policy extraction 2x2, `flow_endpoint` is not a win over `mlp_ref/mlp`.
- `flow_endpoint/mlp` improves over `mlp_ref/flow` and sometimes over `flow_endpoint/flow`, but the MLP reference policy remains the stronger real-eval baseline.

Duplicate submission handling:

- New duplicate arrays `9416210` and `9416209` were submitted before discovering the earlier completed arrays.
- They targeted the same output roots as completed jobs, so they were immediately cancelled with `scancel 9416209 9416210`.

## Flow-MBPO Endpoint H1 Formal Eval

This is a separate pipeline from the PWM-style 2x2 above. It uses Flow-MBPO endpoint synthetic replay plus AWR-style policy extraction.

Candidate:

- `flow_endpoint_seed0_h1_unc0p5_q0p90_cons_r224_s32_anchor1_iter500_s0`

Formal scalar eval:

| Checkpoint | Episodes | Return | Length | Fall |
| --- | ---: | ---: | ---: | ---: |
| final | 40 | `60.8721` | `759.30` | `0.450` |
| best | 40 | `46.1720` | `600.60` | `0.700` |

Rollout/video eval:

| Checkpoint | Episodes | Return | Length | Fall |
| --- | ---: | ---: | ---: | ---: |
| final | 10 | `47.4617` | `625.60` | `0.500` |
| best | 10 | `55.5533` | `707.60` | `0.400` |

Interpretation:

- This remains the strongest endpoint-flow scalar result, but it should not be conflated with the PWM-style `flow_endpoint` WM 2x2.
- The final checkpoint has the best scalar eval; the best checkpoint has stronger rollout/video fall rate.
