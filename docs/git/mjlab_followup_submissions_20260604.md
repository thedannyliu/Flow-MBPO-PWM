# MJLab Follow-Up Submissions - 2026-06-04

## Scope

This note records the 2026-06-04 submissions for four requested tracks:

1. Flow-MBPO endpoint H1 multi-seed plus ratio ablation.
2. PWM collapse diagnostics.
3. Offline mixed dataset to Flow-MBPO-style policy extraction.
4. Controlled 2x2 Flow/MLP matrix.

It also adds dataset-distribution probes: all expert, expert 50%, and no-fall/no-done success-100 proxy data.

## Code Change

Added real-data sampler controls to `run_flow_mbpo_v0_awr_update.py`:

- `--real-quality-mixture`: target real-batch quality mixture, for example `expert:0.5,medium:0.5`.
- `--real-require-no-fall-window`: keep only real windows with no termination flags.
- `--real-require-no-done-window`: keep only real windows with no done flags.

`run_flow_mbpo_awr_row.py` now passes these fields from CSV manifests.

Manifest builder:

- `scripts/experiments/mjlab_qs/build_20260604_followup_manifests.py`

## Validation

Completed before submission:

- `python -m py_compile scripts/experiments/mjlab_qs/run_flow_mbpo_v0_awr_update.py scripts/experiments/mjlab_qs/run_flow_mbpo_awr_row.py`
- `python -m py_compile scripts/experiments/mjlab_qs/build_20260604_followup_manifests.py`
- AWR row dry-runs with `--check-inputs` for:
  - `expert50_medium50`
  - `nofall_nodone_success100_proxy`
  - endpoint H1 fix1 row 0
- Manifest path audit for all generated manifests.
- In-memory sampler smoke in the `pwm` conda env confirmed:
  - no-fall/no-done filtering removes terminal rows.
  - `expert:0.5,medium:0.5` mixture builds separate quality groups and samples from both.

## Cancelled Premature-Stop Submission

Initial AWR manifests used `real_eval_stop_score_below=-80.0`.

H200 row 0 reached the 250-iteration eval and stopped early:

- Job: `9419604_0`
- State: `COMPLETED`
- Output root: `scripts/outputs/mjlab_qs/flow_mbpo_endpoint_h1_multiseed_ablation_20260604/h200/endpoint_h1_r224_s32_anchor1_iter500_s0`
- Real eval at iter 250:
  - return: `17.3596`
  - length: `261.75`
  - fall: `1.0`
  - selection score: `-80.0229`
- Root cause: selection score was just below `-80.0`, so the row stopped before the intended 500 AWR iterations.

Cancelled affected arrays:

| Job ID | Rows | Reason |
| --- | --- | --- |
| `9419604_[1-8]` | endpoint H1 old H200 rows | premature early-stop threshold |
| `9419606_[0-9]` | data distribution old H100 rows | same threshold |

Fix1 manifests disable this stop gate with `real_eval_stop_score_below=-1000000000.0`.

## Active Submissions

### Flow-MBPO Endpoint H1 Multi-Seed/Ablation

Manifest:

- `scripts/experiments/mjlab_qs/manifests/flow_mbpo_endpoint_h1_multiseed_ablation_fix1_h200_20260604.csv`

Job:

- `9419642`
- GPU: H200
- Array: `0-8%3`
- QOS: `embers`

Rows:

| Seeds | Ratios | Synthetic source |
| --- | --- | --- |
| `0,1,2` | `real224/syn32`, `real192/syn64`, `real248/syn8` | `flow_endpoint_ensemble_seed0_h1_unc0p5_q0p90_truncate_check/synthetic_replay.pt` |

Output root:

- `scripts/outputs/mjlab_qs/flow_mbpo_endpoint_h1_multiseed_ablation_fix1_20260604/h200/`

Note: this is multi-seed policy/AWR extraction using the currently validated endpoint H1 synthetic replay from WM seed 0. It does not yet regenerate seed-matched synthetic replay for WM seeds 1/2.

### Offline Dataset Distribution to Flow-MBPO-Style Extraction

Manifest:

- `scripts/experiments/mjlab_qs/manifests/flow_mbpo_data_distribution_awr_fix1_h100_20260604.csv`

Job:

- `9419641`
- GPU: H100
- Array: `0-9%3`
- QOS: `embers`

Rows:

| Distribution | Seeds | Definition |
| --- | --- | --- |
| `mixed_uniform_windows` | `0,1` | `expert,expert_noisy,medium,random_smooth`, uniform over selected windows |
| `expert_only` | `0,1` | only `expert` windows |
| `expert50_medium50` | `0,1` | real batch mixture `expert:0.5,medium:0.5` |
| `expert50_noisy50` | `0,1` | real batch mixture `expert:0.5,expert_noisy:0.5` |
| `nofall_nodone_success100_proxy` | `0,1` | `expert,expert_noisy,medium` windows with no termination and no done flags |

Output root:

- `scripts/outputs/mjlab_qs/flow_mbpo_data_distribution_awr_fix1_20260604/h100/`

Success-100 proxy definition:

- For MJLab locomotion there is no binary task success label in this dataset.
- The proxy used here is no-fall/no-done windows: no `termination` and no `done` flag within the H16 window.

### Controlled 2x2 Flow/MLP Matrix

Manifest:

- `scripts/experiments/mjlab_qs/manifests/rerun_g1_bcwarm_pwm_bcreg10_2x2_allexpert_a100_20260604.csv`

Job:

- `9419605`
- GPU: A100
- Array: `0-11%4`
- QOS: `embers`

Rows:

- WM: `mlp_ref`, `flow_endpoint`
- Policy: `mlp`, `flow`
- Seeds: `0,1,2`
- Dataset filter: all-expert for BC warm start and PWM policy extraction
- BC warm start: `50000`
- PWM policy iterations: `2000`
- BC regularization: `10.0`

Output root:

- `scripts/outputs/mjlab_qs/policy_extraction/rerun_g1_bcwarm_pwm_bcreg10_2x2_allexpert_20260604/`

### PWM Collapse Diagnostics

Submitter:

- `scripts/experiments/mjlab_qs/submit_original_pwm_collapse_probes_20260603.sh`

Job:

- `9419607`
- GPU: L40S
- Array: `0-2%3`
- QOS: `embers`

Rows:

- pretrained policy
- final policy extraction checkpoint
- best policy extraction checkpoint

Output root:

- `scripts/outputs/mjlab_qs/original_pwm_collapse_probe_l40s_20260604/`

## Queue Snapshot

After fix1 submission:

- `9419642_[0-8%3]`: pending on H200.
- `9419641_[0-9%3]`: pending on H100.
- `9419605_[0-11%4]`: pending on A100.
- `9419607_[0-2%3]`: pending on L40S.
- Older NEWT flow jobs `9417059`, `9417060`, `9417061`, `9417062` were also still pending and are separate from this MJLab follow-up.

## Monitoring

Useful commands:

```bash
squeue -u eliu354 -o '%.18i %.9P %.32j %.8u %.2t %.10M %.6D %R'
sacct -j 9419642,9419641,9419605,9419607 --format=JobID,JobName%35,State,ExitCode,Elapsed,NodeList -P
```

Primary logs:

- `logs/slurm/mjlab_qs/flow_mbpo_awr/`
- `logs/slurm/mjlab_qs/policy_extract/`
- `logs/slurm/mjlab_qs/original_pwm_collapse_probe/`

## Monitoring Snapshot - 2026-06-04 02:48 EDT

### Job Status

| Job ID | Track | Status |
| --- | --- | --- |
| `9419642` | Flow-MBPO endpoint H1 multiseed/ratio | `COMPLETED`, 9/9 rows |
| `9419641` | Data-distribution Flow-MBPO-style AWR | `COMPLETED`, 10/10 rows |
| `9419605` | all-expert Flow/MLP 2x2 policy extraction | `PENDING`, 12 rows |
| `9419607` | PWM collapse diagnostics | `PENDING`, 3 rows |

### Endpoint H1 Multiseed/Ratio Results

Job `9419642` completed without Slurm failures. All rows ran to `update_iters=500` with two real-env eval snapshots.

Per-row best real-eval metrics:

| Ratio | Seed | Return | Length | Fall | Selection Score |
| --- | ---: | ---: | ---: | ---: | ---: |
| `real192/syn64` | `0` | `19.8589` | `294.75` | `1.000` | `-77.1936` |
| `real192/syn64` | `1` | `15.1366` | `234.00` | `1.000` | `-82.5234` |
| `real192/syn64` | `2` | `23.4868` | `339.125` | `1.000` | `-73.1220` |
| `real224/syn32` | `0` | `23.2943` | `344.375` | `1.000` | `-73.2619` |
| `real224/syn32` | `1` | `15.2849` | `238.625` | `1.000` | `-82.3289` |
| `real224/syn32` | `2` | `23.5517` | `334.75` | `1.000` | `-73.1008` |
| `real248/syn8` | `0` | `27.9866` | `390.625` | `0.875` | `-55.6071` |
| `real248/syn8` | `1` | `17.4911` | `269.375` | `1.000` | `-79.8152` |
| `real248/syn8` | `2` | `21.0284` | `306.375` | `1.000` | `-75.9078` |

Mean by ratio:

| Ratio | Mean Return | Mean Length | Mean Fall | Mean Score |
| --- | ---: | ---: | ---: | ---: |
| `real192/syn64` | `19.4941` | `289.292` | `1.000` | `-77.6130` |
| `real224/syn32` | `20.7103` | `305.917` | `1.000` | `-76.2305` |
| `real248/syn8` | `22.1687` | `322.125` | `0.958` | `-70.4434` |

Interpretation:

- Lower synthetic ratio (`real248/syn8`) is the least bad variant, but it is still clearly below the BC baseline used by the gate: return `45.8491`, length `594.97`, fall `0.625`.
- This fix1 multiseed sweep does not reproduce the earlier strongest Flow-MBPO endpoint H1 formal result (`final eval40` return `60.8721`, length `759.30`, fall `0.45`).
- The gap suggests the earlier strong candidate was sensitive to details not fully matched by this sweep, likely checkpoint/source artifact, eval protocol, or run configuration differences rather than a robust ratio-only effect.

### Data Distribution AWR Results

Job `9419641` completed without Slurm failures. The sampler summaries in each `summary.json` confirm that the new sampler controls were applied.

Per-row best real-eval metrics:

| Distribution | Seed | Return | Length | Fall | Selection Score |
| --- | ---: | ---: | ---: | ---: | ---: |
| `mixed_uniform_windows` | `0` | `16.0875` | `249.625` | `1.000` | `-81.4162` |
| `mixed_uniform_windows` | `1` | `16.1409` | `259.125` | `1.000` | `-81.2679` |
| `expert_only` | `0` | `18.7047` | `261.25` | `1.000` | `-78.6828` |
| `expert_only` | `1` | `15.9107` | `253.25` | `1.000` | `-81.5568` |
| `expert50_medium50` | `0` | `10.1461` | `155.375` | `1.000` | `-88.3001` |
| `expert50_medium50` | `1` | `10.6156` | `186.375` | `1.000` | `-87.5206` |
| `expert50_noisy50` | `0` | `20.0458` | `270.375` | `1.000` | `-77.2505` |
| `expert50_noisy50` | `1` | `17.3762` | `273.75` | `1.000` | `-79.8863` |
| `nofall_nodone_success100_proxy` | `0` | `13.4583` | `205.5` | `1.000` | `-84.4867` |
| `nofall_nodone_success100_proxy` | `1` | `13.8603` | `228.0` | `1.000` | `-83.8597` |

Mean by distribution:

| Distribution | Mean Return | Mean Length | Mean Fall | Mean Score |
| --- | ---: | ---: | ---: | ---: |
| `mixed_uniform_windows` | `16.1142` | `254.375` | `1.000` | `-81.3421` |
| `expert_only` | `17.3077` | `257.25` | `1.000` | `-80.1198` |
| `expert50_medium50` | `10.3809` | `170.875` | `1.000` | `-87.9104` |
| `expert50_noisy50` | `18.7110` | `272.063` | `1.000` | `-78.5684` |
| `nofall_nodone_success100_proxy` | `13.6593` | `216.75` | `1.000` | `-84.1732` |

Interpretation:

- Data-distribution changes alone did not rescue this Flow-MBPO-style AWR extraction.
- `expert50_noisy50` is the best data-distribution mean, but it is still much worse than the BC baseline and all rows have fall rate `1.0`.
- The no-fall/no-done success-100 proxy did not help; this suggests simply filtering to clean windows is not enough when the synthetic replay / policy update is already destabilizing the policy.

### Current Follow-Up

- Do not cancel `9419605` or `9419607`; they are still pending and cover separate diagnostics.
- The next useful check is comparing these weak fix1 rows against the exact artifact/config of the earlier strong endpoint H1 candidate, especially:
  - policy checkpoint source and seed,
  - synthetic replay path and truncation settings,
  - formal eval checkpoint choice (`final` vs `best`),
  - whether the previous `final eval40` used a different output directory than this fix1 sweep.
