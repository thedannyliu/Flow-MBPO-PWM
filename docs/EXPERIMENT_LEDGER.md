# MJLab QS Experiment Ledger

This ledger records formal rollout evidence. Detailed chronological notes remain in `docs/phaseA/pwm_extension_submission_20260507.md`.

| Date | Stage | Git SHA | W&B | Dataset | Seed | Checkpoint | Real rollout result | Video/report | Conclusion |
|---|---|---|---|---|---:|---|---|---|---|
| 2026-05-28 | `rerun_g1_collector_reference_rollouts_20260528` expert | `83c9768` | `swwdza3g` | native collector checkpoint | 1 | iter15000 collector | return `82.6090`, length `1000.00`, fall `0.000` | `scripts/outputs/mjlab_qs/native_collector_rollouts/.../expert_seed1_iter15000/seed_1/rollout.mp4` | Collector target established. |
| 2026-05-28 | `rerun_g1_collector_reference_rollouts_20260528` expert_noisy | `83c9768` | `cu62wnbc` | native collector checkpoint | 1 | iter15000 collector | return `80.3525`, length `1000.00`, fall `0.000` | `scripts/outputs/mjlab_qs/native_collector_rollouts/.../expert_noisy_seed1_iter15000/seed_1/rollout.mp4` | Noisy expert remains stable. |
| 2026-05-28 | `rerun_g1_collector_reference_rollouts_20260528` medium | `83c9768` | `rh31kxs7` | native collector checkpoint | 2 | iter15000 collector | return `49.1935`, length `653.33`, fall `0.667` | `scripts/outputs/mjlab_qs/native_collector_rollouts/.../medium_seed2_iter15000/seed_2/rollout.mp4` | Medium baseline established. |
| 2026-05-28 | `rerun_g1_collector_reference_rollouts_20260528` random_smooth | `83c9768` | `ltam24zf` | random/reference | 0 | none | return `0.4857`, length `75.33`, fall `1.000` | `scripts/outputs/mjlab_qs/native_collector_rollouts/.../random_smooth/seed_0/rollout.mp4` | Random/reference baseline established. |
| 2026-05-27/28 | `rerun_g1_bc_expert_mlp50k_20260528` | see Phase A log | `jebhwezc`, `ucbm020h`, `xfyb30lu` rollouts | expert,expert_noisy QS windows | 0-2 | final BC policy | return `19.0827`, length `238.22`, fall `0.333` aggregate | `scripts/outputs/mjlab_qs/policy_rollouts/rerun_g1_bc_expert_mlp50k_20260528/...` | Stronger than failed PWM, still far below collector. |
| 2026-05-28 | `rerun_g1_pwm_flow_policy2x2_20260527` | see Phase A log | multiple | QS windows | 0-2 | final and true-best where available | best aggregate variant still negative return and short length | `scripts/outputs/mjlab_qs/reports/rollout_comparison_20260528.csv` | Diagnostic only; no policy-improvement claim. |
| 2026-05-28 | `rerun_g1_bcwarm_pwm_bcreg10_short2k_lr1e4_mlpwm_vs_flowwm_seed0_20260528` | `c8c4a1b` plus later logging fix `c6c74a5` | `0okrhnzj`, `co77axyn`, `r6a6kk1b`, `vpx6ve3v` rollouts | expert,expert_noisy QS windows | 0 | final and true-best | best row return `-1.0999`, length `54.67`, fall `1.000` | `scripts/outputs/mjlab_qs/policy_rollouts/rerun_g1_bcwarm_pwm_bcreg10_short2k_lr1e4_mlpwm_vs_flowwm_seed0_20260528/...` | Stronger BC anchoring reduces failure severity but does not preserve BC behavior. |

Current aggregate report:

```bash
python scripts/experiments/mjlab_qs/export_rollout_comparison.py \
  --output-csv scripts/outputs/mjlab_qs/reports/rollout_comparison_20260528.csv \
  --output-md scripts/outputs/mjlab_qs/reports/rollout_comparison_20260528.md
```
