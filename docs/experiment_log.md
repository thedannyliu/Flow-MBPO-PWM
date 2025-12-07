# Experiment Log (Runs, Metrics, Jobs)

Purpose: Central registry of all training/eval jobs across phases. Record every submitted Slurm job (including smoke tests) with config, seeds, metrics, status, and links to logs/checkpoints. Keep entries concise, in English, and append-only (newest at top).

How to update:
- Add a row for each job as soon as it is submitted; update status/metrics when it finishes or fails.
- Include the Slurm job ID, W&B group/name, config file, and script path so runs are reproducible.
- Point to log files under `logs/slurm/...` and checkpoints under `outputs/...` per the conventions in `docs/master_plan.md` section 10.
- When aggregating multiple seeds, add one row per seed and a short aggregated note once metrics are summarized.

Columns (table):
- `Date`: submission date (YYYY-MM-DD).
- `Phase`: 1 / 1.5 / 2 / 3 (or smoke).
- `Env`: dflex_ant, MT30, MT80, etc.
- `Config`: Hydra alg name and key flags (H, K, K_policy, integrator, regularization tag).
- `Seed`: integer.
- `Script`: Slurm script path used.
- `JobID`: Slurm job ID (`%j`).
- `W&B`: project/group/name or link.
- `Status`: queued/running/succeeded/failed/cancelled.
- `Metrics`: final reward (mean), ESNR, notable diagnostics; `n/a` until available.
- `Logs`: paths to `.out/.err`.
- `Notes`: brief remarks (fail reason, anomalies, checkpoint path).

Template row (copy/paste and fill):
```
| Date       | Phase | Env       | Config                               | Seed | Script                                                | JobID   | W&B                                         | Status    | Metrics                        | Logs                                                     | Notes |
|------------|-------|-----------|--------------------------------------|------|-------------------------------------------------------|---------|----------------------------------------------|-----------|--------------------------------|---------------------------------------------------------|-------|
| 2025-12-07 | smoke | dflex_ant | pwm_5M_baseline_final H=16 K=1       | 42   | scripts/phase1/submit_dflex_ant_baseline_l40s.sh     | <jobid> | flow-mbpo-pwm/phase1_dflex_ant_baseline/... | queued    | n/a                            | logs/slurm/phase1/dflex_ant/train_baseline_seed42_%j... | n/a   |
```

---

## Submitted Jobs

| Date       | Phase | Env       | Config                               | Seed | Script                                            | JobID   | W&B                                    | Status    | Metrics | Logs                                                      | Notes                            |
|------------|-------|-----------|--------------------------------------|------|---------------------------------------------------|---------|----------------------------------------|-----------|---------|-----------------------------------------------------------|----------------------------------|
| 2025-12-07 | smoke | dflex_ant | pwm_5M_baseline_final H=16, 10 epochs| 42   | scripts/smoke_test_baseline.sh                   | 2571819 | n/a                                    | running   | n/a     | logs/smoke/test_baseline_cpu_2571819.out                 | Smoke test 10 steps - running OK |
| 2025-12-07 | 1     | dflex_ant | pwm_5M_baseline_final H=16           | 42   | scripts/phase1/submit_dflex_ant_5M_baseline_l40s.sh | 2571823 | flow-mbpo-pwm/phase1_dflex_ant_baseline | queued   | n/a     | logs/slurm/phase1/dflex_ant/train_5M_baseline_seed42_2571823.out | Phase 1 baseline 5M           |
| 2025-12-07 | 1     | dflex_ant | pwm_5M_flow_v2 H=16 K=4 Heun         | 42   | scripts/phase1/submit_dflex_ant_5M_flow_v2_l40s.sh  | 2571824 | flow-mbpo-pwm/phase1_dflex_ant_flow_v2 | queued   | n/a     | logs/slurm/phase1/dflex_ant/train_5M_flow_v2_seed42_2571824.out | Phase 1 Flow WM 5M            |
| 2025-12-07 | 1     | dflex_ant | pwm_48M_baseline H=16                | 42   | scripts/phase1/submit_dflex_ant_48M_baseline_l40s.sh | 2571831 | flow-mbpo-pwm/phase1_dflex_ant_baseline_48M | queued | n/a | logs/slurm/phase1/dflex_ant/train_48M_baseline_seed42_2571831.out | Phase 1 baseline 48M          |
| 2025-12-07 | 1     | dflex_ant | pwm_48M_flow_v2 H=16 K=4 Heun        | 42   | scripts/phase1/submit_dflex_ant_48M_flow_v2_l40s.sh | 2571832 | flow-mbpo-pwm/phase1_dflex_ant_flow_v2_48M | running | n/a | logs/slurm/phase1/dflex_ant/train_48M_flow_v2_seed42_2571832.out | Phase 1 Flow WM 48M - running |
| 2025-12-07 | 1.5   | dflex_ant | pwm_5M_flow_v2 H=8 K=4 regbase       | 42   | scripts/phase1p5/submit_dflex_ant_5M_flow_v2_H8_regbase_l40s.sh | 2571834 | flow-mbpo-pwm/phase1p5_dflex_ant_flow_v2_H8_base | queued | n/a | logs/slurm/phase1p5/dflex_ant/train_5M_flow_v2_H8_base_seed42_2571834.out | Phase 1.5 H=8 base reg |
| 2025-12-07 | 1.5   | dflex_ant | pwm_5M_flow_v2 H=8 K=4 regstrong     | 42   | scripts/phase1p5/submit_dflex_ant_5M_flow_v2_H8_regstrong_l40s.sh | 2571835 | flow-mbpo-pwm/phase1p5_dflex_ant_flow_v2_H8_strong | queued | n/a | logs/slurm/phase1p5/dflex_ant/train_5M_flow_v2_H8_str_seed42_2571835.out | Phase 1.5 H=8 L2=3e-4 |
| 2025-12-07 | 1.5   | dflex_ant | pwm_5M_flow_v2 H=16 K=4 regbase      | 42   | scripts/phase1p5/submit_dflex_ant_5M_flow_v2_H16_regbase_l40s.sh | 2571836 | flow-mbpo-pwm/phase1p5_dflex_ant_flow_v2_H16_base | queued | n/a | logs/slurm/phase1p5/dflex_ant/train_5M_flow_v2_H16_base_seed42_2571836.out | Phase 1.5 H=16 base reg |
| 2025-12-07 | 1.5   | dflex_ant | pwm_5M_flow_v2 H=16 K=4 regstrong    | 42   | scripts/phase1p5/submit_dflex_ant_5M_flow_v2_H16_regstrong_l40s.sh | 2571837 | flow-mbpo-pwm/phase1p5_dflex_ant_flow_v2_H16_strong | queued | n/a | logs/slurm/phase1p5/dflex_ant/train_5M_flow_v2_H16_str_seed42_2571837.out | Phase 1.5 H=16 L2=3e-4 |
| 2025-12-07 | 2     | dflex_ant | pwm_5M_flowWM_flowpol H=8 K=4 Kpol=2 | 42   | scripts/phase2/submit_dflex_ant_5M_flowWM_flowpolicy_l40s.sh | 2571838 | flow-mbpo-pwm/phase2_dflex_ant_flowWM_flowpolicy | queued | n/a | logs/slurm/phase2/dflex_ant/train_5M_flowWM_flowpol_seed42_2571838.out | Phase 2 Flow WM + Flow policy |
| 2025-12-07 | 2     | dflex_ant | pwm_5M_mlpWM_flowpol H=8 Kpol=2      | 42   | scripts/phase2/submit_dflex_ant_5M_mlpWM_flowpolicy_l40s.sh | 2571839 | flow-mbpo-pwm/phase2_dflex_ant_mlpWM_flowpolicy | running | n/a | logs/slurm/phase2/dflex_ant/train_5M_mlpWM_flowpol_seed42_2571839.out | Phase 2 MLP WM + Flow policy - running |
