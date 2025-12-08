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
| 2025-12-07 | smoke | dflex_ant | pwm_5M_baseline_final H16, 10 epochs| 42   | scripts/smoke_test_baseline.sh                   | 2571819 | n/a                                    | succeeded   | R~32.8 (from job 2572354)     | logs/smoke/test_baseline_cpu_2571819.out                 | Smoke test passed |
| 2025-12-07 | 1     | dflex_ant | pwm_5M_baseline H16                 | 42   | scripts/phase1/submit_dflex_ant_5M_baseline_l40s.sh | 2581563 | flow-mbpo-pwm/phase1_dflex_ant_baseline | running   | 84% (12630/15000 epochs)     | logs/slurm/phase1/dflex_ant/train_5M_baseline_seed42_2581563.out | env_baseline, 128 envs, wandb OK, ETA ~22min |
| 2025-12-07 | 1     | dflex_ant | pwm_5M_flow_v2 H16 K4 Heun         | 42   | scripts/phase1/submit_dflex_ant_5M_flow_v2_l40s.sh  | 2581564 | flow-mbpo-pwm/phase1_dflex_ant_flow_v2 | running   | 31% (6251/20000 epochs)     | logs/slurm/phase1/dflex_ant/train_5M_flow_v2_seed42_2581564.out | Flow WM 5M, 128 envs, ETA ~4.5h |
| 2025-12-07 | 1     | dflex_ant | pwm_48M_baseline H16                | 42   | scripts/phase1/submit_dflex_ant_48M_baseline_l40s.sh | 2581581 | flow-mbpo-pwm/phase1_dflex_ant_baseline_48M | running | 2h elapsed | logs/slurm/phase1/dflex_ant/train_48M_baseline_seed42_2581581.out | Phase 1 baseline 48M |
| 2025-12-07 | 1     | dflex_ant | pwm_48M_flow_v2 H16 K4 Heun        | 42   | scripts/phase1/submit_dflex_ant_48M_flow_v2_l40s.sh | 2581582 | flow-mbpo-pwm/phase1_dflex_ant_flow_v2_48M | running | 2h elapsed | logs/slurm/phase1/dflex_ant/train_48M_flow_v2_seed42_2581582.out | Phase 1 Flow WM 48M |
| 2025-12-07 | 1.5   | dflex_ant | pwm_5M_flow_v2 H16 K4 regbase      | 42   | scripts/phase1p5/submit_dflex_ant_5M_flow_v2_H16_regbase_l40s.sh | 2581583 | flow-mbpo-pwm/phase1p5_dflex_ant_flow_v2_H16_base | queued | n/a | logs/slurm/phase1p5/dflex_ant/train_5M_flow_v2_H16_base_seed42_2581583.out | Phase 1.5 H16 base reg wm_wd=0.0 |
| 2025-12-07 | 1.5   | dflex_ant | pwm_5M_flow_v2 H16 K4 regstrong    | 42   | scripts/phase1p5/submit_dflex_ant_5M_flow_v2_H16_regstrong_l40s.sh | 2581584 | flow-mbpo-pwm/phase1p5_dflex_ant_flow_v2_H16_strong | queued | n/a | logs/slurm/phase1p5/dflex_ant/train_5M_flow_v2_H16_str_seed42_2581584.out | Phase 1.5 H16 L2_3e-4 |
| 2025-12-07 | 1.5   | dflex_ant | pwm_5M_flow_v2 H8 K4 regbase       | 42   | scripts/phase1p5/submit_dflex_ant_5M_flow_v2_H8_regbase_l40s.sh | 2581585 | flow-mbpo-pwm/phase1p5_dflex_ant_flow_v2_H8_base | queued | n/a | logs/slurm/phase1p5/dflex_ant/train_5M_flow_v2_H8_base_seed42_2581585.out | Phase 1.5 H8 base reg wm_wd=0.0 |
| 2025-12-07 | 1.5   | dflex_ant | pwm_5M_flow_v2 H8 K4 regstrong     | 42   | scripts/phase1p5/submit_dflex_ant_5M_flow_v2_H8_regstrong_l40s.sh | 2581586 | flow-mbpo-pwm/phase1p5_dflex_ant_flow_v2_H8_strong | queued | n/a | logs/slurm/phase1p5/dflex_ant/train_5M_flow_v2_H8_str_seed42_2581586.out | Phase 1.5 H8 L2_3e-4 |
| 2025-12-07 | 2     | dflex_ant | pwm_5M_mlpWM_flowpol H8 Kpol2      | 42   | scripts/phase2/submit_dflex_ant_5M_mlpWM_flowpolicy_l40s.sh | 2583678 | flow-mbpo-pwm/phase2_dflex_ant_mlpWM_flowpolicy | pending | n/a | logs/slurm/phase2/dflex_ant/train_5M_mlpWM_flowpol_seed42_2583678.out | Priority 3: MLP WM + Flow policy, waiting for resources |
| 2025-12-07 | 2     | dflex_ant | pwm_5M_flowWM_flowpol H8 K4 Kpol2 | 42   | scripts/phase2/submit_dflex_ant_5M_flowWM_flowpolicy_l40s.sh | 2583679 | flow-mbpo-pwm/phase2_dflex_ant_flowWM_flowpolicy | pending | n/a | logs/slurm/phase2/dflex_ant/train_5M_flowWM_flowpol_seed42_2583679.out | Priority 4: Flow WM + Flow policy, waiting for resources |
