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
