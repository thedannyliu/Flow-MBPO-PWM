# Experiment Log (Runs, Metrics, Jobs)

Purpose: Central registry of all training/eval jobs across phases. Record every submitted Slurm job (including smoke tests) with config, seeds, metrics, status, and links to logs/checkpoints. Keep entries concise, in English, and append-only (newest at top).

## Current Status (Updated 2025-12-08 12:30 EST)

**Phase Completion:**
- Phase 1 (5M Matrix): 3/4 completed, 1 running (Priority 2 @51.8%)
- Phase 1.5 (Ablations): 4 jobs pending (H8/H16 reg sweeps)
- Phase 2 (48M Matrix): 1/4 completed, 1 running, 2 pending

**Key Findings:**
1. **Best 5M Result**: MLP WM + Flow policy (Job 2583678) → Reward **+23.43** (+41.7% over baseline +16.53)
2. **5M Matrix Results**: Priority 1 (MLP+MLP)=+16.53, Priority 3 (MLP+Flow)=**+23.43**, Priority 4 (Flow+Flow)=+16.77, Priority 2 (Flow+MLP)=running
3. **48M Baseline**: MLP+MLP → Reward **+25.14** (+52% over 5M baseline)
4. **Training Bug Clarified**: Progress bar "R=0.0" is cosmetic only. Actual training metrics in wandb (rewards=-policy_loss ~900-1100). See ROOT_CAUSE_ANALYSIS.md
5. **Next Steps**: Complete Phase 1.5 ablations (H=8 investigation), 48M Priority 3&4, additional eval runs for statistical significance

---

## How to update:
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
| 2025-12-08 | 2     | dflex_ant | pwm_48M_flowWM_flowpol H8 K4 Kpol2 | 42   | scripts/phase2/submit_dflex_ant_48M_flowWM_flowpolicy_l40s.sh | 2590207 | flow-mbpo-pwm/phase2_dflex_ant_48M_flowWM_flowpolicy | pending | n/a | logs/slurm/phase2/dflex_ant/train_48M_flowWM_flowpol_seed42_2590207.out | Priority 4 48M: Flow WM + Flow policy |
| 2025-12-08 | 2     | dflex_ant | pwm_48M_mlpWM_flowpol H8 Kpol2     | 42   | scripts/phase2/submit_dflex_ant_48M_mlpWM_flowpolicy_l40s.sh | 2590206 | flow-mbpo-pwm/phase2_dflex_ant_48M_mlpWM_flowpolicy | pending | n/a | logs/slurm/phase2/dflex_ant/train_48M_mlpWM_flowpol_seed42_2590206.out | Priority 3 48M: MLP WM + Flow policy |
| 2025-12-08 | 1.5   | dflex_ant | pwm_5M_flow_v2 H16 K4 regstrong    | 42   | scripts/phase1p5/submit_dflex_ant_5M_flow_v2_H16_regstrong_l40s.sh | 2590205 | flow-mbpo-pwm/phase1p5_dflex_ant_flow_v2_H16_strong | pending | n/a | logs/slurm/phase1p5/dflex_ant/train_5M_flow_v2_H16_str_seed42_2590205.out | Phase 1.5 H16 L2_3e-4 ablation |
| 2025-12-08 | 1.5   | dflex_ant | pwm_5M_flow_v2 H16 K4 regbase      | 42   | scripts/phase1p5/submit_dflex_ant_5M_flow_v2_H16_regbase_l40s.sh | 2590204 | flow-mbpo-pwm/phase1p5_dflex_ant_flow_v2_H16_base | pending | n/a | logs/slurm/phase1p5/dflex_ant/train_5M_flow_v2_H16_base_seed42_2590204.out | Phase 1.5 H16 base reg wm_wd=0.0 ablation |
| 2025-12-08 | 1.5   | dflex_ant | pwm_5M_flow_v2 H8 K4 regstrong     | 42   | scripts/phase1p5/submit_dflex_ant_5M_flow_v2_H8_regstrong_l40s.sh | 2590203 | flow-mbpo-pwm/phase1p5_dflex_ant_flow_v2_H8_strong | pending | n/a | logs/slurm/phase1p5/dflex_ant/train_5M_flow_v2_H8_str_seed42_2590203.out | Phase 1.5 H8 L2_3e-4 ablation |
| 2025-12-08 | 1.5   | dflex_ant | pwm_5M_flow_v2 H8 K4 regbase       | 42   | scripts/phase1p5/submit_dflex_ant_5M_flow_v2_H8_regbase_l40s.sh | 2590202 | flow-mbpo-pwm/phase1p5_dflex_ant_flow_v2_H8_base | pending | n/a | logs/slurm/phase1p5/dflex_ant/train_5M_flow_v2_H8_base_seed42_2590202.out | Phase 1.5 H8 base reg wm_wd=0.0 ablation |
| 2025-12-08 | 1     | dflex_ant | pwm_5M_flow_v2 H16 K4 Heun         | 42   | scripts/phase1/submit_dflex_ant_5M_flow_v2_l40s.sh  | 2588416 | flow-mbpo-pwm/phase1_dflex_ant_flow_v2 | running   | 5h11m elapsed, epoch ~10350/20000 (~51.8%) | logs/slurm/phase1/dflex_ant/train_5M_flow_v2_seed42_2588416.out | Priority 2 RESUBMIT with wandb fix + 450GB. Has CORRECT code, shows R=0.0 bug in progress bar (cosmetic only) |
| 2025-12-07 | 2     | dflex_ant | pwm_5M_flowWM_flowpol H8 K4 Kpol2 | 42   | scripts/phase2/submit_dflex_ant_5M_flowWM_flowpolicy_l40s.sh | 2583679 | flow-mbpo-pwm/phase2_dflex_ant_flowWM_flowpolicy | succeeded | **Eval: R=+16.77** (loss=-16.77), disc=-15.80, L=16.67, runtime 4h12m | logs/slurm/phase2/dflex_ant/train_5M_flowWM_flowpol_seed42_2583679.out | **Priority 4 COMPLETED** Flow WM + Flow policy. Training successful (see ROOT_CAUSE_ANALYSIS.md) |
| 2025-12-07 | 2     | dflex_ant | pwm_5M_mlpWM_flowpol H8 Kpol2      | 42   | scripts/phase2/submit_dflex_ant_5M_mlpWM_flowpolicy_l40s.sh | 2583678 | flow-mbpo-pwm/phase2_dflex_ant_mlpWM_flowpolicy | succeeded | **Eval: R=+23.43** (loss=-23.43), disc=-21.73, L=16.57, runtime 1h58m | logs/slurm/phase2/dflex_ant/train_5M_mlpWM_flowpol_seed42_2583678.out | **Priority 3 COMPLETED** MLP WM + Flow policy. **BEST 5M result!** +41.7% over baseline |
| 2025-12-07 | 1     | dflex_ant | pwm_48M_flow_v2 H16 K4 Heun        | 42   | scripts/phase1/submit_dflex_ant_48M_flow_v2_l40s.sh | 2581582 | flow-mbpo-pwm/phase1_dflex_ant_flow_v2_48M | running | 12h25m elapsed | logs/slurm/phase1/dflex_ant/train_48M_flow_v2_seed42_2581582.out | Priority 2 48M Flow WM, 384GB allocation may be insufficient |
| 2025-12-07 | 1     | dflex_ant | pwm_48M_baseline H16                | 42   | scripts/phase1/submit_dflex_ant_48M_baseline_l40s.sh | 2581581 | flow-mbpo-pwm/phase1_dflex_ant_baseline_48M | succeeded | **Eval: R=+25.14** (loss=-25.14), disc=-23.12, L=18.38, runtime 3h44m, 63GB peak | logs/slurm/phase1/dflex_ant/train_48M_baseline_seed42_2581581.out | **Priority 1 48M COMPLETED** MLP WM + MLP policy baseline (see ROOT_CAUSE_ANALYSIS.md) |
| 2025-12-07 | 1     | dflex_ant | pwm_5M_flow_v2 H16 K4 Heun         | 42   | scripts/phase1/submit_dflex_ant_5M_flow_v2_l40s.sh  | 2581564 | flow-mbpo-pwm/phase1_dflex_ant_flow_v2 | failed | OOM at 393GB (90.9% done, R~30) | logs/slurm/phase1/dflex_ant/train_5M_flow_v2_seed42_2581564.out | Priority 2 FAILED: Flow WM 5M OOM at epoch 18193/20000, exceeded 384GB limit |
| 2025-12-07 | 1     | dflex_ant | pwm_5M_baseline H16                 | 42   | scripts/phase1/submit_dflex_ant_5M_baseline_l40s.sh | 2581563 | flow-mbpo-pwm/phase1_dflex_ant_baseline | succeeded | **Eval: R=+16.53** (loss=-16.53), disc=-15.59, L=17.41, runtime 2h26m, 59GB peak | logs/slurm/phase1/dflex_ant/train_5M_baseline_seed42_2581563.out | **Priority 1 COMPLETED** MLP WM + MLP policy baseline (see ROOT_CAUSE_ANALYSIS.md) |
| 2025-12-07 | smoke | dflex_ant | pwm_5M_baseline_final H16, 10 epochs| 42   | scripts/smoke_test_baseline.sh                   | 2571819 | n/a                                    | succeeded   | R~32.8 (from job 2572354)     | logs/smoke/test_baseline_cpu_2571819.out                 | Smoke test passed |
| 2025-12-07 | 1.5   | dflex_ant | pwm_5M_flow_v2 H16 K4 regbase      | 42   | scripts/phase1p5/submit_dflex_ant_5M_flow_v2_H16_regbase_l40s.sh | 2581583 | flow-mbpo-pwm/phase1p5_dflex_ant_flow_v2_H16_base | cancelled | n/a | logs/slurm/phase1p5/dflex_ant/train_5M_flow_v2_H16_base_seed42_2581583.out | OLD Phase 1.5 cancelled, resubmitted as 2590204 |
| 2025-12-07 | 1.5   | dflex_ant | pwm_5M_flow_v2 H16 K4 regstrong    | 42   | scripts/phase1p5/submit_dflex_ant_5M_flow_v2_H16_regstrong_l40s.sh | 2581584 | flow-mbpo-pwm/phase1p5_dflex_ant_flow_v2_H16_strong | cancelled | n/a | logs/slurm/phase1p5/dflex_ant/train_5M_flow_v2_H16_str_seed42_2581584.out | OLD Phase 1.5 cancelled, resubmitted as 2590205 |
| 2025-12-07 | 1.5   | dflex_ant | pwm_5M_flow_v2 H8 K4 regbase       | 42   | scripts/phase1p5/submit_dflex_ant_5M_flow_v2_H8_regbase_l40s.sh | 2581585 | flow-mbpo-pwm/phase1p5_dflex_ant_flow_v2_H8_base | cancelled | n/a | logs/slurm/phase1p5/dflex_ant/train_5M_flow_v2_H8_base_seed42_2581585.out | OLD Phase 1.5 cancelled, resubmitted as 2590202 |
| 2025-12-07 | 1.5   | dflex_ant | pwm_5M_flow_v2 H8 K4 regstrong     | 42   | scripts/phase1p5/submit_dflex_ant_5M_flow_v2_H8_regstrong_l40s.sh | 2581586 | flow-mbpo-pwm/phase1p5_dflex_ant_flow_v2_H8_strong | cancelled | n/a | logs/slurm/phase1p5/dflex_ant/train_5M_flow_v2_H8_str_seed42_2581586.out | OLD Phase 1.5 cancelled, resubmitted as 2590203 |

---

## Wandb Run Mapping

**Completed Jobs → Wandb Runs:**

| JobID | Phase | Config | Wandb Run ID | Runtime | Rewards (max) | Status | Notes |
|-------|-------|--------|--------------|---------|---------------|--------|-------|
| 2581563 | 1 | MLP+MLP 5M baseline | run-20251207_172413-5wyczum4 | 8680s (2h25m) | 932.7 | Synced | Priority 1, eval R=+16.53 |
| 2581581 | 1 | MLP+MLP 48M baseline | run-20251207_172629-zpf68000 | 13384s (3h43m) | 305.3 | Synced | 48M baseline, eval R=+25.14 |
| 2583678 | 2 | MLP WM + Flow pol 5M | run-20251207_195007-gf51ndbl | 7038s (1h57m) | 97.7 | Synced | Priority 3, **best 5M result** R=+23.43 |
| 2583679 | 2 | Flow WM + Flow pol 5M | run-20251207_214823-kr1jdvtp | 14739s (4h06m) | 1133.2 | Synced | Priority 4, eval R=+16.77 |

**Wandb Project Organization Issue:**
- All completed runs show `project: N/A` in wandb-summary.json (project not logged to summary)
- Need to check actual wandb dashboard or run metadata to verify project assignment
- New submissions (Jobs 2590202-2590207) configured with `++wandb.project="flow-mbpo-pwm"` to ensure proper project

**Wandb Naming Issue:**
- Historical runs may have default/auto-generated names
- New submissions use descriptive naming: `{wm_type}WM_{policy_type}pol_{scale}_seed{seed}_H{horizon}_K{wm_k}`
- Examples: `mlpWM_flowpol_5M_seed42_H8_Kpol2`, `flowWM_mlppol_48M_seed42_H8_K4`

**Actions Taken:**
1. ✅ Mapped all completed jobs to their wandb run IDs by runtime correlation
2. ✅ Fixed wandb config in new submission scripts (phase1p5, phase2 48M jobs)
3. ⏳ TODO: Verify runs appear in correct wandb project dashboard
4. ⏳ TODO: Rename historical runs if needed (can be done via wandb API or dashboard)

