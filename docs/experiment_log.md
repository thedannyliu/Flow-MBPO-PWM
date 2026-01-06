# Experiment Log

> **Purpose**: Persistent experiment registry for all training/evaluation jobs.
> **Fields**: Job ID, Config, Task, Seed, Runtime, Hardware, Status, Final Reward

---

## 🟢 Active Experiments

### Phase 8: WM Pretraining
**Method**: Pretrain WM From Scratch
**Purpose**: Create matched Flow WM and MLP WM checkpoints for fair 2×2 factorial comparison.

| Job ID | WM Type | Config | Iters | Hardware | Runtime | Status | Best Loss | Checkpoint |
|--------|---------|--------|-------|----------|---------|--------|-----------|------------|
| `4013702` | Flow WM | `pwm_48M_mt_flowwm` | 200k | H100 | ~8h36m | 🟢 RUNNING | 1.3040 | `outputs/2026-01-05/19-10-40/logs/flowwm_mt30_best.pt` |
| `4013703` | MLP WM | `pwm_48M_mt_baseline` | 200k | H100 | 2h28m | ✅ COMPLETED | 0.0009 | `outputs/2026-01-05/19-10-40/logs/mlpwm_mt30_best.pt` |

---

## ⏸️ Incomplete Runs (Can Resume)

### Phase 6: Flow 100k (TIMEOUT at 16h)
**Progress**: Reached ~78k/100k epochs (78%)
**Checkpoints exist**: Yes (`model_last.pt`)
**Need**: Additional 8h+ to complete

| Job ID | Config | Task | Seed | Hardware | Progress | Checkpoint |
|--------|--------|------|------|----------|----------|------------|
| `4012538_0` | FullFlow | reacher-easy | 42 | H200 | 78200/100k | `outputs/epoch_sweep/flow_100k/4012547/0_s42/logs/model_last.pt` |
| `4012538_2-8` | FullFlow | All | * | H200 | ~78k/100k | `outputs/epoch_sweep/flow_100k/*/logs/model_last.pt` |

### Phase 6: Flow 50k (TIMEOUT at 8h)
**Progress**: Reached ~39k/50k epochs (78%)
**Checkpoints exist**: Yes
**Need**: Additional 3h+ to complete

| Job ID | Config | Task | Seed | Hardware | Progress |
|--------|--------|------|------|----------|----------|
| `4012536_0-8` | FullFlow | All | * | H100 | 39k/50k |

### Phase 6: 150k (CUDA OOM)
**Status**: Cannot resume (failed at startup)
**Reason**: CUDA Out of Memory on H200
**Resolution**: Need to reduce `wm_batch_size` or use smaller model

---

## ✅ Completed Phases

### Phase 7: Flow Policy Fine-tuning (27 runs)
**Method**: Load Pretrained MLP WM + `finetune_wm=True`
- See `results/phase7_results.csv` for full details
- Best: baseline walker-stand 284.19

### Phase 6: Epoch Sweep (Baseline Complete, Flow Incomplete)
- **15k Baseline**: ✅ 9 runs completed
- **15k FullFlow**: ✅ 9 runs completed
- **50k Baseline**: ✅ 9 runs completed
- **50k FullFlow**: ⏸️ 9 runs TIMEOUT (can resume)
- **100k Baseline**: ✅ 9 runs completed
- **100k FullFlow**: ⏸️ 8 runs TIMEOUT (can resume)
- **150k**: ❌ OOM (cannot resume)

### Phase 5: Flow Tuning (18 runs)
- All completed, see `results/phase5_flow_tuning.csv`

### Phase 4: Full Flow 10k (9 runs)
- All completed

### Phase 3: Pretrained WM (18 runs)
- Best: reacher 983.50, walker 977.35, cheetah 134.97

---

## 📂 Failed Runs Summary
| Phase | Job IDs | Issue | Resolution |
|-------|---------|-------|------------|
| 6-100k Flow | 4012538 | TIMEOUT 16h | ⏸️ Resume with 10h+ |
| 6-50k Flow | 4012536 | TIMEOUT 8h | ⏸️ Resume with 5h+ |
| 6-150k | 4012555/56 | CUDA OOM | ❌ Need config change |
| 8 WM | 4012664/65/915/16 | Config error | ✅ Fixed → 4013702/03 |

---

## Next Steps (Priority Order)

1. **Wait for Phase 8 Flow WM** to complete (~8h more)
2. **Resume Flow 50k/100k** with longer time limits
3. **Run 2×2 factorial** with pretrained WMs
