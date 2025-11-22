# Quick Training Status Reference

**Last Updated**: 2024-11-18 Afternoon

## At a Glance

| Experiment | Job ID | Status | Progress | Peak/Current R | Notes |
|------------|--------|--------|----------|----------------|-------|
| **5M Baseline** | 2314140 | ✓ Complete | 15000/15000 | **1222.15** | Best 5M model |
| **48M Baseline** | 2314141 | ✓ Complete | 15000/15000 | **1253.89** ⭐ | **Best overall** |
| 5M Flow V1 | 2309575 | ✓ Complete | 20000 | 1132.89 | Lost to baseline |
| 5M Flow V2 | 2309576 | ✓ Complete | 20000/20000 | 1197.40 | Best 5M Flow |
| 5M Flow V3 | 2309577 | ✓ Complete | 20000/20000 | 1137.49 | Lost to baseline |
| **48M Flow V1** | 2322456 | 🔄 Running | 54.5% | R=23.49 | ~6h left |
| **48M Flow V2** | 2322458 | 🔄 Running | 27.1% | R=17.36 | ~11h left, **prev peak ~1209** |
| **48M Flow V3** | 2322459 | 🔄 Running | 18.0% | R=102.60 | ~14h left, prev peak ~1182 |
| 48M MT Baseline | 2322520 | ⏸️ Pending | 0% | - | In queue |
| 48M MT Flow V1 | 2322521 | ⏸️ Pending | 0% | - | In queue |
| 48M MT Flow V2 | 2322522 | ⏸️ Pending | 0% | - | In queue |
| 48M MT Flow V3 | 2322523 | ⏸️ Pending | 0% | - | In queue |

---

## Key Questions

### Q1: Did Flow beat Baseline?
- **5M**: ❌ NO (Best Flow: 1197.40 < Baseline: 1222.15)
- **48M**: ⏳ WAITING (Flow V2 showing promise with partial peak ~1209)

### Q2: Which Flow substeps value is best?
- **5M**: substeps=4 (V2: 1197.40)
- **48M**: TBD (V2 with substeps=4 looks most promising)

### Q3: Does larger model help Flow?
- **5M Flow** underperformed 5M Baseline by ~25 R
- **48M Flow** TBD - need final results

### Q4: When will we know the final answer?
- **48M Flow V1**: ~6 hours (fastest, 54% done)
- **48M Flow V2**: ~11 hours (most important)
- **48M Flow V3**: ~14 hours (slowest, 18% done)
- **Full comparison**: November 19 morning

---

## Monitoring Commands

### Check Queue
```bash
squeue -u $USER --format="%.18i %.9P %.50j %.8u %.8T %.10M %.9l %.6D %R"
```

### Check Running Progress
```bash
for job in 2322456 2322458 2322459; do
  log=$(ls PWM/logs/*_${job}.out 2>/dev/null)
  if [ -f "$log" ]; then
    echo "=== Job $job ==="
    tail -5 "$log" | grep -E "\[" | tail -1
  fi
done
```

### Extract Peak R
```bash
for log in PWM/logs/train_48M_flow_*_2322*.out; do
  echo "=== $(basename $log) ==="
  echo "Peak R: $(grep -oP 'R:\K[-\d.]+' "$log" | sort -g | tail -1)"
  echo "Current: $(grep -oP 'R:\K[-\d.]+' "$log" | tail -1)"
done
```

---

## Success Criteria

### Critical Goal
**48M Flow V2** must achieve Peak R > **1253.89** to beat baseline

### Stretch Goals
- Any 48M Flow variant beats 48M Baseline
- 48M Flow shows consistent advantage over baseline
- MT Flow variants beat MT Baseline

### Minimum Acceptable
- All jobs complete without OOM errors ✓
- Collect complete training data for analysis
- Understand why Flow succeeds or fails

---

## File Locations

### Logs
```
PWM/logs/train_*_2322*.out  # Running jobs
PWM/logs/train_*_2314*.out  # Completed jobs
```

### Configs
```
PWM/scripts/cfg/alg/pwm_*_baseline*.yaml
PWM/scripts/cfg/alg/pwm_*_flow_v*.yaml
```

### Results Summary
```
TRAINING_STATUS_NOV18.md  # Full English report
訓練狀態總結_20241118.md  # Full Chinese summary
TRAINING_STATUS_QUICK.md  # This quick reference
```

---

## What to Do Next

1. **Wait**: Let jobs run (check progress every few hours)
2. **Monitor**: Watch for errors in .err logs
3. **When Flow V1 Completes (~6h)**: Extract peak R and update
4. **When Flow V2 Completes (~11h)**: Compare to baseline ⭐
5. **When All Complete (~14h)**: Generate final comparison report
6. **If Flow Wins**: Celebrate and run evaluation 🎉
7. **If Flow Loses**: Analyze why and decide next steps 🤔

---

**Next Check**: 2024-11-18 Evening (after 6pm)  
**Final Results**: 2024-11-19 Morning
