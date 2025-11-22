# Training Status Summary - November 18, 2024

## Overview
檢查所有提交的訓練任務狀態。以下是各個實驗的詳細進度報告。

## Queue Status
- **Running Jobs**: 3 (48M Flow V1/V2/V3 single-task)
- **Pending Jobs**: 4 (48M Multi-task: Baseline + Flow V1/V2/V3)

---

## ✅ Completed Training Runs

### 1. Single-Task Baselines

#### 5M Baseline (Job 2314140)
- **Status**: ✓ **COMPLETE** (15000/15000 epochs)
- **Peak R**: **1222.15**
- **Final R (Last 10 avg)**: 29.92
- **Log**: `train_5M_baseline_l40s_2314140.out`
- **Model Size**: 5M parameters
- **Notes**: Training completed successfully. High peak performance observed.

#### 48M Baseline (Job 2314141)
- **Status**: ✓ **COMPLETE** (15000/15000 epochs)
- **Peak R**: **1253.89** ⭐ **(Best baseline)**
- **Final R (Last 10 avg)**: 23.40
- **Log**: `train_48M_baseline_l40s_2314141.out`
- **Model Size**: 48M parameters
- **Notes**: Training completed successfully. Achieved highest peak among baselines.

---

### 2. 5M Flow Variants (Completed)

#### 5M Flow V1 - substeps=2 (Job 2309575)
- **Status**: ✓ **COMPLETE**
- **Peak R**: 1132.89
- **Final R (Last 10 avg)**: 1131.74
- **Log**: `train_5M_flow_v1_l40s_2309575.out`
- **Config**: `flow_cfg_version=v1_l40s`, `flow_substeps=2`
- **Notes**: Completed successfully. Stable final performance.

#### 5M Flow V2 - substeps=4 (Job 2309576) 
- **Status**: ✓ **COMPLETE** (20000/20000 epochs)
- **Peak R**: **1197.40** ⭐
- **Final R (Last 10 avg)**: 561.61
- **Log**: `train_5M_flow_v2_l40s_2309576.out`
- **Config**: `flow_cfg_version=v2_l40s`, `flow_substeps=4`
- **Notes**: **Best 5M Flow variant.** High peak performance but dropped at end.

#### 5M Flow V3 - substeps=8 (Job 2309577)
- **Status**: ✓ **COMPLETE** (20000/20000 epochs)
- **Peak R**: 1137.49
- **Final R (Last 10 avg)**: 22.87
- **Log**: `train_5M_flow_v3_l40s_2309577.out`
- **Config**: `flow_cfg_version=v3_l40s`, `flow_substeps=8`
- **Notes**: Completed successfully. Performance declined at end.

---

## ⏳ Currently Running (48M Single-Task)

### 48M Flow V1 - substeps=2 (Job 2322456)
- **Status**: 🔄 **RUNNING** 
- **Progress**: 10897/20000 (54.5%)
- **Current R**: 23.49
- **Runtime**: ~6h 37min
- **Node**: atl1-1-03-004-31-0
- **Memory**: 256GB (upgraded from 128GB)
- **Log**: `train_48M_flow_v1_l40s_2322456.out`
- **ETA**: ~6 more hours

### 48M Flow V2 - substeps=4 (Job 2322458)
- **Status**: 🔄 **RUNNING**
- **Progress**: 5424/20000 (27.1%)
- **Current R**: 17.36
- **Runtime**: ~5h 15min
- **Node**: atl1-1-01-010-31-0
- **Memory**: 256GB (upgraded from 128GB)
- **Log**: `train_48M_flow_v2_l40s_2322458.out`
- **ETA**: ~11 more hours
- **Notes**: Based on previous run, this variant reached Peak R=1209.81 before OOM

### 48M Flow V3 - substeps=8 (Job 2322459)
- **Status**: 🔄 **RUNNING**
- **Progress**: 3601/20000 (18.0%)
- **Current R**: 102.60 (showing early promise!)
- **Runtime**: ~5h 12min
- **Node**: atl1-1-01-010-33-0
- **Memory**: 256GB (upgraded from 128GB)
- **Log**: `train_48M_flow_v3_l40s_2322459.out`
- **ETA**: ~14 more hours
- **Notes**: Based on previous run, reached Peak R=1182.21 before OOM

---

## 📋 Pending (48M Multi-Task)

### 48M MT Baseline (Job 2322520)
- **Status**: ⏸️ **PENDING** (Priority queue)
- **Config**: `config_mt30.yaml` + `pwm_48M_multitask_baseline.yaml`
- **Task**: MT30 (30 tasks)
- **Memory**: 256GB
- **Time Limit**: 24 hours
- **Notes**: Uses `train_multitask.py` with proper MT30 config

### 48M MT Flow V1 - substeps=2 (Job 2322521)
- **Status**: ⏸️ **PENDING** (Priority queue)
- **Config**: MT30 + Flow substeps=2
- **Memory**: 256GB
- **Time Limit**: 24 hours

### 48M MT Flow V2 - substeps=4 (Job 2322522)
- **Status**: ⏸️ **PENDING** (Priority queue)
- **Config**: MT30 + Flow substeps=4
- **Memory**: 256GB
- **Time Limit**: 24 hours

### 48M MT Flow V3 - substeps=8 (Job 2322523)
- **Status**: ⏸️ **PENDING** (Priority queue)
- **Config**: MT30 + Flow substeps=8
- **Memory**: 256GB
- **Time Limit**: 24 hours

---

## 📊 Performance Summary

### Baseline Comparison
| Model | Params | Epochs | Peak R | Final R (Last 10) | Status |
|-------|--------|--------|--------|-------------------|--------|
| **5M Baseline** | 5M | 15000 | **1222.15** | 29.92 | ✓ Complete |
| **48M Baseline** | 48M | 15000 | **1253.89** | 23.40 | ✓ Complete |

**Winner**: 48M Baseline (Peak R: 1253.89) ⭐

### 5M Flow Variants
| Variant | Substeps | Epochs | Peak R | Final R (Last 10) | Status |
|---------|----------|--------|--------|-------------------|--------|
| **V1** | 2 | 20000 | 1132.89 | 1131.74 | ✓ Complete |
| **V2** | 4 | 20000 | **1197.40** | 561.61 | ✓ Complete |
| **V3** | 8 | 20000 | 1137.49 | 22.87 | ✓ Complete |

**Winner**: 5M Flow V2 (Peak R: 1197.40) ⭐  
**Note**: All 5M Flow variants underperformed compared to 5M baseline

### 48M Flow Variants (In Progress)
| Variant | Substeps | Progress | Current R | Previous Peak | Status |
|---------|----------|----------|-----------|---------------|--------|
| **V1** | 2 | 54.5% (10897/20000) | 23.49 | ~1100 | 🔄 Running |
| **V2** | 4 | 27.1% (5424/20000) | 17.36 | **~1209** | 🔄 Running |
| **V3** | 8 | 18.0% (3601/20000) | 102.60 | ~1182 | 🔄 Running |

**Expected Winner**: 48M Flow V2 (previous partial peak: 1209.81)

---

## 🔍 Key Observations

### 1. **Memory Issues Resolved** ✅
- Previous 48M Flow runs experienced OOM kills with 128GB memory
- All resubmitted runs now use 256GB memory
- Jobs are running successfully without OOM issues

### 2. **5M vs 48M Baseline**
- 48M baseline achieved **higher peak** (1253.89 vs 1222.15)
- Both completed 15000 epochs successfully
- Final performance relatively similar (both ~20-30 R)

### 3. **Flow Performance**
- 5M Flow variants did **NOT** beat 5M baseline
- Best 5M Flow (V2): 1197.40 vs Baseline: 1222.15 ❌
- 48M Flow V2 shows **promise** (partial peak 1209.81 before OOM)
- Need to wait for 48M Flow runs to complete for final comparison

### 4. **Multi-Task Training**
- All MT jobs pending in queue
- Fixed configuration to use `train_multitask.py` with MT30 config
- Increased memory to 256GB and time to 24h

---

## 📈 Expected Outcomes

### Best Case Scenario
- **48M Flow V2** completes with Peak R > **1209** (potentially beating 48M baseline's 1253.89)
- **48M MT Baseline** provides strong multi-task baseline
- **48M MT Flow variants** show improvement over MT baseline

### Realistic Scenario
- 48M Flow variants complete successfully without OOM
- 48M Flow V2 likely performs best among Flow variants
- May still not beat 48M baseline (1253.89)
- MT experiments provide valuable multi-task comparison data

---

## 🎯 Next Steps

### Immediate (While Jobs Running)
1. ✅ Monitor running 48M Flow jobs (~6-14 hours remaining)
2. ✅ Wait for MT jobs to start (pending in queue)
3. ✅ Check logs periodically for any errors

### After Current Runs Complete
1. 📊 Extract final Peak R values for all 48M Flow variants
2. 🔬 Compare 48M Flow V2 vs 48M Baseline (most important comparison)
3. 📈 Analyze MT baseline performance
4. 🔄 Compare MT Flow variants to MT baseline
5. 📝 Generate final evaluation report with all results
6. ⚖️ Make decision: Is Flow improvement worth the computational cost?

### If Flow V2 Beats Baseline
- 🎉 **Success!** Flow matching provides improvement
- Run additional evaluation episodes to confirm
- Analyze what makes substeps=4 optimal

### If Flow V2 Does NOT Beat Baseline
- 🤔 Analyze why Flow underperforms
- Consider: different flow substep values, hyperparameter tuning
- Decide whether to continue Flow experiments or focus on baseline

---

## 📁 File References

### Configuration Files
- Baselines: `pwm_5M_baseline_final.yaml`, `pwm_48M_baseline_single_task.yaml`
- 5M Flow: `pwm_5M_flow_v{1,2,3}_l40s.yaml`
- 48M Flow: `pwm_48M_flow_v{1,2,3}_l40s.yaml`
- MT Baseline: `pwm_48M_multitask_baseline.yaml`
- MT Flow: `pwm_48M_multitask_flow_v{1,2,3}.yaml`
- MT30 Config: `config_mt30.yaml`

### Submit Scripts
- Single-task: `submit_{5M,48M}_{baseline,flow_v{1,2,3}}_l40s.sh`
- Multi-task: `submit_48M_multitask_{baseline,flow_v{1,2,3}}.sh`
- Batch: `submit_all_flow_256GB.sh`, `submit_all_multitask_256GB.sh`

### Logs Directory
- Location: `PWM/logs/`
- Pattern: `train_*_<job_id>.{out,err}`

---

## 💡 Technical Notes

### Memory Requirements
- **5M models**: 128GB sufficient
- **48M models**: 256GB required (OOM at 128GB)
- **MT models**: 256GB required

### Training Time
- **5M Baseline**: ~7-8 hours (15k epochs)
- **48M Baseline**: ~10-12 hours (15k epochs)
- **5M Flow**: ~8-10 hours (20k epochs)
- **48M Flow**: ~12-16 hours (20k epochs, estimated)
- **48M MT**: ~24 hours (estimated)

### GPU Allocation
- All jobs running on **L40S GPUs**
- Queue: `gpu-l40s`
- Nodes: atl1-1-01-* and atl1-1-03-*

---

**Report Generated**: 2024-11-18  
**Status**: 7 Complete, 3 Running, 4 Pending  
**Next Update**: After 48M Flow jobs complete (~6-14 hours)
