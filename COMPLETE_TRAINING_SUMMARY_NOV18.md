# 完整訓練總結 - Nov 18, 2025

## 🎯 已提交的所有訓練任務

### 第一批：單任務實驗（12小時 GPU）

| Job ID | 模型 | 配置 | 提交時間 | 狀態 |
|--------|------|------|----------|------|
| 2314140 | 5M Baseline | horizon=16 ✅ | 03:57 | ✅ 運行中 (11h 53m) |
| 2314141 | 48M Baseline (單任務) | task_dim=0 | 03:57 | ✅ 運行中 (11h 53m) |
| 2314142 | 48M Flow V1 (單任務) | substeps=2 | 03:57 | ✅ 運行中 (11h 53m) |
| 2314143 | 48M Flow V2 (單任務) ⭐ | substeps=4 | 03:57 | ✅ 運行中 (9h 29m) |
| 2314144 | 48M Flow V3 (單任務) | substeps=8 | 03:57 | ✅ 運行中 (9h 29m) |

### 第二批：多任務實驗（24小時 GPU）

| Job ID | 模型 | 配置 | 提交時間 | 狀態 |
|--------|------|------|----------|------|
| 2314382 | 48M Baseline (多任務) | task_dim=96 | 04:09 | ✅ 運行中 (11m) |
| 2314383 | 48M Flow V1 (多任務) | substeps=2, MT | 04:09 | ⏳ 等待資源 |
| 2314384 | 48M Flow V2 (多任務) ⭐ | substeps=4, MT | 04:09 | ⏳ 等待資源 |
| 2314385 | 48M Flow V3 (多任務) | substeps=8, MT | 04:09 | ⏳ 等待資源 |

**總計：9 個訓練任務**
- 1 個 5M baseline
- 4 個 48M 單任務（baseline + 3 flow）
- 4 個 48M 多任務（baseline + 3 flow）

## 📊 實驗矩陣

```
                 5M              48M Single-task      48M Multi-task
                                 (task_dim=0)         (task_dim=96)
Baseline        2314140          2314141              2314382
Flow V1 (sub=2)    -             2314142              2314383
Flow V2 (sub=4) ⭐  -             2314143 ⭐            2314384 ⭐
Flow V3 (sub=8)    -             2314144              2314385
```

## 🔑 關鍵修復和改進

### 1. ✅ Horizon=16 修復（最重要！）

**問題：** horizon=4 導致 baseline R~292  
**修復：** horizon=16（PWM paper 默認）  
**影響：** 預期 5M baseline 提升到 R~1200

### 2. ✅ eval() Bug 修復

**問題：** 使用 world model reward  
**修復：** 使用真實環境 reward  
**文件：** `PWM/src/pwm/algorithms/pwm.py`

### 3. ✅ 48M 配置創建

**單任務（4個）：**
- Baseline: task_dim=0, batch_size=1024
- Flow V1/V2/V3: substeps=2/4/8

**多任務（4個）：**
- Baseline: task_dim=96, batch_size=256（PWM paper）
- Flow V1/V2/V3: task_dim=96, batch_size=512

## 📁 創建的配置文件

### 單任務（Single-task）
```
PWM/scripts/cfg/alg/
├── pwm_5M_baseline_final.yaml (horizon=16 ✅)
├── pwm_48M_baseline_single_task.yaml
├── pwm_48M_flow_v1_substeps2.yaml
├── pwm_48M_flow_v2_substeps4.yaml ⭐
└── pwm_48M_flow_v3_substeps8.yaml
```

### 多任務（Multi-task）
```
PWM/scripts/cfg/alg/
├── pwm_48M_multitask_baseline.yaml
├── pwm_48M_multitask_flow_v1_substeps2.yaml
├── pwm_48M_multitask_flow_v2_substeps4.yaml ⭐
└── pwm_48M_multitask_flow_v3_substeps8.yaml
```

## 📜 創建的提交腳本

### 單任務
```
PWM/scripts/
├── submit_5M_baseline_l40s_final.sh
├── submit_48M_baseline_l40s.sh
├── submit_48M_flow_v1_l40s.sh
├── submit_48M_flow_v2_l40s.sh
├── submit_48M_flow_v3_l40s.sh
└── submit_all_nov18_experiments.sh (批量)
```

### 多任務
```
PWM/scripts/
├── submit_48M_multitask_baseline.sh
├── submit_48M_multitask_flow_v1.sh
├── submit_48M_multitask_flow_v2.sh
├── submit_48M_multitask_flow_v3.sh
└── submit_all_48M_multitask.sh (批量)
```

## 📚 創建的文檔

1. **EXPERIMENT_PLAN_NOV18.md** - 單任務實驗完整設計
2. **EXPERIMENT_48M_MULTITASK_NOV18.md** - 多任務實驗完整設計
3. **TRAINING_SUBMISSION_SUMMARY_NOV18.md** - 單任務提交總結
4. **CORRECTED_EVALUATION_RESULTS.md** - Nov 17 修正評估
5. **EVALUATION_SUMMARY_ZH.md** - 中文摘要
6. **TRAINING_VISUALIZATION_CORRECTED.md** - 視覺化結果
7. **CRITICAL_EVAL_BUG.md** - eval() bug 文檔
8. **QUICK_REF_NOV18.md** - 快速參考

## 🔄 Git 提交記錄

### PWM 子模組
1. `4e9390f` - 添加 48M 單任務配置和 eval() 修復
2. `d39ddb5` - 添加 48M 多任務配置

### 主倉庫
1. `9513772` - Nov 18 實驗：修復 5M + 添加 48M 配置
2. `0105e67` - 更新 PWM 子模組（單任務）
3. `d72e594` - 添加訓練提交總結文檔
4. `b4d94ae` - 添加多任務實驗文檔

## ⏰ 預期完成時間

### 單任務（12小時限制）
- **5M Baseline:** ~07:00-08:00 (Nov 18)
- **48M models:** ~15:00-16:00 (Nov 18)

### 多任務（24小時限制）
- **All MT models:** ~04:00 (Nov 19)

## 📊 監控命令

### 查看所有作業
```bash
squeue -u $USER
```

### 即時監控
```bash
# 單任務
tail -f PWM/logs/train_5M_baseline_l40s_2314140.out
tail -f PWM/logs/train_48M_flow_v2_l40s_2314143.out

# 多任務
tail -f PWM/logs/train_48M_mt_baseline_2314382.out
tail -f PWM/logs/train_48M_mt_flow_v2_2314384.out
```

### 提取 Peak R 值
```bash
# 單任務
for log in PWM/logs/train_*_2314{140..144}.out; do
    model=$(basename "$log" | sed 's/train_//' | sed 's/_l40s.*//')
    peak=$(grep -oP "R:\K[0-9.]+" "$log" 2>/dev/null | sort -rn | head -1)
    echo "$model: Peak R = ${peak:-N/A}"
done

# 多任務
for log in PWM/logs/train_48M_mt_*_2314{382..385}.out; do
    model=$(basename "$log" | sed 's/train_48M_mt_//' | sed 's/_2314.*//')
    peak=$(grep -oP "R:\K[0-9.]+" "$log" 2>/dev/null | sort -rn | head -1)
    echo "MT $model: Peak R = ${peak:-N/A}"
done
```

## 🎯 預期結果總結

### 單任務
| 模型 | 預期 Peak R | 信心度 | 依據 |
|------|-------------|--------|------|
| 5M Baseline | ~1200 | 高 | Nov 8 + horizon 修復 |
| 48M Baseline | 待觀察 | 中 | 新實驗 |
| 48M Flow V2 ⭐ | ~1200-1400 | 高 | 5M V2 最佳 |

### 多任務
| 模型 | 預期表現 | 信心度 | 依據 |
|------|----------|--------|------|
| 48M MT Baseline | 待觀察 | 中 | PWM paper 多任務 |
| 48M MT Flow V2 ⭐ | 最佳 | 高 | 單任務 V2 成功 |

## 🔍 關鍵配置對比

### Batch Size
- **5M Baseline:** 1024
- **48M Single-task Baseline:** 1024
- **48M Single-task Flow:** 1024
- **48M Multi-task Baseline:** 256（PWM paper）
- **48M Multi-task Flow:** 512（增加穩定性）

### Horizon
- **所有模型:** 16 ✅（修復前是 4）

### GPU Time Limit
- **5M & 48M Single-task:** 12 hours
- **48M Multi-task:** 24 hours ⏰

### Task Dimension
- **Single-task:** task_dim=0, multitask=False
- **Multi-task:** task_dim=96, multitask=True

## ✅ 完成清單

- [x] 修復 eval() bug（真實環境 reward）
- [x] 修復 5M baseline（horizon=16）
- [x] 創建 48M 單任務配置（4個）
- [x] 創建 48M 多任務配置（4個）
- [x] 創建所有提交腳本
- [x] 創建完整文檔
- [x] Git commit 所有更改
- [x] 提交所有訓練任務（9個）
- [ ] 監控訓練進度
- [ ] 收集結果
- [ ] 生成最終報告

## 📈 下一步行動

1. **監控訓練進度**（定期檢查）
2. **提取訓練 R 值**（完成後）
3. **重新評估 checkpoints**（用修復後的 eval）
4. **生成綜合分析報告**：
   - 單任務 vs 多任務對比
   - Flow 在不同設置的表現
   - 最佳配置建議

---

**狀態：** ✅ 所有 9 個訓練任務已提交  
**單任務預計完成：** Nov 18, 15:00-16:00  
**多任務預計完成：** Nov 19, 04:00  
**下一步：** 監控訓練，等待結果

*最後更新: 2025-11-18 04:10 EST*
