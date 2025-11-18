# 快速參考 - Nov 18 訓練實驗

## 🚀 已提交的訓練

| Job ID | 模型 | 狀態 |
|--------|------|------|
| 2314140 | 5M Baseline (horizon=16 ✅) | ✅ 運行中 |
| 2314141 | 48M Baseline | ✅ 運行中 |
| 2314142 | 48M Flow V1 (substeps=2) | ✅ 運行中 |
| 2314143 | 48M Flow V2 (substeps=4) ⭐ | ⏳ 等待 |
| 2314144 | 48M Flow V3 (substeps=8) | ⏳ 等待 |

## 📊 快速監控命令

```bash
# 查看作業狀態
squeue -u $USER

# 即時監控（5M baseline）
tail -f PWM/logs/train_5M_baseline_l40s_2314140.out

# 即時監控（48M Flow V2，推薦）
tail -f PWM/logs/train_48M_flow_v2_l40s_2314143.out

# 查看所有訓練的最新 R 值
for log in PWM/logs/train_*_2314*.out; do
    echo "=== $(basename $log) ==="
    grep "R:" "$log" | tail -5
    echo
done

# 提取 peak R 值
for log in PWM/logs/train_*_2314*.out; do
    model=$(basename "$log" | sed 's/train_//' | sed 's/_l40s.*//')
    peak=$(grep -oP "R:\K[0-9.]+" "$log" | sort -rn | head -1)
    echo "$model: Peak R = $peak"
done
```

## 🔑 關鍵修復

### 1. Horizon=16（最重要！）
- **問題：** horizon=4 導致 baseline R~292
- **修復：** horizon=16（PWM paper 默認）
- **預期：** baseline R~1200

### 2. eval() Bug
- **問題：** 使用 world model reward
- **修復：** 使用真實環境 reward
- **文件：** PWM/src/pwm/algorithms/pwm.py

## 📁 重要文件

### 配置
- `PWM/scripts/cfg/alg/pwm_48M_baseline_single_task.yaml`
- `PWM/scripts/cfg/alg/pwm_48M_flow_v2_substeps4.yaml` ⭐

### 文檔
- `EXPERIMENT_PLAN_NOV18.md` - 完整實驗設計
- `TRAINING_SUBMISSION_SUMMARY_NOV18.md` - 詳細總結

### 日誌
- `PWM/logs/train_5M_baseline_l40s_2314140.out`
- `PWM/logs/train_48M_flow_v2_l40s_2314143.out`

## ⏰ 預期完成時間

- 5M Baseline: ~4 hours (07:00-08:00)
- 48M models: ~8-10 hours (12:00-14:00)

## 🎯 預期結果

| 模型 | 預期 Peak R | 信心度 |
|------|-------------|--------|
| 5M Baseline | ~1200 | 高 |
| 48M Baseline | 待觀察 | 中 |
| 48M Flow V2 ⭐ | ~1200-1400 | 高 |

## ✅ 下一步

1. 等待訓練完成
2. 提取所有 peak R 值
3. 重新評估 checkpoints（用修復後的 eval）
4. 生成最終報告

---
**更新時間:** 2025-11-18 03:57  
**詳細文檔:** TRAINING_SUBMISSION_SUMMARY_NOV18.md
