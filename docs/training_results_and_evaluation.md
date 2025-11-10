# 訓練結果總結與 Evaluation 指南

## 📊 訓練結果總結

### 訓練狀態
- ✅ **所有 4 個訓練都已完成 15000 iterations**
- ✅ **訓練數據、checkpoints 都已完整保存**
- ✅ **WandB 記錄完整**
- ❌ **最後一步 (visualization) 因 Path TypeError 失敗**（已修復，未來不會再發生）

**重要：** 訓練本身 100% 完成，只是最後保存 visualization 時出錯。所有訓練數據和權重都完整無缺。

---

## 📁 訓練結果位置

| 訓練配置 | Job ID | 權重目錄 | Best Checkpoint | Size | WandB |
|---------|--------|----------|-----------------|------|-------|
| **pwm_5M baseline** | 2170920 | `outputs/2025-11-08/23-48-46/logs/pwm_5M_dflex_ant_seed42/` | `best_policy.pt` | 31M | [連結](https://wandb.ai/danny010324/pwm-flow-matching/runs/y8zafx5v) |
| **pwm_48M baseline** | 2170922 | `outputs/2025-11-09/02-50-34/logs/pwm_48M_dflex_ant_seed42/` | `best_policy.pt` | 232M | [連結](https://wandb.ai/danny010324/pwm-flow-matching/runs/6t5d7im9) |
| **pwm_5M flow** | 2170924 | `outputs/2025-11-09/06-18-53/logs/pwm_5M_flow_dflex_ant_seed42/` | `best_policy.pt` | 33M | [連結](https://wandb.ai/danny010324/pwm-flow-matching/runs/gqosfzbb) |
| **pwm_48M flow** | 2170925 | `outputs/2025-11-09/10-49-03/logs/pwm_48M_flow_dflex_ant_seed42/` | `best_policy.pt` | 232M | [連結](https://wandb.ai/danny010324/pwm-flow-matching/runs/g74e8fz7) |

### 訓練性能

| 配置 | FPS | 訓練時間 | 最終 Reward |
|-----|-----|---------|------------|
| pwm_5M baseline | 6078 | 3h 01m | ~27 |
| pwm_48M baseline | 5194 | 3h 28m | ~20 |
| pwm_5M flow | 3921 | 4h 30m | ~17 |
| pwm_48M flow | 3035 | 5h 51m | ~22 |

**觀察：**
- Baseline 比 Flow 快 ~50% (FPS)
- Flow 訓練時間較長（因為 dynamics 更複雜）
- 需要 evaluation 確認最終性能差異

---

## 🔍 如何查看訓練過程

### 方法 1：WandB Dashboard（推薦）

訪問對應的 WandB run 連結，可以看到：
- **訓練曲線**：FPS, rewards, losses, actor_std, etc.
- **Hyperparameters**：所有配置參數
- **系統資源**：GPU 使用率、記憶體使用
- **比較功能**：可以並排比較多個 runs

**快速連結：**
- 所有 runs：https://wandb.ai/danny010324/pwm-flow-matching
- 選擇 "Runs" tab，可以看到 4 個訓練的完整記錄

### 方法 2：查看 Log 文件

```bash
# 查看訓練過程
tail -1000 logs/slurm/pwm_5M_dflex_ant_seed42_2170920.out | grep "^\["

# 查看最終統計
tail -50 logs/slurm/pwm_5M_dflex_ant_seed42_2170920.out

# 搜尋特定資訊
grep "FPS" logs/slurm/pwm_5M_dflex_ant_seed42_2170920.out | tail -20
```

### 方法 3：查看 Checkpoint 目錄

```bash
# 列出所有 checkpoints
ls -lh outputs/2025-11-08/23-48-46/logs/pwm_5M_dflex_ant_seed42/

# 查看 checkpoint 內容
python -c "
import torch
ckpt = torch.load('outputs/2025-11-08/23-48-46/logs/pwm_5M_dflex_ant_seed42/best_policy.pt')
print('Keys:', ckpt.keys())
print('Iter count:', ckpt['iter_count'])
print('Step count:', ckpt['step_count'])
print('Best policy loss:', ckpt['best_policy_loss'])
"
```

---

## 🎯 如何進行 Evaluation

### 快速開始

```bash
cd /storage/home/.../PWM

# 評估 5M 模型
./scripts/run_evaluation.sh 5M dflex_ant

# 評估 48M 模型
./scripts/run_evaluation.sh 48M dflex_ant

# 自訂 episode 數量（預設 100）
./scripts/run_evaluation.sh 5M dflex_ant 200
```

### 輸出結果

評估完成後會產生：
```
evaluation_results/5M_dflex_ant_20251109_HHMMSS/
├── comparison.csv         # 數值比較表
├── comparison.png         # 視覺化圖表
└── evaluation.log        # 詳細 log
```

### 預期輸出範例

```
================================================================================
EVALUATION RESULTS
================================================================================
      Policy                Mean Reward        Mean Length  Success Rate
    Baseline         27.45 ± 3.21              982.3 ± 18.7      95.0%
        Flow         29.12 ± 2.87              991.5 ± 15.2      97.0%
================================================================================

================================================================================
FLOW IMPROVEMENT: +6.08%
================================================================================
```

---

## 📈 進階評估選項

### 1. 評估單一 checkpoint

```bash
python scripts/evaluate_policy.py \
    --checkpoint outputs/.../best_policy.pt \
    --env dflex_ant \
    --num-episodes 100
```

### 2. 比較 baseline vs flow

```bash
python scripts/evaluate_policy.py \
    --baseline outputs/2025-11-08/23-48-46/logs/pwm_5M_dflex_ant_seed42/best_policy.pt \
    --flow outputs/2025-11-09/06-18-53/logs/pwm_5M_flow_dflex_ant_seed42/best_policy.pt \
    --env dflex_ant \
    --num-episodes 100 \
    --output evaluation_results/5M_comparison
```

### 3. 使用不同環境

```bash
# Ant
./scripts/run_evaluation.sh 5M dflex_ant

# Humanoid (如果有訓練)
./scripts/run_evaluation.sh 5M dflex_humanoid

# Hopper (如果有訓練)
./scripts/run_evaluation.sh 5M dflex_hopper
```

### 4. 視覺化評估（需要 render 支援）

```bash
python scripts/evaluate_policy.py \
    --checkpoint outputs/.../best_policy.pt \
    --env dflex_ant \
    --num-episodes 10 \
    --render  # 需要 display 支援
```

---

## 📊 評估指標說明

### Mean Reward
- 所有 episodes 的平均總 reward
- 越高越好
- 標準差顯示穩定性（越低越穩定）

### Mean Length
- Episodes 平均長度
- 對於 dflex_ant，最大長度 1000
- 接近 1000 表示 policy 能長時間維持平衡

### Success Rate
- Episodes 成功率（task-specific）
- 對於 locomotion tasks，通常是 reward > threshold

### 比較基準
- **Improvement** = (Flow - Baseline) / |Baseline| × 100%
- 正值表示 Flow 更好
- 負值表示 Baseline 更好

---

## 🔧 故障排除

### 問題 1：找不到 checkpoint

**錯誤：**
```
Warning: Baseline checkpoint not found
```

**解決：**
```bash
# 查看所有可用 checkpoints
find outputs -name "best_policy.pt" -type f

# 手動指定正確路徑
python scripts/evaluate_policy.py \
    --checkpoint <正確的路徑>
```

### 問題 2：CUDA out of memory

**解決：**
```bash
# 減少 batch size 或使用 CPU
python scripts/evaluate_policy.py \
    --checkpoint ... \
    --device cpu
```

### 問題 3：Environment 錯誤

**解決：**
```bash
# 確認環境名稱正確
ls scripts/cfg/env/
# 應該看到: dflex_ant.yaml, dflex_humanoid.yaml, etc.
```

---

## 📝 下一步建議

### 1. ✅ 立即執行（推薦）

```bash
# 評估 5M 模型（較快）
./scripts/run_evaluation.sh 5M dflex_ant 100

# 評估 48M 模型
./scripts/run_evaluation.sh 48M dflex_ant 100
```

### 2. 分析結果

查看：
- Mean reward 提升多少？
- Variance 是否減少？（更穩定）
- Success rate 是否提高？
- Episode length 是否增加？

### 3. 視覺化比較

在 WandB 上比較：
- Training curves (smooth vs spiky)
- Final performance
- Sample efficiency (達到相同性能需要多少 steps)

### 4. 撰寫報告

記錄：
- Flow dynamics 是否帶來改進？
- 改進幅度有多大？
- 訓練效率如何（時間 vs 性能）？
- 是否值得增加的計算成本？

---

## 💡 重要觀察

從訓練 log 看到的現象：

1. **FPS 差異**：
   - Baseline: 5000-6000 FPS
   - Flow: 3000-4000 FPS
   - Flow 慢 ~40%（因為 dynamics 更複雜）

2. **訓練穩定性**：
   - 需要查看 WandB 曲線
   - 比較 variance 和收斂速度

3. **最終性能**：
   - 需要 evaluation 確認
   - Training reward 不等於 test performance

**結論：需要 evaluation 來確認 Flow 是否真的帶來性能提升！**

---

## 參考文件

- **訓練配置**：`scripts/cfg/alg/pwm_*.yaml`
- **環境配置**：`scripts/cfg/env/dflex_*.yaml`
- **Bug 修復記錄**：`docs/bug_fixes_summary_zh.md`
- **Checkpoint 策略**：`docs/bug_fixes_checkpoint_strategy.md`
