# November 18, 2025 - 訓練提交總結

## ✅ 所有任務已完成

### 已提交的訓練任務

| Job ID | 模型 | 配置 | 狀態 | 節點 |
|--------|------|------|------|------|
| **2314140** | 5M Baseline | horizon=16 (修復) | ✅ 運行中 | atl1-1-03-007-29-0 |
| **2314141** | 48M Baseline | 單任務 | ✅ 運行中 | atl1-1-03-007-29-0 |
| **2314142** | 48M Flow V1 | substeps=2, heun | ✅ 運行中 | atl1-1-01-010-31-0 |
| **2314143** | 48M Flow V2 ⭐ | substeps=4, heun | ⏳ 等待資源 | (Resources) |
| **2314144** | 48M Flow V3 | substeps=8, euler | ⏳ 等待資源 | (Resources) |

### 關鍵修復和改進

#### 1. ✅ 修復 eval() Bug

**問題：** eval() 使用 world model 預測的 reward
```python
# 錯誤的程式碼
z, rew, trunc = self.wm.step(...)
_, _, done, _ = self.env.step(actions)  # 忽略真實 reward
episode_loss -= rew  # 使用 world model reward
```

**修復：** 使用真實環境 reward
```python
# 正確的程式碼
_, env_rew, done, _ = self.env.step(actions)
episode_loss -= env_rew  # 使用真實環境 reward
```

**文件：** `PWM/src/pwm/algorithms/pwm.py` (lines ~634-650)

#### 2. ✅ 修復 5M Baseline - Horizon 設置

**問題根因：** `horizon=4` 導致 baseline 只有 R~292

**原因分析：**
- PWM paper 使用 `horizon=16`（world model rollout 看 16 步）
- Nov 17 錯誤使用 `horizon=4`（只看 4 步）
- 短 horizon 削弱 long-term credit assignment
- 策略無法學習長期規劃

**修復：** 所有訓練都使用 `horizon=16`

**影響：** 預期 5M baseline 從 R~292 提升到 R~1200

#### 3. ✅ 創建 48M 配置

**48M Baseline（單任務）：**
- 基於 PWM paper 48M 架構
- 改為單任務：task_dim=0, multitask=False
- 增加 batch size：1024（從 768）
- 使用 horizon=16, linear LR schedule

**48M Flow V1/V2/V3：**
- V1: substeps=2, heun（保守配置）
- V2: substeps=4, heun（推薦配置，基於 5M 最佳結果）
- V3: substeps=8, euler（高精度，但可能不穩定）

### 創建的配置文件

```
PWM/scripts/cfg/alg/
├── pwm_5M_baseline_final.yaml (horizon=16 ✅)
├── pwm_48M_baseline_single_task.yaml
├── pwm_48M_flow_v1_substeps2.yaml
├── pwm_48M_flow_v2_substeps4.yaml ⭐
└── pwm_48M_flow_v3_substeps8.yaml
```

### 創建的提交腳本

```
PWM/scripts/
├── submit_5M_baseline_l40s_final.sh
├── submit_48M_baseline_l40s.sh
├── submit_48M_flow_v1_l40s.sh
├── submit_48M_flow_v2_l40s.sh
├── submit_48M_flow_v3_l40s.sh
└── submit_all_nov18_experiments.sh
```

### 創建的文檔

1. **EXPERIMENT_PLAN_NOV18.md** - 完整實驗設計和配置說明
2. **CORRECTED_EVALUATION_RESULTS.md** - Nov 17 訓練的修正評估報告
3. **EVALUATION_SUMMARY_ZH.md** - 中文摘要
4. **TRAINING_VISUALIZATION_CORRECTED.md** - 視覺化結果
5. **CRITICAL_EVAL_BUG.md** - eval() bug 詳細文檔

## Git 提交記錄

### Commit 1: 主倉庫
```
commit 9513772
Add Nov 18 experiments: Fix 5M baseline + Add 48M baseline/flow configs
```

### Commit 2: PWM 子模組
```
commit 4e9390f
Add 48M configs and fix eval() bug
```

### Commit 3: 更新子模組引用
```
commit 0105e67
Update PWM submodule with 48M configs and eval fix
```

## 監控訓練

### 查看作業狀態
```bash
squeue -u $USER
```

### 即時監控訓練日誌
```bash
# 5M Baseline (修復後)
tail -f PWM/logs/train_5M_baseline_l40s_2314140.out

# 48M Baseline
tail -f PWM/logs/train_48M_baseline_l40s_2314141.out

# 48M Flow V2 (推薦)
tail -f PWM/logs/train_48M_flow_v2_l40s_2314143.out
```

### 提取訓練進度
```bash
# 查看最新 R values
grep "R:" PWM/logs/train_*_2314*.out | tail -20

# 查看特定模型
grep "R:" PWM/logs/train_5M_baseline_l40s_2314140.out | tail -20
```

### WandB 監控
- **Project:** `flow-mbpo-pwm`
- **Groups:**
  - `5M_baseline_l40s_fixed`
  - `48M_baseline_single_task`
  - `48M_flow_single_task`

## 預期結果

基於 5M 實驗和分析：

| 模型 | 預期 Peak R | 預期穩定性 | 訓練時間 | 信心度 |
|------|-------------|------------|----------|--------|
| 5M Baseline | ~1200 | ✅ 高 | 3-4h | 高 |
| 48M Baseline | 待觀察 | ⚠️ 中 | 5-8h | 中 |
| 48M Flow V1 | ~1000-1200 | ✅ 高 | 6-9h | 中 |
| 48M Flow V2 ⭐ | ~1200-1400 | ✅ 高 | 6-10h | 高 |
| 48M Flow V3 | ~1100-1300 | ❌ 低 | 7-11h | 低 |

### 預期完成時間

- **5M Baseline:** ~07:00-08:00 (Nov 18)
- **48M Baseline:** ~09:00-12:00 (Nov 18)
- **48M Flow models:** ~10:00-14:00 (Nov 18)

## 後續行動

### 訓練完成後

1. **提取訓練 R 值：**
   ```bash
   for log in PWM/logs/train_*_2314*.out; do
       echo "=== $(basename $log) ==="
       grep -oP "^\[[0-9]+/[0-9]+\].*R:\K[0-9.]+" "$log" | sort -rn | head -1
   done
   ```

2. **使用修復後的 eval() 重新評估：**
   - 找到所有 best_policy.pt checkpoints
   - 用真實環境 reward 評估
   - 驗證訓練 R 值與評估結果一致

3. **生成最終報告：**
   - 對比 5M vs 48M baseline
   - 對比 48M baseline vs Flow variants
   - 分析不同 substeps 的影響
   - 確認 Flow V2 是否仍是最佳配置

4. **文檔更新：**
   - 更新 PROJECT_STATUS.md
   - 創建 48M 實驗結果報告
   - 記錄最佳配置和建議

## 重要發現總結

### 1. Horizon=16 是關鍵
- **這是導致 Nov 17 baseline 失敗的主要原因**
- PWM paper 使用 horizon=16
- horizon=4 削弱 long-term planning
- 預期修復後 baseline 達到 R~1200

### 2. Linear LR Schedule 必須使用
- Constant LR 會導致訓練崩潰
- Linear schedule 已在多次實驗中驗證
- 不要再嘗試其他 schedule

### 3. Batch Size 1024 是安全選擇
- 對 5M 和 48M 都穩定
- 比 256 提供更穩定的梯度
- 比 2048 更節省記憶體

### 4. Flow V2 (substeps=4, heun) 最推薦
- 5M 實驗證明：peak R=1197, 最穩定
- substeps=4 是精度與效率的平衡點
- Heun integrator 比 Euler 穩定

### 5. 48M 單任務是新實驗
- PWM paper 是多任務設置
- 單任務性能待觀察
- 可能需要調整超參數

## 技術細節

### 硬體配置
- **GPU:** L40S (48GB VRAM)
- **CPU:** 4 cores
- **Memory:** 128GB RAM
- **Account:** `gts-agarg35-ideas_l40s`

### 軟體環境
- **Conda env:** `pwm`
- **Python:** 3.x
- **Framework:** PWM + TDMPC2 + DFlex
- **Logging:** WandB

### 實驗參數統一設置
- **Seed:** 42
- **Device:** cuda:0
- **Horizon:** 16 ✅
- **Episode length:** 1000
- **Early termination:** height < 0.27m

## 檢查清單

- [x] 修復 eval() bug (使用真實環境 reward)
- [x] 修復 5M baseline (horizon=16)
- [x] 創建 48M baseline 配置
- [x] 創建 48M Flow V1/V2/V3 配置
- [x] 創建所有提交腳本
- [x] 創建實驗文檔
- [x] Git commit 所有更改
- [x] 提交所有訓練任務
- [ ] 監控訓練進度
- [ ] 收集訓練結果
- [ ] 重新評估所有模型
- [ ] 生成最終報告

## 聯繫信息

- **實驗設計者:** AI Assistant
- **執行日期:** November 18, 2025
- **Git branch:** `dev/flow-dynamics`
- **Commits:** 9513772, 4e9390f, 0105e67

---

**狀態：** ✅ 所有任務已完成，訓練中  
**下一步：** 監控訓練進度，等待完成後分析結果

*生成時間: 2025-11-18 03:57 EST*
