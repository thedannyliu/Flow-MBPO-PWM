# November 18, 2025 Training Experiments

## 實驗目標

1. **重現 5M Baseline**：修復 horizon=4 問題，改為 horizon=16（PWM paper 預設值）
2. **建立 48M Baseline**：單任務 DFlex Ant，基於 PWM paper 48M 配置
3. **測試 48M Flow**：三種不同的 substeps 配置（2, 4, 8）

## 關鍵發現與修復

### Baseline 表現不佳的根本原因

**問題：** Nov 17 baseline 只達到 R~292，遠低於預期的 R~1200

**根本原因：** `horizon=4` 而非 `horizon=16`

- PWM paper 使用 `horizon=16`（在 world model rollout 中看 16 步）
- Nov 17 training script 錯誤地使用 `horizon=4`（只看 4 步）
- 短 horizon 嚴重削弱 long-term credit assignment
- 導致策略無法學習長期規劃

**修復：** 所有新訓練都使用 `horizon=16`

## 配置總覽

### 1. 5M Baseline (Fixed)

**配置文件：** `pwm_5M_baseline_final.yaml`

**關鍵設置：**
```yaml
world_model: TDMPC2 (5M parameters)
units: [512, 512]
encoder_units: [256]
latent_dim: 512
task_dim: 0                    # 單任務
wm_batch_size: 1024           
lr_schedule: linear
max_epochs: 15_000
horizon: 16                    # 修復！之前是 4
```

**預期結果：** R ~ 1200（基於 Nov 8 成功訓練）

**訓練腳本：** `submit_5M_baseline_l40s_final.sh`

---

### 2. 48M Baseline (Single-Task)

**配置文件：** `pwm_48M_baseline_single_task.yaml`

**關鍵設置：**
```yaml
world_model: WorldModel (48M parameters)
units: [1792, 1792]           # 48M 寬度
encoder_units: [1792, 1792, 1792]
latent_dim: 768               # 48M latent
task_dim: 0                   # 單任務（不同於 PWM paper 多任務）
wm_batch_size: 1024           # 從 768 增加到 1024 以穩定梯度
lr_schedule: linear
max_epochs: 15_000
horizon: 16
```

**與 PWM paper 差異：**
- PWM paper: 多任務 (task_dim=96, multitask=True)
- 我們的版本: 單任務 (task_dim=0, multitask=False)
- Batch size 增加: 256 → 1024（更穩定的梯度）

**預期結果：** 待觀察（48M 單任務是新實驗）

**訓練腳本：** `submit_48M_baseline_l40s.sh`

---

### 3. 48M Flow V1 (Conservative)

**配置文件：** `pwm_48M_flow_v1_substeps2.yaml`

**關鍵設置：**
```yaml
world_model: FlowWorldModel (48M parameters)
units: [1792, 1792]           # 與 baseline 相同
encoder_units: [1792, 1792, 1792]
latent_dim: 768
task_dim: 0
wm_batch_size: 1024
lr_schedule: linear
max_epochs: 20_000            # Flow 需要更多 iterations
horizon: 16

# Flow-specific
use_flow_dynamics: true
flow_integrator: heun         # 2階方法，穩定
flow_substeps: 2              # 保守：速度快但精度較低
flow_tau_sampling: uniform
```

**設計理念：**
- 保守配置：substeps=2 計算成本最低
- Heun integrator：比 Euler 穩定
- 與 5M V1 對應

**訓練腳本：** `submit_48M_flow_v1_l40s.sh`

---

### 4. 48M Flow V2 (Recommended) ⭐

**配置文件：** `pwm_48M_flow_v2_substeps4.yaml`

**關鍵設置：**
```yaml
world_model: FlowWorldModel (48M parameters)
units: [1792, 1792]
encoder_units: [1792, 1792, 1792]
latent_dim: 768
task_dim: 0
wm_batch_size: 1024
lr_schedule: linear
max_epochs: 20_000
horizon: 16

# Flow-specific (RECOMMENDED)
use_flow_dynamics: true
flow_integrator: heun         # 2階方法，最穩定
flow_substeps: 4              # 推薦：精度與穩定性的最佳平衡
flow_tau_sampling: uniform
```

**設計理念：**
- 基於 5M V2 的成功經驗（peak R=1197, 最穩定）
- substeps=4：5M 實驗中證明是 sweet spot
- Heun + substeps=4 組合最可靠

**預期結果：** 最佳性能與穩定性

**訓練腳本：** `submit_48M_flow_v2_l40s.sh`

---

### 5. 48M Flow V3 (High-Fidelity)

**配置文件：** `pwm_48M_flow_v3_substeps8.yaml`

**關鍵設置：**
```yaml
world_model: FlowWorldModel (48M parameters)
units: [1792, 1792]
encoder_units: [1792, 1792, 1792]
latent_dim: 768
task_dim: 0
wm_batch_size: 1024
lr_schedule: linear
max_epochs: 20_000
horizon: 16

# Flow-specific
use_flow_dynamics: true
flow_integrator: euler        # 1階方法（5M 顯示不穩定）
flow_substeps: 8              # 高 substeps：更多計算，可能有數值問題
flow_tau_sampling: uniform
```

**設計理念：**
- 高精度配置：substeps=8
- 與 5M V3 對應（但 5M V3 表現不穩定）
- 用於對比實驗

**已知風險：**
- 5M V3: 不穩定（peak 1137 → avg last 10: 978）
- Euler + 高 substeps 組合可能有數值問題

**訓練腳本：** `submit_48M_flow_v3_l40s.sh`

---

## 實驗設計對比表

| 模型 | World Model | Params | Latent | Substeps | Integrator | Batch Size | Max Epochs | Horizon |
|------|-------------|--------|--------|----------|------------|------------|------------|---------|
| **5M Baseline** | TDMPC2 | 5M | 512 | - | - | 1024 | 15k | **16** ✅ |
| **48M Baseline** | WorldModel | 48M | 768 | - | - | 1024 | 15k | 16 |
| **48M Flow V1** | FlowWM | 48M | 768 | 2 | heun | 1024 | 20k | 16 |
| **48M Flow V2** ⭐ | FlowWM | 48M | 768 | 4 | heun | 1024 | 20k | 16 |
| **48M Flow V3** | FlowWM | 48M | 768 | 8 | euler | 1024 | 20k | 16 |

## 硬體配置

所有實驗使用：
- **GPU:** 1× L40S (48GB VRAM)
- **CPU:** 4 cores
- **Memory:** 128GB RAM
- **Time limit:** 12 hours
- **Account:** `gts-agarg35-ideas_l40s`

## 提交訓練

### 方式 1: 批量提交所有實驗

```bash
cd /storage/home/hcoda1/9/eliu354/r-agarg35-0/projects/Flow-MBPO-PWM/PWM/scripts
./submit_all_nov18_experiments.sh
```

這會提交所有 5 個訓練任務。

### 方式 2: 個別提交

```bash
cd /storage/home/hcoda1/9/eliu354/r-agarg35-0/projects/Flow-MBPO-PWM/PWM/scripts

# 5M Baseline (horizon=16 fix)
sbatch submit_5M_baseline_l40s_final.sh

# 48M Baseline
sbatch submit_48M_baseline_l40s.sh

# 48M Flow variants
sbatch submit_48M_flow_v1_l40s.sh
sbatch submit_48M_flow_v2_l40s.sh  # RECOMMENDED
sbatch submit_48M_flow_v3_l40s.sh
```

## 監控訓練

### 查看 Job 狀態
```bash
squeue -u $USER
```

### 查看訓練日誌
```bash
# 即時監控
tail -f PWM/logs/train_5M_baseline_l40s_<job_id>.out
tail -f PWM/logs/train_48M_baseline_l40s_<job_id>.out
tail -f PWM/logs/train_48M_flow_v2_l40s_<job_id>.out

# 查看訓練進度（R values）
grep "R:" PWM/logs/train_*_<job_id>.out | tail -20
```

### WandB 監控
所有訓練會記錄到 WandB project: `flow-mbpo-pwm`

- 5M Baseline: group `5M_baseline_l40s_fixed`
- 48M Baseline: group `48M_baseline_single_task`
- 48M Flow: group `48M_flow_single_task`

## 預期結果

基於 5M 實驗的經驗：

| 模型 | 預期 Peak R | 預期穩定性 | 訓練時間 | 信心度 |
|------|-------------|------------|----------|--------|
| 5M Baseline | ~1200 | ✅ 高 | ~3-4h | 高（Nov 8 驗證）|
| 48M Baseline | 待觀察 | ⚠️ 中 | ~5-8h | 中（單任務新實驗）|
| 48M Flow V1 | ~1000-1200 | ✅ 高 | ~6-9h | 中 |
| 48M Flow V2 ⭐ | ~1200-1400 | ✅ 高 | ~6-10h | 高（5M V2 最佳）|
| 48M Flow V3 | ~1100-1300 | ❌ 低 | ~7-11h | 低（5M V3 不穩定）|

## 評估計劃

訓練完成後：

1. **提取訓練 R 值**：從 logs 提取真實環境 reward
2. **使用修復後的 eval()**：重新評估所有 checkpoints
3. **對比分析**：
   - 5M baseline vs 48M baseline
   - 48M baseline vs 48M Flow variants
   - 不同 substeps 的影響
4. **生成報告**：完整的實驗結果分析

## 文件結構

```
PWM/scripts/cfg/alg/
├── pwm_5M_baseline_final.yaml           # 5M baseline (horizon=16 ✅)
├── pwm_48M_baseline_single_task.yaml    # 48M baseline (new)
├── pwm_48M_flow_v1_substeps2.yaml       # 48M + Flow substeps=2
├── pwm_48M_flow_v2_substeps4.yaml       # 48M + Flow substeps=4 ⭐
└── pwm_48M_flow_v3_substeps8.yaml       # 48M + Flow substeps=8

PWM/scripts/
├── submit_5M_baseline_l40s_final.sh     # 5M baseline submit
├── submit_48M_baseline_l40s.sh          # 48M baseline submit
├── submit_48M_flow_v1_l40s.sh           # 48M Flow V1 submit
├── submit_48M_flow_v2_l40s.sh           # 48M Flow V2 submit (推薦)
├── submit_48M_flow_v3_l40s.sh           # 48M Flow V3 submit
└── submit_all_nov18_experiments.sh      # 批量提交所有實驗
```

## 重要注意事項

1. **Horizon=16 是關鍵**：這是導致之前 baseline 失敗的主要原因
2. **Linear LR schedule**：已驗證有效，不要改動
3. **Batch size 1024**：對 5M 和 48M 都是安全選擇
4. **Flow V2 最推薦**：基於 5M 實驗，substeps=4 + heun 是最佳組合
5. **48M 單任務是新實驗**：與 PWM paper 多任務不同，結果待觀察

---

*創建日期: 2025-11-18*  
*實驗設計者: Based on PWM paper + 5M experiment insights*  
*目標: 建立可靠的 baseline + 驗證 Flow matching 在 48M scale 的效果*
