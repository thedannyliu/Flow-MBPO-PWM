# 48M Multi-task Experiments - Nov 18, 2025

## 實驗概述

在單任務（single-task）實驗的基礎上，新增 **48M 多任務（multi-task）** 配置。

### 目標

1. **測試 PWM 原始多任務 baseline**：task_dim=96, multitask=True
2. **測試 Flow matching 在多任務環境的表現**：3 種不同 substeps 配置
3. **對比單任務 vs 多任務**：分析 Flow 在不同設置下的效果

## 配置總覽

### 與單任務的主要差異

| 參數 | 單任務 | 多任務 |
|------|--------|--------|
| `task_dim` | 0 | 96 |
| `multitask` | False | True |
| `wm_batch_size` | 1024 | 256 (baseline) / 512 (flow) |
| `action_dims` | null | ??? (set by config) |
| `tasks` | null | ??? (set by config) |
| **訓練時間限制** | 12h | **24h** ⏰ |

## 配置詳情

### 1. 48M Multi-task Baseline

**配置文件：** `pwm_48M_multitask_baseline.yaml`

**關鍵設置：**
```yaml
# PWM paper 原始多任務配置
world_model: WorldModel (48M)
units: [1792, 1792]
encoder_units: [1792, 1792, 1792]
latent_dim: 768

# Multi-task setup
task_dim: 96                   # PWM paper default
multitask: True
wm_batch_size: 256            # PWM paper default for multi-task

# Training
lr_schedule: linear
max_epochs: 15_000
horizon: 16
```

**設計理念：**
- 完全遵循 PWM paper 的多任務配置
- 用於建立 baseline 性能
- 與單任務 baseline 對比

**訓練腳本：** `submit_48M_multitask_baseline.sh`

---

### 2. 48M Multi-task Flow V1 (Conservative)

**配置文件：** `pwm_48M_multitask_flow_v1_substeps2.yaml`

**關鍵設置：**
```yaml
world_model: FlowWorldModel (48M)
task_dim: 96
multitask: True
wm_batch_size: 512            # 增加到 512（Flow 需要更穩定的梯度）
max_epochs: 20_000            # Flow 需要更多 iterations

# Flow-specific
use_flow_dynamics: true
flow_integrator: heun         # 2階，穩定
flow_substeps: 2              # 保守配置
```

**設計理念：**
- 保守配置：substeps=2，計算成本最低
- Batch size 512：比 baseline 的 256 大，提供更穩定的梯度
- 對應單任務 V1

**訓練腳本：** `submit_48M_multitask_flow_v1.sh`

---

### 3. 48M Multi-task Flow V2 (Recommended) ⭐

**配置文件：** `pwm_48M_multitask_flow_v2_substeps4.yaml`

**關鍵設置：**
```yaml
world_model: FlowWorldModel (48M)
task_dim: 96
multitask: True
wm_batch_size: 512
max_epochs: 20_000

# Flow-specific (RECOMMENDED)
use_flow_dynamics: true
flow_integrator: heun         # 2階，最穩定
flow_substeps: 4              # 推薦：精度與穩定性最佳平衡
```

**設計理念：**
- 基於單任務 V2 的成功經驗（peak R=1197，最穩定）
- substeps=4：5M 和單任務實驗證明是 sweet spot
- 預期在多任務環境也有最佳表現

**訓練腳本：** `submit_48M_multitask_flow_v2.sh`

---

### 4. 48M Multi-task Flow V3 (High-Fidelity)

**配置文件：** `pwm_48M_multitask_flow_v3_substeps8.yaml`

**關鍵設置：**
```yaml
world_model: FlowWorldModel (48M)
task_dim: 96
multitask: True
wm_batch_size: 512
max_epochs: 20_000

# Flow-specific
use_flow_dynamics: true
flow_integrator: euler        # 1階（單任務顯示不穩定）
flow_substeps: 8              # 高 substeps
```

**設計理念：**
- 高精度配置：substeps=8
- 與單任務 V3 對應
- 用於對比實驗

**已知風險：**
- 單任務 V3 表現不穩定
- Euler + 高 substeps 可能有數值問題

**訓練腳本：** `submit_48M_multitask_flow_v3.sh`

---

## 配置對比表

| 模型 | Task Dim | Multitask | Batch Size | Substeps | Integrator | Max Epochs | GPU Time |
|------|----------|-----------|------------|----------|------------|------------|----------|
| **MT Baseline** | 96 | True | 256 | - | - | 15k | 24h |
| **MT Flow V1** | 96 | True | 512 | 2 | heun | 20k | 24h |
| **MT Flow V2** ⭐ | 96 | True | 512 | 4 | heun | 20k | 24h |
| **MT Flow V3** | 96 | True | 512 | 8 | euler | 20k | 24h |

## 提交訓練

### 批量提交所有多任務實驗

```bash
cd /storage/home/hcoda1/9/eliu354/r-agarg35-0/projects/Flow-MBPO-PWM/PWM/scripts
./submit_all_48M_multitask.sh
```

### 個別提交

```bash
# 48M Multi-task Baseline
sbatch submit_48M_multitask_baseline.sh

# 48M Multi-task Flow variants
sbatch submit_48M_multitask_flow_v1.sh
sbatch submit_48M_multitask_flow_v2.sh  # RECOMMENDED
sbatch submit_48M_multitask_flow_v3.sh
```

## 監控訓練

```bash
# 查看作業狀態
squeue -u $USER

# 即時監控
tail -f PWM/logs/train_48M_mt_baseline_<job_id>.out
tail -f PWM/logs/train_48M_mt_flow_v2_<job_id>.out  # RECOMMENDED

# 提取 peak R 值
for log in PWM/logs/train_48M_mt_*_<job_id>.out; do
    model=$(basename "$log" | sed 's/train_48M_mt_//' | sed 's/_.*/...//')
    peak=$(grep -oP "R:\K[0-9.]+" "$log" | sort -rn | head -1)
    echo "$model: Peak R = $peak"
done
```

## 硬體配置

- **GPU:** 1× L40S (48GB VRAM)
- **CPU:** 4 cores
- **Memory:** 128GB RAM
- **Time limit:** **24 hours** ⏰（比單任務多 12 小時）
- **Account:** `gts-agarg35-ideas_l40s`

## 預期結果

### 基於單任務經驗的預測

| 模型 | 預期表現 | 穩定性 | 信心度 | 依據 |
|------|----------|--------|--------|------|
| MT Baseline | 待觀察 | ⚠️ 中 | 中 | PWM paper 多任務結果 |
| MT Flow V1 | 略優於 baseline | ✅ 高 | 中 | 單任務 V1 穩定 |
| MT Flow V2 ⭐ | 最佳 | ✅ 高 | 高 | 單任務 V2 最佳 |
| MT Flow V3 | 高但不穩定 | ❌ 低 | 低 | 單任務 V3 不穩定 |

### 預期完成時間

所有任務預計在 **24 小時內**完成（Nov 19, 04:00 左右）

## 重要注意事項

### 1. Multi-task 配置

⚠️ **當前配置使用 task_dim=96, multitask=True，但仍使用單任務 dflex_ant 環境**

如果要使用真正的多任務（MT30/MT80）：
1. 需要下載 MT30/MT80 dataset
2. 使用 `train_multitask.py` 而非 `train_dflex.py`
3. 修改 config 為 `-cn config_mt30`

### 2. Batch Size 選擇

- **Baseline:** 256（PWM paper 多任務默認）
- **Flow:** 512（增加以提供更穩定的梯度）

### 3. 訓練時間

- **24 小時**：給多任務和 Flow 更多訓練時間
- 如果提早完成，SLURM 會自動結束

## 實驗設計對比

### 單任務 vs 多任務

| 特性 | 單任務 | 多任務 |
|------|--------|--------|
| **環境** | DFlex Ant | MT30 / MT80 (或單環境測試) |
| **Task dim** | 0 | 96 |
| **Batch size** | 1024 | 256 (baseline) / 512 (flow) |
| **目標** | 單一環境最佳性能 | 跨任務泛化能力 |
| **訓練時間** | 12h | 24h |
| **已知結果** | Flow V2 最佳 (R~1197) | 待觀察 |

## 評估計劃

訓練完成後：

1. **提取訓練 R 值**：從 logs 提取真實環境 reward
2. **對比單任務 vs 多任務**：
   - Baseline 性能差異
   - Flow 改進幅度差異
   - 穩定性差異
3. **分析 Flow 在多任務的表現**：
   - 是否仍然是 V2 最佳？
   - 多任務是否需要不同的 substeps？
4. **生成綜合報告**：
   - 單任務結果
   - 多任務結果
   - 對比分析
   - 最終建議

## 文件結構

```
PWM/scripts/cfg/alg/
├── # 單任務配置
├── pwm_48M_baseline_single_task.yaml
├── pwm_48M_flow_v1_substeps2.yaml
├── pwm_48M_flow_v2_substeps4.yaml
├── pwm_48M_flow_v3_substeps8.yaml
│
├── # 多任務配置 (新增)
├── pwm_48M_multitask_baseline.yaml       ✨
├── pwm_48M_multitask_flow_v1_substeps2.yaml ✨
├── pwm_48M_multitask_flow_v2_substeps4.yaml ✨
└── pwm_48M_multitask_flow_v3_substeps8.yaml ✨

PWM/scripts/
├── # 單任務提交腳本
├── submit_48M_baseline_l40s.sh
├── submit_48M_flow_v1_l40s.sh
├── submit_48M_flow_v2_l40s.sh
├── submit_48M_flow_v3_l40s.sh
│
├── # 多任務提交腳本 (新增)
├── submit_48M_multitask_baseline.sh      ✨
├── submit_48M_multitask_flow_v1.sh       ✨
├── submit_48M_multitask_flow_v2.sh       ✨
├── submit_48M_multitask_flow_v3.sh       ✨
└── submit_all_48M_multitask.sh           ✨ (批量提交)
```

## 實驗矩陣

```
                    Single-task              Multi-task
                    (task_dim=0)             (task_dim=96)
                    ─────────────            ─────────────
Baseline            ✅ Running (2314141)     🆕 New
Flow V1 (sub=2)     ✅ Running (2314142)     🆕 New
Flow V2 (sub=4) ⭐   ✅ Running (2314143)     🆕 New
Flow V3 (sub=8)     ✅ Running (2314144)     🆕 New

Total: 4 single-task + 4 multi-task = 8 experiments
```

---

*創建日期: 2025-11-18*  
*實驗類型: 48M Multi-task (Baseline + Flow V1/V2/V3)*  
*訓練時間: 24 hours per job*  
*目標: 驗證 Flow matching 在多任務環境的效果*
