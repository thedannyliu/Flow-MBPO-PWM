# Flow-Matching 動力學與 Baseline PWM 的公平比較指南

**文件狀態**：操作手冊  
**日期**：2025-11-03  
**版本**：v1.0  
**目標**：提供完整、公平、可複現的實驗比較流程

---

## 1. 概述

### 1.1 實驗目標

本指南說明如何公平地比較兩個 PWM 變體：

1. **Baseline PWM**：使用 MLP 動力學模型（MSE loss）
2. **Flow PWM**：使用 Flow-Matching 動力學模型（rectified flow loss）

### 1.2 核心原則

- **參數量對齊**：兩者的 world model 參數總數必須在 ±2% 範圍內
- **其他組件不變**：Encoder、Reward、Actor、Critic 完全相同
- **數據與種子一致**：使用相同的離線數據集與隨機種子
- **透明記錄**：記錄所有關鍵指標，包括牆鐘時間與計算量

---

## 2. 環境設置

### 2.1 確認安裝

```bash
cd /storage/home/hcoda1/9/eliu354/r-agarg35-0/projects/Flow-MBPO-PWM/PWM
conda activate pwm  # 或你的環境名稱

# 確認新模組可以導入
python -c "from pwm.models.flow_world_model import FlowWorldModel; print('Flow model OK')"
python -c "from pwm.utils.integrators import heun_step, euler_step; print('Integrators OK')"
```

### 2.2 驗證參數量

建議先跑一個小腳本驗證參數量是否符合 ±2% 要求：

```python
# scripts/verify_param_parity.py
import torch
from hydra import compose, initialize
from hydra.utils import instantiate

with initialize(config_path="cfg"):
    # Baseline config
    cfg_base = compose(config_name="config", overrides=["alg=pwm_48M"])
    wm_base = instantiate(
        cfg_base.alg.world_model_config,
        observation_dim=100,  # 替換為實際值
        action_dim=20,
        latent_dim=768,
    )
    p_base = wm_base.total_params
    print(f"Baseline WM parameters: {p_base:,}")
    
    # Flow config
    cfg_flow = compose(config_name="config", overrides=["alg=pwm_48M_flow"])
    wm_flow = instantiate(
        cfg_flow.alg.world_model_config,
        observation_dim=100,
        action_dim=20,
        latent_dim=768,
    )
    p_flow = wm_flow.total_params
    print(f"Flow WM parameters: {p_flow:,}")
    
    diff_pct = abs(p_flow - p_base) / p_base * 100
    print(f"Difference: {diff_pct:.2f}%")
    
    if diff_pct > 2.0:
        print("WARNING: Difference exceeds 2%! Adjust units in pwm_48M_flow.yaml")
    else:
        print("✓ Parameter parity satisfied")
```

**執行**：
```bash
python scripts/verify_param_parity.py
```

如果差距超過 2%，需要調整 `pwm_48M_flow.yaml` 中的 `units` 欄位（例如 `[1788, 1788]` → `[1790, 1790]`）。

---

## 3. 實驗設定

### 3.1 任務與數據集

建議的測試集（按優先級排序）：

1. **單任務 dflex**（快速驗證）：
   - `dflex_ant`
   - `dflex_humanoid`
   - `dflex_hopper`

2. **多任務 MT30**（中等規模）
   - 使用 `scripts/cfg/config_mt30.yaml`

3. **多任務 MT80**（完整評估）
   - 使用 `scripts/cfg/config_mt80.yaml`

### 3.2 配置檔案對應

| 實驗類型 | Baseline 配置 | Flow 配置 |
|---------|--------------|-----------|
| 單任務 | `alg=pwm` 或 `alg=pwm_48M` | `alg=pwm_48M_flow` |
| MT30 | 修改 `config_mt30.yaml` 使用 `pwm_48M` | 修改 `config_mt30.yaml` 使用 `pwm_48M_flow` |
| MT80 | 修改 `config_mt80.yaml` 使用 `pwm_48M` | 修改 `config_mt80.yaml` 使用 `pwm_48M_flow` |

### 3.3 隨機種子

為了統計顯著性，每個設定跑 **3 個種子**：

```yaml
general:
  seed: 42  # 種子 1
  # seed: 123  # 種子 2
  # seed: 456  # 種子 3
```

---

## 4. 運行實驗

### 4.1 Baseline 實驗

**單任務範例（Ant）**：
```bash
python scripts/train_dflex.py \
    env=dflex_ant \
    alg=pwm_48M \
    general.seed=42 \
    general.run_wandb=True \
    wandb.project=flow-pwm-comparison \
    wandb.group=baseline-ant \
    general.logdir=logs/baseline_ant_seed42
```

**多任務範例（MT30）**：
```bash
python scripts/train_multitask.py \
    --config-name config_mt30 \
    alg=pwm_48M \
    general.seed=42 \
    general.run_wandb=True \
    wandb.project=flow-pwm-comparison \
    wandb.group=baseline-mt30 \
    general.logdir=logs/baseline_mt30_seed42
```

### 4.2 Flow 實驗

**單任務範例（Ant）**：
```bash
python scripts/train_dflex.py \
    env=dflex_ant \
    alg=pwm_48M_flow \
    general.seed=42 \
    general.run_wandb=True \
    wandb.project=flow-pwm-comparison \
    wandb.group=flow-ant \
    general.logdir=logs/flow_ant_seed42
```

**多任務範例（MT30）**：
```bash
python scripts/train_multitask.py \
    --config-name config_mt30 \
    alg=pwm_48M_flow \
    general.seed=42 \
    general.run_wandb=True \
    wandb.project=flow-pwm-comparison \
    wandb.group=flow-mt30 \
    general.logdir=logs/flow_mt30_seed42
```

### 4.3 超參數掃描（可選）

如果想測試不同的積分器或子步數：

```bash
# Heun K=2 (default)
python scripts/train_dflex.py env=dflex_ant alg=pwm_48M_flow \
    alg.flow_integrator=heun alg.flow_substeps=2

# Heun K=4
python scripts/train_dflex.py env=dflex_ant alg=pwm_48M_flow \
    alg.flow_integrator=heun alg.flow_substeps=4

# Euler K=2 (ablation)
python scripts/train_dflex.py env=dflex_ant alg=pwm_48M_flow \
    alg.flow_integrator=euler alg.flow_substeps=2
```

---

## 5. 監控指標

### 5.1 必須記錄的指標

訓練過程中應記錄：

| 指標類別 | 具體指標 | 用途 |
|---------|---------|------|
| **策略表現** | `rewards`, `policy_loss`, `episode_lengths` | 主要評估 |
| **World Model** | `wm_loss`, `dynamics_loss`, `reward_loss` | 模型品質 |
| **優化品質** | `actor_grad_norm`, `critic_grad_norm` | 訓練穩定性 |
| **計算成本** | `fps`, wall-clock time | 效率對比 |
| **梯度品質** | ESNR (可選，需額外實作) | 理論驗證 |

### 5.2 WandB 記錄

所有指標會自動記錄到 WandB（如果 `general.run_wandb=True`）。建議配置：

```yaml
wandb:
  project: flow-pwm-comparison
  entity: your_username
  group: baseline-ant  # 或 flow-ant, baseline-mt30, etc.
```

在 WandB 中可以：
- 按 `group` 聚合多個種子的結果
- 比較 baseline vs flow 的學習曲線
- 導出數據到 CSV 進行統計檢驗

---

## 6. 結果分析

### 6.1 主要比較維度

#### (A) 策略表現

- **最終回報**：訓練結束時的平均 episode reward
- **樣本效率**：達到特定回報閾值所需的訓練步數
- **穩定性**：跨種子的方差

**統計檢驗**：使用 Welch's t-test 或 bootstrapping 檢驗差異顯著性。

#### (B) 模型品質

- **Dynamics Loss**：Flow 的 rectified flow loss vs Baseline 的 MSE
- **Reward Loss**：兩者應該相近（因為 reward head 相同）

#### (C) 優化品質

- **梯度範數**：Flow 是否帶來更穩定的梯度？
- **ESNR**（可選）：Flow 是否提升訊噪比？

#### (D) 計算成本

- **訓練時間**：牆鐘時間（秒）
- **FPS**：每秒處理的環境步數
- **速度比**：`time_flow / time_baseline`

**註**：Flow 由於積分器會慢一些（Heun K=2 約為 2× velocity evaluations），但期望以更好的梯度品質換取更快的收斂。

### 6.2 可視化建議

1. **學習曲線**：
   ```python
   import matplotlib.pyplot as plt
   plt.plot(steps_baseline, rewards_baseline, label='Baseline', alpha=0.7)
   plt.plot(steps_flow, rewards_flow, label='Flow', alpha=0.7)
   plt.fill_between(...)  # 標準差
   plt.xlabel('Training Steps')
   plt.ylabel('Episode Reward')
   plt.legend()
   ```

2. **Dynamics Loss 對比**：
   ```python
   plt.subplot(1, 2, 1)
   plt.plot(baseline_dyn_loss, label='Baseline MSE')
   plt.subplot(1, 2, 2)
   plt.plot(flow_dyn_loss, label='Flow Matching')
   ```

3. **梯度範數分佈**：
   ```python
   plt.hist(baseline_grad_norms, alpha=0.5, label='Baseline')
   plt.hist(flow_grad_norms, alpha=0.5, label='Flow')
   ```

### 6.3 驗收標準（來自 plan）

根據 `flow-world-model-plan.md` Section 12：

> Flow variant meets or exceeds baseline on ≥50% tasks in reward under parameter parity

**最低要求**：
- Flow 在至少一半任務上的最終回報 ≥ Baseline
- 參數量差距 ≤ 2%
- 無嚴重的訓練不穩定（NaN、爆炸等）

**理想目標**：
- Flow 在多數任務上顯著優於 Baseline
- ESNR 指標顯示更高的訊噪比
- 牆鐘時間增加 ≤ 2× 但收斂步數減少

---

## 7. 常見問題與除錯

### 7.1 參數量不匹配

**症狀**：`verify_param_parity.py` 顯示差距 > 2%

**解決**：
1. 調整 `pwm_48M_flow.yaml` 中的 `units`
2. 公式：`new_width ≈ old_width - (param_diff / (2 * num_layers * latent_dim))`
3. 重新驗證直到差距 < 2%

### 7.2 Flow 訓練不穩定

**症狀**：NaN loss、梯度爆炸

**可能原因與解決**：
1. **積分器不穩定**：減少 `flow_substeps` 或換用 Euler
2. **學習率過高**：降低 `model_lr`（例如 3e-4 → 1e-4）
3. **梯度裁剪**：檢查 `wm_grad_norm` 設定（預設 20.0）

### 7.3 Flow 比 Baseline 慢很多

**預期行為**：Heun K=2 約慢 1.5-2×（因為每步 2 次 velocity evaluation）

**優化建議**：
1. 確認沒有不必要的同步點或 logging
2. 可嘗試 K=1 的 Euler（快但可能不穩定）
3. Profile 瓶頸：`python -m cProfile scripts/train_dflex.py ...`

### 7.4 缺少預訓練數據

**症狀**：`general.pretrain` 路徑找不到

**解決**：
1. 確認數據路徑正確（通常在 `results/` 或專門的 data 目錄）
2. 或者先用小規模環境測試（不預訓練）
3. 參考原 PWM README 的數據準備說明

---

## 8. 報告結果

### 8.1 表格範例

| 任務 | Baseline Reward | Flow Reward | Δ (%) | Baseline Time (min) | Flow Time (min) | 速度比 |
|------|----------------|-------------|-------|---------------------|-----------------|--------|
| Ant  | 5234 ± 120     | 5612 ± 95   | +7.2  | 45                  | 72              | 1.6×   |
| Humanoid | 4521 ± 200 | 4890 ± 150  | +8.2  | 120                 | 195             | 1.625× |
| ...  | ...            | ...         | ...   | ...                 | ...             | ...    |
| **平均** | - | - | **+X%** | - | - | **Y×** |

### 8.2 文字描述範例

> 在 3 個單任務與 MT30 多任務設定下，Flow-Matching 動力學相比 Baseline MLP 動力學，在保持 ±1.5% 參數量差距的前提下，平均提升最終回報 **X%**（p < 0.05, Welch's t-test）。雖然訓練時間增加約 **Y×**（主要來自 Heun 積分器的額外 velocity evaluation），但梯度品質指標（ESNR）顯著提升，表明 flow-based 動力學帶來更平滑的優化地形。

### 8.3 程式碼與配置發布

為了可複現性，建議提供：
1. 完整的 config 檔案（包括所有超參數）
2. 預訓練數據來源或生成腳本
3. 隨機種子列表
4. 環境依賴（`environment.yaml` 或 `requirements.txt`）

---

## 9. 進階：計算量對齊實驗（可選）

### 9.1 動機

Flow 由於積分器會做更多前向傳播，如果想「公平」比較，可以控制：

- **Baseline**：單步 MLP forward
- **Flow**：Heun K=2（約 2× forward）

則可以讓 Baseline 做 2× 的訓練迭代，使總計算量相近。

### 9.2 實作方式

```yaml
# baseline_compute_matched.yaml
alg:
  wm_iterations: 16  # 原本 8，加倍以匹配 Flow 的計算量
```

### 9.3 報告格式

| 對比方式 | Baseline | Flow | 結論 |
|---------|---------|------|------|
| **參數量對齊** | A | B | Flow 效果好 X% |
| **計算量對齊** | A' (2× iters) | B | Flow 仍好 Y% 或相當 |

---

## 10. Checklist

實驗前確認：

- [ ] 環境安裝完成，能導入 `FlowWorldModel`
- [ ] 參數量驗證通過（差距 < 2%）
- [ ] WandB 配置正確（project、entity、group）
- [ ] 隨機種子已設定（建議 3 個：42, 123, 456）
- [ ] 數據路徑正確（如需預訓練）
- [ ] Baseline 與 Flow 使用**相同**的數據、環境、其他超參數

實驗後確認：

- [ ] 所有 runs 正常完成（無崩潰、無 NaN）
- [ ] 記錄了完整的 metrics（reward, loss, time, grad norm）
- [ ] 導出 WandB 數據或 CSV 供統計分析
- [ ] 計算了平均值、標準差、顯著性檢驗
- [ ] 撰寫了實驗報告（表格 + 圖表 + 文字描述）

---

## 11. 參考文獻

- **PWM 原論文**：Policy Learning with Multi-Task World Models (2407.02466v3)
- **Flow Matching**：Flow Matching for Generative Modeling (ICLR 2023)
- **Rectified Flow**：Flow Straight and Fast: Learning to Generate and Transfer Data with Rectified Flow
- **本專案設計文檔**：`docs/flow-world-model-plan.md`

---

## 12. 聯絡與支援

如遇問題：

1. 檢查 `flow-world-model-plan.md` 的數學定義與設計決策
2. 查看 `PWM/src/pwm/models/flow_world_model.py` 與 `integrators.py` 的實作註釋
3. 確認 config 檔案語法正確（YAML 縮排、Hydra overrides）
4. 開 GitHub Issue 並附上：
   - 完整的錯誤訊息
   - 使用的 config 檔案
   - 環境資訊（Python、PyTorch 版本）

---

**最後更新**：2025-11-03  
**作者**：Flow-MBPO-PWM Team  
**版本**：1.0

祝實驗順利！🚀
