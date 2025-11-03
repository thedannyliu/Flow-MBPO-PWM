# Flow-Matching PWM: Quick Start Guide

快速開始指南，5 分鐘內跑起來！

---

## 1. 環境確認

```bash
# 進入專案目錄
cd /storage/home/hcoda1/9/eliu354/r-agarg35-0/projects/Flow-MBPO-PWM/PWM

# 確認環境
conda activate pwm  # 或你的環境名稱

# 測試導入
python -c "from pwm.models.flow_world_model import FlowWorldModel; print('✓ OK')"
```

---

## 2. 驗證參數量

```bash
# 快速檢查參數量是否在 ±2% 範圍
python scripts/verify_param_parity.py --obs-dim 100 --act-dim 20
```

應該看到：`✓ PASS: Difference X.X% <= 2.0%`

---

## 3. 跑第一個實驗（單任務）

### Baseline

```bash
python scripts/train_dflex.py \
    env=dflex_ant \
    alg=pwm_48M \
    general.seed=42 \
    general.logdir=logs/baseline_ant_seed42
```

### Flow

```bash
python scripts/train_dflex.py \
    env=dflex_ant \
    alg=pwm_48M_flow \
    general.seed=42 \
    general.logdir=logs/flow_ant_seed42
```

---

## 4. 監控訓練

### 方式 1: 終端輸出

訓練時會看到：

```
[1/15000]  R:123.4  T:50.0  H:16.0  S:1024  FPS:320  pi_loss:-123.4  v_loss:0.56  wm_loss:1.23
```

關鍵指標：
- `R` = Reward（越高越好）
- `wm_loss` = World model loss（應該下降）
- `FPS` = 訓練速度

### 方式 2: WandB（推薦）

在配置中啟用：

```bash
python scripts/train_dflex.py \
    env=dflex_ant \
    alg=pwm_48M_flow \
    general.run_wandb=True \
    wandb.project=my-flow-experiments \
    wandb.entity=your_username
```

然後去 https://wandb.ai 查看即時曲線。

---

## 5. 比較結果

訓練完成後：

```python
# 讀取 logs
import pandas as pd
import matplotlib.pyplot as plt

# 假設訓練記錄到 logs/
baseline = pd.read_csv("logs/baseline_ant_seed42/metrics.csv")
flow = pd.read_csv("logs/flow_ant_seed42/metrics.csv")

# 畫學習曲線
plt.plot(baseline['step'], baseline['rewards'], label='Baseline', alpha=0.7)
plt.plot(flow['step'], flow['rewards'], label='Flow', alpha=0.7)
plt.xlabel('Training Steps')
plt.ylabel('Episode Reward')
plt.legend()
plt.savefig('comparison.png')
```

---

## 6. 常見問題

### Q: 跑不起來？

**A**: 檢查：
1. 環境是否安裝正確（`pip list | grep torch`）
2. PWM 路徑是否在 `PYTHONPATH`（`echo $PYTHONPATH`）
3. 數據路徑是否正確（如需預訓練資料）

### Q: NaN loss？

**A**: 降低學習率：

```bash
python scripts/train_dflex.py \
    env=dflex_ant \
    alg=pwm_48M_flow \
    alg.model_lr=1e-4  # 從 3e-4 降低
```

### Q: Flow 比 Baseline 慢很多？

**A**: 正常！Heun K=2 約慢 1.5-2×（因為多評估一次 velocity）。  
如果想要快一點，可以用 Euler K=1：

```bash
python scripts/train_dflex.py \
    env=dflex_ant \
    alg=pwm_48M_flow \
    alg.flow_integrator=euler \
    alg.flow_substeps=1
```

但 Euler 可能不如 Heun 穩定。

### Q: 如何調整超參數？

**A**: 常見的可調項：

- `alg.flow_integrator`: `heun` 或 `euler`
- `alg.flow_substeps`: `1`, `2`, `4`（越大越精確但越慢）
- `alg.flow_tau_sampling`: `uniform` 或 `midpoint`
- `alg.model_lr`: 學習率（建議範圍 `1e-4` ~ `3e-4`）

---

## 7. 下一步

完成單任務驗證後：

1. **跑多種子**：`seed=42,123,456` 各跑一遍
2. **多任務**：試試 MT30 或 MT80
3. **完整分析**：參考 `flow-dynamics-comparison-guide.md`

---

## 8. 需要幫助？

1. 查看 `docs/FLOW_IMPLEMENTATION_SUMMARY.md` 瞭解實作細節
2. 查看 `docs/flow-dynamics-comparison-guide.md` 瞭解實驗協議
3. 查看 `docs/flow-world-model-plan.md` 瞭解設計決策

或者開一個 GitHub Issue！

---

**祝實驗順利！** 🎉
