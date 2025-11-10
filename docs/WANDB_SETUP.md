# WandB 設置指南

## 📊 為什麼需要 WandB？

WandB (Weights & Biases) 提供：
- 🔍 **實時監控**: 訓練過程中實時查看 metrics
- 📈 **互動圖表**: 豐富的可視化和對比功能
- 🔄 **實驗追蹤**: 自動記錄所有超參數和結果
- 👥 **團隊協作**: 分享實驗結果和配置
- 📝 **完整記錄**: 永久保存實驗歷史

## 🚀 快速設置（5分鐘）

### 步驟 1: 獲取 API Key

1. 訪問 https://wandb.ai/authorize
2. 如果沒有帳號，先註冊（免費）
3. 複製您的 API key（類似：`a1b2c3d4e5f6...`）

### 步驟 2: 在 PACE 登錄節點上登入

```bash
# SSH 到 PACE Phoenix 登錄節點
ssh your_username@login-phoenix.pace.gatech.edu

# 激活 PWM 環境
conda activate pwm

# 登入 WandB（一次性設置）
wandb login

# 貼上您的 API key 並按 Enter
# 看到 "Successfully logged in" 即成功！
```

### 步驟 3: 驗證登入狀態

```bash
# 檢查是否已登入
wandb status

# 應該看到類似輸出：
# Logged in? True
# Current username: your_username
```

### 步驟 4: 提交作業

```bash
# 現在可以提交作業，WandB 會自動啟用
./scripts/submit_job.sh single pwm_48M_flow dflex_ant 42

# 作業開始後，在輸出中會看到 WandB 連結
# 點擊連結即可在瀏覽器中查看實時訓練進度！
```

## 📁 配置個性化設置

### 設置 WandB 團隊/組織

如果您屬於某個 WandB 團隊：

```bash
# 方法 1: 環境變數（推薦）
export WANDB_ENTITY="your-team-name"

# 方法 2: 在提交作業時指定
WANDB_ENTITY=your-team-name ./scripts/submit_job.sh single pwm_48M_flow dflex_ant 42
```

### 設置項目名稱

```bash
# 默認項目名稱：flow-pwm-comparison
# 自定義項目名稱：
export WANDB_PROJECT="my-custom-project"
```

### 完整配置範例

在 `~/.bashrc` 中添加（永久設置）：

```bash
# WandB 配置
export WANDB_ENTITY="your-team-name"    # 可選
export WANDB_PROJECT="flow-pwm-experiments"
```

然後：
```bash
source ~/.bashrc
```

## 🔍 使用 WandB 監控訓練

### 實時查看

1. 提交作業後，查看 SLURM 輸出：
   ```bash
   tail -f logs/slurm/pwm_48M_flow_dflex_ant_seed42_*.out
   ```

2. 找到 WandB URL（類似）：
   ```
   wandb: 🚀 View run at https://wandb.ai/your-username/flow-pwm-comparison/runs/xxx
   ```

3. 在瀏覽器中打開連結

### 主要功能

- **Overview**: 訓練摘要和關鍵指標
- **Charts**: 互動式圖表（reward, loss, gradient norms 等）
- **System**: GPU/CPU 使用率、記憶體消耗
- **Logs**: 完整的訓練日誌
- **Files**: 保存的模型檔案

### 比較多個實驗

1. 在 WandB 網頁界面中：
2. 點擊左側 "Runs" 標籤
3. 勾選要比較的實驗
4. 點擊 "Compare" 按鈕
5. 即可並排查看所有圖表！

## 🛠️ 進階選項

### 禁用 WandB（如需要）

```bash
# 方法 1: 環境變數
export USE_WANDB=false
./scripts/submit_job.sh single pwm_48M_flow dflex_ant 42

# 方法 2: 直接修改 SLURM 腳本
# 在 slurm_single_gpu.sh 中設置：
USE_WANDB=${USE_WANDB:-false}
```

### Offline 模式

如果網路不穩定，可以使用 offline 模式：

```bash
export WANDB_MODE=offline
./scripts/submit_job.sh single pwm_48M_flow dflex_ant 42

# 訓練完成後，在有網路的地方同步：
wandb sync logs/slurm/wandb/
```

### 設置 Tags 和 Notes

在訓練腳本中添加：

```python
wandb.config.update({
    'tags': ['baseline', 'ant', 'experiment-v1'],
    'notes': 'Testing baseline performance on Ant task'
})
```

## 📊 記錄的 Metrics

我們的實現自動記錄以下 metrics：

### 訓練指標
- `reward` / `policy_loss`: 策略表現
- `actor_loss`: Actor 網路損失
- `value_loss`: Critic 網路損失
- `wm_loss`: 世界模型總損失
  - `dynamics_loss`: 動力學預測損失
  - `reward_loss`: 獎勵預測損失

### 優化指標
- `actor_grad_norm`: Actor 梯度範數
- `critic_grad_norm`: Critic 梯度範數
- `wm_grad_norm`: 世界模型梯度範數
- `actor_lr`: 學習率（如有調度）

### 系統指標
- `fps`: 每秒樣本數
- `episode_length`: Episode 長度
- `rollout_len`: Rollout 長度

### 每 200 epochs 記錄
- 所有網路的梯度直方圖
- 參數分佈

## 🎯 最佳實踐

### 實驗命名

使用有意義的名稱：
```bash
export WANDB_NAME="ant_baseline_lr5e-4_seed42"
```

### 使用 Groups

將相關實驗組織在一起：
```bash
export WANDB_GROUP="baseline-comparison"
```

### Tags

使用 tags 方便過濾：
```python
wandb.config.update({
    'tags': ['baseline', 'production', 'paper-v1']
})
```

### 定期檢查

建議：
- 訓練開始後 5 分鐘檢查一次，確保正常
- 每小時檢查一次進度
- 發現異常立即終止作業（`scancel <job_id>`）

## 🐛 故障排除

### 問題 1: "wandb: ERROR Not logged in"

**解決**:
```bash
conda activate pwm
wandb login
# 輸入 API key
```

### 問題 2: "wandb: ERROR API key not found"

**解決**:
```bash
# 重新登入
wandb logout
wandb login YOUR_API_KEY
```

### 問題 3: WandB 連線超時

**解決**:
```bash
# 使用 offline 模式
export WANDB_MODE=offline
# 或增加超時時間
export WANDB_INIT_TIMEOUT=300
```

### 問題 4: 看不到 GPU metrics

WandB 自動記錄 GPU metrics，如果看不到：
1. 確認 `nvidia-smi` 可用
2. 檢查 WandB agent 版本：`pip install --upgrade wandb`

### 問題 5: 實驗沒有出現在 dashboard

1. 檢查網路連線
2. 確認 project 和 entity 名稱正確
3. 查看 SLURM 輸出中的錯誤訊息

## 📖 相關資源

- **WandB 官方文檔**: https://docs.wandb.ai
- **快速入門**: https://docs.wandb.ai/quickstart
- **Python API**: https://docs.wandb.ai/ref/python
- **最佳實踐**: https://docs.wandb.ai/guides/track/best-practices

## ✅ 檢查清單

設置前：
- [ ] 註冊 WandB 帳號
- [ ] 獲取 API key

設置步驟：
- [ ] SSH 到 PACE 登錄節點
- [ ] 激活 PWM 環境
- [ ] 運行 `wandb login`
- [ ] 輸入 API key
- [ ] 驗證：`wandb status`

使用步驟：
- [ ] 提交作業
- [ ] 在輸出中找到 WandB URL
- [ ] 在瀏覽器中打開監控頁面
- [ ] 享受實時訓練監控！🎉

---

**提示**: 完成登入後，所有後續的作業都會自動使用 WandB，無需重複設置！
