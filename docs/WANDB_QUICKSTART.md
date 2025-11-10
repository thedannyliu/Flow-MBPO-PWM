# 🚀 WandB 快速設置（2分鐘）

## 第一次使用前（只需做一次）

```bash
# 1. SSH 到 PACE 登錄節點
ssh your_username@login-phoenix.pace.gatech.edu

# 2. 激活環境
conda activate pwm

# 3. 登入 WandB
wandb login
# 然後貼上您的 API key（從 https://wandb.ai/authorize 獲取）

# 4. 驗證登入
wandb status
# 看到 "Logged in? True" 即成功！
```

## 提交作業（正常使用）

```bash
# WandB 會自動啟用（因為您已經登入）
./scripts/submit_job.sh single pwm_48M_flow dflex_ant 42

# 查看輸出找到 WandB URL
tail -f logs/slurm/pwm_48M_flow_dflex_ant_seed42_*.out
# 找到類似這樣的行：
# wandb: 🚀 View run at https://wandb.ai/...
# 在瀏覽器中打開這個連結！
```

## 如果遇到問題

### 錯誤: "wandb: ERROR Not logged in"

```bash
# 重新登入
conda activate pwm
wandb login YOUR_API_KEY
```

### 想暫時禁用 WandB

```bash
# 設置環境變數
export USE_WANDB=false
./scripts/submit_job.sh single pwm_48M_flow dflex_ant 42
```

## 完整指南

查看 `docs/WANDB_SETUP.md` 了解：
- 進階配置
- 團隊設置
- 故障排除
- 最佳實踐

---

**注意**: 登入一次後，所有後續作業都會自動使用 WandB！
