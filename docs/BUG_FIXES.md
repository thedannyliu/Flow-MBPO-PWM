# 🐛 問題修復說明

## 已修復的問題

### 1. ❌ Hydra 配置錯誤: `wandb.name`
**問題**: SLURM 腳本試圖設置 `wandb.name`，但這個欄位不在配置結構中

**原因**: 
- `train_dflex.py` 中的 `create_wandb_run()` 函數已經自動生成實驗名稱
- WandB 配置檔案（`config.yaml`）中沒有 `name` 欄位

**修復**: 移除了 SLURM 腳本中的 `wandb.name=$RUN_NAME` 參數

### 2. ❌ WandB `notes` 欄位必填錯誤
**問題**: 原始代碼要求 `notes` 欄位必填，但配置中是空的

**修復**: 改為 `.get("notes", "")` 允許空值

### 3. ✅ WandB 登入檢查
**新增**: 自動檢測 WandB 登入狀態
- 已登入 → 啟用 WandB 記錄
- 未登入 → 顯示警告並繼續（不使用 WandB）

## 修改的文件

1. ✅ `slurm_single_gpu.sh` - 移除 `wandb.name` 參數
2. ✅ `slurm_multi_gpu.sh` - 移除 `wandb.name` 參數  
3. ✅ `train_dflex.py` - 修復 `notes` 欄位處理

## 🚀 現在可以正常使用了

### 測試命令

```bash
# 確保在正確的環境中
conda activate pwm

# 提交測試作業
./scripts/submit_job.sh single pwm_5M dflex_ant 42

# 監控輸出
tail -f logs/slurm/pwm_5M_dflex_ant_seed42_*.out
```

### 預期輸出

應該看到：
```
WandB: Already logged in ✓
==============================================
Experiment Configuration
==============================================
Task: dflex_ant
Algorithm: pwm_5M
Seed: 42
Run Name: dflex_ant_pwm_5M_seed42_20251107_HHMMSS
WandB Enabled: True
WandB Project: flow-pwm-comparison
==============================================

Starting training...
==============================================
wandb: Currently logged in as: your-username
wandb: 🚀 View run at https://wandb.ai/...
```

然後訓練應該正常進行！

## 🎯 完整測試流程

### 1. 登入 WandB（如果還沒登入）
```bash
conda activate pwm
wandb login
# 貼上 API key
```

### 2. 測試 5M 模型（快速）
```bash
./scripts/submit_job.sh single pwm_5M dflex_ant 42
```

### 3. 檢查作業狀態
```bash
squeue -u $USER
```

### 4. 監控輸出
```bash
# 查看實時日誌
tail -f logs/slurm/pwm_5M_dflex_ant_seed42_*.out

# 或查看訓練日誌
tail -f logs/slurm/training_*.log
```

### 5. 訓練完成後檢查結果
```bash
# 應該看到日誌目錄
ls -lh logs/pwm_5M_dflex_ant_seed42/

# 應該有以下文件：
# - best_policy.pt
# - final_policy.pt
# - 訓練圖表（如果可視化成功）
```

## 🔍 如果還有問題

### 查看完整錯誤
```bash
# 查看 SLURM 錯誤輸出
cat logs/slurm/pwm_5M_dflex_ant_seed42_*.err

# 查看標準輸出
cat logs/slurm/pwm_5M_dflex_ant_seed42_*.out

# 設置完整錯誤追蹤
HYDRA_FULL_ERROR=1 python scripts/train_dflex.py \
    env=dflex_ant \
    alg=pwm_5M \
    general.seed=42 \
    general.run_wandb=True
```

### 檢查配置
```bash
# 驗證配置正確性
python scripts/train_dflex.py --help

# 測試不使用 WandB
./scripts/submit_job.sh single pwm_5M dflex_ant 42
# 然後在 SLURM 腳本中設置：
export USE_WANDB=false
```

## ✅ 檢查清單

設置：
- [x] 修復 `wandb.name` 配置錯誤
- [x] 修復 `notes` 欄位處理
- [x] 添加 WandB 登入檢查
- [x] 測試命令正確性

使用前：
- [ ] 確認已登入 WandB：`wandb status`
- [ ] 激活正確環境：`conda activate pwm`
- [ ] 使用正確參數順序：`algorithm task seed`

測試：
- [ ] 提交測試作業
- [ ] 監控輸出
- [ ] 檢查日誌目錄是否創建
- [ ] 確認訓練正常運行

## 📝 注意事項

1. **WandB 實驗名稱**：由 `train_dflex.py` 自動生成，格式為 `{algorithm}_{env_name}`

2. **WandB Project**: 默認是 `flow-pwm-comparison`，可以通過環境變數修改：
   ```bash
   export WANDB_PROJECT="my-project"
   ```

3. **WandB Entity**: 如果為空，會使用您的個人帳號

4. **日誌目錄**: 格式為 `logs/{algorithm}_{task}_seed{seed}`

5. **可視化**: 訓練完成後會自動嘗試生成可視化，如果失敗不影響訓練結果

---

現在重新提交作業應該可以正常工作了！🎉
