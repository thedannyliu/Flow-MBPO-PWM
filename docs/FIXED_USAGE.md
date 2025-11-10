# 修復後的使用指南

## 🔧 問題修復

已修復兩個主要問題：

1. **WandB 交互式登錄問題**: 
   - SLURM 批次作業無法交互式輸入 WandB API key
   - **解決方案**: 在 SLURM 腳本中設置 `WANDB_MODE=disabled` 並使用 `general.run_wandb=False`

2. **參數順序說明**:
   - 明確了正確的命令參數順序

## ✅ 正確的使用方法

### 單 GPU 實驗

**正確的參數順序**: `./scripts/submit_job.sh single <algorithm> <task> <seed> [gpu_type]`

```bash
# ❌ 錯誤 - 參數順序錯誤
./scripts/submit_job.sh single pwm_5M_flow dflex_ant 42

# ✅ 正確 - algorithm 在前，task 在後
./scripts/submit_job.sh single pwm_5M dflex_ant 42
./scripts/submit_job.sh single pwm_5M_flow dflex_ant 42
./scripts/submit_job.sh single pwm_48M dflex_ant 42
./scripts/submit_job.sh single pwm_48M_flow dflex_ant 42

# 使用不同種子
./scripts/submit_job.sh single pwm_48M_flow dflex_ant 123
./scripts/submit_job.sh single pwm_48M_flow dflex_humanoid 42

# 指定 GPU 類型（可選）
./scripts/submit_job.sh single pwm_48M_flow dflex_ant 42 H200
./scripts/submit_job.sh single pwm_48M_flow dflex_ant 42 H100
```

### 多 GPU 並行實驗

```bash
# 4個不同種子（multi_seed 策略）
./scripts/submit_job.sh multi multi_seed dflex_ant 42

# 4個不同任務（multi_task 策略）
./scripts/submit_job.sh multi multi_task dflex_ant 42

# Baseline vs Flow 比較（baseline_vs_flow 策略）
./scripts/submit_job.sh multi baseline_vs_flow dflex_ant 42
```

## 📊 監控作業

```bash
# 查看作業狀態
squeue -u $USER

# 查看實時輸出（使用正確的日誌文件名）
tail -f logs/slurm/pwm_5M_dflex_ant_seed42_*.out

# 查看錯誤（如果有）
tail -f logs/slurm/pwm_5M_dflex_ant_seed42_*.err

# 查看所有正在運行的作業
watch -n 5 'squeue -u $USER'
```

## 🔍 檢查結果

訓練完成後，結果保存在以下位置：

```bash
# 日誌目錄結構
logs/
├── pwm_5M_dflex_ant_seed42/           # 訓練日誌和模型
│   ├── best_policy.pt
│   ├── final_policy.pt
│   ├── learning_curves.png           # 自動生成的圖表
│   ├── gradient_norms.png
│   └── ...
└── slurm/                             # SLURM 作業日誌
    ├── pwm_5M_dflex_ant_seed42_2143135.out
    ├── pwm_5M_dflex_ant_seed42_2143135.err
    └── ...
```

## 🎯 可用的配置

### 算法選項
- `pwm_5M` - 5M 參數基線（快速測試）
- `pwm_5M_flow` - 5M 參數 flow（快速測試）
- `pwm_48M` - 48M 參數基線（完整實驗）
- `pwm_48M_flow` - 48M 參數 flow（完整實驗）

### 任務選項
- `dflex_ant` - Ant 機器人
- `dflex_humanoid` - Humanoid 機器人
- `dflex_hopper` - Hopper 機器人
- `dflex_anymal` - Anymal 四足機器人
- `dflex_snu_humanoid` - SNU Humanoid

### GPU 類型選項
- `H100` - H100 GPU
- `H200` - H200 GPU（預設）
- `A100` - A100 GPU
- `L40S` - L40S GPU

## 📝 範例工作流程

### 快速測試（推薦先做）
```bash
# 1. 提交 5M 模型快速測試（約30分鐘）
./scripts/submit_job.sh single pwm_5M dflex_ant 42

# 2. 查看作業狀態
squeue -u $USER

# 3. 監控輸出
tail -f logs/slurm/pwm_5M_dflex_ant_seed42_*.out

# 4. 檢查結果
ls -lh logs/pwm_5M_dflex_ant_seed42/
```

### 完整實驗
```bash
# 1. 提交 baseline 實驗
./scripts/submit_job.sh single pwm_48M dflex_ant 42

# 2. 提交 flow 實驗
./scripts/submit_job.sh single pwm_48M_flow dflex_ant 42

# 3. 等待完成後比較結果
python scripts/compare_runs.py \
    --run-dirs \
        logs/pwm_48M_dflex_ant_seed42 \
        logs/pwm_48M_flow_dflex_ant_seed42 \
    --labels baseline flow \
    --output-dir analysis/baseline_vs_flow/
```

### 多種子統計分析
```bash
# 1. 提交 4 個不同種子的 flow 實驗
./scripts/submit_job.sh multi multi_seed dflex_ant 42

# 2. 等待完成後生成統計分析
python scripts/compare_runs.py \
    --task dflex_ant \
    --algorithm pwm_48M_flow \
    --seeds 42 123 456 789 \
    --output-dir analysis/flow_seeds/
```

## ⚠️ 注意事項

1. **WandB 已禁用**: 
   - 目前配置下，WandB 在批次作業中被禁用
   - 如需啟用，需要先在登錄節點運行 `wandb login YOUR_API_KEY`
   - 然後修改 SLURM 腳本中的 `WANDB_MODE=disabled` 為 `WANDB_MODE=online`

2. **參數順序很重要**:
   - 必須是 `algorithm task seed`，不是 `task algorithm seed`

3. **日誌目錄**:
   - 確保 `logs/slurm/` 目錄存在
   - 腳本會自動創建，但首次運行前最好手動創建

4. **GPU 可用性**:
   - 使用 `sinfo -p gpu-h200` 檢查 H200 GPU 可用性
   - 如果 H200 不可用，可以嘗試 H100 或其他 GPU

## 🐛 故障排除

### 問題：作業立即失敗
**檢查**:
```bash
# 查看錯誤日誌
cat logs/slurm/pwm_5M_dflex_ant_seed42_*.err

# 檢查輸出日誌
cat logs/slurm/pwm_5M_dflex_ant_seed42_*.out
```

### 問題：找不到日誌目錄
**原因**: 訓練失敗，沒有創建日誌目錄
**解決**: 檢查 SLURM 錯誤日誌找出訓練失敗的原因

### 問題：仍然遇到 WandB 登錄提示
**解決**:
```bash
# 方法1: 在登錄節點登錄 WandB
wandb login YOUR_API_KEY

# 方法2: 確認 SLURM 腳本中已設置
export WANDB_MODE=disabled
```

### 問題：參數平衡驗證失敗
**解決**:
```bash
# 激活環境
conda activate pwm

# 運行驗證腳本
python scripts/verify_param_parity.py

# 如果失敗，查看建議的 units 值並更新配置
```

## ✨ 後續步驟

1. ✅ 修復已完成 - WandB 已禁用
2. ✅ 參數順序已明確說明
3. 📋 運行快速測試驗證修復
4. 🚀 提交完整實驗

祝實驗順利！🎉
