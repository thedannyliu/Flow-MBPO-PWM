# 快速參考卡片 - PWM Flow-Matching 實驗

## ⚡ 第一次使用（必做）

```bash
# 1. 登入 WandB（只需做一次）
conda activate pwm
wandb login
# 貼上 API key（從 https://wandb.ai/authorize 獲取）

# 2. 驗證登入
wandb status  # 應該顯示 "Logged in? True"
```

**詳細步驟**: 查看 `docs/WANDB_QUICKSTART.md`

## 🚀 快速開始

### 正確命令格式
```bash
./scripts/submit_job.sh single <algorithm> <task> <seed> [gpu_type]
                               ^^^^^^^^^^  ^^^^^^  ^^^^
                               第1參數     第2參數  第3參數
```

### 常用命令
```bash
# 5M 快速測試（~30分鐘）
./scripts/submit_job.sh single pwm_5M dflex_ant 42
./scripts/submit_job.sh single pwm_5M_flow dflex_ant 42

# 48M 完整實驗（~數小時）
./scripts/submit_job.sh single pwm_48M dflex_ant 42
./scripts/submit_job.sh single pwm_48M_flow dflex_ant 42

# 不同種子
./scripts/submit_job.sh single pwm_48M_flow dflex_humanoid 123

# 指定 GPU
./scripts/submit_job.sh single pwm_48M_flow dflex_ant 42 H100
```

## 📊 監控命令

```bash
# 查看作業
squeue -u $USER

# 實時日誌
tail -f logs/slurm/*.out

# 查看特定作業
tail -f logs/slurm/pwm_48M_flow_dflex_ant_seed42_*.out

# 取消作業
scancel <JOB_ID>
```

## 📁 文件位置

```
logs/
├── pwm_48M_dflex_ant_seed42/      # 訓練結果
│   ├── best_policy.pt              # 最佳模型
│   ├── final_policy.pt             # 最終模型
│   └── *.png                       # 自動生成圖表
└── slurm/                          # SLURM 日誌
    ├── pwm_48M_dflex_ant_seed42_*.out  # 標準輸出
    └── pwm_48M_dflex_ant_seed42_*.err  # 錯誤輸出
```

## 🎯 參數選項

| 類別 | 選項 | 說明 |
|------|------|------|
| **算法** | `pwm_5M` | 5M 基線（快速） |
| | `pwm_5M_flow` | 5M flow（快速） |
| | `pwm_48M` | 48M 基線（完整） |
| | `pwm_48M_flow` | 48M flow（完整） |
| **任務** | `dflex_ant` | Ant 機器人 |
| | `dflex_humanoid` | Humanoid |
| | `dflex_hopper` | Hopper |
| | `dflex_anymal` | Anymal |
| **GPU** | `H200` | H200 (預設) |
| | `H100` | H100 |
| | `A100` | A100 |
| | `L40S` | L40S |

## ⚡ 多 GPU 並行

```bash
# 4個種子並行
./scripts/submit_job.sh multi multi_seed dflex_ant 42

# 4個任務並行
./scripts/submit_job.sh multi multi_task dflex_ant 42

# Baseline vs Flow
./scripts/submit_job.sh multi baseline_vs_flow dflex_ant 42
```

## 🔧 故障排除

| 問題 | 解決方案 |
|------|----------|
| WandB 登錄提示 | ✅ 已修復 - WandB 已禁用 |
| 參數順序錯誤 | 使用 `algorithm task seed` 順序 |
| 作業失敗 | 查看 `logs/slurm/*.err` |
| 找不到日誌 | 訓練失敗，檢查錯誤日誌 |

## 📖 完整文檔

- **使用指南**: `docs/FIXED_USAGE.md`
- **完整教程**: `docs/flow-dynamics-comparison-guide.md`
- **集群指南**: `docs/PACE_USAGE_GUIDE.md`
- **快速入門**: `docs/QUICKSTART.md`

## ✅ 檢查清單

- [ ] 激活環境: `conda activate pwm`
- [ ] 創建日誌目錄: `mkdir -p logs/slurm`
- [ ] 檢查參數順序: `algorithm task seed`
- [ ] 提交作業: `./scripts/submit_job.sh ...`
- [ ] 監控進度: `squeue -u $USER`
- [ ] 查看結果: `ls logs/*/`

---
**注意**: 正確的參數順序是 `algorithm task seed`，不是 `task algorithm seed`！
