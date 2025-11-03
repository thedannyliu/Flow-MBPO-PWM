# PACE Phoenix Cluster 使用指南

本指南說明如何在 Georgia Tech PACE Phoenix cluster 上運行 PWM Flow-Matching 實驗。

---

## 1. 環境設置

### 1.1 首次設置

```bash
# SSH 登入
ssh your_gt_username@login-phoenix.pace.gatech.edu

# 進入專案目錄
cd /storage/home/hcoda1/9/eliu354/r-agarg35-0/projects/Flow-MBPO-PWM/PWM

# 加載模組
module load anaconda3
module load cuda/12.1

# 創建/激活 conda 環境
conda activate pwm

# 安裝依賴（首次）
pip install tqdm wandb seaborn
```

### 1.2 WandB 設置

```bash
# 登入 WandB（首次）
wandb login

# 或者使用 API key
export WANDB_API_KEY="your_api_key_here"
```

---

## 2. 提交任務

### 2.1 使用輔助腳本（推薦）

腳本已經幫你處理好所有參數：

```bash
# 給腳本執行權限（首次）
chmod +x scripts/submit_job.sh

# 單 GPU 任務
./scripts/submit_job.sh single pwm_48M dflex_ant 42
./scripts/submit_job.sh single pwm_48M_flow dflex_humanoid 123

# 多 GPU 任務 - 4 個種子
./scripts/submit_job.sh multi multi_seed dflex_ant 42

# 多 GPU 任務 - 4 個不同任務
./scripts/submit_job.sh multi multi_task dflex_ant

# 多 GPU 任務 - Baseline vs Flow 對比
./scripts/submit_job.sh multi baseline_vs_flow dflex_ant 42
```

### 2.2 直接使用 sbatch

如果你想更細緻地控制參數：

```bash
# 單 GPU
sbatch \
    --export=ALL,TASK=dflex_ant,ALGORITHM=pwm_48M,SEED=42 \
    scripts/slurm_single_gpu.sh

# 多 GPU
sbatch \
    --export=ALL,STRATEGY=multi_seed,TASK=dflex_ant,SEED=42 \
    scripts/slurm_multi_gpu.sh
```

---

## 3. 監控任務

### 3.1 查看隊列狀態

```bash
# 查看你的所有任務
squeue -u $USER

# 詳細信息
squeue -u $USER -l

# 查看特定任務
squeue -j <job_id>
```

### 3.2 查看輸出

```bash
# 實時查看輸出
tail -f logs/slurm/pwm_flow_single_<job_id>.out

# 實時查看錯誤
tail -f logs/slurm/pwm_flow_single_<job_id>.err

# 查看訓練日誌
tail -f logs/slurm/training_<job_id>.log
```

### 3.3 WandB 監控

在訓練開始後，訪問 https://wandb.ai/your_entity/flow-pwm-comparison

實時查看：
- 學習曲線
- Loss 變化
- 梯度範數
- FPS 和訓練速度
- 系統資源使用

---

## 4. 管理任務

### 4.1 取消任務

```bash
# 取消單個任務
scancel <job_id>

# 取消你的所有任務
scancel -u $USER

# 取消特定名稱的任務
scancel --name=pwm_flow_single
```

### 4.2 查看任務信息

```bash
# 查看已完成任務的詳細信息
sacct -j <job_id> --format=JobID,JobName,State,ExitCode,Elapsed,MaxRSS

# 查看最近的任務
sacct -u $USER --starttime=today
```

---

## 5. 結果分析

### 5.1 自動生成的視覺化

訓練完成後，視覺化會自動生成在：
```
logs/<algorithm>_<task>_seed<seed>/visualizations/
```

包含：
- `learning_curves.png` - 學習曲線（Rewards, Losses, FPS）
- `world_model_losses.png` - World Model 損失
- `gradient_norms.png` - 梯度範數
- `summary_statistics.png` - 統計摘要

### 5.2 手動生成視覺化

如果自動生成失敗，可以手動運行：

```bash
python scripts/generate_visualizations.py \
    --log-dir logs/pwm_48M_flow_dflex_ant_seed42
```

### 5.3 比較多個實驗

```bash
# 比較不同種子
python scripts/compare_runs.py \
    --task dflex_ant \
    --algorithm pwm_48M_flow \
    --seeds 42 123 456

# 比較 Baseline vs Flow
python scripts/compare_runs.py \
    --run-dirs logs/pwm_48M_dflex_ant_seed42 logs/pwm_48M_flow_dflex_ant_seed42 \
    --run-names "Baseline" "Flow" \
    --output-dir logs/comparisons/baseline_vs_flow_ant
```

---

## 6. 資源使用建議

### 6.1 單 GPU 任務

適合：
- 初步測試
- 單個實驗
- 資源受限時

優點：
- 簡單直接
- 不需要考慮並行

缺點：
- 一次只能跑一個實驗
- 多個種子需要依次提交

### 6.2 多 GPU 任務（4×H100）

適合：
- 需要多個種子（統計顯著性）
- 需要測試多個任務
- 直接比較 baseline vs flow
- 趕時間的情況

優點：
- 4 倍加速（4 個實驗並行）
- 充分利用資源
- 一次提交完成多個實驗

策略選擇：
- `multi_seed`: 同任務不同種子，用於統計
- `multi_task`: 不同任務，用於廣泛評估
- `baseline_vs_flow`: 直接對比，各 2 個種子

---

## 7. 故障排除

### 7.1 任務被取消

檢查：
```bash
# 查看原因
sacct -j <job_id> --format=JobID,State,ExitCode,Reason

# 常見原因:
# - OUT_OF_MEMORY: 增加 --mem
# - TIMEOUT: 增加 --time
# - NODE_FAIL: 重新提交即可
```

### 7.2 CUDA 錯誤

```bash
# 檢查 GPU 可見性
echo $CUDA_VISIBLE_DEVICES

# 檢查 PyTorch CUDA
python -c "import torch; print(torch.cuda.is_available())"

# 檢查 GPU 狀態
nvidia-smi
```

### 7.3 WandB 同步問題

```bash
# 離線模式（如果網路問題）
export WANDB_MODE=offline

# 之後手動同步
wandb sync logs/<run_dir>/wandb/run-*
```

### 7.4 權限問題

```bash
# 確保腳本有執行權限
chmod +x scripts/*.sh
chmod +x scripts/*.py
```

---

## 8. 最佳實踐

### 8.1 實驗組織

```
logs/
├── pwm_48M_dflex_ant_seed42/          # Baseline, Ant, seed 42
├── pwm_48M_dflex_ant_seed123/         # Baseline, Ant, seed 123
├── pwm_48M_flow_dflex_ant_seed42/     # Flow, Ant, seed 42
├── pwm_48M_flow_dflex_ant_seed123/    # Flow, Ant, seed 123
└── comparisons/
    └── baseline_vs_flow_ant/          # 比較結果
```

### 8.2 命名規範

WandB runs 會自動命名為：
```
<task>_<algorithm>_seed<seed>_<timestamp>
```

Group 會自動設為：
```
<algorithm>-<task>
```

這樣可以輕鬆在 WandB UI 中按 group 過濾和比較。

### 8.3 資源配置

根據任務調整：

**小型任務（如 Ant, Hopper）：**
```bash
#SBATCH --mem=64GB
#SBATCH --time=24:00:00
```

**大型任務（如 Humanoid, MT30）：**
```bash
#SBATCH --mem=128GB
#SBATCH --time=48:00:00
```

**預訓練 World Model：**
```bash
#SBATCH --mem=256GB
#SBATCH --time=72:00:00
```

---

## 9. 完整實驗流程範例

### 9.1 單任務比較（3 個種子）

```bash
# 1. Baseline - 3 seeds (可以用單 GPU 依次提交)
./scripts/submit_job.sh single pwm_48M dflex_ant 42
./scripts/submit_job.sh single pwm_48M dflex_ant 123
./scripts/submit_job.sh single pwm_48M dflex_ant 456

# 2. Flow - 3 seeds (或用多 GPU 一次提交)
./scripts/submit_job.sh multi multi_seed dflex_ant 42
# 這會跑 seeds: 42, 123, 456, 789

# 3. 等待完成後比較
python scripts/compare_runs.py \
    --task dflex_ant \
    --algorithm pwm_48M \
    --seeds 42 123 456

python scripts/compare_runs.py \
    --task dflex_ant \
    --algorithm pwm_48M_flow \
    --seeds 42 123 456 789
```

### 9.2 直接對比（推薦）

```bash
# 一次提交，直接對比 baseline vs flow (各2個種子)
./scripts/submit_job.sh multi baseline_vs_flow dflex_ant 42

# 完成後會自動生成比較圖表
```

---

## 10. Checklist

提交任務前確認：

- [ ] 已登入 PACE Phoenix
- [ ] 已激活 conda 環境
- [ ] 已登入 WandB
- [ ] 數據路徑正確（如需預訓練）
- [ ] 修改了 SLURM 腳本中的：
  - [ ] `#SBATCH --account=` (你的 account)
  - [ ] `#SBATCH --mail-user=` (你的 email)
  - [ ] `export WANDB_ENTITY=` (你的 WandB entity)
- [ ] 腳本有執行權限 (`chmod +x`)
- [ ] logs 目錄存在

提交後：

- [ ] 檢查任務是否在隊列中 (`squeue -u $USER`)
- [ ] 監控輸出文件
- [ ] 在 WandB 上查看訓練進度
- [ ] 訓練完成後查看視覺化
- [ ] 比較多個實驗結果

---

## 11. 聯絡支援

**PACE 支援：**
- Email: pace-support@oit.gatech.edu
- 文檔: https://docs.pace.gatech.edu/

**專案相關：**
- 查看 `docs/flow-dynamics-comparison-guide.md`
- 查看 `docs/FLOW_IMPLEMENTATION_SUMMARY.md`
- 開 GitHub Issue

---

**祝實驗順利！** 🚀
