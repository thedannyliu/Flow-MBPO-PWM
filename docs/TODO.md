# Flow-Matching PWM - 使用者 TODO 清單

To-DO:
在您目前的腳本設定中，「使用多 GPU」的含義是實驗並行，而不是模型並行。

您腳本做的是：申請 4 個 GPU，然後同時進行 4 個獨立的實驗，每個實驗使用 1 個 GPU。例如，用 4 個 GPU 同時跑 4 個不同隨機種子的 pwm_48M 實驗。
您腳本沒做的：用 4 個 GPU 加速單一一個 pwm_48M 實驗。這需要修改 Python 程式碼本身（例如使用 DistributedDataParallel），遠比修改提交腳本複雜。
我將假設您的目標是前者（實驗並行），因為這符合您腳本的設計。

現有腳本的限制
我發現 submit_job.sh 在設計上存在一個小缺陷：在 multi 模式下，它沒有提供一個參數讓您指定要運行的演算法（algorithm），導致 multi_seed 這個策略會一直預設使用 pwm_48M。

我們可以修正這個問題。

我建議修改 submit_job.sh，讓 multi 模式可以接收 algorithm 參數。這只需要做一個小小的改動。

修改計畫
我將修改 submit_job.sh 腳本，把 multi 模式的用法從： $0 multi <strategy> <task> [base_seed]

改成： $0 multi <strategy> <algorithm> <task> [base_seed]

這樣您就可以明確指定要對哪個演算法進行多 GPU 的並行實驗了。

## 🔧 首次設置（5-10分鐘）

### 1. 創建 Conda 環境
```bash
cd /storage/home/hcoda1/9/eliu354/r-agarg35-0/projects/Flow-MBPO-PWM/PWM

# 加載模組
module load anaconda3/2023.09-0

# 創建環境（這只需要做一次）
conda env create -f environment.yaml

# 激活環境
conda activate pwm

# 安裝 PWM
pip install -e .
```

### 2. 個性化配置

#### 更新 SLURM 腳本中的電子郵件
編輯以下文件，找到 `#SBATCH --mail-user=` 這一行，更新為您的電子郵件：
- `scripts/slurm_single_gpu.sh`
- `scripts/slurm_multi_gpu.sh`

#### （可選）設置 WandB
如果您想使用 WandB 進行實驗追蹤：
```bash
wandb login YOUR_API_KEY
```

或者在 `pwm.py` 中找到 `wandb_logger` 初始化處，設置 `entity='your-wandb-team'`

### 3. 驗證設置
```bash
# 激活環境
conda activate pwm

# 驗證參數平衡（應該 < 2% 差異）
python scripts/verify_param_parity.py

# 如果看到 "✓ PASS"，則設置成功！
```

## 🧪 快速測試（30分鐘 - 推薦先做）

運行一個小型實驗確保一切工作正常：

```bash
# 基線（5M 模型，100 epochs）
python scripts/train_dflex.py \
    general=dflex_ant \
    alg=pwm_5M \
    general.epochs=100 \
    seed=42

# Flow（5M 模型，100 epochs）
python scripts/train_dflex.py \
    general=dflex_ant \
    alg=pwm_5M_flow \
    general.epochs=100 \
    seed=42
```

檢查：
- [ ] 訓練開始沒有錯誤
- [ ] 進度條顯示正確
- [ ] 在 `logs/` 目錄下生成圖表

## 🚀 完整實驗（數小時 - 在集群上運行）

### 選項 A: 單個實驗
```bash
# 基線
./scripts/submit_job.sh single dflex_ant pwm_48M 42

# Flow
./scripts/submit_job.sh single dflex_ant pwm_48M_flow 42
```

### 選項 B: 多種子並行（推薦）
```bash
# 4個種子並行運行
./scripts/submit_job.sh multi dflex_ant pwm_48M_flow multi_seed
```

### 選項 C: 基線 vs Flow 比較
```bash
# 同時運行 baseline 和 flow（各2個種子）
./scripts/submit_job.sh multi dflex_ant pwm_48M,pwm_48M_flow baseline_vs_flow
```

### 監控作業
```bash
# 查看作業狀態
squeue -u $USER

# 查看實時輸出
tail -f slurm-JOBID.out
```

## 📊 分析結果

### 單個運行
```bash
# 重新生成可視化
python scripts/generate_visualizations.py \
    --log-dir logs/dflex_ant/pwm_48M_flow/seed42
```

### 比較多個種子
```bash
# 比較同一算法的多個種子
python scripts/compare_runs.py \
    --task dflex_ant \
    --algorithm pwm_48M_flow \
    --seeds 42 43 44 45 \
    --output-dir analysis/flow_seeds/
```

### 比較 Baseline vs Flow
```bash
# 比較兩種方法
python scripts/compare_runs.py \
    --run-dirs \
        logs/dflex_ant/pwm_48M/seed42 \
        logs/dflex_ant/pwm_48M_flow/seed42 \
    --labels baseline flow \
    --output-dir analysis/baseline_vs_flow/
```

## 📖 需要幫助？

### 快速參考
- **5分鐘入門**: `docs/QUICKSTART.md`
- **詳細指南**: `docs/flow-dynamics-comparison-guide.md`
- **集群使用**: `docs/PACE_USAGE_GUIDE.md`
- **完整總結**: `docs/IMPLEMENTATION_COMPLETE.md`

### 常見問題

#### Q: 參數平衡驗證失敗？
查看 `scripts/verify_param_parity.py` 的輸出，它會建議新的 `units` 值。

#### Q: SLURM 作業失敗？
1. 檢查 `slurm-JOBID.out` 文件
2. 確認帳戶名稱正確（gts-agarg35）
3. 檢查 GPU 可用性：`sinfo -p phoenix-gpu-h100`

#### Q: NaN 損失？
1. 降低學習率
2. 檢查觀測值歸一化
3. 確認梯度裁剪已啟用

#### Q: WandB 登錄失敗？
```bash
wandb login YOUR_API_KEY
```
或在代碼中設置 `enabled=False` 禁用 WandB

## ✅ 檢查清單

### 設置階段
- [ ] 創建 conda 環境
- [ ] 激活環境並安裝 PWM
- [ ] 更新電子郵件地址（SLURM 腳本）
- [ ] （可選）設置 WandB
- [ ] 運行參數平衡驗證

### 測試階段
- [ ] 運行快速測試（5M 模型）
- [ ] 檢查訓練輸出
- [ ] 檢查可視化生成

### 實驗階段
- [ ] 提交完整實驗（48M 模型）
- [ ] 監控作業狀態
- [ ] 等待訓練完成

### 分析階段
- [ ] 生成可視化圖表
- [ ] 比較不同運行
- [ ] 分析 baseline vs flow

## 🎯 預期結果

### 訓練指標
- **Reward** 應該持續增長
- **Loss** 應該下降
- **Gradient Norms** 應該穩定

### 可視化
在 `logs/` 目錄下應該看到：
- `learning_curves.png`
- `world_model_losses.png`
- `gradient_norms.png`
- `training_summary.png`

### WandB（如果啟用）
應該看到詳細的指標、梯度直方圖和訓練進度

## 🎉 完成！

按照此清單完成後，您就可以運行 Flow-Matching PWM 實驗並與基線進行比較了！

祝實驗順利！🚀

---

**注意**: 如果遇到任何問題，請查看 `docs/IMPLEMENTATION_COMPLETE.md` 的故障排除部分。
