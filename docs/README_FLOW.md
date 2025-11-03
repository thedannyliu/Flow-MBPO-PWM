# Flow-Matching World Model 實作摘要

## 📁 項目結構

```
PWM/
├── src/pwm/
│   ├── models/
│   │   ├── flow_world_model.py      ✅ Flow-matching 世界模型
│   │   └── world_model.py           (原有基線模型)
│   ├── utils/
│   │   ├── integrators.py           ✅ ODE 積分器 (Heun, Euler)
│   │   ├── esnr.py                  ✅ ESNR 計算
│   │   ├── monitoring.py            ✅ 訓練監控工具
│   │   ├── visualization.py         ✅ 自動可視化
│   │   └── reproducibility.py       ✅ 實驗可重現性
│   └── algorithms/
│       └── pwm.py                   ✅ PWM 算法（已修改）
├── scripts/
│   ├── cfg/alg/
│   │   ├── pwm_48M_flow.yaml        ✅ 48M flow 配置
│   │   └── pwm_5M_flow.yaml         ✅ 5M flow 配置
│   ├── slurm_single_gpu.sh          ✅ 單 GPU SLURM 腳本
│   ├── slurm_multi_gpu.sh           ✅ 多 GPU SLURM 腳本
│   ├── submit_job.sh                ✅ 作業提交輔助腳本
│   ├── verify_param_parity.py       ✅ 參數平衡驗證
│   ├── generate_visualizations.py   ✅ 可視化生成
│   └── compare_runs.py              ✅ 運行比較
└── docs/
    ├── flow-dynamics-comparison-guide.md      ✅ 實驗比較指南（中文）
    ├── PACE_USAGE_GUIDE.md                    ✅ PACE 集群指南（中文）
    ├── QUICKSTART.md                          ✅ 快速入門（中文）
    ├── FLOW_IMPLEMENTATION_SUMMARY.md         ✅ 實作總結（英文）
    └── IMPLEMENTATION_COMPLETE.md             ✅ 完整總結（中文）
```

## ✅ 核心實現

### 1. Flow-Matching 動力學
- **速度場**: v_θ(z, a, τ) 其中 τ ∈ [0,1]
- **訓練**: 整流流目標 ||v_θ(z_τ, a, τ) - v*||²，其中 v* = z_target - z_start
- **推理**: ODE 積分 dz/dτ = v_θ(z, a, τ) 使用 Heun 方法（RK2）

### 2. 參數平衡
- **基線**: units=[1792, 1792]，總參數 ≈ 48M
- **Flow**: units=[1788, 1788]，補償 +1 時間維度
- **差異**: < 2% (符合要求)

### 3. 增強功能
- ✅ 進度條和 ETA（tqdm）
- ✅ WandB 詳細日誌
- ✅ 自動可視化生成
- ✅ 數據一致性驗證
- ✅ SLURM 集群部署

## 🚀 快速開始

### 環境設置
```bash
# 加載模組
module load anaconda3/2023.09-0

# 創建環境（第一次）
cd PWM
conda env create -f environment.yaml
conda activate pwm
pip install -e .
```

### 本地測試
```bash
# 基線（5M 模型）
python scripts/train_dflex.py general=dflex_ant alg=pwm_5M general.epochs=100 seed=42

# Flow（5M 模型）
python scripts/train_dflex.py general=dflex_ant alg=pwm_5M_flow general.epochs=100 seed=42
```

### 集群運行
```bash
# 單 GPU
./scripts/submit_job.sh single dflex_ant pwm_48M_flow 42

# 4×GPU 並行（多種子）
./scripts/submit_job.sh multi dflex_ant pwm_48M_flow multi_seed
```

## 📊 關鍵指標

| 指標 | 說明 |
|-----|------|
| **Reward** | 策略回報（越高越好） |
| **WM Loss** | 世界模型總損失 |
| ↳ Dynamics Loss | 動力學預測損失 |
| ↳ Reward Loss | 獎勵預測損失 |
| **Actor Loss** | 策略損失 |
| **Value Loss** | 價值函數損失 |
| **Gradient Norms** | 訓練穩定性指標 |

## 📖 文檔索引

1. **快速入門**: 查看 `docs/QUICKSTART.md`（5分鐘上手）
2. **完整指南**: 查看 `docs/flow-dynamics-comparison-guide.md`（12個部分）
3. **集群使用**: 查看 `docs/PACE_USAGE_GUIDE.md`（PACE Phoenix）
4. **技術細節**: 查看 `docs/FLOW_IMPLEMENTATION_SUMMARY.md`
5. **完整總結**: 查看 `docs/IMPLEMENTATION_COMPLETE.md`

## 🎯 實驗流程

```
1. 環境設置 → 2. 參數驗證 → 3. 快速測試 → 4. 完整實驗 → 5. 結果分析
   (5分鐘)     (1分鐘)        (30分鐘)      (數小時)       (隨時)
```

## 📧 技術支持

遇到問題？查看：
- `docs/IMPLEMENTATION_COMPLETE.md` 的故障排除部分
- `docs/flow-dynamics-comparison-guide.md` 第11節（常見問題）

祝實驗順利！🚀
