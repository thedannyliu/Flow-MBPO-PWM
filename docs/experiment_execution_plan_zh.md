# 單任務在線 RL 實驗執行計畫

> **日期**：2026-02-28  
> **分支**：`dev/unified-pwm-resume-20260223` (commit `c063efb`)  
> **叢集**：Georgia Tech PACE-ICE  
> **範圍**：Smoke → 正式訓練（Confirm），跳過 Pilot

---

## 1. 實驗總覽

### 1.1 任務面板（8 個任務）

| 套件 | 任務 | Hydra env | 複雜度 |
|------|------|-----------|--------|
| dFlex | hopper | `gym_hopper_mujoco` | 低 |
| dFlex | ant | `gym_ant_mujoco` | 中 |
| dFlex | anymal | `mjlab_velocity_flat_unitree_go2` | 中 |
| dFlex | humanoid | `gym_humanoid_mujoco` | 中高 |
| dFlex | snu_humanoid | `mjlab_velocity_flat_unitree_g1` | 高 |
| mjlab | leap_left_grasp_asymmetric | `mjlab_leap_left_grasp_asymmetric` | 中 |
| mjlab | tracking_rough_unitree_g1 | `mjlab_tracking_rough_unitree_g1` | 中高 |
| mjlab | leap_left_inhand_pen_twirl | `mjlab_leap_left_inhand_pen_twirl` | 高 |

### 1.2 方法矩陣（4 種組合，2×2 factorial）

| 方法代號 | 演算法配置 | World Model | Policy |
|----------|-----------|-------------|--------|
| `mlpwm_mlppolicy` | `pwm_5M_baseline_final` | MLP | MLP |
| `flowwm_mlppolicy` | `pwm_5M_flow_v2_substeps4` | Flow | MLP |
| `mlpwm_flowpolicy` | `pwm_5M_flowpolicy` | MLP | Flow |
| `flowwm_flowpolicy` | `pwm_5M_fullflow` | Flow | Flow |

### 1.3 兩階段流程

| 階段 | Seeds | 每 seed 的行數 | 總行數 | Epoch 數 | 目的 |
|------|-------|---------------|--------|----------|------|
| **Smoke** | 1 (seed=0) | 8×4 = 32 | 32 | 200 | 驗證所有任務×方法組合跑得通 |
| **Confirm** | 10 (seed 0–9) | 8×4 = 32 | 320 | 15000 | 產出可報告的統計結果 |

> **注意**：跳過 Pilot 階段。Smoke 驗證通過後直接進入 Confirm 正式訓練。

---

## 2. Smoke 階段（當前步驟）

### 2.1 Manifest 已就緒

Manifest 檔案：`scripts/experiments/single_task_online/manifests/smoke_formal_all_4methods_20260301.csv`

- 32 行（8 任務 × 4 方法 × seed=0）
- 每行最多 200 epochs
- 已通過 dry-run 驗證

### 2.2 提交指令

```bash
# === L40S（已驗證過，最穩定，優先提交）===
bash scripts/experiments/single_task_online/submit_manifest_array.sh \
  --manifest scripts/experiments/single_task_online/manifests/smoke_formal_all_4methods_20260301.csv \
  --gpu-type L40S --time 08:00:00 \
  --conda-env flow-mbpo \
  --python-bin /storage/ice1/2/9/eliu354/conda_envs/flow-mbpo/bin/python

# === H100（等 canary job 4136399 通過後提交）===
bash scripts/experiments/single_task_online/submit_manifest_array.sh \
  --manifest scripts/experiments/single_task_online/manifests/smoke_formal_all_4methods_20260301.csv \
  --gpu-type H100 --time 08:00:00 \
  --conda-env flow-mbpo \
  --python-bin /storage/ice1/2/9/eliu354/conda_envs/flow-mbpo/bin/python

# === H200（等 canary job 4136400 通過後提交）===
bash scripts/experiments/single_task_online/submit_manifest_array.sh \
  --manifest scripts/experiments/single_task_online/manifests/smoke_formal_all_4methods_20260301.csv \
  --gpu-type H200 --time 08:00:00 \
  --conda-env flow-mbpo \
  --python-bin /storage/ice1/2/9/eliu354/conda_envs/flow-mbpo/bin/python
```

### 2.3 通過標準

全部 32 行滿足以下條件即算通過：
1. 訓練 exit code = 0
2. 產出 checkpoint（`best_policy.pt` 或 `final_policy.pt`）
3. 產出 `eval_summary.json` + rollout artifacts
4. 至少 1 個 dFlex 和 1 個 mjlab 在每種方法上成功
5. WandB 收到訓練與評估指標

---

## 3. Confirm 階段（Smoke 通過後）

### 3.1 生成 Confirm Manifest

```bash
# 生成 10 seeds × 4 methods × 8 tasks = 320 行
PYTHONPATH=src:$PYTHONPATH python scripts/experiments/single_task_online/build_manifest.py \
  --stage confirm \
  --methods mlpwm_mlppolicy,flowwm_mlppolicy,mlpwm_flowpolicy,flowwm_flowpolicy \
  --wandb-project flow-mbpo-single-task-online-confirm-20260301 \
  --output scripts/experiments/single_task_online/manifests/confirm_all_4methods_20260301.csv
```

### 3.2 提交指令

```bash
# Confirm 需要更長的時間和更多記憶體
bash scripts/experiments/single_task_online/submit_manifest_array.sh \
  --manifest scripts/experiments/single_task_online/manifests/confirm_all_4methods_20260301.csv \
  --gpu-type H100 --time 48:00:00 --mem 256G \
  --conda-env flow-mbpo \
  --python-bin /storage/ice1/2/9/eliu354/conda_envs/flow-mbpo/bin/python
```

### 3.3 資源估算

| GPU 類型 | 每行估計時間 | 320 行 GPU 時數 | 並行 16 個的日曆天數 |
|----------|------------|----------------|-------------------|
| H100 | 12–24h | 3840–7680h | 5–10 天 |
| H200 | 10–20h | 3200–6400h | 4–8 天 |
| L40S | 16–30h | 5120–9600h | 7–14 天 |

---

## 4. 如何利用 GPU 同時跑多個實驗

PACE-ICE 提供兩種方式在有限 GPU 資源下最大化吞吐量：

### 4.1 方式一：Slurm Array Job（一個 GPU 跑一個實驗）

這是**預設且最穩定**的模式。每個 Slurm array task 使用一張 GPU 跑一行 manifest。

```
submit_manifest_array.sh
  └─ sbatch --array=0-31%16
       ├─ task 0 → GPU #1 → manifest row 0
       ├─ task 1 → GPU #2 → manifest row 1
       ├─ ...
       └─ task 31 → GPU #N → manifest row 31
```

**重點參數**：
- `--array=0-31%16`：代表 32 個 task（0 到 31），最多同時排 16 個
- `%16` 是併行上限（Slurm concurrency cap），設太高會佔滿整個 partition

**使用時機**：
- 大部分情況下用這個即可
- 每行實驗佔一整張 GPU 的 VRAM
- 適合所有 stage（smoke / confirm）

**提交範例**：
```bash
bash scripts/experiments/single_task_online/submit_manifest_array.sh \
  --manifest <manifest.csv> \
  --gpu-type H100 --time 24:00:00 \
  --conda-env flow-mbpo \
  --python-bin /storage/ice1/2/9/eliu354/conda_envs/flow-mbpo/bin/python
```

### 4.2 方式二：Packed Mode（一個 GPU 同時跑多個實驗）

當單個實驗的 VRAM 用量較小時（例如 smoke 階段的 200 epochs、小 batch），可以讓**一張 GPU 同時跑 2–4 個實驗**，大幅節省排隊時間。

```
submit_manifest_packed_array.sh
  └─ sbatch --array=0-7%2
       ├─ task 0 (pack_index=0, pack_size=4) → 1 GPU
       │    ├─ row 0 (subprocess 1) ─┐
       │    ├─ row 1 (subprocess 2) ─┤ 同時跑 2 個
       │    ├─ row 2 (等前面完成後)   │
       │    └─ row 3 (等前面完成後)   │
       │                              ↓
       │    （一張 GPU 依序消化 4 行，最多 2 行併行）
       │
       ├─ task 1 (pack_index=1) → 1 GPU → rows 4-7
       └─ ...
```

**重點參數**：

| 參數 | 意義 | 建議值 |
|------|------|--------|
| `--pack-size N` | 每個 Slurm task 消化幾行 manifest | 4（smoke）, 2（confirm） |
| `--runs-per-gpu N` | 同一張 GPU 上最多同時跑幾個 subprocess | 2（安全值）|
| `--max-concurrent-jobs N` | 最多同時排幾個 Slurm array task | 2–4 |

**Warp Cache 隔離**：

在 packed mode 下，mjlab 任務使用 Warp 進行即時 CUDA 編譯。多個 row 共用同一份 cache 會出現 PCH 檔案競態（race condition）。我們的 `run_manifest_pack.py` 已自動處理此問題：每個 row 使用獨立的 `WARP_CACHE_DIR`。

```
~/.cache/warp/job_<SLURM_JOB_ID>/row_0/
~/.cache/warp/job_<SLURM_JOB_ID>/row_1/
...
```

**提交範例**：
```bash
# 32 行 manifest，每 4 行一包，每包同時跑 2 個 → 8 個 Slurm tasks
bash scripts/experiments/single_task_online/submit_manifest_packed_array.sh \
  --manifest scripts/experiments/single_task_online/manifests/smoke_formal_all_4methods_20260301.csv \
  --gpu-type L40S --time 08:00:00 \
  --pack-size 4 --runs-per-gpu 2 --max-concurrent-jobs 4 \
  --conda-env flow-mbpo \
  --python-bin /storage/ice1/2/9/eliu354/conda_envs/flow-mbpo/bin/python
```

**使用時機**：
- Smoke 階段（VRAM 用量低，epochs 少）
- 想加速 quick check 或 canary 測試
- ⚠️ **不建議用在 Confirm**：15000 epochs 的 VRAM 用量高，packed 容易 OOM

### 4.3 兩種模式的比較

| | Array (1:1) | Packed (N:1) |
|---|---|---|
| 每 GPU 實驗數 | 1 | 2–4 |
| 佔用 Slurm tasks 數 | = manifest 行數 | = 行數 ÷ pack_size |
| 排隊壓力 | 高（需要很多 GPU slots） | 低（少量 slots 即可） |
| OOM 風險 | 低 | 中~高（取決於任務 VRAM） |
| 適用階段 | 全部 | Smoke |
| Warp cache 問題 | 無 | 已自動隔離 |

### 4.4 最佳化策略建議

1. **Smoke 用 Packed**：32 行只需 8 個 Slurm tasks，排隊更快
2. **Confirm 用 Array**：320 行各佔一張 GPU，用 `%16` 或 `%32` 控制同時跑的數量
3. **多 GPU 類型分散提交**：L40S、H100、H200 各提交一批，分散排隊壓力
4. **時間估算**：
   - Smoke: 每行 2–6h → 用 `--time 08:00:00`
   - Confirm: 每行 12–30h → 用 `--time 48:00:00`

---

## 5. 監控與除錯

### 5.1 查看 Job 狀態

```bash
# 查看所有排隊/執行中的 job
squeue -u $USER

# 查看特定 array job 的所有 task
squeue -j <JOB_ID>

# 查看完成的 job 結果
sacct -j <JOB_ID> --format=JobID,State,ExitCode,Elapsed,MaxRSS
```

### 5.2 查看 Log 檔案

```bash
# Slurm stdout/stderr
ls logs/slurm/single_task_online/smoke/

# 看特定 task 的 log
cat logs/slurm/single_task_online/smoke/sto_smoke_L40S_<ARRAY_JOB_ID>_<TASK_ID>.out
```

### 5.3 查看訓練輸出

```bash
# 每個任務的輸出在 outputs/ 目錄下
ls outputs/<date>/<time>/

# 檢查 checkpoint
ls outputs/<date>/<time>/best_policy.pt
ls outputs/<date>/<time>/final_policy.pt

# 檢查 eval 結果
cat outputs/<date>/<time>/eval/eval_summary.json
```

### 5.4 WandB 檢查

- Project: 對應 manifest 中的 `wandb_project` 欄位
- 確認每個 run 都有 `stage`, `method`, `task` 等 metadata tag
- 確認有 `rewards`, `policy_loss` 等訓練指標

---

## 6. 時間線

```
2/28  ──── Smoke 提交 (L40S) ────────────────────── 32 rows, ~1-2 天
3/01  ──── 等 H100/H200 canary 結果
3/02  ──── Smoke 提交 (H100/H200)
3/03  ──── Smoke 結果確認 ──────────────────────────
3/03  ──── Confirm manifest 生成 + 提交 ────────── 320 rows
3/03~3/14 ──── Confirm 正式訓練中 ──────────────── ~10 天
3/14  ──── 結果收集 + 分析 ─────────────────────────
```
