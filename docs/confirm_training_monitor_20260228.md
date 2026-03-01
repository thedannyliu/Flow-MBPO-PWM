# 正式訓練（Confirm）實驗監測紀錄

> **最後更新**：2026-02-28 21:45 EST  
> **分支**：`dev/unified-pwm-resume-20260223` (commit `3eccf3c`)  
> **叢集**：PACE-ICE (`ice-gpu` partition)  
> **WandB Project**：[`flow-mbpo-formal-training`](https://wandb.ai/danny010324/flow-mbpo-formal-training)

---

## 1. 實驗總覽

### 1.1 基本資訊

| 項目 | 值 |
|------|-----|
| **階段** | Confirm（正式訓練） |
| **Manifest** | `manifests/confirm_all_4methods_10seeds_20260301.csv` |
| **總行數** | 320（8 任務 × 4 方法 × 10 seeds） |
| **Epochs** | 15,000 |
| **Eval Runs** | 40 |
| **GPU** | L40S（PACE-ICE） |
| **Slurm Job ID** | `4136966` |
| **時間限制** | 16h |
| **Array 併行上限** | %16 |

### 1.2 Smoke 驗證結果

在提交 Confirm 前已通過完整 Smoke 驗證：

| 項目 | 結果 |
|------|------|
| L40S Smoke Job | `4136884` — 32/32 ✅ COMPLETED |
| Checkpoints | 32/32 `final_policy.pt` 產出 |
| Eval Artifacts | 32/32 `eval_summary.json` 產出 |
| WandB Sync | 所有 run 已同步至 `flow-mbpo-formal-experiments` |

---

## 2. 當前狀態

### 2.1 Job 狀態總覽

| 狀態 | 數量 | 備註 |
|------|------|------|
| ✅ COMPLETED | ~0 | 15000 epochs 需要數小時 |
| 🔄 RUNNING | ~42 | 正在跑 |
| ❌ FAILED | 5 | 見 §2.2 |
| 🕐 PENDING | ~273 | 排隊中 |

### 2.2 已知失敗（5 行）

| Row | Task | Method | Seed | 原因 |
|-----|------|--------|------|------|
| 0 | hopper | mlpwm_mlppolicy | 0 | Replay buffer trajectory length 不足 |
| 3 | hopper | mlpwm_mlppolicy | 3 | 同上 |
| 6 | hopper | mlpwm_mlppolicy | 6 | 同上 |
| 7 | hopper | mlpwm_mlppolicy | 7 | 同上 |
| 9 | hopper | mlpwm_mlppolicy | 9 | 同上 |

**錯誤訊息**：`RuntimeError: Did not find a single trajectory with sufficient length (length range: 8 - 12 / required=17)`

> 這是 hopper 任務在 `num_envs=64` 時的 replay buffer 初始化問題。Hopper 的 episode 天生很短（8-12 steps），而 sampler 需要至少 17 步的 trajectory。其他任務不受影響。

**處理方案**：
1. 減少 `num_envs`（例如 32）重新跑這 5 行
2. 或修改 replay buffer 的 `min_traj_len` setting

---

## 3. 方法 × 任務 × Seed 矩陣

### Row 範圍對照表

每 40 行為一個任務組（4 methods × 10 seeds）：

| Row 範圍 | Task |
|----------|------|
| 0–39 | hopper |
| 40–79 | ant |
| 80–119 | anymal |
| 120–159 | humanoid |
| 160–199 | snu_humanoid |
| 200–239 | leap_left_grasp_asymmetric |
| 240–279 | tracking_rough_unitree_g1 |
| 280–319 | leap_left_inhand_pen_twirl |

每個任務組內的方法順序：
- 每 10 行為一個方法（10 seeds）
- 方法順序：`mlpwm_mlppolicy` → `flowwm_mlppolicy` → `mlpwm_flowpolicy` → `flowwm_flowpolicy`

---

## 4. WandB 標注詳情

每個 run 自動帶有：

| 標注類型 | 欄位 | 範例 |
|----------|------|------|
| **Tags** | `stage_confirm`, `task_ant`, `method_flowwm_mlppolicy`, `gpu_type_L40S`, `seed_3` | 自動產生 |
| **Name** | `confirm_gym_ant_flowwm_mlppolicy_s3_default` | 唯一識別碼 |
| **Group** | `single_task_online_confirm_gym` | 按 stage + suite 分組 |
| **Notes** | 含 GPU type, node, job ID | 排查用 |
| **Config** | `experiment.*`, `runtime.slurm.*` | 完整可追溯 |

### 在 WandB 中常用篩選

```
# 看所有 hopper 實驗
tags: task_hopper

# 看 flow world model 的所有結果
tags: method_flowwm_mlppolicy OR method_flowwm_flowpolicy

# 看特定 seed
tags: seed_0

# 比較四種方法在 ant 上的表現
tags: task_ant
Group by: experiment.method
```

---

## 5. 監測指令

### 5.1 快速狀態

```bash
# 總覽
sacct -j 4136966 --format=State --noheader | grep -v "^$" | sort | uniq -c

# 詳細（含時長）
sacct -j 4136966 --format=JobID%20,State%12,ExitCode,Elapsed%12 --noheader | grep -v "extern\|batch" | head -30

# 排隊中的 job
squeue -u $USER --format="%.12i %.2t %.10M %.25R" --noheader | head -15
```

### 5.2 查看訓練進度

```bash
# 看特定 row 的訓練 log（例如 row 40 = ant, mlpwm_mlppolicy, seed=0）
tail -20 logs/slurm/single_task_online/confirm/sto_confirm_L40S_4136966_40.out

# 看錯誤
tail -20 logs/slurm/single_task_online/confirm/sto_confirm_L40S_4136966_40.err
```

### 5.3 檢查 checkpoint 產出

```bash
# 統計 checkpoint 數量
find scripts/outputs/single_task_online/confirm/ -name "best_policy.pt" | wc -l
find scripts/outputs/single_task_online/confirm/ -name "final_policy.pt" | wc -l

# 總輸出大小
du -sh scripts/outputs/single_task_online/confirm/
```

### 5.4 WandB 即時查看

打開 [flow-mbpo-formal-training](https://wandb.ai/danny010324/flow-mbpo-formal-training)，可以互動篩選。

---

## 6. 資源估算

| 項目 | 估算 |
|------|------|
| 每行 wall-clock | 6-16h（取決於任務複雜度） |
| 總 GPU 時數 | ~2000-5000h |
| 16h time limit | 簡單任務（hopper, ant）可完成；複雜任務可能需 checkpoint resume |
| 排隊等待 | %16 cap，320 行需 20 輪 × 每輪 6-16h |
| 預計全部完成 | 3-7 天（取決於 GPU 排程） |

### 若需 checkpoint resume

部分複雜任務（humanoid, snu_humanoid, tracking）在 16h 內可能無法跑完 15000 epochs。`run_manifest_job.py` 已內建自動 resume 機制：

```bash
# 重新提交同一個 manifest，會自動從 latest_checkpoint.pt 續跑
bash scripts/experiments/single_task_online/submit_manifest_array.sh \
  --manifest scripts/experiments/single_task_online/manifests/confirm_all_4methods_10seeds_20260301.csv \
  --gpu-type L40S --time 16:00:00 \
  --conda-env flow-mbpo \
  --python-bin /storage/ice1/2/9/eliu354/conda_envs/flow-mbpo/bin/python
```

---

## 7. 後續步驟

1. **持續監測** 當前 320 行的完成進度
2. **處理 hopper 失敗**：準備 `num_envs=32` 的 override 重跑 5 行
3. **Checkpoint resume**：16h 到期後，重新提交讓未完成的 row 自動續跑
4. **結果收集**：待全部完成後，用 WandB API 拉取 metrics 做統計分析
