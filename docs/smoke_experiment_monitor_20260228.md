# 正式訓練實驗監測紀錄

> **最後更新**：2026-02-28 21:15 EST  
> **分支**：`dev/unified-pwm-resume-20260223` (commit `c063efb`)  
> **叢集**：PACE-ICE (`ice-gpu` partition)

---

## 1. 實驗提交總覽

### 1.1 Smoke 階段（Stage 1）

| GPU 類型 | Slurm Job ID | 總行數 | 併行上限 | 時間限制 | 提交時間 | 狀態 |
|----------|-------------|--------|---------|---------|---------|------|
| **H100** | `4136626` | 32 | %16 | 8h | 2026-02-28 21:12 | 🕐 PENDING |
| **H200** | `4136628` | 32 | %16 | 8h | 2026-02-28 21:12 | 🕐 PENDING |
| **L40S** | `4136632` | 32 | %16 | 8h | 2026-02-28 21:12 | 🔄 RUNNING (3/32) |

### 1.2 Manifest 資訊

- **檔案**：`scripts/experiments/single_task_online/manifests/smoke_formal_all_4methods_20260301.csv`
- **組合**：8 任務 × 4 方法 × 1 seed (seed=0) = 32 行
- **Epoch 數**：200
- **Eval runs**：8
- **WandB Project**：`flow-mbpo-single-task-online-smoke-formal-20260301`

---

## 2. 設定對齊驗證（已通過）

### 2.1 Manifest 參數

| 參數 | 值 | 狀態 |
|------|-----|------|
| Stage | `smoke` | ✅ |
| Seeds | `0` | ✅ |
| Max epochs | `200` | ✅ |
| Eval runs | `8` | ✅ |
| Hparam profile | `default` | ✅ |
| Method→Alg 映射 | 4/4 正確 | ✅ |
| 任務覆蓋 | 8/8 完整 | ✅ |
| Factorial 完整性 | 32/32 無重複 | ✅ |

### 2.2 算法配置確認

| 方法 | 配置檔 | World Model | Policy | Actor units | WM units |
|------|--------|-------------|--------|-------------|----------|
| `mlpwm_mlppolicy` | `pwm_5M_baseline_final` | MLP `WorldModel` | MLP `ActorStochasticMLP` | [400,200,100] | [512,512] |
| `flowwm_mlppolicy` | `pwm_5M_flow_v2_substeps4` | Flow `FlowWorldModel` | MLP `ActorStochasticMLP` | [400,200,100] | [512,512] |
| `mlpwm_flowpolicy` | `pwm_5M_flowpolicy` | MLP `WorldModel` | Flow `ActorFlowODE` | [400,200,100] | [512,512] |
| `flowwm_flowpolicy` | `pwm_5M_fullflow` | Flow `FlowWorldModel` | Flow `ActorFlowODE` | [400,200,100] | [512,512] |

> ✅ 所有配置共用 critic=[400,200], encoder=[256], num_bins=101。僅 WM type 和 Policy type 有差異，實驗設計公平。

### 2.3 Slurm 提交設定一致性

| 參數 | H100 | H200 | L40S |
|------|------|------|------|
| Partition | `ice-gpu` | `ice-gpu` | `ice-gpu` |
| Account | `coc` | `coc` | `coc` |
| GRES | `gpu:h100:1` | `gpu:h200:1` | `gpu:l40s:1` |
| Time | 8h | 8h | 8h |
| Memory | 128G | 128G | 128G |
| CPUs | 16 | 16 | 16 |
| Conda env | `flow-mbpo` | `flow-mbpo` | `flow-mbpo` |
| Array concurrency | %16 | %16 | %16 |

---

## 3. 任務 × 方法 矩陣

### 3.1 Row Index 對照表

| Row | Task | Method | Run Key |
|-----|------|--------|---------|
| 0 | hopper | mlpwm_mlppolicy | smoke_gym_hopper_mlpwm_mlppolicy_s0_default |
| 1 | hopper | flowwm_mlppolicy | smoke_gym_hopper_flowwm_mlppolicy_s0_default |
| 2 | hopper | mlpwm_flowpolicy | smoke_gym_hopper_mlpwm_flowpolicy_s0_default |
| 3 | hopper | flowwm_flowpolicy | smoke_gym_hopper_flowwm_flowpolicy_s0_default |
| 4 | ant | mlpwm_mlppolicy | smoke_gym_ant_mlpwm_mlppolicy_s0_default |
| 5 | ant | flowwm_mlppolicy | smoke_gym_ant_flowwm_mlppolicy_s0_default |
| 6 | ant | mlpwm_flowpolicy | smoke_gym_ant_mlpwm_flowpolicy_s0_default |
| 7 | ant | flowwm_flowpolicy | smoke_gym_ant_flowwm_flowpolicy_s0_default |
| 8 | anymal | mlpwm_mlppolicy | smoke_mjlab_proxy_anymal_mlpwm_mlppolicy_s0_default |
| 9 | anymal | flowwm_mlppolicy | smoke_mjlab_proxy_anymal_flowwm_mlppolicy_s0_default |
| 10 | anymal | mlpwm_flowpolicy | smoke_mjlab_proxy_anymal_mlpwm_flowpolicy_s0_default |
| 11 | anymal | flowwm_flowpolicy | smoke_mjlab_proxy_anymal_flowwm_flowpolicy_s0_default |
| 12 | humanoid | mlpwm_mlppolicy | smoke_gym_humanoid_mlpwm_mlppolicy_s0_default |
| 13 | humanoid | flowwm_mlppolicy | smoke_gym_humanoid_flowwm_mlppolicy_s0_default |
| 14 | humanoid | mlpwm_flowpolicy | smoke_gym_humanoid_mlpwm_flowpolicy_s0_default |
| 15 | humanoid | flowwm_flowpolicy | smoke_gym_humanoid_flowwm_flowpolicy_s0_default |
| 16 | snu_humanoid | mlpwm_mlppolicy | smoke_mjlab_proxy_snu_humanoid_mlpwm_mlppolicy_s0_default |
| 17 | snu_humanoid | flowwm_mlppolicy | smoke_mjlab_proxy_snu_humanoid_flowwm_mlppolicy_s0_default |
| 18 | snu_humanoid | mlpwm_flowpolicy | smoke_mjlab_proxy_snu_humanoid_mlpwm_flowpolicy_s0_default |
| 19 | snu_humanoid | flowwm_flowpolicy | smoke_mjlab_proxy_snu_humanoid_flowwm_flowpolicy_s0_default |
| 20 | leap_left_grasp | mlpwm_mlppolicy | smoke_mjlab_leap_left_grasp_asymmetric_mlpwm_mlppolicy_s0_default |
| 21 | leap_left_grasp | flowwm_mlppolicy | smoke_mjlab_leap_left_grasp_asymmetric_flowwm_mlppolicy_s0_default |
| 22 | leap_left_grasp | mlpwm_flowpolicy | smoke_mjlab_leap_left_grasp_asymmetric_mlpwm_flowpolicy_s0_default |
| 23 | leap_left_grasp | flowwm_flowpolicy | smoke_mjlab_leap_left_grasp_asymmetric_flowwm_flowpolicy_s0_default |
| 24 | tracking_rough | mlpwm_mlppolicy | smoke_mjlab_tracking_rough_unitree_g1_mlpwm_mlppolicy_s0_default |
| 25 | tracking_rough | flowwm_mlppolicy | smoke_mjlab_tracking_rough_unitree_g1_flowwm_mlppolicy_s0_default |
| 26 | tracking_rough | mlpwm_flowpolicy | smoke_mjlab_tracking_rough_unitree_g1_mlpwm_flowpolicy_s0_default |
| 27 | tracking_rough | flowwm_flowpolicy | smoke_mjlab_tracking_rough_unitree_g1_flowwm_flowpolicy_s0_default |
| 28 | inhand_pen_twirl | mlpwm_mlppolicy | smoke_mjlab_leap_left_inhand_pen_twirl_mlpwm_mlppolicy_s0_default |
| 29 | inhand_pen_twirl | flowwm_mlppolicy | smoke_mjlab_leap_left_inhand_pen_twirl_flowwm_mlppolicy_s0_default |
| 30 | inhand_pen_twirl | mlpwm_flowpolicy | smoke_mjlab_leap_left_inhand_pen_twirl_mlpwm_flowpolicy_s0_default |
| 31 | inhand_pen_twirl | flowwm_flowpolicy | smoke_mjlab_leap_left_inhand_pen_twirl_flowwm_flowpolicy_s0_default |

---

## 4. 即時監測指令

### 4.1 查看所有 Job 狀態

```bash
# 總覽
squeue -u $USER --format="%.12i %.9P %.25j %.2t %.10M %.6D %.25R"

# 看特定 array job 詳細
squeue -j 4136626   # H100
squeue -j 4136628   # H200
squeue -j 4136632   # L40S

# 看已完成的 job 結果（含失敗）
sacct -j 4136626 --format=JobID,State,ExitCode,Elapsed,MaxRSS --parsable2
sacct -j 4136628 --format=JobID,State,ExitCode,Elapsed,MaxRSS --parsable2
sacct -j 4136632 --format=JobID,State,ExitCode,Elapsed,MaxRSS --parsable2
```

### 4.2 GPU 使用率監測

若某個 job 正在跑，可以登入該節點查看 GPU 使用率：
```bash
# 找出跑在哪個節點
squeue -j <JOB_ID> --format="%N"

# SSH 到節點後
nvidia-smi
# 或者使用 srun
srun --jobid=<JOB_ID> nvidia-smi
```

### 4.3 Log 檔案路徑

```bash
# Stdout / Stderr
ls logs/slurm/single_task_online/smoke/

# H100 的 task 0 log
cat logs/slurm/single_task_online/smoke/sto_smoke_H100_4136626_0.out

# H200 的 task 5 log
cat logs/slurm/single_task_online/smoke/sto_smoke_H200_4136628_5.out

# L40S 的 task 10 log
cat logs/slurm/single_task_online/smoke/sto_smoke_L40S_4136632_10.out
```

### 4.4 快速統計完成情況

```bash
# 統計各 job 的完成/失敗/執行中狀態
for JOB in 4136626 4136628 4136632; do
  echo "=== Job $JOB ==="
  sacct -j $JOB --format=State --noheader | sort | uniq -c | sort -rn
done
```

### 4.5 檢查產出是否完整

```bash
# 搜尋所有 smoke 訓練產出
find outputs/ -name "eval_summary.json" -newer scripts/experiments/single_task_online/manifests/smoke_formal_all_4methods_20260301.csv 2>/dev/null | wc -l

# 檢查特定 run 的完整度
for dir in outputs/2026-03-*/; do
  ckpt=$(ls "$dir"/{best_policy,final_policy}.pt 2>/dev/null | head -1)
  eval_json=$(ls "$dir"/eval/eval_summary.json 2>/dev/null)
  echo "$dir: ckpt=${ckpt:-(missing)} eval=${eval_json:-(missing)}"
done
```

---

## 5. 進度追蹤表

以下表格需手動更新（或等 job 完成後用指令填入）：

### H100 Job `4136626`

| Task ID | Row | Task × Method | 狀態 | 節點 | 時長 | 備註 |
|---------|-----|---------------|------|------|------|------|
| 0 | 0 | hopper × mlpwm_mlppolicy | 🕐 PENDING | - | - | - |
| 1 | 1 | hopper × flowwm_mlppolicy | 🕐 PENDING | - | - | - |
| ... | ... | ... | ... | ... | ... | ... |

### H200 Job `4136628`

| Task ID | Row | Task × Method | 狀態 | 節點 | 時長 | 備註 |
|---------|-----|---------------|------|------|------|------|
| 0 | 0 | hopper × mlpwm_mlppolicy | 🕐 PENDING | - | - | - |
| ... | ... | ... | ... | ... | ... | ... |

### L40S Job `4136632`

| Task ID | Row | Task × Method | 狀態 | 節點 | 時長 | 備註 |
|---------|-----|---------------|------|------|------|------|
| 0 | 0 | hopper × mlpwm_mlppolicy | 🔄 RUNNING | atl1-1-03-004-21-0 | 0:01 | 初始化中 |
| 1 | 1 | hopper × flowwm_mlppolicy | 🔄 RUNNING | atl1-1-03-004-21-0 | 0:01 | 初始化中 |
| 2 | 2 | hopper × mlpwm_flowpolicy | 🔄 RUNNING | atl1-1-03-004-23-0 | 0:01 | 初始化中 |
| 3–31 | 3–31 | 其餘 | 🕐 PENDING | - | - | 排隊中 |

---

## 6. 異常處理 SOP

| 問題 | 檢查方式 | 處理方式 |
|------|---------|---------|
| Job FAILED (exit ≠ 0) | `sacct -j <JOB_ID>` | 查 `.err` log，修復後用 `sbatch` 重跑單行 |
| OOM | log 中出現 `CUDA out of memory` | 降低 `num_envs` 或改用更大 VRAM 的 GPU |
| Timeout | Job state = `TIMEOUT` | 增加 `--time` 或用 packed mode 分批 |
| WandB 未接收指標 | 登入 WandB project 查看 | 檢查 `WANDB_API_KEY` 和網路 |
| Checkpoint 缺失 | `ls outputs/*/best_policy.pt` | 檢查 `alg.save_interval` 設定 |
| Eval artifacts 缺失 | `ls outputs/*/eval/` | 檢查 eval script 是否被跳過 |

---

## 7. 後續計畫

1. **Smoke 完成後**（預計 3–6 小時內）
   - 統計所有 32 行的完成情況
   - 確認所有 checkpoint + eval artifacts 產出
   - 檢查 WandB 指標

2. **GPU 使用率評估**
   - 若 smoke 期間 GPU 使用率 < 50%，考慮 Confirm 階段使用 Packed mode
   - 目標：最大化 GPU 利用率與實驗迭代次數

3. **直接進入 Confirm**（跳過 Pilot）
   - 生成 Confirm manifest（10 seeds × 4 methods × 8 tasks = 320 行）
   - 根據 smoke 結果決定使用哪種 GPU 類型
   - 提交正式訓練
