# 正式訓練（Confirm）實驗監測紀錄

> **最後更新**：2026-02-28 22:10 EST  
> **分支**：`dev/unified-pwm-resume-20260223` (commit `5ec8b4a`)  
> **叢集**：PACE-ICE (`ice-gpu` partition, 16h time limit)  
> **WandB Project（正式訓練）**：[`flow-mbpo-formal-training`](https://wandb.ai/danny010324/flow-mbpo-formal-training)  
> **WandB Project（Smoke）**：[`flow-mbpo-formal-experiments`](https://wandb.ai/danny010324/flow-mbpo-formal-experiments)

---

## 1. 當前 Job 總覽

### 1.1 正式訓練（Confirm）

| GPU 類型 | Job ID | 狀態 | Rows | 備註 |
|----------|--------|------|------|------|
| **H100** | `4137131` | 🕐 PENDING | 320 | H100 節點近滿（58/64 CPUs used），等候排程 |

> [!NOTE]
> H200 提交被 `QOSMaxSubmitJobPerUserLimit` 擋住（一次只能有一個 320-row array job）。  
> 若 H100 長時間無法排到，可考慮取消改投 L40S（有 24 idle GPUs）。

### 1.2 Smoke 驗證結果

| 項目 | 結果 |
|------|------|
| 第一次 Smoke（無 horizon fix） | L40S `4136884` — 32/32 ✅ ALL COMPLETED |
| Hopper horizon fix 驗證 | L40S `4137103` — hopper rows RUNNING（非 trajectory 失敗），8 rows因 WandB init_timeout 失敗（transient） |
| Hopper Fix | `alg.horizon=7` 成功解決 replay buffer trajectory 長度問題 |

---

## 2. 實驗設計

### 2.1 基本參數

| 項目 | 值 |
|------|-----|
| **階段** | Confirm（正式訓練） |
| **Manifest** | `manifests/confirm_all_4methods_10seeds_20260301.csv` |
| **總行數** | 320 = 8 tasks × 4 methods × 10 seeds |
| **Epochs** | 15,000 |
| **Eval Runs** | 40 |
| **WandB Project** | `flow-mbpo-formal-training` |
| **Time Limit** | 16h（支援 checkpoint resume 續跑） |

### 2.2 方法矩陣

| Method Key | World Model | Policy | Alg Config |
|------------|-------------|--------|------------|
| `mlpwm_mlppolicy` | MLP | MLP | `pwm_5M_baseline_final` |
| `flowwm_mlppolicy` | Flow | MLP | `pwm_5M_flow_v2_substeps4` |
| `mlpwm_flowpolicy` | MLP | Flow | `pwm_5M_flowpolicy` |
| `flowwm_flowpolicy` | Flow | Flow | `pwm_5M_fullflow` |

### 2.3 任務矩陣

| Task | Suite | Env Config | `num_envs` | `alg.horizon` | Complexity |
|------|-------|-----------|------------|---------------|------------|
| hopper | gym | `gym_hopper_mujoco` | 64 | **7** | low |
| ant | gym | `gym_ant_mujoco` | 64 | 16 (default) | medium |
| anymal | mjlab_proxy | `mjlab_velocity_flat_unitree_go2` | 128 | 16 | medium |
| humanoid | gym | `gym_humanoid_mujoco` | 64 | 16 | medium_high |
| snu_humanoid | mjlab_proxy | `mjlab_velocity_flat_unitree_g1` | 128 | 16 | high |
| leap_left_grasp | mjlab | `mjlab_leap_left_grasp_asymmetric` | 128 | 16 | medium |
| tracking_rough | mjlab | `mjlab_tracking_rough_unitree_g1` | 96 | **1** | medium_high |
| inhand_pen_twirl | mjlab | `mjlab_leap_left_inhand_pen_twirl` | 64 | 16 | high |

### 2.4 Row → Task 對照（每 40 行一組）

| Row 範圍 | Task | 方法排列（每 10 行一個 method × 10 seeds） |
|----------|------|-------------------------------------------|
| 0–39 | hopper | mlpwm→flowwm→mlpwm_flow→fullflow |
| 40–79 | ant | 同上 |
| 80–119 | anymal | 同上 |
| 120–159 | humanoid | 同上 |
| 160–199 | snu_humanoid | 同上 |
| 200–239 | leap_left_grasp | 同上 |
| 240–279 | tracking_rough | 同上 |
| 280–319 | inhand_pen_twirl | 同上 |

---

## 3. WandB 標注系統

### 3.1 自動產生的 Tags

每個 WandB run 由 `train_dflex.py:create_wandb_run()` 自動從 `experiment.*` config 建立 tags：

| Tag 格式 | 用途 |
|----------|------|
| `stage_confirm` | 實驗階段 |
| `suite_gym` / `suite_mjlab` | 任務套件 |
| `task_hopper` | 具體任務 |
| `method_flowwm_mlppolicy` | 方法組合 |
| `gpu_type_H100` | GPU 類型（自動偵測） |
| `hparam_profile_default` | 超參配置 |
| `seed_0` | 隨機種子 |
| `single_task_online` | 實驗類型 |

### 3.2 常用 WandB 篩選

```
# 比較四種方法在 ant 上
tags: task_ant
Group by: config.experiment.method

# 看 Flow WM 所有結果
tags: method_flowwm_mlppolicy OR method_flowwm_flowpolicy

# 看 H100 vs L40S 速度差異
tags: gpu_type_H100 OR gpu_type_L40S
```

---

## 4. 已知問題與修復

| 問題 | 原因 | 修復 | 狀態 |
|------|------|------|------|
| Hopper trajectory 太短 | `alg.horizon=16` 需 17-step, episode 只有 8-12 steps | `alg.horizon=7` (只需 8 steps) | ✅ 已修復 |
| WandB init timeout | 多 run 同時 init, 網路壅塞 | Transient, 自動 retry 或 resume | ⚠️ 偶發 |
| QOS 限制 | 一次只能有一個 320-row array job | 無法繞過, 選擇最佳 GPU 提交 | ℹ️ 限制 |

---

## 5. 監測指令

### 5.1 快速狀態

```bash
# 總覽
sacct -j 4137131 --format=State --noheader | grep -v "^$" | sort | uniq -c

# 看失敗
sacct -j 4137131 --format=JobID%20,State%12,ExitCode --noheader | grep -v "extern\|batch" | grep FAILED

# 排隊
squeue -u $USER --format="%.12i %.9P %.25j %.2t %.10M %.25R" --noheader
```

### 5.2 訓練進度

```bash
# 看特定 row（例如 row 40 = ant mlpwm_mlppolicy seed=0）
tail -20 logs/slurm/single_task_online/confirm/sto_confirm_H100_4137131_40.out
tail -20 logs/slurm/single_task_online/confirm/sto_confirm_H100_4137131_40.err
```

### 5.3 Checkpoint 統計

```bash
find scripts/outputs/single_task_online/confirm/ -name "final_policy.pt" | wc -l   # 完成數
find scripts/outputs/single_task_online/confirm/ -name "best_policy.pt" | wc -l    # 最佳策略
du -sh scripts/outputs/single_task_online/confirm/                                  # 總大小
```

### 5.4 GPU 資源

```bash
sinfo -p ice-gpu -t idle,mix --format="%N %G %T %e %C" | grep -E "(h100|h200|l40s)"
```

---

## 6. Checkpoint Resume 機制

`run_manifest_job.py` 內建自動 resume：重新提交同一 manifest，已完成的 row 會跳過（如果 `final_policy.pt` 存在），未完成的會從 `latest_checkpoint.pt` 續跑。

```bash
# 16h 到期後重新提交（自動 resume）
bash scripts/experiments/single_task_online/submit_manifest_array.sh \
  --manifest scripts/experiments/single_task_online/manifests/confirm_all_4methods_10seeds_20260301.csv \
  --gpu-type H100 --time 16:00:00 \
  --conda-env flow-mbpo \
  --python-bin /storage/ice1/2/9/eliu354/conda_envs/flow-mbpo/bin/python
```

---

## 7. 時間線

| 時間 | 事件 |
|------|------|
| 21:00 | 提交 Smoke（L40S `4136884`）— 32/32 PASS |
| 21:25 | 提交第一次 Confirm（L40S `4136966`）— 9 hopper failures |
| 21:45 | 識別 hopper replay buffer trajectory 長度問題 |
| 21:55 | 加入 `alg.horizon=7` fix, 驗證 smoke pass |
| 22:00 | 提交正式 Confirm（H100 `4137131`）— 320 rows |
| 22:05 | H200 提交被 QOS 限制擋住 |
