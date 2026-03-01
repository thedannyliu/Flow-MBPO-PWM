# 正式訓練（Confirm）監測紀錄

> **最後更新**：2026-03-01 06:15 EST  
> **分支**：`dev/unified-pwm-resume-20260223` (commit `e1e98ba`)  
> **WandB（正式）**：[`flow-mbpo-formal-training`](https://wandb.ai/danny010324/flow-mbpo-formal-training)  
> **WandB（Smoke）**：[`flow-mbpo-formal-experiments`](https://wandb.ai/danny010324/flow-mbpo-formal-experiments)

---

## 1. 目前 Job 狀態

| GPU | Job ID | 狀態 | 已完成 | 執行中 | 失敗 | 待排 | 備註 |
|-----|--------|------|--------|--------|------|------|------|
| **L40S** | `4137162` | 🔄 RUNNING | 138 | 24 | **0** | 158 | 主要 Job，%16 concurrency |

> [!NOTE]
> QOS 限制每位使用者只能同時提交一個 320-row array job（`QOSMaxSubmitJobPerUserLimit`），  
> H100/H200 無法同時提交。L40S 是目前的唯一 Job。

---

## 2. 任務完成統計（06:15 EST 快照）

| Task | Suite | DONE | RUN | FAIL | PEND | Wall-clock/row |
|------|-------|------|-----|------|------|----------------|
| **hopper** | gym | **40** ✅ | 0 | 0 | 0 | ~1h 20m |
| **ant** | gym | 6 | 8 | 0 | 26 | ~2h 33m |
| anymal | mjlab_proxy | 0 | 0 | 0 | 40 | 待排隊 |
| humanoid | gym | 0 | 0 | 0 | 40 | 待排隊 |
| snu_humanoid | mjlab_proxy | 0 | 0 | 0 | 40 | 待排隊 |
| leap_left_grasp | mjlab | 0 | 0 | 0 | 40 | 待排隊 |
| tracking_rough | mjlab | 0 | 0 | 0 | 40 | 待排隊 |
| inhand_pen_twirl | mjlab | 0 | 0 | 0 | 40 | 待排隊 |
| **TOTAL** | | **46** | **8** | **0** | **266** | |

> 注：`sacct` 僅顯示已排程過的 rows，PEND 數字為估算。

---

## 3. 修復歷程（重要錯誤一覽）

| 錯誤 | 原因 | 修復 | Commit |
|------|------|------|--------|
| Hopper replay buffer fail | `alg.horizon=16` 需 17-step trajectory，但 episode 僅 8-12 steps | `alg.horizon=7`（加到 `build_manifest.py:TASK_EXTRA_OVERRIDES`） | `5ec8b4a` |
| WandB notes 解析錯誤 | GPU metadata 在 `hydra_quote()` 之外，導致空格破壞 override | 移入 `hydra_quote()` 內 | `dea298b` |
| WandB pydantic tag 64 字元超限 | 整個 tag list 當一個字串傳入 pydantic | 改在 `train_dflex.py:create_wandb_run()` 自動從 `experiment.*` 建 tags | `dea298b` |
| WandB init timeout（偶發） | 多 run 同時 init，網路壅塞 | Transient — 下次 resume 可用 checkpoint 繼續 | — |

---

## 4. WandB 標注系統

每個 run 自動產生（`train_dflex.py:create_wandb_run()`）：

```
Tags:    stage_confirm, suite_gym, task_hopper, method_mlpwm_mlppolicy,
         gpu_type_L40S, hparam_profile_default, seed_0,
         single_task_online, online_rl, from_scratch
Group:   single_task_online_confirm_gym
Name:    confirm_gym_hopper_mlpwm_mlppolicy_s0_default
Notes:   ...task=hopper, method=mlpwm_mlppolicy... | GPU=L40S node=... job=4137162
Config:  experiment.{stage, suite, task, method, gpu_type, slurm_job_id, slurm_node}
         runtime.slurm.{job_id, array_task_id, node_name, partition}
```

### WandB 篩選範例

```
# 比較 4 種方法在 ant 上的表現
tags: task_ant
Group by: config.experiment.method

# 所有 Flow WM 實驗
tags: method_flowwm_mlppolicy OR method_flowwm_flowpolicy

# 只看特定 seed 的穩定性
tags: seed_3

# 交叉（e.g., hopper + MLP policy）
tags: task_hopper AND (method_mlpwm_mlppolicy OR method_flowwm_mlppolicy)
```

---

## 5. 監測指令

```bash
# 快速總覽
sacct -j 4137162 --format=State --noheader | grep -v "^$" | sort | uniq -c

# 查看失敗
sacct -j 4137162 --format=JobID%20,State%12,ExitCode,Elapsed%12 --noheader \
  | grep -v "extern\|batch" | grep FAILED

# 排隊查看
squeue -u $USER --format="%.12i %.9P %.25j %.2t %.10M %.6D %.25R" --noheader

# 訓練 log（Row 47 例，snu_humanoid）
tail -5 logs/slurm/single_task_online/confirm/sto_confirm_L40S_4137162_47.out

# Checkpoint 數量
find scripts/outputs/single_task_online/confirm/ -name "final_policy.pt" | wc -l
du -sh scripts/outputs/single_task_online/confirm/

# GPU 資源
sinfo -p ice-gpu -t idle,mix --format="%N %G %T %C" | grep -E "(l40s|h100|h200)"
```

---

## 6. Checkpoint Resume 機制

每個 row 的 `run_manifest_job.py` 在啟動時自動檢查：

1. `logs/latest_checkpoint.pt` → 從此續跑（最優先）
2. `logs/final_policy.pt` → 已完成，跳過
3. 都沒有 → 從頭訓練

**16h 到期後的重新提交**（自動 resume）：

```bash
bash scripts/experiments/single_task_online/submit_manifest_array.sh \
  --manifest scripts/experiments/single_task_online/manifests/confirm_all_4methods_10seeds_20260301.csv \
  --gpu-type L40S --time 16:00:00 \
  --conda-env flow-mbpo \
  --python-bin /storage/ice1/2/9/eliu354/conda_envs/flow-mbpo/bin/python
```

> 已完成的 row 會因有 `final_policy.pt` 而快速跳過，不會重新訓練。

---

## 7. 時間線

| 時間（EST） | 事件 |
|-------------|------|
| 02/28 21:00 | 提交 Smoke（L40S `4136884`）— 32/32 PASS |
| 02/28 21:25 | 提交第一次 Confirm（L40S `4136966`）— 9 hopper 失敗 |
| 02/28 21:45 | 識別 replay buffer trajectory 問題 |
| 02/28 21:55 | 加入 `alg.horizon=7` fix，commit `5ec8b4a` |
| 02/28 22:06 | 提交正式 Confirm（L40S `4137162`）— 320 rows |
| 02/28 22:22+ | 前 16 rows 開始 RUNNING（hopper rows 0-15） |
| 03/01 00:36 | hopper 全部 40 rows COMPLETED ✅ |
| 03/01 01:30 | ant rows 開始執行 |
| 03/01 06:15 | 138 rows COMPLETE, 24 RUNNING, 0 FAILED |

---

## 8. 後續行動

| 行動 | 時間點 | 指令 |
|------|--------|------|
| 16h job 到期，重新提交 resume | ~03/01 14:00 EST | 同上 `submit_manifest_array.sh` |
| 採集 WandB 結果 | 所有 320 rows 完成後 | `wandb api` or download csv |
| 統計分析 | 320 rows 全完成 | 見 `results/` 目錄 |
