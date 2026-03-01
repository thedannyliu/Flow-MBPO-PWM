# 正式訓練實驗監測紀錄

> **最後更新**：2026-02-28 21:32 EST  
> **分支**：`dev/unified-pwm-resume-20260223` (commit `dea298b`)  
> **叢集**：PACE-ICE (`ice-gpu` partition)  
> **WandB Project**：[`flow-mbpo-formal-experiments`](https://wandb.ai/danny010324/flow-mbpo-formal-experiments)

---

## 1. 當前 Smoke 實驗狀態

| GPU 類型 | Slurm Job ID | 狀態 | 已完成 | 執行中 | 待排隊 |
|----------|-------------|------|--------|--------|--------|
| **H100** | `4136881` | 🕐 PENDING | 0 | 0 | 32 |
| **H200** | `4136882` | 🕐 PENDING | 0 | 0 | 32 |
| **L40S** | `4136884` | 🔄 RUNNING | 4 | 6+ | ~22 |

### Manifest 設定

- **檔案**：`manifests/smoke_formal_all_4methods_20260301.csv`
- **總行數**：32（8 任務 × 4 方法 × seed=0）
- **Epochs**：200
- **WandB Project**：`flow-mbpo-formal-experiments`（統一專案）

---

## 2. WandB 實驗標注

每個 WandB run 自動包含以下資訊：

### 2.1 WandB Tags（自動產生）

由 `train_dflex.py` 根據 `experiment.*` config 自動建立：

| Tag 格式 | 範例 | 用途 |
|----------|------|------|
| `stage_<X>` | `stage_smoke` | 實驗階段 |
| `suite_<X>` | `suite_gym`, `suite_mjlab` | 任務套件 |
| `task_<X>` | `task_hopper`, `task_anymal` | 具體任務 |
| `method_<X>` | `method_flowwm_mlppolicy` | 方法組合 |
| `gpu_type_<X>` | `gpu_type_L40S` | GPU 類型 |
| `hparam_profile_<X>` | `hparam_profile_default` | 超參配置 |
| `seed_<N>` | `seed_0` | 隨機種子 |
| `single_task_online` | - | 實驗類型 |
| `online_rl` | - | 訓練方式 |
| `from_scratch` | - | 從頭訓練 |

### 2.2 WandB Config（自動記錄）

```yaml
experiment:
  run_key: "smoke_gym_hopper_mlpwm_mlppolicy_s0_default"
  stage: "smoke"
  suite: "gym"
  task: "hopper"
  method: "mlpwm_mlppolicy"
  hparam_profile: "default"
  gpu_type: "L40S"           # ← 自動偵測
  slurm_job_id: "4136884"
  slurm_node: "atl1-1-03-004-21-0"

runtime.slurm:
  job_id: "..."
  array_job_id: "..."
  array_task_id: "0"
  node_name: "atl1-1-03-004-21-0"
  cluster_name: "..."
  partition: "ice-gpu"
```

### 2.3 WandB Notes（自動附加 GPU 資訊）

`Single-task online RL from scratch on PACE-ICE. task=hopper, method=mlpwm_mlppolicy, profile=default | GPU=L40S node=atl1-1-03-004-21-0 job=4136884`

### 2.4 WandB Run Name

格式：`<stage>_<suite>_<task>_<method>_s<seed>_<profile>`  
範例：`smoke_gym_hopper_mlpwm_mlppolicy_s0_default`

### 2.5 WandB Group

格式：`single_task_online_<stage>_<suite>`  
範例：`single_task_online_smoke_gym`

---

## 3. 在 WandB 中如何篩選實驗

### 按方法篩選
```
tags: method_flowwm_mlppolicy
```

### 按任務篩選
```
tags: task_hopper
```

### 按 GPU 類型篩選
```
tags: gpu_type_H100
```

### 交叉篩選（例：H100 上的 Flow WM + MLP Policy 在 humanoid 上）
```
tags: gpu_type_H100 AND method_flowwm_mlppolicy AND task_humanoid
```

---

## 4. 監測指令

```bash
# 查看所有 job 狀態
squeue -u $USER --format="%.12i %.9P %.25j %.2t %.10M %.6D %.25R"

# 統計各 job 完成數
for JOB in 4136881 4136882 4136884; do
  echo "=== Job $JOB ==="
  sacct -j $JOB --format=State --noheader | grep -v "^$" | sort | uniq -c
done

# 查看特定 task 的 log
cat logs/slurm/single_task_online/smoke/sto_smoke_L40S_4136884_<TASK_ID>.out
cat logs/slurm/single_task_online/smoke/sto_smoke_L40S_4136884_<TASK_ID>.err
```

---

## 5. Row → Task × Method 對照表

| Row | Task | Method |
|-----|------|--------|
| 0–3 | hopper | mlpwm→flowwm→mlpwm_flow→fullflow |
| 4–7 | ant | 同上 |
| 8–11 | anymal | 同上 |
| 12–15 | humanoid | 同上 |
| 16–19 | snu_humanoid | 同上 |
| 20–23 | leap_left_grasp | 同上 |
| 24–27 | tracking_rough | 同上 |
| 28–31 | inhand_pen_twirl | 同上 |

---

## 6. 已知問題與修復

| 問題 | 原因 | 修復 | Commit |
|------|------|------|--------|
| Hydra notes 解析錯誤 | GPU 資訊在 quotes 外 | 移入 `hydra_quote()` 內 | `dea298b` |
| Hydra tags 列表歧義 | `a,b,c` 需 `[a,b,c]` | 移至 `train_dflex.py` 自動建立 | `dea298b` |
| WandB pydantic tag 64 字元限制 | 整個列表當成單一 tag | 從 `experiment.*` 拆分建立 | `dea298b` |
| 舊 checkpoint 導致意外 resume | 前次失敗留下 outputs | 清除 `scripts/outputs/` | 手動 |
