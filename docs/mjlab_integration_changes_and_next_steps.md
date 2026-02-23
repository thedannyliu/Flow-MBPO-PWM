# mjlab 整合改動與下一步（Flow-MBPO-PWM）

> 日期：2026-02-19  
> 範圍：先完成 codebase 整合（不處理環境安裝），可進入 smoke test。

---

## 1. 本次完成內容（Codebase Integration）

## 1.1 新增 mjlab 正式 adapter（主訓練路徑可用）

- 新增 `src/flow_mbpo_pwm/envs/mjlab_pwm_adapter.py`
- 新增 `src/flow_mbpo_pwm/envs/__init__.py`

重點能力：

- 將 mjlab/Gymnasium 風格介面轉為 PWM 需要的介面：
  - `reset(grads=True/False)`
  - `step(actions) -> (obs, reward, done, info)`
  - `info` 保證輸出：`termination`, `truncation`, `obs_before_reset`, `primal`
- `obs_dict -> flat state` 支援：
  - 優先 `obs_key`（預設 `state`）
  - 候選 key（`state/policy/observation/obs`）
  - 若都沒有，自動按 key 排序後拼接
- `done = terminated | truncated` 對齊
- `num_envs / num_obs / num_actions / episode_length` 對齊
- 提供 `get_diagnostics()` 回傳 adapter 風險診斷指標（可上 W&B）

## 1.2 Hydra env config 與訓練入口整合

- 新增 `scripts/cfg/env/mjlab_walker.yaml`
  - `_target_ = flow_mbpo_pwm.envs.mjlab_pwm_adapter.create_mjlab_pwm_env`
  - 含 `task_id / num_envs / device / episode_length / action_repeat / obs_key`
  - 含 done/reset 安全參數（strict/expect_auto_reset/fallback warning）
- 新增 `scripts/train_online.py`
  - 作為 `train_dflex.py` 別名入口，避免名稱只綁 dFlex 的誤導
  - 不破壞既有 `train_dflex.py` 流程

## 1.3 Profiling 與 W&B 全量記錄

- 修改 `src/flow_mbpo_pwm/utils/time_report.py`
  - 新增 `get_time(name)` 與 `get_timer_totals()`
- 修改 `src/flow_mbpo_pwm/algorithms/pwm.py`
  - 新增 timer：`env step`
  - `compute_actor_loss()` 內對 `env.step` 進行 profile 計時
  - 每個 epoch 計算並輸出 profiling metrics：
    - `profile/*_seconds`
    - `profile/*_pct`
    - `profile/epoch_wall_seconds`
  - 每個 epoch 都上傳 W&B（不再每 50 epoch 才 log）
  - 上傳 adapter diagnostics（`env/*`）
  - 訓練結束上傳總計時（`profile/total_*_seconds`）
  - warmup 階段（buffer 尚未有 episode）也會記錄 W&B warmup 指標

## 1.4 評估路徑相容 mjlab

- 修改 `scripts/eval/eval_pwm.py`
  - 移除對 dflex 的硬編碼建立流程
  - 改為直接使用 run 的 Hydra config 實例化環境（含 mjlab）
  - 支援 `create_mjlab_pwm_env` target 偵測
- 修改 `scripts/evaluate_policy.py`
  - 新增由 `scripts/cfg/env/<env>.yaml` 建環境的路徑
  - 不再只支援 `dflex_ant`

## 1.5 smoke scripts

- 新增 `scripts/mjlab/smoke_train.sh`
  - 主訓練路徑 smoke command（Hydra + W&B）
- 新增 `scripts/mjlab/smoke_adapter_semantics.py`
  - 不依賴 mjlab 安裝，直接驗證 `obs_before_reset` done/reset 語義

---

## 2. 文件風險對策對照（已落地）

對照 `docs/mjlab_migration_assessment.md`：

## 2.1 最高風險：`obs_before_reset` 語義不一致

已處理：

- adapter 先檢查 `info` 內 terminal obs 欄位（`obs_before_reset/final_observation/...`）
- 若缺失，fallback 用 `pre_step_obs` 填入 done transition
- 可設定 `fail_on_missing_terminal_obs=true` 直接 fail fast
- 可設定 `strict_terminal_obs=true + expect_auto_reset=true` 在 smoke 階段做污染檢查
- `get_diagnostics()` 提供以下追蹤：
  - `terminal_obs_from_info`
  - `terminal_obs_from_fallback`
  - `terminal_obs_equal_next_obs`
  - 對應 ratio 指標

## 2.2 速度增益不線性（需 profiling）

已處理：

- 訓練迴圈提供 section-level profiling（actor/critic/wm/env step 等）
- 每 epoch 上傳 W&B，可直接看占比曲線（`profile/*_pct`）
- 結束時另上傳總計時（`profile/total_*_seconds`）

---

## 3. Smoke Test 指南

## 3.1 先跑語義 smoke（不依賴 mjlab 安裝）

```bash
python scripts/mjlab/smoke_adapter_semantics.py
```

預期輸出：

- `MJLabPWMAdapter smoke semantics: PASS`

## 3.2 再跑主訓練路徑 smoke（需已安裝 mjlab）

```bash
WANDB_PROJECT=flow-mbpo-mjlab-smoke \
scripts/mjlab/smoke_train.sh
```

可調參數（環境變數）：

- `SEED`
- `NUM_ENVS`
- `MAX_EPOCHS`
- `ENV_NAME`（預設 `mjlab_walker`）
- `ALG_NAME`（預設 `pwm_5M_baseline_final`）

---

## 4. W&B 指標總覽（這次新增）

## 4.1 Profiling 指標（每 epoch）

- `profile/epoch_wall_seconds`
- `profile/compute_actor_loss_seconds`, `profile/compute_actor_loss_pct`
- `profile/forward_simulation_seconds`, `profile/forward_simulation_pct`
- `profile/env_step_seconds`, `profile/env_step_pct`
- `profile/backward_simulation_seconds`, `profile/backward_simulation_pct`
- `profile/actor_training_seconds`, `profile/actor_training_pct`
- `profile/prepare_critic_dataset_seconds`, `profile/prepare_critic_dataset_pct`
- `profile/critic_training_seconds`, `profile/critic_training_pct`
- `profile/world_model_training_seconds`, `profile/world_model_training_pct`

## 4.2 總計時（訓練結束）

- `profile/training_time_total_seconds`
- `profile/training_time_avg_epoch_seconds`
- `profile/training_time_median_epoch_seconds`
- `profile/total_*_seconds`

## 4.3 Adapter 風險監控（每 epoch）

- `env/step_calls`
- `env/done_events`
- `env/done_terminated`
- `env/done_truncated`
- `env/terminal_obs_from_info`
- `env/terminal_obs_from_fallback`
- `env/terminal_obs_equal_next_obs`
- `env/terminal_obs_info_ratio`
- `env/terminal_obs_fallback_ratio`
- `env/terminal_obs_equal_next_obs_ratio`

---

## 5. 下一步（Phase B / C）

## 5.1 Phase B：正確性對齊（建議先做）

1. 在真實 mjlab 任務上跑 `MAX_EPOCHS=100~300` smoke（baseline + flow 各一）
2. 觀察 `env/terminal_obs_fallback_ratio`
   - 若高，優先在 mjlab wrapper 層提供 `final_observation`
3. 抽樣檢查 done transition：
   - `obs_before_reset` 與 `next_obs` 差異分佈
4. 補一個 `scripts/eval/eval_pwm_mjlab.py`（如果你們偏好分離式 eval）

## 5.2 Phase C：正式實驗準備

1. 固定 task/seed/budget 跑 E1-E4（先 3-5 seeds）
2. 擴到 10 seeds confirmatory
3. 使用 W&B dashboard 分析：
   - `profile/env_step_pct` vs `profile/world_model_training_pct`
   - `env/terminal_obs_*` 指標是否穩定
4. 再進入 ablation（WM uncertainty、demo leverage、reward pipeline）

---

## 6. 已知限制（本次刻意不做）

- 未處理套件安裝與版本衝突（依需求，環境先不管）
- 未導入 DDP / 分散式訓練
- 未改動既有大批 submit 腳本（僅新增 `scripts/mjlab/` smoke 腳本）

