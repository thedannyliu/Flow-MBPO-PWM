# Flow-MBPO-PWM 遷移到 mjlab 的詳細評估與實驗方案

## 1) 結論先行

- 遷移到 `mjlab` 是可行的，而且對「單任務 online 訓練（`scripts/train_dflex.py`）」最可能帶來迭代加速。
- 目前專案其實已經有一部分遷移基礎（`frank/pwm_env_adapter.py`、`scripts/test_pwm_playground.py`），但**尚未接入正式 Hydra 訓練路徑**，所以還不能直接用在主實驗。
- 最大技術風險是：PWM 目前依賴 `info["obs_before_reset"]` 來避免自動 reset 造成的 replay 汙染；`mjlab` API 預設回傳是 Gymnasium 風格 `(obs, reward, terminated, truncated, info)`，需要額外 adapter 邏輯來對齊。

## 2) 目前 codebase 與環境耦合現況

你目前的主訓練迴圈（`scripts/train_dflex.py` + `src/flow_mbpo_pwm/algorithms/pwm.py`）與環境 API 有以下強耦合：

- `env = instantiate(cfg.env.config, ...)`（Hydra 直接建 env）
- `env` 必須提供：
  - `num_envs`, `num_obs`, `num_actions`, `episode_length`
  - `observation_space.shape`, `action_space.shape`
  - `reset(grads=True/False)`
  - `step(actions) -> (obs, reward, done, info)`
  - `info` 需含 `termination`, `truncation`, `obs_before_reset`, `primal`

關鍵依賴點在 `src/flow_mbpo_pwm/algorithms/pwm.py` 的 `compute_actor_loss()`：  
done 後會用 `obs_before_reset` 寫入 replay，避免把 reset 後觀測誤當成 episode terminal 觀測。

## 3) mjlab 官方能力與你需求的對齊

根據官方文件（Installation / Migration / API）：

- `mjlab` 是以 PyTorch 為核心，支援 GPU 加速與向量化環境。
- 安裝與執行建議走 `uv` workflow（`uv sync`）。
- 遷移指南強調介面接近 Gymnasium / Isaac Lab 路徑，`step` 與 `reset` 介面是標準 RL 風格。
- `ManagerBasedRlEnv.step()` 回傳為 `(obs_dict, reward, terminated, truncated, info)`，`reset()` 回傳 `(obs_dict, info)`。
- 分散式訓練建議是 `torch.distributed` 一卡一 process 的 DDP 架構。

這和你現有 PWM 訓練邏輯是「可對接但非零成本」：

- 可直接對齊：`terminated/truncated`、向量化 `num_envs`、PyTorch tensor 路徑。
- 需要補 adapter：obs dict flatten、`reset(grads=True)`、`obs_before_reset` 兼容層。

## 4) 必改項目（以最小可行遷移為目標）

## 4.1 新增 mjlab 專用 adapter（必要）

建議新增：

- `src/flow_mbpo_pwm/envs/mjlab_pwm_adapter.py`

責任：

- 封裝 `mjlab` env 為 PWM 介面：
  - 輸出 `num_envs / num_obs / num_actions / episode_length`
  - `reset(grads=True)` 回傳 cache obs（不重置）
  - `step()` 將 Gymnasium 輸出轉為 PWM 期望格式
- 將 `obs_dict` 固定 flatten 成單一 state 向量（先做 state-only，避免初期複雜度）
- 統一 `done = terminated | truncated`

## 4.2 新增 Hydra env config（必要）

建議新增：

- `scripts/cfg/env/mjlab_<task>.yaml`（例如先做一個 locomotion task）

內容包含：

- `_target_`: 指向 `create_mjlab_pwm_env(...)` 工廠
- `task_id` / `num_envs` / `device` / `episode_length` / `action_repeat` 等

## 4.3 訓練入口兼容（必要）

`scripts/train_dflex.py` 可先不改檔名，但建議：

- 新增註解或 alias script（例如 `scripts/train_online.py`），避免名稱誤導（已不只 DFlex）

## 4.4 評估腳本兼容（建議）

目前 `scripts/eval/eval_pwm.py` 與 `scripts/evaluate_policy.py` 直接 hardcode `dflex_*`。  
建議至少新增：

- `scripts/eval/eval_pwm_mjlab.py`（或重構既有 create_env 支援 `mjlab_*`）

否則訓練後評估流程會斷。

## 4.5 提交腳本與實驗管理（建議）

目前大量 `scripts/*submit*.sh` 都是 `env=dflex_*`。  
建議新增 `scripts/mjlab/` 子目錄放新的 smoke / submit 腳本，先不要覆蓋原 dflex 腳本。

## 5) 主要風險與對策

## 5.1 最高風險：`obs_before_reset` 語義不一致

風險：

- 若 adapter 無法提供正確 terminal 前觀測，replay 可能混入 reset 後觀測，影響 WM 訓練品質與比較公平性。

對策：

1. 先檢查 `mjlab info` 是否已有 terminal obs 欄位（若有，直接映射）。
2. 若無，實作「正確模式 adapter」：
   - 在 env step 流程中顯式取得 reset 前觀測（可能需 wrapper/subclass）。
3. 在 smoke test 增加 assertion：
   - done 時 `obs_before_reset` 與 reset 後 `obs` 必須可區分（非全相等）。

## 5.2 相依套件衝突

風險：

- 你目前 `environment.yaml` 偏向舊環境（Torch 2.3 / CUDA 11.8 + dFlex）；
- `mjlab` 生態（JAX/MJX/warp/mujoco 新版）可能衝突。

對策：

- 建立**獨立環境**（例如 `flow-mbpo-mjlab`），不要直接覆寫現有 `pwm` 環境。
- 保留 dflex 舊流程可回退。

## 5.3 速度增益可能不線性

風險：

- 即使 simulator 更快，整體 wall-clock 可能仍受 WM 更新與 Python 邏輯影響。

對策：

- 遷移後第一步先做 profiling：
  - `env.step` 時間占比
  - `world model training` 時間占比
  - 端到端 `fps`（你現有 log 已有）
- 若 env 佔比 < 30%，再加速模擬器的收益會有限，需同步優化訓練程式。

## 6) 建議遷移路線（分三階段）

## Phase A: 可跑通（1-2 天）

目標：跑通 single-task smoke，不追求最終公平比較。

- 建立 `mjlab_pwm_adapter.py`
- 新增一個 `mjlab` env config
- 用 `max_epochs=100~300`、`num_envs=32~64` 跑 baseline smoke
- 驗證：
  - 無 NaN
  - replay 有資料（`buffer.num_eps > 0`）
  - eval 能完成

## Phase B: 正確性對齊（2-4 天）

目標：補齊 `obs_before_reset` 語義與評估鏈。

- 完成 done/reset 邊界觀測的正確處理
- 新增 `mjlab` 版 eval
- 用 2-3 seeds 跑短實驗，確認 learning curve 穩定

## Phase C: 正式實驗（1-2 週）

目標：在 `mjlab` 上做公平 baseline vs flow 比較。

- 固定同一 task、同一 seed set、同一 budget
- 先做 4 組最小矩陣：
  1. MLP-WM + MLP-Policy
  2. Flow-WM + MLP-Policy
  3. MLP-WM + Flow-Policy
  4. Flow-WM + Flow-Policy
- 報告同時給：
  - 最終 reward / success
  - 收斂速度（wall-clock 與 env steps）
  - FPS 與成本（GPU 小時）

## 7) 在 mjlab 上如何進行實驗（建議 workflow）

## 7.1 環境準備

- 使用官方建議的 `uv` 或獨立 conda+pip 環境
- 先跑 `mjlab` 官方 quick-start / play 例子，確認 GPU 與 rendering backend 正常

## 7.2 專案內第一個 smoke command（建議形態）

```bash
python scripts/train_dflex.py \
  env=mjlab_<task> \
  alg=pwm_5M_baseline_final \
  general.seed=0 \
  alg.max_epochs=300 \
  env.config.num_envs=64 \
  general.run_wandb=true
```

接著把 `alg` 切到 flow 版本跑同條件 smoke，比對：

- 每秒步數（FPS）
- 同 epochs 的 reward 曲線
- 是否出現 done/reset 邊界異常

## 7.3 正式對照實驗規範

為了讓結果可解釋：

- 除目標因子外全部固定（seed、epochs、horizon、batch、lr、eval_freq）
- baseline 與 flow 使用同一批 task/initialization protocol
- 若你同時保留 dflex 與 mjlab 實驗：
  - 不要跨 simulator 直接比較絕對 reward
  - 只比較「同 simulator 內」方法差異

## 8) 這次評估後的執行建議

建議你直接做：

1. 先走 **Phase A**，目標 48 小時內拿到第一條可用 learning curve。  
2. 確認 `obs_before_reset` 後再進 **Phase B**，否則不要大規模丟資源。  
3. 完成 B 後再啟動正式 seed 擴展與 ablation。  

---

## 參考資料（官方）

- mjlab GitHub: https://github.com/mujocolab/mjlab
- mjlab docs: https://mujocolab.github.io/mjlab/
- Installation Guide: https://mujocolab.github.io/mjlab/source/installation_guide.html
- Migration Guide: https://mujocolab.github.io/mjlab/source/migration_guide.html
- Environment API (`ManagerBasedRlEnv`): https://mujocolab.github.io/mjlab/source/api/envs.html#mjlab.envs.manager_based_rl_env.ManagerBasedRlEnv
- Distributed Training Guide: https://mujocolab.github.io/mjlab/source/distributed_training_guide.html
- PyPI (版本與發布時間): https://pypi.org/project/mjlab/
