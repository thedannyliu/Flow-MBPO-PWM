# PWM 與 Flow-MBPO Pipeline 流程差異對照（公平比較版）

## 1. 對照範圍

本文件比對以下兩份 pseudo code：

- `docs/original_pwm_pipeline_pseudocode.md`
- `docs/flow_mbpo_pipeline_pseudocode.md`

目標是確認兩者在流程上哪些是「同一件事」、哪些是「方法學差異」，以避免不公平比較。

## 2. 先說結論

- 兩者在「訓練骨架」上大致一致：都以 `actor + critic + world model` 為核心，且離線任務訓練階段都走 `update(obs, act, rew, task_ids, finetune_wm)` 的模式。
- 真正主要差異不在高層 loop，而在：
  - world model 動力學學習目標（MSE latent regression vs flow matching）
  - policy/world model 架構選擇（MLP vs Flow）
  - Flow-MBPO 額外明確切出 Stage A（WM pretrain）與 2x2 因子化分支設計。
- 若要公平比較，必須把「除了目標因子外」的資料、初始化、更新預算、checkpoint 規則全部對齊。

## 3. 共同流程（可視為已對齊）

## 3.1 多任務離線訓練主幹

- 皆使用 MT30/MT80 task set，依 task 過濾離線資料進 replay buffer。
- 皆可先載入 world model checkpoint，再做 task-specific policy training。
- 皆在每個訓練 epoch 中執行 actor/critic 更新，並定期在真實環境 eval、保存 best/last。

## 3.2 更新模組組成

- 皆包含三類更新：
  - Actor：基於 imagined rollout 最大化回報（以最小化負回報形式）
  - Critic：TD 型目標（含 TD-lambda 設定）
  - World model：由 replay buffer 取樣 `H+1` slice 做 dynamics + reward loss

## 3.3 單任務 online 路徑

- 兩份文件都保留 `train_dflex.py` 路徑，表示仍有 online 與真實環境互動版本。

## 4. 關鍵差異（流程與方法）

## 4.1 訓練階段切分方式不同

- Original PWM 文件把流程寫成：
  - `single_task_dflex`（online）
  - `multitask_offline_extraction`（offline extraction）
  - `PRETRAIN_WORLD_MODEL` 為可選前置
- Flow-MBPO 文件明確固定成兩階段：
  - Stage A: `pretrain_multitask_wm.py`（只訓練 WM）
  - Stage B: `train_multitask.py`（訓練 policy/critic，可選是否 finetune WM）

影響：
- Flow-MBPO 對 WM 預訓練流程與 checkpoint 產物定義更明確，實驗追蹤較容易。

## 4.2 變體設計方式不同（Flow-MBPO 有明確因子化）

- Original PWM 文件主要描述單一 PWM 設計（外加 paper/code reward 實作差異註記）。
- Flow-MBPO 文件明確給 2x2 因子化變體：
  - Baseline（MLP WM + MLP policy）
  - FlowWM（Flow WM + MLP policy）
  - FlowPolicy（MLP WM + Flow policy）
  - FullFlow（Flow WM + Flow policy）

影響：
- Flow-MBPO 允許拆解「WM 架構」與「policy 架構」的獨立貢獻；Original PWM 文件未以因子化實驗為主軸。

## 4.3 WM dynamics loss 定義不同（最核心）

- Original PWM：`L_dyn` 是 rollout latent 的 MSE（`F_phi(z_t, a_t)` 對 `E_phi(obs_{t+1})`）。
- Flow-MBPO（Flow WM 分支）：`L_dyn` 改為 Flow Matching loss，並引入 ODE 積分設定（integrator、substeps、tau sampling）。

影響：
- 這是演算法本體差異，不只是實作細節；若要公平，必須把其他因子固定。

## 4.4 Actor 架構可能不同

- Original PWM：預設 stochastic MLP actor。
- Flow-MBPO：可切換為 `ActorFlowODE`（flow policy）。

影響：
- policy expressiveness 與訓練動力學改變，屬於第二個主要因子。

## 4.5 WM rollout transition 機制不同（僅 Flow WM 分支）

- Original PWM：一步 latent transition（MLP dynamics）。
- Flow-MBPO FlowWM：`wm.step` 可能透過 flow integrator/substeps 近似連續動力。

影響：
- 每步 rollout 的計算路徑、數值誤差型態、訓練成本都可能不同。

## 4.6 預訓練 checkpoint 使用規約更明確

- Original PWM：描述可由 `load_wm()` 載入 checkpoint（常見是 TD-MPC2 轉入 WM）。
- Flow-MBPO：強調分支需配對 checkpoint 來源（MLP-WM 分支用 MLP-WM ckpt，Flow-WM 分支用 Flow-WM ckpt）。

影響：
- 若 checkpoint 配對錯誤，會混入初始化偏差，導致比較失真。

## 4.7 Reward 表徵註記重點不同

- Original PWM 文件特別標註 paper 與 code 的 reward loss 表徵差異（two-hot vs MSE，且 multi-task 有 `num_bins=101` 情境）。
- Flow-MBPO 文件主體寫法聚焦 MSE reward，未把 paper/code 差異當主軸。

影響：
- 若你要和論文數字直接對齊，需要額外確認 reward head/target encoding 設定是否一致。

## 5. 公平比較必對齊清單

以下項目若不對齊，結果通常不可直接解讀為「Flow 比 MLP 好/差」：

- 相同資料來源與過濾規則：
  - 同一批 `.pt` 檔
  - 同 task id 過濾條件
- 相同訓練預算：
  - `wm_pretrain_iters`、`epochs`、每 epoch update 次數
  - critic iterations、batch size、horizon `H`
- 相同隨機性控制：
  - seeds、eval episodes、eval frequency
- 相同優化超參：
  - actor/critic/WM learning rates、scheduler、gradient clipping
- 相同 `finetune_wm` 策略：
  - 比較組必須同時開啟或同時關閉
- 相同 checkpoint protocol：
  - Baseline/FlowPolicy 用 MLP-WM 起始
  - FlowWM/FullFlow 用 Flow-WM 起始
- 相同環境與回報統計設定：
  - normalization / RMS / reward scaling 一致

## 6. 建議的最小公平實驗矩陣

建議至少做以下 4 組，同一組控制條件：

1. `MLP-WM + MLP-Policy`（baseline）
2. `Flow-WM + MLP-Policy`（只看 WM 因子）
3. `MLP-WM + Flow-Policy`（只看 policy 因子）
4. `Flow-WM + Flow-Policy`（交互效果）

解讀方式：

- (2)-(1)：WM 換成 Flow 的增益
- (3)-(1)：Policy 換成 Flow 的增益
- (4)-(2) 與 (3)-(1) 是否一致：可看交互作用是否存在

## 7. 一句話總結

兩者「流程骨架」大致一致，差異主要集中在 WM/Policy 架構與 dynamics learning objective；只要把資料、初始化、訓練預算與 checkpoint protocol 嚴格對齊，就可以做有意義且公平的比較。
