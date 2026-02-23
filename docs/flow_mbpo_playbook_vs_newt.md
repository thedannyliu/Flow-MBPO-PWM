# Flow-MBPO 訓練技巧整合與 Newt 公平比較指南

> 目的：把你指定的 4 篇論文轉成「可直接落地的技巧、實驗方向、與公平比較規範」，同時考慮你們正遷移到 `mjlab` 的現況。

---

## 1) 研究策略（先定義勝利條件）

你現在的實際目標可拆成三層：

1. **內部勝利**：Flow-MBPO 穩定且可重複地贏過你們自己的 PWM baseline。  
2. **機制證據**：不是只看 reward，要能解釋「為什麼 Flow 在哪些設定有效」。  
3. **外部競爭**：和 Newt（`Learning Massively Multitask World Models for Continuous Control`）做可辯護的公平比較。  

---

## 2) 論文萃取的可落地技巧

## 2.1 Policy 端（FPO / FPO++）可直接借鑑

| 技巧 | 來自 | 對你們的實作建議 | 優先級 |
|---|---|---|---|
| `per-sample ratio`（非 per-action） | FPO++ 2602.02481 | 若你們做 flow policy 的 policy-gradient 版本，ratio clipping 改為 sample-level | 高 |
| `ASPO` 非對稱 trust region | FPO++ | 正 advantage 用 PPO clipping；負 advantage 用更保守懲罰項，抑制崩潰 | 中高 |
| `zero-sampling` 測試策略 | FPO++ | 訓練用隨機 noise，eval 用 `epsilon=0`，降低部署方差與延遲 | 高 |
| CFM/ratio clamp + gradient-preserving clamp | FPO++ Appendix | 針對 flow loss 大幅震盪時，加入 clamp 防 NaN 與爆梯度 | 中高 |
| `Nmc`（tau, epsilon）抽樣數作為效率旋鈕 | FPO 2507.21053 | 先固定小 Nmc（如 4/8）做效率版，再做 Nmc ablation | 中 |
| 目標參數化（epsilon-target vs velocity-target） | FPO | 先做 `epsilon-target` 與 `velocity-target` 對照，觀察尺度穩定性 | 中 |

關鍵 insight：

- Flow policy 的核心優勢常出現在 **under-conditioned / 多模態 action** 場景。  
- 若只在 fully-conditioned、低不確定性任務上比較，容易看不出差距。

## 2.2 World Model 端（Newt + RWM-O）可直接借鑑

| 技巧 | 來自 | 對你們的實作建議 | 優先級 |
|---|---|---|---|
| Demo 驅動 pretrain（全部模組一起 pretrain） | Newt 2511.19584 | 不只 pretrain encoder；連 dynamics/reward/policy prior 一起 pretrain | 高 |
| Demo 的 4 重利用（pretrain / constrained planning / oversample / BC regularization） | Newt | 你們即使無 planner，也可保留 3 項：pretrain + oversample + actor BC regularization | 高 |
| Demo 與 online buffer 50/50 取樣 | Newt | 直接加入 replay sampling ratio ablation（50/50、30/70） | 高 |
| Reward/Value 離散化（bins + CE）+ log-space value | Newt | 你們目前 reward 多為 MSE，可把 reward head 當正式 ablation 軸 | 中高 |
| Per-task gamma（依 episode length） | Newt | 多任務時避免單一 gamma 對短/長回合都不合適 | 中高 |
| Ensemble epistemic uncertainty + reward penalty | RWM-O 2504.16680 | 在 WM rollout reward 上加 `r_tilde = r - lambda * u`，抑制 hallucination exploitation | 高 |
| `lambda` 懲罰係數掃描（小會投機、大會太保守） | RWM-O | 固定掃 `lambda ∈ {0, 0.5, 1.0, 2.0}`，常見最優在中間 | 高 |

關鍵 insight：

- 你的判斷是對的：**WM 訓練細節通常比 policy 架構更決定成敗**。  
- Flow policy 可以「work」，但若 WM 不穩，actor 仍會吃到壞梯度。

## 2.3 對你們專案最重要的「先做再說」項目

| 優先順序 | 項目 | 原因 |
|---|---|---|
| P0 | `WM uncertainty penalty` + `lambda` 掃描 | 直接抑制 model exploitation，通常立刻提升穩定性 |
| P0 | `demo oversampling` + `actor BC regularization` | 最低工程成本、對早期學習最有效 |
| P1 | reward head：MSE vs two-hot/CE | 避免把 reward pipeline 差異誤判成 Flow 效果 |
| P1 | flow policy 的 per-sample ratio / zero-sampling | 提升 flow policy 可訓練性與部署穩定性 |
| P2 | ASPO、CFM clamp、target parameterization | 進一步榨取上限與穩定性 |

---

## 3) 建議新增的實驗表（補你現有 spec）

你已經有主表 (`E1-E4`) 很好，這裡補 3 張「WM 導向」表：

## Table-WM1：不確定性與保守性

| Exp ID | 設定 | 目的 | 預期 |
|---|---|---|---|
| WM1-0 | baseline（無 uncertainty） | 參考組 | 易出現 hallucination exploitation |
| WM1-1 | ensemble uncertainty + `lambda=0.5` | 輕保守 | 有機會提升泛化 |
| WM1-2 | ensemble uncertainty + `lambda=1.0` | 中保守 | 常見最佳折衷 |
| WM1-3 | ensemble uncertainty + `lambda=2.0` | 強保守 | 可能過度保守、回報下降 |

## Table-WM2：Demo 利用策略拆解

| Exp ID | pretrain | oversample | BC reg | 目的 |
|---|---:|---:|---:|---|
| WM2-0 | ✗ | ✗ | ✗ | 純 online 參考 |
| WM2-1 | ✓ | ✗ | ✗ | 只看 pretrain 效益 |
| WM2-2 | ✓ | ✓ | ✗ | 加入資料分佈穩定化 |
| WM2-3 | ✓ | ✓ | ✓ | Newt-style full leverage |

## Table-WM3：Reward pipeline 對齊

| Exp ID | reward head | regression space | 目的 |
|---|---|---|---|
| WM3-0 | scalar MSE | raw reward | 目前實作 |
| WM3-1 | scalar MSE | symlog/normalized | 低成本穩定化 |
| WM3-2 | two-hot + CE | binned/log-space | 對齊 Newt/PWM 類設計 |

---

## 4) 如何和 Newt 做「公平且可辯護」比較

## 4.1 先講底線

**若任務集合不同（MMBench vs mjlab task suite），不能宣稱「全面超越 Newt」。**  
只能說：

- 在你定義的 mjlab benchmark 上超越 Newt-style baseline，或  
- 在 MMBench 子集上達到/超過 Newt 報告值。  

## 4.2 三層比較協議（建議同時做）

## Track A：嚴格可比（最硬）

- 平台：直接用 Newt 官方 code + MMBench。  
- 設定：同 observation mode（先 state-only）、同步數 budget、同 seeds 數、同 aggregation。  
- 目標：對齊 Newt 論文/CSV 的 normalized score。  

這是唯一可以正面說「比 Newt 好」的主戰場。

## Track B：mjlab 內部競賽（工程主線）

- 平台：全部在 `mjlab`。  
- baseline：`MLP-WM + MLP-policy`（你們 PWM 版本）+ `Newt-style trick ablations`。  
- 目標：先把 Flow-MBPO 在 mjlab 做到穩定贏 baseline。  

這條線解決「你們現在迭代速度慢」的現實問題。

## Track C：跨平台轉移（有研究價值）

- 先在 MMBench/TD-MPC2 類資料 pretrain（含 demo），再 transfer 到 mjlab 任務。  
- 比較「from-scratch vs pretrain+finetune」在樣本效率與 wall-clock 的差距。  

這條線可以把你們論文故事連到「generalist pretrain -> target adaptation」。

## 4.3 比較時必須鎖死的欄位

| 類別 | 必對齊項目 |
|---|---|
| Data | 同 demo 來源、同過濾規則、同 train/eval split |
| Budget | 同 env steps、同 wall-clock 截止、同更新頻率 |
| Model | 參數量級（例如 5M/20M）與 rollout budget（H、K、Kpol） |
| Eval | 同 seeds、同 eval episodes、同 deterministic/stochastic protocol |
| 統計 | 同 primary endpoint（AUC + final return）與同 bootstrap/FDR |

---

## 5) 考慮 mjlab 的可執行落地方案

## 5.1 最小可行比較（4 週內）

1. `mjlab` 跑通 `E1-E4` 在 3 個代表任務（低/中/高維）  
2. 加上 `WM1`（uncertainty penalty）  
3. 加上 `WM2`（demo leverage）  
4. 每組 5 seeds 做 pilot；保留前二名設定進 10 seeds confirmatory

## 5.2 你現在就可以用的「競賽式」指標板

| 指標 | 定義 | 用途 |
|---|---|---|
| `Score@EqualSteps` | 固定 env steps 的 normalized AUC | 比 sample efficiency |
| `Score@EqualTime` | 固定 wall-clock 的 final return | 比實務迭代效率 |
| `Stability` | collapse rate / NaN rate | 比可訓練性 |
| `Generalization` | OOD/noise/perturbation 下性能保留率 | 比穩健性 |

---

## 6) 建議你們論文敘事（避免空泛）

你可以把主敘事寫成：

1. **Flow policy 已被文獻證實可行**，但在你們設定中，性能上限主要受 WM 品質約束。  
2. 透過 Newt/RWM-O 啟發的訓練策略（demo leverage + uncertainty-aware world model），Flow-MBPO 的穩定性和上限同時提升。  
3. 在公平協議下（EqualSteps/EqualTime/EqualCompute），Flow-MBPO 在高維與長 horizon 場景展現優勢。  

---

## 7) 下一步（務實執行順序）

1. 先在 `mjlab` 實作 `WM uncertainty penalty`（`WM1`）與 `demo oversampling`（`WM2`）。  
2. 同步補 flow policy 的 `zero-sampling` 與 per-sample ratio（若走 FPO-style 更新）。  
3. 啟動 Track B（mjlab 主線）拿第一批穩定結果，再啟動 Track A（MMBench 對 Newt 嚴格可比）。  

---

## Sources

- Flow Matching Policy Gradients (arXiv:2507.21053): https://arxiv.org/abs/2507.21053  
- Flow Policy Gradients for Robot Control (arXiv:2602.02481): https://arxiv.org/abs/2602.02481  
- Learning Massively Multitask World Models for Continuous Control (arXiv:2511.19584): https://arxiv.org/abs/2511.19584  
- Offline Robotic World Model Learning Robotic Policies without a Physics Simulator (arXiv:2504.16680): https://arxiv.org/abs/2504.16680  
- Newt official repo README/examples: https://github.com/nicklashansen/newt  
- Newt model card/checkpoints summary: https://huggingface.co/nicklashansen/newt  
- MMBench dataset card: https://huggingface.co/datasets/nicklashansen/mmbench  
- mjlab official repo/readme: https://github.com/mujocolab/mjlab  
- mjlab docs home: https://mujocolab.github.io/mjlab/  
- mjlab PyPI: https://pypi.org/project/mjlab/
