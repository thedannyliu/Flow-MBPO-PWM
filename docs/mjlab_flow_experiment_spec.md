# Flow-MBPO @ mjlab 實驗與 Ablation 規格（可直接執行版）

> Version: `v1.0`  
> Frozen Date: `2026-02-18`  
> 目的：把「要跑哪些實驗、為什麼跑、如何統計、如何記錄」一次定義清楚，後續只需照表執行與填寫結果。

---

## 0. 核心研究主張與對應限制

我們把研究敘事明確拆成可驗證假說（Hypotheses）：

| Hypothesis ID | 研究主張 | 對應舊限制（PWM/MLP） | 若成立代表什麼 |
|---|---|---|---|
| H1 | Flow WM 在高維任務優於 MLP WM | MLP latent dynamics 在高維/複雜動力下誤差累積快 | Flow dynamics 對複雜轉移更穩定 |
| H2 | Flow Policy 在固定 WM 下優於 MLP Policy | Gaussian/MLP policy 表達能力有限 | Flow policy 提供更有效探索或 action manifold |
| H3 | Flow 的增益不是只來自「算更久」 | Flow 模型通常每 step 計算更重 | 在等算力/等時間下仍有增益 |
| H4 | Flow 在長 horizon 或模型誤差累積下較穩定 | PWM 中 H 增大時 surrogate gradient 容易劣化 | Flow 對 rollout mismatch 較不敏感 |
| H5 | Flow latent representation 更具可分性/可用性 | MLP latent 對下游控制訊號不夠豐富 | latent probe 與 dynamics 指標支持機制解釋 |

### 這張表為什麼要跑

- 先把故事從「感覺比較好」變成「每個 claim 都可被反駁」。

### 跑完可得 insight

- 最終 paper 的主敘事可直接映射到 H1-H5，不會變成結果堆疊而無因果脈絡。

---

## 1. 實驗總體設計（分層執行）

| 層級 | 目的 | Seeds | 用途 |
|---|---|---|---|
| L0 Smoke | 確認可跑、無 API/NaN 問題 | 1-2 | 開發期，不能做結論 |
| L1 Pilot | 觀察趨勢、過濾明顯失敗設定 | 3-5 | 決定是否進入完整統計 |
| L2 Confirmatory | 正式檢定假說 | 10（主實驗）/5（次要 ablation） | 用於報告與結論 |

### 為什麼要跑

- 把「工程除錯」與「科學結論」分離，避免拿 smoke 結果做主張。

### 跑完可得 insight

- 能清楚知道哪些設定只是可跑、哪些設定具統計可信度。

---

## 2. mjlab 任務面板定義（先固定再跑）

先固定一個可覆蓋維度與難度的 task panel；不要邊跑邊換 task。

| Panel Slot | 維度/難度目的 | 選擇規則 | 實際 task_id（待填） | Obs Dim | Act Dim | Episode Len |
|---|---|---|---|---:|---:|---:|
| MJ-L1 | 低維控制 | obs/act 落在面板最低 30% |  |  |  |  |
| MJ-L2 | 低維控制 | 與 MJ-L1 不同動力型態 |  |  |  |  |
| MJ-M1 | 中維 locomotion | obs/act 中位區間 |  |  |  |  |
| MJ-M2 | 中維 locomotion | 與 MJ-M1 不同 reward shaping |  |  |  |  |
| MJ-H1 | 高維 humanoid/complex | obs/act 高位 30% |  |  |  |  |
| MJ-H2 | 高維 humanoid/complex | 與 MJ-H1 不同接觸型態 |  |  |  |  |

### 為什麼要跑

- H1/H5 的核心在「維度與複雜度」，不先固定面板會導致 cherry-picking。

### 跑完可得 insight

- 可做跨維度趨勢分析，而非單一 task 的偶然結果。

---

## 3. 主實驗：2x2 因子設計（必跑）

固定除了 WM/Policy 架構外的所有條件，做完整 2x2。

| Exp ID | WM | Policy | 主要對應假說 | Seeds (L2) | Budget | 主要指標 |
|---|---|---|---|---:|---|---|
| E1 | MLP | MLP | Baseline | 10 | 固定 env steps + 固定 wall-clock 報告 | AUC, Final Return, Success |
| E2 | Flow | MLP | H1 | 10 | 同 E1 | 同上 + WM losses |
| E3 | MLP | Flow | H2 | 10 | 同 E1 | 同上 + action stats |
| E4 | Flow | Flow | H1+H2 interaction | 10 | 同 E1 | 同上 |

每個 `Exp ID` 都在 `MJ-L1..MJ-H2` 全部 task 跑一輪。

### 這張表為什麼要跑

- 這是最核心證據：可分離 WM 因子與 Policy 因子，避免混雜解讀。

### 跑完可得 insight

- `(E2-E1)` = Flow WM 純增益；`(E3-E1)` = Flow Policy 純增益；`(E4-E2)` 可看交互作用是否存在。

---

## 4. 限制導向 Ablations（回答「為何有效」）

## 4.1 Horizon / Rollout 穩定性（H4）

| Exp ID | 固定設定 | 改動軸 | 值 | Seeds | 指標 | 期望觀察 |
|---|---|---|---|---:|---|---|
| A1 | E1/E2 的最佳預設 | Horizon H | 4, 8, 16 | 5 | Reward, ESNR, collapse rate | Flow 在較長 H 衰退較慢 |
| A2 | Flow-WM only | Substeps K | 2, 4, 8 | 5 | Reward vs time, WM loss | 找到效能/計算折衷點 |
| A3 | Flow-WM only | tau sampling | uniform, midpoint | 5 | 穩定性與方差 | 判斷訓練訊號是否更平滑 |

### 為什麼要跑

- Flow 常被質疑只是更大/更慢模型；這組要回答「在哪些 dynamics 條件下有效」。

### 跑完可得 insight

- 可定位 Flow 有效區間（例如長 horizon）並形成可轉移的設計準則。

## 4.2 Reward pipeline 對齊（避免錯誤歸因）

| Exp ID | 固定設定 | 改動軸 | 值 | Seeds | 指標 | 目的 |
|---|---|---|---|---:|---|---|
| A4 | E1/E2 | Reward loss | MSE vs Two-hot/CE(若實作) | 5 | Reward, WM reward loss | 排除 reward head 差異造成假增益 |
| A5 | E2 | FM source state | rollout-based vs teacher-forced | 5 | ESNR,穩定性,最終回報 | 驗證 correction-flow 機制 |

### 為什麼要跑

- 避免把 pipeline 差異誤判成 Flow 架構優勢。

### 跑完可得 insight

- 可更乾淨回答「Flow dynamics 本身」是否有貢獻。

---

## 5. 機制驗證實驗（支持 representation 敘事，H5）

## 5.1 Latent Probe 套件

| Exp ID | Probe 任務 | 訓練資料來源 | 量測 | 比較組 |
|---|---|---|---|---|
| M1 | `z_t -> r_t` 線性回歸 | replay samples | `R^2` / MSE | E1 vs E2 |
| M2 | `z_t -> done_t` 分類 | replay samples | AUC / F1 | E1 vs E2 |
| M3 | `z_t, a_t -> z_{t+1}` 線性近似誤差 | replay samples | one-step error | E1 vs E2 |
| M4 | task ID / command 預測（multi-task） | multi-task buffer | accuracy / NMI | E1 vs E2 |

### 為什麼要跑

- 你的敘事是「Flow latent 更豐富」，就需要不依賴最終 reward 的中介證據。

### 跑完可得 insight

- 若高維任務 `E2>E1` 且 probe 指標同步改善，能更有力支持 representation 機制。

## 5.2 Jacobian / Sensitivity 指標（可選但強力）

| Exp ID | 指標 | 定義 | 目的 |
|---|---|---|---|
| M5 | `||∂z_{t+1}/∂a_t||` 分佈 | local controllability proxy | 檢查 action 對 latent 的可控性 |
| M6 | `||∂V/∂z||` 與 actor grad SNR | gradient quality proxy | 連結到 FoG 可訓練性 |

### 為什麼要跑

- 把「Flow 比較穩」轉成可量測的梯度與敏感度訊號。

### 跑完可得 insight

- 可解釋為什麼某些 task 上 reward 提升顯著、某些不顯著。

---

## 6. Robustness / OOD 實驗（H4/H5 延伸）

| Exp ID | Shift 類型 | 設定 | 比較組 | Seeds | 指標 |
|---|---|---|---|---:|---|
| R1 | Observation noise | test-time obs 加噪（3 強度） | E1/E2/E3/E4 | 5 | degrade slope |
| R2 | Dynamics perturbation | 質量/摩擦/延遲小幅偏移 | E1/E2/E3/E4 | 5 | robustness gap |
| R3 | Partial observability | 隱藏部分觀測通道 | E1/E2/E3/E4 | 5 | 成功率與恢復能力 |

### 為什麼要跑

- 如果 Flow 只是 overfit 原分佈，實用價值有限。

### 跑完可得 insight

- 能驗證 Flow 的增益是否在分佈偏移時仍保留。

---

## 7. mjlab 遷移效益與計算公平性（H3）

## 7.1 Simulator 與吞吐校準

| Exp ID | 目的 | 設定 | 輸出 |
|---|---|---|---|
| S0 | API/資料正確性 | 2 seeds, 300 epochs | 無 NaN、buffer 正常、eval 正常 |
| S1 | 吞吐 benchmark | 固定 config 比較 env FPS | `env_steps/s`, `train_steps/s` |
| S2 | Time breakdown | profiling | env.step% / WM update% / critic% |

### 為什麼要跑

- 先確認「慢」是否真來自 simulator，再決定優化重點。

### 跑完可得 insight

- 若 env 只占小比例，換 simulator 不會顯著加速，需改訓練熱點。

## 7.2 公平比較協議（必遵守）

每一個主結果都需同時提供三種比較：

| 協議 | 控制方式 | 用途 |
|---|---|---|
| Equal-EnvSteps | 固定互動 steps | 比 sample efficiency |
| Equal-WallClock | 固定訓練時間 | 比實務迭代效率 |
| Equal-ComputeIndex | 固定 `H/K/Kpol` 對應計算代理 | 排除「算更多」因素 |

### 為什麼要跑

- 防止 reviewer 認為 Flow 提升只是因為算更久。

### 跑完可得 insight

- 你能明確說明 Flow 在哪個公平框架下仍成立。

---

## 8. 統計嚴謹性規範（固定分析計畫）

## 8.1 主要與次要終點（Pre-register）

| 層級 | 指標 | 定義 |
|---|---|---|
| Primary | `Normalized AUC` | 在固定 steps 區間下，reward curve 面積（task-normalized） |
| Primary | `Final Return` | 最後評估窗口平均 |
| Secondary | `Success Rate` | task success 平均 |
| Secondary | Stability | collapse rate / NaN rate |
| Mechanistic | Probe / Jacobian 指標 | 見 M1-M6 |

## 8.2 檢定方法（建議固定）

| 問題類型 | 檢定 | 報告內容 |
|---|---|---|
| 同 task, 兩方法比較 | Paired bootstrap CI（seed 配對） | effect size + 95% CI |
| 多 task 聚合 | Hierarchical bootstrap（task × seed） | overall effect + CI |
| 多重比較 | Benjamini-Hochberg FDR | q-value |
| 實務顯著性 | 最小效果門檻 `>=5%` 相對提升 | 是否達門檻 |

## 8.3 結論門檻（避免 p-hacking）

某假說判定「支持」需同時滿足：

1. Primary endpoint 至少一項達顯著（CI 不跨 0，且 FDR 後仍成立）  
2. 實務效果門檻達成（`>=5%`）  
3. 至少 2 個 task bin（L/M/H）方向一致

---

## 9. 可復現性與執行規範（Run Hygiene）

| 類別 | 必做項目 |
|---|---|
| Code | 記錄 git commit hash、dirty state、patch 摘要 |
| Config | 保存 Hydra resolved config + checksum |
| Env | 保存 `pip freeze`/`uv.lock`、CUDA driver、GPU 型號 |
| Seed | 固定 seed list（主實驗 10 seeds）並跨方法共用 |
| Runtime | 記錄 node ID、開始/結束時間、故障重跑原因 |
| Artifact | checkpoint、eval csv、raw logs 路徑固定化 |
| Failure policy | OOM/硬體故障可重跑；演算法 NaN 不可 silently 重跑，需記錄為 failure |

---

## 10. 結果填寫模板（跑完直接填）

## 10.1 Run-level 登記表

| Run ID | Exp ID | Task | Seed | Commit | Config Hash | Status | Final Return | AUC | Success | Runtime(h) | Node | Notes |
|---|---|---|---:|---|---|---|---:|---:|---:|---:|---|---|
|  |  |  |  |  |  |  |  |  |  |  |  |  |

## 10.2 聚合統計表（每個 Exp ID × Task）

| Exp ID | Task | N | Mean Final Return | Std | Mean AUC | 95% CI vs Baseline | Effect Size(%) | Verdict |
|---|---|---:|---:|---:|---:|---|---:|---|
|  |  |  |  |  |  |  |  |  |

## 10.3 假說總結表（寫論文可直接引用）

| Hypothesis | 支持程度（Yes/Partial/No） | 主要證據（Exp IDs） | 反例/限制 | 下一步 |
|---|---|---|---|---|
| H1 |  |  |  |  |
| H2 |  |  |  |  |
| H3 |  |  |  |  |
| H4 |  |  |  |  |
| H5 |  |  |  |  |

---

## 11. 建議執行順序（你可以直接照這個跑）

1. `S0 -> S1 -> S2`（先確保 mjlab 遷移可用且真的加速）  
2. `E1-E4` 在 `MJ-L1/MJ-M1/MJ-H1` 做 L1 pilot（3-5 seeds）  
3. pilot 通過後，`E1-E4` 全 task panel 跑 L2 confirmatory（10 seeds）  
4. 跑 `A1-A5`（5 seeds）定位有效區間與失效區間  
5. 跑 `M1-M6` 補齊機制證據  
6. 跑 `R1-R3` 做 robustness 收尾  
7. 填 `10.1~10.3`，輸出論文主表與附錄表

---

## 12. 與現有文件的關係

- 本文件是「可執行實驗規格（Spec）」；  
- `docs/experiment_log.md` 是「每次 run 的流水帳與結果登記」；  
- `docs/master_plan.md` 是「高層研究路線圖」。  

建議：之後所有新實驗都先對應一個 `Exp ID`，再提交 job。
