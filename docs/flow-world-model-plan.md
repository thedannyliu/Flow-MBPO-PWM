Title: PWM Flow-Matching World Model — Precise Design, Interfaces, and Fair-Comparison Protocol

Status: Authoritative plan (all future changes must conform)
Owners: PWM extension (flow-based dynamics)
Scope: Introduce a flow-matching dynamics into PWM with rigorous software design, explicit interfaces, parameter/compute parity, and mathematically defined diagnostics.

---

**1) Goals and Non-Goals**
- Goals
  - Replace the baseline MLP next-state dynamics with a conditional flow-matching dynamics to improve gradient quality for policy learning, without altering actor/critic objectives.
  - Preserve strict separation of concerns: models implement single-step functions; the algorithm handles rollout and loss.
- Non-Goals
  - No change to encoder and reward architectures besides parity alignment.
  - No hidden “duck typing” or runtime `hasattr` checks.
  - No coupling of rollout/loss into model classes.

---

**2) Minimal Software Design (pragmatic first iteration)**
- No new abstract base classes or strategy modules in the initial iteration.
- Do not modify the existing world model class structure.
- Add a single `if/else` switch inside `pwm.py::compute_wm_loss`:
  - If `cfg.alg.use_flow_dynamics == False` → use the current baseline MSE dynamics loss (unchanged codepath).
  - If `cfg.alg.use_flow_dynamics == True` → compute the flow-matching dynamics loss and integrate with the selected integrator (Section 4).
- Add a `# TODO` in code: if more loss types are added in the future, refactor to a strategy/ABC pattern.
- Responsibility separation is preserved: rollout and loss remain in the algorithm layer; model exposes `encode/next/reward/step` only (no loss helpers in the model).

---

**3) Mathematical Specification**
- Latent Encoding and Reward
  - `z_t = E_φ(o_t)`, `r̂_t = R_φ(z_t, a_t)`; identical architectures for MLP and Flow variants.

- Baseline Dynamics (for reference)
  - `z_{t+1} = F_φ(z_t, a_t)` (MLP)
  - Dynamics loss: `L_dyn = (1/H) Σ_{t=0}^{H-1} γ^t || z_pred_{t+1} − z_tgt_{t+1} ||^2`, where `z_tgt_{t+1} = E_φ(o_{t+1})`.

- Flow-Matching Dynamics
  - Velocity field: `v_θ(z, a, τ)` with explicit time conditioning `τ ∈ [0, 1]`.
  - Target construction per step `t`:
    - `z_s = z_pred_t`, `z_tgt = E_φ(o_{t+1})`.
    - Sample `τ ~ U[0,1]` (or midpoint `0.5`).
    - Interpolate `z_τ = (1−τ) z_s + τ z_tgt`.
    - Rectified-flow target: `v* = z_tgt − z_s`.
  - Per-step loss: `ℓ_t = || v_θ(z_τ, a_t, τ) − v* ||^2`.
  - Aggregation: `L_dyn = (1/H) Σ_{t=0}^{H-1} γ^t ℓ_t`.

- Reward Loss (shared)
  - `L_rew = mean_t [ ((R_φ(z_t, a_t) − r_t)^2) · γ^t ]` with two-hot handling identical to baseline.
  - Total: `L_wm = L_dyn + L_rew`.

---

**4) ODE Integrator Choice and Justification**
- Default: Heun’s method (explicit trapezoidal / RK2)
  - For fixed sub-steps `K`, `dt = 1/K`, per sub-step:
    - `k1 = v_θ(z, a, t_k)`
    - `z' = z + dt · k1`
    - `k2 = v_θ(z', a, t_k + dt)`
    - `z ← z + (dt/2) · (k1 + k2)`
  - Rationale: Substantial stability improvement vs Euler with ~2× field evals; materially better multi-step behavior at small extra cost.
- Ablation: Euler (K identical) for completeness.
- We report both reward and wall-clock under the chosen K; compute-aware analysis is specified in Section 7.

---

**5) Fairness: Parameter Parity（務實版）**
- 參數量對齊（必要）：
  - `P_base`：現有 WM（encoder+dynamics+reward）的參數總數。
  - `P_flow`：flow 版本的參數總數。
  - 規則：`|P_flow − P_base| / P_base ≤ 0.02`（1–2% 範圍內）。
  - 作法：維持相同深度，必要時微調 hidden width 以補償 +1 的時間通道，直到達到 1–2% 的範圍。
  - 驗證與紀錄：初始化時記錄 `P_base、P_flow、差距百分比`。若超過 2%，以 warning log 提示並繼續訓練；在報告中透明呈現實際差距與其可能影響。
- 不做「計算量對齊」消融於第一階段。專注在參數對齊下的核心比較。牆鐘時間仍需回報以供參考。

---

**6) ESNR：定義與測量（保留數學、鬆綁實作）**
- 目標：衡量 actor 更新的梯度訊噪比。
- 定義（數學不變）：
  - 令微批次梯度為 `g_i`，`μ = (1/M) Σ_i g_i`，`E[||g||^2] = (1/M) Σ_i ||g_i||^2`，`Var = E[||g||^2] − ||μ||^2`。
  - `ESNR = ||μ||^2 / max(Var, ε)`，並可報告 `ESNR_dB = 10 log10(ESNR)`。
- 實作自由度：不在計畫中固定 `M`、記錄頻率或是否採用 EMA；這些作為可調超參數，在實作時根據效能/開銷權衡設定，並於報告中註明。

---

**7) Algorithmic Rollout and Loss (Responsibility Separation)**
- Location: `PWM/src/pwm/algorithms/dynamics_loss.py` (new) and `PWM/src/pwm/algorithms/pwm.py` (existing).
- The algorithm performs the H-step rollout, independent of model internals:
  - `z0 = model.encode(obs[0])`.
  - For `t = 0..H−1`: `z_{t+1} = model.next(z_t, a_t)` via the chosen model.
  - Collect `zs = [z0..zH]`.
- Dynamics losses via strategy:
  - MLP baseline: `MSE(z_{t+1}, E_φ(o_{t+1}))` with `γ^t` weights.
  - Flow-matching: compute per-step rectified-flow loss as in Section 3, using the chosen integrator from Section 4 to produce `z_{t+1}`.
- Reward loss identical to baseline using `zs[:-1]` and `act`.
- No loss computation resides inside model classes.

---

**8) Configuration (Hydra) and Wiring**
- Add `alg.dynamics_loss._target_` to select the loss strategy explicitly (`MLPDynamicsLoss` or `FlowMatchingDynamicsLoss`).
- Add `alg.integrator` group with fields: `type ∈ {heun, euler}`, `substeps (K)`, `tau_sampling ∈ {uniform, midpoint}`.
- Add `alg.model._target_` to choose between `MLPWorldModel` and `FlowWorldModel` (no duck typing).

---

**9) Implementation Steps and Files**
- Models
  - Add `pwm/models/base.py` with `BaseWorldModel` ABC.
  - Refactor current `world_model.py` to expose `MLPWorldModel` (wrapping existing logic; no functional changes).
  - Add `pwm/models/flow_world_model.py` implementing the velocity field dynamics; encoder/reward identical to baseline.
- Algorithm
  - Add `pwm/algorithms/dynamics_loss.py` with `MLPDynamicsLoss` and `FlowMatchingDynamicsLoss` (+ integrator implementations and unit-tested math helpers).
  - Update `pwm/algorithms/pwm.py` to instantiate the selected strategy via Hydra and call it in `compute_wm_loss`.
- Configs
  - Add `scripts/cfg/alg/pwm_48M_flow.yaml` mirroring baseline with `model._target_` / `dynamics_loss._target_` / `integrator` overrides.
- Instrumentation
  - Add ESNR logging utility in `pwm/utils/metrics.py` implementing the estimators in Section 6.

---

**10) Verification and Invariants（精簡）**
- 參數量：記錄 `P_base, P_flow, 差距%`；若超 2% 發出 warning（不中斷），並於結果中揭露。
- 健全檢查（sanity checks）：檢測 NaN、過大梯度或潛在向量爆炸（可用梯度裁剪與簡單範數門檻），超標則早停或重設 batch。
- 不為積分器編寫單元測試；Heun 實作以簡單自檢（單步邏輯與張量形狀）為主。

---

**11) Experimental Protocol**
- Datasets: identical offline data per task.
- Seeds: 3 seeds per setting.
- Settings: MT30 and MT80; optionally 1–2 dflex single tasks.
- Variants:
  - Baseline: `MLPWorldModel + MLPDynamicsLoss`.
  - Flow: `FlowWorldModel + FlowMatchingDynamicsLoss` (Heun, `K ∈ {2,4}`), parameter parity enforced.
  - Optional compute-aware: choose `K` to approximate eval count parity; report separately.
- Metrics per run: episode reward, success, WM losses, ESNR (actor), grad norms, wall-clock.

---

**12) Acceptance Criteria**
- Code respects the interfaces and responsibility separation defined here (no runtime `hasattr`, no loss in model classes).
- Parameter parity constraint satisfied; runs abort if violated.
- Flow variant meets or exceeds baseline on ≥50% tasks in reward under parameter parity, and shows improved ESNR trends on at least one benchmark.
- All results include wall-clock and evaluation count reporting.

---

**13) Rationale For Key Choices**
- Interfaces via ABCs improve maintainability, explicit coupling, and testing vs ad-hoc runtime checks.
- Heun integrator balances stability and cost, improving multi-step accuracy critical for PWM’s H-step unrolls.
- Parameter parity removes capacity confounds; compute-aware ablation separates architecture benefits from raw throughput.
- ESNR definition is explicit, computable with bounded overhead, and directly tied to H1.

---

File Reference (authoritative plan file): docs/flow-world-model-plan.md:1
