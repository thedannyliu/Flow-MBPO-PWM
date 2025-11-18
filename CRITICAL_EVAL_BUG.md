# 🚨 重大發現：Eval()函數有嚴重Bug！

**Date**: November 18, 2025  
**Priority**: 🔴 CRITICAL

---

## ❌ 發現的問題

### 1. Eval()使用World Model的Reward，不是真實環境！

查看 `PWM/src/pwm/algorithms/pwm.py` line 578的eval()函數：

```python
def eval(self, num_games, deterministic=True):
    # ...
    
    # 使用world model預測下一個狀態和reward
    if self.use_flow_dynamics:
        res = self.wm.step(z, actions, ...)  # ← 這裡！
    else:
        res = self.wm.step(z, actions, task=None)
    
    z, rew, trunc = res  # ← reward來自world model！
    
    # 環境只用來獲取done信號
    _, _, done, _ = self.env.step(actions)  # ← 忽略了環境的真實reward！
    
    episode_loss -= rew  # ← 使用world model的reward計算loss
```

**這意味著**:
- ✅ Actions來自actor (正確)
- ❌ Rewards來自world model預測 (錯誤！)
- ✅ Done信號來自環境 (正確)
- ❌ Episode loss基於world model reward (錯誤！)

### 2. 這解釋了所有異常現象

#### Loss都是0.00
- World model預測的reward可能接近0
- 或者reward預測有問題
- 不反映真實性能

#### V1 Episode Length = 1000
- Flow world model可能預測太樂觀
- 從不預測會摔倒
- 實際上可能早就摔倒了

#### Baseline "崩潰"
- 不是真的崩潰
- 只是world model預測變差
- 真實性能未知

---

## 🔍 驗證

### 檢查訓練日誌

所有模型的eval結果:
```
Baseline: loss = 0.00, len = 15.90
Flow V1:  loss = 0.00, len = 1000.00  ← 異常！
Flow V2:  loss = 0.00, len = 21.60
Flow V3:  loss = 0.00, len = 15.88
```

**共同點**: 所有loss都是0.00  
**異常**: V1的length達到最大值1000

### Reward來源確認

訓練過程中的reward (來自真實環境):
```
Flow V1訓練過程:
[61/20000]  R:130.66  ← 真實環境reward
[62/20000]  R:169.80
...
[後期]      R:1130+   ← 訓練很成功！

但Eval結果:
loss = 0.00  ← World model reward
```

---

## 🎯 正確的Evaluation方法

### 應該使用真實環境的Reward

```python
def eval(self, num_games, deterministic=True):
    # ...
    
    actions = self.actor(obs, deterministic=deterministic)  # 直接用obs
    
    # 使用真實環境
    obs, rew, done, _ = self.env.step(actions)  # ← 使用環境的reward！
    
    episode_loss -= rew  # ← 使用真實reward
```

### 或者使用evaluate_policy.py

`PWM/scripts/evaluate_policy.py` 有正確的評估函數：
```python
def evaluate_policy(agent, env, num_episodes=100):
    # 使用真實環境step
    obs, reward, done, info = env.step(action)
    # 使用真實reward
    episode_reward += reward
```

---

## 📊 重新評估需求

### 之前的所有"結果"都不可靠

| 模型 | 之前報告的 | 實際情況 |
|------|-----------|---------|
| Baseline | R~141, length=15.90 | ❓ 未知 (需要真實eval) |
| Flow V1 | R~1133, length=1000 | ❓ 未知 (world model過度樂觀) |
| Flow V2 | R~1197→561, length=21.60 | ❓ 未知 |
| Flow V3 | R~1137, length=15.88 | ❓ 未知 |

**唯一可靠的指標**: 訓練過程中的R值
- 這些來自真實環境
- Baseline訓練R: peak ~292
- Flow V1訓練R: peak ~1133
- Flow V2訓練R: peak ~1197
- Flow V3訓練R: peak ~1137

---

## ✅ 需要做的事

### 立即 (今天)

1. **修復eval()函數**
   ```python
   # 選項1: 直接用環境
   obs, rew, done, _ = self.env.step(actions)
   # 不要用world model的reward
   
   # 選項2: 使用evaluate_policy.py腳本
   ```

2. **重新評估所有模型**
   ```bash
   python scripts/evaluate_policy.py \
     --checkpoint outputs/2025-11-17/.../best_policy.pt \
     --num-episodes 100
   ```

3. **對比訓練R vs 評估R**
   - 訓練R: 來自真實環境 ✅
   - 評估R: 應該類似或稍高 (deterministic policy)

### 驗證問題

4. **檢查為什麼之前報告"R ~ 1200 (PWM paper baseline)"**
   - 找到之前成功的訓練記錄
   - 確認是用什麼方法評估的
   - 可能用的是evaluate_policy.py?

5. **理解world model quality**
   - 為什麼V1的world model預測length=1000?
   - 是flow dynamics太樂觀?
   - 還是reward model有問題?

---

## 🤔 之前的結論需要修正

### 錯誤的分析鏈

1. ❌ "V1 length=1000表示完美完成episode"
   - 實際: World model預測的，不是真實環境

2. ❌ "Baseline崩潰了"
   - 實際: World model預測變差，真實性能未知

3. ❌ "V1是最穩定的模型"
   - 實際: 無法從當前數據得出

4. ❌ "所有loss=0.00是正常的"
   - 實際: 這是bug的症狀

### 唯一可信的數據

✅ **訓練過程中的R值** (來自真實環境):
```
Baseline:  peak ~292
Flow V1:   peak ~1133  (3.88x)
Flow V2:   peak ~1197  (4.10x)
Flow V3:   peak ~1137  (3.89x)
```

這些是真實的性能指標！

---

## 🎯 Action Items

### Priority 1: 修復並重新評估

```bash
# 1. 使用evaluate_policy.py正確評估
cd PWM
python scripts/evaluate_policy.py \
  --checkpoint outputs/2025-11-17/22-07-53/baseline/best_policy.pt \
  --num-episodes 100 \
  --env dflex_ant

# 2. 對比所有模型
for model in baseline flow_v1 flow_v2 flow_v3; do
  python scripts/evaluate_policy.py \
    --checkpoint outputs/.../$model/best_policy.pt \
    --num-episodes 100 \
    --output results_$model.json
done

# 3. 生成對比報告
python scripts/compare_results.py results_*.json
```

### Priority 2: 修復eval()函數

在 `pwm.py` 中修改eval():
```python
def eval(self, num_games, deterministic=True):
    # ...
    
    # 選項A: 純環境評估 (推薦)
    obs = self.env.reset()
    if self.obs_rms is not None:
        obs = self.obs_rms.normalize(obs)
    
    actions = self.actor(obs, deterministic=deterministic)
    obs, rew, done, _ = self.env.step(actions)  # 使用真實reward
    
    # 選項B: 仍使用world model但記錄真實reward
    # (用於對比world model accuracy)
```

---

## 📌 總結

### 關鍵發現

1. **Eval()函數有嚴重bug** - 使用world model reward而非環境reward
2. **所有之前的"評估結果"都不可靠** - length=1000, loss=0.00等都不是真實表現
3. **唯一可信的是訓練R** - 顯示Flow確實比baseline好3.9-4.1倍
4. **需要立即用真實環境重新評估所有模型**

### 你說得對！

- ✅ Baseline應該達到~1200 (如果之前成功過)
- ✅ Length=1000確實很可疑
- ✅ 需要重新檢查和調整

### 下一步

1. 修復eval()或使用evaluate_policy.py
2. 重新評估所有checkpoints
3. 驗證Flow的真實改進幅度
4. 找到之前成功的baseline配置

---

*Critical Bug Report*  
*Date: November 18, 2025*  
*Status: 🔴 需要立即修復和重新評估*
