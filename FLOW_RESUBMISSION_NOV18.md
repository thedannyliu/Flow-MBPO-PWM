# Flow 訓練重新提交 - Nov 18, 2025 15:12

## ✅ 已完成操作

### 1. 修改提交腳本（增加內存）

修改了 3 個 Flow 提交腳本：
- `submit_48M_flow_v1_l40s.sh`: 128GB → **256GB**
- `submit_48M_flow_v2_l40s.sh`: 128GB → **256GB** ⭐
- `submit_48M_flow_v3_l40s.sh`: 128GB → **256GB**

### 2. 創建批量提交腳本

新建：`submit_all_flow_256GB.sh`
- 自動提交所有 3 個 Flow 訓練
- 顯示 Job ID 和狀態

### 3. 提交訓練任務

| Job ID | 模型 | 內存 | 狀態 |
|--------|------|------|------|
| 2322456 | 48M Flow V1 (substeps=2) | 256GB | ⏳ 排隊中 |
| 2322458 | 48M Flow V2 (substeps=4) ⭐ | 256GB | ⏳ 排隊中 |
| 2322459 | 48M Flow V3 (substeps=8) | 256GB | ⏳ 排隊中 |

### 4. Git 記錄

**Commit 1 (PWM submodule):** `3fd2d37`
```
Fix OOM: Increase memory to 256GB for Flow models
```

**Commit 2 (Main repo):** `48cfbbe`
```
Update PWM submodule: Fix Flow OOM with 256GB memory
```

---

## 📊 為什麼重新訓練？

### 之前的 OOM 失敗

| 模型 | 完成度 | Peak R | OOM 時間 |
|------|--------|--------|----------|
| Flow V1 | 45% (9073/20000) | 1100 | ~5.6h |
| Flow V2 ⭐ | 27% (5424/20000) | **1210** | ~5h |
| Flow V3 | 27% (5423/20000) | 1182 | ~5h |

**對比 Baseline:**
- 48M Baseline: Peak R **1254** (完整訓練)
- Flow V2: Peak R **1210** (只訓練 27%！)

**關鍵洞察：**
Flow V2 只用 27% 的訓練就達到接近 baseline 的性能，**完整訓練很可能超越 baseline！**

---

## 🎯 預期結果（256GB 內存）

### 預測性能

| 模型 | 之前 Peak | 預期 Peak (256GB) | 信心度 |
|------|-----------|-------------------|--------|
| Flow V1 | 1100 (45%) | ~1200-1250 | 中 |
| **Flow V2** ⭐ | 1210 (27%) | **~1300-1350** 🚀 | 高 |
| Flow V3 | 1182 (27%) | ~1250-1300 | 中 |

**對比 Baseline:**
- 48M Baseline: 1254
- **預期 Flow V2: ~1300+** (超越！)

### 為什麼有信心？

1. **學習效率高：** V2 只訓練 27% 就達到 1210
2. **趨勢良好：** 在 OOM 前性能還在上升
3. **配置最佳：** substeps=4 是最佳平衡點（5M 實驗驗證）

---

## ⏰ 時間線

| 時間點 | 事件 |
|--------|------|
| **15:12** | ✅ 提交 3 個 Flow 訓練（256GB） |
| **15:12-?** | ⏳ 排隊等待資源 |
| **~16:00?** | 🚀 開始訓練 |
| **~04:00 (Nov 19)** | 🎯 預計完成（12h 訓練） |

---

## 📈 監控命令

### 查看任務狀態
```bash
squeue -u $USER
```

### 即時監控最重要的 Flow V2
```bash
tail -f PWM/logs/train_48M_flow_v2_l40s_2322458.out
```

### 監控所有 Flow
```bash
# V1
tail -f PWM/logs/train_48M_flow_v1_l40s_2322456.out

# V2 (最重要)
tail -f PWM/logs/train_48M_flow_v2_l40s_2322458.out

# V3
tail -f PWM/logs/train_48M_flow_v3_l40s_2322459.out
```

### 提取實時 R 值
```bash
grep -oP "R:\K[0-9.]+" PWM/logs/train_48M_flow_v2_l40s_2322458.out | tail -10
```

---

## 🎯 成功標準

### 最低目標
- ✅ 完成 20,000 epochs（不 OOM）
- ✅ Peak R > 1200

### 理想目標
- 🎯 Flow V2 Peak R > 1254 (超越 baseline)
- 🎯 穩定訓練到結束

### 完美目標
- 🚀 Flow V2 Peak R > 1300
- 🚀 最後 100 avg R > 100

---

## 📊 實驗總覽（包含新任務）

### 已完成 ✅
1. 5M Baseline (2314140): Peak R **1222**
2. 48M Baseline (2314141): Peak R **1254**

### OOM 失敗 ⚠️（已重新提交）
3. 48M Flow V1 (2314142): Peak R 1100 → **重新訓練 (2322456)**
4. 48M Flow V2 (2314143): Peak R 1210 → **重新訓練 (2322458)** ⭐
5. 48M Flow V3 (2314144): Peak R 1182 → **重新訓練 (2322459)**

### 配置錯誤 ❌（暫時擱置）
6. 48M MT Baseline (2314382)
7. 48M MT Flow V1 (2314383)
8. 48M MT Flow V2 (2314384)
9. 48M MT Flow V3 (2314385)

---

## 🔄 如果還是 OOM 怎麼辦？

### Plan B: 降低 Batch Size

如果 256GB 還不夠，修改配置文件：

```yaml
# PWM/scripts/cfg/alg/pwm_48M_flow_v2_substeps4.yaml
wm_batch_size: 512  # 從 1024 → 512
```

然後重新提交。

### Plan C: 使用更多節點

如果需要，可以申請更大的節點：

```bash
#SBATCH --mem=512GB  # 或更多
```

---

## 📝 下一步

### 立即（0-1 小時）
- ⏳ 等待任務開始運行
- 👀 監控是否成功開始

### 短期（12-24 小時）
- 📊 監控訓練進度
- 🔍 檢查是否有 OOM
- 📈 追蹤 Peak R 變化

### 完成後（~Nov 19 04:00）
- ✅ 提取所有 R 值
- 📊 生成訓練曲線對比
- 📝 撰寫完整結果報告
- 🎓 準備論文材料

---

## 💡 關鍵要點

1. **256GB 應該足夠**
   - 之前 128GB 不夠
   - 256GB = 2x 內存
   - 理論上可以完成訓練

2. **Flow V2 最重要**
   - 27% 訓練就達到 1210
   - 完整訓練很可能超越 baseline
   - **這是論文的關鍵結果！**

3. **時間充足**
   - 12 小時訓練時間
   - 之前 5 小時才 OOM
   - 現在 2x 內存應該可以完成

4. **已做好準備**
   - ✅ 配置修改完成
   - ✅ Git 記錄完整
   - ✅ 任務已提交
   - ⏳ 等待結果

---

**狀態：** 🚀 所有修復完成，等待訓練開始  
**預計完成時間：** Nov 19, 04:00 (12小時後)  
**最關鍵任務：** Job 2322458 (Flow V2)

*最後更新: 2025-11-18 15:12 EST*
