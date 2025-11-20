# 48M訓練狀態報告 - Nov 20, 2025

## 單任務訓練結果總結

### 1. Baseline (pwm_48M_baseline_l40s)
- **Job ID**: 2314141
- **狀態**: ✅ 部分完成 (因12小時限制中斷)
- **進度**: 3787/15000步 (25%)
- **最終性能**: R≈1254
- **評價**: 🌟優秀 - 達到預期性能
- **備註**: 無world model loss，僅訓練policy

### 2. Flow V1 (substeps=2, heun)
- **Job ID**: 2322456
- **狀態**: ✅ 接近完成
- **進度**: 18410/20000步 (92%)
- **最終性能**: R≈1229
- **評價**: 🌟優秀 - 與baseline相當
- **備註**: wm_loss=1.30 (穩定)

### 3. Flow V2 (substeps=4, heun) - **異常**
- **Job ID**: 2322458
- **狀態**: ⚠️ 訓練失敗
- **進度**: 12720/20000步 (64%)
- **最終性能**: R≈17 (極低!)
- **評價**: ❌ 失敗 - 性能極差
- **問題**: 可能的配置錯誤或超參數問題
- **備註**: wm_loss=1.30 (正常)，但reward極低
- **已提交**: Job 2344573 (30小時重新訓練)

### 4. Flow V3 (substeps=8, heun)
- **Job ID**: 2322459
- **狀態**: 🔄 訓練中斷 (需繼續)
- **進度**: 10893/20000步 (54%)
- **最終性能**: R≈1040
- **評價**: ✅ 良好 - 接近baseline性能
- **備註**: wm_loss=1.30 (穩定)
- **已提交**: Job 2344575 (30小時繼續訓練)

## Early Stopping
- ❌ **PWM沒有實現early stopping機制**
- 訓練會持續到指定的epochs數
- 所有中斷都是由於時間限制

## 多任務訓練修復記錄

### 問題1: Hydra ConfigCompositionException
- **錯誤**: `wandb.name`無法被override
- **原因**: config檔案中wandb section缺少name/notes欄位
- **修復**: ✅ 添加所有需要的wandb欄位到config_mt30.yaml和config_mt80.yaml
- **Commit**: fix: Add wandb.notes field and fix task names

### 問題2: metaworld API不兼容
- **錯誤**: `AttributeError: module 'metaworld' has no attribute 'MT30'`
- **原因**: 當前metaworld版本使用MT10/MT25/MT50，不是MT30/MT80
- **修復**: ✅ 更新為使用MT50
- **Commit**: fix: Update metaworld API to use MT50

### 問題3: 任務名稱版本不匹配
- **錯誤**: `ValueError: Task assembly-v2 not found`
- **原因**: MT50使用v3版本任務名稱 (assembly-v3)，但配置使用v2
- **修復**: ✅ 添加自動v2→v3轉換邏輯
- **Commit**: fix: Update metaworld API to use MT50 with v2->v3 task name conversion

### 問題4: OmegaConf hasattr問題
- **錯誤**: `MissingMandatoryValue: Missing mandatory value: episode_length`
- **原因**: `hasattr(cfg, 'key')`在OmegaConf中會觸發異常
- **修復**: ✅ 改用try-except處理
- **Commit**: fix: Replace hasattr with try-except for OmegaConf compatibility

## 當前運行的任務

### 單任務繼續訓練 (30小時)
1. **Flow V2 Continue** - Job 2344573
   - 重新訓練，診斷之前的低性能問題
   - 時限: 30小時
   
2. **Flow V3 Continue** - Job 2344575
   - 從~10893步繼續訓練到20000步
   - 時限: 30小時

### 多任務訓練 (24小時)
1. **MT Baseline** - Job 2344577
   - 48M參數，MT30 benchmark
   - Task: assembly-v3
   
2. **MT Flow V2** - Job 2344569
   - 48M參數，Flow dynamics (substeps=4, heun)
   - Task: assembly-v3

## Git提交記錄

```bash
# PWM子模塊提交
cd PWM
git commit -m "fix: Update metaworld API to use MT50 instead of MT30/MT80"
git commit -m "fix: Update metaworld API to use MT50 with v2->v3 task name conversion"  
git commit -m "fix: Replace hasattr with try-except for OmegaConf compatibility"
```

## 下一步行動

1. ✅ 監控Flow V2/V3繼續訓練 (30小時)
2. ✅ 監控多任務baseline和Flow V2訓練 (24小時)
3. 🔍 分析Flow V2為何失敗 - 檢查超參數配置
4. 📊 比較所有模型的最終性能
5. 📝 準備最終實驗報告

## 配置說明

### 單任務配置
- Environment: dflex_ant
- Horizon: 16
- Device: cuda:0
- Seed: 42

### 多任務配置
- Benchmark: MT50 (使用MT30 subset)
- Tasks: 30個Meta-World任務
- Task Dim: 64 (MT30) / 96 (MT80)
- Horizon: 16
- Device: cuda:0
- Seed: 42
