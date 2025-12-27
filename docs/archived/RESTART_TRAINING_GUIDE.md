# 重新訓練指南

> Bug 已修復，準備從頭開始訓練

---

## 📋 執行步驟

### 1. 停止當前訓練

如果訓練正在運行：
```bash
Ctrl+C
```

### 2. 備份受影響的檢查點

保留診斷記錄：
```bash
# Windows
move checkpoints checkpoints_backup_scheduler_bug

# Linux/Mac
mv checkpoints checkpoints_backup_scheduler_bug
```

### 3. 創建新檢查點目錄

```bash
mkdir checkpoints
```

### 4. 啟動全新訓練

```bash
python train_pipeline_pytorch.py
```

應該看到：
```
✅ 使用優化配置（默認）
關鍵改進：
  • DIRICHLET_EPSILON: 0.35 → 0.15 (-57%)
  • LR_DECAY_STEPS: 100 → 200 (+100%)
  • REPLAY_BUFFER_SIZE: 400K → 800K (+100%)
  • C_PUCT: 2.0 → 1.5 (-25%)
  • BATCH_SIZE: 512 → 1024 (+100%)

開始從頭訓練...
```

---

## 🎯 預期效果（Bug 修復後）

### 前 10 次迭代
```
策略損失: 5.42 → 5.3 (明顯下降，不是 0.01 級別)
價值損失: 應穩定下降
梯度範數: 保持 > 0.5
```

### 30 次迭代
```
策略損失: < 5.0 ✅ 達成第一目標
價值標準差: > 0.5
遊戲長度: 30+ 步
```

### 50 次迭代
```
策略損失: < 4.8
價值標準差: 0.5-0.7 (健康範圍)
遊戲開始出現防守
```

### 100 次迭代
```
策略損失: < 4.5
完整的攻防對局 (50-100 步)
可以戰勝隨機玩家 (100% 勝率)
```

---

## 🔍 關鍵監控指標

### 每 10 次迭代檢查

**策略損失**：
- ✅ 每 10 次迭代應降低 0.1-0.2
- ❌ 如果降低 < 0.05，可能有問題

**價值損失**：
- ✅ 應穩定下降
- ❌ 如果上升，檢查 VALUE_LOSS_WEIGHT

**梯度範數**：
- ✅ 保持在 0.5-5.0 範圍
- ❌ 如果 < 0.5，梯度消失

**價值標準差**：
- ✅ 保持 > 0.5
- ❌ 如果持續下降到 < 0.3，價值網路崩潰

---

## ⚠️ 異常處理

### 迭代 30 檢查點

**如果策略損失仍 > 5.2**：
```python
# 進一步降低探索
config.DIRICHLET_EPSILON = 0.10  # 從 0.15 → 0.10
config.C_PUCT = 1.0              # 從 1.5 → 1.0
```

**如果價值損失持續上升**：
```python
# 增加價值損失權重
config.VALUE_LOSS_WEIGHT = 1.5   # 從 1.0 → 1.5
```

**如果梯度消失**：
```python
# 調整學習率或批次大小
config.LEARNING_RATE = 0.0002    # 從 0.0001 → 0.0002
# 或
config.BATCH_SIZE = 512          # 從 1024 → 512 (如果 GPU 記憶體不足)
```

---

## 📊 與之前訓練的對比

### 會話 #1: 原始配置 (迭代 1-226)
- 探索過高 (DIRICHLET_EPSILON=0.35)
- 策略損失停滯在 5.28
- 價值標準差崩潰到 0.11

### 會話 #2: Bug 影響 (迭代 1-22)
- 優化配置未生效（調度器 Bug）
- 策略損失幾乎無改善 (5.416 → 5.402)
- 價值損失惡化 (0.598 → 0.946)

### 會話 #3: Bug 修復後（現在）
- ✅ 優化配置正確應用
- ✅ 調度器正確同步
- ✅ 預期正常學習曲線

---

## 📝 訓練日誌

建議同時監控：

```bash
# 終端 1: 訓練
python train_pipeline_pytorch.py

# 終端 2: 查看訓練歷史
watch -n 10 'tail -20 checkpoints/training_history.json'

# 或使用 Python 查看
python -c "
import json
with open('checkpoints/training_history.json') as f:
    h = json.load(f)
print(f'迭代: {len(h[\"policy_loss\"])}')
print(f'策略損失: {h[\"policy_loss\"][-1]:.3f}')
print(f'價值損失: {h[\"value_loss\"][-1]:.3f}')
print(f'價值標準差: {h[\"value_std\"][-1]:.3f}')
"
```

---

## ✅ 成功標準

### 迭代 30
- [x] 策略損失 < 5.0
- [x] 價值標準差 > 0.5
- [x] 梯度範數 > 0.5

### 迭代 50
- [x] 策略損失 < 4.8
- [x] 遊戲長度 > 30 步
- [x] 開始出現防守

### 迭代 100
- [x] 策略損失 < 4.5
- [x] 完整攻防對局
- [x] 戰勝隨機玩家 100%

---

## 🔗 相關文檔

- `TRAINING_FAILURE_REPORT.md` - 完整訓練記錄和 Bug 分析
- `OPTIMIZED_CONFIG.md` - 優化配置說明
- `loss_analysis.md` - 損失分析報告
- `CONFIG_INVESTIGATION_REPORT.md` - Bug 技術診斷報告

---

**準備開始了嗎？**

```bash
# 執行這些命令開始訓練
move checkpoints checkpoints_backup_scheduler_bug
mkdir checkpoints
python train_pipeline_pytorch.py
```

祝訓練順利！ 🚀
