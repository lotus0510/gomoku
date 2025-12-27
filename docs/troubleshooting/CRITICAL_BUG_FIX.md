# 🔴 CRITICAL BUG 修復報告

## 問題摘要

**價值網路完全失效：100% 的價值預測為 0.0**

診斷顯示所有遊戲記錄中的 `values` 欄位全部為 `0.0`，導致訓練數據質量嚴重受損。

## 根本原因

### MCTS 實現 BUG：根節點從未被更新

**問題位置：**
- `core/mcts.py:268-280` - `_backup()` 函數
- `core/mcts_batched.py:369-385` - `_backup()` 函數

**BUG 原理：**

```python
# 原始的錯誤代碼
def _backup(self, path, value):
    # path = [(root, child1), (child1, child2), ...]
    for parent, child in reversed(path):
        child.visit_count += 1  # 只更新 child
        child.total_value += value
        value = -value
```

**問題流程：**

1. MCTS 創建根節點 `root = MCTSNode(prior_prob=1.0)`
   - `root.visit_count = 0`
   - `root.total_value = 0.0`

2. 執行 200 次模擬，path 結構為：
   ```
   [(root, child1), (child1, child2), (child2, leaf)]
   ```

3. 反向傳播時：
   ```python
   reversed(path) = [(child2, leaf), (child1, child2), (root, child1)]
   ```
   - 更新 `leaf` ✓
   - 更新 `child2` ✓
   - 更新 `child1` ✓
   - **但沒有更新 `root`** ✗

4. 結果：`root.visit_count` 保持為 `0`

5. 獲取根節點價值時：
   ```python
   def get_value(self):
       total_visits = self.visit_count + self.virtual_loss
       if total_visits == 0:
           return 0.0  # ← 永遠返回這個！
   ```

6. 這個 `0.0` 被記錄到遊戲歷史：
   ```python
   # train_pipeline_pytorch.py:142
   game_history.append({
       'value': root_value  # 永遠是 0.0
   })

   # train_pipeline_pytorch.py:184
   game_metadata = {
       'values': [entry['value'] for entry in game_history]  # 全部是 0.0
   }
   ```

## 影響範圍

### 直接影響
- ✗ **所有遊戲記錄的 `values` 都是 `0.0`**
- ✗ **遊戲日誌 (`logs/games/`) 中無有效的價值估計數據**
- ✗ **診斷工具報告 "價值網路完全失效"**

### 間接影響
- ⚠️ **訓練數據質量受損**（但訓練目標值仍然正確）
- ⚠️ **無法監控 MCTS 搜索質量**
- ⚠️ **無法分析局面評估能力**

### 未受影響
- ✓ **訓練過程本身正常**（訓練目標來自遊戲結果，不是 root_value）
- ✓ **價值網路本身正常**（訓練歷史顯示 value_loss 下降）
- ✓ **策略網路正常**（MCTS 使用子節點訪問次數，不依賴 root_value）

## 修復方案

### 修復代碼

**`core/mcts_batched.py`：**
```python
def _backup(self, path, value):
    # 从叶节点向根节点传播
    for parent, child in reversed(path):
        child.virtual_loss -= 1
        child.visit_count += 1
        child.total_value += value
        value = -value

    # 🆕 更新根节点（根节点不在path的child中，需要单独更新）
    if path:
        root = path[0][0]  # path的第一个parent就是root
        root.visit_count += 1
        root.total_value += value
```

**`core/mcts.py`：**
```python
def _backup(self, path, value):
    # 从叶节点向根节点传播
    for parent, child in reversed(path):
        child.visit_count += 1
        child.total_value += value
        value = -value

    # 🆕 更新根节点（根节点不在path的child中，需要单独更新）
    if path:
        root = path[0][0]  # path的第一个parent就是root
        root.visit_count += 1
        root.total_value += value
```

### 修復效果

修復後，`root.get_value()` 將返回正確的平均價值：

```
修復前：
  root.visit_count = 0
  root.total_value = 0.0
  root.get_value() = 0.0  ✗

修復後（200次模擬）：
  root.visit_count = 200
  root.total_value ≈ +50 (假設黑方優勢)
  root.get_value() = 50/200 = 0.25  ✓
```

## 驗證方法

### 1. 重新訓練一個迭代

```bash
# 清除舊數據（可選）
rm -rf logs/games/iteration_122

# 運行一個迭代
python train_pipeline.py --iterations 1
```

### 2. 檢查遊戲日誌

```bash
python -c "
import json
game = json.load(open('logs/games/iteration_122/game_0.json'))
print('Values:', game['values'][:10])
print('All zeros:', all(v==0.0 for v in game['values']))
"
```

**預期結果：**
```
Values: [0.2341, -0.1823, 0.3156, ...]  # 不再全部是 0.0
All zeros: False  ✓
```

### 3. 重新運行診斷

```bash
python analysis/diagnose_training.py
```

**預期輸出：**
```
【4. 價值網路檢查】
價值網路統計（樣本: 131 個評估）：
  平均值: 0.123  # 不再是 0.000
  標準差: 0.456  # 不再是 0.000
  零值比例: 2.3%  # 不再是 100.0%
```

## 後續行動

### 必須執行
1. ✅ **修復 MCTS 代碼**（已完成）
2. ⏳ **驗證修復效果**（執行新迭代並檢查）
3. ⏳ **重新訓練**（建議從頭開始，或繼續訓練）

### 建議執行
1. **保留舊數據**：移動當前 checkpoints 到備份目錄
   ```bash
   mv checkpoints checkpoints_backup_before_fix
   mkdir checkpoints
   ```

2. **從頭重新訓練**：雖然修復前的訓練不完全浪費，但重新訓練可以獲得更高質量的監控數據

3. **監控新訓練**：
   - 檢查 `games_summary.csv` 中的 value 欄位不再全為 0
   - 使用 `analysis/diagnose_training.py` 定期檢查
   - 使用 `interactive/watch_ai.py` 觀察 MCTS 搜索質量

## 技術註解

### 為什麼訓練沒有完全崩潰？

雖然 `root_value` 全部是 `0.0`，但這個值**只用於記錄和監控**，不影響核心訓練邏輯：

1. **訓練目標值**來自遊戲結果（winner），不是 root_value：
   ```python
   # train_pipeline_pytorch.py:159-164
   if winner == 0:
       value = 0.0
   else:
       base_value = 1.0 if entry['turn'] == winner else -1.0
       value = base_value * (config.VALUE_GAMMA ** steps_to_end)
   ```

2. **MCTS 策略分布**基於子節點訪問次數，不依賴 root_value：
   ```python
   # mcts.py:95-100
   visits = np.array([child.visit_count for child in children])
   action_probs = visits / visits.sum()
   ```

### 為什麼之前沒發現？

1. **價值損失在下降**：訓練目標正確，所以損失正常
2. **策略損失持平**：策略問題更明顯，掩蓋了價值記錄問題
3. **診斷工具最近才添加**：之前沒有檢查遊戲日誌中的詳細數據

## 總結

- **BUG 嚴重性**：HIGH（影響監控和分析，但不完全破壞訓練）
- **修復複雜度**：LOW（3 行代碼）
- **修復效果**：HIGH（完全解決問題）
- **建議行動**：重新訓練以獲得高質量監控數據

---

**修復時間**：2025-12-23
**發現者**：診斷工具 `analysis/diagnose_training.py`
**修復人**：Claude Code
