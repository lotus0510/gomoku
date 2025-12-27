# 獎勵機制的視角解析

## 核心問題：從誰的角度計算獎勵？

**答案**：**從當前玩家的角度**，不是固定從黑棋或白棋的角度。

---

## 代碼實現

### 關鍵代碼（train_pipeline_pytorch.py:162）

```python
# 遊戲結束後，為每個歷史位置計算價值
for i, entry in enumerate(game_history):
    steps_to_end = len(game_history) - i - 1

    if winner == 0:
        value = 0.0  # 平局
    else:
        # 關鍵！從當前玩家角度
        base_value = 1.0 if entry['turn'] == winner else -1.0
        value = base_value * (VALUE_GAMMA ** steps_to_end)
```

### 變數說明

- `entry['turn']`：該步驟下棋的玩家（1=黑棋，2=白棋）
- `winner`：遊戲的最終獲勝者（1=黑棋獲勝，2=白棋獲勝，0=平局）
- `base_value`：
  - +1.0：該步驟的玩家 == 獲勝者（這個玩家贏了）
  - -1.0：該步驟的玩家 != 獲勝者（這個玩家輸了）

---

## 具體範例

假設一局遊戲 10 步結束，黑棋獲勝（winner = 1），VALUE_GAMMA = 0.995：

| 步數 | 玩家 | entry['turn'] | winner | turn==winner? | base_value | steps_to_end | 最終 value |
|-----|------|--------------|--------|---------------|------------|--------------|-----------|
| 1 | 黑 | 1 | 1 | ✓ | +1.0 | 9 | +0.956 |
| 2 | 白 | 2 | 1 | ✗ | -1.0 | 8 | -0.961 |
| 3 | 黑 | 1 | 1 | ✓ | +1.0 | 7 | +0.966 |
| 4 | 白 | 2 | 1 | ✗ | -1.0 | 6 | -0.971 |
| 5 | 黑 | 1 | 1 | ✓ | +1.0 | 5 | +0.975 |
| 6 | 白 | 2 | 1 | ✗ | -1.0 | 4 | -0.980 |
| 7 | 黑 | 1 | 1 | ✓ | +1.0 | 3 | +0.985 |
| 8 | 白 | 2 | 1 | ✗ | -1.0 | 2 | -0.990 |
| 9 | 黑 | 1 | 1 | ✓ | +1.0 | 1 | +0.995 |
| 10 | 白 | 2 | 1 | ✗ | -1.0 | 0 | -1.000 |

### 觀察

1. **黑棋的所有步驟**：價值都是正數（+0.956 ~ +0.995）
   - 因為黑棋贏了，所以黑棋的每個決策都被標記為「好的」

2. **白棋的所有步驟**：價值都是負數（-0.961 ~ -1.000）
   - 因為白棋輸了，所以白棋的每個決策都被標記為「壞的」

3. **越接近終點，價值絕對值越大**
   - 第 10 步：|value| = 1.000（最關鍵）
   - 第 1 步：|value| = 0.956（較不關鍵）

---

## 為什麼從當前玩家角度？

### 1. 神經網路只有一個

不是分別訓練「黑棋網路」和「白棋網路」，而是用**同一個網路**：

```python
# 黑棋下棋時
value = model(black_perspective_state)  # 預測黑棋勝率

# 白棋下棋時
value = model(white_perspective_state)  # 預測白棋勝率
```

**同一個模型**通過觀察不同的棋盤狀態（翻轉顏色）來為兩方服務。

### 2. 狀態表示的設計

查看 `core/game_state.py` 的 `to_input()` 方法：

```python
def to_input(self):
    """轉換為神經網路輸入（3 通道）"""
    input_array = np.zeros((self.size, self.size, 3), dtype=np.float32)

    # 通道 0: 當前玩家的棋子
    # 通道 1: 對手的棋子
    # 通道 2: 當前玩家指示器（全1或全0）

    current_player = self.get_current_player()

    for i in range(self.size):
        for j in range(self.size):
            if self.board[i, j] == current_player:
                input_array[i, j, 0] = 1.0  # 我的棋子
            elif self.board[i, j] != 0:
                input_array[i, j, 1] = 1.0  # 對手的棋子

    input_array[:, :, 2] = float(current_player - 1)

    return input_array
```

**關鍵**：
- 通道 0 永遠是「當前玩家的棋子」
- 通道 1 永遠是「對手的棋子」
- 網路不知道「黑」或「白」，只知道「我」和「對手」

### 3. AlphaZero 原版設計

這是 AlphaZero 論文的標準做法：

> "The neural network takes the board position as input and outputs a value v, which is the expected outcome from the current player's perspective."

**從當前玩家角度的優勢**：
- ✅ 一個網路服務雙方
- ✅ 訓練數據量翻倍（每個位置從兩個角度看）
- ✅ 對稱性和泛化性更好

---

## 訓練時如何處理？

### 數據生成

```python
# 黑棋第 1 步
state_black_view = game.get_state()  # 黑棋視角
# → 通道0=黑棋子，通道1=白棋子
# → 如果黑棋贏，value = +0.956

# 白棋第 2 步
state_white_view = game.get_state()  # 白棋視角
# → 通道0=白棋子，通道1=黑棋子
# → 如果黑棋贏（白棋輸），value = -0.961
```

### 神經網路學習

網路學習的模式：

**輸入**：
```
通道 0: 我的棋子位置
通道 1: 對手的棋子位置
通道 2: 當前玩家標記
```

**輸出**：
```
value: [-1, +1]
  +1 = 我（當前玩家）會贏
  -1 = 我（當前玩家）會輸
   0 = 平局
```

---

## 實戰驗證

### 測試場景

假設訓練後的模型：

```python
import torch
from core.neural_net import create_enhanced_model
from core.game_state import GameState

model = create_enhanced_model()
model.load_state_dict(torch.load('checkpoints/best_model.pth'))
model.eval()

# 創建一個黑棋優勢局面
game = GameState()
# ... 設置一個黑棋即將獲勝的局面

# 從黑棋角度評估
state_black = game.to_input()
_, value_black = model(torch.FloatTensor(state_black).unsqueeze(0))
print(f"黑棋視角的價值: {value_black.item():.3f}")  # 預期: +0.8 左右

# 切換到白棋下棋
game.make_move(some_move)

# 從白棋角度評估（同一局面）
state_white = game.to_input()
_, value_white = model(torch.FloatTensor(state_white).unsqueeze(0))
print(f"白棋視角的價值: {value_white.item():.3f}")  # 預期: -0.8 左右
```

**預期結果**：
- 黑棋視角：+0.8（黑棋優勢）
- 白棋視角：-0.8（白棋劣勢）
- 同一個局面，不同視角，價值相反

---

## 常見誤解

### ❌ 誤解 1：「網路固定預測黑棋勝率」

**錯誤理解**：
```
value = 0.8 → 黑棋 80% 會贏
value = -0.8 → 黑棋 20% 會贏
```

**正確理解**：
```
value = 0.8 → 當前玩家 80% 會贏
value = -0.8 → 當前玩家 20% 會贏
```

### ❌ 誤解 2：「需要兩個網路」

**錯誤想法**：分別訓練黑棋網路和白棋網路

**正確做法**：一個網路 + 視角轉換
- 黑棋下棋：輸入「黑棋視角狀態」
- 白棋下棋：輸入「白棋視角狀態」

### ❌ 誤解 3：「白棋總是負值」

**錯誤理解**：白棋的訓練標籤總是負數

**正確理解**：
- 白棋獲勝的局 → 白棋所有步驟都是正值
- 白棋失敗的局 → 白棋所有步驟都是負值

---

## 數學驗證

### 對稱性檢查

如果實現正確，應該滿足：

```python
# 對於同一個局面
value_from_black = model(state_black_perspective)
value_from_white = model(state_white_perspective)

# 應該有（近似）
value_from_black ≈ -value_from_white
```

因為：
- 黑棋的優勢 = 白棋的劣勢
- 局面是零和博弈

### 實際檢查腳本

```python
#!/usr/bin/env python3
"""驗證視角對稱性"""
import torch
import numpy as np
from core.neural_net import create_enhanced_model
from core.game_state import GameState

model = create_enhanced_model()
model.load_state_dict(torch.load('checkpoints/best_model.pth'))
model.eval()

# 創建測試局面
game = GameState()
game.make_move(7, 7)  # 黑棋
game.make_move(7, 8)  # 白棋
game.make_move(8, 7)  # 黑棋

# 當前是白棋下
state_white = game.to_input()
_, value_white = model(torch.FloatTensor(state_white).unsqueeze(0))

# 模擬黑棋視角（需要手動翻轉）
state_white_array = state_white.copy()
state_black_view = np.zeros_like(state_white_array)
state_black_view[:, :, 0] = state_white_array[:, :, 1]  # 對手 → 我
state_black_view[:, :, 1] = state_white_array[:, :, 0]  # 我 → 對手
state_black_view[:, :, 2] = 1 - state_white_array[:, :, 2]  # 翻轉玩家

_, value_black = model(torch.FloatTensor(state_black_view).unsqueeze(0))

print(f"白棋視角價值: {value_white.item():.4f}")
print(f"黑棋視角價值: {value_black.item():.4f}")
print(f"總和: {value_white.item() + value_black.item():.4f}")
print(f"對稱性檢查: {'✓ 通過' if abs(value_white.item() + value_black.item()) < 0.1 else '✗ 失敗'}")
```

---

## 總結

### 核心要點

1. **獎勵視角**：從**當前玩家**角度，不是固定黑/白棋
2. **價值標籤**：
   - 獲勝方的所有步驟：正值
   - 失敗方的所有步驟：負值
3. **神經網路**：一個網路通過視角轉換服務雙方
4. **狀態表示**：通道 0 永遠是「當前玩家」，通道 1 永遠是「對手」

### 為什麼這樣設計？

| 方案 | 優點 | 缺點 |
|-----|------|------|
| **當前玩家視角**（採用） | 一個網路、數據翻倍、對稱性好 | 需要視角轉換 |
| 固定黑棋視角 | 實現簡單 | 需要兩個網路或性能差 |

### 驗證方法

運行上面的對稱性檢查腳本，確認：
```
value_black + value_white ≈ 0
```

---

**相關文檔**：
- [TRAINING_CONCEPTS.md](TRAINING_CONCEPTS.md) - 訓練概念
- [DEVELOPER_GUIDE.md](DEVELOPER_GUIDE.md) - 代碼結構
- `train_pipeline_pytorch.py:152-170` - 獎勵計算代碼
- `core/game_state.py:to_input()` - 狀態表示

**最後更新**: 2025-12-23
