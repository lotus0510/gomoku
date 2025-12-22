# 分析工具目錄

這個目錄包含所有用於訓練分析和可視化的工具。

## 📊 圖表生成工具

### `plot_comprehensive.py` - 綜合圖表分析（⭐ 推薦）

**功能**: 生成 10 張詳細的分析圖表

**用法**:
```bash
python analysis/plot_comprehensive.py
```

**生成的圖表**:

1. **訓練指標總覽** (`1_training_metrics.png`)
   - 總損失變化
   - 梯度範數趨勢
   - 學習率調度
   - 對隨機玩家勝率

2. **損失分解分析** (`2_loss_breakdown.png`)
   - 三種損失對比
   - 策略損失詳細
   - 價值損失詳細
   - 損失標準差

3. **梯度健康度分析** (`3_gradient_analysis.png`)
   - 梯度範數時間序列
   - 梯度範數分佈
   - 梯度穩定性
   - 梯度健康度統計

4. **價值網路分析** (`4_value_network_metrics.png`)
   - 價值 MAE
   - 價值標準差
   - 策略 Top-1 機率
   - 策略熵

5. **遊戲長度分佈** (`5_game_length_distribution.png`)
   - 整體長度分佈
   - 按迭代的長度變化
   - 長度分類統計
   - 最近10次迭代箱型圖

6. **勝率趨勢分析** (`6_win_rate_trends.png`)
   - 黑白勝率對比
   - 勝率堆疊圖
   - 勝率不平衡度
   - 最近趨勢

7. **策略演化分析** (`7_strategy_evolution.png`)
   - 超短局比例趨勢
   - 短局比例趨勢
   - 策略演化階段圖
   - Edge-Rush 健康度評分

8. **遊戲模式分析** (`8_game_patterns.png`)
   - 長度vs勝率散點圖
   - 按長度分類的勝率
   - 最近10次迭代熱圖
   - 統計摘要表

9. **綜合儀表板** (`9_comprehensive_dashboard.png`)
   - 9 個關鍵指標的總覽
   - 一頁看清整體訓練狀態

10. **指標相關性分析** (`10_correlation_matrix.png`)
    - 各指標間的相關性矩陣
    - 發現潛在關聯

**輸出位置**: `analysis/plots/`

---

## 📈 實時監控工具

### `monitor_training.py` - 實時訓練監控

**功能**: 實時顯示訓練進度

**用法**:
```bash
python analysis/monitor_training.py
```

**顯示內容**:
- 已完成迭代數
- 最新損失值
- 損失歷史（最近10次）
- 評估結果
- 統計信息

**特點**:
- 每5秒自動刷新
- 清屏顯示，易於閱讀
- Ctrl+C 退出

---

## 📊 進展分析工具

### `analyze_progress.py` - 訓練進展分析

**功能**: 詳細分析訓練進展，判斷是否需要繼續訓練

**用法**:
```bash
python analysis/analyze_progress.py
```

**分析內容**:
1. **梯度範數分析**
   - 最近10次迭代的梯度值
   - 健康度評估（0.5-5.0）

2. **遊戲長度變化** ⭐ 關鍵指標
   - 每次迭代的平均步數
   - 超短局（<15步）比例
   - 趨勢判斷

3. **價值網路指標**
   - 價值MAE變化
   - 改善情況

4. **總結建議**
   - 當前問題診斷
   - 是否繼續訓練
   - 預期效果

**輸出**: 文字報告，包含建議

---

## 🎮 遊戲可視化工具

### `visualize_game.py` - 遊戲棋譜可視化

**功能**: 可視化特定遊戲的棋譜

**用法**:
```bash
python analysis/visualize_game.py
```

**交互方式**:
1. 輸入迭代次數
2. 輸入遊戲編號
3. 顯示棋盤狀態和移動序列

**顯示內容**:
- ASCII 棋盤
- 每步移動詳情
- 勝負結果
- MCTS 統計信息（如果有）

---

## 🔧 使用建議

### 訓練中

**邊訓練邊監控**:
```bash
# 終端 1：啟動訓練
python train_pipeline_pytorch.py

# 終端 2：實時監控
python analysis/monitor_training.py
```

### 訓練後

**生成完整分析**:
```bash
# 1. 生成所有圖表（推薦）
python analysis/plot_comprehensive.py

# 2. 查看詳細文字分析
python analysis/analyze_progress.py

# 3. 查看特定遊戲
python analysis/visualize_game.py
```

---

## 📁 輸出文件

```
analysis/
├── plots/                      # 圖表輸出目錄
│   ├── 1_training_metrics.png
│   ├── 2_loss_breakdown.png
│   ├── 3_gradient_analysis.png
│   ├── 4_value_network_metrics.png
│   ├── 5_game_length_distribution.png
│   ├── 6_win_rate_trends.png
│   ├── 7_strategy_evolution.png
│   ├── 8_game_patterns.png
│   ├── 9_comprehensive_dashboard.png
│   └── 10_correlation_matrix.png
│
├── plot_comprehensive.py       # 圖表生成工具
├── monitor_training.py         # 實時監控
├── analyze_progress.py         # 進展分析
├── visualize_game.py           # 遊戲可視化
└── README.md                   # 本文件
```

---

## 💡 常見用途

### 檢查訓練健康度
```bash
python analysis/analyze_progress.py
```
查看梯度是否穩定，是否需要調整學習率。

### 診斷 Edge-Rush 問題
```bash
python analysis/plot_comprehensive.py
```
查看圖表 7（策略演化）和圖表 8（遊戲模式）。

### 決定是否繼續訓練
```bash
python analysis/analyze_progress.py
```
查看「總結建議」部分的建議。

### 對比不同配置效果
1. 訓練配置 A → 生成圖表 → 保存到 `plots_configA/`
2. 訓練配置 B → 生成圖表 → 保存到 `plots_configB/`
3. 對比兩組圖表

---

## 🎯 快速開始

**第一次使用**:
```bash
# 確保已安裝依賴
pip install matplotlib seaborn pandas numpy

# 生成所有圖表
python analysis/plot_comprehensive.py

# 查看結果
cd analysis/plots
# 使用圖片查看器打開 PNG 文件
```

**定期檢查**:
```bash
# 每訓練 10-20 次迭代後
python analysis/plot_comprehensive.py
python analysis/analyze_progress.py
```

---

## ❓ 常見問題

**Q: 圖表生成失敗？**
- 確保 `checkpoints/training_history.json` 存在
- 確保 `logs/games/games_summary.csv` 存在
- 檢查是否有訓練數據

**Q: 中文顯示亂碼？**
- Windows: 確保系統安裝了「微軟雅黑」字體
- Linux: 安裝中文字體包
- macOS: 通常自帶中文字體

**Q: 圖表太大/太小？**
- 編輯 `plot_comprehensive.py`
- 修改 `figsize=(16, 10)` 參數
- 或修改 `dpi=300` 參數

---

## 🔗 相關文檔

- 訓練指南: `docs/TRAINING_GUIDE.md`
- 獎勵機制: `docs/REWARD_MECHANISM.md`
- 性能優化: `docs/PERFORMANCE.md`
