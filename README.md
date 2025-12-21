# 現代化深度學習五子棋訓練系統 (PyTorch Edition)

## 概述

這是一個基於 AlphaZero 架構的現代化五子棋 AI 訓練系統，已全面遷移至 **PyTorch** 框架，提供更高效的混合精度訓練和更穩定的多進程架構。

## 核心改進 (PyTorch vs TensorFlow)

### 1. 深度學習框架遷移

- ✅ **PyTorch**: 全面遷移至 PyTorch，代碼更清晰，調試更容易。
- ✅ **混合精度訓練 (AMP)**: 使用 `torch.cuda.amp`，顯著減少顯存佔用並加速訓練。
- ✅ **模型優化**: SE-ResNet 架構優化，包含 10 個殘差塊、128 濾波器和 SE 注意力機制。

### 2. 資料處理與訓練

- ✅ **高效數據管道**: 優化的 GPU 數據傳輸 (Pinned Memory)。
- ✅ **數據增強**: 8 種對稱變換並行處理。
- ✅ **優先級經驗回放**: 更高效的數據採樣。

### 3. 用戶體驗

- ✅ **人機對戰**: 新增 `play_vs_ai.py`，直接與 AI 對弈。
- ✅ **觀戰增強**: `watch_ai.py` 支持模型自動適配和播放控制。
- ✅ **可視化分析**: 改進的 `analyze_games.py`，支持中文圖表。

## 項目結構

```
gomoku/
├── core/                      # 核心模塊
│   ├── neural_net.py         # PyTorch SE-ResNet 模型
│   ├── mcts_batched.py       # 優化的批量 MCTS
│   ├── game_state.py         # 連接遊戲邏輯與神經網絡
│   └── game.py               # [保留] 遊戲核心邏輯
│
├── training/                  # 訓練模塊
│   ├── config.py             # 訓練配置
│   ├── replay_buffer.py      # 優先級經驗回放
│   └── game_logger.py        # 遊戲日誌系統
│
├── checkpoints/               # 模型和日誌保存目錄
├── train_pipeline_pytorch.py  # [New] PyTorch 主訓練腳本
├── play_vs_ai.py             # [New] 人機對戰腳本
├── watch_ai.py               # [Updated] AI 觀戰腳本
├── monitor_training.py       # 實時訓練監控
├── analyze_games.py          # 訓練後分析工具
└── main.py                   # 雙人對戰 UI
```

## 快速開始

### 1. 快速測試 (驗證環境)

運行以下命令進行快速端到端測試：

```bash
python train_pipeline_pytorch.py -f -i 20 -g 20
```

- `-f`: 快速測試配置 (小模型, 少量 MCTS)
- 預計時間: ~5-10 分鐘 (NVIDIA GPU)

### 2. 完整訓練 (Training)

```bash
python reset_training.py  # 清理舊數據 (可選)
python train_pipeline_pytorch.py -i 1000 -g 100
```

- 使用完整配置 (10 ResBlocks, 128 Filters, 200 MCTS sims)
- 訓練 1000 次迭代，每次迭代 100 局自我對弈。

### 3. 人機對戰 (Human vs AI) 🎉

訓練完成後，您可以親自挑戰 AI：

```bash
python play_vs_ai.py
```

- **功能**:
  - 自動加載最新的訓練模型。
  - 默認人類執黑先行。
  - 雖然初期 AI 可能很弱，但隨著訓練進行，它會變得非常強大。

### 4. 觀看 AI 自我對弈

```bash
python watch_ai.py
```

- **新功能**:
  - **空白鍵 (Space)**: 暫停/繼續自動播放。
  - **R 鍵**: 重新開始。

## 訓練監控

### 實時監控

在訓練過程中，開啟新終端運行：

```bash
python monitor_training.py
```

這將實時顯示損失值下降趨勢、當前迭代信息和評估結果。

### 事後分析

訓練結束後，生成詳細的分析圖表：

```bash
python analyze_games.py
```

生成的圖表將保存在 `logs/games/` 目錄下，包含勝率變化、步數趨勢和策略熵分析。

## 常見問題 (FAQ)

### Q: 為什麼以前的 `train_pipeline.py` 不能用了？

A: 為了更好的性能和維護性，我們全面遷移到了 PyTorch。舊的 TensorFow 腳本現已歸檔，請使用 `train_pipeline_pytorch.py`。

### Q: `watch_ai.py` 報錯 "Weights only load failed"？

A: 這是 `torch.load` 的安全機制。我們已更新了腳本，現在它優先嘗試加載安全的 `latest_model.pth`，或者您可以忽略該警告（我們加載的是自己訓練的模型，是安全的）。

### Q: 如何切換回 TensorFlow？

A: 雖然不建議，但舊代碼仍保留在項目中。不過新功能（如 `play_vs_ai.py`）僅支持 PyTorch 模型。

---

**祝您訓練愉快！打造最強五子棋 AI！** 🚀
