# 開發者指南 (Developer Guide)

## 項目概覽 (Project Overview)

**Gomoku AI** 是一個基於深度強化學習 (Deep Reinforcement Learning) 的五子棋人工智慧項目。其核心算法靈感來自 AlphaZero，結合了深度神經網絡 (Deep Neural Networks) 和蒙特卡洛樹搜索 (MCTS) 進行自我對弈訓練。

### 核心目標

- 訓練一個達到人類頂尖水平的五子棋 AI。
- 實現高效、穩定的訓練流程。
- 提供清晰的代碼結構和文檔，便於研究和二次開發。

### 技術棧 (Tech Stack)

- **語言**: Python 3.9+
- **深度學習框架**: PyTorch (遷移自 TensorFlow)
- **計算加速**: CUDA (NVIDIA GPU), Mixed Precision (AMP)
- **數據分析**: Pandas, Matplotlib

## 核心功能 (Core Features)

### 1. 深度神經網絡 (SE-ResNet)

- **架構**: 殘差網絡 (ResNet) 結合 Squeeze-and-Excitation (SE) 注意力機制。
- **輸入**: $15 \times 15 \times 3$ (本方棋子, 對手棋子, 當前對手標識)。
- **輸出**:
  - **策略頭 (Policy Head)**: 落子概率分佈 ($15 \times 15$ + 1 pass)。
  - **價值頭 (Value Head)**: 當前局面勝率估計 (-1.0 至 1.0)。

### 2. 蒙特卡洛樹搜索 (MCTS)

- **PUCT 算法**: 平衡探索 (Exploration) 與利用 (Exploitation)。
- **虛擬損失 (Virtual Loss)**: 支持多線程/並行搜索。
- **批量推理 (Batched MCTS)**: 優化神經網絡推理效率。

### 3. 自我對弈訓練 (Self-Play Training)

- **多進程架構**: 並行執行多個遊戲實例收集數據。
- **數據增強**: 利用棋盤對稱性 (旋轉、翻轉) 擴充訓練數據 (8 倍)。
- **優先級經驗回放 (PER)**: 提高訓練數據利用率。

## 系統架構 (System Architecture)

```mermaid
graph TD
    subgraph Training Loop [訓練循環]
        SP[自我對弈 (Self-Play)] --> |生成遊戲數據| DA[數據增強 (Data Augmentation)]
        DA --> |(s, p, v)| RB[經驗回放緩衝區 (Replay Buffer)]
        RB --> |Batch 採樣| TRAIN[模型訓練 (Training)]
        TRAIN --> |更新權重| SP
        TRAIN --> |定期評估| EVAL[評估 (Evaluation)]
    end

    subgraph Components [核心組件]
        NN[SE-ResNet 模型]
        MCTS[MCTS 搜索引擎]
        AMP[混合精度 (GradScaler)]
    end

    SP -.-> MCTS
    MCTS -.-> NN
    TRAIN -.-> AMP
```

## 詳細功能與規則 (Detailed Features)

### 訓練配置

所有超參數均在 `training/config.py` 中定義。關鍵參數包括：

- `BOARD_SIZE`: 15
- `NUM_WORKERS`:並行進程數
- `LEARNING_RATE`: 初始學習率 (配合 StepLR 衰減)
- `USE_AMP`: 是否啟用混合精度

### 數據流 (Data Flow)

1.  **State**: `GameState` 對象 -> `to_input()` -> Numpy `(H,W,C)`
2.  **Model Input**: Numpy `(H,W,C)` -> Transpose -> Tensor `(B,C,H,W)`
3.  **Model Output**: Policy Logits, Value -> Loss Calculation

### 驗證與測試

- **快速測試**: `python train_pipeline_pytorch.py --fast-test`
- **可視化**: `analyze_games.py` 分析訓練日誌和指標。

## 變更日誌 (Changelog)

### [2025-12-21] PyTorch Migration

- **新增**: 完整的 PyTorch 訓練流程 `train_pipeline_pytorch.py`。
- **改進**: 啟用混合精度訓練，顯著提升 NVIDIA GPU 上的性能。
- **優化**: 修復了多進程數據序列化問題。
- **文檔**: 重組項目結構，建立 `docs/`, `legacy/` 目錄。

---

_詳細更新請參閱 `docs/LOG.md`_
