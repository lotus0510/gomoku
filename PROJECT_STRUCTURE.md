# 五子棋 AI 專案結構

## 核心訓練文件

### 主要訓練流程

- **`train_pipeline_pytorch.py`** ✨ - PyTorch 訓練流程（主要使用）
- **`train_pytorch.bat`** - 快速啟動訓練腳本

### 遊戲和 AI

- **`game.py`** - 五子棋遊戲核心邏輯
- **`main.py`** - 遊戲主入口（人機對弈）
- **`watch_ai.py`** - 觀看 AI 自我對弈

## 目錄結構

```
gomoku/
├── core/                          # 核心模組
│   ├── neural_net.py             # PyTorch 神經網絡（SE-ResNet）
│   ├── mcts.py                   # 蒙特卡洛樹搜索
│   ├── mcts_batched.py           # 批量 MCTS（性能優化）
│   ├── game_state.py             # 遊戲狀態管理
│   └── data_augmentation.py      # 數據增強（8種對稱）
│
├── training/                      # 訓練工具
│   ├── config.py                 # 訓練配置
│   ├── replay_buffer.py          # 優先級經驗回放
│   └── game_logger.py            # 遊戲日誌記錄
│
├── evaluation/                    # 評估工具
│   └── arena.py                  # AI 對戰競技場
│
├── utils/                         # 工具腳本
│   ├── test_mcts_pytorch.py      # MCTS 測試
│   ├── verify_pytorch.py         # PyTorch 環境驗證
│   └── diagnose_gpu.py           # GPU 診斷
│
├── docs/                          # 文檔
│   ├── PERFORMANCE_EVALUATION.md # 性能評估
│   ├── PERFORMANCE_OPTIMIZATION.md # 性能優化
│   └── install_pytorch.md        # PyTorch 安裝指南
│
├── legacy/                        # 舊代碼（保留參考）
│   └── train_pipeline_tf.py      # TensorFlow 訓練流程
│
├── checkpoints/                   # 模型檢查點
│   ├── latest_model.pth          # 最新模型
│   ├── training_history.json     # 訓練歷史
│   └── summaries/                # 迭代摘要
│
├── logs/                          # 訓練日誌
│   └── games/                    # 遊戲詳細記錄
│
├── analyze_games.py              # 遊戲分析工具
├── benchmark_performance.py      # 性能基準測試
├── monitor_games.py              # 監控遊戲進度
├── monitor_training.py           # 監控訓練進度
├── reset_training.py             # 重置訓練數據
│
├── README.md                     # 專案說明
├── TODO.md                       # 待辦事項
├── requirements.txt              # Python 依賴
└── pyproject.toml               # 專案配置
```

## 快速開始

### 訓練 AI

```bash
# 快速測試（5次迭代）
python train_pipeline_pytorch.py --fast-test

# 完整訓練（1000次迭代）
python train_pipeline_pytorch.py

# 自定義訓練
python train_pipeline_pytorch.py --iterations 100 --games 50
```

### 觀看 AI 對弈

```bash
python watch_ai.py checkpoints/latest_model.pth
```

### 與 AI 對弈

```bash
python main.py
```

## 技術棧

- **深度學習**: PyTorch + CUDA (混合精度訓練)
- **模型架構**: SE-ResNet (10 層殘差塊 + 注意力機制)
- **訓練方法**: AlphaZero 風格自我對弈 + 優先級經驗回放
- **並行處理**: 多進程自我對弈（8 個進程）
- **性能優化**: 批量 MCTS、混合精度、GPU 加速

## 文件統計

- 總文件數：16 個主要文件
- 核心模組：6 個
- 訓練工具：3 個
- 文檔：3 個
- 代碼行數：~3000+ 行
