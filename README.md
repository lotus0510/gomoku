# 五子棋 AI 訓練系統 (AlphaZero Architecture)

基於 AlphaZero 架構的現代化五子棋 AI 訓練系統，使用 **PyTorch** 框架實現高效的深度強化學習。

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 📋 目錄

- [核心特性](#核心特性)
- [系統需求](#系統需求)
- [安裝指南](#安裝指南)
- [快速開始](#快速開始)
- [專案結構](#專案結構)
- [訓練配置詳解](#訓練配置詳解)
- [使用指南](#使用指南)
- [監控與診斷](#監控與診斷)
- [性能優化](#性能優化)
- [常見問題](#常見問題)
- [完整文檔](#完整文檔)
- [更新日誌](#更新日誌)

---

## 📚 完整文檔

本 README 提供快速開始指南。完整文檔請參考：

### 快速導覽
- **[📖 文檔索引](docs/INDEX.md)** - 完整文檔導覽和分類
- **[🔧 環境設置](docs/guides/SETUP_GUIDE.md)** - 詳細安裝和配置指南
- **[🎓 訓練概念](docs/guides/TRAINING_CONCEPTS.md)** - 深入理解訓練機制和獎勵設計
- **[👨‍💻 開發指南](docs/guides/DEVELOPER_GUIDE.md)** - 代碼結構和開發文檔
- **[📊 監控指標](docs/guides/MONITORING_METRICS.md)** - 訓練監控詳解

### 問題排查
- **[🐛 BUG 修復報告](docs/troubleshooting/CRITICAL_BUG_FIX.md)** - MCTS 根節點問題（已修復）
- **[❓ 常見問題](docs/troubleshooting/)** - 問題診斷和解決方案

### 分析工具
- **[🔍 診斷工具](analysis/)** - 訓練分析和診斷腳本
  ```bash
  python analysis/diagnose_training.py  # 完整診斷
  python analysis/plot_comprehensive.py # 生成分析圖表
  ```

---

## 🎯 核心特性

### AlphaZero 架構
- **自我對弈訓練**：無需人類棋譜，完全從零開始學習
- **MCTS + 深度學習**：蒙特卡羅樹搜索結合神經網絡策略與價值評估
- **批量化 MCTS**：並行推理，8x 性能提升
- **優先級經驗回放**：智能採樣重要訓練數據

### 神經網絡架構
- **SE-ResNet**：10 層殘差塊 + Squeeze-and-Excitation 注意力機制
- **128 濾波器**：深度特徵提取
- **雙頭輸出**：
  - 策略頭（Policy）：15×15 = 225 維動作概率分佈
  - 價值頭（Value）：局面評估 [-1, 1]
- **混合精度訓練 (AMP)**：顯存優化，訓練加速

### 訓練系統
- **多進程自我對弈**：8 個並行進程
- **動態學習率調度**：StepLR 自動衰減
- **梯度裁剪**：防止梯度爆炸
- **檢查點管理**：自動保存與恢復
- **實時監控**：6 大核心指標追蹤

### 用戶體驗
- **人機對戰**：與訓練後的 AI 對弈
- **觀戰模式**：觀看 AI 自我對弈
- **可視化分析**：豐富的圖表與統計數據
- **遊戲日誌系統**：完整記錄每局棋譜與策略

---

## 💻 系統需求

### 最低需求
- **CPU**: 4 核心 (推薦 8 核心以上)
- **RAM**: 8GB (推薦 16GB)
- **GPU**: NVIDIA GPU with CUDA (推薦 6GB+ VRAM)
- **存儲**: 10GB 可用空間
- **作業系統**: Windows 10/11, Linux, macOS

### 推薦配置
- **CPU**: Intel i7 / AMD Ryzen 7 或更高
- **RAM**: 16GB+
- **GPU**: NVIDIA RTX 3060 或更高 (8GB+ VRAM)
- **存儲**: SSD 20GB+

### 軟體需求
- Python 3.8 - 3.11
- CUDA 11.8+ (GPU 訓練)
- cuDNN 8.0+

---

## 📦 安裝指南

### 1. 克隆專案

```bash
git clone <repository-url>
cd gomoku
```

### 2. 創建虛擬環境

```bash
# Windows
python -m venv .venv
.venv\Scripts\activate

# Linux/macOS
python3 -m venv .venv
source .venv/bin/activate
```

### 3. 安裝依賴

```bash
# 安裝 PyTorch (CUDA 版本，根據你的 CUDA 版本調整)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# 安裝其他依賴
pip install -r requirements.txt
```

### 4. 驗證安裝

```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"
```

預期輸出：
```
PyTorch: 2.x.x
CUDA available: True
```

---

## 🚀 快速開始

### 1. 快速測試（5-10 分鐘）

驗證環境配置是否正確：

```bash
python train_pipeline.py -f -i 20 -g 20
```

- `-f`: 快速測試模式（小模型：3 ResBlocks, 64 濾波器, 50 MCTS 模擬）
- `-i 20`: 運行 20 次迭代
- `-g 20`: 每次迭代 20 局遊戲

### 2. 完整訓練（推薦）

```bash
# 可選：清理舊訓練數據
python reset_training.py

# 開始完整訓練
python train_pipeline.py -i 1000 -g 100
```

預期時間：
- **無 GPU**: 約 100-200 小時
- **有 GPU (RTX 3060)**: 約 20-40 小時
- **高端 GPU (RTX 4090)**: 約 10-15 小時

### 3. 從檢查點恢復訓練

```bash
python train_pipeline.py --resume checkpoints/checkpoint_iter_50.pth
```

### 4. 人機對戰

```bash
python play_vs_ai.py
```

操作說明：
- 左鍵點擊下棋
- 默認人類執黑（先手）

### 5. 觀看 AI 自我對弈

```bash
python watch_ai.py
```

控制：
- **空白鍵**: 暫停/繼續
- **R**: 重新開始

---

## 📁 專案結構

```
gomoku/
├── core/                           # 核心模組
│   ├── game.py                    # 五子棋遊戲邏輯
│   ├── game_state.py              # 遊戲狀態管理
│   ├── neural_net.py              # PyTorch 神經網絡（SE-ResNet）
│   ├── mcts.py                    # 標準 MCTS 實現
│   └── mcts_batched.py            # 批量化 MCTS（8x 加速）
│
├── training/                       # 訓練模組
│   ├── config.py                  # 🔧 訓練超參數配置
│   ├── replay_buffer.py           # 優先級經驗回放緩衝區
│   └── game_logger.py             # 遊戲詳細日誌系統
│
├── analysis/                       # 分析工具
│   ├── diagnose_training.py       # 訓練健康診斷
│   ├── plot_comprehensive.py      # 綜合可視化分析
│   ├── value_gamma_analysis.py    # 價值折扣分析
│   └── README.md                  # 分析工具說明
│
├── docs/                           # 文檔
│   ├── MONITORING_METRICS.md      # 監控指標說明
│   └── TRAINING_IMPROVEMENT_PLAN.md  # 訓練改進計劃
│
├── checkpoints/                    # 模型檢查點
│   ├── checkpoint_iter_*.pth      # 迭代檢查點
│   ├── latest_model.pth           # 最新模型
│   ├── training_history.json      # 訓練歷史數據
│   └── games/                     # 遊戲詳細記錄
│
├── logs/                           # 日誌輸出
│   ├── games/                     # 遊戲統計圖表
│   └── tensorboard/               # TensorBoard 日誌
│
├── train_pipeline.py               # 🎯 主訓練腳本
├── play_vs_ai.py                  # 人機對戰
├── watch_ai.py                    # AI 觀戰
├── monitor_training.py            # 實時監控
├── analyze_games.py               # 訓練後分析
├── benchmark_performance.py       # 性能基準測試
├── reset_training.py              # 重置訓練環境
└── README.md                      # 本文件
```

---

## ⚙️ 訓練配置詳解

所有訓練超參數位於 `training/config.py`，以下為詳細說明：

### 🏗️ 模型架構

```python
# training/config.py

BOARD_SIZE = 15              # 棋盤大小（標準五子棋）
NUM_RES_BLOCKS = 10          # 殘差塊數量（深度）
NUM_FILTERS = 128            # 卷積濾波器數（寬度）
SE_RATIO = 8                 # Squeeze-Excitation 壓縮比
L2_REG = 1e-4               # L2 正則化係數
DROPOUT_RATE = 0.3           # Dropout 率（防止過擬合）
VALUE_HEAD_HIDDEN = 512      # 價值頭隱藏層大小
```

**調整建議**：
- **更深模型**：增加 `NUM_RES_BLOCKS` (10 → 15)，但需要更多訓練時間
- **更寬模型**：增加 `NUM_FILTERS` (128 → 256)，顯存需求翻倍
- **過擬合**：提高 `DROPOUT_RATE` (0.3 → 0.4) 或 `L2_REG` (1e-4 → 5e-4)

---

### 🔍 MCTS 設置

```python
MCTS_SIMULATIONS = 200       # 每步 MCTS 模擬次數
MCTS_BATCH_SIZE = 8          # 批量推理大小
C_PUCT = 2.0                 # PUCT 探索常數（UCB 權重）
DIRICHLET_ALPHA = 0.3        # Dirichlet 噪聲 alpha
DIRICHLET_EPSILON = 0.35     # Dirichlet 噪聲混合比例
USE_BATCHED_MCTS = True      # 是否使用批量 MCTS
```

**參數說明**：

| 參數 | 作用 | 效果 |
|------|------|------|
| `MCTS_SIMULATIONS` | 搜索深度 | 越高越強，但越慢（200 是平衡值） |
| `C_PUCT` | 探索 vs 利用 | 越高越探索未知路徑（2.0 推薦） |
| `DIRICHLET_EPSILON` | 開局隨機性 | 越高越多樣化（0.35 防止過早收斂） |

**調整建議**：
- **訓練階段**：`MCTS_SIMULATIONS = 200`，`DIRICHLET_EPSILON = 0.35`
- **評估階段**：`MCTS_SIMULATIONS = 400`，`DIRICHLET_EPSILON = 0.0`（無噪聲）
- **快速測試**：`MCTS_SIMULATIONS = 50`

---

### 🌡️ 溫度控制

```python
TEMP_THRESHOLD_MOVE = 30     # 前 30 步使用溫度 1.0（探索）
TEMP_FINAL_MOVE = 60         # 60 步後使用極低溫度（確定性）
```

**溫度概念**：
- **溫度 = 1.0**：按 MCTS 訪問次數比例隨機選擇（高探索）
- **溫度 → 0**：選擇訪問次數最多的動作（確定性）

**時間線**：
```
步數 1-30:  溫度 1.0  → 充分探索開局變化
步數 31-60: 溫度線性下降 → 逐漸收斂
步數 60+:   溫度 0.01 → 確定性收官
```

---

### 📚 訓練超參數

```python
ITERATIONS = 1000            # 總迭代次數
GAMES_PER_ITERATION = 100    # 每次迭代自我對弈局數
BATCH_SIZE = 512             # 訓練批次大小
EPOCHS_PER_ITERATION = 5     # 每次迭代訓練 epoch 數

# 學習率調度
LEARNING_RATE = 0.0001       # 初始學習率
LR_DECAY_STEPS = 100         # 每 100 次迭代衰減一次
LR_DECAY_RATE = 0.8          # 學習率衰減率（lr *= 0.8）
MIN_LEARNING_RATE = 1e-5     # 最低學習率下限

GRADIENT_CLIP_NORM = 1.0     # 梯度裁剪範數
```

**學習率時間線**：
```
迭代 0-99:    lr = 0.0001
迭代 100-199: lr = 0.00008  (0.0001 × 0.8)
迭代 200-299: lr = 0.000064 (0.00008 × 0.8)
迭代 300+:    lr = 0.000051 → 0.00001 (最低)
```

**調整建議**：
- **訓練不穩定**：降低 `LEARNING_RATE` (1e-4 → 5e-5)
- **訓練過慢**：增加 `BATCH_SIZE` (512 → 1024) 或 `EPOCHS_PER_ITERATION` (5 → 10)
- **早期過擬合**：減少 `LR_DECAY_STEPS` (100 → 50)

---

### 💾 經驗回放

```python
REPLAY_BUFFER_SIZE = 400000  # 回放緩衝區大小（默認配置，約 10 次迭代）
                             # 優化配置: 800000 (約 40 次迭代)
PRIORITIZED_ALPHA = 0.6      # 優先級指數（0 = 均勻採樣）
PRIORITIZED_BETA = 0.4       # 重要性採樣補償
PRIORITIZED_BETA_INCREMENT = 0.001  # Beta 增長率
```

**優先級回放原理**：
- 損失越大的樣本，被採樣概率越高
- `PRIORITIZED_ALPHA` 控制優先級強度（0.6 是平衡值）
- `PRIORITIZED_BETA` 補償採樣偏差（從 0.4 逐漸增長到 1.0）

---

### 🎯 價值折扣（重要！）

```python
VALUE_GAMMA = 0.995          # 價值折扣因子
```

**作用**：控制快速獲勝的激勵強度

**計算公式**：
```
位置價值 = 基礎價值 × (VALUE_GAMMA ^ 距離終點步數)
```

**具體效果**（50 步遊戲）：

| VALUE_GAMMA | 第 1 步價值 | 第 50 步價值 | 差距 | 效果 |
|-------------|------------|-------------|------|------|
| 1.000 | 1.000 | 1.000 | 0% | ❌ 無效率概念，可能拖延 |
| 0.998 | 0.906 | 1.000 | 7.7% | ✅ 輕微偏好快速（保守） |
| **0.995** | **0.778** | **1.000** | **18.2%** | ✅ **明確偏好快速（推薦）** |
| 0.99 | 0.605 | 1.000 | 33.1% | ❌ 過度追求速度（Edge-Rush 風險） |

**當前配置 (0.995) 的含義**：
- ✅ 鼓勵快速決勝（18.2% 獎勵）
- ✅ 不會犧牲策略質量
- ⚠️ 需監控是否出現過度激進策略

**調整建議**：
- **出現 Edge-Rush**：降低到 0.997 或 0.998
- **遊戲過長（>60 步）**：降低到 0.99
- **需要深度策略**：提高到 0.998 或 1.0

---

### 📊 評估與日誌

```python
EVAL_FREQUENCY = 5           # 每 5 次迭代評估一次
EVAL_GAMES = 50              # 評估遊戲局數
PROMOTION_THRESHOLD = 0.55   # 模型晉級閾值（勝率 >55%）

CHECKPOINT_FREQUENCY = 1     # 每次迭代保存檢查點
MAX_CHECKPOINTS = 1000       # 保留所有檢查點
SAVE_BEST_MODEL = True       # 自動保存最佳模型

LOG_FREQUENCY = 1            # 每次迭代記錄日誌
ENABLE_GAME_LOGGING = True   # 記錄遊戲詳細數據
DETAILED_GAME_LOG_FREQUENCY = 10  # 每 10 局保存詳細棋譜
```

---

### 🔧 損失權重

```python
POLICY_LOSS_WEIGHT = 1.0     # 策略損失權重
VALUE_LOSS_WEIGHT = 1.0      # 價值損失權重
```

**總損失**：
```
Loss = POLICY_LOSS_WEIGHT × CrossEntropy(policy_pred, policy_target)
     + VALUE_LOSS_WEIGHT × MSE(value_pred, value_target)
```

**調整建議**：
- **價值網路不收斂**：提高 `VALUE_LOSS_WEIGHT` (1.0 → 1.5)
- **策略過於隨機**：提高 `POLICY_LOSS_WEIGHT` (1.0 → 1.5)

---

## 📖 使用指南

### 訓練流程

#### 1. 配置檢查

```bash
# 驗證配置
python -c "from training.config import TrainingConfig; print(TrainingConfig())"
```

#### 2. 開始訓練

```bash
# 完整訓練（1000 次迭代）
python train_pipeline.py -i 1000 -g 100

# 自定義迭代數和遊戲數
python train_pipeline.py -i 500 -g 50

# 從檢查點恢復
python train_pipeline.py --resume checkpoints/checkpoint_iter_100.pth
```

#### 3. 監控訓練（另開終端）

```bash
# 實時監控
python monitor_training.py

# 每 5 次迭代診斷
python analysis/diagnose_training.py
```

#### 4. 訓練後分析

```bash
# 生成綜合分析圖表
python analysis/plot_comprehensive.py

# 分析價值折扣效果
python analysis/value_gamma_analysis.py
```

---

### 對弈與評估

#### 人機對戰

```bash
python play_vs_ai.py
```

**選項**：
- 默認執黑（先手）
- 可在代碼中修改 `human_color = 2` 切換為執白

#### 觀看 AI 對弈

```bash
python watch_ai.py

# 指定模型
python watch_ai.py --model checkpoints/checkpoint_iter_500.pth

# 指定 MCTS 模擬次數
python watch_ai.py --simulations 400
```

#### 性能基準測試

```bash
python benchmark_performance.py
```

輸出：
- MCTS 模擬速度（sims/sec）
- 神經網絡推理速度（inferences/sec）
- 批量 vs 標準 MCTS 對比

---

### 數據管理

#### 重置訓練環境

```bash
python reset_training.py
```

**警告**：這將刪除所有檢查點和訓練歷史！

#### 備份重要檢查點

```bash
# 手動備份
mkdir backup
cp checkpoints/checkpoint_iter_500.pth backup/
cp checkpoints/training_history.json backup/

# 壓縮備份
tar -czf backup_iter500.tar.gz checkpoints/checkpoint_iter_500.pth checkpoints/training_history.json
```

---

## 📈 監控與診斷

### 六大核心監控指標

#### 1. 勝率平衡
- **健康範圍**：黑棋 45-58%
- **問題**：<30% 或 >70% 表示訓練失衡
- **查看**：`training_history.json` → `black_win_rate`

#### 2. 遊戲長度
- **健康範圍**：25-45 步
- **問題**：<20 步（可能 Edge-Rush）, >60 步（拖延）
- **查看**：`training_history.json` → `avg_game_length`

#### 3. 策略質量（Top-1 機率）
- **健康範圍**：0.15-0.45
- **問題**：<0.1（過於隨機）, >0.8（過度自信）
- **查看**：`checkpoints/games/games_iter_*.json` → `policy_top1`

#### 4. 價值網路標準差
- **健康範圍**：>0.3
- **問題**：<0.2（判別力下降）, 0.0（完全失效）
- **查看**：`training_history.json` → `value_std`

#### 5. 梯度範數
- **健康範圍**：0.5-2.0
- **問題**：<0.1（梯度消失）, >5.0（梯度爆炸）
- **查看**：`training_history.json` → `grad_norm`

#### 6. 勝率振盪
- **健康範圍**：std <0.1, 翻轉 <2 次/10 迭代
- **問題**：std >0.15 或頻繁翻轉（訓練不穩定）
- **查看**：分析工具自動計算

---

### 自動診斷工具

```bash
# 運行診斷
python analysis/diagnose_training.py
```

**輸出範例**：
```
🟢 [OK] 勝率平衡健康
   迭代 95-100: 黑棋 52.3% ± 3.1%

⚠️  [WARNING] 遊戲長度偏短
   當前平均: 22.5 步 (建議: >25 步)

🔴 [CRITICAL] 價值網路判別力下降
   value_std = 0.18 (建議: >0.3)
   建議: 檢查是否勝率失衡或數據分佈問題
```

---

### 可視化分析

```bash
# 生成綜合圖表（6 個子圖）
python analysis/plot_comprehensive.py
```

**生成圖表**：
1. 勝率變化趨勢（黑棋 vs 白棋）
2. 價值網路健康度（std 與健康範圍）
3. 策略質量（Top-1, 熵）
4. 遊戲統計（長度、超短局比例）
5. 訓練指標（損失、梯度）
6. 勝率振盪分析（滾動 std, 翻轉次數）

保存位置：`analysis/comprehensive_analysis.png`

---

### 監控最佳實踐

#### 訓練期間（每 5-10 次迭代）

```bash
# 1. 快速檢查訓練歷史
tail -n 50 checkpoints/training_history.json

# 2. 運行診斷
python analysis/diagnose_training.py

# 3. 查看最新遊戲統計
ls -lt checkpoints/games/ | head
```

#### 停止條件

**立即停止** 🔴：
- 勝率連續 3 次 <25% 或 >75%
- 價值 std <0.1 持續 5 次迭代
- 梯度範數 <0.05 或 >10.0
- Top-1 >0.95（策略完全固化）

**需要調整** ⚠️：
- 遊戲長度持續下降趨勢（可能 Edge-Rush）
- 損失不再下降（學習停滯）
- 超短局 (≤15 步) >50% 持續 10 次迭代

---

## 🔧 性能優化

### GPU 優化

#### 1. 混合精度訓練（已啟用）

```python
# train_pipeline.py
scaler = torch.cuda.amp.GradScaler()
with torch.cuda.amp.autocast():
    policy_logits, value = model(states)
```

**效果**：
- 顯存減少 ~40%
- 訓練速度提升 ~30%

#### 2. 批量大小調整

根據 GPU 顯存調整 `BATCH_SIZE`：

| GPU VRAM | 推薦 BATCH_SIZE | 備註 |
|----------|-----------------|------|
| 6GB | 256-512 | 保守配置 |
| 8GB | 768-1024 | **激進配置** (你的硬件) |
| 12GB | 1024-2048 | 高性能 |
| 16GB+ | 2048+ | 專業級 |

```python
# training/config.py
BATCH_SIZE = 512   # 默認配置
BATCH_SIZE = 1024  # 優化配置 (8GB VRAM 激進設定)
```

#### 3. DataLoader 優化

```python
# train_pipeline.py 已優化
DataLoader(
    dataset,
    batch_size=config.BATCH_SIZE,
    shuffle=True,
    num_workers=4,      # 多進程加載
    pin_memory=True,    # 固定內存（GPU 加速）
    persistent_workers=True  # 保持 worker 進程
)
```

---

### CPU 優化

#### 1. 多進程自我對弈

```python
# training/config.py
NUM_WORKERS = 8   # 默認配置
NUM_WORKERS = 16  # 優化配置 (14核20緒 激進設定)
```

推薦設定：
- 4 核 CPU: `NUM_WORKERS = 4`
- 8 核 CPU: `NUM_WORKERS = 8`
- 12 核 CPU: `NUM_WORKERS = 10-12`
- **14 核 20 緒**: `NUM_WORKERS = 16` **← 你的硬件 (激進配置)**
- 16 核以上: `NUM_WORKERS = 14-18`（保留部分核心給系統）

#### 2. 批量 MCTS

```python
USE_BATCHED_MCTS = True  # 必須啟用！
MCTS_BATCH_SIZE = 8       # 批量推理大小
```

**效果**：比標準 MCTS 快 8 倍

---

### 磁碟 I/O 優化

#### 1. 減少詳細日誌頻率

```python
# training/config.py
DETAILED_GAME_LOG_FREQUENCY = 20  # 從 10 → 20（減少寫入）
```

#### 2. 使用 SSD

將 `checkpoints/` 目錄放在 SSD 上：

```bash
# Linux/macOS 符號連結範例
mv checkpoints /path/to/ssd/gomoku_checkpoints
ln -s /path/to/ssd/gomoku_checkpoints checkpoints
```

---

## ❓ 常見問題

### 訓練相關

#### Q: 訓練多久能擊敗人類新手？

**A**: 約 100-200 次迭代（10-20 小時，有 GPU）。初期 AI 很弱，但學習速度很快。

---

#### Q: 為什麼勝率不穩定，一下黑贏一下白贏？

**A**: 這是自我對弈的正常現象：
- 迭代 0-50：勝率振盪劇烈（模型探索中）
- 迭代 50-200：逐漸穩定
- 迭代 200+：黑棋應穩定在 52-58%（先手優勢）

如果持續振盪：
1. 檢查 `VALUE_GAMMA` 是否過低（<0.995）
2. 增加 `DIRICHLET_EPSILON` (0.35 → 0.4)
3. 降低學習率 `LEARNING_RATE` (1e-4 → 5e-5)

---

#### Q: 什麼是 Edge-Rush？如何避免？

**A**: Edge-Rush 是指 AI 學到沿棋盤邊緣快速連五的策略，忽略策略質量。

**檢測**：
- 平均遊戲長度 <15 步
- 超短局比例 >60%
- 大量棋子在第 0 行或第 14 行

**解決方法**：
1. 提高 `VALUE_GAMMA` (0.99 → 0.995 或 0.998)
2. 增加探索性：`DIRICHLET_EPSILON` (0.25 → 0.35)
3. 回滾到健康檢查點重新訓練

---

#### Q: 價值網路輸出全是 0.0，怎麼辦？

**A**: 價值網路崩潰，通常由勝率極度失衡引起（如白棋 90% 勝率）。

**恢復步驟**：
1. 停止訓練
2. 回滾到健康迭代（勝率 45-55%）：
   ```bash
   python train_pipeline.py --resume checkpoints/checkpoint_iter_XX.pth
   ```
3. 調整配置：
   - 提高 `VALUE_LOSS_WEIGHT` (1.0 → 1.5)
   - 增加探索性（見上文）

---

#### Q: 訓練後期損失不再下降？

**A**: 可能原因：
1. **學習率過低**：檢查當前迭代，學習率可能已衰減到最低
2. **過擬合**：模型已學習到當前數據的極限
3. **局部最優**：陷入次優策略

**解決方法**：
- 提高學習率：修改 `MIN_LEARNING_RATE` (1e-5 → 5e-5)
- 增加數據多樣性：提高 `DIRICHLET_EPSILON`
- 增加模型容量：`NUM_RES_BLOCKS` (10 → 12)

---

### 技術問題

#### Q: CUDA out of memory 錯誤？

**A**: 顯存不足，嘗試：
1. 降低 `BATCH_SIZE` (512 → 256)
2. 減少 `NUM_RES_BLOCKS` (10 → 8) 或 `NUM_FILTERS` (128 → 96)
3. 確保啟用混合精度訓練（已默認啟用）
4. 關閉其他 GPU 程序

---

#### Q: 多進程訓練報錯？

**A**: Windows 上多進程需要特殊處理：

```python
# train_pipeline.py 頂部已處理
if __name__ == '__main__':
    multiprocessing.set_start_method('spawn', force=True)
```

如果仍有問題：
1. 降低 `NUM_WORKERS` (8 → 4)
2. 檢查是否有殺毒軟件干擾

---

#### Q: 無法加載檢查點？

**A**: 可能原因：
1. **模型架構不匹配**：檢查點是舊版本模型
2. **文件損壞**：訓練中斷時保存失敗

**解決**：
```python
# 檢查檢查點內容
checkpoint = torch.load('checkpoints/checkpoint_iter_50.pth', map_location='cpu')
print(checkpoint.keys())  # 應有: model_state_dict, optimizer_state_dict, iteration

# 手動加載模型權重
model.load_state_dict(checkpoint['model_state_dict'])
```

---

### 配置問題

#### Q: 如何切換快速測試模式？

**A**: 使用 `-f` 標誌或手動配置：

```python
# 方法 1: 命令行
python train_pipeline.py -f -i 10 -g 10

# 方法 2: 代碼修改
config = TrainingConfig.get_fast_test_config()
```

---

#### Q: 如何只訓練策略網路（不訓練價值網路）？

**A**: 調整損失權重：

```python
# training/config.py
POLICY_LOSS_WEIGHT = 1.0
VALUE_LOSS_WEIGHT = 0.0  # 設為 0
```

**警告**：這會嚴重影響 MCTS 性能！

---

## 📝 更新日誌

### v2.1.0 (2024-12-22) - 配置優化版

#### 🎯 核心改進
- **VALUE_GAMMA 優化**: 1.0 → 0.995（18.2% 效率激勵）
  - 明確鼓勵快速獲勝，避免拖延戰術
  - 平衡速度與策略質量

- **探索性增強**: 防止過早收斂到次優策略
  - `DIRICHLET_EPSILON`: 0.25 → 0.35 (+40%)
  - `C_PUCT`: 1.5 → 2.0 (+33%)
  - `TEMP_THRESHOLD_MOVE`: 25 → 30 (+20%)
  - `TEMP_FINAL_MOVE`: 50 → 60 (+20%)

- **學習率調度優化**: 更頻繁衰減，提升後期穩定性
  - `LR_DECAY_STEPS`: 400 → 100（4x 頻繁）
  - `LR_DECAY_RATE`: 0.7 → 0.8（更溫和）

#### 📊 新增工具
- `analysis/diagnose_training.py` - 自動訓練健康診斷
- `analysis/plot_comprehensive.py` - 6 大指標綜合可視化
- `analysis/value_gamma_analysis.py` - 價值折扣效果分析
- `docs/MONITORING_METRICS.md` - 監控指標完整文檔
- `FINAL_RECOMMENDATION.md` - 配置建議總結

#### 🐛 問題修復
- 修復價值網路崩潰問題（0.0 輸出）
- 修復 Edge-Rush 策略（VALUE_GAMMA 過低導致）
- 修復勝率振盪過大問題

---

### v2.0.0 (2024-11) - PyTorch 重構版

#### 🔄 框架遷移
- ✅ 完全遷移至 PyTorch
- ✅ 混合精度訓練 (AMP)
- ✅ 批量化 MCTS (8x 加速)
- ✅ 優先級經驗回放

#### 🎮 新功能
- 人機對戰模式 (`play_vs_ai.py`)
- AI 觀戰模式 (`watch_ai.py`)
- 實時訓練監控 (`monitor_training.py`)

#### 🏗️ 架構優化
- SE-ResNet 模型（10 ResBlocks, 128 Filters）
- 遊戲詳細日誌系統
- 自動檢查點管理

---

### v1.0.0 (2024-10) - 初始版本

- 基於 TensorFlow 的 AlphaZero 實現
- 標準 MCTS + 深度學習
- 基本訓練與評估功能

---

## 📚 參考資料

### 論文
- [Mastering the Game of Go without Human Knowledge (AlphaGo Zero)](https://www.nature.com/articles/nature24270)
- [Mastering Chess and Shogi by Self-Play with a General Reinforcement Learning Algorithm (AlphaZero)](https://arxiv.org/abs/1712.01815)
- [Squeeze-and-Excitation Networks](https://arxiv.org/abs/1709.01507)

### 相關專案
- [AlphaZero_Gomoku](https://github.com/junxiaosong/AlphaZero_Gomoku)
- [KataGo](https://github.com/lightvector/KataGo)

### 教學資源
- [強化學習入門](https://spinningup.openai.com/)
- [深度學習花書](https://www.deeplearningbook.org/)

---

## 📄 授權

MIT License - 詳見 `LICENSE` 文件

---

## 🤝 貢獻

歡迎提交 Issue 和 Pull Request！

### 開發指南
1. Fork 本專案
2. 創建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 開啟 Pull Request

---

## 📧 聯繫方式

如有問題或建議，請開啟 GitHub Issue。

---

**祝您訓練愉快！打造最強五子棋 AI！** 🚀🎯
