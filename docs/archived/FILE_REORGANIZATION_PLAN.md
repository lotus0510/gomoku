# 檔案整理計劃

## 📊 當前檔案清單（按類別）

### 🟢 核心系統檔案（保留）

**訓練系統**
- `train_pipeline_pytorch.py` - 主訓練腳本
- `training/config.py` - 配置文件
- `training/game_logger.py` - 遊戲日誌
- `training/replay_buffer.py` - 經驗回放緩衝區
- `training/__init__.py`

**遊戲核心**
- `game.py` - 五子棋遊戲邏輯
- `core/game_state.py` - 遊戲狀態管理
- `core/mcts.py` - MCTS 實現
- `core/mcts_batched.py` - 批次化 MCTS
- `core/neural_net.py` - 神經網路模型
- `core/data_augmentation.py` - 數據增強
- `core/__init__.py`

**評估系統**
- `evaluation/arena.py` - 對戰競技場
- `evaluation/elo_rating.py` - ELO 評分系統
- `evaluation/__init__.py`

**工具系統**
- `utils/diagnose_gpu.py` - GPU 診斷
- `utils/verify_pytorch.py` - PyTorch 驗證
- `utils/test_mcts_pytorch.py` - MCTS 測試
- `utils/__init__.py`

**配置**
- `requirements.txt` - Python 依賴
- `pyproject.toml` - 專案配置（如果存在）

---

### 🟡 分析與監控腳本（需要整理）

**當前存在的檔案：**
1. `analyze_progress.py` - 訓練進展分析 ⭐
2. `analyze_asymmetry.py` - 黑白不對稱分析 ⭐
3. `detailed_analysis.py` - 詳細分析
4. `analyze_games.py` - 遊戲分析
5. `analyze_csv_metrics.py` - CSV 指標分析
6. `reward_analysis_simple.py` - 獎勵機制分析 ⭐
7. `monitor_training.py` - 訓練監控 ⭐
8. `monitor_games.py` - 遊戲監控
9. `check_history.py` - 歷史檢查
10. `visualize_game.py` - 遊戲可視化

**整理建議：合併為 3 個腳本**

#### 方案 A：按功能合併

```
analysis/
├── training_monitor.py      # 合併：monitor_training.py + analyze_progress.py
├── game_analyzer.py          # 合併：analyze_games.py + analyze_csv_metrics.py + visualize_game.py
└── pattern_analyzer.py       # 合併：analyze_asymmetry.py + reward_analysis_simple.py
```

#### 方案 B：保留核心，刪除重複

**保留（3個核心腳本）：**
- `analyze_progress.py` - 最常用的訓練進展分析
- `monitor_training.py` - 實時監控
- `visualize_game.py` - 遊戲可視化

**刪除（功能重複）：**
- `analyze_asymmetry.py` - 功能可以整合到 analyze_progress.py
- `detailed_analysis.py` - 與 analyze_progress.py 重複
- `analyze_games.py` - 功能重複
- `analyze_csv_metrics.py` - 功能重複
- `reward_analysis_simple.py` - 一次性分析，已完成任務
- `monitor_games.py` - 與 monitor_training.py 重複
- `check_history.py` - 功能簡單，可內聯

---

### 🟠 臨時/測試檔案（刪除）

- `temp_analyze_csv.py` ❌
- `temp_summary.py` ❌
- `temp_summary_v2.py` ❌
- `update_checkpoint_config.py` ❌ （一次性工具，已完成）
- `quick_test.py` ❌ （功能已整合到主訓練腳本 -f 參數）

---

### 🔵 互動與遊戲檔案（保留）

- `play_vs_ai.py` - 人機對戰
- `watch_ai.py` - 觀看 AI 對戰
- `main.py` - 主入口（如果有用）

---

### 📚 文檔檔案（整理）

**當前文檔：**
- `README.md` - 專案說明
- `PROJECT_STRUCTURE.md` - 專案結構
- `TRACKING_METRICS.md` - 指標追蹤
- `REWARD_MECHANISM.md` - 獎勵機制說明 ⭐
- `REWARD_IMPORTANCE.md` - 獎勵重要性 ⭐
- `PERFORMANCE_EVALUATION.md` - 性能評估（根目錄）
- `PERFORMANCE_OPTIMIZATION.md` - 性能優化（根目錄）
- `docs/DEVELOPER_GUIDE.md`
- `docs/install_pytorch.md`
- `docs/LOG.md`
- `docs/PERFORMANCE_EVALUATION.md`
- `docs/PERFORMANCE_OPTIMIZATION.md`

**整理建議：**

```
docs/
├── README.md                          # 專案總覽
├── SETUP.md                           # 安裝與設置（合併 install_pytorch.md）
├── ARCHITECTURE.md                    # 系統架構（合併 PROJECT_STRUCTURE.md）
├── TRAINING_GUIDE.md                  # 訓練指南
├── REWARD_MECHANISM.md                # 獎勵機制（移入 docs/）
├── PERFORMANCE.md                     # 性能文檔（合併兩個 PERFORMANCE 文件）
└── CHANGELOG.md                       # 變更日誌（取代 LOG.md）
```

**刪除重複：**
- 根目錄的 `PERFORMANCE_EVALUATION.md` 和 `PERFORMANCE_OPTIMIZATION.md` ❌
- `TRACKING_METRICS.md` ❌ （可以整合到 TRAINING_GUIDE.md）

---

### 🔴 遺留檔案（刪除或歸檔）

- `legacy/train_pipeline_tf.py` ❌ （TensorFlow 舊版本）
- `start_training.sh` ⚠️ （檢查是否還在使用）
- `analysis_results.txt` ❌ （臨時結果文件）

---

### 🛠️ 管理腳本（保留）

- `reset_training.py` - 訓練重置工具 ✅

---

## 📋 建議的最終結構

```
gomoku/
├── train_pipeline_pytorch.py          # 主訓練腳本
├── reset_training.py                  # 重置工具
├── requirements.txt
├── pyproject.toml
│
├── core/                               # 核心系統
│   ├── game_state.py
│   ├── mcts.py
│   ├── mcts_batched.py
│   ├── neural_net.py
│   └── data_augmentation.py
│
├── training/                           # 訓練系統
│   ├── config.py
│   ├── game_logger.py
│   └── replay_buffer.py
│
├── evaluation/                         # 評估系統
│   ├── arena.py
│   └── elo_rating.py
│
├── analysis/                           # 分析工具 ⭐ 新建目錄
│   ├── training_monitor.py           # 實時監控
│   ├── analyze_progress.py           # 進展分析
│   └── visualize_game.py              # 遊戲可視化
│
├── interactive/                        # 互動遊戲 ⭐ 新建目錄
│   ├── play_vs_ai.py
│   └── watch_ai.py
│
├── utils/                              # 工具函數
│   ├── diagnose_gpu.py
│   └── verify_pytorch.py
│
└── docs/                               # 文檔
    ├── README.md
    ├── SETUP.md
    ├── ARCHITECTURE.md
    ├── TRAINING_GUIDE.md
    ├── REWARD_MECHANISM.md
    └── PERFORMANCE.md
```

---

## 🎯 整理步驟建議

### 第一階段：刪除明確無用的檔案

```bash
# 1. 刪除臨時檔案
rm temp_analyze_csv.py
rm temp_summary.py
rm temp_summary_v2.py
rm update_checkpoint_config.py
rm analysis_results.txt

# 2. 刪除測試檔案
rm quick_test.py

# 3. 刪除 legacy
rm -rf legacy/

# 4. 刪除重複的分析腳本
rm analyze_games.py
rm analyze_csv_metrics.py
rm analyze_asymmetry.py
rm detailed_analysis.py
rm reward_analysis_simple.py
rm monitor_games.py
rm check_history.py
```

### 第二階段：整理文檔

```bash
# 1. 移動重複的 PERFORMANCE 文件
rm PERFORMANCE_EVALUATION.md
rm PERFORMANCE_OPTIMIZATION.md

# 2. 移動 REWARD 文檔到 docs/
mv REWARD_MECHANISM.md docs/
mv REWARD_IMPORTANCE.md docs/

# 3. 合併重複文檔（手動處理）
# 將 PROJECT_STRUCTURE.md 內容合併到 docs/ARCHITECTURE.md
# 將 TRACKING_METRICS.md 內容合併到 docs/TRAINING_GUIDE.md
```

### 第三階段：創建新目錄結構

```bash
# 1. 創建新目錄
mkdir -p analysis
mkdir -p interactive

# 2. 移動檔案
mv analyze_progress.py analysis/
mv monitor_training.py analysis/
mv visualize_game.py analysis/

mv play_vs_ai.py interactive/
mv watch_ai.py interactive/
mv main.py interactive/  # 如果 main.py 是互動入口
```

### 第四階段：清理測試工具

```bash
# 移動或刪除不常用的測試工具
mv utils/test_mcts_pytorch.py utils/archive/  # 或直接刪除
```

---

## 📊 整理前後對比

### 整理前
- 總檔案數：~60 個 Python 檔案
- 分析腳本：10 個（功能重複）
- 文檔：分散在多處
- 臨時檔案：5+ 個

### 整理後
- 總檔案數：~35 個 Python 檔案
- 分析腳本：3 個（精簡核心）
- 文檔：統一在 docs/
- 臨時檔案：0 個

**減少：~40% 的檔案數量**

---

## ✅ 推薦的整理策略

### 保守策略（推薦新手）
1. 只刪除明確無用的檔案（temp_*, legacy/）
2. 創建 analysis/ 和 interactive/ 目錄
3. 移動檔案但不刪除功能性腳本
4. 整理文檔到 docs/

### 激進策略（推薦有經驗者）
1. 刪除所有重複功能的腳本
2. 只保留 3 個核心分析腳本
3. 合併所有文檔
4. 清理到最小必要集合

---

## 🚀 執行計劃

你想要：
- **選項 A**：我提供刪除/移動命令，你審查後執行？
- **選項 B**：我直接執行保守策略的整理？
- **選項 C**：我直接執行激進策略的整理？
- **選項 D**：自訂策略（告訴我你想保留/刪除什麼）

請選擇一個選項，我會立即執行！
