# 開發日誌 (Development Log)

本文檔記錄 Gomoku AI 項目的詳細開發更新和變更記錄。

## [2025-12-21] PyTorch 遷移與系統現代化

### 重大變更

從 TensorFlow 完全遷移至 PyTorch 框架，以提升訓練效率、靈活性和開發體驗。

### 詳細更新內容

#### 1. 訓練流程重構 (`train_pipeline_pytorch.py`)

- **框架遷移**: `tf.keras` -> `torch.nn` + `torch.optim`。
- **混合精度訓練 (AMP)**: 引入 `torch.cuda.amp.autocast` 和 `GradScaler`，在 NVIDIA RTX 3070 Ti 上預期獲得 2-3 倍加速。
- **優化器升級**: 切換至 `AdamW`，內置權重衰減 (L2 正則化)，棄用手動 L2 添加到損失函數的方式。
- **損失函數優化**:
  - 策略頭: `CrossEntropyLoss`
  - 價值頭: `HuberLoss` (delta=1.0)
- **多進程優化**: 修復了 multiprocessing 序列化問題，將數據增強函數移至全局作用域。
- **維度適配**: 實現了 NumPy `(H, W, C)` 到 PyTorch `(C, H, W)` 的自動轉換。

#### 2. 核心組件更新

- **模型架構 (`core/neural_net.py`)**:
  - 實現 SE-ResNet (Squeeze-and-Excitation Residual Network)。
  - 支持動態計算設備選擇 (CUDA/CPU)。
- **MCTS (`core/mcts*.py`)**:
  - 適配 PyTorch 模型推理接口。
  - 保留了標準和批量 (Batched) 兩種實現。

#### 3. 基礎設施改進

- **檢查點管理**:
  - 統一使用 `.pth` 格式。
  - 包含模型權重、優化器狀態、調度器狀態、Scaler 狀態和訓練歷史。
- **分析工具**: 修復 `analyze_games.py` 中文顯示問題，支持 Windows 常見中文字體。

#### 4. 專案存儲庫清理

- 移除舊的 TensorFlow 訓練腳本和備份。
- 整理目錄結構：`docs/`, `legacy/`, `utils/`。

### 測試與驗證

- **快速測試**: 1 迭代 verify pipeline 通過。
- **擴展測試**: 10 迭代穩定性測試通過，勝率達到 94%。
- **功能驗證**: 混合精度、檢查點恢復、多進程自我對弈均正常工作。
