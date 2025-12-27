# 環境設置指南

## 系統需求

### 硬體需求
- **GPU**: NVIDIA GPU with CUDA support (推薦 RTX 3060 或更高)
- **RAM**: 至少 8GB (推薦 16GB+)
- **存儲**: 至少 5GB 可用空間

### 軟體需求
- **Python**: 3.10
- **CUDA**: 11.8 或 12.1
- **操作系統**: Windows 10/11, Linux, macOS

## 快速開始

### 1. 克隆專案

```bash
git clone <repository-url>
cd gomoku
```

### 2. 創建虛擬環境

**Windows:**
```bash
python -m venv .venv
.venv\Scripts\activate
```

**Linux/macOS:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. 安裝 PyTorch (GPU 版本)

根據你的 CUDA 版本選擇：

**CUDA 11.8 (推薦):**
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

**CUDA 12.1:**
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

**CPU 版本（無 GPU）:**
```bash
pip install torch torchvision torchaudio
```

### 4. 安裝其他依賴

```bash
pip install -r requirements.txt
```

如果沒有 requirements.txt，手動安裝：
```bash
pip install numpy pandas matplotlib seaborn
```

### 5. 驗證安裝

運行驗證腳本：
```bash
python utils/verify_installation.py
```

預期輸出：
```
✅ Python 版本: 3.10.x
✅ PyTorch 已安裝: 2.x.x
✅ CUDA 可用: True
✅ GPU 設備: NVIDIA GeForce RTX ...
✅ 所有依賴已安裝
```

## 常見問題

### Q1: CUDA 不可用

**症狀**: `torch.cuda.is_available()` 返回 `False`

**解決方案**:
1. 檢查 NVIDIA 驅動版本:
   ```bash
   nvidia-smi
   ```
   - CUDA 11.8 需要驅動 >= 452.39
   - CUDA 12.1 需要驅動 >= 527.41

2. 重新安裝正確版本的 PyTorch

3. 重啟電腦

### Q2: 內存不足錯誤

**症狀**: `RuntimeError: CUDA out of memory`

**解決方案**:
1. 在 `training/config.py` 中降低批次大小:
   ```python
   BATCH_SIZE = 128  # 從 256 降至 128
   ```

2. 減少 MCTS 模擬次數:
   ```python
   MCTS_SIMULATIONS = 100  # 從 200 降至 100
   ```

3. 使用 CPU 訓練（較慢）:
   ```python
   DEVICE = 'cpu'
   ```

### Q3: 模組找不到

**症狀**: `ModuleNotFoundError: No module named 'xxx'`

**解決方案**:
```bash
# 確保虛擬環境已激活
# Windows
.venv\Scripts\activate

# Linux/macOS
source .venv/bin/activate

# 重新安裝依賴
pip install -r requirements.txt
```

### Q4: 中文字體顯示問題

**症狀**: 圖表中的中文顯示為方框

**解決方案**:
1. 運行字體驗證:
   ```bash
   python verify_font.py
   ```

2. 如果需要，安裝中文字體:
   - **Windows**: 已內建 Microsoft JhengHei
   - **Linux**:
     ```bash
     sudo apt-get install fonts-noto-cjk
     ```
   - **macOS**: 已內建 Heiti TC

## 進階配置

### 使用 conda 環境（可選）

```bash
# 創建 conda 環境
conda create -n gomoku python=3.10
conda activate gomoku

# 安裝 PyTorch
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

# 安裝其他依賴
pip install numpy pandas matplotlib seaborn
```

### 多 GPU 訓練（實驗性）

如果有多個 GPU，可以在 `training/config.py` 中配置:

```python
# 使用特定 GPU
DEVICE = 'cuda:0'  # 使用第一個 GPU

# 或使用 DataParallel (需要代碼修改)
USE_DATA_PARALLEL = True
```

### Docker 部署（進階）

建立 Dockerfile:
```dockerfile
FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime

WORKDIR /app
COPY . /app

RUN pip install -r requirements.txt

CMD ["python", "train_pipeline.py"]
```

構建並運行:
```bash
docker build -t gomoku-ai .
docker run --gpus all -v $(pwd)/checkpoints:/app/checkpoints gomoku-ai
```

## 下一步

安裝完成後，請參考：
- [README.md](../../README.md) - 專案概述和使用指南
- [DEVELOPER_GUIDE.md](DEVELOPER_GUIDE.md) - 開發者指南
- [TRAINING_CONCEPTS.md](TRAINING_CONCEPTS.md) - 訓練概念和原理

開始訓練：
```bash
python train_pipeline.py
```
