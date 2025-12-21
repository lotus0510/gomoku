# PyTorch 安装指南

## 安装命令

```bash
# 激活虚拟环境
.\venv\Scripts\activate

# 卸载 TensorFlow（可选，如果想完全切换）
# pip uninstall tensorflow tensorflow-intel tensorflow-estimator -y

# 安装 PyTorch with CUDA 11.8 support（推荐）
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# 或者 CUDA 12.1（如果你的驱动支持）
# pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

## 验证安装

```bash
python verify_pytorch.py
```

预期输出：
```
✅ PyTorch 已安装
✅ CUDA 可用
✅ GPU: NVIDIA GeForce RTX ...
```

## 版本信息

- PyTorch: 2.x
- CUDA: 11.8 或 12.1
- Python: 3.10
