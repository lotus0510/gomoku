"""检查 PyTorch GPU 是否可用"""

import torch

print("=" * 60)
print("PyTorch GPU 检测脚本")
print("=" * 60)

# 检查 PyTorch 版本
print(f"\n📦 PyTorch 版本: {torch.__version__}")

# 检查 CUDA 可用性
cuda_available = torch.cuda.is_available()
print(f"\n🎮 CUDA 可用: {cuda_available}")

if cuda_available:
    # GPU 数量
    gpu_count = torch.cuda.device_count()
    print(f"   GPU 数量: {gpu_count}")

    # 当前 GPU
    current_device = torch.cuda.current_device()
    print(f"   当前设备: {current_device}")

    # GPU 详细信息
    for i in range(gpu_count):
        print(f"\n   GPU {i}:")
        print(f"   名称: {torch.cuda.get_device_name(i)}")
        print(f"   计算能力: {torch.cuda.get_device_capability(i)}")

        # 内存信息
        props = torch.cuda.get_device_properties(i)
        print(f"   总内存: {props.total_memory / 1024**3:.2f} GB")
        print(f"   多处理器数量: {props.multi_processor_count}")

    # CUDA 版本
    print(f"\n🔧 CUDA 编译版本: {torch.version.cuda}")

    # cuDNN 版本
    if torch.backends.cudnn.is_available():
        print(f"   cuDNN 版本: {torch.backends.cudnn.version()}")
        print(f"   cuDNN 启用: {torch.backends.cudnn.enabled}")

    # 测试 GPU 计算
    print("\n🧪 测试 GPU 计算...")
    try:
        device = torch.device('cuda:0')
        x = torch.rand(1000, 1000, device=device)
        y = torch.rand(1000, 1000, device=device)
        z = torch.matmul(x, y)
        print("   ✅ GPU 计算成功!")
        print(f"   结果形状: {z.shape}")
        print(f"   设备: {z.device}")
    except Exception as e:
        print(f"   ❌ GPU 计算失败: {e}")

else:
    print("\n❌ CUDA 不可用!")
    print("\n可能的原因:")
    print("   1. 安装的是 CPU 版本的 PyTorch")
    print("   2. NVIDIA 驱动未安装或版本过旧")
    print("   3. CUDA 版本不兼容")

    print("\n💡 解决方案:")
    print("   1. 确认 NVIDIA 驱动已安装: nvidia-smi")
    print("   2. 重新安装 GPU 版本:")
    print("      pip uninstall torch torchvision torchaudio")
    print("      pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121")

# 显示可用设备
print("\n🖥️  PyTorch 设备:")
print(f"   默认设备: {torch.device('cuda' if torch.cuda.is_available() else 'cpu')}")

print("\n" + "=" * 60)
