"""PyTorch GPU 验证脚本"""

print("=" * 60)
print("PyTorch GPU 验证")
print("=" * 60)

# 1. PyTorch 安装检测
print("\n1. PyTorch 安装:")
try:
    import torch
    print(f"   ✅ PyTorch 版本: {torch.__version__}")
    print(f"   安装路径: {torch.__file__}")
except ImportError:
    print(f"   ❌ PyTorch 未安装")
    print(f"   请运行: pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118")
    exit(1)

# 2. CUDA 支持检测
print("\n2. CUDA 支持:")
print(f"   PyTorch 编译 CUDA 版本: {torch.version.cuda}")
print(f"   cuDNN 版本: {torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else 'N/A'}")

# 3. GPU 检测
print("\n3. GPU 检测:")
if torch.cuda.is_available():
    print(f"   ✅ CUDA 可用")
    print(f"   GPU 数量: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        print(f"   GPU {i}: {torch.cuda.get_device_name(i)}")
        props = torch.cuda.get_device_properties(i)
        print(f"      显存: {props.total_memory / 1024**3:.1f} GB")
        print(f"      计算能力: {props.major}.{props.minor}")
else:
    print(f"   ❌ CUDA 不可用")
    print(f"   将使用 CPU 模式")

# 4. 测试 GPU 运算
print("\n4. GPU 运算测试:")
try:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"   当前设备: {device}")

    # 创建张量
    x = torch.randn(1000, 1000, device=device)
    y = torch.randn(1000, 1000, device=device)

    # 矩阵乘法
    import time
    start = time.time()
    z = torch.matmul(x, y)
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    elapsed = time.time() - start

    print(f"   ✅ 运算成功")
    print(f"   矩阵乘法耗时: {elapsed*1000:.2f} ms")

except Exception as e:
    print(f"   ❌ 运算失败: {e}")

# 5. 混合精度支持
print("\n5. 混合精度支持:")
if torch.cuda.is_available():
    try:
        from torch.cuda.amp import autocast, GradScaler

        with autocast():
            x_fp16 = torch.randn(10, 10, device='cuda')
            y_fp16 = torch.matmul(x_fp16, x_fp16)

        print(f"   ✅ 自动混合精度 (AMP) 可用")
        print(f"   FP16 运算正常")
    except Exception as e:
        print(f"   ⚠️  混合精度测试失败: {e}")
else:
    print(f"   ⚠️  需要 GPU 才能使用混合精度")

# 6. 推荐配置
print("\n" + "=" * 60)
print("推荐配置")
print("=" * 60)

if torch.cuda.is_available():
    print("\n✅ GPU 可用，推荐配置：")
    print("   - 使用混合精度训练 (torch.cuda.amp)")
    print("   - 批量大小可以增大（更多显存）")
    print("   - 启用 cudnn.benchmark 加速")
    print("\n代码示例：")
    print("   torch.backends.cudnn.benchmark = True")
    print("   with torch.cuda.amp.autocast():")
    print("       output = model(input)")
else:
    print("\n⚠️  GPU 不可用，使用 CPU 模式")
    print("   - 训练速度会较慢")
    print("   - 建议减小批量大小")
    print("   - 建议减小模型大小")

print("\n" + "=" * 60)
