"""GPU 诊断脚本"""

import sys
print("=" * 60)
print("GPU 环境诊断")
print("=" * 60)

# 1. Python 版本
print(f"\n1. Python 版本:")
print(f"   {sys.version}")

# 2. TensorFlow 版本
print(f"\n2. TensorFlow 检测:")
try:
    import tensorflow as tf
    print(f"   ✅ TensorFlow 版本: {tf.__version__}")
    print(f"   安装路径: {tf.__file__}")

    # 检查是否编译了 CUDA 支持
    print(f"\n   CUDA 支持:")
    print(f"   - 是否编译 CUDA: {tf.test.is_built_with_cuda()}")
    print(f"   - 是否编译 GPU 支持: {tf.test.is_built_with_gpu_support()}")

except ImportError as e:
    print(f"   ❌ TensorFlow 未安装: {e}")
    sys.exit(1)

# 3. GPU 设备检测
print(f"\n3. GPU 设备:")
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print(f"   ✅ 检测到 {len(gpus)} 个 GPU:")
    for i, gpu in enumerate(gpus):
        print(f"      GPU {i}: {gpu}")
        # 获取 GPU 详细信息
        try:
            gpu_details = tf.config.experimental.get_device_details(gpu)
            print(f"         详细信息: {gpu_details}")
        except:
            pass
else:
    print(f"   ❌ 未检测到 GPU")

# 4. CUDA 库检测
print(f"\n4. CUDA 库:")
try:
    cuda_version = tf.sysconfig.get_build_info()['cuda_version']
    cudnn_version = tf.sysconfig.get_build_info()['cudnn_version']
    print(f"   ✅ CUDA 版本: {cuda_version}")
    print(f"   ✅ cuDNN 版本: {cudnn_version}")
except Exception as e:
    print(f"   ⚠️  无法获取 CUDA 信息: {e}")

# 5. 测试 GPU 可用性
print(f"\n5. GPU 可用性测试:")
if tf.test.is_gpu_available():
    print(f"   ✅ GPU 可用（旧版 API）")
else:
    print(f"   ❌ GPU 不可用（旧版 API）")

# 新版 API
if gpus:
    print(f"   ✅ GPU 可用（新版 API）")

    # 测试简单运算
    print(f"\n6. GPU 运算测试:")
    try:
        with tf.device('/GPU:0'):
            a = tf.constant([[1.0, 2.0], [3.0, 4.0]])
            b = tf.constant([[1.0, 2.0], [3.0, 4.0]])
            c = tf.matmul(a, b)
        print(f"   ✅ GPU 运算成功")
        print(f"   结果设备: {c.device}")
    except Exception as e:
        print(f"   ❌ GPU 运算失败: {e}")
else:
    print(f"   ❌ GPU 不可用（新版 API）")

# 7. NVIDIA 驱动检测（Windows）
print(f"\n7. NVIDIA 驱动检测:")
try:
    import subprocess
    result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
    if result.returncode == 0:
        print(f"   ✅ nvidia-smi 可用")
        # 只打印前几行
        lines = result.stdout.split('\n')[:10]
        for line in lines:
            print(f"   {line}")
    else:
        print(f"   ❌ nvidia-smi 失败")
except Exception as e:
    print(f"   ⚠️  nvidia-smi 不可用: {e}")

print("\n" + "=" * 60)
print("诊断完成")
print("=" * 60)

# 建议
print("\n💡 建议:")
if not gpus:
    print("\n如果您有 NVIDIA GPU 但未检测到，请检查:")
    print("1. 安装了 CUDA Toolkit (https://developer.nvidia.com/cuda-downloads)")
    print("2. 安装了 cuDNN (https://developer.nvidia.com/cudnn)")
    print("3. TensorFlow 版本与 CUDA 版本匹配:")
    print("   - TensorFlow 2.10-2.15: CUDA 11.2, cuDNN 8.1")
    print("   - TensorFlow 2.16+: CUDA 12.x, cuDNN 8.9+")
    print("\n4. 重新安装 TensorFlow GPU 版本:")
    print("   pip uninstall tensorflow")
    print("   pip install tensorflow[and-cuda]  # TensorFlow 2.16+")
    print("   或")
    print("   pip install tensorflow-gpu  # 旧版本")
