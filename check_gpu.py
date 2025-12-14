"""检查 GPU 是否可用"""

import tensorflow as tf
import sys

print("=" * 60)
print("GPU 检测脚本")
print("=" * 60)

# 检查 TensorFlow 版本
print(f"\n📦 TensorFlow 版本: {tf.__version__}")

# 检查是否为 GPU 版本
print(f"   构建信息: {tf.sysconfig.get_build_info().get('cuda_version', 'CPU-only')}")

# 列出所有设备
print("\n🖥️  可用设备:")
devices = tf.config.list_physical_devices()
for device in devices:
    print(f"   - {device.device_type}: {device.name}")

# 检查 GPU
gpus = tf.config.list_physical_devices('GPU')
print(f"\n🎮 GPU 数量: {len(gpus)}")

if gpus:
    print("\n✅ GPU 已检测到!")
    for i, gpu in enumerate(gpus):
        print(f"\n   GPU {i}:")
        print(f"   名称: {gpu.name}")

        # 获取 GPU 详细信息
        try:
            gpu_details = tf.config.experimental.get_device_details(gpu)
            if gpu_details:
                print(f"   详情: {gpu_details}")
        except:
            pass

        # 检查内存增长设置
        try:
            memory_growth = tf.config.experimental.get_memory_growth(gpu)
            print(f"   内存增长: {memory_growth}")
        except:
            pass

    # 测试 GPU 计算
    print("\n🧪 测试 GPU 计算...")
    try:
        with tf.device('/GPU:0'):
            a = tf.random.normal([1000, 1000])
            b = tf.random.normal([1000, 1000])
            c = tf.matmul(a, b)
        print("   ✅ GPU 计算成功!")

    except Exception as e:
        print(f"   ❌ GPU 计算失败: {e}")
else:
    print("\n❌ 未检测到 GPU!")
    print("\n可能的原因:")
    print("   1. 安装的是 CPU 版本的 TensorFlow")
    print("   2. CUDA/cuDNN 未安装或版本不匹配")
    print("   3. GPU 驱动问题")
    print("   4. 环境变量未设置")

    print("\n💡 解决方案:")
    print("   1. 检查是否安装了 NVIDIA 驱动")
    print("   2. 安装 CUDA 和 cuDNN")
    print("   3. 重新安装 TensorFlow GPU 版本:")
    print("      pip uninstall tensorflow")
    print("      pip install tensorflow[and-cuda]")

# 检查 CUDA 可用性（底层）
print("\n🔧 CUDA 可用性:")
print(f"   tf.test.is_built_with_cuda(): {tf.test.is_built_with_cuda()}")
print(f"   tf.test.is_gpu_available(): ", end="")
try:
    # 注意：这个方法在新版本中已弃用
    print(tf.test.is_gpu_available(cuda_only=True))
except:
    print("方法已弃用，使用上面的检测结果")

print("\n" + "=" * 60)
