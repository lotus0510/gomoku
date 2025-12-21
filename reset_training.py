#!/usr/bin/env python3
"""重置训练数据 - 删除所有模型和历史记录"""

import os
import shutil
import sys


def confirm_reset():
    """确认是否重置"""
    print("=" * 60)
    print("  重置训练数据")
    print("=" * 60)
    print()
    print("即将删除以下内容：")
    print("  ✗ checkpoints/ - 所有模型和训练历史")
    print("  ✗ logs/ - 游戏日志和 TensorBoard 日志")
    print("  ✗ __pycache__/ - Python 缓存")
    print("  ✗ training_visualization.png - 可视化图片")
    print()

    response = input("确认删除所有训练数据? (yes/no): ").strip().lower()
    return response in ['yes', 'y', '是']


def reset_training():
    """执行重置"""
    # 删除检查点目录
    if os.path.exists('checkpoints'):
        print("[1/4] 删除 checkpoints/...")
        try:
            shutil.rmtree('checkpoints')
            os.makedirs('checkpoints', exist_ok=True)
            print("      ✓ 完成")
        except Exception as e:
            print(f"      ✗ 错误: {e}")
    else:
        print("[1/4] checkpoints/ 不存在，跳过")

    # 删除日志目录
    if os.path.exists('logs'):
        print("[2/4] 删除 logs/...")
        try:
            shutil.rmtree('logs')
            os.makedirs('logs', exist_ok=True)
            print("      ✓ 完成")
        except Exception as e:
            print(f"      ✗ 错误: {e}")
    else:
        print("[2/4] logs/ 不存在，跳过")

    # 删除 Python 缓存
    print("[3/4] 删除 Python 缓存...")
    cache_count = 0

    # 删除当前目录的 __pycache__
    if os.path.exists('__pycache__'):
        try:
            shutil.rmtree('__pycache__')
            cache_count += 1
        except Exception as e:
            print(f"      ✗ 删除 __pycache__ 失败: {e}")

    # 递归删除所有子目录的 __pycache__
    for root, dirs, files in os.walk('.'):
        if '__pycache__' in dirs:
            cache_path = os.path.join(root, '__pycache__')
            try:
                shutil.rmtree(cache_path)
                cache_count += 1
            except Exception:
                pass

    print(f"      ✓ 删除了 {cache_count} 个缓存目录")

    # 删除可视化图片
    print("[4/4] 删除可视化图片...")
    viz_file = 'training_visualization.png'
    if os.path.exists(viz_file):
        try:
            os.remove(viz_file)
            print("      ✓ 完成")
        except Exception as e:
            print(f"      ✗ 错误: {e}")
    else:
        print("      - 文件不存在，跳过")

    print()
    print("=" * 60)
    print("  清理完成！")
    print("=" * 60)
    print()
    print("现在可以重新开始训练：")
    print("  python train_pipeline_pytorch.py --fast-test")
    print()


def main():
    """主函数"""
    if not confirm_reset():
        print("\n取消操作")
        sys.exit(0)

    print()
    print("正在清理...")
    print()

    reset_training()


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n取消操作")
    except Exception as e:
        print(f"\n\n错误: {e}")
    finally:
        input("\n按 Enter 键退出...")
