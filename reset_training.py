#!/usr/bin/env python3
"""重置训练数据 - 提供多种清理选项"""

import os
import shutil
import sys
from datetime import datetime
import zipfile


def get_directory_size(path):
    """获取目录大小（MB）"""
    total = 0
    try:
        for dirpath, dirnames, filenames in os.walk(path):
            for f in filenames:
                fp = os.path.join(dirpath, f)
                if os.path.exists(fp):
                    total += os.path.getsize(fp)
    except Exception:
        pass
    return total / (1024 * 1024)  # Convert to MB


def show_current_status():
    """显示当前训练数据状态"""
    print("=" * 70)
    print("  当前训练数据状态")
    print("=" * 70)
    print()

    # 检查 checkpoints
    if os.path.exists('checkpoints'):
        size = get_directory_size('checkpoints')
        model_count = len([f for f in os.listdir('checkpoints') if f.endswith('.pth')])
        print(f"📁 checkpoints/  - {size:.1f} MB")
        print(f"   • {model_count} 个模型文件")
        if os.path.exists('checkpoints/training_history.json'):
            import json
            with open('checkpoints/training_history.json', 'r') as f:
                history = json.load(f)
                print(f"   • {len(history.get('iterations', []))} 次迭代记录")
    else:
        print("📁 checkpoints/  - 不存在")

    # 检查 logs
    if os.path.exists('logs'):
        size = get_directory_size('logs')
        print(f"\n📁 logs/  - {size:.1f} MB")

        if os.path.exists('logs/games/games_summary.csv'):
            with open('logs/games/games_summary.csv', 'r') as f:
                game_count = sum(1 for line in f) - 1  # 减去表头
            print(f"   • {game_count} 局游戏记录")

        # 统计 PNG 文件
        png_count = 0
        if os.path.exists('logs/games'):
            png_count = len([f for f in os.listdir('logs/games') if f.endswith('.png')])
        if png_count > 0:
            print(f"   • {png_count} 个分析图表")
    else:
        print("\n📁 logs/  - 不存在")

    # 检查缓存
    cache_dirs = []
    for root, dirs, files in os.walk('.'):
        if '__pycache__' in dirs:
            cache_dirs.append(os.path.join(root, '__pycache__'))

    if cache_dirs:
        cache_size = sum(get_directory_size(d) for d in cache_dirs)
        print(f"\n📁 __pycache__/  - {cache_size:.1f} MB ({len(cache_dirs)} 个目录)")

    print()


def create_backup():
    """创建备份压缩包"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_name = f"training_backup_{timestamp}.zip"

    print(f"正在创建备份: {backup_name}")
    print()

    try:
        with zipfile.ZipFile(backup_name, 'w', zipfile.ZIP_DEFLATED) as zipf:
            # 备份 checkpoints
            if os.path.exists('checkpoints'):
                print("[1/2] 备份 checkpoints/...")
                for root, dirs, files in os.walk('checkpoints'):
                    for file in files:
                        file_path = os.path.join(root, file)
                        arcname = os.path.relpath(file_path, '.')
                        zipf.write(file_path, arcname)
                print("      ✓ 完成")

            # 备份 logs
            if os.path.exists('logs'):
                print("[2/2] 备份 logs/...")
                for root, dirs, files in os.walk('logs'):
                    for file in files:
                        file_path = os.path.join(root, file)
                        arcname = os.path.relpath(file_path, '.')
                        zipf.write(file_path, arcname)
                print("      ✓ 完成")

        backup_size = os.path.getsize(backup_name) / (1024 * 1024)
        print()
        print(f"✓ 备份完成: {backup_name} ({backup_size:.1f} MB)")
        return True

    except Exception as e:
        print(f"\n✗ 备份失败: {e}")
        return False


def reset_complete():
    """完全重置（选项1）"""
    print("\n正在执行完全重置...")
    print()

    # 删除 checkpoints
    if os.path.exists('checkpoints'):
        print("[1/4] 删除 checkpoints/...")
        try:
            shutil.rmtree('checkpoints')
            os.makedirs('checkpoints', exist_ok=True)
            os.makedirs('checkpoints/summaries', exist_ok=True)
            print("      ✓ 完成")
        except Exception as e:
            print(f"      ✗ 错误: {e}")
    else:
        print("[1/4] checkpoints/ 不存在，跳过")

    # 删除 logs
    if os.path.exists('logs'):
        print("[2/4] 删除 logs/...")
        try:
            shutil.rmtree('logs')
            os.makedirs('logs', exist_ok=True)
            os.makedirs('logs/games', exist_ok=True)
            print("      ✓ 完成")
        except Exception as e:
            print(f"      ✗ 错误: {e}")
    else:
        print("[2/4] logs/ 不存在，跳过")

    # 删除缓存
    print("[3/4] 删除 Python 缓存...")
    cache_count = 0
    for root, dirs, files in os.walk('.'):
        if '__pycache__' in dirs:
            try:
                shutil.rmtree(os.path.join(root, '__pycache__'))
                cache_count += 1
            except Exception:
                pass
    print(f"      ✓ 删除了 {cache_count} 个缓存目录")

    # 删除可视化文件
    print("[4/4] 删除分析文件...")
    viz_files = ['training_visualization.png']
    deleted = 0
    for f in viz_files:
        if os.path.exists(f):
            try:
                os.remove(f)
                deleted += 1
            except Exception:
                pass
    print(f"      ✓ 删除了 {deleted} 个文件")


def reset_models_only():
    """只删除模型（选项2）"""
    print("\n正在删除模型文件，保留日志...")
    print()

    if os.path.exists('checkpoints'):
        print("[1/1] 删除 checkpoints/...")
        try:
            shutil.rmtree('checkpoints')
            os.makedirs('checkpoints', exist_ok=True)
            os.makedirs('checkpoints/summaries', exist_ok=True)
            print("      ✓ 完成")
            print("\n✓ logs/ 已保留，可用于分析")
        except Exception as e:
            print(f"      ✗ 错误: {e}")
    else:
        print("[1/1] checkpoints/ 不存在，跳过")


def reset_logs_only():
    """只删除日志（选项3）"""
    print("\n正在删除日志，保留模型...")
    print()

    if os.path.exists('logs'):
        print("[1/1] 删除 logs/...")
        try:
            shutil.rmtree('logs')
            os.makedirs('logs', exist_ok=True)
            os.makedirs('logs/games', exist_ok=True)
            print("      ✓ 完成")
            print("\n✓ checkpoints/ 已保留，可继续训练")
        except Exception as e:
            print(f"      ✗ 错误: {e}")
    else:
        print("[1/1] logs/ 不存在，跳过")


def backup_then_reset():
    """备份后重置（选项4）"""
    print()
    if create_backup():
        print()
        reset_complete()
    else:
        print("\n✗ 备份失败，取消重置操作")
        return


def show_menu():
    """显示菜单"""
    print("=" * 70)
    print("  选择清理选项")
    print("=" * 70)
    print()
    print("  1. 完全重置 - 删除所有训练数据（模型 + 日志）")
    print("  2. 只删除模型 - 保留 CSV 日志用于分析")
    print("  3. 只删除日志 - 保留模型继续训练")
    print("  4. 备份后重置 - 先创建 ZIP 备份再完全重置")
    print("  5. 取消")
    print()

    while True:
        try:
            choice = input("请选择 (1-5): ").strip()
            if choice in ['1', '2', '3', '4', '5']:
                return choice
            print("无效选择，请输入 1-5")
        except KeyboardInterrupt:
            return '5'


def confirm_action(action_desc):
    """确认操作"""
    print()
    print("=" * 70)
    response = input(f"确认{action_desc}? (yes/no): ").strip().lower()
    return response in ['yes', 'y', '是', 'y']


def main():
    """主函数"""
    print("\n")
    show_current_status()

    choice = show_menu()

    if choice == '5':
        print("\n取消操作")
        sys.exit(0)

    # 根据选择执行相应操作
    actions = {
        '1': ('完全重置', reset_complete),
        '2': ('只删除模型', reset_models_only),
        '3': ('只删除日志', reset_logs_only),
        '4': ('备份后完全重置', backup_then_reset),
    }

    action_name, action_func = actions[choice]

    if not confirm_action(action_name):
        print("\n取消操作")
        sys.exit(0)

    action_func()

    print()
    print("=" * 70)
    print("  清理完成！")
    print("=" * 70)
    print()
    print("现在可以开始训练：")
    print("  python train_pipeline_pytorch.py")
    print("  python train_pipeline_pytorch.py -f  # 快速测试")
    print()


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n取消操作")
    except Exception as e:
        print(f"\n\n错误: {e}")
        import traceback
        traceback.print_exc()
    finally:
        input("\n按 Enter 键退出...")
