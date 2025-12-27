"""查看配置變更記錄

使用方法：
    python tools/view_config_changes.py

作者：Claude Code
日期：2025-12-27
"""
import json
import os
import sys
from pathlib import Path

# 添加項目根目錄到路徑
sys.path.insert(0, str(Path(__file__).parent.parent))


def view_config_changes(log_file='checkpoints/config_changes.json'):
    """查看配置變更記錄"""

    if not os.path.exists(log_file):
        print(f"📋 暫無配置變更記錄")
        print(f"   文件: {log_file}")
        return

    try:
        with open(log_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        print(f"❌ 讀取記錄失敗: {e}")
        return

    changes = data.get('changes', [])
    metadata = data.get('metadata', {})

    print("=" * 80)
    print("📊 配置變更記錄")
    print("=" * 80)
    print(f"總變更次數: {metadata.get('total_changes', len(changes))}")
    print()

    if not changes:
        print("暫無變更記錄")
        return

    # 顯示每次變更
    for i, change in enumerate(changes, 1):
        print(f"\n{'─' * 80}")
        print(f"變更 #{i}")
        print(f"{'─' * 80}")
        print(f"時間:     {change['timestamp']}")
        print(f"迭代:     {change['iteration']}")
        print(f"類型:     {change['change_type']}")
        if change.get('reason'):
            print(f"原因:     {change['reason']}")

        print(f"\n變更的參數 ({change['change_count']} 個):")
        print(f"{'─' * 80}")
        print(f"{'參數名稱':<30} {'舊值':<15} {'新值':<15} {'變化':<10}")
        print(f"{'─' * 80}")

        for param_name, param_change in change['changes'].items():
            old_val = param_change['old']
            new_val = param_change['new']
            change_pct = param_change.get('change_percent')

            # 格式化值
            if isinstance(old_val, float):
                old_str = f"{old_val:.6f}"
                new_str = f"{new_val:.6f}"
            else:
                old_str = str(old_val)
                new_str = str(new_val)

            # 格式化變化百分比
            if change_pct is not None:
                change_str = f"{change_pct:+.1f}%"
            else:
                change_str = "N/A"

            print(f"{param_name:<30} {old_str:<15} {new_str:<15} {change_str:<10}")

    print("\n" + "=" * 80)
    print(f"共 {len(changes)} 次配置變更")
    print("=" * 80)


def main():
    """主函數"""
    import argparse

    parser = argparse.ArgumentParser(description='查看配置變更記錄')
    parser.add_argument('--file', type=str, default='checkpoints/config_changes.json',
                       help='記錄文件路徑')
    args = parser.parse_args()

    view_config_changes(args.file)


if __name__ == '__main__':
    main()
