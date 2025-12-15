"""实时监控游戏日志 - 观察最新的游戏数据"""

import os
import time
import pandas as pd
from datetime import datetime


def clear_screen():
    """清屏"""
    os.system('cls' if os.name == 'nt' else 'clear')


def monitor_games(log_dir='logs/games', refresh_interval=3):
    """
    实时监控游戏日志

    Args:
        log_dir: 日志目录
        refresh_interval: 刷新间隔（秒）
    """
    summary_file = os.path.join(log_dir, 'games_summary.csv')

    print("🎮 实时游戏监控")
    print("=" * 60)
    print(f"日志文件: {summary_file}")
    print(f"刷新间隔: {refresh_interval}秒")
    print("按 Ctrl+C 退出")
    print("=" * 60)

    last_count = 0

    try:
        while True:
            if not os.path.exists(summary_file):
                print(f"\n⏳ 等待日志文件生成...")
                print(f"   请先运行: python train_pipeline.py")
                time.sleep(refresh_interval)
                continue

            # 读取CSV
            df = pd.read_csv(summary_file)
            current_count = len(df)

            # 检测新增游戏
            if current_count > last_count:
                clear_screen()
                print("=" * 60)
                print(f"🎮 实时游戏监控 - {datetime.now().strftime('%H:%M:%S')}")
                print("=" * 60)

                # 总体统计
                print(f"\n📊 总体统计")
                print(f"   总游戏数: {len(df)}")
                print(f"   平均步数: {df['num_moves'].mean():.1f}")
                print(f"   黑胜率: {df['black_win'].sum()/len(df)*100:.1f}%")
                print(f"   白胜率: {df['white_win'].sum()/len(df)*100:.1f}%")
                print(f"   平局率: {df['draw'].sum()/len(df)*100:.1f}%")

                # 当前迭代统计
                if 'iteration' in df.columns and len(df) > 0:
                    current_iter = df['iteration'].max()
                    current_iter_df = df[df['iteration'] == current_iter]

                    print(f"\n📍 当前迭代: {current_iter}")
                    print(f"   已完成游戏: {len(current_iter_df)}")
                    print(f"   平均步数: {current_iter_df['num_moves'].mean():.1f}")
                    print(f"   平均策略熵: {current_iter_df['avg_policy_entropy'].mean():.3f}")

                # 最近10局
                print(f"\n🎯 最近10局游戏")
                print("-" * 60)
                recent = df.tail(10)[['iteration', 'game_num', 'num_moves', 'winner', 'game_duration_sec']]
                recent.columns = ['迭代', '局号', '步数', '胜者', '时长(秒)']
                print(recent.to_string(index=False))

                # 新增游戏提示
                new_games = current_count - last_count
                if new_games > 0:
                    print(f"\n✅ 新增 {new_games} 局游戏")

                # 趋势分析（如果有足够数据）
                if len(df) >= 20:
                    recent_20 = df.tail(20)
                    older_20 = df.tail(40).head(20) if len(df) >= 40 else df.head(20)

                    recent_avg = recent_20['num_moves'].mean()
                    older_avg = older_20['num_moves'].mean()

                    print(f"\n📈 趋势分析（最近20局 vs 之前20局）")
                    print(f"   平均步数: {recent_avg:.1f} vs {older_avg:.1f}", end="")
                    if recent_avg > older_avg * 1.1:
                        print(" ✅ 游戏变长（AI在学习）")
                    elif recent_avg < older_avg * 0.9:
                        print(" ⚠️  游戏变短")
                    else:
                        print(" ➡️  稳定")

                    recent_entropy = recent_20['avg_policy_entropy'].mean()
                    older_entropy = older_20['avg_policy_entropy'].mean()
                    print(f"   策略熵: {recent_entropy:.3f} vs {older_entropy:.3f}", end="")
                    if recent_entropy < older_entropy * 0.95:
                        print(" ✅ 越来越确定")
                    else:
                        print(" ⚠️  仍在探索")

                last_count = current_count

            else:
                # 无新数据，仅更新时间
                print(f"\r⏳ 等待新游戏... (最后更新: {datetime.now().strftime('%H:%M:%S')})", end="")

            time.sleep(refresh_interval)

    except KeyboardInterrupt:
        print("\n\n👋 监控已停止")


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='实时监控游戏日志')
    parser.add_argument('--log-dir', default='logs/games', help='日志目录')
    parser.add_argument('--interval', type=int, default=3, help='刷新间隔（秒）')

    args = parser.parse_args()

    monitor_games(log_dir=args.log_dir, refresh_interval=args.interval)
