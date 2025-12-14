"""训练监控脚本 - 实时显示训练进度"""

import json
import time
import os
import sys
from datetime import datetime


def clear_screen():
    """清屏"""
    os.system('cls' if os.name == 'nt' else 'clear')


def format_time(seconds):
    """格式化时间"""
    if seconds < 60:
        return f"{seconds:.0f}秒"
    elif seconds < 3600:
        return f"{seconds/60:.1f}分钟"
    else:
        return f"{seconds/3600:.1f}小时"


def display_training_status():
    """显示训练状态"""
    history_path = 'checkpoints/training_history.json'

    if not os.path.exists(history_path):
        print("⏳ 等待训练开始...")
        print(f"   正在等待文件: {history_path}")
        return False

    try:
        with open(history_path, 'r', encoding='utf-8') as f:
            history = json.load(f)
    except Exception as e:
        print(f"❌ 读取训练历史失败: {e}")
        return False

    if not history['iterations']:
        print("⏳ 训练刚开始，等待第一次迭代...")
        return False

    # 清屏
    clear_screen()

    # 获取最新数据
    iterations = history['iterations']
    total_loss = history['total_loss']
    policy_loss = history['policy_loss']
    value_loss = history['value_loss']
    win_rates = history['win_rate_vs_random']

    current_iter = iterations[-1]

    # 标题
    print("=" * 60)
    print("🎮 五子棋 AI 训练监控")
    print("=" * 60)
    print(f"⏰ 更新时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    # 进度信息
    print("📊 训练进度")
    print("-" * 60)
    print(f"   已完成迭代: {len(iterations)} 次")
    print(f"   最新迭代: 第 {current_iter} 次")
    print()

    # 损失值
    print("📉 损失值（最新）")
    print("-" * 60)
    print(f"   总损失:   {total_loss[-1]:.4f}")
    print(f"   策略损失: {policy_loss[-1]:.4f}")
    print(f"   价值损失: {value_loss[-1]:.4f}")

    # 如果有多次迭代，显示趋势
    if len(iterations) >= 2:
        total_change = total_loss[-1] - total_loss[-2]
        trend = "📉 下降" if total_change < 0 else "📈 上升"
        print(f"   趋势: {trend} ({total_change:+.4f})")
    print()

    # 损失历史（最近10次）
    if len(iterations) > 1:
        print("📊 损失历史（最近10次）")
        print("-" * 60)
        start_idx = max(0, len(iterations) - 10)
        for i in range(start_idx, len(iterations)):
            iter_num = iterations[i]
            loss = total_loss[i]
            print(f"   迭代 {iter_num:3d}: {loss:.4f}")
        print()

    # 评估结果
    latest_win_rate = None
    for i in range(len(win_rates) - 1, -1, -1):
        if win_rates[i] is not None:
            latest_win_rate = win_rates[i]
            latest_eval_iter = iterations[i]
            break

    if latest_win_rate is not None:
        print("🎯 评估结果（vs 随机玩家）")
        print("-" * 60)
        print(f"   迭代 {latest_eval_iter}: 胜率 {latest_win_rate:.1%}")

        # 评估历史
        eval_history = [(iterations[i], win_rates[i])
                       for i in range(len(win_rates))
                       if win_rates[i] is not None]

        if len(eval_history) > 1:
            print("   历史:")
            for iter_num, rate in eval_history[-5:]:  # 最近5次
                print(f"      迭代 {iter_num:3d}: {rate:.1%}")
        print()

    # 统计信息
    print("📈 统计信息")
    print("-" * 60)
    print(f"   最佳损失: {min(total_loss):.4f} (迭代 {iterations[total_loss.index(min(total_loss))]})")
    if latest_win_rate:
        all_win_rates = [r for r in win_rates if r is not None]
        if all_win_rates:
            print(f"   最高胜率: {max(all_win_rates):.1%}")
    print()

    # 提示信息
    print("=" * 60)
    print("💡 提示:")
    print("   - 按 Ctrl+C 停止监控")
    print("   - 运行 'python watch_ai.py' 观看 AI 下棋")
    print("   - 损失值下降 = 模型在学习")
    print("=" * 60)

    return True


def main():
    """主函数"""
    print("启动训练监控...")
    print("按 Ctrl+C 退出\n")

    refresh_interval = 5  # 刷新间隔（秒）

    try:
        while True:
            success = display_training_status()

            if not success:
                # 如果还没开始训练，等待时间短一点
                time.sleep(2)
            else:
                # 正常刷新间隔
                time.sleep(refresh_interval)

    except KeyboardInterrupt:
        print("\n\n监控已停止")
        sys.exit(0)


if __name__ == '__main__':
    main()
