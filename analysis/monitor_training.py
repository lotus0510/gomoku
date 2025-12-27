"""训练监控脚本 - 实时显示关键指标和健康分析"""

import json
import time
import os
import sys
from datetime import datetime
from io import StringIO


def clear_screen():
    """清屏"""
    os.system('cls' if os.name == 'nt' else 'clear')


def get_status_icon(value, healthy_range, higher_is_better=True):
    """
    获取健康状态图标

    Args:
        value: 当前值
        healthy_range: (min, max) 健康范围
        higher_is_better: True表示越高越好，False表示越低越好

    Returns:
        (icon, color_code) tuple
    """
    if healthy_range is None:
        return "●", ""

    min_val, max_val = healthy_range

    if min_val <= value <= max_val:
        return "✓", "\033[92m"  # 绿色
    elif higher_is_better and value < min_val * 0.8:
        return "✗", "\033[91m"  # 红色
    elif not higher_is_better and value > max_val * 1.2:
        return "✗", "\033[91m"  # 红色
    else:
        return "⚠", "\033[93m"  # 黄色


def get_trend(values, window=5):
    """
    获取趋势

    Args:
        values: 数值列表
        window: 窗口大小

    Returns:
        (trend_icon, trend_text, change_percent)
    """
    if len(values) < window:
        return "━", "数据不足", 0.0

    recent = values[-window:]
    first_half = sum(recent[:window//2]) / (window//2)
    second_half = sum(recent[window//2:]) / (window - window//2)

    change_percent = ((second_half - first_half) / (abs(first_half) + 1e-8)) * 100

    if abs(change_percent) < 1:
        return "━", "稳定", change_percent
    elif change_percent < 0:
        return "↓", "下降", change_percent
    else:
        return "↑", "上升", change_percent


def display_training_status():
    """显示训练状态

    Returns:
        bool: 是否成功顯示
    """
    history_path = 'checkpoints/training_history.json'

    # 清屏
    clear_screen()

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

    if not history.get('iterations'):
        print("⏳ 训练刚开始，等待第一次迭代...")
        return False

    # 获取数据
    iterations = history['iterations']
    current_iter = iterations[-1]
    total_iters = 1000  # 默认值，可从配置读取

    # 基础指标
    policy_loss = history.get('policy_loss', [])
    value_loss = history.get('value_loss', [])

    # 关键指标
    value_mae = history.get('value_mae', [])
    value_std = history.get('value_std', [])
    gradient_norm = history.get('gradient_norm', [])
    policy_top1 = history.get('policy_top1_prob', [])
    policy_entropy = history.get('policy_entropy_train', [])
    learning_rate = history.get('learning_rate', [])
    win_rates = history.get('win_rate_vs_random', [])

    # 标题
    print("=" * 80)
    print("🎮 五子棋 AI 訓練監控 - 關鍵指標儀表板")
    print("=" * 80)
    print(f"⏰ 更新時間: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"📊 進度: {current_iter}/{total_iters} ({current_iter/total_iters*100:.1f}%)")
    print()

    # 核心健康指标
    print("🏥 核心健康指標")
    print("-" * 80)

    if value_mae:
        mae = value_mae[-1]
        icon, color = get_status_icon(mae, (0.70, 0.85), higher_is_better=False)
        trend_icon, trend_text, change = get_trend(value_mae)
        print(f"  {color}{icon}\033[0m 價值MAE:      {mae:.4f}  {trend_icon} {trend_text:4s} ({change:+.1f}%)  " +
              f"[目標: 0.70-0.85]")

        if mae > 0.95:
            print(f"     \033[91m⚠ 警告: 價值網絡崩潰風險！\033[0m")
        elif mae > 0.85:
            print(f"     \033[93m⚠ 注意: MAE偏高，需密切監控\033[0m")

    if value_std:
        std = value_std[-1]
        icon, color = get_status_icon(std, (0.3, 1.0), higher_is_better=True)
        trend_icon, trend_text, change = get_trend(value_std)
        print(f"  {color}{icon}\033[0m 價值Std:      {std:.4f}  {trend_icon} {trend_text:4s} ({change:+.1f}%)  " +
              f"[目標: >0.30]")

        if std < 0.2:
            print(f"     \033[91m⚠ 警告: 價值網絡判別力崩潰！\033[0m")

    if policy_loss:
        p_loss = policy_loss[-1]
        icon, color = get_status_icon(p_loss, (4.0, 5.5), higher_is_better=False)
        trend_icon, trend_text, change = get_trend(policy_loss)
        print(f"  {color}{icon}\033[0m 策略損失:     {p_loss:.4f}  {trend_icon} {trend_text:4s} ({change:+.1f}%)  " +
              f"[目標: <5.5]")

    if gradient_norm:
        grad = gradient_norm[-1]
        icon, color = get_status_icon(grad, (0.5, 5.0), higher_is_better=True)
        trend_icon, trend_text, change = get_trend(gradient_norm)
        print(f"  {color}{icon}\033[0m 梯度範數:     {grad:.4f}  {trend_icon} {trend_text:4s} ({change:+.1f}%)  " +
              f"[目標: 0.5-5.0]")

        if grad < 0.1:
            print(f"     \033[91m⚠ 警告: 梯度消失！\033[0m")
        elif grad > 10:
            print(f"     \033[91m⚠ 警告: 梯度爆炸！\033[0m")

    print()

    # 策略质量指标
    print("🎯 策略質量")
    print("-" * 80)

    if policy_top1:
        top1 = policy_top1[-1]
        icon, color = get_status_icon(top1, (0.15, 0.35), higher_is_better=False)
        print(f"  {color}{icon}\033[0m Top1概率:     {top1:.4f}  [目標: 0.15-0.35]")

        if top1 > 0.5:
            print(f"     \033[93m⚠ 套路化風險: 策略過度自信\033[0m")

    if policy_entropy:
        entropy = policy_entropy[-1]
        icon, color = get_status_icon(entropy, (2.0, 4.0), higher_is_better=True)
        print(f"  {color}{icon}\033[0m 策略熵:       {entropy:.4f}  [目標: 2.0-4.0]")

    # 获取最新胜率
    latest_win_rate = None
    for wr in reversed(win_rates):
        if wr is not None:
            latest_win_rate = wr
            break

    if latest_win_rate is not None:
        icon, color = get_status_icon(latest_win_rate, (0.85, 1.0), higher_is_better=True)
        print(f"  {color}{icon}\033[0m 勝率 (vs隨機): {latest_win_rate:.1%}  [目標: >85%]")

    print()

    # 训练状态
    print("⚙️ 訓練狀態")
    print("-" * 80)

    if learning_rate:
        lr = learning_rate[-1]
        print(f"  📉 學習率:       {lr:.6f}")

    if value_loss:
        v_loss = value_loss[-1]
        print(f"  📊 價值損失:     {v_loss:.4f}")

    print()

    # 问题预警
    warnings = []

    if value_mae and value_mae[-1] > 0.85:
        # 检查连续上升
        if len(value_mae) >= 5:
            recent_mae = value_mae[-5:]
            increases = sum(recent_mae[i] > recent_mae[i-1] for i in range(1, 5))
            if increases >= 3:
                warnings.append("🔴 價值MAE連續上升，可能即將崩潰")

    if value_std and value_std[-1] < 0.3:
        warnings.append("🔴 價值Std過低，判別力下降")

    if gradient_norm and gradient_norm[-1] < 0.1:
        warnings.append("🔴 梯度消失，學習停滯")

    if policy_top1 and policy_top1[-1] > 0.5:
        warnings.append("🟡 策略過度自信，可能套路化")

    if policy_loss and len(policy_loss) >= 20:
        # 检查策略损失是否plateaued
        recent_p_loss = policy_loss[-20:]
        if max(recent_p_loss) - min(recent_p_loss) < 0.1:
            warnings.append("🟡 策略損失平穩20次迭代，可能停滯")

    if warnings:
        print("⚠️ 問題預警")
        print("-" * 80)
        for warning in warnings:
            print(f"  {warning}")
        print()

    # 最近趋势（简化版）
    if len(iterations) >= 5:
        print("📈 最近5次趨勢")
        print("-" * 80)
        start_idx = len(iterations) - 5

        print(f"  迭代    策略損失   價值MAE   梯度")
        print(f"  " + "-" * 42)

        for i in range(start_idx, len(iterations)):
            iter_num = iterations[i]
            p_loss = policy_loss[i] if i < len(policy_loss) else 0
            mae = value_mae[i] if i < len(value_mae) else 0
            grad = gradient_norm[i] if i < len(gradient_norm) else 0

            print(f"  {iter_num:4d}    {p_loss:7.4f}    {mae:7.4f}   {grad:5.2f}")

        print()

    # 提示信息
    print("=" * 80)
    print("💡 快捷操作:")
    print("  • Ctrl+C - 停止監控")
    print("  • python analysis/plot_comprehensive.py - 生成完整圖表")
    print("  • python analysis/diagnose_training.py - 完整健康診斷")
    print("=" * 80)

    return True


def main():
    """主函数"""
    print("啟動訓練監控...")
    print("按 Ctrl+C 退出\n")

    refresh_interval = 5  # 刷新间隔（秒）

    try:
        while True:
            # 显示训练状态
            success = display_training_status()

            if not success:
                time.sleep(2)
            else:
                time.sleep(refresh_interval)

    except KeyboardInterrupt:
        print("\n\n監控已停止")
        sys.exit(0)


if __name__ == '__main__':
    main()
