"""游戏日志分析工具 - 查看和分析训练游戏数据"""

import os
import json
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


def load_games_summary(log_dir='logs/games'):
    """
    加载游戏汇总CSV

    Returns:
        pandas.DataFrame
    """
    summary_file = os.path.join(log_dir, 'games_summary.csv')
    if not os.path.exists(summary_file):
        print(f"❌ 未找到汇总文件: {summary_file}")
        return None

    df = pd.read_csv(summary_file)
    print(f"✅ 加载 {len(df)} 局游戏数据")
    return df


def load_iteration_summary(log_dir='logs/games'):
    """
    加载迭代汇总JSONL

    Returns:
        list of dict
    """
    summary_file = os.path.join(log_dir, 'iteration_summary.jsonl')
    if not os.path.exists(summary_file):
        print(f"❌ 未找到迭代汇总: {summary_file}")
        return []

    iterations = []
    with open(summary_file, 'r', encoding='utf-8') as f:
        for line in f:
            iterations.append(json.loads(line))

    print(f"✅ 加载 {len(iterations)} 次迭代统计")
    return iterations


def analyze_game_length(df):
    """分析游戏长度趋势"""
    print("\n" + "=" * 60)
    print("📊 游戏长度分析")
    print("=" * 60)

    print(f"总游戏数: {len(df)}")
    print(f"平均步数: {df['num_moves'].mean():.1f}")
    print(f"中位数步数: {df['num_moves'].median():.1f}")
    print(f"最短游戏: {df['num_moves'].min()} 步")
    print(f"最长游戏: {df['num_moves'].max()} 步")

    # 按迭代分组统计
    by_iteration = df.groupby('iteration')['num_moves'].agg(['mean', 'std', 'count'])
    print(f"\n按迭代统计（最近5次）:")
    print(by_iteration.tail())


def analyze_win_rate(df):
    """分析胜率分布"""
    print("\n" + "=" * 60)
    print("🎯 胜率分析")
    print("=" * 60)

    total_games = len(df)
    black_wins = df['black_win'].sum()
    white_wins = df['white_win'].sum()
    draws = df['draw'].sum()

    print(f"黑棋胜率: {black_wins/total_games*100:.1f}% ({black_wins}/{total_games})")
    print(f"白棋胜率: {white_wins/total_games*100:.1f}% ({white_wins}/{total_games})")
    print(f"平局率: {draws/total_games*100:.1f}% ({draws}/{total_games})")

    # 检查先手优势
    if abs(black_wins - white_wins) > total_games * 0.1:
        advantage = "黑棋" if black_wins > white_wins else "白棋"
        print(f"\n⚠️  {advantage}有明显先手优势")


def analyze_policy_entropy(df):
    """分析策略熵趋势"""
    print("\n" + "=" * 60)
    print("📈 策略熵分析（探索vs利用）")
    print("=" * 60)

    print(f"平均策略熵: {df['avg_policy_entropy'].mean():.3f}")
    print(f"最小策略熵: {df['avg_policy_entropy'].min():.3f} (更确定)")
    print(f"最大策略熵: {df['avg_policy_entropy'].max():.3f} (更探索)")

    # 按迭代查看熵的变化
    by_iteration = df.groupby('iteration')['avg_policy_entropy'].mean()
    print(f"\n策略熵趋势（最近5次迭代）:")
    print(by_iteration.tail())

    if len(by_iteration) > 5:
        recent_trend = by_iteration.tail(5).values
        if recent_trend[-1] < recent_trend[0]:
            print("✅ 策略熵下降 = AI越来越确定（好现象）")
        else:
            print("⚠️  策略熵上升 = AI仍在探索")


def analyze_iteration_progress(iterations):
    """分析迭代进度"""
    print("\n" + "=" * 60)
    print("⏱️  训练进度分析")
    print("=" * 60)

    if not iterations:
        print("无迭代数据")
        return

    latest = iterations[-1]
    print(f"最新迭代: {latest['iteration']}")
    print(f"平均步数: {latest['avg_moves']:.1f}")
    print(f"黑胜/白胜/平局: {latest['black_wins']}/{latest['white_wins']}/{latest['draws']}")
    print(f"平均游戏时长: {latest['avg_game_duration']:.1f}秒")

    # 步数增长趋势
    if len(iterations) >= 10:
        recent_moves = [it['avg_moves'] for it in iterations[-10:]]
        if recent_moves[-1] > recent_moves[0] * 1.2:
            print("\n✅ 游戏步数增长 = AI学会了更复杂的策略")
        elif recent_moves[-1] < recent_moves[0] * 0.8:
            print("\n⚠️  游戏步数减少 = 可能过度追求速胜")


def plot_training_progress(df, iterations, save_path='logs/games/analysis.png'):
    """绘制训练进度图表"""
    try:
        import matplotlib.pyplot as plt
        import matplotlib
        matplotlib.use('Agg')  # 无GUI后端

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('训练游戏分析', fontsize=16, fontproperties='SimHei')

        # 1. 游戏长度趋势
        ax = axes[0, 0]
        by_iteration = df.groupby('iteration')['num_moves'].agg(['mean', 'std'])
        ax.plot(by_iteration.index, by_iteration['mean'], marker='o', label='平均步数')
        ax.fill_between(by_iteration.index,
                        by_iteration['mean'] - by_iteration['std'],
                        by_iteration['mean'] + by_iteration['std'],
                        alpha=0.3)
        ax.set_xlabel('迭代次数')
        ax.set_ylabel('步数')
        ax.set_title('游戏长度趋势')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # 2. 胜率分布
        ax = axes[0, 1]
        by_iteration_wins = df.groupby('iteration')[['black_win', 'white_win', 'draw']].sum()
        by_iteration_wins.plot(kind='bar', stacked=True, ax=ax)
        ax.set_xlabel('迭代次数')
        ax.set_ylabel('游戏数')
        ax.set_title('胜负分布')
        ax.legend(['黑胜', '白胜', '平局'])
        ax.grid(True, alpha=0.3)

        # 3. 策略熵趋势
        ax = axes[1, 0]
        by_iteration_entropy = df.groupby('iteration')['avg_policy_entropy'].mean()
        ax.plot(by_iteration_entropy.index, by_iteration_entropy.values, marker='s', color='green')
        ax.set_xlabel('迭代次数')
        ax.set_ylabel('策略熵')
        ax.set_title('策略熵趋势（越低越确定）')
        ax.grid(True, alpha=0.3)

        # 4. 游戏时长
        ax = axes[1, 1]
        if iterations:
            iter_nums = [it['iteration'] for it in iterations]
            durations = [it['avg_game_duration'] for it in iterations]
            ax.plot(iter_nums, durations, marker='^', color='orange')
            ax.set_xlabel('迭代次数')
            ax.set_ylabel('时长（秒）')
            ax.set_title('平均游戏时长')
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\n📊 图表已保存: {save_path}")

    except ImportError:
        print("\n⚠️  未安装matplotlib，跳过绘图")
    except Exception as e:
        print(f"\n⚠️  绘图失败: {e}")


def show_recent_games(df, n=10):
    """显示最近N局游戏"""
    print("\n" + "=" * 60)
    print(f"🎮 最近 {n} 局游戏")
    print("=" * 60)

    recent = df.tail(n)[['iteration', 'game_num', 'num_moves', 'winner', 'avg_policy_entropy']]
    print(recent.to_string(index=False))


def main():
    """主函数"""
    print("=" * 60)
    print("🎮 五子棋训练游戏分析工具")
    print("=" * 60)

    # 加载数据
    df = load_games_summary()
    iterations = load_iteration_summary()

    if df is None or len(df) == 0:
        print("\n❌ 没有找到游戏数据")
        print("请先运行训练：python train_pipeline.py")
        return

    # 分析
    analyze_game_length(df)
    analyze_win_rate(df)
    analyze_policy_entropy(df)
    analyze_iteration_progress(iterations)
    show_recent_games(df, n=10)

    # 绘图
    plot_training_progress(df, iterations)

    print("\n" + "=" * 60)
    print("✅ 分析完成！")
    print("=" * 60)


if __name__ == '__main__':
    main()
