"""游戏日志分析工具 - 查看和分析训练游戏数据"""

import os
import json
import pandas as pd
from pathlib import Path

# 不在顶层导入 matplotlib，避免无 GUI 环境报错
# import matplotlib.pyplot as plt

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


def load_training_history(checkpoint_dir='checkpoints'):
    """加载训练历史 JSON"""
    history_file = os.path.join(checkpoint_dir, 'training_history.json')
    if not os.path.exists(history_file):
        return {}
    
    try:
        with open(history_file, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"⚠️  加载训练历史失败: {e}")
        return {}


def analyze_game_length(df):
    """分析游戏长度趋势"""
    print("\n" + "=" * 60)
    print("📊 游戏长度分析")
    print("=" * 60)

    print(f"总游戏数: {len(df)}")
    print(f"平均步数: {df['num_moves'].mean():.1f}")
    if len(df) > 0:
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
    if total_games == 0:
        return

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
    if len(df) > 0:
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


def setup_chinese_font():
    """设置中文字体"""
    import matplotlib.pyplot as plt
    import matplotlib

    plt.rcParams['axes.unicode_minus'] = False

    found_font = False
    chinese_fonts = ['Microsoft JhengHei', 'Microsoft YaHei', 'SimHei', 'SimSun', 'Arial Unicode MS']

    for font in chinese_fonts:
        try:
            if font in [f.name for f in matplotlib.font_manager.fontManager.ttflist]:
                plt.rcParams['font.sans-serif'] = [font] + plt.rcParams['font.sans-serif']
                print(f"✅ 使用字体: {font}")
                found_font = True
                break
        except:
            continue

    if not found_font:
        print("⚠️  未找到常见中文字体，图表文字可能显示为方框")


def plot_training_loss(training_history, save_path='logs/games/training_loss.png'):
    """绘制训练损失图表"""
    try:
        import matplotlib.pyplot as plt
        import matplotlib
        import numpy as np

        matplotlib.use('Agg')
        setup_chinese_font()

        if not training_history or 'total_loss' not in training_history:
            print("⚠️  无损失数据")
            return

        iterations = training_history['iterations']
        total_loss = training_history['total_loss']
        policy_loss = training_history['policy_loss']
        value_loss = training_history['value_loss']

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('训练损失分析', fontsize=16)

        # 1. 总损失趋势
        ax = axes[0, 0]
        ax.plot(iterations, total_loss, marker='o', linewidth=2, color='red', label='总损失')
        ax.set_xlabel('迭代次数')
        ax.set_ylabel('损失值')
        ax.set_title('总损失趋势')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # 2. 策略损失 vs 价值损失
        ax = axes[0, 1]
        ax.plot(iterations, policy_loss, marker='s', linewidth=2, color='blue', label='策略损失')
        ax.plot(iterations, value_loss, marker='^', linewidth=2, color='green', label='价值损失')
        ax.set_xlabel('迭代次数')
        ax.set_ylabel('损失值')
        ax.set_title('策略损失 vs 价值损失')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # 3. 损失下降率（移动平均）
        ax = axes[1, 0]
        if len(total_loss) > 5:
            window = 5
            ma_loss = np.convolve(total_loss, np.ones(window)/window, mode='valid')
            ma_iters = iterations[window-1:]
            ax.plot(ma_iters, ma_loss, linewidth=2, color='purple', label=f'{window}迭代移动平均')
            ax.plot(iterations, total_loss, alpha=0.3, color='red', label='原始数据')
        else:
            ax.plot(iterations, total_loss, linewidth=2, color='purple')
        ax.set_xlabel('迭代次数')
        ax.set_ylabel('损失值')
        ax.set_title('总损失平滑趋势')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # 4. 损失统计
        ax = axes[1, 1]
        loss_data = [total_loss, policy_loss, value_loss]
        labels = ['总损失', '策略损失', '价值损失']
        bp = ax.boxplot(loss_data, labels=labels, patch_artist=True)
        for patch, color in zip(bp['boxes'], ['red', 'blue', 'green']):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)
        ax.set_ylabel('损失值')
        ax.set_title('损失分布统计')
        ax.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"📊 损失图表已保存: {save_path}")
        plt.close()

    except Exception as e:
        print(f"⚠️  绘制损失图表失败: {e}")
        import traceback
        traceback.print_exc()


def plot_training_progress(df, iterations, training_history=None, save_path='logs/games/analysis.png'):
    """绘制训练进度图表"""
    try:
        import matplotlib.pyplot as plt
        import matplotlib
        from matplotlib.font_manager import FontProperties

        matplotlib.use('Agg')  # 无GUI后端
        setup_chinese_font()

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('训练游戏分析', fontsize=16)

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

        # 4. AI vs 随机玩家胜率
        ax = axes[1, 1]
        if training_history:
            iterations = training_history.get('iterations', [])
            win_rates = training_history.get('win_rate_vs_random', [])
            
            # 过滤掉 None 值
            valid_data = [(i, w) for i, w in zip(iterations, win_rates) if w is not None]
            if valid_data:
                valid_iters, valid_rates = zip(*valid_data)
                ax.plot(valid_iters, valid_rates, marker='*', color='red', markersize=10, linestyle='-')
                ax.set_ylim(-0.05, 1.05)
                ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
                for i, r in zip(valid_iters, valid_rates):
                    # 在点上方标注百分比
                    ax.text(i, r + 0.02, f"{r:.0%}", ha='center', fontsize=9, color='darkred')
            
            ax.set_xlabel('迭代次数')
            ax.set_ylabel('胜率')
            ax.set_title('AI vs 随机玩家胜率')
            ax.grid(True, alpha=0.3)
        else:
            # 如果没有历史数据，回退到显示游戏时长
            ax.text(0.5, 0.5, "无评估数据", ha='center', va='center')

        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\n📊 图表已保存: {save_path}")

    except ImportError:
        print("\n⚠️  未安装matplotlib，跳过绘图")
    except Exception as e:
        print(f"\n⚠️  绘图失败: {e}")
        import traceback
        traceback.print_exc()


def plot_performance_analysis(df, save_path='logs/games/performance.png'):
    """绘制性能分析图表"""
    try:
        import matplotlib.pyplot as plt
        import matplotlib
        import numpy as np

        matplotlib.use('Agg')
        setup_chinese_font()

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('MCTS 性能分析', fontsize=16)

        # 1. MCTS 时间趋势
        ax = axes[0, 0]
        by_iteration = df.groupby('iteration')['avg_mcts_time'].agg(['mean', 'std'])
        ax.plot(by_iteration.index, by_iteration['mean'], marker='o', linewidth=2, color='orange')
        ax.fill_between(by_iteration.index,
                        by_iteration['mean'] - by_iteration['std'],
                        by_iteration['mean'] + by_iteration['std'],
                        alpha=0.3, color='orange')
        ax.set_xlabel('迭代次数')
        ax.set_ylabel('MCTS 时间 (秒/步)')
        ax.set_title('MCTS 平均时间趋势')
        ax.grid(True, alpha=0.3)

        # 2. 游戏时长趋势
        ax = axes[0, 1]
        by_iteration = df.groupby('iteration')['game_duration_sec'].agg(['mean', 'std'])
        ax.plot(by_iteration.index, by_iteration['mean'], marker='s', linewidth=2, color='brown')
        ax.fill_between(by_iteration.index,
                        by_iteration['mean'] - by_iteration['std'],
                        by_iteration['mean'] + by_iteration['std'],
                        alpha=0.3, color='brown')
        ax.set_xlabel('迭代次数')
        ax.set_ylabel('时长 (秒)')
        ax.set_title('游戏时长趋势')
        ax.grid(True, alpha=0.3)

        # 3. MCTS 时间 vs 游戏步数
        ax = axes[1, 0]
        scatter = ax.scatter(df['num_moves'], df['avg_mcts_time'],
                           alpha=0.3, c=df['iteration'], cmap='viridis', s=10)
        ax.set_xlabel('游戏步数')
        ax.set_ylabel('MCTS 时间 (秒/步)')
        ax.set_title('MCTS 时间 vs 游戏步数')
        plt.colorbar(scatter, ax=ax, label='迭代次数')
        ax.grid(True, alpha=0.3)

        # 4. 效率分析：每步耗时
        ax = axes[1, 1]
        df['time_per_move'] = df['game_duration_sec'] / df['num_moves']
        by_iteration = df.groupby('iteration')['time_per_move'].mean()
        ax.plot(by_iteration.index, by_iteration.values, marker='d', linewidth=2, color='teal')
        ax.set_xlabel('迭代次数')
        ax.set_ylabel('每步耗时 (秒)')
        ax.set_title('每步平均耗时')
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"📊 性能图表已保存: {save_path}")
        plt.close()

    except Exception as e:
        print(f"⚠️  绘制性能图表失败: {e}")
        import traceback
        traceback.print_exc()


def plot_data_distribution(df, save_path='logs/games/distribution.png'):
    """绘制数据分布图表"""
    try:
        import matplotlib.pyplot as plt
        import matplotlib
        import numpy as np

        matplotlib.use('Agg')
        setup_chinese_font()

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('数据分布分析', fontsize=16)

        # 1. 游戏步数分布
        ax = axes[0, 0]
        ax.hist(df['num_moves'], bins=30, edgecolor='black', alpha=0.7, color='skyblue')
        ax.axvline(df['num_moves'].mean(), color='red', linestyle='--',
                   label=f'均值: {df["num_moves"].mean():.1f}')
        ax.axvline(df['num_moves'].median(), color='green', linestyle='--',
                   label=f'中位数: {df["num_moves"].median():.1f}')
        ax.set_xlabel('游戏步数')
        ax.set_ylabel('频数')
        ax.set_title('游戏步数分布')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')

        # 2. 策略熵分布
        ax = axes[0, 1]
        ax.hist(df['avg_policy_entropy'], bins=30, edgecolor='black', alpha=0.7, color='lightgreen')
        ax.axvline(df['avg_policy_entropy'].mean(), color='red', linestyle='--',
                   label=f'均值: {df["avg_policy_entropy"].mean():.3f}')
        ax.set_xlabel('策略熵')
        ax.set_ylabel('频数')
        ax.set_title('策略熵分布')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')

        # 3. 胜负分布（饼图）
        ax = axes[1, 0]
        black_wins = df['black_win'].sum()
        white_wins = df['white_win'].sum()
        draws = df['draw'].sum()
        sizes = [black_wins, white_wins, draws]
        labels = [f'黑胜 ({black_wins})', f'白胜 ({white_wins})', f'平局 ({draws})']
        colors = ['#2c3e50', '#ecf0f1', '#95a5a6']
        ax.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
        ax.set_title('整体胜负分布')

        # 4. 先手优势演化
        ax = axes[1, 1]
        by_iteration = df.groupby('iteration').agg({
            'black_win': 'sum',
            'white_win': 'sum'
        })
        by_iteration['black_rate'] = by_iteration['black_win'] / (by_iteration['black_win'] + by_iteration['white_win'])
        ax.plot(by_iteration.index, by_iteration['black_rate'], marker='o', linewidth=2)
        ax.axhline(y=0.5, color='red', linestyle='--', label='50% (均衡)')
        ax.set_xlabel('迭代次数')
        ax.set_ylabel('黑方胜率')
        ax.set_title('先手优势演化')
        ax.set_ylim(0, 1)
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"📊 分布图表已保存: {save_path}")
        plt.close()

    except Exception as e:
        print(f"⚠️  绘制分布图表失败: {e}")
        import traceback
        traceback.print_exc()


def plot_correlation_analysis(df, save_path='logs/games/correlation.png'):
    """绘制相关性分析图表"""
    try:
        import matplotlib.pyplot as plt
        import matplotlib
        import numpy as np

        matplotlib.use('Agg')
        setup_chinese_font()

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('相关性分析', fontsize=16)

        # 1. 策略熵 vs 游戏步数
        ax = axes[0, 0]
        scatter = ax.scatter(df['avg_policy_entropy'], df['num_moves'],
                           alpha=0.3, c=df['iteration'], cmap='plasma', s=10)
        ax.set_xlabel('策略熵')
        ax.set_ylabel('游戏步数')
        ax.set_title('策略熵 vs 游戏步数')
        plt.colorbar(scatter, ax=ax, label='迭代次数')

        # 计算相关系数
        corr = np.corrcoef(df['avg_policy_entropy'], df['num_moves'])[0, 1]
        ax.text(0.05, 0.95, f'相关系数: {corr:.3f}',
                transform=ax.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        ax.grid(True, alpha=0.3)

        # 2. 策略熵 vs 胜率
        ax = axes[0, 1]
        by_entropy_bin = pd.cut(df['avg_policy_entropy'], bins=10)
        win_by_entropy = df.groupby(by_entropy_bin)['black_win'].mean()
        entropy_centers = [interval.mid for interval in win_by_entropy.index]
        ax.plot(entropy_centers, win_by_entropy.values, marker='o', linewidth=2)
        ax.set_xlabel('策略熵')
        ax.set_ylabel('黑方胜率')
        ax.set_title('策略熵 vs 胜率')
        ax.grid(True, alpha=0.3)

        # 3. 收敛性分析（策略熵标准差）
        ax = axes[1, 0]
        by_iteration = df.groupby('iteration')['avg_policy_entropy'].std()
        ax.plot(by_iteration.index, by_iteration.values, marker='s', linewidth=2, color='purple')
        ax.set_xlabel('迭代次数')
        ax.set_ylabel('策略熵标准差')
        ax.set_title('策略一致性（标准差越低越收敛）')
        ax.grid(True, alpha=0.3)

        # 4. 相关性热图
        ax = axes[1, 1]
        corr_cols = ['num_moves', 'game_duration_sec', 'avg_mcts_time', 'avg_policy_entropy']
        corr_matrix = df[corr_cols].corr()

        im = ax.imshow(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1)
        ax.set_xticks(range(len(corr_cols)))
        ax.set_yticks(range(len(corr_cols)))
        ax.set_xticklabels(['步数', '时长', 'MCTS', '策略熵'], rotation=45, ha='right')
        ax.set_yticklabels(['步数', '时长', 'MCTS', '策略熵'])

        # 添加数值标注
        for i in range(len(corr_cols)):
            for j in range(len(corr_cols)):
                text = ax.text(j, i, f'{corr_matrix.iloc[i, j]:.2f}',
                             ha='center', va='center', color='black', fontsize=10)

        ax.set_title('特征相关性矩阵')
        plt.colorbar(im, ax=ax)

        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"📊 相关性图表已保存: {save_path}")
        plt.close()

    except Exception as e:
        print(f"⚠️  绘制相关性图表失败: {e}")
        import traceback
        traceback.print_exc()


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
    history = load_training_history()

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
    
    # 显示评估结果
    if history and 'win_rate_vs_random' in history:
        print("\n" + "=" * 60)
        print("🏆 AI vs 随机玩家评估")
        print("=" * 60)
        win_rates = history['win_rate_vs_random']
        iters = history['iterations']
        has_eval = False
        for i, wr in zip(iters, win_rates):
            if wr is not None:
                print(f"迭代 {i}: 胜率 {wr:.1%}")
                has_eval = True
        if not has_eval:
            print("暂无评估数据")

    # 绘图
    print("\n" + "=" * 60)
    print("📊 生成图表")
    print("=" * 60)

    plot_training_loss(history)
    plot_training_progress(df, iterations, history)
    plot_performance_analysis(df)
    plot_data_distribution(df)
    plot_correlation_analysis(df)

    print("\n" + "=" * 60)
    print("✅ 分析完成！")
    print("=" * 60)
    print("\n生成的图表:")
    print("  1. logs/games/training_loss.png - 训练损失分析")
    print("  2. logs/games/analysis.png - 训练进度分析")
    print("  3. logs/games/performance.png - MCTS 性能分析")
    print("  4. logs/games/distribution.png - 数据分布分析")
    print("  5. logs/games/correlation.png - 相关性分析")


if __name__ == '__main__':
    main()
