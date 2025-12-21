"""游戏日志分析工具（增强版）- 整合所有分析功能"""

import os
import json
import pandas as pd
import numpy as np
from pathlib import Path

def load_all_data(games_dir='logs/games', checkpoint_dir='checkpoints'):
    """加载所有训练数据"""
    data = {}

    # 1. 游戏汇总 CSV
    games_csv = os.path.join(games_dir, 'games_summary.csv')
    if os.path.exists(games_csv):
        data['games_df'] = pd.read_csv(games_csv)
        print(f"✅ 加载游戏汇总: {len(data['games_df'])} 局")
    else:
        print(f"⚠️  未找到: {games_csv}")
        data['games_df'] = None

    # 2. 迭代汇总 JSONL
    iter_jsonl = os.path.join(games_dir, 'iteration_summary.jsonl')
    if os.path.exists(iter_jsonl):
        iterations = []
        with open(iter_jsonl, 'r', encoding='utf-8') as f:
            for line in f:
                iterations.append(json.loads(line))
        data['iterations'] = iterations
        print(f"✅ 加载迭代汇总: {len(iterations)} 次")
    else:
        print(f"⚠️  未找到: {iter_jsonl}")
        data['iterations'] = []

    # 3. 训练历史 JSON
    history_json = os.path.join(checkpoint_dir, 'training_history.json')
    if os.path.exists(history_json):
        with open(history_json, 'r') as f:
            data['history'] = json.load(f)
        print(f"✅ 加载训练历史")
    else:
        print(f"⚠️  未找到: {history_json}")
        data['history'] = {}

    return data


def analyze_all_metrics(df):
    """综合分析所有指标"""
    print("\n" + "=" * 80)
    print("📊 综合数据分析")
    print("=" * 80)

    if df is None or len(df) == 0:
        print("无数据")
        return None

    # 基本统计
    stats = {
        'total_games': len(df),
        'total_iterations': df['iteration'].nunique(),
        'iter_range': (df['iteration'].min(), df['iteration'].max()),

        # 游戏长度
        'avg_moves': df['num_moves'].mean(),
        'med_moves': df['num_moves'].median(),
        'min_moves': df['num_moves'].min(),
        'max_moves': df['num_moves'].max(),
        'std_moves': df['num_moves'].std(),

        # 胜负
        'black_wins': df['black_win'].sum(),
        'white_wins': df['white_win'].sum(),
        'draws': df['draw'].sum(),
        'black_win_rate': df['black_win'].sum() / len(df),
        'white_win_rate': df['white_win'].sum() / len(df),

        # 时间性能
        'avg_duration': df['game_duration_sec'].mean(),
        'avg_mcts_time': df['avg_mcts_time'].mean(),
        'min_mcts_time': df['avg_mcts_time'].min(),
        'max_mcts_time': df['avg_mcts_time'].max(),

        # 策略熵
        'avg_entropy': df['avg_policy_entropy'].mean(),
        'min_entropy': df['avg_policy_entropy'].min(),
        'max_entropy': df['avg_policy_entropy'].max(),
    }

    # 打印统计
    print(f"\n【训练进度】")
    print(f"  总游戏数: {stats['total_games']:,}")
    print(f"  迭代范围: {stats['iter_range'][0]} - {stats['iter_range'][1']}")
    print(f"  总迭代数: {stats['total_iterations']}")

    print(f"\n【游戏长度】")
    print(f"  平均: {stats['avg_moves']:.1f} 步")
    print(f"  中位数: {stats['med_moves']:.1f} 步")
    print(f"  范围: {stats['min_moves']} - {stats['max_moves']} 步")
    print(f"  标准差: {stats['std_moves']:.1f}")

    print(f"\n【胜负统计】")
    print(f"  黑方胜: {stats['black_wins']} ({stats['black_win_rate']*100:.1f}%)")
    print(f"  白方胜: {stats['white_wins']} ({stats['white_win_rate']*100:.1f}%)")
    print(f"  平局: {stats['draws']} ({stats['draws']/stats['total_games']*100:.1f}%)")

    # 先手优势检测
    win_diff = abs(stats['black_wins'] - stats['white_wins'])
    if win_diff > stats['total_games'] * 0.1:
        advantage = "黑方" if stats['black_wins'] > stats['white_wins'] else "白方"
        print(f"  ⚠️  {advantage}有明显优势 (差距 {win_diff} 局)")
    else:
        print(f"  ✅ 黑白平衡良好")

    print(f"\n【MCTS 性能】")
    print(f"  平均 MCTS 时间: {stats['avg_mcts_time']:.3f} 秒/步")
    print(f"  范围: {stats['min_mcts_time']:.3f} - {stats['max_mcts_time']:.3f} 秒")
    print(f"  平均游戏时长: {stats['avg_duration']:.1f} 秒")

    print(f"\n【策略熵】")
    print(f"  平均: {stats['avg_entropy']:.3f}")
    print(f"  范围: {stats['min_entropy']:.3f} - {stats['max_entropy']:.3f}")

    # 按迭代分析趋势
    by_iter = df.groupby('iteration').agg({
        'num_moves': 'mean',
        'black_win': 'sum',
        'avg_mcts_time': 'mean',
        'avg_policy_entropy': 'mean'
    })

    if len(by_iter) >= 5:
        print(f"\n【趋势分析】（最近5次 vs 最早5次）")
        early = by_iter.head(5)
        recent = by_iter.tail(5)

        # 游戏长度变化
        early_moves = early['num_moves'].mean()
        recent_moves = recent['num_moves'].mean()
        moves_change = (recent_moves - early_moves) / early_moves * 100
        print(f"  游戏长度: {early_moves:.1f} → {recent_moves:.1f} ({moves_change:+.1f}%)")
        if moves_change > 20:
            print(f"    ✅ AI 学会更复杂的策略")
        elif moves_change < -20:
            print(f"    ⚠️  AI 追求快速结束")

        # MCTS 时间变化
        early_mcts = early['avg_mcts_time'].mean()
        recent_mcts = recent['avg_mcts_time'].mean()
        mcts_change = (recent_mcts - early_mcts) / early_mcts * 100
        print(f"  MCTS 时间: {early_mcts:.3f} → {recent_mcts:.3f} ({mcts_change:+.1f}%)")

        # 策略熵变化
        early_entropy = early['avg_policy_entropy'].mean()
        recent_entropy = recent['avg_policy_entropy'].mean()
        entropy_change = (recent_entropy - early_entropy) / early_entropy * 100
        print(f"  策略熵: {early_entropy:.3f} → {recent_entropy:.3f} ({entropy_change:+.1f}%)")
        if entropy_change < -5:
            print(f"    ✅ AI 越来越确定（好）")
        elif entropy_change > 5:
            print(f"    ⚠️  AI 仍在探索")

    return stats


def plot_enhanced_analysis(df, iterations, history, output_dir='logs/analysis'):
    """生成增强版图表"""
    try:
        import matplotlib.pyplot as plt
        import matplotlib
        import seaborn as sns
        from matplotlib.font_manager import FontProperties

        matplotlib.use('Agg')

        # 中文字体设置
        plt.rcParams['axes.unicode_minus'] = False
        chinese_fonts = ['Microsoft JhengHei', 'Microsoft YaHei', 'SimHei', 'SimSun']
        for font in chinese_fonts:
            if font in [f.name for f in matplotlib.font_manager.fontManager.ttflist]:
                plt.rcParams['font.sans-serif'] = [font] + plt.rcParams['font.sans-serif']
                print(f"✅ 使用字体: {font}")
                break

        Path(output_dir).mkdir(parents=True, exist_ok=True)

        # === 图表 1: 训练指标趋势 (3x2) ===
        fig, axes = plt.subplots(3, 2, figsize=(16, 14))
        fig.suptitle('训练指标趋势分析', fontsize=16, fontweight='bold')

        by_iter = df.groupby('iteration').agg({
            'num_moves': ['mean', 'std'],
            'black_win': 'sum',
            'white_win': 'sum',
            'draw': 'sum',
            'avg_mcts_time': 'mean',
            'avg_policy_entropy': 'mean',
            'game_duration_sec': 'mean'
        })
        by_iter.columns = ['_'.join(col).strip('_') for col in by_iter.columns]
        games_per_iter = df.groupby('iteration').size()
        by_iter['black_win_rate'] = by_iter['black_win_sum'] / games_per_iter

        # 1.1 游戏长度趋势
        ax = axes[0, 0]
        ax.plot(by_iter.index, by_iter['num_moves_mean'], marker='o', linewidth=2, label='平均步数')
        ax.fill_between(by_iter.index,
                        by_iter['num_moves_mean'] - by_iter['num_moves_std'],
                        by_iter['num_moves_mean'] + by_iter['num_moves_std'],
                        alpha=0.2)
        ax.set_xlabel('迭代次数', fontsize=11)
        ax.set_ylabel('步数', fontsize=11)
        ax.set_title('游戏长度趋势', fontsize=12, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # 1.2 黑方胜率
        ax = axes[0, 1]
        ax.plot(by_iter.index, by_iter['black_win_rate'], marker='s', linewidth=2, color='darkblue')
        ax.axhline(y=0.5, color='r', linestyle='--', alpha=0.5, label='50% (平衡)')
        ax.set_xlabel('迭代次数', fontsize=11)
        ax.set_ylabel('黑方胜率', fontsize=11)
        ax.set_title('黑方胜率趋势', fontsize=12, fontweight='bold')
        ax.set_ylim(0, 1)
        ax.legend()
        ax.grid(True, alpha=0.3)

        # 1.3 MCTS 时间
        ax = axes[1, 0]
        ax.plot(by_iter.index, by_iter['avg_mcts_time_mean'], marker='^', linewidth=2, color='orange')
        ax.set_xlabel('迭代次数', fontsize=11)
        ax.set_ylabel('MCTS 时间 (秒)', fontsize=11)
        ax.set_title('MCTS 性能趋势', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)

        # 1.4 策略熵
        ax = axes[1, 1]
        ax.plot(by_iter.index, by_iter['avg_policy_entropy_mean'], marker='D', linewidth=2, color='purple')
        ax.set_xlabel('迭代次数', fontsize=11)
        ax.set_ylabel('策略熵', fontsize=11)
        ax.set_title('策略熵趋势 (越低越确定)', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)

        # 1.5 胜负堆叠图
        ax = axes[2, 0]
        win_data = by_iter[['black_win_sum', 'white_win_sum', 'draw_sum']]
        win_data.plot(kind='bar', stacked=True, ax=ax, color=['#2E86AB', '#A23B72', '#F18F01'])
        ax.set_xlabel('迭代次数', fontsize=11)
        ax.set_ylabel('游戏数', fontsize=11)
        ax.set_title('胜负分布', fontsize=12, fontweight='bold')
        ax.legend(['黑胜', '白胜', '平局'])
        ax.grid(True, alpha=0.3, axis='y')
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)

        # 1.6 AI vs 随机玩家
        ax = axes[2, 1]
        if history and 'win_rate_vs_random' in history:
            iters = history.get('iterations', [])
            win_rates = history.get('win_rate_vs_random', [])
            valid_data = [(i, w) for i, w in zip(iters, win_rates) if w is not None]

            if valid_data:
                valid_iters, valid_rates = zip(*valid_data)
                ax.plot(valid_iters, valid_rates, marker='*', color='red', markersize=12, linewidth=2)
                ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)

                # 标注百分比
                for i, r in zip(valid_iters, valid_rates):
                    ax.text(i, r + 0.02, f"{r:.0%}", ha='center', fontsize=9)

                ax.set_ylim(-0.05, 1.05)
                ax.set_xlabel('迭代次数', fontsize=11)
                ax.set_ylabel('胜率', fontsize=11)
                ax.set_title('AI vs 随机玩家胜率', fontsize=12, fontweight='bold')
                ax.grid(True, alpha=0.3)
            else:
                ax.text(0.5, 0.5, '无评估数据', ha='center', va='center', fontsize=14)
                ax.set_title('AI vs 随机玩家胜率', fontsize=12, fontweight='bold')
        else:
            ax.text(0.5, 0.5, '无评估数据', ha='center', va='center', fontsize=14)
            ax.set_title('AI vs 随机玩家胜率', fontsize=12, fontweight='bold')

        plt.tight_layout()
        plot_path = Path(output_dir) / 'training_trends.png'
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        print(f"✅ 保存图表: {plot_path}")
        plt.close()

        # === 图表 2: 数据分布 (2x2) ===
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('数据分布分析', fontsize=16, fontweight='bold')

        # 2.1 游戏长度分布
        ax = axes[0, 0]
        ax.hist(df['num_moves'], bins=30, edgecolor='black', alpha=0.7, color='steelblue')
        ax.axvline(df['num_moves'].mean(), color='r', linestyle='--', linewidth=2,
                   label=f'平均: {df["num_moves"].mean():.1f}')
        ax.set_xlabel('步数', fontsize=11)
        ax.set_ylabel('频数', fontsize=11)
        ax.set_title('游戏长度分布', fontsize=12, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')

        # 2.2 MCTS 时间分布
        ax = axes[0, 1]
        ax.hist(df['avg_mcts_time'], bins=30, edgecolor='black', alpha=0.7, color='orange')
        ax.axvline(df['avg_mcts_time'].mean(), color='r', linestyle='--', linewidth=2,
                   label=f'平均: {df["avg_mcts_time"].mean():.3f}s')
        ax.set_xlabel('MCTS 时间 (秒)', fontsize=11)
        ax.set_ylabel('频数', fontsize=11)
        ax.set_title('MCTS 时间分布', fontsize=12, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')

        # 2.3 策略熵分布
        ax = axes[1, 0]
        ax.hist(df['avg_policy_entropy'], bins=30, edgecolor='black', alpha=0.7, color='purple')
        ax.axvline(df['avg_policy_entropy'].mean(), color='r', linestyle='--', linewidth=2,
                   label=f'平均: {df["avg_policy_entropy"].mean():.3f}')
        ax.set_xlabel('策略熵', fontsize=11)
        ax.set_ylabel('频数', fontsize=11)
        ax.set_title('策略熵分布', fontsize=12, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')

        # 2.4 游戏时长分布
        ax = axes[1, 1]
        ax.hist(df['game_duration_sec'], bins=30, edgecolor='black', alpha=0.7, color='green')
        ax.axvline(df['game_duration_sec'].mean(), color='r', linestyle='--', linewidth=2,
                   label=f'平均: {df["game_duration_sec"].mean():.1f}s')
        ax.set_xlabel('游戏时长 (秒)', fontsize=11)
        ax.set_ylabel('频数', fontsize=11)
        ax.set_title('游戏时长分布', fontsize=12, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()
        plot_path = Path(output_dir) / 'distributions.png'
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        print(f"✅ 保存图表: {plot_path}")
        plt.close()

        # === 图表 3: 相关性热力图 ===
        fig, ax = plt.subplots(figsize=(10, 8))
        corr_cols = ['num_moves', 'game_duration_sec', 'avg_mcts_time', 'avg_policy_entropy']
        corr_matrix = df[corr_cols].corr()

        sns.heatmap(corr_matrix, annot=True, fmt='.3f', cmap='coolwarm', center=0,
                    square=True, linewidths=1, cbar_kws={"shrink": 0.8}, ax=ax)
        ax.set_title('训练指标相关性矩阵', fontsize=14, fontweight='bold', pad=15)

        plt.tight_layout()
        plot_path = Path(output_dir) / 'correlation_matrix.png'
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        print(f"✅ 保存图表: {plot_path}")
        plt.close()

        print(f"\n✅ 所有图表已保存到: {Path(output_dir).absolute()}")

    except ImportError as e:
        print(f"⚠️  缺少依赖: {e}")
        print("   请安装: pip install matplotlib seaborn")
    except Exception as e:
        print(f"⚠️  绘图失败: {e}")
        import traceback
        traceback.print_exc()


def save_summary_report(stats, output_path='logs/analysis/summary.txt'):
    """保存文本摘要报告"""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("五子棋 AI 训练分析报告\n")
        f.write("=" * 80 + "\n\n")

        f.write("【训练进度】\n")
        f.write(f"  总游戏数: {stats['total_games']:,}\n")
        f.write(f"  迭代范围: {stats['iter_range'][0]} - {stats['iter_range'][1]}\n")
        f.write(f"  总迭代数: {stats['total_iterations']}\n\n")

        f.write("【游戏长度】\n")
        f.write(f"  平均: {stats['avg_moves']:.1f} 步\n")
        f.write(f"  中位数: {stats['med_moves']:.1f} 步\n")
        f.write(f"  范围: {stats['min_moves']} - {stats['max_moves']} 步\n\n")

        f.write("【胜负统计】\n")
        f.write(f"  黑方胜率: {stats['black_win_rate']*100:.1f}%\n")
        f.write(f"  白方胜率: {stats['white_win_rate']*100:.1f}%\n")
        f.write(f"  平局率: {stats['draws']/stats['total_games']*100:.1f}%\n\n")

        f.write("【MCTS 性能】\n")
        f.write(f"  平均 MCTS 时间: {stats['avg_mcts_time']:.3f} 秒/步\n")
        f.write(f"  平均游戏时长: {stats['avg_duration']:.1f} 秒\n\n")

        f.write("【策略熵】\n")
        f.write(f"  平均: {stats['avg_entropy']:.3f}\n")
        f.write(f"  范围: {stats['min_entropy']:.3f} - {stats['max_entropy']:.3f}\n")

    print(f"✅ 保存摘要报告: {output_path}")


def main():
    """主函数"""
    print("=" * 80)
    print("🎮 五子棋训练分析工具（增强版）")
    print("=" * 80)

    # 加载所有数据
    print("\n📁 加载数据...")
    data = load_all_data()

    df = data.get('games_df')
    if df is None or len(df) == 0:
        print("\n❌ 没有找到游戏数据")
        print("请先运行训练: python train_pipeline_pytorch.py")
        return

    # 综合分析
    stats = analyze_all_metrics(df)

    # 显示最近游戏
    print("\n" + "=" * 80)
    print("🎮 最近 10 局游戏")
    print("=" * 80)
    recent = df.tail(10)[['iteration', 'game_num', 'num_moves', 'winner', 'avg_policy_entropy', 'avg_mcts_time']]
    print(recent.to_string(index=False))

    # 生成图表
    print("\n" + "=" * 80)
    print("📊 生成可视化图表")
    print("=" * 80)
    plot_enhanced_analysis(df, data.get('iterations', []), data.get('history', {}))

    # 保存摘要
    if stats:
        save_summary_report(stats)

    print("\n" + "=" * 80)
    print("✅ 分析完成！")
    print("=" * 80)


if __name__ == '__main__':
    main()
