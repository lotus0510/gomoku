"""分析训练日志 CSV 数据"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

print("=" * 80)
print("五子棋 AI 训练日志分析")
print("=" * 80)

# 读取数据
csv_path = "logs/games/games_summary.csv"
print(f"\n读取数据: {csv_path}")

try:
    df = pd.read_csv(csv_path)
    print(f"✅ 成功读取 {len(df)} 条记录")
except FileNotFoundError:
    print(f"❌ 文件不存在: {csv_path}")
    exit(1)

# 数据概览
print("\n" + "=" * 80)
print("数据概览")
print("=" * 80)
print(f"\n数据形状: {df.shape}")
print(f"列名: {list(df.columns)}")
print(f"\n前5行:")
print(df.head())

print(f"\n数据类型:")
print(df.dtypes)

print(f"\n基本统计:")
print(df.describe())

# 训练进度
print("\n" + "=" * 80)
print("训练进度")
print("=" * 80)
min_iter = df['iteration'].min()
max_iter = df['iteration'].max()
total_iters = max_iter - min_iter + 1
games_per_iter = df.groupby('iteration').size().mean()

print(f"\n迭代范围: {min_iter} - {max_iter}")
print(f"总迭代次数: {total_iters}")
print(f"总游戏局数: {len(df)}")
print(f"平均每次迭代游戏数: {games_per_iter:.1f}")

# 胜负统计
print("\n" + "=" * 80)
print("胜负统计")
print("=" * 80)

winner_counts = df['winner'].value_counts()
total_games = len(df)

print(f"\n总游戏数: {total_games}")
print(f"黑方胜 (1): {winner_counts.get(1, 0)} ({winner_counts.get(1, 0)/total_games*100:.1f}%)")
print(f"白方胜 (2): {winner_counts.get(2, 0)} ({winner_counts.get(2, 0)/total_games*100:.1f}%)")
if 0 in winner_counts:
    print(f"平局 (0): {winner_counts.get(0, 0)} ({winner_counts.get(0, 0)/total_games*100:.1f}%)")

# 按迭代分析胜率变化
print("\n按迭代分析胜率变化:")
iter_stats = df.groupby('iteration').agg({
    'winner': lambda x: (x == 1).sum() / len(x),  # 黑方胜率
    'num_moves': 'mean',
    'game_duration_sec': 'mean',
    'avg_mcts_time': 'mean',
    'avg_policy_entropy': 'mean'
}).rename(columns={'winner': 'black_win_rate'})

print(f"\n前5次迭代:")
print(iter_stats.head())
print(f"\n后5次迭代:")
print(iter_stats.tail())

# 游戏时长分析
print("\n" + "=" * 80)
print("游戏时长分析")
print("=" * 80)

print(f"\n平均移动步数: {df['num_moves'].mean():.1f}")
print(f"移动步数范围: {df['num_moves'].min()} - {df['num_moves'].max()}")
print(f"平均游戏时长: {df['game_duration_sec'].mean():.1f} 秒")
print(f"游戏时长范围: {df['game_duration_sec'].min():.1f} - {df['game_duration_sec'].max():.1f} 秒")

# MCTS 性能分析
print("\n" + "=" * 80)
print("MCTS 性能分析")
print("=" * 80)

print(f"\n平均 MCTS 时间: {df['avg_mcts_time'].mean():.3f} 秒/步")
print(f"MCTS 时间范围: {df['avg_mcts_time'].min():.3f} - {df['avg_mcts_time'].max():.3f} 秒")

# 按迭代分析 MCTS 时间变化
print(f"\n各迭代平均 MCTS 时间:")
mcts_by_iter = df.groupby('iteration')['avg_mcts_time'].mean()
print(f"第 1 次迭代: {mcts_by_iter.iloc[0]:.3f} 秒")
print(f"第 {len(mcts_by_iter)} 次迭代: {mcts_by_iter.iloc[-1]:.3f} 秒")
print(f"变化: {(mcts_by_iter.iloc[-1] - mcts_by_iter.iloc[0]) / mcts_by_iter.iloc[0] * 100:+.1f}%")

# 策略熵分析
print("\n" + "=" * 80)
print("策略熵分析")
print("=" * 80)

print(f"\n平均策略熵: {df['avg_policy_entropy'].mean():.3f}")
print(f"策略熵范围: {df['avg_policy_entropy'].min():.3f} - {df['avg_policy_entropy'].max():.3f}")

print(f"\n策略熵变化（越低说明模型越确定）:")
entropy_by_iter = df.groupby('iteration')['avg_policy_entropy'].mean()
print(f"第 1 次迭代: {entropy_by_iter.iloc[0]:.3f}")
print(f"第 {len(entropy_by_iter)} 次迭代: {entropy_by_iter.iloc[-1]:.3f}")
print(f"变化: {(entropy_by_iter.iloc[-1] - entropy_by_iter.iloc[0]) / entropy_by_iter.iloc[0] * 100:+.1f}%")

# 创建可视化
print("\n" + "=" * 80)
print("生成可视化图表")
print("=" * 80)

output_dir = Path("logs/analysis")
output_dir.mkdir(parents=True, exist_ok=True)

# 设置中文字体（如果需要）
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

# 1. 胜率变化
fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# 1.1 黑方胜率
axes[0, 0].plot(iter_stats.index, iter_stats['black_win_rate'], marker='o', linewidth=2)
axes[0, 0].axhline(y=0.5, color='r', linestyle='--', label='50% (均衡)')
axes[0, 0].set_xlabel('Iteration')
axes[0, 0].set_ylabel('Black Win Rate')
axes[0, 0].set_title('Black Win Rate by Iteration')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# 1.2 平均移动步数
axes[0, 1].plot(iter_stats.index, iter_stats['num_moves'], marker='o', linewidth=2, color='green')
axes[0, 1].set_xlabel('Iteration')
axes[0, 1].set_ylabel('Average Moves')
axes[0, 1].set_title('Average Game Length by Iteration')
axes[0, 1].grid(True, alpha=0.3)

# 1.3 MCTS 时间
axes[1, 0].plot(iter_stats.index, iter_stats['avg_mcts_time'], marker='o', linewidth=2, color='orange')
axes[1, 0].set_xlabel('Iteration')
axes[1, 0].set_ylabel('MCTS Time (sec)')
axes[1, 0].set_title('Average MCTS Time by Iteration')
axes[1, 0].grid(True, alpha=0.3)

# 1.4 策略熵
axes[1, 1].plot(iter_stats.index, iter_stats['avg_policy_entropy'], marker='o', linewidth=2, color='purple')
axes[1, 1].set_xlabel('Iteration')
axes[1, 1].set_ylabel('Policy Entropy')
axes[1, 1].set_title('Average Policy Entropy by Iteration')
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plot_path = output_dir / "training_metrics.png"
plt.savefig(plot_path, dpi=150, bbox_inches='tight')
print(f"✅ 保存图表: {plot_path}")
plt.close()

# 2. 分布图
fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# 2.1 移动步数分布
axes[0, 0].hist(df['num_moves'], bins=30, edgecolor='black', alpha=0.7)
axes[0, 0].set_xlabel('Number of Moves')
axes[0, 0].set_ylabel('Frequency')
axes[0, 0].set_title('Distribution of Game Length')
axes[0, 0].axvline(df['num_moves'].mean(), color='r', linestyle='--', label=f'Mean: {df["num_moves"].mean():.1f}')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# 2.2 游戏时长分布
axes[0, 1].hist(df['game_duration_sec'], bins=30, edgecolor='black', alpha=0.7, color='green')
axes[0, 1].set_xlabel('Game Duration (sec)')
axes[0, 1].set_ylabel('Frequency')
axes[0, 1].set_title('Distribution of Game Duration')
axes[0, 1].axvline(df['game_duration_sec'].mean(), color='r', linestyle='--',
                   label=f'Mean: {df["game_duration_sec"].mean():.1f}s')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# 2.3 MCTS 时间分布
axes[1, 0].hist(df['avg_mcts_time'], bins=30, edgecolor='black', alpha=0.7, color='orange')
axes[1, 0].set_xlabel('MCTS Time (sec)')
axes[1, 0].set_ylabel('Frequency')
axes[1, 0].set_title('Distribution of MCTS Time')
axes[1, 0].axvline(df['avg_mcts_time'].mean(), color='r', linestyle='--',
                   label=f'Mean: {df["avg_mcts_time"].mean():.3f}s')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

# 2.4 策略熵分布
axes[1, 1].hist(df['avg_policy_entropy'], bins=30, edgecolor='black', alpha=0.7, color='purple')
axes[1, 1].set_xlabel('Policy Entropy')
axes[1, 1].set_ylabel('Frequency')
axes[1, 1].set_title('Distribution of Policy Entropy')
axes[1, 1].axvline(df['avg_policy_entropy'].mean(), color='r', linestyle='--',
                   label=f'Mean: {df["avg_policy_entropy"].mean():.3f}')
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plot_path = output_dir / "distributions.png"
plt.savefig(plot_path, dpi=150, bbox_inches='tight')
print(f"✅ 保存图表: {plot_path}")
plt.close()

# 3. 相关性分析
print("\n" + "=" * 80)
print("相关性分析")
print("=" * 80)

correlation_cols = ['num_moves', 'game_duration_sec', 'avg_mcts_time', 'avg_policy_entropy']
corr_matrix = df[correlation_cols].corr()

print("\n相关系数矩阵:")
print(corr_matrix)

# 热力图
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, fmt='.3f', cmap='coolwarm', center=0,
            square=True, linewidths=1, cbar_kws={"shrink": 0.8})
plt.title('Correlation Matrix of Training Metrics')
plt.tight_layout()
plot_path = output_dir / "correlation_matrix.png"
plt.savefig(plot_path, dpi=150, bbox_inches='tight')
print(f"✅ 保存图表: {plot_path}")
plt.close()

# 保存统计摘要
print("\n" + "=" * 80)
print("保存统计摘要")
print("=" * 80)

summary_path = output_dir / "training_summary.txt"
with open(summary_path, 'w', encoding='utf-8') as f:
    f.write("=" * 80 + "\n")
    f.write("五子棋 AI 训练日志分析摘要\n")
    f.write("=" * 80 + "\n\n")

    f.write(f"训练进度:\n")
    f.write(f"  迭代范围: {min_iter} - {max_iter}\n")
    f.write(f"  总迭代次数: {total_iters}\n")
    f.write(f"  总游戏局数: {len(df)}\n")
    f.write(f"  平均每次迭代游戏数: {games_per_iter:.1f}\n\n")

    f.write(f"胜负统计:\n")
    f.write(f"  黑方胜率: {winner_counts.get(1, 0)/total_games*100:.1f}%\n")
    f.write(f"  白方胜率: {winner_counts.get(2, 0)/total_games*100:.1f}%\n\n")

    f.write(f"游戏时长:\n")
    f.write(f"  平均移动步数: {df['num_moves'].mean():.1f}\n")
    f.write(f"  平均游戏时长: {df['game_duration_sec'].mean():.1f} 秒\n\n")

    f.write(f"MCTS 性能:\n")
    f.write(f"  平均 MCTS 时间: {df['avg_mcts_time'].mean():.3f} 秒/步\n")
    f.write(f"  第 1 次迭代: {mcts_by_iter.iloc[0]:.3f} 秒\n")
    f.write(f"  第 {len(mcts_by_iter)} 次迭代: {mcts_by_iter.iloc[-1]:.3f} 秒\n")
    f.write(f"  变化: {(mcts_by_iter.iloc[-1] - mcts_by_iter.iloc[0]) / mcts_by_iter.iloc[0] * 100:+.1f}%\n\n")

    f.write(f"策略熵:\n")
    f.write(f"  平均策略熵: {df['avg_policy_entropy'].mean():.3f}\n")
    f.write(f"  第 1 次迭代: {entropy_by_iter.iloc[0]:.3f}\n")
    f.write(f"  第 {len(entropy_by_iter)} 次迭代: {entropy_by_iter.iloc[-1]:.3f}\n")
    f.write(f"  变化: {(entropy_by_iter.iloc[-1] - entropy_by_iter.iloc[0]) / entropy_by_iter.iloc[0] * 100:+.1f}%\n")

print(f"✅ 保存摘要: {summary_path}")

print("\n" + "=" * 80)
print("分析完成！")
print("=" * 80)
print(f"\n输出目录: {output_dir.absolute()}")
print("生成的文件:")
print("  - training_metrics.png (训练指标趋势)")
print("  - distributions.png (数据分布)")
print("  - correlation_matrix.png (相关性矩阵)")
print("  - training_summary.txt (统计摘要)")
