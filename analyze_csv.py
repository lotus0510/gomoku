#!/usr/bin/env python3
"""分析 games_summary.csv 的训练数据"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False

# 读取CSV文件
df = pd.read_csv('logs/games/games_summary.csv')

print("=" * 80)
print("📊 五子棋训练数据分析报告")
print("=" * 80)

# 1. 基本统计
print("\n【1. 数据概览】")
print(f"总游戏场数: {len(df):,}")
print(f"训练迭代数: {df['iteration'].max()}")
print(f"每次迭代平均游戏数: {len(df) / df['iteration'].max():.1f}")
print(f"数据时间跨度: 第 {df['iteration'].min()} 到第 {df['iteration'].max()} 次迭代")

# 2. 按迭代分组分析
print("\n【2. 训练进度分析】")
iter_stats = df.groupby('iteration').agg({
    'num_moves': ['mean', 'std', 'min', 'max'],
    'game_duration_sec': 'mean',
    'avg_mcts_time': 'mean',
    'black_win': 'sum',
    'white_win': 'sum',
    'avg_policy_entropy': 'mean',
    'avg_policy_top1_prob': 'mean'
}).round(2)

# 计算胜率
iter_stats['black_win_rate'] = (df.groupby('iteration')['black_win'].sum() /
                                 df.groupby('iteration').size() * 100).round(1)
iter_stats['white_win_rate'] = (df.groupby('iteration')['white_win'].sum() /
                                 df.groupby('iteration').size() * 100).round(1)

# 3. 早期 vs 晚期对比
early_iters = df[df['iteration'] <= 5]
late_iters = df[df['iteration'] >= df['iteration'].max() - 5]

print("\n【3. 早期 vs 晚期对比】")
print(f"\n早期训练 (迭代 1-5, {len(early_iters)} 场游戏):")
print(f"  平均步数: {early_iters['num_moves'].mean():.1f} ± {early_iters['num_moves'].std():.1f}")
print(f"  平均策略熵: {early_iters['avg_policy_entropy'].mean():.3f}")
print(f"  平均Top1概率: {early_iters['avg_policy_top1_prob'].mean():.3f}")
print(f"  黑棋胜率: {early_iters['black_win'].mean() * 100:.1f}%")
print(f"  平均游戏时长: {early_iters['game_duration_sec'].mean():.1f} 秒")

print(f"\n晚期训练 (迭代 {df['iteration'].max()-5}-{df['iteration'].max()}, {len(late_iters)} 场游戏):")
print(f"  平均步数: {late_iters['num_moves'].mean():.1f} ± {late_iters['num_moves'].std():.1f}")
print(f"  平均策略熵: {late_iters['avg_policy_entropy'].mean():.3f}")
print(f"  平均Top1概率: {late_iters['avg_policy_top1_prob'].mean():.3f}")
print(f"  黑棋胜率: {late_iters['black_win'].mean() * 100:.1f}%")
print(f"  平均游戏时长: {late_iters['game_duration_sec'].mean():.1f} 秒")

# 计算改进
print(f"\n📈 训练改进:")
steps_improvement = ((late_iters['num_moves'].mean() - early_iters['num_moves'].mean()) /
                     early_iters['num_moves'].mean() * 100)
print(f"  步数变化: {steps_improvement:+.1f}%")

entropy_improvement = ((early_iters['avg_policy_entropy'].mean() - late_iters['avg_policy_entropy'].mean()) /
                       early_iters['avg_policy_entropy'].mean() * 100)
print(f"  策略熵降低: {entropy_improvement:.1f}% (越低越好，说明AI更有信心)")

confidence_improvement = ((late_iters['avg_policy_top1_prob'].mean() - early_iters['avg_policy_top1_prob'].mean()) /
                          early_iters['avg_policy_top1_prob'].mean() * 100)
print(f"  决策信心提升: {confidence_improvement:+.1f}%")

# 4. 胜负分析
print("\n【4. 胜负统计】")
total_black_wins = df['black_win'].sum()
total_white_wins = df['white_win'].sum()
total_draws = df['draw'].sum()
print(f"黑棋获胜: {total_black_wins} 场 ({total_black_wins/len(df)*100:.1f}%)")
print(f"白棋获胜: {total_white_wins} 场 ({total_white_wins/len(df)*100:.1f}%)")
print(f"平局: {total_draws} 场 ({total_draws/len(df)*100:.1f}%)")

# 5. 性能分析
print("\n【5. 性能指标】")
print(f"平均MCTS时间: {df['avg_mcts_time'].mean():.3f} 秒/步")
print(f"最快MCTS: {df['avg_mcts_time'].min():.3f} 秒/步")
print(f"最慢MCTS: {df['avg_mcts_time'].max():.3f} 秒/步")
print(f"平均游戏时长: {df['game_duration_sec'].mean():.1f} 秒")

# 6. 异常检测
print("\n【6. 异常游戏检测】")
very_short = df[df['num_moves'] < 15]
very_long = df[df['num_moves'] > 80]
print(f"超短游戏 (<15步): {len(very_short)} 场 ({len(very_short)/len(df)*100:.1f}%)")
print(f"超长游戏 (>80步): {len(very_long)} 场 ({len(very_long)/len(df)*100:.1f}%)")

# 7. 每次迭代的关键指标
print("\n【7. 各迭代关键指标】")
print("迭代 | 游戏数 | 平均步数 | 策略熵 | 黑胜率 | 白胜率")
print("-" * 60)
for iteration in sorted(df['iteration'].unique()):
    iter_df = df[df['iteration'] == iteration]
    avg_moves = iter_df['num_moves'].mean()
    avg_entropy = iter_df['avg_policy_entropy'].mean()
    black_wr = iter_df['black_win'].mean() * 100
    white_wr = iter_df['white_win'].mean() * 100
    print(f"{iteration:4d} | {len(iter_df):6d} | {avg_moves:8.1f} | {avg_entropy:6.3f} | {black_wr:5.1f}% | {white_wr:5.1f}%")

# 8. 生成可视化
print("\n【8. 生成可视化图表...】")
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle('五子棋训练数据分析', fontsize=16, fontweight='bold')

# 图1: 平均步数趋势
iter_avg_moves = df.groupby('iteration')['num_moves'].mean()
axes[0, 0].plot(iter_avg_moves.index, iter_avg_moves.values, marker='o', linewidth=2)
axes[0, 0].set_xlabel('迭代次数')
axes[0, 0].set_ylabel('平均步数')
axes[0, 0].set_title('平均步数趋势')
axes[0, 0].grid(True, alpha=0.3)

# 图2: 策略熵趋势
iter_entropy = df.groupby('iteration')['avg_policy_entropy'].mean()
axes[0, 1].plot(iter_entropy.index, iter_entropy.values, marker='o', color='orange', linewidth=2)
axes[0, 1].set_xlabel('迭代次数')
axes[0, 1].set_ylabel('平均策略熵')
axes[0, 1].set_title('策略熵趋势 (越低越好)')
axes[0, 1].grid(True, alpha=0.3)

# 图3: 黑白胜率
iter_black_wr = df.groupby('iteration')['black_win'].mean() * 100
iter_white_wr = df.groupby('iteration')['white_win'].mean() * 100
axes[0, 2].plot(iter_black_wr.index, iter_black_wr.values, marker='o', label='黑棋胜率', linewidth=2)
axes[0, 2].plot(iter_white_wr.index, iter_white_wr.values, marker='s', label='白棋胜率', linewidth=2)
axes[0, 2].set_xlabel('迭代次数')
axes[0, 2].set_ylabel('胜率 (%)')
axes[0, 2].set_title('黑白胜率对比')
axes[0, 2].legend()
axes[0, 2].grid(True, alpha=0.3)

# 图4: MCTS时间趋势
iter_mcts = df.groupby('iteration')['avg_mcts_time'].mean()
axes[1, 0].plot(iter_mcts.index, iter_mcts.values, marker='o', color='green', linewidth=2)
axes[1, 0].set_xlabel('迭代次数')
axes[1, 0].set_ylabel('平均MCTS时间 (秒)')
axes[1, 0].set_title('MCTS搜索时间趋势')
axes[1, 0].grid(True, alpha=0.3)

# 图5: Top1概率趋势
iter_top1 = df.groupby('iteration')['avg_policy_top1_prob'].mean()
axes[1, 1].plot(iter_top1.index, iter_top1.values, marker='o', color='red', linewidth=2)
axes[1, 1].set_xlabel('迭代次数')
axes[1, 1].set_ylabel('平均Top1概率')
axes[1, 1].set_title('决策信心趋势 (越高越好)')
axes[1, 1].grid(True, alpha=0.3)

# 图6: 步数分布直方图
axes[1, 2].hist(early_iters['num_moves'], bins=30, alpha=0.5, label=f'早期 (迭代1-5)', edgecolor='black')
axes[1, 2].hist(late_iters['num_moves'], bins=30, alpha=0.5, label=f'晚期 (迭代{df["iteration"].max()-5}-{df["iteration"].max()})', edgecolor='black')
axes[1, 2].set_xlabel('游戏步数')
axes[1, 2].set_ylabel('频数')
axes[1, 2].set_title('游戏步数分布对比')
axes[1, 2].legend()
axes[1, 2].grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('logs/games/training_analysis.png', dpi=300, bbox_inches='tight')
print("✅ 图表已保存至: logs/games/training_analysis.png")

print("\n" + "=" * 80)
print("✅ 分析完成！")
print("=" * 80)
