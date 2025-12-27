#!/usr/bin/env python3
"""
綜合圖表分析工具 - 生成詳細的訓練與遊戲分析圖表

用途：
- 訓練指標分析（損失、梯度、學習率等）
- 遊戲統計分析（長度分佈、勝率、策略模式等）
- 性能趨勢分析（時間序列、相關性等）
- 策略演化分析（edge-rush 趨勢、防守模式等）
"""

import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import os
import warnings
warnings.filterwarnings('ignore')

# 設置 seaborn 風格（先設置，避免覆蓋字體）
sns.set_style("whitegrid")
sns.set_palette("husl")

# 設置中文字體（在 seaborn 之後設置，確保不被覆蓋）
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimSun', 'Microsoft JhengHei']
plt.rcParams['axes.unicode_minus'] = False

# 設置 legend 使用相同字體
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['legend.framealpha'] = 0.8


class ComprehensivePlotter:
    """綜合繪圖分析器"""

    def __init__(self, checkpoint_dir='checkpoints', log_dir='logs/games', output_dir='analysis/plots'):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.log_dir = Path(log_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # 載入數據
        self.load_data()

    def load_data(self):
        """載入訓練歷史和遊戲數據"""
        print("載入數據...")

        # 訓練歷史
        history_path = self.checkpoint_dir / 'training_history.json'
        if history_path.exists():
            with open(history_path, 'r') as f:
                self.history = json.load(f)
            print(f"  ✓ 訓練歷史: {len(self.history['iterations'])} 次迭代")
        else:
            print("  ✗ 找不到訓練歷史")
            self.history = None

        # 遊戲數據
        games_path = self.log_dir / 'games_summary.csv'
        if games_path.exists():
            self.games_df = pd.read_csv(games_path)
            print(f"  ✓ 遊戲數據: {len(self.games_df)} 局遊戲")
        else:
            print("  ✗ 找不到遊戲數據")
            self.games_df = None

        # 配置變更記錄
        config_changes_path = self.checkpoint_dir / 'config_changes.json'
        if config_changes_path.exists():
            with open(config_changes_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                self.config_changes = data.get('changes', [])
            print(f"  ✓ 配置變更: {len(self.config_changes)} 次")
        else:
            print("  ✗ 找不到配置變更記錄")
            self.config_changes = []

    def mark_config_changes(self, ax, iterations=None):
        """在圖表上標記配置變更點

        Args:
            ax: matplotlib axes 對象
            iterations: 迭代數列表（用於確定 y 軸範圍），如果為 None 則使用當前 ax 的範圍
        """
        if not self.config_changes:
            return

        # 獲取 y 軸範圍
        if iterations is not None:
            ymin, ymax = ax.get_ylim()
        else:
            ymin, ymax = ax.get_ylim()

        # 標記每次配置變更
        for change in self.config_changes:
            iter_num = change['iteration']
            change_type = change['change_type']
            change_count = change['change_count']

            # 根據變更類型選擇顏色
            if change_type == 'auto_reset':
                color = 'red'
                label = '自動重置'
            elif change_type == 'manual':
                color = 'orange'
                label = '手動調整'
            else:  # resume
                color = 'blue'
                label = '恢復訓練'

            # 畫垂直虛線
            ax.axvline(x=iter_num, color=color, linestyle='--', alpha=0.5, linewidth=1.5)

            # 添加註解（只顯示在圖表上方）
            y_pos = ymax - (ymax - ymin) * 0.05  # 靠近頂部
            ax.text(iter_num, y_pos, f'{label}\n({change_count}個參數)',
                   rotation=90, verticalalignment='top', horizontalalignment='right',
                   fontsize=8, color=color, alpha=0.7,
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7, edgecolor=color))

    def plot_all(self):
        """生成所有圖表"""
        print("\n生成圖表...")

        if self.history:
            self.plot_training_metrics()
            self.plot_loss_breakdown()
            self.plot_gradient_analysis()
            self.plot_value_network_metrics()

        if self.games_df is not None:
            self.plot_game_length_distribution()
            self.plot_win_rate_trends()
            self.plot_strategy_evolution()
            self.plot_game_patterns()

        if self.history and self.games_df is not None:
            self.plot_comprehensive_dashboard()
            self.plot_correlation_matrix()

        print(f"\n✓ 所有圖表已保存到: {self.output_dir}")

    def plot_training_metrics(self):
        """繪製訓練指標總覽"""
        print("  生成: 訓練指標總覽...")

        fig, axes = plt.subplots(2, 2, figsize=(16, 10))
        fig.suptitle('訓練指標總覽', fontsize=16, fontweight='bold')

        iterations = self.history['iterations']

        # 1. 總損失
        ax = axes[0, 0]
        ax.plot(iterations, self.history['total_loss'], 'b-', linewidth=2, label='總損失')
        ax.fill_between(iterations, self.history['total_loss'], alpha=0.3)
        ax.set_xlabel('迭代次數')
        ax.set_ylabel('損失值')
        ax.set_title('總損失變化')
        ax.grid(True, alpha=0.3)
        ax.legend()

        # 2. 梯度範數
        ax = axes[0, 1]
        grad_norms = self.history['gradient_norm']
        ax.plot(iterations, grad_norms, 'g-', linewidth=2, label='梯度範數')
        ax.axhline(y=0.5, color='r', linestyle='--', alpha=0.5, label='健康下限')
        ax.axhline(y=5.0, color='r', linestyle='--', alpha=0.5, label='健康上限')
        ax.fill_between(iterations, 0.5, 5.0, color='green', alpha=0.1)
        ax.set_xlabel('迭代次數')
        ax.set_ylabel('梯度範數')
        ax.set_title('梯度範數變化（健康範圍: 0.5-5.0）')
        ax.grid(True, alpha=0.3)
        ax.legend()
        ax.set_yscale('log')

        # 3. 學習率
        ax = axes[1, 0]
        lr = self.history['learning_rate']
        ax.plot(iterations, lr, 'orange', linewidth=2, marker='o', markersize=4)
        ax.set_xlabel('迭代次數')
        ax.set_ylabel('學習率')
        ax.set_title('學習率調度')
        ax.grid(True, alpha=0.3)
        ax.set_yscale('log')

        # 4. 勝率 vs 隨機玩家
        ax = axes[1, 1]
        win_rates = self.history['win_rate_vs_random']
        # 只繪製非 null 的數據點
        valid_iters = [iterations[i] for i in range(len(win_rates)) if win_rates[i] is not None]
        valid_rates = [win_rates[i] for i in range(len(win_rates)) if win_rates[i] is not None]

        if valid_rates:
            ax.plot(valid_iters, valid_rates, 'ro-', linewidth=2, markersize=8, label='勝率')
            ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='50% 基準')
            ax.set_xlabel('迭代次數')
            ax.set_ylabel('勝率')
            ax.set_title('對隨機玩家勝率')
            ax.set_ylim([0, 1])
            ax.grid(True, alpha=0.3)
            ax.legend()
        else:
            ax.text(0.5, 0.5, '無評估數據', ha='center', va='center',
                   transform=ax.transAxes, fontsize=14)
            ax.set_title('對隨機玩家勝率（無數據）')

        # 在所有子圖上標記配置變更
        for ax in axes.flat:
            self.mark_config_changes(ax)

        plt.tight_layout()
        plt.savefig(self.output_dir / '1_training_metrics.png', dpi=300, bbox_inches='tight')
        plt.close()

    def plot_loss_breakdown(self):
        """繪製損失分解分析"""
        print("  生成: 損失分解分析...")

        fig, axes = plt.subplots(2, 2, figsize=(16, 10))
        fig.suptitle('損失分解分析', fontsize=16, fontweight='bold')

        iterations = self.history['iterations']

        # 1. 三種損失對比
        ax = axes[0, 0]
        ax.plot(iterations, self.history['total_loss'], 'b-', linewidth=2, label='總損失', alpha=0.8)
        ax.plot(iterations, self.history['policy_loss'], 'g-', linewidth=2, label='策略損失', alpha=0.8)
        ax.plot(iterations, self.history['value_loss'], 'r-', linewidth=2, label='價值損失', alpha=0.8)
        ax.set_xlabel('迭代次數')
        ax.set_ylabel('損失值')
        ax.set_title('損失組成對比')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # 2. 策略損失詳細
        ax = axes[0, 1]
        policy_loss = self.history['policy_loss']
        ax.plot(iterations, policy_loss, 'g-', linewidth=2)

        # 計算趨勢線
        z = np.polyfit(iterations, policy_loss, 2)
        p = np.poly1d(z)
        ax.plot(iterations, p(iterations), 'r--', linewidth=2, alpha=0.5, label='趨勢')

        ax.set_xlabel('迭代次數')
        ax.set_ylabel('策略損失')
        ax.set_title('策略損失變化（交叉熵）')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # 3. 價值損失詳細
        ax = axes[1, 0]
        value_loss = self.history['value_loss']
        ax.plot(iterations, value_loss, 'r-', linewidth=2)

        # 計算趨勢線
        z = np.polyfit(iterations, value_loss, 2)
        p = np.poly1d(z)
        ax.plot(iterations, p(iterations), 'b--', linewidth=2, alpha=0.5, label='趨勢')

        ax.set_xlabel('迭代次數')
        ax.set_ylabel('價值損失')
        ax.set_title('價值損失變化（MSE）')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # 4. 損失標準差（穩定性）
        ax = axes[1, 1]
        if 'loss_std' in self.history:
            loss_std = self.history['loss_std']
            ax.plot(iterations, loss_std, 'purple', linewidth=2, marker='o', markersize=4)
            ax.set_xlabel('迭代次數')
            ax.set_ylabel('損失標準差')
            ax.set_title('訓練穩定性（損失標準差）')
            ax.grid(True, alpha=0.3)
        else:
            ax.text(0.5, 0.5, '無標準差數據', ha='center', va='center',
                   transform=ax.transAxes, fontsize=14)

        # 標記配置變更
        for ax in axes.flat:
            self.mark_config_changes(ax)

        plt.tight_layout()
        plt.savefig(self.output_dir / '2_loss_breakdown.png', dpi=300, bbox_inches='tight')
        plt.close()

    def plot_gradient_analysis(self):
        """繪製梯度分析"""
        print("  生成: 梯度分析...")

        fig, axes = plt.subplots(2, 2, figsize=(16, 10))
        fig.suptitle('梯度健康度分析', fontsize=16, fontweight='bold')

        iterations = self.history['iterations']
        grad_norms = self.history['gradient_norm']

        # 過濾無效值（inf, NaN）
        grad_norms_array = np.array(grad_norms)
        valid_mask = np.isfinite(grad_norms_array)
        grad_norms_valid = grad_norms_array[valid_mask]
        iterations_valid = np.array(iterations)[valid_mask]

        # 1. 梯度範數時間序列
        ax = axes[0, 0]
        ax.plot(iterations_valid, grad_norms_valid, 'b-', linewidth=2, marker='o', markersize=4)
        ax.axhline(y=0.5, color='r', linestyle='--', alpha=0.5, label='下限 0.5')
        ax.axhline(y=5.0, color='r', linestyle='--', alpha=0.5, label='上限 5.0')
        ax.fill_between(iterations_valid, 0.5, 5.0, color='green', alpha=0.1, label='健康範圍')
        ax.set_xlabel('迭代次數')
        ax.set_ylabel('梯度範數')
        ax.set_title('梯度範數趨勢')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # 2. 梯度範數分佈
        ax = axes[0, 1]
        if len(grad_norms_valid) > 0:
            ax.hist(grad_norms_valid, bins=30, color='blue', alpha=0.7, edgecolor='black')
            ax.axvline(x=0.5, color='r', linestyle='--', linewidth=2, label='下限')
            ax.axvline(x=5.0, color='r', linestyle='--', linewidth=2, label='上限')
            ax.axvline(x=np.mean(grad_norms_valid), color='g', linestyle='-', linewidth=2,
                      label=f'平均 {np.mean(grad_norms_valid):.2f}')
        else:
            ax.text(0.5, 0.5, '無有效梯度數據', ha='center', va='center', transform=ax.transAxes)
        ax.set_xlabel('梯度範數')
        ax.set_ylabel('頻率')
        ax.set_title('梯度範數分佈')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # 3. 梯度穩定性（標準差）
        ax = axes[1, 0]
        if 'gradient_norm_std' in self.history:
            grad_std = self.history['gradient_norm_std']
            ax.plot(iterations, grad_std, 'orange', linewidth=2, marker='s', markersize=4)
            ax.set_xlabel('迭代次數')
            ax.set_ylabel('梯度標準差')
            ax.set_title('梯度穩定性（批次內標準差）')
            ax.grid(True, alpha=0.3)
        else:
            ax.text(0.5, 0.5, '無標準差數據', ha='center', va='center',
                   transform=ax.transAxes, fontsize=14)

        # 4. 梯度健康度評分
        ax = axes[1, 1]
        # 計算健康度：在 0.5-5.0 範圍內的百分比（使用有效值）
        if len(grad_norms_valid) > 0:
            healthy_count = sum(1 for g in grad_norms_valid if 0.5 <= g <= 5.0)
            healthy_pct = healthy_count / len(grad_norms_valid) * 100

            categories = ['健康\n(0.5-5.0)', '過小\n(<0.5)', '過大\n(>5.0)']
            counts = [
                healthy_count,
                sum(1 for g in grad_norms_valid if g < 0.5),
                sum(1 for g in grad_norms_valid if g > 5.0)
            ]
            colors = ['green', 'orange', 'red']

            ax.bar(categories, counts, color=colors, alpha=0.7, edgecolor='black')
            ax.set_ylabel('迭代次數')
            ax.set_title(f'梯度健康度統計（健康度: {healthy_pct:.1f}%）')

            # 添加數值標籤
            for i, (cat, count) in enumerate(zip(categories, counts)):
                ax.text(i, count, str(count), ha='center', va='bottom', fontsize=12, fontweight='bold')

            ax.grid(True, alpha=0.3, axis='y')
        else:
            ax.text(0.5, 0.5, '無有效梯度數據', ha='center', va='center', transform=ax.transAxes)

        # 標記配置變更
        for ax in axes.flat:
            self.mark_config_changes(ax)

        plt.tight_layout()
        plt.savefig(self.output_dir / '3_gradient_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()

    def plot_value_network_metrics(self):
        """繪製價值網路指標"""
        print("  生成: 價值網路分析...")

        fig, axes = plt.subplots(2, 2, figsize=(16, 10))
        fig.suptitle('價值網路性能分析', fontsize=16, fontweight='bold')

        iterations = self.history['iterations']

        # 1. 價值 MAE
        ax = axes[0, 0]
        value_mae = self.history['value_mae']
        ax.plot(iterations, value_mae, 'r-', linewidth=2, marker='o', markersize=4)

        # 趨勢線
        z = np.polyfit(iterations, value_mae, 2)
        p = np.poly1d(z)
        ax.plot(iterations, p(iterations), 'b--', linewidth=2, alpha=0.5, label='趨勢')

        ax.set_xlabel('迭代次數')
        ax.set_ylabel('MAE')
        ax.set_title('價值網路 MAE（越低越好）')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # 2. 價值標準差 + 零值比例（新增！）
        ax = axes[0, 1]
        if 'value_std' in self.history:
            value_std = self.history['value_std']
            ax.plot(iterations, value_std, 'purple', linewidth=2, marker='s', markersize=4, label='價值標準差')

            # 添加健康範圍標記
            ax.axhspan(0.3, 1.0, alpha=0.1, color='green', label='健康範圍 (>0.3)')
            ax.axhspan(0.0, 0.3, alpha=0.1, color='red')

            ax.set_xlabel('迭代次數')
            ax.set_ylabel('標準差')
            ax.set_title('價值預測標準差（<0.3 表示失去判別力）')
            ax.legend()
            ax.grid(True, alpha=0.3)

        # 3. 策略 Top-1 機率
        ax = axes[1, 0]
        if 'policy_top1_prob' in self.history:
            top1_prob = self.history['policy_top1_prob']
            ax.plot(iterations, top1_prob, 'g-', linewidth=2, marker='o', markersize=4)
            ax.set_xlabel('迭代次數')
            ax.set_ylabel('Top-1 機率')
            ax.set_title('策略集中度（Top-1 機率，越高越自信）')
            ax.grid(True, alpha=0.3)

        # 4. 策略熵
        ax = axes[1, 1]
        if 'policy_entropy_train' in self.history:
            entropy = self.history['policy_entropy_train']
            ax.plot(iterations, entropy, 'orange', linewidth=2, marker='s', markersize=4)
            ax.set_xlabel('迭代次數')
            ax.set_ylabel('熵值')
            ax.set_title('策略熵（越低越確定）')
            ax.grid(True, alpha=0.3)

        # 標記配置變更（這是最重要的圖表之一）
        for ax in axes.flat:
            self.mark_config_changes(ax)

        plt.tight_layout()
        plt.savefig(self.output_dir / '4_value_network_metrics.png', dpi=300, bbox_inches='tight')
        plt.close()

    def plot_game_length_distribution(self):
        """繪製遊戲長度分佈"""
        print("  生成: 遊戲長度分佈...")

        fig, axes = plt.subplots(2, 2, figsize=(16, 10))
        fig.suptitle('遊戲長度分析', fontsize=16, fontweight='bold')

        # 1. 整體長度分佈
        ax = axes[0, 0]
        game_lengths = self.games_df['num_moves']
        ax.hist(game_lengths, bins=50, color='blue', alpha=0.7, edgecolor='black')
        ax.axvline(x=game_lengths.mean(), color='r', linestyle='--', linewidth=2,
                  label=f'平均: {game_lengths.mean():.1f}')
        ax.axvline(x=game_lengths.median(), color='g', linestyle='--', linewidth=2,
                  label=f'中位數: {game_lengths.median():.1f}')
        ax.set_xlabel('遊戲步數')
        ax.set_ylabel('頻率')
        ax.set_title('遊戲長度分佈')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # 2. 按迭代的長度變化
        ax = axes[0, 1]
        iter_avg_length = self.games_df.groupby('iteration')['num_moves'].mean()
        ax.plot(iter_avg_length.index, iter_avg_length.values, 'b-',
               linewidth=2, marker='o', markersize=4)

        # 趨勢線
        x = iter_avg_length.index
        y = iter_avg_length.values
        z = np.polyfit(x, y, 2)
        p = np.poly1d(z)
        ax.plot(x, p(x), 'r--', linewidth=2, alpha=0.5, label='趨勢')

        ax.set_xlabel('迭代次數')
        ax.set_ylabel('平均步數')
        ax.set_title('平均遊戲長度演化')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # 3. 長度分類統計
        ax = axes[1, 0]
        ultra_short = len(self.games_df[self.games_df['num_moves'] <= 12])
        short = len(self.games_df[(self.games_df['num_moves'] > 12) &
                                  (self.games_df['num_moves'] <= 20)])
        medium = len(self.games_df[(self.games_df['num_moves'] > 20) &
                                   (self.games_df['num_moves'] <= 40)])
        long = len(self.games_df[self.games_df['num_moves'] > 40])

        categories = ['超短局\n(≤12步)', '短局\n(13-20步)', '中局\n(21-40步)', '長局\n(>40步)']
        counts = [ultra_short, short, medium, long]
        colors = ['red', 'orange', 'green', 'blue']

        bars = ax.bar(categories, counts, color=colors, alpha=0.7, edgecolor='black')
        ax.set_ylabel('遊戲數量')
        ax.set_title('遊戲長度分類統計')

        # 添加百分比標籤
        total = len(self.games_df)
        for bar, count in zip(bars, counts):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{count}\n({count/total*100:.1f}%)',
                   ha='center', va='bottom', fontsize=10, fontweight='bold')

        ax.grid(True, alpha=0.3, axis='y')

        # 4. 箱型圖（按迭代分組，最近10次）
        ax = axes[1, 1]
        recent_iters = sorted(self.games_df['iteration'].unique())[-10:]
        recent_data = self.games_df[self.games_df['iteration'].isin(recent_iters)]

        boxplot_data = [recent_data[recent_data['iteration'] == it]['num_moves'].values
                       for it in recent_iters]

        bp = ax.boxplot(boxplot_data, labels=[str(it) for it in recent_iters],
                       patch_artist=True)

        # 美化箱型圖
        for patch in bp['boxes']:
            patch.set_facecolor('lightblue')
            patch.set_alpha(0.7)

        ax.set_xlabel('迭代次數')
        ax.set_ylabel('遊戲步數')
        ax.set_title('最近 10 次迭代的長度分佈（箱型圖）')
        ax.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()
        plt.savefig(self.output_dir / '5_game_length_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()

    def plot_win_rate_trends(self):
        """繪製勝率趨勢"""
        print("  生成: 勝率趨勢分析...")

        fig, axes = plt.subplots(2, 2, figsize=(16, 10))
        fig.suptitle('勝率趨勢分析', fontsize=16, fontweight='bold')

        # 計算每次迭代的勝率
        iter_stats = self.games_df.groupby('iteration').agg({
            'black_win': 'sum',
            'white_win': 'sum',
            'draw': 'sum',
            'num_moves': 'count'
        }).reset_index()

        iter_stats['black_rate'] = iter_stats['black_win'] / iter_stats['num_moves']
        iter_stats['white_rate'] = iter_stats['white_win'] / iter_stats['num_moves']
        iter_stats['draw_rate'] = iter_stats['draw'] / iter_stats['num_moves']

        # 1. 黑白勝率對比
        ax = axes[0, 0]
        ax.plot(iter_stats['iteration'], iter_stats['black_rate'],
               'ko-', linewidth=2, markersize=6, label='黑棋勝率')
        ax.plot(iter_stats['iteration'], iter_stats['white_rate'],
               'wo-', linewidth=2, markersize=6, markeredgecolor='black', label='白棋勝率')
        ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='50% 基準')
        ax.set_xlabel('迭代次數')
        ax.set_ylabel('勝率')
        ax.set_title('黑白勝率演化')
        ax.set_ylim([0, 1])
        ax.legend()
        ax.grid(True, alpha=0.3)

        # 2. 勝率堆疊圖
        ax = axes[0, 1]
        ax.fill_between(iter_stats['iteration'], 0, iter_stats['black_rate'],
                       color='black', alpha=0.7, label='黑勝')
        ax.fill_between(iter_stats['iteration'], iter_stats['black_rate'],
                       iter_stats['black_rate'] + iter_stats['white_rate'],
                       color='white', edgecolor='black', alpha=0.7, label='白勝')
        ax.fill_between(iter_stats['iteration'],
                       iter_stats['black_rate'] + iter_stats['white_rate'],
                       1.0, color='gray', alpha=0.5, label='平局')
        ax.set_xlabel('迭代次數')
        ax.set_ylabel('比例')
        ax.set_title('勝負比例堆疊圖')
        ax.set_ylim([0, 1])
        ax.legend()
        ax.grid(True, alpha=0.3)

        # 3. 勝率不平衡度
        ax = axes[1, 0]
        imbalance = abs(iter_stats['black_rate'] - 0.5)
        ax.plot(iter_stats['iteration'], imbalance, 'r-',
               linewidth=2, marker='o', markersize=4)
        ax.fill_between(iter_stats['iteration'], 0, imbalance, color='red', alpha=0.3)
        ax.set_xlabel('迭代次數')
        ax.set_ylabel('不平衡度')
        ax.set_title('黑白勝率不平衡度（|黑勝率 - 0.5|）')
        ax.grid(True, alpha=0.3)

        # 4. 勝率振盪分析（新增！）
        ax = axes[1, 1]

        # 計算滾動窗口的標準差和翻轉次數
        window_size = 5
        if len(iter_stats) >= window_size:
            rolling_std = []
            flip_counts = []

            for i in range(window_size - 1, len(iter_stats)):
                window_rates = iter_stats['black_rate'].iloc[i-window_size+1:i+1].values
                rolling_std.append(np.std(window_rates))

                # 計算翻轉次數
                flips = 0
                for j in range(1, len(window_rates)):
                    if (window_rates[j-1] > 0.5) != (window_rates[j] > 0.5):
                        flips += 1
                flip_counts.append(flips)

            rolling_iters = iter_stats['iteration'].iloc[window_size-1:].values

            # 雙Y軸
            ax2 = ax.twinx()

            line1 = ax.plot(rolling_iters, rolling_std, 'r-', linewidth=2,
                           marker='o', markersize=4, label='振盪標準差')
            ax2_line = ax2.plot(rolling_iters, flip_counts, 'b--', linewidth=2,
                               marker='s', markersize=4, label='翻轉次數')

            # 健康範圍標記
            ax.axhspan(0, 0.1, alpha=0.1, color='green')
            ax.axhspan(0.2, 1.0, alpha=0.1, color='red')

            ax.set_xlabel('迭代次數')
            ax.set_ylabel('勝率標準差（5次滾動）', color='r')
            ax2.set_ylabel('優勢翻轉次數', color='b')
            ax.set_title(f'勝率振盪分析（窗口={window_size}）')
            ax.tick_params(axis='y', labelcolor='r')
            ax2.tick_params(axis='y', labelcolor='b')

            # 合併圖例
            lines = line1 + ax2_line
            labels = [l.get_label() for l in lines]
            ax.legend(lines, labels, loc='upper left')
            ax.grid(True, alpha=0.3)
        else:
            ax.text(0.5, 0.5, f'需要至少 {window_size} 次迭代',
                   ha='center', va='center', transform=ax.transAxes, fontsize=12)

        plt.tight_layout()
        plt.savefig(self.output_dir / '6_win_rate_trends.png', dpi=300, bbox_inches='tight')
        plt.close()

    def plot_strategy_evolution(self):
        """繪製策略演化分析"""
        print("  生成: 策略演化分析...")

        fig, axes = plt.subplots(2, 2, figsize=(16, 10))
        fig.suptitle('策略演化分析（Edge-Rush 問題）', fontsize=16, fontweight='bold')

        # 計算每次迭代的統計
        iter_stats = []
        for iteration in sorted(self.games_df['iteration'].unique()):
            iter_df = self.games_df[self.games_df['iteration'] == iteration]

            stats = {
                'iteration': iteration,
                'ultra_short_pct': len(iter_df[iter_df['num_moves'] <= 12]) / len(iter_df) * 100,
                'short_pct': len(iter_df[iter_df['num_moves'] <= 20]) / len(iter_df) * 100,
                'avg_length': iter_df['num_moves'].mean(),
                'black_win_rate': iter_df['black_win'].mean(),
            }
            iter_stats.append(stats)

        stats_df = pd.DataFrame(iter_stats)

        # 1. 超短局比例趨勢
        ax = axes[0, 0]
        ax.plot(stats_df['iteration'], stats_df['ultra_short_pct'],
               'r-', linewidth=2, marker='o', markersize=6)
        ax.fill_between(stats_df['iteration'], 0, stats_df['ultra_short_pct'],
                       color='red', alpha=0.3)
        ax.axhline(y=40, color='orange', linestyle='--', linewidth=2,
                  alpha=0.5, label='警戒線 40%')
        ax.set_xlabel('迭代次數')
        ax.set_ylabel('超短局比例 (%)')
        ax.set_title('超短局（≤12步）比例演化')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # 2. 短局比例趨勢
        ax = axes[0, 1]
        ax.plot(stats_df['iteration'], stats_df['short_pct'],
               'orange', linewidth=2, marker='s', markersize=6)
        ax.fill_between(stats_df['iteration'], 0, stats_df['short_pct'],
                       color='orange', alpha=0.3)
        ax.set_xlabel('迭代次數')
        ax.set_ylabel('短局比例 (%)')
        ax.set_title('短局（≤20步）比例演化')
        ax.grid(True, alpha=0.3)

        # 3. 策略演化階段圖
        ax = axes[1, 0]

        # 定義階段
        ax.axvspan(0, 20, alpha=0.2, color='red', label='階段1: 簡單策略')
        ax.axvspan(20, 50, alpha=0.2, color='orange', label='階段2: 學習防守')
        ax.axvspan(50, 200, alpha=0.2, color='green', label='階段3: 複雜戰術')

        ax.plot(stats_df['iteration'], stats_df['avg_length'],
               'b-', linewidth=3, marker='o', markersize=6, label='平均遊戲長度')

        ax.set_xlabel('迭代次數')
        ax.set_ylabel('平均步數')
        ax.set_title('策略演化階段與遊戲長度')
        ax.legend(loc='upper left')
        ax.grid(True, alpha=0.3)

        # 4. Edge-Rush 健康度評分
        ax = axes[1, 1]

        # 計算健康度分數（超短局越少越好）
        health_score = 100 - stats_df['ultra_short_pct']

        colors = ['red' if s < 50 else 'orange' if s < 70 else 'green'
                 for s in health_score]

        bars = ax.bar(stats_df['iteration'], health_score, color=colors,
                     alpha=0.7, edgecolor='black')
        ax.axhline(y=50, color='red', linestyle='--', linewidth=2,
                  alpha=0.5, label='不健康 (<50)')
        ax.axhline(y=70, color='orange', linestyle='--', linewidth=2,
                  alpha=0.5, label='尚可 (<70)')
        ax.set_xlabel('迭代次數')
        ax.set_ylabel('健康度分數')
        ax.set_title('策略健康度（100 - 超短局%，越高越好）')
        ax.set_ylim([0, 100])
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()
        plt.savefig(self.output_dir / '7_strategy_evolution.png', dpi=300, bbox_inches='tight')
        plt.close()

    def plot_game_patterns(self):
        """繪製遊戲模式分析"""
        print("  生成: 遊戲模式分析...")

        fig, axes = plt.subplots(2, 2, figsize=(16, 10))
        fig.suptitle('遊戲模式深度分析', fontsize=16, fontweight='bold')

        # 1. 長度 vs 勝率散點圖
        ax = axes[0, 0]

        black_wins = self.games_df[self.games_df['black_win'] == 1]
        white_wins = self.games_df[self.games_df['white_win'] == 1]

        ax.scatter(black_wins['num_moves'], black_wins['iteration'],
                  c='black', alpha=0.5, s=20, label='黑勝')
        ax.scatter(white_wins['num_moves'], white_wins['iteration'],
                  c='white', edgecolors='black', alpha=0.5, s=20, label='白勝')

        ax.set_xlabel('遊戲步數')
        ax.set_ylabel('迭代次數')
        ax.set_title('遊戲長度與迭代分佈（黑白勝負）')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # 2. 按長度分類的勝率
        ax = axes[1, 0]

        # 定義長度區間
        bins = [0, 12, 20, 40, 100]
        labels = ['≤12步', '13-20步', '21-40步', '>40步']

        self.games_df['length_category'] = pd.cut(self.games_df['num_moves'],
                                                  bins=bins, labels=labels)

        category_stats = self.games_df.groupby('length_category').agg({
            'black_win': 'mean',
            'white_win': 'mean',
            'num_moves': 'count'
        }).reset_index()

        x = np.arange(len(category_stats))
        width = 0.35

        bars1 = ax.bar(x - width/2, category_stats['black_win'], width,
                      label='黑勝率', color='black', alpha=0.7)
        bars2 = ax.bar(x + width/2, category_stats['white_win'], width,
                      label='白勝率', color='white', edgecolor='black', alpha=0.7)

        ax.set_xlabel('遊戲長度分類')
        ax.set_ylabel('勝率')
        ax.set_title('不同長度分類下的黑白勝率')
        ax.set_xticks(x)
        ax.set_xticklabels(category_stats['length_category'])
        ax.set_ylim([0, 1])
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')

        # 添加遊戲數量標註
        for i, count in enumerate(category_stats['num_moves']):
            ax.text(i, 0.95, f'n={count}', ha='center', va='top',
                   fontsize=9, color='gray')

        # 3. 最近迭代的策略模式熱圖
        ax = axes[0, 1]

        recent_iters = sorted(self.games_df['iteration'].unique())[-10:]
        recent_df = self.games_df[self.games_df['iteration'].isin(recent_iters)]

        # 創建矩陣：迭代 x 長度分類
        matrix_data = []
        for it in recent_iters:
            iter_df = recent_df[recent_df['iteration'] == it]
            counts = [
                len(iter_df[iter_df['num_moves'] <= 12]),
                len(iter_df[(iter_df['num_moves'] > 12) & (iter_df['num_moves'] <= 20)]),
                len(iter_df[(iter_df['num_moves'] > 20) & (iter_df['num_moves'] <= 40)]),
                len(iter_df[iter_df['num_moves'] > 40])
            ]
            matrix_data.append(counts)

        matrix = np.array(matrix_data).T

        im = ax.imshow(matrix, aspect='auto', cmap='YlOrRd', interpolation='nearest')
        ax.set_xticks(range(len(recent_iters)))
        ax.set_xticklabels(recent_iters)
        ax.set_yticks(range(4))
        ax.set_yticklabels(labels)
        ax.set_xlabel('迭代次數')
        ax.set_title('最近 10 次迭代的長度分佈熱圖')

        # 添加數值標籤
        for i in range(4):
            for j in range(len(recent_iters)):
                text = ax.text(j, i, int(matrix[i, j]),
                             ha="center", va="center", color="black", fontsize=9)

        plt.colorbar(im, ax=ax, label='遊戲數量')

        # 4. 統計摘要表
        ax = axes[1, 1]
        ax.axis('off')

        # 生成統計表
        summary_text = f"""
【整體統計】
總遊戲數: {len(self.games_df)}
迭代範圍: {self.games_df['iteration'].min()} - {self.games_df['iteration'].max()}

【長度統計】
平均步數: {self.games_df['num_moves'].mean():.1f}
中位數: {self.games_df['num_moves'].median():.1f}
最短: {self.games_df['num_moves'].min()} 步
最長: {self.games_df['num_moves'].max()} 步

【策略分佈】
超短局 (≤12步): {len(self.games_df[self.games_df['num_moves'] <= 12])} ({len(self.games_df[self.games_df['num_moves'] <= 12])/len(self.games_df)*100:.1f}%)
短局 (13-20步): {len(self.games_df[(self.games_df['num_moves'] > 12) & (self.games_df['num_moves'] <= 20)])} ({len(self.games_df[(self.games_df['num_moves'] > 12) & (self.games_df['num_moves'] <= 20)])/len(self.games_df)*100:.1f}%)
中局 (21-40步): {len(self.games_df[(self.games_df['num_moves'] > 20) & (self.games_df['num_moves'] <= 40)])} ({len(self.games_df[(self.games_df['num_moves'] > 20) & (self.games_df['num_moves'] <= 40)])/len(self.games_df)*100:.1f}%)
長局 (>40步): {len(self.games_df[self.games_df['num_moves'] > 40])} ({len(self.games_df[self.games_df['num_moves'] > 40])/len(self.games_df)*100:.1f}%)

【勝率統計】
黑勝率: {self.games_df['black_win'].mean()*100:.1f}%
白勝率: {self.games_df['white_win'].mean()*100:.1f}%
平局率: {self.games_df['draw'].mean()*100:.1f}%
"""

        ax.text(0.1, 0.9, summary_text, transform=ax.transAxes,
               fontsize=11, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        plt.tight_layout()
        plt.savefig(self.output_dir / '8_game_patterns.png', dpi=300, bbox_inches='tight')
        plt.close()

    def plot_comprehensive_dashboard(self):
        """繪製綜合儀表板"""
        print("  生成: 綜合儀表板...")

        fig = plt.figure(figsize=(20, 12))
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        fig.suptitle('訓練綜合儀表板', fontsize=18, fontweight='bold')

        iterations = self.history['iterations']

        # 1. 總損失
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.plot(iterations, self.history['total_loss'], 'b-', linewidth=2)
        ax1.set_title('總損失', fontweight='bold')
        ax1.set_xlabel('迭代')
        ax1.set_ylabel('Loss')
        ax1.grid(True, alpha=0.3)

        # 2. 梯度健康度
        ax2 = fig.add_subplot(gs[0, 1])
        grad_norms = self.history['gradient_norm']
        colors = ['green' if 0.5 <= g <= 5.0 else 'red' for g in grad_norms]
        ax2.scatter(iterations, grad_norms, c=colors, s=50, alpha=0.6)
        ax2.axhline(y=0.5, color='red', linestyle='--', alpha=0.3)
        ax2.axhline(y=5.0, color='red', linestyle='--', alpha=0.3)
        ax2.set_title('梯度健康度', fontweight='bold')
        ax2.set_xlabel('迭代')
        ax2.set_ylabel('梯度範數')
        ax2.set_yscale('log')
        ax2.grid(True, alpha=0.3)

        # 3. 價值 MAE
        ax3 = fig.add_subplot(gs[0, 2])
        ax3.plot(iterations, self.history['value_mae'], 'r-', linewidth=2)
        ax3.set_title('價值網路 MAE', fontweight='bold')
        ax3.set_xlabel('迭代')
        ax3.set_ylabel('MAE')
        ax3.grid(True, alpha=0.3)

        # 4. 遊戲長度演化
        ax4 = fig.add_subplot(gs[1, 0])
        iter_avg_length = self.games_df.groupby('iteration')['num_moves'].mean()
        ax4.plot(iter_avg_length.index, iter_avg_length.values, 'g-',
                linewidth=2, marker='o', markersize=4)
        ax4.set_title('平均遊戲長度', fontweight='bold')
        ax4.set_xlabel('迭代')
        ax4.set_ylabel('步數')
        ax4.grid(True, alpha=0.3)

        # 5. 黑白勝率
        ax5 = fig.add_subplot(gs[1, 1])
        iter_stats = self.games_df.groupby('iteration').agg({
            'black_win': 'mean',
            'white_win': 'mean'
        })
        ax5.plot(iter_stats.index, iter_stats['black_win'], 'ko-',
                linewidth=2, markersize=4, label='黑')
        ax5.plot(iter_stats.index, iter_stats['white_win'], 'wo-',
                linewidth=2, markersize=4, markeredgecolor='black', label='白')
        ax5.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
        ax5.set_title('黑白勝率', fontweight='bold')
        ax5.set_xlabel('迭代')
        ax5.set_ylabel('勝率')
        ax5.set_ylim([0, 1])
        ax5.legend()
        ax5.grid(True, alpha=0.3)

        # 6. Edge-Rush 比例
        ax6 = fig.add_subplot(gs[1, 2])
        ultra_short_pct = self.games_df.groupby('iteration').apply(
            lambda x: len(x[x['num_moves'] <= 12]) / len(x) * 100
        )
        ax6.plot(ultra_short_pct.index, ultra_short_pct.values, 'r-',
                linewidth=2, marker='o', markersize=4)
        ax6.fill_between(ultra_short_pct.index, 0, ultra_short_pct.values,
                        color='red', alpha=0.3)
        ax6.axhline(y=40, color='orange', linestyle='--', alpha=0.5, label='警戒線')
        ax6.set_title('超短局比例 (Edge-Rush)', fontweight='bold')
        ax6.set_xlabel('迭代')
        ax6.set_ylabel('比例 (%)')
        ax6.legend()
        ax6.grid(True, alpha=0.3)

        # 7. 遊戲長度分佈（最新迭代）
        ax7 = fig.add_subplot(gs[2, 0])
        latest_iter = self.games_df['iteration'].max()
        latest_games = self.games_df[self.games_df['iteration'] == latest_iter]
        ax7.hist(latest_games['num_moves'], bins=30, color='blue', alpha=0.7, edgecolor='black')
        ax7.set_title(f'最新迭代 ({latest_iter}) 長度分佈', fontweight='bold')
        ax7.set_xlabel('步數')
        ax7.set_ylabel('頻率')
        ax7.grid(True, alpha=0.3, axis='y')

        # 8. 策略健康度分數
        ax8 = fig.add_subplot(gs[2, 1])
        health_scores = 100 - ultra_short_pct
        colors = ['red' if s < 50 else 'orange' if s < 70 else 'green'
                 for s in health_scores]
        ax8.bar(health_scores.index, health_scores.values, color=colors, alpha=0.7)
        ax8.axhline(y=50, color='red', linestyle='--', alpha=0.5)
        ax8.axhline(y=70, color='orange', linestyle='--', alpha=0.5)
        ax8.set_title('策略健康度分數', fontweight='bold')
        ax8.set_xlabel('迭代')
        ax8.set_ylabel('分數')
        ax8.set_ylim([0, 100])
        ax8.grid(True, alpha=0.3, axis='y')

        # 9. 關鍵指標表
        ax9 = fig.add_subplot(gs[2, 2])
        ax9.axis('off')

        # 計算關鍵指標
        latest_loss = self.history['total_loss'][-1]
        latest_grad = self.history['gradient_norm'][-1]
        latest_mae = self.history['value_mae'][-1]
        avg_game_length = self.games_df['num_moves'].mean()
        ultra_short_overall = len(self.games_df[self.games_df['num_moves'] <= 12]) / len(self.games_df) * 100
        black_wr = self.games_df['black_win'].mean() * 100

        metrics_text = f"""
【當前狀態】迭代: {iterations[-1]}

【訓練指標】
總損失: {latest_loss:.3f}
梯度範數: {latest_grad:.3f}
價值MAE: {latest_mae:.3f}

【遊戲統計】
平均長度: {avg_game_length:.1f} 步
超短局: {ultra_short_overall:.1f}%
黑勝率: {black_wr:.1f}%

【健康度評估】
梯度: {'✓ 健康' if 0.5 <= latest_grad <= 5.0 else '✗ 異常'}
策略: {'✓ 良好' if ultra_short_overall < 40 else '✗ Edge-Rush'}
平衡: {'✓ 平衡' if 40 < black_wr < 60 else '✗ 不平衡'}
"""

        ax9.text(0.1, 0.9, metrics_text, transform=ax9.transAxes,
                fontsize=12, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))

        plt.savefig(self.output_dir / '9_comprehensive_dashboard.png', dpi=300, bbox_inches='tight')
        plt.close()

    def plot_correlation_matrix(self):
        """繪製相關性矩陣"""
        print("  生成: 指標相關性分析...")

        # 準備數據
        data_dict = {
            '迭代': self.history['iterations'],
            '總損失': self.history['total_loss'],
            '策略損失': self.history['policy_loss'],
            '價值損失': self.history['value_loss'],
            '梯度範數': self.history['gradient_norm'],
            '價值MAE': self.history['value_mae'],
        }

        # 添加遊戲統計
        iter_game_stats = self.games_df.groupby('iteration').agg({
            'num_moves': 'mean',
            'black_win': 'mean'
        }).reset_index()

        # 合併數據
        df = pd.DataFrame(data_dict)
        df = df.merge(iter_game_stats, left_on='迭代', right_on='iteration', how='left')
        df['平均步數'] = df['num_moves']
        df['黑勝率'] = df['black_win']

        # 選擇要分析的列
        cols_to_analyze = ['總損失', '策略損失', '價值損失', '梯度範數',
                          '價值MAE', '平均步數', '黑勝率']
        correlation_df = df[cols_to_analyze].corr()

        # 繪製相關性矩陣
        fig, ax = plt.subplots(figsize=(12, 10))

        im = ax.imshow(correlation_df, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')

        # 設置刻度
        ax.set_xticks(np.arange(len(cols_to_analyze)))
        ax.set_yticks(np.arange(len(cols_to_analyze)))
        ax.set_xticklabels(cols_to_analyze, rotation=45, ha='right')
        ax.set_yticklabels(cols_to_analyze)

        # 添加數值標籤
        for i in range(len(cols_to_analyze)):
            for j in range(len(cols_to_analyze)):
                text = ax.text(j, i, f'{correlation_df.iloc[i, j]:.2f}',
                             ha="center", va="center",
                             color="white" if abs(correlation_df.iloc[i, j]) > 0.5 else "black",
                             fontsize=10, fontweight='bold')

        ax.set_title('訓練指標相關性矩陣', fontsize=16, fontweight='bold', pad=20)

        # 添加顏色條
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('相關係數', rotation=270, labelpad=20)

        plt.tight_layout()
        plt.savefig(self.output_dir / '10_correlation_matrix.png', dpi=300, bbox_inches='tight')
        plt.close()


def main():
    """主函數"""
    print("=" * 80)
    print("綜合圖表分析工具")
    print("=" * 80)
    print()

    plotter = ComprehensivePlotter()
    plotter.plot_all()

    print("\n" + "=" * 80)
    print("✓ 完成！所有圖表已生成")
    print("=" * 80)
    print(f"\n查看圖表: {plotter.output_dir}")
    print()
    print("生成的圖表:")
    for i, name in enumerate([
        "1_training_metrics.png - 訓練指標總覽",
        "2_loss_breakdown.png - 損失分解分析",
        "3_gradient_analysis.png - 梯度健康度分析",
        "4_value_network_metrics.png - 價值網路分析",
        "5_game_length_distribution.png - 遊戲長度分佈",
        "6_win_rate_trends.png - 勝率趨勢分析",
        "7_strategy_evolution.png - 策略演化分析",
        "8_game_patterns.png - 遊戲模式分析",
        "9_comprehensive_dashboard.png - 綜合儀表板",
        "10_correlation_matrix.png - 指標相關性分析"
    ], 1):
        print(f"  {i}. {name}")


if __name__ == '__main__':
    main()
