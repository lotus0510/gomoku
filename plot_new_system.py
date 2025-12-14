"""新训练系统的可视化工具"""

import matplotlib.pyplot as plt
import json
import os
import numpy as np

def plot_new_training_history():
    """绘制新训练系统的历史数据"""

    # 尝试读取新系统的历史
    new_history_file = 'checkpoints/training_history.json'
    old_history_file = 'training_history.json'

    if os.path.exists(new_history_file):
        history_file = new_history_file
    elif os.path.exists(old_history_file):
        history_file = old_history_file
    else:
        print("错误：找不到训练历史文件")
        print("  尝试的路径：")
        print(f"    - {new_history_file}")
        print(f"    - {old_history_file}")
        return

    print(f"从 '{history_file}' 读取数据...")
    with open(history_file, 'r', encoding='utf-8') as f:
        history = json.load(f)

    iterations = history.get('iterations', [])
    total_loss = history.get('total_loss', [])
    policy_loss = history.get('policy_loss', [])
    value_loss = history.get('value_loss', [])
    win_rate = history.get('win_rate_vs_random', [])

    if not total_loss:
        print("错误：历史数据为空")
        return

    # 创建图表
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))

    # === 损失曲线 ===
    ax1.plot(iterations, total_loss, label='Total Loss',
             linestyle='--', color='red', marker='x', linewidth=2)
    ax1.plot(iterations, policy_loss, label='Policy Loss',
             marker='o', linewidth=1.5)
    ax1.plot(iterations, value_loss, label='Value Loss',
             marker='s', linewidth=1.5)

    # 标注最低损失
    if total_loss:
        min_idx = np.argmin(total_loss)
        min_val = total_loss[min_idx]
        min_iter = iterations[min_idx]
        ax1.annotate(f'Min: {min_val:.4f}',
                    xy=(min_iter, min_val),
                    xytext=(min_iter + 1, min_val + 0.1),
                    arrowprops=dict(arrowstyle='->', color='red'),
                    fontsize=10, color='red')

    ax1.set_title('Modern Gomoku AI - Training Loss', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Iteration', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.legend(loc='upper right', fontsize=10)
    ax1.grid(True, alpha=0.3)

    # 添加统计信息
    if total_loss:
        info_text = (
            f"Iterations: {len(iterations)}\n"
            f"Latest Loss: {total_loss[-1]:.4f}\n"
            f"  Policy: {policy_loss[-1]:.4f}\n"
            f"  Value: {value_loss[-1]:.4f}\n"
            f"Min Total: {min(total_loss):.4f}\n"
            f"Avg Total: {np.mean(total_loss):.4f}"
        )
        ax1.text(0.02, 0.98, info_text,
                transform=ax1.transAxes,
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
                fontsize=9, family='monospace')

    # === 胜率曲线 ===
    # 过滤掉 None 值
    valid_win_rate = [(i, w) for i, w in zip(iterations, win_rate) if w is not None]

    if valid_win_rate:
        valid_iters, valid_rates = zip(*valid_win_rate)
        ax2.plot(valid_iters, valid_rates,
                label='Win Rate vs Random',
                color='green', marker='o', linewidth=2, markersize=8)
        ax2.axhline(y=0.95, color='red', linestyle='--',
                   label='Target: 95%', alpha=0.5)
        ax2.fill_between(valid_iters, 0, valid_rates,
                        alpha=0.2, color='green')

        # 标注最新胜率
        latest_rate = valid_rates[-1]
        latest_iter = valid_iters[-1]
        ax2.annotate(f'{latest_rate:.1%}',
                    xy=(latest_iter, latest_rate),
                    xytext=(latest_iter - 1, latest_rate + 0.05),
                    arrowprops=dict(arrowstyle='->', color='green'),
                    fontsize=11, color='green', fontweight='bold')

        ax2.set_title('Evaluation: Win Rate vs Random Player',
                     fontsize=14, fontweight='bold')
        ax2.set_xlabel('Iteration', fontsize=12)
        ax2.set_ylabel('Win Rate', fontsize=12)
        ax2.set_ylim(0, 1.05)
        ax2.legend(loc='lower right', fontsize=10)
        ax2.grid(True, alpha=0.3)

        # 添加胜率统计
        win_info = (
            f"Evaluations: {len(valid_rates)}\n"
            f"Latest: {valid_rates[-1]:.1%}\n"
            f"Best: {max(valid_rates):.1%}\n"
            f"Avg: {np.mean(valid_rates):.1%}"
        )
        ax2.text(0.02, 0.98, win_info,
                transform=ax2.transAxes,
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5),
                fontsize=9, family='monospace')
    else:
        ax2.text(0.5, 0.5, 'No evaluation data yet\n(eval_frequency not reached)',
                ha='center', va='center', fontsize=14, color='gray',
                transform=ax2.transAxes)
        ax2.set_title('Evaluation: Win Rate vs Random Player',
                     fontsize=14, fontweight='bold')

    plt.tight_layout()

    # 保存图表
    output_file = 'training_visualization.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"\n图表已保存到: {output_file}")

    print("\n正在显示图表...")
    plt.show()


if __name__ == '__main__':
    print("=" * 60)
    print("Modern Gomoku AI - Training Visualization")
    print("=" * 60)
    plot_new_training_history()
