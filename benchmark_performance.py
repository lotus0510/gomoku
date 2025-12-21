"""性能基準測試 - 比較優化前後的速度"""

import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import time
import numpy as np
from core.neural_net import create_enhanced_model
from core.mcts import MCTS
from core.mcts_batched import BatchedMCTS
from core.game_state import GameState
from training.config import TrainingConfig


def benchmark_mcts(mcts_class, config, num_games=5):
    """
    測試MCTS性能

    Args:
        mcts_class: MCTS類（MCTS或BatchedMCTS）
        config: 配置對象
        num_games: 測試遊戲局數

    Returns:
        平均每步耗時（秒）
    """
    # 創建模型
    model = create_enhanced_model(
        board_size=config.BOARD_SIZE,
        num_res_blocks=config.NUM_RES_BLOCKS,
        num_filters=config.NUM_FILTERS,
        se_ratio=config.SE_RATIO,
        l2_reg=config.L2_REG
    )

    # 創建MCTS
    mcts = mcts_class(model, config)

    total_time = 0
    total_steps = 0

    for game_idx in range(num_games):
        state = GameState(board_size=config.BOARD_SIZE)
        move_count = 0
        max_moves = 30  # 限制最多30步

        while not state.is_game_over() and move_count < max_moves:
            start = time.time()
            action_probs, _ = mcts.search(state, add_noise=False)
            elapsed = time.time() - start

            total_time += elapsed
            total_steps += 1

            # 選擇最佳移動
            move_idx = np.argmax(action_probs)
            row = move_idx // config.BOARD_SIZE
            col = move_idx % config.BOARD_SIZE

            state.make_move(row, col)
            move_count += 1

    avg_time_per_step = total_time / total_steps if total_steps > 0 else 0
    return avg_time_per_step, total_steps


def main():
    """運行性能基準測試"""
    print("=" * 80)
    print("性能基準測試 - MCTS優化效果")
    print("=" * 80)

    # 使用快速測試配置
    config = TrainingConfig.get_fast_test_config()
    config.MCTS_SIMULATIONS = 100  # 使用100次模擬
    config.MCTS_BATCH_SIZE = 8     # 批量大小8

    print(f"\n配置:")
    print(f"  MCTS模擬次數: {config.MCTS_SIMULATIONS}")
    print(f"  批量大小: {config.MCTS_BATCH_SIZE}")
    print(f"  測試遊戲數: 5局")
    print(f"  模型: {config.NUM_RES_BLOCKS}層ResNet, {config.NUM_FILTERS}個濾波器")

    # 測試標準MCTS
    print(f"\n{'=' * 80}")
    print("測試 1: 標準MCTS（單個推理）")
    print("=" * 80)

    start = time.time()
    avg_time_standard, steps_standard = benchmark_mcts(MCTS, config, num_games=5)
    total_time_standard = time.time() - start

    print(f"  總耗時: {total_time_standard:.2f} 秒")
    print(f"  總步數: {steps_standard}")
    print(f"  平均每步: {avg_time_standard:.3f} 秒")
    print(f"  推理次數: ~{steps_standard * config.MCTS_SIMULATIONS:,} 次")

    # 測試批量MCTS
    print(f"\n{'=' * 80}")
    print("測試 2: 批量MCTS（批量推理）")
    print("=" * 80)

    start = time.time()
    avg_time_batched, steps_batched = benchmark_mcts(BatchedMCTS, config, num_games=5)
    total_time_batched = time.time() - start

    print(f"  總耗時: {total_time_batched:.2f} 秒")
    print(f"  總步數: {steps_batched}")
    print(f"  平均每步: {avg_time_batched:.3f} 秒")
    print(f"  批量推理次數: ~{steps_batched * config.MCTS_SIMULATIONS // config.MCTS_BATCH_SIZE:,} 次")

    # 計算加速比
    speedup = avg_time_standard / avg_time_batched if avg_time_batched > 0 else 0
    total_speedup = total_time_standard / total_time_batched if total_time_batched > 0 else 0

    print(f"\n{'=' * 80}")
    print("性能對比")
    print("=" * 80)
    print(f"  每步加速比: {speedup:.2f}x")
    print(f"  總體加速比: {total_speedup:.2f}x")
    print(f"  時間節省: {(1 - 1/total_speedup)*100:.1f}%")

    # 預估訓練時間節省
    print(f"\n{'=' * 80}")
    print("訓練時間預估（100局/迭代，30步/局）")
    print("=" * 80)

    games_per_iter = 100
    steps_per_game = 30
    total_steps = games_per_iter * steps_per_game

    time_standard = total_steps * avg_time_standard / 60  # 分鐘
    time_batched = total_steps * avg_time_batched / 60    # 分鐘

    print(f"  標準MCTS: {time_standard:.1f} 分鐘")
    print(f"  批量MCTS: {time_batched:.1f} 分鐘")
    print(f"  每次迭代節省: {time_standard - time_batched:.1f} 分鐘")
    print(f"  1000次迭代節省: {(time_standard - time_batched) * 1000 / 60:.1f} 小時")

    print(f"\n{'=' * 80}")
    print("✓ 性能測試完成！")
    print("=" * 80)


if __name__ == '__main__':
    main()
