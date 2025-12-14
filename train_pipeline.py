"""主训练流程 - 现代化五子棋AI训练系统"""

import os

# 靜音 TensorFlow C++ 層的 INFO 日誌，避免多進程刷屏
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import json
import argparse
import numpy as np
import multiprocessing
from datetime import datetime
from tensorflow.keras.optimizers import Adam
import tensorflow as tf

# GPU 配置
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        # 使用第一个 GPU (NVIDIA RTX 3070 Ti)
        tf.config.set_visible_devices(gpus[0], 'GPU')
        # 允许 GPU 内存按需增长，避免占用全部显存
        tf.config.experimental.set_memory_growth(gpus[0], True)
        print(f"✅ 使用 GPU: {gpus[0].name}")
    except RuntimeError as e:
        print(f"⚠️  GPU 配置错误: {e}")

from game import GomokuGame
from core.neural_net import create_enhanced_model, prepare_input
from core.mcts import MCTS
from core.game_state import GameState
from core.data_augmentation import get_symmetries
from training.config import TrainingConfig
from training.replay_buffer import PrioritizedReplayBuffer
from evaluation.elo_rating import EloRating
from evaluation.arena import Arena


def run_self_play_game_worker(args):
    """
    自我对弈worker（多进程）

    Args:
        args: (game_num, model_weights_path, config_dict)

    Returns:
        list of (state, policy, value) tuples
    """
    game_num, model_weights_path, config_dict = args

    # 重建配置对象
    config = TrainingConfig()
    for key, value in config_dict.items():
        setattr(config, key, value)

    print(f"  [进程 {os.getpid()}] 开始第 {game_num + 1} 局...")

    # 创建模型
    model = create_enhanced_model(
        board_size=config.BOARD_SIZE,
        num_res_blocks=config.NUM_RES_BLOCKS,
        num_filters=config.NUM_FILTERS,
        se_ratio=config.SE_RATIO,
        l2_reg=config.L2_REG
    )

    # 加载权重
    if os.path.exists(model_weights_path):
        model.load_weights(model_weights_path)

    # 创建MCTS
    mcts = MCTS(model, config)

    # 创建游戏
    state = GameState(board_size=config.BOARD_SIZE)
    game_history = []

    # 游戏循环
    move_count = 0
    max_moves = config.BOARD_SIZE * config.BOARD_SIZE

    while not state.is_game_over() and move_count < max_moves:
        # MCTS搜索
        action_probs, root_value = mcts.search(state, add_noise=True)

        # 温度采样
        if move_count < config.TEMP_THRESHOLD_MOVE:
            temperature = 1.0
        elif move_count < config.TEMP_FINAL_MOVE:
            temperature = 0.5
        else:
            temperature = 0.01

        action = mcts.get_action_with_temperature(action_probs, temperature)

        # 记录状态
        game_history.append({
            'state': state.to_input(),  # (15, 15, 3)
            'policy': action_probs,      # (225,)
            'turn': state.get_current_player()
        })

        # 执行动作
        state.make_move(*action)
        move_count += 1

    # 游戏结束，计算价值标签
    winner = state.get_winner()
    training_data = []

    for i, entry in enumerate(game_history):
        steps_to_end = len(game_history) - i - 1

        if winner == 0:
            value = 0.0  # 平局
        else:
            base_value = 1.0 if entry['turn'] == winner else -1.0
            # 价值折扣
            value = base_value * (config.VALUE_GAMMA ** steps_to_end)

        training_data.append((
            entry['state'],
            entry['policy'],
            value
        ))

    print(f"  [进程 {os.getpid()}] 第 {game_num + 1} 局完成，{move_count}步")

    # 清理内存
    del model
    del mcts
    import gc
    gc.collect()

    return training_data


def evaluate_vs_random(model, config, num_games=20):
    """
    评估vs随机玩家

    Returns:
        win_rate
    """
    class RandomPlayer:
        def get_action(self, state):
            import random
            legal_moves = state.get_legal_moves()
            return random.choice(legal_moves) if legal_moves else None

    class ModelPlayer:
        def __init__(self, model, board_size):
            self.model = model
            self.board_size = board_size

        def get_action(self, state):
            nn_input = state.to_input()
            nn_input = np.expand_dims(nn_input, axis=0)
            policy, _ = self.model.predict(nn_input, verbose=0)

            legal_moves = state.get_legal_moves()
            if not legal_moves:
                return None

            # Mask非法移动
            legal_indices = [r * self.board_size + c for r, c in legal_moves]
            masked_policy = np.zeros_like(policy[0])
            masked_policy[legal_indices] = policy[0][legal_indices]

            if masked_policy.sum() > 0:
                move_idx = np.argmax(masked_policy)
            else:
                import random
                move_idx = random.choice(legal_indices)

            row, col = divmod(move_idx, self.board_size)
            return (row, col)

    arena = Arena(board_size=config.BOARD_SIZE)
    model_player = ModelPlayer(model, config.BOARD_SIZE)
    random_player = RandomPlayer()

    results = arena.compete(model_player, random_player, num_games=num_games)

    return results['player1_win_rate']


def train(config, resume_from=None):
    """
    主训练函数

    Args:
        config: TrainingConfig对象
        resume_from: 恢复训练的模型路径（可选）
    """
    print("=" * 60)
    print("现代化深度学习五子棋训练系统")
    print("=" * 60)
    print(f"\n配置:\n{config}\n")

    # 创建目录
    os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)

    # 创建模型
    print("创建模型...")
    model = create_enhanced_model(
        board_size=config.BOARD_SIZE,
        num_res_blocks=config.NUM_RES_BLOCKS,
        num_filters=config.NUM_FILTERS,
        se_ratio=config.SE_RATIO,
        l2_reg=config.L2_REG,
        dropout_rate=config.DROPOUT_RATE,
        value_head_hidden=config.VALUE_HEAD_HIDDEN
    )

    # 编译模型
    model.compile(
        optimizer=Adam(learning_rate=config.LEARNING_RATE, clipnorm=config.GRADIENT_CLIP_NORM),
        loss={
            'policy_output': 'categorical_crossentropy',
            'value_output': tf.keras.losses.Huber(delta=1.0)
        },
        loss_weights={
            'policy_output': config.POLICY_LOSS_WEIGHT,
            'value_output': config.VALUE_LOSS_WEIGHT
        }
    )

    # 加载已有模型
    model_weights_path = os.path.join(config.CHECKPOINT_DIR, 'latest_model.weights.h5')
    start_iteration = 0

    if resume_from and os.path.exists(resume_from):
        print(f"从 {resume_from} 恢复训练...")
        model.load_weights(resume_from)
    elif os.path.exists(model_weights_path):
        print(f"加载已有模型 {model_weights_path}...")
        model.load_weights(model_weights_path)

    # 创建经验回放缓冲区
    replay_buffer = PrioritizedReplayBuffer(
        capacity=config.REPLAY_BUFFER_SIZE,
        alpha=config.PRIORITIZED_ALPHA,
        beta=config.PRIORITIZED_BETA
    )

    # 训练历史
    history = {
        'iterations': [],
        'total_loss': [],
        'policy_loss': [],
        'value_loss': [],
        'win_rate_vs_random': []
    }

    # 训练循环
    for iteration in range(start_iteration, config.ITERATIONS):
        print(f"\n{'=' * 60}")
        print(f"迭代 {iteration + 1}/{config.ITERATIONS}")
        print(f"{'=' * 60}")

        # 1. 自我对弈
        print(f"\n[1/4] 自我对弈 {config.GAMES_PER_ITERATION} 局（{config.NUM_WORKERS}个进程）...")

        # 保存当前模型供worker使用
        model.save_weights(model_weights_path)

        # 准备参数
        config_dict = {k: v for k, v in config.__dict__.items() if not k.startswith('_')}
        args_list = [(i, model_weights_path, config_dict) for i in range(config.GAMES_PER_ITERATION)]

        # 多进程自我对弈
        with multiprocessing.Pool(processes=config.NUM_WORKERS) as pool:
            results = pool.map(run_self_play_game_worker, args_list)

        # 收集数据
        all_games_data = [item for game_data in results for item in game_data]
        print(f"  收集到 {len(all_games_data)} 步训练数据")

        # 2. 数据增强
        print(f"\n[2/4] 数据增强（8种对称变换）...")
        augmented_data = []
        for state, policy, value in all_games_data:
            # 随机选择一种对称变换
            symmetries = get_symmetries(state, policy, config.BOARD_SIZE)
            aug_state, aug_policy = symmetries[np.random.randint(8)]
            augmented_data.append((aug_state, aug_policy, value))

        # 添加到回放缓冲区
        for data in augmented_data:
            replay_buffer.add(data)

        print(f"  缓冲区大小: {len(replay_buffer)}/{config.REPLAY_BUFFER_SIZE}")

        # 3. 训练模型
        print(f"\n[3/4] 训练模型（{config.EPOCHS_PER_ITERATION} epochs）...")

        if len(replay_buffer) < config.BATCH_SIZE:
            print(f"  缓冲区数据不足，跳过训练")
            continue

        epoch_losses = []
        for epoch in range(config.EPOCHS_PER_ITERATION):
            # 从缓冲区采样
            samples, weights, indices = replay_buffer.sample(config.BATCH_SIZE)

            if len(samples) == 0:
                continue

            # 准备批次数据
            X = np.array([s[0] for s in samples])
            y_policy = np.array([s[1] for s in samples])
            y_value = np.array([s[2] for s in samples])

            # 训练
            loss_dict = model.train_on_batch(
                X,
                [y_policy, y_value],
                sample_weight=[weights, weights],
                return_dict=True
            )

            epoch_losses.append(loss_dict)

        # 计算平均损失
        avg_loss = {
            'loss': np.mean([l['loss'] for l in epoch_losses]),
            'policy_output_loss': np.mean([l['policy_output_loss'] for l in epoch_losses]),
            'value_output_loss': np.mean([l['value_output_loss'] for l in epoch_losses])
        }

        print(f"  总损失: {avg_loss['loss']:.4f}, "
              f"策略: {avg_loss['policy_output_loss']:.4f}, "
              f"价值: {avg_loss['value_output_loss']:.4f}")

        # 4. 评估
        win_rate = None
        if (iteration + 1) % config.EVAL_FREQUENCY == 0:
            print(f"\n[4/4] 评估模型 vs 随机玩家...")
            win_rate = evaluate_vs_random(model, config, num_games=config.EVAL_GAMES)
            print(f"  胜率: {win_rate:.3f}")

        # 记录历史
        history['iterations'].append(iteration + 1)
        history['total_loss'].append(avg_loss['loss'])
        history['policy_loss'].append(avg_loss['policy_output_loss'])
        history['value_loss'].append(avg_loss['value_output_loss'])
        history['win_rate_vs_random'].append(win_rate)

        # 保存模型
        if (iteration + 1) % config.CHECKPOINT_FREQUENCY == 0:
            checkpoint_path = os.path.join(
                config.CHECKPOINT_DIR,
                f'model_iter_{iteration+1}.weights.h5'
            )
            model.save_weights(checkpoint_path)
            print(f"  保存检查点: {checkpoint_path}")

        # 保存最新模型
        model.save_weights(model_weights_path)

        # 保存训练历史（转换 NumPy 类型为 Python 原生类型）
        history_path = os.path.join(config.CHECKPOINT_DIR, 'training_history.json')

        # 转换所有值为 Python 原生类型
        def convert_to_native(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (np.float32, np.float64)):
                return float(obj)
            elif isinstance(obj, (np.int32, np.int64)):
                return int(obj)
            elif isinstance(obj, dict):
                return {k: convert_to_native(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_native(item) for item in obj]
            else:
                return obj

        history_native = convert_to_native(history)

        with open(history_path, 'w') as f:
            json.dump(history_native, f, indent=2)

    print(f"\n{'=' * 60}")
    print("训练完成！")
    print(f"{'=' * 60}")
    print(f"最终模型: {model_weights_path}")
    print(f"训练历史: {history_path}")


if __name__ == '__main__':
    multiprocessing.freeze_support()

    parser = argparse.ArgumentParser(description='现代化五子棋AI训练')
    parser.add_argument('--fast-test', '-f', action='store_true', help='使用快速测试配置')
    parser.add_argument('--iterations', '-i', type=int, help='训练迭代次数')
    parser.add_argument('--games', '-g', type=int, help='每次迭代游戏局数')
    parser.add_argument('--resume', '-r', type=str, help='恢复训练的模型路径')

    args = parser.parse_args()

    # 选择配置
    if args.fast_test:
        config = TrainingConfig.get_fast_test_config()
        print("使用快速测试配置")
    else:
        config = TrainingConfig.get_full_config()
        print("使用完整训练配置")

    # 覆盖配置
    if args.iterations:
        config.ITERATIONS = args.iterations
    if args.games:
        config.GAMES_PER_ITERATION = args.games

    # 开始训练
    train(config, resume_from=args.resume)
