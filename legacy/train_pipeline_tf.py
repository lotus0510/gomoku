"""主训练流程 - 现代化五子棋AI训练系统"""

import os

# 靜音 TensorFlow C++ 層的 INFO 日誌，避免多進程刷屏
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import json
import argparse
import time
import random
import traceback
import numpy as np
import multiprocessing
from datetime import datetime
from tensorflow.keras.optimizers import Adam
import tensorflow as tf

# ===== Phase 1: 混合精度训练 + XLA 编译优化 =====
# 注意：当前使用 CPU 训练，混合精度已禁用
# GPU 配置完成后，将 'float32' 改回 'mixed_float16'
try:
    from tensorflow.keras import mixed_precision

    # CPU 模式：使用 FP32（混合精度在 CPU 上收益很小）
    # GPU 模式：改为 'mixed_float16' 以获得 2-3x 加速
    policy = mixed_precision.Policy('float32')  # ← CPU 模式
    mixed_precision.set_global_policy(policy)
    print(f"✅ 训练精度策略: {policy.name}")

    if policy.name == 'float32':
        print(f"   ⚠️  当前使用 CPU/FP32 模式")
        print(f"   💡 GPU 配置后改为 'mixed_float16' 可获得 2-3x 加速")
    else:
        print(f"   计算 dtype: {policy.compute_dtype}")
        print(f"   变量 dtype: {policy.variable_dtype}")

    # 启用 XLA 编译（CPU 上也有小幅帮助）
    tf.config.optimizer.set_jit(True)
    print("✅ XLA 编译优化已启用")

    MIXED_PRECISION_ENABLED = (policy.name == 'mixed_float16')
except Exception as e:
    print(f"⚠️  优化启用失败: {e}")
    print("   将使用默认配置训练")
    MIXED_PRECISION_ENABLED = False

# GPU 配置（只在主进程打印信息）
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        # 支持多GPU或通过环境变量指定
        gpu_id = int(os.environ.get('CUDA_VISIBLE_DEVICES', '0').split(',')[0])
        selected_gpu = gpus[gpu_id] if gpu_id < len(gpus) else gpus[0]

        # 允许 GPU 内存按需增长
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)

        # 只在主进程打印（避免多进程时重复输出）
        if __name__ == '__main__':
            print(f"✅ 检测到 {len(gpus)} 个GPU，使用: {selected_gpu.name}")
    except (RuntimeError, ValueError, IndexError) as e:
        if __name__ == '__main__':
            print(f"⚠️  GPU 配置错误: {e}，使用CPU")

from core.neural_net import create_enhanced_model
from core.mcts import MCTS
from core.mcts_batched import BatchedMCTS
from core.game_state import GameState
from core.data_augmentation import get_symmetries
from training.config import TrainingConfig
from training.replay_buffer import PrioritizedReplayBuffer
from training.game_logger import GameLogger
from evaluation.arena import Arena


# 全局变量用于缓存模型（每个worker进程一份）
_worker_model = None
_worker_config = None


def init_worker(model_weights_path, config_dict):
    """
    初始化worker进程（进程池创建时调用一次）

    Args:
        model_weights_path: 模型权重路径
        config_dict: 配置字典
    """
    global _worker_model, _worker_config

    # 重建配置
    _worker_config = TrainingConfig()
    for key, value in config_dict.items():
        setattr(_worker_config, key, value)

    # 创建并加载模型（每个进程只做一次）
    _worker_model = create_enhanced_model(
        board_size=_worker_config.BOARD_SIZE,
        num_res_blocks=_worker_config.NUM_RES_BLOCKS,
        num_filters=_worker_config.NUM_FILTERS,
        se_ratio=_worker_config.SE_RATIO,
        l2_reg=_worker_config.L2_REG
    )

    if os.path.exists(model_weights_path):
        _worker_model.load_weights(model_weights_path)

    # 静默初始化，避免8个进程同时打印造成混乱
    # print(f"  [进程 {os.getpid()}] 模型已初始化")


def run_self_play_game_worker(game_num):
    """
    自我对弈worker（多进程，使用缓存的模型）

    Args:
        game_num: 游戏编号

    Returns:
        tuple: (training_data, game_metadata)
            - training_data: list of (state, policy, value) tuples
            - game_metadata: dict with game statistics
    """
    global _worker_model, _worker_config

    try:
        start_time = time.time()

        print(f"  [进程 {os.getpid()}] 开始第 {game_num + 1} 局...")

        # 使用缓存的模型和配置
        model = _worker_model
        config = _worker_config

        # 创建MCTS（根据配置选择批量或标准版本）
        use_batched = getattr(config, 'USE_BATCHED_MCTS', False)
        if use_batched:
            mcts = BatchedMCTS(model, config)
        else:
            mcts = MCTS(model, config)

        # 创建游戏
        state = GameState(board_size=config.BOARD_SIZE)
        game_history = []
        moves_list = []  # 记录实际移动
        mcts_times = []  # 记录每步MCTS时间

        # 游戏循环
        move_count = 0
        max_moves = config.BOARD_SIZE * config.BOARD_SIZE

        while not state.is_game_over() and move_count < max_moves:
            mcts_start = time.time()
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

            mcts_times.append(time.time() - mcts_start)

            # 记录状态
            game_history.append({
                'state': state.to_input(),  # (15, 15, 3)
                'policy': action_probs,      # (225,)
                'turn': state.get_current_player(),
                'value': root_value  # 根节点价值估计
            })

            # 记录实际移动
            moves_list.append(action)

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

        game_duration = time.time() - start_time

        print(f"  [进程 {os.getpid()}] 第 {game_num + 1} 局完成，{move_count}步，{game_duration:.1f}秒")

        # 构建游戏元数据
        game_metadata = {
            'num_moves': move_count,
            'winner': winner,
            'duration': game_duration,
            'avg_mcts_time': np.mean(mcts_times) if mcts_times else 0,
            'moves': moves_list,
            'policies': [entry['policy'] for entry in game_history],
            'values': [entry['value'] for entry in game_history]
        }

        # 清理MCTS（模型是共享的，不删除）
        del mcts

        return training_data, game_metadata

    except Exception as e:
        print(f"  [进程 {os.getpid()}] 错误：第 {game_num + 1} 局失败 - {str(e)}")
        traceback.print_exc()
        return [], {}  # 返回空数据，避免整个训练崩溃


def save_iteration_summary(iteration, config, avg_loss, win_rate, selfplay_time,
                           training_time, eval_time, iteration_time, replay_buffer_size,
                           games_metadata, model):
    """
    保存迭代摘要文件（繁體中文白話版）

    Args:
        iteration: 迭代編號
        config: 訓練配置
        avg_loss: 平均損失字典
        win_rate: 對隨機玩家勝率
        selfplay_time: 自我對弈耗時
        training_time: 模型訓練耗時
        eval_time: 評估耗時
        iteration_time: 迭代總耗時
        replay_buffer_size: 緩衝區大小
        games_metadata: 遊戲元數據列表
        model: 模型對象
    """
    # 計算遊戲統計
    if games_metadata:
        avg_moves = np.mean([g['num_moves'] for g in games_metadata])
        avg_mcts_time = np.mean([g['avg_mcts_time'] for g in games_metadata])
        black_wins = sum(1 for g in games_metadata if g['winner'] == 1)
        white_wins = sum(1 for g in games_metadata if g['winner'] == -1)
        draws = sum(1 for g in games_metadata if g['winner'] == 0)
        total_games = len(games_metadata)
    else:
        avg_moves = avg_mcts_time = black_wins = white_wins = draws = total_games = 0

    # 獲取當前學習率
    current_lr = model.optimizer.learning_rate
    if hasattr(current_lr, 'numpy'):
        current_lr = float(current_lr.numpy())
    elif hasattr(current_lr, '__call__'):
        current_lr = float(current_lr(model.optimizer.iterations))
    else:
        current_lr = float(current_lr)

    # 計算模型參數量
    trainable_params = sum([np.prod(v.shape) for v in model.trainable_weights])

    # 準備摘要內容
    summary_lines = [
        "=" * 80,
        f"第 {iteration} 次迭代訓練摘要",
        "=" * 80,
        "",
        "【基本指標】",
        f"  • 總損失：{avg_loss['loss']:.4f}",
        f"  • 策略損失：{avg_loss['policy_output_loss']:.4f}",
        f"  • 價值損失：{avg_loss['value_output_loss']:.4f}",
    ]

    if win_rate is not None:
        summary_lines.append(f"  • 對隨機玩家勝率：{win_rate:.1%}")
    else:
        summary_lines.append(f"  • 對隨機玩家勝率：本次未評估")

    summary_lines.extend([
        "",
        "【性能統計】",
        f"  • 自我對弈耗時：{selfplay_time:.1f} 秒（{selfplay_time/60:.1f} 分鐘）",
        f"  • 模型訓練耗時：{training_time:.1f} 秒",
    ])

    if eval_time > 0:
        summary_lines.append(f"  • 模型評估耗時：{eval_time:.1f} 秒")

    summary_lines.extend([
        f"  • 迭代總耗時：{iteration_time:.1f} 秒（{iteration_time/60:.1f} 分鐘）",
        "",
        "【遊戲統計】",
        f"  • 本次對弈局數：{total_games} 局",
        f"  • 平均每局步數：{avg_moves:.1f} 步",
        f"  • 平均 MCTS 搜索時間：{avg_mcts_time:.3f} 秒/步",
        f"  • 黑方勝率：{black_wins}/{total_games} = {black_wins/total_games*100 if total_games > 0 else 0:.1f}%",
        f"  • 白方勝率：{white_wins}/{total_games} = {white_wins/total_games*100 if total_games > 0 else 0:.1f}%",
        f"  • 平局率：{draws}/{total_games} = {draws/total_games*100 if total_games > 0 else 0:.1f}%",
        "",
        "【系統狀態】",
        f"  • 經驗回放緩衝區：{replay_buffer_size}/{config.REPLAY_BUFFER_SIZE} ({replay_buffer_size/config.REPLAY_BUFFER_SIZE*100:.1f}% 已使用)",
        f"  • 當前學習率：{current_lr:.6f}",
        f"  • 模型參數量：{trainable_params:,} 個可訓練參數",
        f"  • 並行工作進程：{config.NUM_WORKERS} 個",
        "",
        "【訓練進度】",
        f"  • 已完成：{iteration}/{config.ITERATIONS} 次迭代（{iteration/config.ITERATIONS*100:.1f}%）",
        f"  • 預計剩餘時間：{(config.ITERATIONS - iteration) * iteration_time / 60:.1f} 分鐘",
        "",
        "=" * 80,
        f"摘要生成時間：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "=" * 80,
    ])

    # 保存到文件
    summary_dir = os.path.join(config.CHECKPOINT_DIR, 'summaries')
    os.makedirs(summary_dir, exist_ok=True)

    summary_path = os.path.join(summary_dir, f'iteration_{iteration:04d}_summary.txt')

    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(summary_lines))

    print(f"  保存迭代摘要: {summary_path}")


def evaluate_vs_random(model, config, num_games=20):
    """
    评估vs随机玩家

    Returns:
        win_rate
    """
    class RandomPlayer:
        def get_action(self, state):
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

    # 创建学习率调度器
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=config.LEARNING_RATE,
        decay_steps=config.LR_DECAY_STEPS,
        decay_rate=config.LR_DECAY_RATE,
        staircase=True
    )

    # 创建优化器
    optimizer = Adam(learning_rate=lr_schedule, clipnorm=config.GRADIENT_CLIP_NORM)

    # 如果启用混合精度，包装优化器以支持损失缩放
    if MIXED_PRECISION_ENABLED:
        optimizer = mixed_precision.LossScaleOptimizer(optimizer)
        print(f"✅ 优化器已包装为 LossScaleOptimizer（防止下溢）")

    # 编译模型
    model.compile(
        optimizer=optimizer,
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

    # 创建游戏日志记录器（可以通过配置禁用）
    enable_game_logging = getattr(config, 'ENABLE_GAME_LOGGING', True)
    detailed_frequency = getattr(config, 'DETAILED_GAME_LOG_FREQUENCY', 10)
    game_logger = GameLogger(
        log_dir='logs/games',
        enabled=enable_game_logging,
        detailed_frequency=detailed_frequency
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
        iteration_start_time = time.time()

        print(f"\n{'=' * 60}")
        print(f"迭代 {iteration + 1}/{config.ITERATIONS}")
        print(f"{'=' * 60}")

        # 1. 自我对弈
        print(f"\n[1/4] 自我对弈 {config.GAMES_PER_ITERATION} 局（{config.NUM_WORKERS}个进程）...")
        selfplay_start_time = time.time()

        # 保存当前模型供worker使用
        model.save_weights(model_weights_path)

        # 准备参数
        config_dict = {k: v for k, v in config.__dict__.items() if not k.startswith('_')}
        game_numbers = list(range(config.GAMES_PER_ITERATION))

        # 多进程自我对弈（使用initializer避免重复加载模型）
        with multiprocessing.Pool(
            processes=config.NUM_WORKERS,
            initializer=init_worker,
            initargs=(model_weights_path, config_dict)
        ) as pool:
            results = pool.map(run_self_play_game_worker, game_numbers)

        # 收集训练数据和游戏元数据
        all_games_data = []
        all_games_metadata = []
        for training_data, game_metadata in results:
            all_games_data.extend(training_data)
            if game_metadata:  # 如果不为空
                all_games_metadata.append(game_metadata)

        selfplay_time = time.time() - selfplay_start_time
        print(f"  收集到 {len(all_games_data)} 步训练数据，{len(all_games_metadata)} 局游戏")
        print(f"  自我对弈耗时: {selfplay_time:.1f}秒 ({selfplay_time/60:.1f}分钟)")

        # 记录每局游戏到日志
        for game_num, metadata in enumerate(all_games_metadata):
            game_logger.log_game(iteration + 1, game_num, metadata)

        # 2. 数据增强（并行化）
        print(f"\n[2/4] 数据增强（8种对称变换，并行处理）...")
        augment_start_time = time.time()

        # 并行处理数据增强
        def augment_single_data(data_tuple):
            """对单个数据进行8种对称变换"""
            state, policy, value = data_tuple
            symmetries = get_symmetries(state, policy, config.BOARD_SIZE)
            return [(aug_state, aug_policy, value) for aug_state, aug_policy in symmetries]

        # 使用多进程并行处理（使用CPU核心数的一半避免过载）
        num_aug_workers = max(1, config.NUM_WORKERS // 2)
        with multiprocessing.Pool(processes=num_aug_workers) as pool:
            augmented_results = pool.map(augment_single_data, all_games_data)

        # 展平结果
        augmented_data = [item for sublist in augmented_results for item in sublist]

        augment_time = time.time() - augment_start_time
        print(f"  增强后数据: {len(all_games_data)} -> {len(augmented_data)} (8x)")
        print(f"  数据增强耗时: {augment_time:.1f}秒（使用{num_aug_workers}个进程）")

        # 添加到回放缓冲区
        for data in augmented_data:
            replay_buffer.add(data)

        print(f"  缓冲区大小: {len(replay_buffer)}/{config.REPLAY_BUFFER_SIZE}")

        # 3. 训练模型
        print(f"\n[3/4] 训练模型（{config.EPOCHS_PER_ITERATION} epochs）...")
        training_start_time = time.time()

        if len(replay_buffer) < config.BATCH_SIZE:
            print(f"  缓冲区数据不足，跳过训练")
            continue

        epoch_losses = []
        last_indices = None
        last_X = None
        last_y_value = None

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

            # 保存最后一个批次用于更新优先级
            last_indices = indices
            last_X = X
            last_y_value = y_value

        # 更新优先级：使用最后一个批次计算 TD-error
        if last_indices is not None and len(last_indices) > 0:
            # 获取当前预测
            _, pred_values = model.predict(last_X, verbose=0)
            pred_values = pred_values.flatten()

            # 计算 TD-error（价值预测误差）
            td_errors = np.abs(pred_values - last_y_value)

            # 更新优先级
            replay_buffer.update_priorities(last_indices, td_errors)

        # 计算平均损失
        avg_loss = {
            'loss': np.mean([l['loss'] for l in epoch_losses]),
            'policy_output_loss': np.mean([l['policy_output_loss'] for l in epoch_losses]),
            'value_output_loss': np.mean([l['value_output_loss'] for l in epoch_losses])
        }

        # ===== Phase 1: NaN 检测和处理 =====
        if np.isnan(avg_loss['loss']) or np.isinf(avg_loss['loss']):
            print(f"\n🔴 检测到 NaN/Inf 损失！")
            print(f"   总损失: {avg_loss['loss']}")
            print(f"   策略损失: {avg_loss['policy_output_loss']}")
            print(f"   价值损失: {avg_loss['value_output_loss']}")

            if MIXED_PRECISION_ENABLED:
                print(f"\n⚠️  可能是混合精度训练导致的数值不稳定")
                print(f"   建议:")
                print(f"   1. 检查学习率是否过大")
                print(f"   2. 增加梯度裁剪")
                print(f"   3. 如果问题持续，在 train_pipeline.py 开头设置:")
                print(f"      mixed_precision.Policy('float32')")

            # 保存问题检查点
            error_checkpoint = os.path.join(
                config.CHECKPOINT_DIR,
                f'error_iter_{iteration+1}_nan.weights.h5'
            )
            model.save_weights(error_checkpoint)
            print(f"   已保存错误检查点: {error_checkpoint}")

            # 选择：继续或停止
            print(f"\n   训练将继续，但结果可能不可靠")

        training_time = time.time() - training_start_time
        print(f"  总损失: {avg_loss['loss']:.4f}, "
              f"策略: {avg_loss['policy_output_loss']:.4f}, "
              f"价值: {avg_loss['value_output_loss']:.4f}")
        print(f"  模型训练耗时: {training_time:.1f}秒")

        # 如果启用混合精度，打印损失缩放信息
        if MIXED_PRECISION_ENABLED and hasattr(optimizer, 'loss_scale'):
            current_scale = optimizer.loss_scale
            if hasattr(current_scale, '_current_loss_scale'):
                print(f"  当前损失缩放: {current_scale._current_loss_scale}")
            elif hasattr(current_scale, 'numpy'):
                print(f"  当前损失缩放: {current_scale.numpy()}")

        # 4. 评估
        win_rate = None
        eval_time = 0
        if (iteration + 1) % config.EVAL_FREQUENCY == 0:
            print(f"\n[4/4] 评估模型 vs 随机玩家...")
            eval_start_time = time.time()
            win_rate = evaluate_vs_random(model, config, num_games=config.EVAL_GAMES)
            eval_time = time.time() - eval_start_time
            print(f"  胜率: {win_rate:.3f}")
            print(f"  评估耗时: {eval_time:.1f}秒")

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

        # 完成本次迭代的游戏日志
        game_logger.finish_iteration(iteration + 1)

        # 打印本次迭代总耗时
        iteration_time = time.time() - iteration_start_time
        print(f"\n{'=' * 60}")
        print(f"迭代 {iteration + 1} 总耗时: {iteration_time:.1f}秒 ({iteration_time/60:.1f}分钟)")
        print(f"{'=' * 60}")

        # 保存迭代摘要（繁體中文白話版）
        save_iteration_summary(
            iteration=iteration + 1,
            config=config,
            avg_loss=avg_loss,
            win_rate=win_rate,
            selfplay_time=selfplay_time,
            training_time=training_time,
            eval_time=eval_time,
            iteration_time=iteration_time,
            replay_buffer_size=len(replay_buffer),
            games_metadata=all_games_metadata,
            model=model
        )

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
