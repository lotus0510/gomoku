"""PyTorch 訓練流程 - 現代化五子棋 AI 訓練系統"""

import os
import json
import argparse
import time
import random
import traceback
import numpy as np
import multiprocessing
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.cuda.amp import autocast

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
_worker_device = None


def init_worker(model_state_path, config_dict, device_str):
    """
    初始化worker进程（进程池创建时调用一次）
    
    Args:
        model_state_path: 模型状态字典路径
        config_dict: 配置字典
        device_str: 设备字符串 ('cuda' 或 'cpu')
    """
    global _worker_model, _worker_config, _worker_device
    
    # 设置设备
    _worker_device = torch.device(device_str)
    
    # 重建配置
    _worker_config = TrainingConfig()
    for key, value in config_dict.items():
        setattr(_worker_config, key, value)
    
    # 创建模型
    _worker_model, _ = create_enhanced_model(
        board_size=_worker_config.BOARD_SIZE,
        num_res_blocks=_worker_config.NUM_RES_BLOCKS,
        num_filters=_worker_config.NUM_FILTERS,
        se_ratio=_worker_config.SE_RATIO,
        dropout_rate=_worker_config.DROPOUT_RATE,
        value_head_hidden=_worker_config.VALUE_HEAD_HIDDEN,
        device=_worker_device
    )
    
    # 加载权重
    if os.path.exists(model_state_path):
        state_dict = torch.load(model_state_path, map_location=_worker_device, weights_only=True)
        _worker_model.load_state_dict(state_dict)
    
    # 设置为评估模式
    _worker_model.eval()

# ... (skipping to next chunk)




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
    global _worker_model, _worker_config, _worker_device
    
    try:
        start_time = time.time()
        
        print(f"  [进程 {os.getpid()}] 开始第 {game_num + 1} 局...")
        
        # 使用缓存的模型和配置
        model = _worker_model
        config = _worker_config
        device = _worker_device
        
        # 创建MCTS（根据配置选择批量或标准版本）
        use_batched = getattr(config, 'USE_BATCHED_MCTS', False)
        if use_batched:
            mcts = BatchedMCTS(model, config, device=device)
        else:
            mcts = MCTS(model, config, device=device)
        
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
        # 修正：白棋可能是 2 或者是 -1（兼容旧代码）
        white_wins = sum(1 for g in games_metadata if g.get('winner') == 2 or g.get('winner') == -1)
        draws = sum(1 for g in games_metadata if g['winner'] == 0)
        total_games = len(games_metadata)
    else:
        avg_moves = avg_mcts_time = black_wins = white_wins = draws = total_games = 0
    
    # 獲取當前學習率（PyTorch）
    current_lr = model.optimizer.param_groups[0]['lr']
    
    # 計算模型參數量
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # 準備摘要內容
    summary_lines = [
        "=" * 80,
        f"第 {iteration} 次迭代訓練摘要",
        "=" * 80,
        "",
        "【基本指標】",
        f"  • 總損失：{avg_loss['loss']:.4f}",
        f"  • 策略損失：{avg_loss['policy_loss']:.4f}",
        f"  • 價值損失：{avg_loss['value_loss']:.4f}",
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


def evaluate_vs_random(model, config, device, num_games=20):
    """
    评估vs随机玩家
    
    Args:
        model: PyTorch 模型
        config: 训练配置
        device: 设备
        num_games: 评估游戏局数
    
    Returns:
        win_rate: 胜率
    """
    class RandomPlayer:
        def get_action(self, state):
            legal_moves = state.get_legal_moves()
            return random.choice(legal_moves) if legal_moves else None
    
    class ModelPlayer:
        def __init__(self, model, board_size, device):
            self.model = model
            self.board_size = board_size
            self.device = device
        
        def get_action(self, state):
            # 准备输入 (H, W, C) -> (1, C, H, W)
            nn_input = state.to_input()
            nn_input = torch.from_numpy(nn_input).float().permute(2, 0, 1).unsqueeze(0).to(self.device)
            
            # 推理
            self.model.eval()
            with torch.no_grad():
                policy, _ = self.model(nn_input)
            
            # 转换为 NumPy
            policy = policy.cpu().numpy()[0]
            
            legal_moves = state.get_legal_moves()
            if not legal_moves:
                return None
            
            # Mask非法移动
            legal_indices = [r * self.board_size + c for r, c in legal_moves]
            masked_policy = np.zeros_like(policy)
            masked_policy[legal_indices] = policy[legal_indices]
            
            if masked_policy.sum() > 0:
                move_idx = np.argmax(masked_policy)
            else:
                move_idx = random.choice(legal_indices)
            
            row, col = divmod(move_idx, self.board_size)
            return (row, col)
    
    arena = Arena(board_size=config.BOARD_SIZE)
    model_player = ModelPlayer(model, config.BOARD_SIZE, device)
    random_player = RandomPlayer()
    
    results = arena.compete(model_player, random_player, num_games=num_games)
    
    return results['player1_win_rate']




def augment_single_data(data_tuple):
    """对单个数据进行8种对称变换"""
    state, policy, value = data_tuple
    board_size = int(np.sqrt(len(policy)))  # 从 policy 推断棋盘大小
    symmetries = get_symmetries(state, policy, board_size)
    return [(aug_state, aug_policy, value) for aug_state, aug_policy in symmetries]


def train(config, resume_from=None):
    """
    主训练函数
    
    Args:
        config: TrainingConfig对象
        resume_from: 恢复训练的检查点路径（可选）
    """
    print("=" * 60)
    print("PyTorch 現代化深度學習五子棋訓練系統")
    print("=" * 60)
    print(f"\n配置:\n{config}\n")
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用設備: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"混合精度訓練: 已啟用")
    else:
        print(f"混合精度訓練: 已禁用 (CPU 模式)")
    
    # 创建目录
    os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)
    
    # 创建模型
    print("\n創建模型...")
    model, l2_reg = create_enhanced_model(
        board_size=config.BOARD_SIZE,
        num_res_blocks=config.NUM_RES_BLOCKS,
        num_filters=config.NUM_FILTERS,
        se_ratio=config.SE_RATIO,
        dropout_rate=config.DROPOUT_RATE,
        value_head_hidden=config.VALUE_HEAD_HIDDEN,
        device=device
    )
    
    # 创建优化器（使用 AdamW，内置 L2 正则化）
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config.LEARNING_RATE,
        weight_decay=l2_reg
    )
    
    # 创建学习率调度器
    scheduler = optim.lr_scheduler.StepLR(
        optimizer,
        step_size=config.LR_DECAY_STEPS,
        gamma=config.LR_DECAY_RATE
    )
    
    # 创建混合精度训练的 GradScaler（仅 GPU）
    use_amp = torch.cuda.is_available()
    scaler = torch.amp.GradScaler('cuda') if use_amp else None
    
    # 将优化器附加到模型（用于 save_iteration_summary）
    model.optimizer = optimizer
    
    # 加载检查点
    start_iteration = 0
    model_state_path = os.path.join(config.CHECKPOINT_DIR, 'latest_model.pth')
    
    if resume_from and os.path.exists(resume_from):
        print(f"從檢查點恢復訓練: {resume_from}")
        checkpoint = torch.load(resume_from, map_location=device, weights_only=True)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        if use_amp and 'scaler_state_dict' in checkpoint:
            scaler.load_state_dict(checkpoint['scaler_state_dict'])
        start_iteration = checkpoint.get('iteration', 0)
        print(f"從第 {start_iteration + 1} 次迭代繼續")
    elif os.path.exists(model_state_path):
        print(f"載入已有模型: {model_state_path}")
        model.load_state_dict(torch.load(model_state_path, map_location=device, weights_only=True))
    
    # 创建经验回放缓冲区
    replay_buffer = PrioritizedReplayBuffer(
        capacity=config.REPLAY_BUFFER_SIZE,
        alpha=config.PRIORITIZED_ALPHA,
        beta=config.PRIORITIZED_BETA
    )
    
    # 创建游戏日志记录器
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
        print(f"\n[1/4] 自我對弈 {config.GAMES_PER_ITERATION} 局（{config.NUM_WORKERS}個進程）...")
        selfplay_start_time = time.time()
        
        # 保存当前模型供worker使用
        torch.save(model.state_dict(), model_state_path)
        
        # 准备参数
        config_dict = {k: v for k, v in config.__dict__.items() if not k.startswith('_')}
        game_numbers = list(range(config.GAMES_PER_ITERATION))
        device_str = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # 多进程自我对弈
        with multiprocessing.Pool(
            processes=config.NUM_WORKERS,
            initializer=init_worker,
            initargs=(model_state_path, config_dict, device_str)
        ) as pool:
            results = pool.map(run_self_play_game_worker, game_numbers)
        
        # 收集训练数据和游戏元数据
        all_games_data = []
        all_games_metadata = []
        for training_data, game_metadata in results:
            all_games_data.extend(training_data)
            if game_metadata:
                all_games_metadata.append(game_metadata)
        
        selfplay_time = time.time() - selfplay_start_time
        print(f"  收集到 {len(all_games_data)} 步訓練數據，{len(all_games_metadata)} 局遊戲")
        print(f"  自我對弈耗時: {selfplay_time:.1f}秒 ({selfplay_time/60:.1f}分鐘)")
        
        # 记录每局游戏到日志
        for game_num, metadata in enumerate(all_games_metadata):
            game_logger.log_game(iteration + 1, game_num, metadata)
        
        # 2. 数据增强
        print(f"\n[2/4] 數據增強（8種對稱變換，並行處理）...")
        augment_start_time = time.time()
        
        # 使用多进程并行处理
        num_aug_workers = max(1, config.NUM_WORKERS // 2)
        with multiprocessing.Pool(processes=num_aug_workers) as pool:
            augmented_results = pool.map(augment_single_data, all_games_data)
        
        # 展平结果
        augmented_data = [item for sublist in augmented_results for item in sublist]
        
        augment_time = time.time() - augment_start_time
        print(f"  增強後數據: {len(all_games_data)} -> {len(augmented_data)} (8x)")
        print(f"  數據增強耗時: {augment_time:.1f}秒（使用{num_aug_workers}個進程）")
        
        # 添加到回放缓冲区
        for data in augmented_data:
            replay_buffer.add(data)
        
        print(f"  緩衝區大小: {len(replay_buffer)}/{config.REPLAY_BUFFER_SIZE}")
        
        # 3. 训练模型
        print(f"\n[3/4] 訓練模型（{config.EPOCHS_PER_ITERATION} epochs）...")
        training_start_time = time.time()
        
        if len(replay_buffer) < config.BATCH_SIZE:
            print(f"  緩衝區數據不足，跳過訓練")
            continue
        
        model.train()
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
            X = np.array([s[0] for s in samples])  # (batch, H, W, C)
            y_policy = np.array([s[1] for s in samples])
            y_value = np.array([s[2] for s in samples])
            
            # 转换为 PyTorch Tensor 并调整维度 (B, H, W, C) -> (B, C, H, W)
            X = torch.from_numpy(X).float().permute(0, 3, 1, 2).to(device)
            y_policy = torch.from_numpy(y_policy).float().to(device)
            y_value = torch.from_numpy(y_value).float().to(device)
            weights_tensor = torch.from_numpy(weights).float().to(device)
            
            # 前向传播（使用混合精度）
            optimizer.zero_grad()
            
            if use_amp:
                with autocast():
                    policy_pred, value_pred = model(X)
                    
                    # 计算损失
                    policy_loss = F.cross_entropy(
                        policy_pred, 
                        y_policy, 
                        reduction='none'
                    )
                    value_loss = F.huber_loss(
                        value_pred.squeeze(), 
                        y_value, 
                        reduction='none',
                        delta=1.0
                    )
                    
                    # 加权损失
                    loss = (
                        policy_loss * weights_tensor * config.POLICY_LOSS_WEIGHT +
                        value_loss * weights_tensor * config.VALUE_LOSS_WEIGHT
                    ).mean()
                
                # 反向传播
                scaler.scale(loss).backward()
                
                # 梯度裁剪
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.GRADIENT_CLIP_NORM)
                
                # 优化器步骤
                scaler.step(optimizer)
                scaler.update()
            else:
                # CPU 模式（无混合精度）
                policy_pred, value_pred = model(X)
                
                # 计算损失
                policy_loss = F.cross_entropy(
                    policy_pred, 
                    y_policy, 
                    reduction='none'
                )
                value_loss = F.huber_loss(
                    value_pred.squeeze(), 
                    y_value, 
                    reduction='none',
                    delta=1.0
                )
                
                # 加权损失
                loss = (
                    policy_loss * weights_tensor * config.POLICY_LOSS_WEIGHT +
                    value_loss * weights_tensor * config.VALUE_LOSS_WEIGHT
                ).mean()
                
                # 反向传播
                loss.backward()
                
                # 梯度裁剪
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.GRADIENT_CLIP_NORM)
                
                # 优化器步骤
                optimizer.step()
            
            # 记录损失
            epoch_losses.append({
                'loss': loss.item(),
                'policy_loss': policy_loss.mean().item(),
                'value_loss': value_loss.mean().item()
            })
            
            # 保存最后一个批次用于更新优先级
            last_indices = indices
            last_X = X
            last_y_value = y_value
        
        # 学习率调度
        scheduler.step()
        
        # 更新优先级：使用最后一个批次计算 TD-error
        if last_indices is not None and len(last_indices) > 0:
            model.eval()
            with torch.no_grad():
                _, pred_values = model(last_X)
                pred_values = pred_values.cpu().numpy().flatten()
            
            # 计算 TD-error
            last_y_value_np = last_y_value.cpu().numpy()
            td_errors = np.abs(pred_values - last_y_value_np)
            
            # 更新优先级
            replay_buffer.update_priorities(last_indices, td_errors)
        
        # 计算平均损失
        avg_loss = {
            'loss': np.mean([l['loss'] for l in epoch_losses]),
            'policy_loss': np.mean([l['policy_loss'] for l in epoch_losses]),
            'value_loss': np.mean([l['value_loss'] for l in epoch_losses])
        }
        
        # NaN 检测
        if np.isnan(avg_loss['loss']) or np.isinf(avg_loss['loss']):
            print(f"\n🔴 檢測到 NaN/Inf 損失！")
            print(f"   總損失: {avg_loss['loss']}")
            print(f"   策略損失: {avg_loss['policy_loss']}")
            print(f"   價值損失: {avg_loss['value_loss']}")
            
            # 保存错误检查点
            error_checkpoint = os.path.join(
                config.CHECKPOINT_DIR,
                f'error_iter_{iteration+1}_nan.pth'
            )
            torch.save({
                'iteration': iteration,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()
            }, error_checkpoint)
            print(f"   已保存錯誤檢查點: {error_checkpoint}")
            print(f"\n   訓練將繼續，但結果可能不可靠")
        
        training_time = time.time() - training_start_time
        print(f"  總損失: {avg_loss['loss']:.4f}, "
              f"策略: {avg_loss['policy_loss']:.4f}, "
              f"價值: {avg_loss['value_loss']:.4f}")
        print(f"  模型訓練耗時: {training_time:.1f}秒")
        
        # 4. 评估
        win_rate = None
        eval_time = 0
        if (iteration + 1) % config.EVAL_FREQUENCY == 0:
            print(f"\n[4/4] 評估模型 vs 隨機玩家...")
            eval_start_time = time.time()
            win_rate = evaluate_vs_random(model, config, device, num_games=config.EVAL_GAMES)
            eval_time = time.time() - eval_start_time
            print(f"  勝率: {win_rate:.3f}")
            print(f"  評估耗時: {eval_time:.1f}秒")
        
        # 记录历史
        history['iterations'].append(iteration + 1)
        history['total_loss'].append(avg_loss['loss'])
        history['policy_loss'].append(avg_loss['policy_loss'])
        history['value_loss'].append(avg_loss['value_loss'])
        history['win_rate_vs_random'].append(win_rate)
        
        # 保存检查点
        if (iteration + 1) % config.CHECKPOINT_FREQUENCY == 0:
            checkpoint_path = os.path.join(
                config.CHECKPOINT_DIR,
                f'checkpoint_iter_{iteration+1}.pth'
            )
            torch.save({
                'iteration': iteration + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'scaler_state_dict': scaler.state_dict() if scaler else None,
                'history': history,
                'config': config_dict
            }, checkpoint_path)
            print(f"  保存檢查點: {checkpoint_path}")
        
        # 保存最新模型
        torch.save(model.state_dict(), model_state_path)
        
        # 保存训练历史
        history_path = os.path.join(config.CHECKPOINT_DIR, 'training_history.json')
        
        def convert_to_native(obj):
            """转换为 Python 原生类型"""
            if isinstance(obj, (np.ndarray, torch.Tensor)):
                return obj.tolist() if isinstance(obj, np.ndarray) else obj.cpu().numpy().tolist()
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
        print(f"迭代 {iteration + 1} 總耗時: {iteration_time:.1f}秒 ({iteration_time/60:.1f}分鐘)")
        print(f"{'=' * 60}")
        
        # 保存迭代摘要
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
    print("訓練完成！")
    print(f"{'=' * 60}")
    print(f"最終模型: {model_state_path}")
    print(f"訓練歷史: {history_path}")


if __name__ == '__main__':
    multiprocessing.freeze_support()
    
    parser = argparse.ArgumentParser(description='PyTorch 現代化五子棋AI訓練')
    parser.add_argument('--fast-test', '-f', action='store_true', help='使用快速測試配置')
    parser.add_argument('--iterations', '-i', type=int, help='訓練迭代次數')
    parser.add_argument('--games', '-g', type=int, help='每次迭代遊戲局數')
    parser.add_argument('--resume', '-r', type=str, help='恢復訓練的檢查點路徑')
    
    args = parser.parse_args()
    
    # 选择配置
    if args.fast_test:
        config = TrainingConfig.get_fast_test_config()
        print("使用快速測試配置")
    else:
        config = TrainingConfig.get_full_config()
        print("使用完整訓練配置")
    
    # 覆盖配置
    if args.iterations:
        config.ITERATIONS = args.iterations
    if args.games:
        config.GAMES_PER_ITERATION = args.games
    
    # 开始训练
    train(config, resume_from=args.resume)
