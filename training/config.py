"""训练配置文件 - 集中管理所有超参数"""

class TrainingConfig:
    """训练配置类"""

    # ===== 棋盘设置 =====
    BOARD_SIZE = 15

    # ===== 模型架构 =====
    NUM_RES_BLOCKS = 10      # 残差块数量（从3增加到10）
    NUM_FILTERS = 128        # 卷积滤波器数（从64增加到128）
    # 注意：優化配置會調整為 6 ResBlocks, 96 Filters 以適合 CPU 推理
    SE_RATIO = 8             # SE注意力压缩比
    L2_REG = 1e-4           # L2正则化系数
    DROPOUT_RATE = 0.3       # Dropout率
    VALUE_HEAD_HIDDEN = 512  # 价值头隐藏层大小

    # ===== MCTS 设置 =====
    MCTS_SIMULATIONS = 100   # 每步MCTS模拟次数（CPU推理用，从200降至100）
    MCTS_BATCH_SIZE = 8      # 批量推理大小（批量MCTS专用）
    C_PUCT = 2.0             # PUCT探索常数（从1.5→2.0，增强探索）
    DIRICHLET_ALPHA = 0.3    # Dirichlet噪声alpha
    DIRICHLET_EPSILON = 0.35 # Dirichlet噪声混合比例（从0.25→0.35，+40%探索性）
    USE_BATCHED_MCTS = True  # 是否使用批量MCTS（显著提升性能）

    # ===== 温度控制 =====
    TEMP_THRESHOLD_MOVE = 30 # 前30步使用温度1.0（从25→30，延长探索期）
    TEMP_FINAL_MOVE = 60     # 60步后使用极低温度（从50→60）

    # ===== 训练超参数 =====
    ITERATIONS = 1000        # 总迭代次数
    GAMES_PER_ITERATION = 100  # 每次迭代自我对弈局数
    BATCH_SIZE = 512         # 训练批次大小
    LEARNING_RATE = 0.0001   # 初始学习率（适中，AlphaZero标准）
    MIN_LEARNING_RATE = 1e-5 # 最低学习率下限（防止梯度消失）
    LR_DECAY_STEPS = 100     # 学习率衰减步数（从400→100，更频繁衰减）
    LR_DECAY_RATE = 0.8      # 学习率衰减率（从0.7→0.8，更温和衰减）
    GRADIENT_CLIP_NORM = 1.0 # 梯度裁剪范数
    GRADIENT_MIN_THRESHOLD = 0.1  # 梯度过小警告阈值（<0.1视为梯度消失）
    EPOCHS_PER_ITERATION = 5 # 每次迭代训练epoch数

    # ===== 经验回放 =====
    REPLAY_BUFFER_SIZE = 400000  # 回放缓冲区大小 (约保存10次迭代的数据)
    PRIORITIZED_ALPHA = 0.6     # 优先级回放alpha
    PRIORITIZED_BETA = 0.4      # 优先级回放beta（重要性采样）
    PRIORITIZED_BETA_INCREMENT = 0.001  # beta增长率

    # ===== 价值标签 =====
    VALUE_GAMMA = 0.995      # 价值折扣因子（从1.0→0.995，中等偏好效率，18.2%差距）

    # ===== 评估设置 =====
    EVAL_FREQUENCY = 5       # 每N次迭代评估一次（更频繁监控）
    EVAL_GAMES = 50          # 评估游戏局数
    PROMOTION_THRESHOLD = 0.55  # 模型晋级胜率阈值

    # ===== 并行设置 =====
    NUM_WORKERS = 8          # 自我对弈并行进程数 (默认配置)

    # ===== 检查点设置 =====
    CHECKPOINT_FREQUENCY = 1   # 每N次迭代保存检查点（从50→5，更頻繁備份）
    MAX_CHECKPOINTS = 1000       # 保留最近N个检查点（从10→20）
    SAVE_BEST_MODEL = True     # 自動保存最佳模型（基於勝率）
    BEST_MODEL_METRIC = 'win_rate'  # 最佳模型評判標準（win_rate/loss）

    # ===== 日志设置 =====
    LOG_FREQUENCY = 1        # 每N次迭代记录日志
    TENSORBOARD_DIR = 'logs/tensorboard'
    CHECKPOINT_DIR = 'checkpoints'
    ENABLE_GAME_LOGGING = True  # 是否记录每局游戏详细数据
    DETAILED_GAME_LOG_FREQUENCY = 10  # 每N局保存详细游戏数据（棋谱、策略）

    # ===== 损失权重 =====
    POLICY_LOSS_WEIGHT = 1.0
    VALUE_LOSS_WEIGHT = 1.0  # 提升价值损失权重（原0.5太低）

    @classmethod
    def get_fast_test_config(cls):
        """快速测试配置（用于验证代码）"""
        config = cls()
        config.NUM_RES_BLOCKS = 3
        config.NUM_FILTERS = 64
        config.MCTS_SIMULATIONS = 50
        config.ITERATIONS = 5
        config.GAMES_PER_ITERATION = 10
        config.BATCH_SIZE = 64
        config.REPLAY_BUFFER_SIZE = 20000  # 增加缓冲区大小，避免数据溢出 (7000+ per iter)
        config.REPLAY_SAMPLE_SIZE = 500
        config.EVAL_FREQUENCY = 2
        config.CHECKPOINT_FREQUENCY = 1
        return config

    @classmethod
    def get_full_config(cls):
        """完整训练配置（原始配置，已知問題：探索過高）"""
        return cls()

    @classmethod
    def get_optimized_config(cls):
        """
        優化配置 - 解決當前訓練問題

        基於 226 次迭代的診斷結果：
        - 策略損失 5.28 (隨機 5.42，僅好 2.6%)
        - 價值標準差 0.11 (目標 0.5-0.7，崩潰)
        - 梯度範數 0.38 (健康 0.5-5.0，消失)

        核心問題：
        1. 探索參數過高 (35%) → 訓練目標太隨機
        2. 學習率過早衰減 → 無法適應新數據
        3. 溫度控制過於探索 → 策略不確定

        優化策略：
        1. 大幅降低探索 (35% → 15%)
        2. 延後學習率衰減 (100 → 200)
        3. 收緊溫度控制
        4. 增加緩衝區容量
        """
        config = cls()

        # ===== 核心優化 =====

        # 1. 【關鍵修改】大幅降低探索噪聲
        config.DIRICHLET_EPSILON = 0.15  # 原 0.35 → 0.15 (-57%)
        # 預期效果：
        # - 訓練目標更明確（85% 來自神經網路，15% 隨機）
        # - 策略損失應在 20-30 次迭代降到 < 5.0
        # - Top-1 機率從 0.008 提升到 0.1-0.2

        # 2. 延後學習率衰減
        config.LR_DECAY_STEPS = 200  # 原 100 → 200 (+100%)
        # 預期效果：
        # - 前 200 次迭代保持 LR = 0.0001
        # - 網路有足夠時間學習基本策略
        # - 避免過早進入微調階段

        # 3. 收緊溫度控制（減少前期探索）
        config.TEMP_THRESHOLD_MOVE = 20  # 原 30 → 20 (-33%)
        config.TEMP_FINAL_MOVE = 40      # 原 60 → 40 (-33%)
        # 預期效果：
        # - 前 20 步探索，20-40 步過渡，40+ 步確定
        # - 遊戲更快進入確定性階段
        # - 減少無意義的後期探索

        # 4. 增加緩衝區容量
        config.REPLAY_BUFFER_SIZE = 800000  # 原 400,000 → 800,000 (+100%)
        # 預期效果：
        # - 保存 ~40 次迭代的數據（原 ~20 次）
        # - 更好的樣本多樣性
        # - 訓練更穩定，減少過擬合

        # 5. 調整 C_PUCT（平衡探索與利用）
        config.C_PUCT = 1.5  # 原 2.0 → 1.5 (-25%)
        # 預期效果：
        # - MCTS 更傾向選擇高價值節點
        # - 減少過度探索未知節點
        # - 配合低 DIRICHLET_EPSILON 使用

        # ===== 次要優化 =====

        # 6. 增加每次迭代的訓練強度
        config.EPOCHS_PER_ITERATION = 10  # 原 5 → 10 (+100%)
        # 預期效果：
        # - 每次迭代更充分地學習數據
        # - 加快收斂速度
        # - 配合大緩衝區使用

        # 7. 調整訓練批次大小（針對 8GB VRAM）
        config.BATCH_SIZE = 768  # 折衷方案：512→768（1024 會 OOM）
        # 預期效果：
        # - 更穩定的梯度估計（比 512 略好）
        # - 充分利用 8GB VRAM（約 5-6GB）
        # - 如果 OOM 請降回 512

        # 8. 調整梯度裁剪
        config.GRADIENT_CLIP_NORM = 5.0  # 原 1.0 → 5.0
        # 預期效果：
        # - 允許更大的梯度更新
        # - 幫助價值網路從崩潰狀態恢復
        # - 配合增大的批次使用

        # ===== 硬件優化 (方案B: 激進配置) =====
        # 針對 8GB VRAM + 16GB RAM + 14核20緒 CPU

        # 9. 平衡 GPU 記憶體與 CPU 多核心
        config.NUM_WORKERS = 10  # 原 8 → 10（平衡性能與記憶體）
        # 預期效果：
        # - 10 個並行 workers 使用 GPU 推理
        # - 8GB VRAM 使用率約 80-85%（每個 worker 約 0.8-1GB）
        # - CPU 利用率適中（10/20 線程）

        # 10. 增加每次迭代遊戲數
        config.GAMES_PER_ITERATION = 100  # 原 100 → 150 (+50%)
        # 預期效果：
        # - 配合 16 workers 使用
        # - 更多樣化的訓練數據
        # - 每次迭代收集更多經驗

        return config

    def __repr__(self):
        """打印配置摘要"""
        return f"""
TrainingConfig:
  Model: {self.NUM_RES_BLOCKS} ResBlocks, {self.NUM_FILTERS} Filters
  MCTS: {self.MCTS_SIMULATIONS} simulations
  Training: {self.ITERATIONS} iterations, {self.GAMES_PER_ITERATION} games/iter
  Batch: {self.BATCH_SIZE}, LR: {self.LEARNING_RATE}
  Buffer: {self.REPLAY_BUFFER_SIZE}
        """.strip()


if __name__ == '__main__':
    # 测试配置
    print("=" * 80)
    print("配置對比")
    print("=" * 80)

    full_config = TrainingConfig.get_full_config()
    opt_config = TrainingConfig.get_optimized_config()

    print("\n關鍵參數對比：")
    print(f"{'參數':<30} {'原始配置':<15} {'優化配置':<15} {'變化':<15}")
    print("-" * 80)

    comparisons = [
        ("DIRICHLET_EPSILON", full_config.DIRICHLET_EPSILON, opt_config.DIRICHLET_EPSILON),
        ("LR_DECAY_STEPS", full_config.LR_DECAY_STEPS, opt_config.LR_DECAY_STEPS),
        ("TEMP_THRESHOLD_MOVE", full_config.TEMP_THRESHOLD_MOVE, opt_config.TEMP_THRESHOLD_MOVE),
        ("TEMP_FINAL_MOVE", full_config.TEMP_FINAL_MOVE, opt_config.TEMP_FINAL_MOVE),
        ("REPLAY_BUFFER_SIZE", full_config.REPLAY_BUFFER_SIZE, opt_config.REPLAY_BUFFER_SIZE),
        ("C_PUCT", full_config.C_PUCT, opt_config.C_PUCT),
        ("EPOCHS_PER_ITERATION", full_config.EPOCHS_PER_ITERATION, opt_config.EPOCHS_PER_ITERATION),
        ("BATCH_SIZE", full_config.BATCH_SIZE, opt_config.BATCH_SIZE),
        ("GRADIENT_CLIP_NORM", full_config.GRADIENT_CLIP_NORM, opt_config.GRADIENT_CLIP_NORM),
    ]

    for name, old_val, new_val in comparisons:
        if isinstance(old_val, float):
            change = f"{(new_val/old_val - 1)*100:+.1f}%"
        else:
            change = f"{(new_val/old_val - 1)*100:+.1f}%"
        print(f"{name:<30} {old_val:<15} {new_val:<15} {change:<15}")

    print("\n" + "=" * 80)
    print("使用方法：")
    print("=" * 80)
    print("\n在 train_pipeline_pytorch.py 中使用優化配置：")
    print("  config = TrainingConfig.get_optimized_config()")
    print("\n或在命令行中：")
    print("  python train_pipeline_pytorch.py --optimized")
    print("\n" + "=" * 80)
