"""训练配置文件 - 集中管理所有超参数"""

class TrainingConfig:
    """训练配置类"""

    # ===== 棋盘设置 =====
    BOARD_SIZE = 15

    # ===== 模型架构 =====
    NUM_RES_BLOCKS = 10      # 残差块数量（从3增加到10）
    NUM_FILTERS = 128        # 卷积滤波器数（从64增加到128）
    SE_RATIO = 8             # SE注意力压缩比
    L2_REG = 1e-4           # L2正则化系数
    DROPOUT_RATE = 0.3       # Dropout率
    VALUE_HEAD_HIDDEN = 512  # 价值头隐藏层大小

    # ===== MCTS 设置 =====
    MCTS_SIMULATIONS = 200   # 每步MCTS模拟次数
    C_PUCT = 1.5             # PUCT探索常数
    DIRICHLET_ALPHA = 0.3    # Dirichlet噪声alpha
    DIRICHLET_EPSILON = 0.25 # Dirichlet噪声混合比例

    # ===== 温度控制 =====
    TEMP_THRESHOLD_MOVE = 30 # 前30步使用温度1.0
    TEMP_FINAL_MOVE = 50     # 50步后使用极低温度

    # ===== 训练超参数 =====
    ITERATIONS = 1000        # 总迭代次数
    GAMES_PER_ITERATION = 100  # 每次迭代自我对弈局数
    BATCH_SIZE = 512         # 训练批次大小
    LEARNING_RATE = 0.001    # 初始学习率
    LR_DECAY_STEPS = 200     # 学习率衰减步数
    LR_DECAY_RATE = 0.1      # 学习率衰减率
    GRADIENT_CLIP_NORM = 1.0 # 梯度裁剪范数
    EPOCHS_PER_ITERATION = 5 # 每次迭代训练epoch数

    # ===== 经验回放 =====
    REPLAY_BUFFER_SIZE = 50000  # 回放缓冲区大小
    PRIORITIZED_ALPHA = 0.6     # 优先级回放alpha
    PRIORITIZED_BETA = 0.4      # 优先级回放beta（重要性采样）
    PRIORITIZED_BETA_INCREMENT = 0.001  # beta增长率

    # ===== 价值标签 =====
    VALUE_GAMMA = 0.99       # 价值折扣因子

    # ===== 评估设置 =====
    EVAL_FREQUENCY = 10      # 每N次迭代评估一次
    EVAL_GAMES = 50          # 评估游戏局数
    PROMOTION_THRESHOLD = 0.55  # 模型晋级胜率阈值

    # ===== 并行设置 =====
    NUM_WORKERS = 8          # 自我对弈并行进程数

    # ===== 检查点设置 =====
    CHECKPOINT_FREQUENCY = 50  # 每N次迭代保存检查点
    MAX_CHECKPOINTS = 10     # 保留最近N个检查点

    # ===== 日志设置 =====
    LOG_FREQUENCY = 1        # 每N次迭代记录日志
    TENSORBOARD_DIR = 'logs/tensorboard'
    CHECKPOINT_DIR = 'checkpoints'
    ENABLE_GAME_LOGGING = True  # 是否记录每局游戏详细数据

    # ===== 损失权重 =====
    POLICY_LOSS_WEIGHT = 1.0
    VALUE_LOSS_WEIGHT = 0.5

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
        config.REPLAY_BUFFER_SIZE = 1000
        config.REPLAY_SAMPLE_SIZE = 500
        config.EVAL_FREQUENCY = 2
        config.CHECKPOINT_FREQUENCY = 5
        return config

    @classmethod
    def get_full_config(cls):
        """完整训练配置"""
        return cls()

    def __repr__(self):
        """打印配置摘要"""
        return f"""
TrainingConfig:
  Model: {self.NUM_RES_BLOCKS} ResBlocks, {self.NUM_FILTERS} Filters
  MCTS: {self.MCTS_SIMULATIONS} simulations
  Training: {self.ITERATIONS} iterations, {self.GAMES_PER_ITERATION} games/iter
  Batch: {self.BATCH_SIZE}, LR: {self.LEARNING_RATE}
  Buffer: {self.REPLAY_BUFFER_SIZE}, Sample: {self.REPLAY_SAMPLE_SIZE}
        """.strip()


if __name__ == '__main__':
    # 测试配置
    full_config = TrainingConfig.get_full_config()
    print("=== 完整配置 ===")
    print(full_config)

    print("\n=== 快速测试配置 ===")
    test_config = TrainingConfig.get_fast_test_config()
    print(test_config)
