"""MCTS - 蒙特卡罗树搜索算法 (PyTorch 版本)"""

import numpy as np
import math
import torch
from core.game_state import GameState


class MCTSNode:
    """MCTS树节点"""

    def __init__(self, prior_prob=0.0):
        """
        初始化节点

        Args:
            prior_prob: 先验概率 P(s,a) 来自神经网络
        """
        self.visit_count = 0        # N(s,a) - 访问次数
        self.total_value = 0.0      # W(s,a) - 累计价值
        self.prior_prob = prior_prob # P(s,a) - 先验概率
        self.children = {}          # 子节点字典 {action: MCTSNode}

    def get_value(self):
        """获取节点平均价值 Q(s,a) = W(s,a) / N(s,a)"""
        if self.visit_count == 0:
            return 0.0
        return self.total_value / (self.visit_count + 1e-8)

    def is_leaf(self):
        """是否为叶节点"""
        return len(self.children) == 0

    def __repr__(self):
        return f"MCTSNode(N={self.visit_count}, W={self.total_value:.2f}, P={self.prior_prob:.3f})"


class MCTS:
    """蒙特卡罗树搜索"""

    def __init__(self, model, config, device='cuda' if torch.cuda.is_available() else 'cpu'):
        """
        初始化MCTS

        Args:
            model: PyTorch 神经网络模型
            config: 配置对象
            device: 设备 ('cuda' 或 'cpu')
        """
        self.model = model
        self.device = device
        self.c_puct = config.C_PUCT
        self.num_simulations = config.MCTS_SIMULATIONS
        self.dirichlet_alpha = config.DIRICHLET_ALPHA
        self.dirichlet_epsilon = config.DIRICHLET_EPSILON
        self.board_size = config.BOARD_SIZE

    def search(self, state, add_noise=True):
        """
        执行MCTS搜索

        Args:
            state: GameState 游戏状态
            add_noise: 是否添加Dirichlet噪声（训练时True，评估时False）

        Returns:
            (action_probs, value) tuple
            - action_probs: (board_size^2,) 改进的策略分布
            - value: 根节点价值估计
        """
        root = MCTSNode(prior_prob=1.0)

        # 扩展根节点
        self._expand(root, state)

        # 根节点添加Dirichlet噪声增加探索
        if add_noise:
            self._add_dirichlet_noise(root, state)

        # 执行N次模拟
        for _ in range(self.num_simulations):
            # 克隆状态用于模拟
            sim_state = state.clone()

            # Selection + Expansion + Backup
            self._simulate(root, sim_state)

        # 收集访问次数分布
        action_probs = self._get_action_probs(root, state)

        # 根节点价值估计
        root_value = root.get_value()

        return action_probs, root_value

    def _simulate(self, node, state):
        """
        单次MCTS模拟

        Args:
            node: 当前节点
            state: 当前游戏状态
        """
        # 1. Selection: 递归选择子节点直到叶节点
        path = []  # 记录路径用于反向传播
        current_node = node

        while not current_node.is_leaf() and not state.is_game_over():
            # 选择最佳子节点
            action, child_node = self._select_child(current_node, state)
            path.append((current_node, child_node))

            # 执行动作
            state.make_move(*action)
            current_node = child_node

        # 2. Expansion & Evaluation
        if state.is_game_over():
            # 游戏结束，使用真实结果
            winner = state.get_winner()
            current_player = state.get_current_player()

            if winner == 0:
                value = 0.0  # 平局
            elif winner == current_player:
                value = 1.0  # 当前玩家赢
            else:
                value = -1.0  # 当前玩家输
        else:
            # 叶节点：扩展并评估
            value = self._expand(current_node, state)

        # 3. Backup: 反向传播价值
        self._backup(path, value)

    def _select_child(self, node, state):
        """
        选择PUCT值最大的子节点

        Args:
            node: 父节点
            state: 当前状态

        Returns:
            (action, child_node) tuple
        """
        best_score = -float('inf')
        best_action = None
        best_child = None

        legal_moves = state.get_legal_moves()
        parent_visits = node.visit_count

        for action in legal_moves:
            action_key = action  # (row, col)
            if action_key in node.children:
                child = node.children[action_key]
                score = self._puct_score(child, parent_visits)

                if score > best_score:
                    best_score = score
                    best_action = action
                    best_child = child

        return best_action, best_child

    def _puct_score(self, node, parent_visits):
        """
        计算PUCT分数

        PUCT = Q(s,a) + c_puct * P(s,a) * sqrt(N(s)) / (1 + N(s,a))

        Args:
            node: 子节点
            parent_visits: 父节点访问次数

        Returns:
            PUCT分数
        """
        # Q值：平均价值
        q_value = node.get_value()

        # U值：探索奖励
        u_value = (
            self.c_puct *
            node.prior_prob *
            math.sqrt(parent_visits) /
            (1 + node.visit_count)
        )

        return q_value + u_value

    def _expand(self, node, state):
        """
        扩展节点：调用神经网络评估

        Args:
            node: 要扩展的节点
            state: 当前状态

        Returns:
            价值估计
        """
        # 准备神经网络输入
        nn_input = state.to_input()
        nn_input = np.expand_dims(nn_input, axis=0)  # 添加批次维度

        # 转换为 PyTorch 格式 (NHWC -> NCHW)
        nn_input = np.transpose(nn_input, (0, 3, 1, 2))

        # 转换为 PyTorch 张量
        nn_input = torch.from_numpy(nn_input).float().to(self.device)

        # 神经网络推理（无梯度）
        self.model.eval()
        with torch.no_grad():
            policy, value = self.model(nn_input)

        # 转换回 NumPy
        policy = policy.cpu().numpy()[0]  # 移除批次维度
        value = value.cpu().numpy()[0, 0]

        # 只对合法移动创建子节点
        legal_moves = state.get_legal_moves()
        legal_moves_indices = [r * self.board_size + c for r, c in legal_moves]

        # Mask非法移动
        masked_policy = np.zeros_like(policy)
        masked_policy[legal_moves_indices] = policy[legal_moves_indices]

        # 重新归一化
        if masked_policy.sum() > 0:
            masked_policy /= masked_policy.sum()
        else:
            # 如果所有概率都是0，使用均匀分布
            masked_policy[legal_moves_indices] = 1.0 / len(legal_moves_indices)

        # 创建子节点
        for move in legal_moves:
            row, col = move
            move_index = row * self.board_size + col
            prior_prob = masked_policy[move_index]

            node.children[move] = MCTSNode(prior_prob=prior_prob)

        return value

    def _add_dirichlet_noise(self, root, state):
        """
        为根节点添加Dirichlet噪声以增加探索

        Args:
            root: 根节点
            state: 根状态
        """
        legal_moves = state.get_legal_moves()
        noise = np.random.dirichlet([self.dirichlet_alpha] * len(legal_moves))

        for i, move in enumerate(legal_moves):
            if move in root.children:
                child = root.children[move]
                # 混合先验概率和噪声
                child.prior_prob = (
                    (1 - self.dirichlet_epsilon) * child.prior_prob +
                    self.dirichlet_epsilon * noise[i]
                )

    def _backup(self, path, value):
        """
        反向传播价值

        Args:
            path: [(parent, child), ...] 路径
            value: 叶节点价值
        """
        # 从叶节点向根节点传播
        for parent, child in reversed(path):
            child.visit_count += 1
            child.total_value += value
            value = -value  # 价值在对手视角下取反

        # 更新根节点（根节点不在path的child中，需要单独更新）
        if path:
            root = path[0][0]  # path的第一个parent就是root
            root.visit_count += 1
            root.total_value += value

    def _get_action_probs(self, root, state):
        """
        从MCTS树中提取改进的策略分布

        Args:
            root: 根节点
            state: 根状态

        Returns:
            (board_size^2,) 策略向量
        """
        action_probs = np.zeros(self.board_size * self.board_size, dtype=np.float32)

        # 收集所有合法移动的访问次数
        visits = []
        moves = []

        for move, child in root.children.items():
            row, col = move
            move_index = row * self.board_size + col
            visits.append(child.visit_count)
            moves.append(move_index)

        visits = np.array(visits, dtype=np.float32)

        # 将访问次数转换为概率分布
        if visits.sum() > 0:
            probs = visits / visits.sum()
            for i, move_index in enumerate(moves):
                action_probs[move_index] = probs[i]

        return action_probs

    def get_action_with_temperature(self, action_probs, temperature=1.0):
        """
        使用温度参数采样动作

        Args:
            action_probs: (board_size^2,) 策略分布
            temperature: 温度参数
                - temperature=1.0: 按概率采样（探索）
                - temperature→0: 选择最大概率（贪婪）

        Returns:
            (row, col) 选择的动作
        """
        if temperature == 0:
            # 贪婪选择
            move_index = np.argmax(action_probs)
        else:
            # 温度采样
            # 应用温度
            powered_probs = action_probs ** (1.0 / temperature)

            # 重新归一化
            if powered_probs.sum() > 0:
                powered_probs /= powered_probs.sum()
            else:
                # 如果全为0，均匀采样
                powered_probs = (action_probs > 0).astype(float)
                powered_probs /= powered_probs.sum()

            # 采样
            move_index = np.random.choice(
                len(action_probs),
                p=powered_probs
            )

        row = move_index // self.board_size
        col = move_index % self.board_size

        return (row, col)


if __name__ == '__main__':
    from training.config import TrainingConfig
    from core.neural_net import create_enhanced_model
    import tensorflow as tf

    print("=== 测试MCTS算法 ===\n")

    # 创建配置
    config = TrainingConfig.get_fast_test_config()
    config.MCTS_SIMULATIONS = 50  # 快速测试用50次

    # 创建模型
    print("创建模型...")
    model = create_enhanced_model(
        board_size=config.BOARD_SIZE,
        num_res_blocks=config.NUM_RES_BLOCKS,
        num_filters=config.NUM_FILTERS
    )

    # 创建MCTS
    print(f"创建MCTS（{config.MCTS_SIMULATIONS}次模拟）...")
    mcts = MCTS(model, config)

    # 创建测试状态
    state = GameState(board_size=15)

    # 在中心下几步棋
    state.make_move(7, 7)
    state.make_move(7, 8)
    state.make_move(8, 7)

    print(f"\n当前状态: {state}")
    print(f"合法移动数: {len(state.get_legal_moves())}")

    # 执行MCTS搜索
    print(f"\n执行MCTS搜索...")
    action_probs, value = mcts.search(state, add_noise=True)

    print(f"  搜索完成")
    print(f"  根节点价值估计: {value:.4f}")
    print(f"  策略向量和: {action_probs.sum():.6f}")

    # 找到top5移动
    top5_indices = np.argsort(action_probs)[-5:][::-1]
    print(f"\n  Top 5 移动:")
    for i, idx in enumerate(top5_indices):
        row, col = idx // 15, idx % 15
        prob = action_probs[idx]
        if prob > 0:
            print(f"    {i+1}. ({row}, {col}): {prob:.4f}")

    # 测试温度采样
    print(f"\n测试温度采样...")
    for temp in [1.0, 0.5, 0.1, 0.0]:
        move = mcts.get_action_with_temperature(action_probs, temperature=temp)
        prob = action_probs[move[0] * 15 + move[1]]
        print(f"  温度={temp:.1f}: 移动{move}, 概率{prob:.4f}")

    print("\n✓ MCTS算法测试通过！")
