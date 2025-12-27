"""MCTS批量推理版本 - 使用虛擬損失技術支持批量神經網絡推理 (PyTorch 版本)"""

import numpy as np
import math
import torch
from core.game_state import GameState


class MCTSNode:
    """MCTS树节点（支持虚拟损失）"""

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
        self.virtual_loss = 0       # 虚拟损失计数（用于并行搜索）

    def get_value(self):
        """获取节点平均价值 Q(s,a) = W(s,a) / N(s,a)"""
        # 考虑虚拟损失的访问次数
        total_visits = self.visit_count + self.virtual_loss
        if total_visits == 0:
            return 0.0
        return self.total_value / (total_visits + 1e-8)

    def is_leaf(self):
        """是否为叶节点"""
        return len(self.children) == 0

    def __repr__(self):
        return f"MCTSNode(N={self.visit_count}, W={self.total_value:.2f}, P={self.prior_prob:.3f}, VL={self.virtual_loss})"


class BatchedMCTS:
    """蒙特卡罗树搜索（批量推理版本）"""

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

        # 批量推理参数
        self.batch_size = getattr(config, 'MCTS_BATCH_SIZE', 8)  # 每批处理8个叶节点

    def search(self, state, add_noise=True):
        """
        执行MCTS搜索（使用批量推理）

        Args:
            state: GameState 游戏状态
            add_noise: 是否添加Dirichlet噪声（训练时True，评估时False）

        Returns:
            (action_probs, value) tuple
            - action_probs: (board_size^2,) 改进的策略分布
            - value: 根节点价值估计
        """
        root = MCTSNode(prior_prob=1.0)

        # 扩展根节点（单个推理）
        self._expand_single(root, state)

        # 根节点添加Dirichlet噪声增加探索
        if add_noise:
            self._add_dirichlet_noise(root, state)

        # 批量执行模拟
        num_batches = (self.num_simulations + self.batch_size - 1) // self.batch_size

        for batch_idx in range(num_batches):
            # 当前批次的模拟数量
            current_batch_size = min(
                self.batch_size,
                self.num_simulations - batch_idx * self.batch_size
            )

            # 收集一批叶节点
            leaf_nodes = []
            leaf_states = []
            paths = []

            for _ in range(current_batch_size):
                # 克隆状态用于模拟
                sim_state = state.clone()

                # Selection: 选择到叶节点
                path, leaf_node = self._select_to_leaf(root, sim_state)

                # 如果游戏未结束，收集叶节点
                if not sim_state.is_game_over():
                    leaf_nodes.append(leaf_node)
                    leaf_states.append(sim_state)
                    paths.append((path, None))  # None占位，稍后填入value
                else:
                    # 游戏结束，直接计算价值
                    winner = sim_state.get_winner()
                    current_player = sim_state.get_current_player()

                    if winner == 0:
                        value = 0.0  # 平局
                    elif winner == current_player:
                        value = 1.0  # 当前玩家赢
                    else:
                        value = -1.0  # 当前玩家输

                    paths.append((path, value))

            # 批量推理叶节点
            if leaf_nodes:
                values = self._expand_batch(leaf_nodes, leaf_states)

                # 填入价值
                value_idx = 0
                for i, (path, val) in enumerate(paths):
                    if val is None:  # 需要填入的
                        paths[i] = (path, values[value_idx])
                        value_idx += 1

            # 批量反向传播
            for path, value in paths:
                if value is not None:
                    self._backup(path, value)

        # 收集访问次数分布
        action_probs = self._get_action_probs(root, state)

        # 根节点价值估计
        root_value = root.get_value()

        return action_probs, root_value

    def _select_to_leaf(self, root, state):
        """
        选择路径直到叶节点（使用虚拟损失）

        Args:
            root: 根节点
            state: 游戏状态（会被修改）

        Returns:
            (path, leaf_node) tuple
            - path: [(parent, child), ...] 路径
            - leaf_node: 叶节点
        """
        path = []
        current_node = root

        while not current_node.is_leaf() and not state.is_game_over():
            # 选择最佳子节点
            action, child_node = self._select_child(current_node, state)

            # 添加虚拟损失
            child_node.virtual_loss += 1

            path.append((current_node, child_node))

            # 执行动作
            state.make_move(*action)
            current_node = child_node

        return path, current_node

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
        parent_visits = node.visit_count + node.virtual_loss

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
        计算PUCT分数（考虑虚拟损失）

        PUCT = Q(s,a) + c_puct * P(s,a) * sqrt(N(s)) / (1 + N(s,a))

        Args:
            node: 子节点
            parent_visits: 父节点访问次数（含虚拟损失）

        Returns:
            PUCT分数
        """
        # Q值：平均价值（已考虑虚拟损失）
        q_value = node.get_value()

        # U值：探索奖励
        u_value = (
            self.c_puct *
            node.prior_prob *
            math.sqrt(parent_visits) /
            (1 + node.visit_count + node.virtual_loss)
        )

        return q_value + u_value

    def _expand_single(self, node, state):
        """
        扩展单个节点：调用神经网络评估

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

        # 创建子节点
        self._create_children(node, state, policy)

        return value

    def _expand_batch(self, nodes, states):
        """
        批量扩展节点：批量调用神经网络评估

        Args:
            nodes: 要扩展的节点列表
            states: 对应的状态列表

        Returns:
            价值估计列表
        """
        if not nodes:
            return []

        # 准备批量神经网络输入
        batch_inputs = np.array([state.to_input() for state in states])

        # 转换为 PyTorch 格式 (NHWC -> NCHW)
        batch_inputs = np.transpose(batch_inputs, (0, 3, 1, 2))

        # 转换为 PyTorch 张量
        batch_inputs = torch.from_numpy(batch_inputs).float().to(self.device)

        # 批量神经网络推理（无梯度）
        self.model.eval()
        with torch.no_grad():
            batch_policies, batch_values = self.model(batch_inputs)

        # 转换回 NumPy
        batch_policies = batch_policies.cpu().numpy()
        batch_values = batch_values.cpu().numpy()

        # 为每个节点创建子节点
        values = []
        for i, (node, state) in enumerate(zip(nodes, states)):
            policy = batch_policies[i]
            value = batch_values[i, 0]

            self._create_children(node, state, policy)
            values.append(value)

        return values

    def _create_children(self, node, state, policy):
        """
        为节点创建子节点

        Args:
            node: 父节点
            state: 当前状态
            policy: 策略向量
        """
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
        反向传播价值（移除虚拟损失）

        Args:
            path: [(parent, child), ...] 路径
            value: 叶节点价值
        """
        # 从叶节点向根节点传播
        for parent, child in reversed(path):
            # 移除虚拟损失
            child.virtual_loss -= 1

            # 更新统计
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
