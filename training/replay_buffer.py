"""经验回放缓冲区 - 优先级回放（Prioritized Experience Replay）"""

import numpy as np
from collections import deque
import random


class ReplayBuffer:
    """简单的经验回放缓冲区（FIFO）"""

    def __init__(self, capacity=50000):
        """
        初始化缓冲区

        Args:
            capacity: 缓冲区最大容量
        """
        self.buffer = deque(maxlen=capacity)
        self.capacity = capacity

    def add(self, experience):
        """
        添加经验

        Args:
            experience: (state, policy, value) tuple
        """
        self.buffer.append(experience)

    def sample(self, batch_size):
        """
        随机采样批次

        Args:
            batch_size: 批次大小

        Returns:
            list of experiences
        """
        sample_size = min(batch_size, len(self.buffer))
        return random.sample(self.buffer, sample_size)

    def __len__(self):
        return len(self.buffer)

    def clear(self):
        """清空缓冲区"""
        self.buffer.clear()


class PrioritizedReplayBuffer:
    """优先级经验回放缓冲区"""

    def __init__(self, capacity=50000, alpha=0.6, beta=0.4, beta_increment=0.001):
        """
        初始化优先级回放缓冲区

        Args:
            capacity: 缓冲区最大容量
            alpha: 优先级指数 (0=均匀采样, 1=完全按优先级)
            beta: 重要性采样修正指数
            beta_increment: beta每次采样的增长量
        """
        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta
        self.beta_increment = beta_increment
        self.beta_max = 1.0

        self.buffer = deque(maxlen=capacity)
        self.priorities = deque(maxlen=capacity)

    def add(self, experience, td_error=None):
        """
        添加经验到缓冲区

        Args:
            experience: (state, policy, value, ...) tuple
            td_error: TD-error（可选，用于计算优先级）
        """
        if td_error is None:
            # 如果没有TD-error，给予最大优先级
            if len(self.priorities) > 0:
                priority = max(self.priorities)
            else:
                priority = 1.0
        else:
            # 优先级 = |TD-error| + epsilon
            priority = (abs(td_error) + 1e-6) ** self.alpha

        self.buffer.append(experience)
        self.priorities.append(priority)

    def sample(self, batch_size):
        """
        按优先级采样批次

        Args:
            batch_size: 批次大小

        Returns:
            (samples, weights, indices) tuple
            - samples: 采样的经验列表
            - weights: 重要性采样权重
            - indices: 采样的索引（用于更新优先级）
        """
        if len(self.buffer) == 0:
            return [], [], []

        sample_size = min(batch_size, len(self.buffer))

        # 计算采样概率
        priorities_array = np.array(self.priorities, dtype=np.float32)
        probs = priorities_array / priorities_array.sum()

        # 按概率采样
        indices = np.random.choice(
            len(self.buffer),
            size=sample_size,
            p=probs,
            replace=False
        )

        # 采样经验
        samples = [self.buffer[i] for i in indices]

        # 计算重要性采样权重
        # w_i = (N * P(i))^(-beta)
        weights = (len(self.buffer) * probs[indices]) ** (-self.beta)
        weights /= weights.max()  # 归一化

        # 增加beta（退火到1.0）
        self.beta = min(self.beta_max, self.beta + self.beta_increment)

        return samples, weights, indices

    def update_priorities(self, indices, td_errors):
        """
        更新采样样本的优先级

        Args:
            indices: 样本索引列表
            td_errors: 对应的TD-error列表
        """
        for idx, error in zip(indices, td_errors):
            priority = (abs(error) + 1e-6) ** self.alpha
            self.priorities[idx] = priority

    def __len__(self):
        return len(self.buffer)

    def clear(self):
        """清空缓冲区"""
        self.buffer.clear()
        self.priorities.clear()


if __name__ == '__main__':
    print("=== 测试经验回放缓冲区 ===\n")

    # 测试简单回放缓冲区
    print("1. 测试简单回放缓冲区")
    simple_buffer = ReplayBuffer(capacity=100)

    # 添加一些经验
    for i in range(50):
        state = np.random.rand(15, 15, 3)
        policy = np.random.rand(225)
        policy /= policy.sum()
        value = np.random.randn()
        simple_buffer.add((state, policy, value))

    print(f"   缓冲区大小: {len(simple_buffer)}")

    # 采样
    batch = simple_buffer.sample(batch_size=10)
    print(f"   采样批次大小: {len(batch)}")
    print(f"   样本类型: {type(batch[0])}")

    # 测试优先级回放缓冲区
    print("\n2. 测试优先级回放缓冲区")
    priority_buffer = PrioritizedReplayBuffer(
        capacity=100,
        alpha=0.6,
        beta=0.4
    )

    # 添加经验（带不同优先级）
    for i in range(50):
        state = np.random.rand(15, 15, 3)
        policy = np.random.rand(225)
        policy /= policy.sum()
        value = np.random.randn()
        td_error = np.random.rand() * (i + 1) / 50  # 模拟递增的TD-error

        priority_buffer.add((state, policy, value), td_error=td_error)

    print(f"   缓冲区大小: {len(priority_buffer)}")
    print(f"   Beta值: {priority_buffer.beta:.4f}")

    # 采样
    samples, weights, indices = priority_buffer.sample(batch_size=10)
    print(f"   采样批次大小: {len(samples)}")
    print(f"   重要性权重范围: [{weights.min():.4f}, {weights.max():.4f}]")
    print(f"   采样索引: {indices}")

    # 更新优先级
    new_td_errors = np.random.rand(len(indices)) * 2
    priority_buffer.update_priorities(indices, new_td_errors)
    print(f"   优先级已更新")

    # 多次采样观察beta增长
    print("\n3. 测试Beta退火")
    initial_beta = priority_buffer.beta
    for i in range(10):
        _, _, _ = priority_buffer.sample(batch_size=10)

    final_beta = priority_buffer.beta
    print(f"   初始Beta: {initial_beta:.4f}")
    print(f"   最终Beta: {final_beta:.4f}")
    print(f"   增长量: {final_beta - initial_beta:.4f}")

    # 测试容量限制
    print("\n4. 测试容量限制")
    small_buffer = PrioritizedReplayBuffer(capacity=20)

    for i in range(30):
        experience = (i, i, i)  # 简单经验
        small_buffer.add(experience, td_error=float(i))

    print(f"   添加30个经验后，缓冲区大小: {len(small_buffer)}")
    print(f"   应该为20（容量限制）: {len(small_buffer) == 20}")

    print("\n✓ 经验回放缓冲区测试通过！")
