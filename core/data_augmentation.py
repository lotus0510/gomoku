"""数据增强模块 - 利用棋盘的8种对称性扩充训练数据"""

import numpy as np


def get_symmetries(board, policy, board_size=15):
    """
    生成棋盘和策略的8种对称变换

    五子棋棋盘具有8种对称性：
    - 4种旋转（0°, 90°, 180°, 270°）
    - 2种翻转×4种旋转 = 8种

    Args:
        board: (board_size, board_size, 3) 棋盘状态
        policy: (board_size * board_size,) 策略向量
        board_size: 棋盘大小

    Returns:
        list of (board, policy) tuples, 共8个
    """
    symmetries = []

    # 将策略从向量转为矩阵以便旋转
    policy_matrix = policy.reshape(board_size, board_size)

    # 4种旋转（0°, 90°, 180°, 270°）
    for k in range(4):
        rotated_board = np.rot90(board, k, axes=(0, 1))  # 只旋转前两个维度
        rotated_policy = np.rot90(policy_matrix, k)
        symmetries.append((
            rotated_board.copy(),
            rotated_policy.flatten().copy()
        ))

    # 水平翻转后再旋转4次
    flipped_board = np.fliplr(board)
    flipped_policy = np.fliplr(policy_matrix)

    for k in range(4):
        rotated_board = np.rot90(flipped_board, k, axes=(0, 1))
        rotated_policy = np.rot90(flipped_policy, k)
        symmetries.append((
            rotated_board.copy(),
            rotated_policy.flatten().copy()
        ))

    return symmetries


def apply_random_symmetry(board, policy, board_size=15):
    """
    随机应用一种对称变换（用于训练时增强）

    Args:
        board: (board_size, board_size, 3) 棋盘状态
        policy: (board_size * board_size,) 策略向量
        board_size: 棋盘大小

    Returns:
        (augmented_board, augmented_policy) tuple
    """
    symmetries = get_symmetries(board, policy, board_size)
    return symmetries[np.random.randint(8)]


def test_symmetries():
    """测试对称变换的正确性"""
    board_size = 15

    # 创建测试棋盘：在(0,0)放黑棋，(1,1)放白棋
    test_board = np.zeros((board_size, board_size, 3), dtype=np.float32)
    test_board[0, 0, 0] = 1  # 黑棋在左上角
    test_board[1, 1, 1] = 1  # 白棋在(1,1)

    # 创建测试策略：最高概率在(2,2)
    test_policy = np.zeros(board_size * board_size, dtype=np.float32)
    test_policy[2 * board_size + 2] = 0.9
    test_policy[3 * board_size + 3] = 0.1
    test_policy /= test_policy.sum()

    print("原始棋盘状态：")
    print(f"  黑棋位置: (0, 0)")
    print(f"  白棋位置: (1, 1)")
    print(f"  策略最大值位置: (2, 2)")

    # 测试8种对称变换
    symmetries = get_symmetries(test_board, test_policy, board_size)

    print(f"\n生成了 {len(symmetries)} 种对称变换")

    for i, (aug_board, aug_policy) in enumerate(symmetries):
        # 找到黑棋位置
        black_positions = np.where(aug_board[:, :, 0] == 1)
        white_positions = np.where(aug_board[:, :, 1] == 1)
        policy_max_idx = np.argmax(aug_policy)
        policy_max_pos = (policy_max_idx // board_size, policy_max_idx % board_size)

        print(f"  变换 {i}: 黑棋{list(zip(black_positions[0], black_positions[1]))}, "
              f"白棋{list(zip(white_positions[0], white_positions[1]))}, "
              f"策略最大{policy_max_pos}")

    # 验证策略和为1
    for i, (_, aug_policy) in enumerate(symmetries):
        policy_sum = aug_policy.sum()
        assert abs(policy_sum - 1.0) < 1e-5, f"变换 {i} 策略和不为1: {policy_sum}"

    print("\n✓ 所有变换的策略和均为1")
    print("✓ 数据增强模块测试通过")


if __name__ == '__main__':
    # 运行测试
    test_symmetries()

    # 测试随机增强
    board_size = 15
    test_board = np.random.rand(board_size, board_size, 3).astype(np.float32)
    test_policy = np.random.rand(board_size * board_size).astype(np.float32)
    test_policy /= test_policy.sum()

    print("\n测试随机增强（10次）：")
    for i in range(10):
        aug_board, aug_policy = apply_random_symmetry(test_board, test_policy, board_size)
        print(f"  第{i+1}次: 棋盘形状{aug_board.shape}, 策略形状{aug_policy.shape}, "
              f"策略和{aug_policy.sum():.6f}")

    print("\n✓ 随机增强功能正常")
