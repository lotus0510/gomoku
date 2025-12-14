"""游戏状态包装器 - 为MCTS优化的游戏状态管理"""

import numpy as np
import copy
from game import GomokuGame


class GameState:
    """
    游戏状态包装器

    为MCTS提供优化的接口：
    - 快速复制
    - 哈希缓存
    - 统一接口
    """

    def __init__(self, board_size=15, game=None):
        """
        初始化游戏状态

        Args:
            board_size: 棋盘大小
            game: GomokuGame实例（可选，用于复制）
        """
        if game is None:
            self.game = GomokuGame(board_size=board_size)
        else:
            self.game = game

        self.board_size = board_size
        self._hash = None  # 缓存哈希值

    def clone(self):
        """
        快速复制游戏状态

        Returns:
            新的GameState实例
        """
        # 深拷贝game对象
        new_game = GomokuGame(board_size=self.board_size)
        new_game.board = self.game.board.copy()
        new_game.turn = self.game.turn
        new_game.game_over = self.game.game_over
        new_game.winner = self.game.winner
        new_game.last_move = self.game.last_move

        return GameState(board_size=self.board_size, game=new_game)

    def get_legal_moves(self):
        """
        获取所有合法落子位置

        Returns:
            list of (row, col) tuples
        """
        return self.game.get_legal_moves()

    def make_move(self, row, col):
        """
        在指定位置落子

        Args:
            row: 行坐标
            col: 列坐标

        Returns:
            (game_over, winner) tuple
        """
        self._hash = None  # 清除哈希缓存
        return self.game.make_move(row, col)

    def is_game_over(self):
        """游戏是否结束"""
        return self.game.game_over

    def get_winner(self):
        """获取赢家（1=黑棋，2=白棋，0=平局，None=未结束）"""
        return self.game.winner

    def get_current_player(self):
        """获取当前玩家（1或2）"""
        return self.game.turn

    def get_board_state(self):
        """
        获取棋盘状态数组

        Returns:
            (board_size, board_size) NumPy数组
        """
        return self.game.get_board_state()

    def to_input(self):
        """
        转换为神经网络输入格式

        Returns:
            (board_size, board_size, 3) 张量
        """
        board = self.game.get_board_state()
        turn = self.game.turn
        board_size = self.board_size

        # 通道1: 当前玩家棋子
        player_channel = (board == turn).astype(float)

        # 通道2: 对手棋子
        opponent_turn = 2 if turn == 1 else 1
        opponent_channel = (board == opponent_turn).astype(float)

        # 通道3: 当前回合标识
        turn_channel = np.ones((board_size, board_size), dtype=float) if turn == 1 else np.zeros((board_size, board_size), dtype=float)

        # 堆叠为3通道
        input_tensor = np.stack([player_channel, opponent_channel, turn_channel], axis=-1)

        return input_tensor

    def get_hash(self):
        """
        获取状态哈希值（用于缓存）

        Returns:
            int哈希值
        """
        if self._hash is None:
            # 使用棋盘状态和当前玩家计算哈希
            board_tuple = tuple(self.game.board.flatten()) + (self.game.turn,)
            self._hash = hash(board_tuple)

        return self._hash

    def __hash__(self):
        """支持作为字典键"""
        return self.get_hash()

    def __eq__(self, other):
        """状态相等性比较"""
        if not isinstance(other, GameState):
            return False

        return (
            np.array_equal(self.game.board, other.game.board) and
            self.game.turn == other.game.turn
        )

    def __repr__(self):
        """字符串表示"""
        return f"GameState(turn={self.game.turn}, moves={np.sum(self.game.board != 0)}, over={self.game.game_over})"


if __name__ == '__main__':
    print("=== 测试游戏状态包装器 ===\n")

    # 创建游戏状态
    state = GameState(board_size=15)
    print(f"初始状态: {state}")
    print(f"合法移动数: {len(state.get_legal_moves())}")
    print(f"当前玩家: {state.get_current_player()}")

    # 测试落子
    print("\n下几步棋...")
    state.make_move(7, 7)
    print(f"  第1步后: {state}")

    state.make_move(7, 8)
    print(f"  第2步后: {state}")

    # 测试克隆
    print("\n测试克隆...")
    cloned_state = state.clone()
    print(f"  原状态: {state}")
    print(f"  克隆状态: {cloned_state}")
    print(f"  棋盘相等: {np.array_equal(state.get_board_state(), cloned_state.get_board_state())}")

    # 修改克隆不影响原状态
    cloned_state.make_move(8, 7)
    print(f"  克隆修改后: {cloned_state}")
    print(f"  原状态未变: {state}")

    # 测试哈希
    print("\n测试哈希...")
    hash1 = state.get_hash()
    hash2 = cloned_state.get_hash()
    print(f"  原状态哈希: {hash1}")
    print(f"  克隆状态哈希: {hash2}")
    print(f"  哈希不同: {hash1 != hash2}")

    # 测试输入转换
    print("\n测试神经网络输入...")
    nn_input = state.to_input()
    print(f"  输入形状: {nn_input.shape}")
    print(f"  通道1（我方）非零: {np.sum(nn_input[:,:,0] > 0)}")
    print(f"  通道2（对方）非零: {np.sum(nn_input[:,:,1] > 0)}")
    print(f"  通道3（回合）均值: {np.mean(nn_input[:,:,2])}")

    print("\n✓ 游戏状态包装器测试通过！")
