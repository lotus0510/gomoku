import numpy as np

class GomokuGame:
    def __init__(self, board_size=15):
        """
        初始化遊戲。
        :param board_size: 棋盤大小 (預設為 15)。
        """
        self.board_size = board_size
        self.reset()

    def reset(self):
        """
        重置遊戲狀態。
        """
        self.board = np.zeros((self.board_size, self.board_size), dtype=int)
        self.turn = 1  # 1 代表黑棋, 2 代表白棋
        self.game_over = False
        self.winner = None
        self.last_move = None

    def make_move(self, row, col):
        """
        在指定位置下棋。
        :param row: 行
        :param col: 列
        :return: (game_over, winner) - 回傳遊戲是否結束以及贏家是誰。
                 如果點擊無效，回傳 (self.game_over, self.winner)。
        """
        if self.game_over or not (0 <= row < self.board_size and 0 <= col < self.board_size and self.board[row, col] == 0):
            return self.game_over, self.winner

        self.board[row, col] = self.turn
        self.last_move = (row, col)

        if self.check_win((row, col)):
            self.game_over = True
            self.winner = self.turn
        elif np.all(self.board != 0): # 棋盤已滿，平局
            self.game_over = True
            self.winner = 0 # 0 代表平局

        # 交換玩家
        if not self.game_over:
            self.turn = 2 if self.turn == 1 else 1

        return self.game_over, self.winner

    def check_win(self, last_move):
        """
        根據最後一步棋，檢查當前玩家是否獲勝。
        :param last_move: 最後下棋的位置 (row, col)。
        :return: True 如果當前玩家獲勝, 否則 False。
        """
        row, col = last_move
        player = self.board[row, col]
        
        # 四個方向: 水平, 垂直, 主對角線, 副對角線
        directions = [(0, 1), (1, 0), (1, 1), (1, -1)]
        
        for dr, dc in directions:
            count = 1
            # 檢查正方向
            for i in range(1, 5):
                r, c = row + i * dr, col + i * dc
                if 0 <= r < self.board_size and 0 <= c < self.board_size and self.board[r, c] == player:
                    count += 1
                else:
                    break
            # 檢查反方向
            for i in range(1, 5):
                r, c = row - i * dr, col - i * dc
                if 0 <= r < self.board_size and 0 <= c < self.board_size and self.board[r, c] == player:
                    count += 1
                else:
                    break
            
            if count >= 5:
                return True
        return False

    def get_board_state(self):
        """
        回傳當前的棋盤狀態。
        :return: np.array
        """
        return self.board

    def get_legal_moves(self):
        """
        回傳所有可以下的有效位置。
        :return: list of tuples, e.g., [(0, 0), (0, 1), ...]
        """
        return list(zip(*np.where(self.board == 0)))
