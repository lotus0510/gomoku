"""竞技场系统 - 模型对战评估"""

import random
from core.game_state import GameState


class Arena:
    """竞技场系统"""

    def __init__(self, board_size=15):
        """
        初始化竞技场

        Args:
            board_size: 棋盘大小
        """
        self.board_size = board_size

    def play_game(self, player1, player2, verbose=False):
        """
        进行一局游戏

        Args:
            player1: 玩家1（应有get_action(state)方法）
            player2: 玩家2
            verbose: 是否打印详细信息

        Returns:
            winner (1=player1, 2=player2, 0=平局)
        """
        state = GameState(board_size=self.board_size)
        players = {1: player1, 2: player2}

        move_count = 0
        max_moves = self.board_size * self.board_size

        while not state.is_game_over() and move_count < max_moves:
            current_player_id = state.get_current_player()
            current_player = players[current_player_id]

            # 获取玩家动作
            action = current_player.get_action(state)

            if action is None:
                # 玩家无法选择动作
                if verbose:
                    print(f"玩家{current_player_id}无法选择动作")
                break

            # 执行动作
            state.make_move(*action)
            move_count += 1

            if verbose:
                print(f"第{move_count}步: 玩家{current_player_id} -> {action}")

        winner = state.get_winner()
        if winner is None:
            winner = 0  # 超过最大步数，判为平局

        if verbose:
            print(f"游戏结束，赢家: {winner}")

        return winner

    def compete(self, player1, player2, num_games=50, name1="Player1", name2="Player2"):
        """
        让两个玩家对弈多局

        Args:
            player1: 玩家1
            player2: 玩家2
            num_games: 对弈局数
            name1: 玩家1名称
            name2: 玩家2名称

        Returns:
            results dict with stats
        """
        results = {
            'player1_wins': 0,
            'player2_wins': 0,
            'draws': 0,
            'games': []
        }

        for game_num in range(num_games):
            # 轮流先手
            if game_num % 2 == 0:
                winner = self.play_game(player1, player2)
                if winner == 1:
                    results['player1_wins'] += 1
                elif winner == 2:
                    results['player2_wins'] += 1
                else:
                    results['draws'] += 1
            else:
                winner = self.play_game(player2, player1)
                if winner == 1:
                    results['player2_wins'] += 1
                elif winner == 2:
                    results['player1_wins'] += 1
                else:
                    results['draws'] += 1

            results['games'].append(winner)

        # 计算统计
        total_games = num_games
        results['player1_win_rate'] = results['player1_wins'] / total_games
        results['player2_win_rate'] = results['player2_wins'] / total_games
        results['draw_rate'] = results['draws'] / total_games

        # 计算得分
        results['player1_score'] = results['player1_wins'] + 0.5 * results['draws']
        results['player2_score'] = results['player2_wins'] + 0.5 * results['draws']

        return results

    def should_promote(self, new_player, current_best, num_games=100, threshold=0.55):
        """
        判断新模型是否应该晋级为最佳模型

        Args:
            new_player: 新模型
            current_best: 当前最佳模型
            num_games: 对弈局数
            threshold: 胜率阈值

        Returns:
            True if new_player should be promoted
        """
        results = self.compete(new_player, current_best, num_games=num_games)
        win_rate = results['player1_win_rate'] + 0.5 * results['draw_rate']

        return win_rate >= threshold


if __name__ == '__main__':
    print("=== 测试竞技场系统 ===\n")

    # 创建测试玩家（随机玩家）
    class RandomPlayer:
        def get_action(self, state):
            legal_moves = state.get_legal_moves()
            if not legal_moves:
                return None
            return random.choice(legal_moves)

    # 创建竞技场
    arena = Arena(board_size=15)

    # 创建两个随机玩家
    player1 = RandomPlayer()
    player2 = RandomPlayer()

    # 进行对弈
    print("进行10局对弈...")
    results = arena.compete(player1, player2, num_games=10, name1="Random1", name2="Random2")

    print(f"\n结果:")
    print(f"  Player1 胜: {results['player1_wins']}")
    print(f"  Player2 胜: {results['player2_wins']}")
    print(f"  平局: {results['draws']}")
    print(f"  Player1 胜率: {results['player1_win_rate']:.3f}")

    print("\n✓ 竞技场系统测试通过！")
