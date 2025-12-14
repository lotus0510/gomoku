"""ELO评级系统"""


class EloRating:
    """ELO评级系统"""

    def __init__(self, k_factor=32, initial_rating=1500):
        """
        初始化ELO系统

        Args:
            k_factor: K因子（决定评分变化幅度）
            initial_rating: 初始评分
        """
        self.k_factor = k_factor
        self.initial_rating = initial_rating
        self.ratings = {}  # {player_name: rating}

    def get_rating(self, player):
        """获取玩家评分"""
        return self.ratings.get(player, self.initial_rating)

    def expected_score(self, rating1, rating2):
        """
        计算预期得分

        Args:
            rating1: 玩家1的评分
            rating2: 玩家2的评分

        Returns:
            玩家1的预期得分 [0, 1]
        """
        return 1.0 / (1.0 + 10 ** ((rating2 - rating1) / 400.0))

    def update_rating(self, player1, player2, score):
        """
        更新评分

        Args:
            player1: 玩家1名称
            player2: 玩家2名称
            score: 玩家1的得分（1.0=赢, 0.5=平局, 0.0=输）

        Returns:
            (new_rating1, new_rating2) tuple
        """
        r1 = self.get_rating(player1)
        r2 = self.get_rating(player2)

        # 计算预期得分
        expected1 = self.expected_score(r1, r2)
        expected2 = 1 - expected1

        # 更新评分
        new_r1 = r1 + self.k_factor * (score - expected1)
        new_r2 = r2 + self.k_factor * ((1 - score) - expected2)

        # 保存新评分
        self.ratings[player1] = new_r1
        self.ratings[player2] = new_r2

        return new_r1, new_r2

    def get_all_ratings(self):
        """获取所有玩家评分"""
        return self.ratings.copy()


if __name__ == '__main__':
    print("=== 测试ELO评级系统 ===\n")

    elo = EloRating(k_factor=32, initial_rating=1500)

    # 模拟比赛
    print("模拟比赛...")
    elo.update_rating("ModelA", "ModelB", score=1.0)  # A赢
    print(f"  比赛1（A赢B）: A={elo.get_rating('ModelA'):.1f}, B={elo.get_rating('ModelB'):.1f}")

    elo.update_rating("ModelA", "ModelB", score=0.0)  # B赢
    print(f"  比赛2（B赢A）: A={elo.get_rating('ModelA'):.1f}, B={elo.get_rating('ModelB'):.1f}")

    elo.update_rating("ModelA", "ModelB", score=0.5)  # 平局
    print(f"  比赛3（平局）: A={elo.get_rating('ModelA'):.1f}, B={elo.get_rating('ModelB'):.1f}")

    # 连续胜利
    for i in range(10):
        elo.update_rating("ModelA", "ModelB", score=1.0)

    print(f"\n  A连胜10局后: A={elo.get_rating('ModelA'):.1f}, B={elo.get_rating('ModelB'):.1f}")

    print("\n✓ ELO评级系统测试通过！")
