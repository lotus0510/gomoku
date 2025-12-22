"""训练游戏日志记录器 - 记录每局游戏的详细数据"""

import os
import json
import csv
from datetime import datetime
import numpy as np


class GameLogger:
    """记录每局游戏的详细信息"""

    def __init__(self, log_dir='logs/games', enabled=True, detailed_frequency=10):
        """
        初始化游戏日志记录器

        Args:
            log_dir: 日志目录
            enabled: 是否启用日志（可以设为False节省磁盘空间）
            detailed_frequency: 每N局保存详细数据（设为1则每局都保存）
        """
        self.log_dir = log_dir
        self.enabled = enabled
        self.detailed_frequency = detailed_frequency

        if self.enabled:
            os.makedirs(log_dir, exist_ok=True)

            # 创建CSV汇总文件
            self.summary_file = os.path.join(log_dir, 'games_summary.csv')
            self._init_summary_file()

            # 当前迭代的游戏列表
            self.current_iteration_games = []

    def _init_summary_file(self):
        """初始化CSV汇总文件（扩展版）"""
        if not os.path.exists(self.summary_file):
            with open(self.summary_file, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow([
                    'iteration', 'game_num', 'num_moves', 'winner',
                    'game_duration_sec', 'avg_mcts_time', 'black_win',
                    'white_win', 'draw', 'avg_policy_entropy',
                    # 新增策略质量指标
                    'avg_policy_top1_prob', 'policy_diversity', 'max_policy_entropy', 'min_policy_entropy'
                ])

    def log_game(self, iteration, game_num, game_data):
        """
        记录一局游戏的数据

        Args:
            iteration: 当前迭代次数
            game_num: 游戏编号
            game_data: 游戏数据字典，包含：
                - moves: 移动列表
                - winner: 胜者 (1=黑, 2=白, 0=平局)
                - num_moves: 总步数
                - positions: 每步的棋盘状态（可选）
                - policies: 每步的策略分布（可选）
                - values: 每步的价值估计（可选）
                - mcts_stats: MCTS统计信息（可选）
        """
        if not self.enabled:
            return

        try:
            # 构建游戏记录
            game_record = {
                'iteration': iteration,
                'game_num': game_num,
                'timestamp': datetime.now().isoformat(),
                'num_moves': game_data.get('num_moves', 0),
                'winner': game_data.get('winner', 0),
                'black_win': 1 if game_data.get('winner') == 1 else 0,
                'white_win': 1 if game_data.get('winner') == 2 else 0,
                'draw': 1 if game_data.get('winner') == 0 else 0,
            }

            # 计算策略质量指标
            if 'policies' in game_data:
                entropies = []
                top1_probs = []

                for policy in game_data['policies']:
                    # 只计算非零概率的熵
                    valid_probs = policy[policy > 1e-8]
                    if len(valid_probs) > 0:
                        entropy = -np.sum(valid_probs * np.log(valid_probs + 1e-8))
                        entropies.append(entropy)

                    # Top-1 概率（最优动作的置信度）
                    top1_prob = np.max(policy)
                    top1_probs.append(top1_prob)

                game_record['avg_policy_entropy'] = np.mean(entropies) if entropies else 0
                game_record['avg_policy_top1_prob'] = np.mean(top1_probs) if top1_probs else 0
                game_record['max_policy_entropy'] = np.max(entropies) if entropies else 0
                game_record['min_policy_entropy'] = np.min(entropies) if entropies else 0
                # 策略多样性（熵的标准差）
                game_record['policy_diversity'] = np.std(entropies) if len(entropies) > 1 else 0
            else:
                game_record['avg_policy_entropy'] = 0
                game_record['avg_policy_top1_prob'] = 0
                game_record['max_policy_entropy'] = 0
                game_record['min_policy_entropy'] = 0
                game_record['policy_diversity'] = 0

            # 添加时间统计
            game_record['game_duration_sec'] = game_data.get('duration', 0)
            game_record['avg_mcts_time'] = game_data.get('avg_mcts_time', 0)

            # 写入CSV汇总（包含新增指标）
            with open(self.summary_file, 'a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow([
                    game_record['iteration'],
                    game_record['game_num'],
                    game_record['num_moves'],
                    game_record['winner'],
                    game_record['game_duration_sec'],
                    game_record['avg_mcts_time'],
                    game_record['black_win'],
                    game_record['white_win'],
                    game_record['draw'],
                    game_record['avg_policy_entropy'],
                    game_record['avg_policy_top1_prob'],
                    game_record['policy_diversity'],
                    game_record['max_policy_entropy'],
                    game_record['min_policy_entropy']
                ])

            # 添加到当前迭代游戏列表
            self.current_iteration_games.append(game_record)

            # 保存详细的游戏数据（根据配置频率）
            if self.detailed_frequency > 0 and (game_num % self.detailed_frequency == 0 or game_num == 0):
                self._save_detailed_game(iteration, game_num, game_data, game_record)

        except Exception as e:
            print(f"  [日志警告] 保存游戏 {game_num} 失败: {e}")

    def _save_detailed_game(self, iteration, game_num, game_data, game_record):
        """保存详细的游戏数据（包含棋谱、策略等）"""
        detailed_dir = os.path.join(self.log_dir, f'iteration_{iteration}')
        os.makedirs(detailed_dir, exist_ok=True)

        detailed_file = os.path.join(detailed_dir, f'game_{game_num}.json')

        # 构建详细数据（转换numpy为list）
        detailed_data = {
            **{k: (float(v) if isinstance(v, (np.floating, np.integer)) else v)
               for k, v in game_record.items()},
            'moves': game_data.get('moves', []),
        }

        # 保存策略分布（只保存top5位置）
        if 'policies' in game_data:
            detailed_data['top_policies'] = []
            for policy in game_data['policies']:
                top5_indices = np.argsort(policy)[-5:][::-1]
                top5 = [(int(idx // 15), int(idx % 15), float(policy[idx]))
                        for idx in top5_indices]
                detailed_data['top_policies'].append(top5)

        # 保存价值估计
        if 'values' in game_data:
            detailed_data['values'] = [float(v) for v in game_data['values']]

        with open(detailed_file, 'w', encoding='utf-8') as f:
            json.dump(detailed_data, f, indent=2, ensure_ascii=False)

    def finish_iteration(self, iteration):
        """
        完成一次迭代，保存迭代汇总

        Args:
            iteration: 迭代次数
        """
        if not self.enabled or not self.current_iteration_games:
            return

        try:
            # 计算迭代统计
            stats = {
                'iteration': iteration,
                'total_games': len(self.current_iteration_games),
                'avg_moves': float(np.mean([g['num_moves'] for g in self.current_iteration_games])),
                'black_wins': sum(g['black_win'] for g in self.current_iteration_games),
                'white_wins': sum(g['white_win'] for g in self.current_iteration_games),
                'draws': sum(g['draw'] for g in self.current_iteration_games),
                'avg_entropy': float(np.mean([g['avg_policy_entropy'] for g in self.current_iteration_games])),
                'avg_game_duration': float(np.mean([g['game_duration_sec'] for g in self.current_iteration_games])),
            }

            # 保存迭代汇总
            iteration_summary_file = os.path.join(self.log_dir, 'iteration_summary.jsonl')
            with open(iteration_summary_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(stats, ensure_ascii=False) + '\n')

            print(f"  [日志] 迭代 {iteration} 统计:")
            print(f"    总游戏数: {stats['total_games']}")
            print(f"    平均步数: {stats['avg_moves']:.1f}")
            print(f"    黑胜/白胜/平局: {stats['black_wins']}/{stats['white_wins']}/{stats['draws']}")
            print(f"    平均策略熵: {stats['avg_entropy']:.3f}")

            # 清空当前迭代游戏列表
            self.current_iteration_games = []

        except Exception as e:
            print(f"  [日志警告] 保存迭代统计失败: {e}")

    def get_statistics(self, last_n_iterations=10):
        """
        获取最近N次迭代的统计信息

        Args:
            last_n_iterations: 统计最近多少次迭代

        Returns:
            dict: 统计信息
        """
        if not self.enabled:
            return {}

        try:
            iteration_summary_file = os.path.join(self.log_dir, 'iteration_summary.jsonl')
            if not os.path.exists(iteration_summary_file):
                return {}

            # 读取最近N次迭代
            stats_list = []
            with open(iteration_summary_file, 'r', encoding='utf-8') as f:
                for line in f:
                    stats_list.append(json.loads(line))

            recent_stats = stats_list[-last_n_iterations:] if len(stats_list) > last_n_iterations else stats_list

            if not recent_stats:
                return {}

            return {
                'iterations': [s['iteration'] for s in recent_stats],
                'avg_moves_trend': [s['avg_moves'] for s in recent_stats],
                'black_win_rate_trend': [s['black_wins'] / s['total_games'] for s in recent_stats],
                'avg_entropy_trend': [s['avg_entropy'] for s in recent_stats],
            }

        except Exception as e:
            print(f"  [日志警告] 读取统计信息失败: {e}")
            return {}


if __name__ == '__main__':
    # 测试游戏日志记录器
    print("=== 测试游戏日志记录器 ===\n")

    logger = GameLogger(log_dir='logs/test_games', enabled=True)

    # 模拟记录几局游戏
    for iteration in range(1, 3):
        print(f"迭代 {iteration}:")

        for game_num in range(5):
            game_data = {
                'num_moves': np.random.randint(20, 60),
                'winner': np.random.choice([0, 1, 2]),
                'duration': np.random.uniform(10, 30),
                'avg_mcts_time': np.random.uniform(0.1, 0.5),
                'policies': [np.random.dirichlet([0.3] * 225) for _ in range(5)],
                'values': np.random.randn(5),
                'moves': [(i % 15, i // 15) for i in range(5)]
            }

            logger.log_game(iteration, game_num, game_data)

        logger.finish_iteration(iteration)

    # 获取统计信息
    stats = logger.get_statistics(last_n_iterations=2)
    print(f"\n统计信息:")
    print(f"  迭代: {stats.get('iterations', [])}")
    print(f"  平均步数趋势: {stats.get('avg_moves_trend', [])}")
    print(f"  黑棋胜率趋势: {stats.get('black_win_rate_trend', [])}")

    print("\n✓ 游戏日志记录器测试通过！")
    print(f"  检查 logs/test_games/ 查看生成的日志文件")
