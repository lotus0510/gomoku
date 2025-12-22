#!/usr/bin/env python3
"""訓練健康度診斷工具 - 從多個數據源抓取問題徵兆"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
import sys

class TrainingDiagnostics:
    """訓練診斷分析器"""

    def __init__(self, checkpoint_dir='checkpoints', log_dir='logs/games'):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.log_dir = Path(log_dir)

        # 載入數據
        self.load_data()

    def load_data(self):
        """載入所有數據源"""
        # 1. 訓練歷史
        history_path = self.checkpoint_dir / 'training_history.json'
        if history_path.exists():
            with open(history_path, 'r') as f:
                self.history = json.load(f)
        else:
            self.history = None

        # 2. 遊戲匯總
        games_path = self.log_dir / 'games_summary.csv'
        if games_path.exists():
            self.games_df = pd.read_csv(games_path)
        else:
            self.games_df = None

    def diagnose_all(self):
        """執行完整診斷"""
        print("=" * 80)
        print("訓練健康度診斷報告")
        print("=" * 80)
        print()

        issues = []

        # 1. 檢查勝率平衡
        print("【1. 勝率平衡性檢查】")
        print("-" * 80)
        winrate_issues = self.check_winrate_balance()
        issues.extend(winrate_issues)
        print()

        # 2. 檢查遊戲長度
        print("【2. 遊戲長度分析】")
        print("-" * 80)
        length_issues = self.check_game_length()
        issues.extend(length_issues)
        print()

        # 3. 檢查策略質量
        print("【3. 策略質量檢查】")
        print("-" * 80)
        policy_issues = self.check_policy_quality()
        issues.extend(policy_issues)
        print()

        # 4. 檢查價值網路
        print("【4. 價值網路檢查】")
        print("-" * 80)
        value_issues = self.check_value_network()
        issues.extend(value_issues)
        print()

        # 5. 檢查梯度健康度
        print("【5. 梯度健康度檢查】")
        print("-" * 80)
        gradient_issues = self.check_gradient_health()
        issues.extend(gradient_issues)
        print()

        # 6. 檢查訓練穩定性
        print("【6. 訓練穩定性檢查】")
        print("-" * 80)
        stability_issues = self.check_training_stability()
        issues.extend(stability_issues)
        print()

        # 總結
        print("=" * 80)
        print("診斷總結")
        print("=" * 80)
        if not issues:
            print("✅ 未發現嚴重問題，訓練狀態健康")
        else:
            print(f"⚠️  發現 {len(issues)} 個問題：\n")
            for i, issue in enumerate(issues, 1):
                severity = issue['severity']
                emoji = "🔴" if severity == "critical" else "⚠️" if severity == "warning" else "ℹ️"
                print(f"{emoji} [{severity.upper()}] {issue['title']}")
                print(f"   {issue['detail']}")
                if 'suggestion' in issue:
                    print(f"   建議: {issue['suggestion']}")
                print()

        return issues

    def check_winrate_balance(self):
        """檢查勝率平衡性"""
        issues = []

        if self.games_df is None:
            print("⚠️  無遊戲數據")
            return issues

        # 最近N次迭代的統計
        recent_n = 5
        recent_iters = sorted(self.games_df['iteration'].unique())[-recent_n:]

        print(f"最近 {recent_n} 次迭代的勝率：\n")
        print("迭代 | 黑棋勝率 | 白棋勝率 | 狀態")
        print("-----|---------|---------|------")

        for iter_num in recent_iters:
            iter_games = self.games_df[self.games_df['iteration'] == iter_num]
            black_wins = (iter_games['winner'] == 1).sum()
            white_wins = (iter_games['winner'] == 2).sum()
            total = len(iter_games)

            black_rate = black_wins / total * 100
            white_rate = white_wins / total * 100

            # 判斷狀態
            if 45 <= black_rate <= 55:
                status = "✅ 平衡"
            elif 40 <= black_rate <= 60:
                status = "⚠️  輕微偏斜"
            elif 25 <= black_rate <= 75:
                status = "⚠️  明顯偏斜"
            else:
                status = "🔴 嚴重失衡"
                issues.append({
                    'severity': 'critical',
                    'title': f'迭代 {iter_num} 勝率嚴重失衡',
                    'detail': f'黑棋 {black_rate:.1f}% vs 白棋 {white_rate:.1f}%',
                    'suggestion': '考慮回滾到健康迭代或調整探索參數'
                })

            print(f"{iter_num:4d} | {black_rate:7.1f}% | {white_rate:7.1f}% | {status}")

        # 檢查趨勢
        if len(recent_iters) >= 3:
            iter_stats = []
            for iter_num in recent_iters:
                iter_games = self.games_df[self.games_df['iteration'] == iter_num]
                black_rate = (iter_games['winner'] == 1).sum() / len(iter_games) * 100
                iter_stats.append(black_rate)

            # 檢查是否持續偏斜
            if all(x < 25 or x > 75 for x in iter_stats[-3:]):
                issues.append({
                    'severity': 'critical',
                    'title': '勝率持續失衡（連續3次）',
                    'detail': f'最近3次迭代黑棋勝率: {iter_stats[-3:]:.1f}%',
                    'suggestion': '訓練已陷入局部最優，建議重新開始或回滾'
                })

        return issues

    def check_game_length(self):
        """檢查遊戲長度"""
        issues = []

        if self.games_df is None:
            print("⚠️  無遊戲數據")
            return issues

        recent_iter = self.games_df['iteration'].max()
        recent_games = self.games_df[self.games_df['iteration'] == recent_iter]

        avg_length = recent_games['num_moves'].mean()
        ultra_short_ratio = (recent_games['num_moves'] <= 12).sum() / len(recent_games) * 100
        short_ratio = (recent_games['num_moves'] <= 20).sum() / len(recent_games) * 100

        print(f"迭代 {recent_iter} 的遊戲長度統計：")
        print(f"  平均長度: {avg_length:.1f} 步")
        print(f"  超短局（≤12步）: {ultra_short_ratio:.1f}%")
        print(f"  短局（≤20步）: {short_ratio:.1f}%")

        # 判斷
        if ultra_short_ratio > 50:
            issues.append({
                'severity': 'critical',
                'title': 'Edge-Rush 問題嚴重',
                'detail': f'超短局佔 {ultra_short_ratio:.1f}%（正常應 <20%）',
                'suggestion': '模型學到簡單速勝策略，需增加探索或懲罰短局'
            })
        elif ultra_short_ratio > 30:
            issues.append({
                'severity': 'warning',
                'title': 'Edge-Rush 問題中等',
                'detail': f'超短局佔 {ultra_short_ratio:.1f}%',
                'suggestion': '建議監控，如持續惡化需調整配置'
            })

        if avg_length < 15:
            issues.append({
                'severity': 'warning',
                'title': '平均遊戲長度過短',
                'detail': f'平均 {avg_length:.1f} 步（正常應 >20 步）',
                'suggestion': '遊戲缺乏深度，策略可能過於簡單'
            })

        return issues

    def check_policy_quality(self):
        """檢查策略質量（從詳細遊戲記錄）"""
        issues = []

        # 從CSV檢查策略指標
        if self.games_df is None or 'avg_policy_top1_prob' not in self.games_df.columns:
            print("⚠️  無策略質量數據")
            return issues

        recent_iter = self.games_df['iteration'].max()
        recent_games = self.games_df[self.games_df['iteration'] == recent_iter]

        avg_top1 = recent_games['avg_policy_top1_prob'].mean()
        avg_entropy = recent_games['avg_policy_entropy'].mean()

        print(f"迭代 {recent_iter} 的策略質量：")
        print(f"  平均 Top-1 機率: {avg_top1:.3f}")
        print(f"  平均策略熵: {avg_entropy:.3f}")

        # 判斷
        if avg_top1 > 0.9:
            issues.append({
                'severity': 'critical',
                'title': '策略過度自信',
                'detail': f'Top-1 機率 {avg_top1:.3f}（正常應 0.1-0.5）',
                'suggestion': '模型過擬合到單一策略，缺乏探索'
            })
        elif avg_top1 < 0.1:
            issues.append({
                'severity': 'critical',
                'title': '策略接近隨機',
                'detail': f'Top-1 機率 {avg_top1:.3f}（正常應 0.1-0.5）',
                'suggestion': '策略網路崩潰，無法學習有效策略'
            })

        if avg_entropy < 0.5:
            issues.append({
                'severity': 'warning',
                'title': '策略熵過低',
                'detail': f'策略熵 {avg_entropy:.3f}（策略過於確定）',
                'suggestion': '可能過擬合，建議增加探索'
            })

        return issues

    def check_value_network(self):
        """檢查價值網路（需要讀取詳細遊戲記錄）"""
        issues = []

        # 尋找最新迭代的詳細記錄
        if self.games_df is None:
            print("⚠️  無遊戲數據")
            return issues

        recent_iter = self.games_df['iteration'].max()
        iter_dir = self.log_dir / f'iteration_{recent_iter}'

        if not iter_dir.exists():
            print(f"⚠️  找不到迭代 {recent_iter} 的詳細記錄")
            return issues

        # 讀取幾個遊戲樣本
        game_files = list(iter_dir.glob('game_*.json'))[:5]

        if not game_files:
            print("⚠️  找不到詳細遊戲記錄")
            return issues

        all_values = []
        for game_file in game_files:
            with open(game_file, 'r') as f:
                game_data = json.load(f)
                if 'values' in game_data:
                    all_values.extend(game_data['values'])

        if all_values:
            value_array = np.array(all_values)
            value_mean = np.mean(value_array)
            value_std = np.std(value_array)
            zero_ratio = (value_array == 0.0).sum() / len(value_array) * 100

            print(f"價值網路統計（樣本: {len(all_values)} 個評估）：")
            print(f"  平均值: {value_mean:.3f}")
            print(f"  標準差: {value_std:.3f}")
            print(f"  零值比例: {zero_ratio:.1f}%")

            # 判斷
            if zero_ratio > 80:
                issues.append({
                    'severity': 'critical',
                    'title': '價值網路完全失效',
                    'detail': f'{zero_ratio:.1f}% 的評估都是 0.0',
                    'suggestion': '價值網路無法判別局面，訓練已崩潰'
                })
            elif value_std < 0.1:
                issues.append({
                    'severity': 'warning',
                    'title': '價值網路缺乏判別力',
                    'detail': f'標準差僅 {value_std:.3f}（正常應 >0.3）',
                    'suggestion': '價值網路輸出過於均勻，需檢查訓練'
                })
        else:
            print("⚠️  無價值數據")

        return issues

    def check_gradient_health(self):
        """檢查梯度健康度"""
        issues = []

        if self.history is None:
            print("⚠️  無訓練歷史")
            return issues

        recent_n = 5
        recent_grads = self.history['gradient_norm'][-recent_n:]

        grad_mean = np.mean(recent_grads)
        grad_std = np.std(recent_grads)

        print(f"最近 {recent_n} 次迭代的梯度範數：")
        print(f"  平均值: {grad_mean:.3f}")
        print(f"  標準差: {grad_std:.3f}")
        print(f"  範圍: {min(recent_grads):.3f} - {max(recent_grads):.3f}")

        # 判斷
        if grad_mean < 0.1:
            issues.append({
                'severity': 'critical',
                'title': '梯度消失',
                'detail': f'梯度範數 {grad_mean:.3f} < 0.1',
                'suggestion': '學習停滯，建議增加學習率或重置訓練'
            })
        elif grad_mean > 5.0:
            issues.append({
                'severity': 'critical',
                'title': '梯度爆炸',
                'detail': f'梯度範數 {grad_mean:.3f} > 5.0',
                'suggestion': '訓練不穩定，建議降低學習率或檢查數據'
            })
        elif 0.5 <= grad_mean <= 2.0:
            print("  ✅ 梯度範圍健康（0.5-2.0）")

        return issues

    def check_training_stability(self):
        """檢查訓練穩定性（勝率振盪）"""
        issues = []

        if self.games_df is None:
            print("⚠️  無遊戲數據")
            return issues

        # 計算每次迭代的黑棋勝率
        iterations = sorted(self.games_df['iteration'].unique())
        if len(iterations) < 10:
            print(f"⚠️  迭代次數不足（僅 {len(iterations)} 次）")
            return issues

        recent_iters = iterations[-10:]
        winrates = []

        for iter_num in recent_iters:
            iter_games = self.games_df[self.games_df['iteration'] == iter_num]
            black_rate = (iter_games['winner'] == 1).sum() / len(iter_games)
            winrates.append(black_rate)

        # 計算振盪程度
        winrate_std = np.std(winrates)

        # 檢查翻轉次數（從黑優勢變白優勢，或反之）
        flips = 0
        for i in range(1, len(winrates)):
            prev_black_advantage = winrates[i-1] > 0.5
            curr_black_advantage = winrates[i] > 0.5
            if prev_black_advantage != curr_black_advantage:
                flips += 1

        print(f"最近 10 次迭代的勝率穩定性：")
        print(f"  勝率標準差: {winrate_std:.3f}")
        print(f"  優勢翻轉次數: {flips}")

        # 判斷
        if winrate_std > 0.2 and flips >= 3:
            issues.append({
                'severity': 'critical',
                'title': '訓練劇烈振盪',
                'detail': f'勝率標準差 {winrate_std:.3f}，翻轉 {flips} 次',
                'suggestion': '策略在黑白優勢間來回振盪，訓練不穩定'
            })
        elif winrate_std > 0.15:
            issues.append({
                'severity': 'warning',
                'title': '訓練存在振盪',
                'detail': f'勝率標準差 {winrate_std:.3f}',
                'suggestion': '訓練不夠穩定，建議調整學習率或探索參數'
            })
        else:
            print("  ✅ 訓練穩定")

        return issues


if __name__ == '__main__':
    diagnostics = TrainingDiagnostics()
    issues = diagnostics.diagnose_all()

    # 返回錯誤碼
    if any(issue['severity'] == 'critical' for issue in issues):
        sys.exit(1)  # 有嚴重問題
    elif any(issue['severity'] == 'warning' for issue in issues):
        sys.exit(2)  # 有警告
    else:
        sys.exit(0)  # 正常
