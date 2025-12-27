"""停滯檢測器 - 自動判斷訓練是否需要調整參數"""

import json
import os
from datetime import datetime


class StagnationDetector:
    """
    停滯檢測器

    基於多維度指標計算訓練停滯分數 (0-100)
    分數越高 = 越停滯 = 越需要調整參數
    """

    def __init__(self, config):
        self.window = 10  # 檢測窗口 (最近N次迭代)
        self.check_frequency = 5  # 每5次迭代檢查一次
        self.history = []
        self.config = config

    def calculate_score(self, training_history):
        """
        計算停滯分數 (0-100)

        評分標準:
        - 0-50: 健康,繼續訓練
        - 50-70: 輕度停滯,考慮調整
        - 70-85: 中度停滯,建議調整
        - 85-100: 嚴重停滯,立即調整
        """
        score = 0

        if len(training_history['iterations']) < self.window:
            return 0  # 數據不足,不判定

        # 1. 策略損失停滯檢查 (權重: 40分)
        policy_score = self._check_policy_loss(training_history)
        score += policy_score

        # 2. 價值MAE檢查 (權重: 30分)
        mae_score = self._check_mae(training_history)
        score += mae_score

        # 3. 勝率檢查 (權重: 20分)
        wr_score = self._check_win_rate(training_history)
        score += wr_score

        # 4. 梯度檢查 (權重: 10分)
        grad_score = self._check_gradient(training_history)
        score += grad_score

        return min(100, score)

    def _check_policy_loss(self, history):
        """檢查策略損失改善情況 (滿分40)"""
        recent_policy = history['policy_loss'][-self.window:]

        if len(recent_policy) < 2:
            return 0

        improvement = (recent_policy[0] - recent_policy[-1]) / recent_policy[0]

        # 評分邏輯
        if improvement < 0:  # 惡化了
            return 40
        elif improvement < 0.002:  # < 0.2%
            return 40
        elif improvement < 0.005:  # < 0.5%
            return 30
        elif improvement < 0.01:   # < 1%
            return 20
        else:
            return 0

    def _check_mae(self, history):
        """檢查MAE趨勢 (滿分30)"""
        recent_mae = history['value_mae'][-self.window:]

        if len(recent_mae) < 2:
            return 0

        # MAE應該下降
        change = (recent_mae[-1] - recent_mae[0]) / recent_mae[0]

        # 評分邏輯
        if change > 0.05:  # 惡化超過5%
            return 30
        elif change > 0:   # 任何惡化
            return 25
        elif change > -0.05:  # 改善不足5%
            return 20
        elif change > -0.10:  # 改善不足10%
            return 10
        else:
            return 0

    def _check_win_rate(self, history):
        """檢查勝率變化 (滿分20)"""
        recent_wr = [w for w in history['win_rate_vs_random'][-self.window:]
                     if w is not None]

        if len(recent_wr) < 2:
            return 0

        # 檢查是否下降或波動過大
        if recent_wr[-1] < recent_wr[0] - 0.05:  # 下降超過5%
            return 20
        elif abs(recent_wr[-1] - recent_wr[0]) > 0.15:  # 波動超過15%
            return 15
        elif recent_wr[-1] <= recent_wr[0]:  # 無提升
            return 10
        else:
            return 0

    def _check_gradient(self, history):
        """檢查梯度健康度 (滿分10)"""
        recent_grad = history['gradient_norm'][-5:]  # 最近5次

        if len(recent_grad) < 5:
            return 0

        avg_grad = sum(recent_grad) / len(recent_grad)

        # 評分邏輯
        if avg_grad < 0.3:  # 嚴重消失
            return 10
        elif avg_grad < 0.5:  # 偏低
            return 7
        elif avg_grad > 10.0:  # 過高(可能不穩定)
            return 5
        else:
            return 0

    def should_adjust(self, current_iteration):
        """
        判斷是否需要調整參數

        Returns:
            (bool, str): (是否調整, 調整策略)
            策略: 'gentle', 'aggressive', 'reset', None
        """
        # 只在檢查點執行
        if current_iteration % self.check_frequency != 0:
            return False, None

        # 讀取訓練歷史
        history_path = 'checkpoints/training_history.json'
        if not os.path.exists(history_path):
            return False, None

        try:
            with open(history_path, 'r') as f:
                history = json.load(f)
        except:
            return False, None

        # 計算停滯分數
        score = self.calculate_score(history)

        # 記錄歷史
        self.history.append({
            'iteration': current_iteration,
            'score': score,
            'timestamp': datetime.now().isoformat()
        })

        # 判定邏輯
        if score >= 85:
            return True, 'reset'  # 嚴重停滯,重置式調整
        elif score >= 70:
            # 檢查是否連續停滯
            if len(self.history) >= 2 and self.history[-2]['score'] >= 70:
                return True, 'aggressive'  # 連續停滯,激進調整
            return True, 'aggressive'
        elif score >= 50:
            # 檢查是否連續停滯
            if len(self.history) >= 2 and self.history[-2]['score'] >= 50:
                return True, 'aggressive'  # 連續輕度停滯,升級為激進
            return True, 'gentle'  # 首次輕度停滯,溫和調整
        else:
            return False, None

    def get_status_report(self, training_history):
        """獲取詳細狀態報告"""
        score = self.calculate_score(training_history)

        # 分項分數
        policy_score = self._check_policy_loss(training_history)
        mae_score = self._check_mae(training_history)
        wr_score = self._check_win_rate(training_history)
        grad_score = self._check_gradient(training_history)

        report = {
            'total_score': score,
            'breakdown': {
                'policy_loss': policy_score,
                'mae': mae_score,
                'win_rate': wr_score,
                'gradient': grad_score
            },
            'status': self._get_status_level(score),
            'recommendation': self._get_recommendation(score)
        }

        return report

    def _get_status_level(self, score):
        """獲取狀態等級"""
        if score < 50:
            return 'healthy'
        elif score < 70:
            return 'mild_stagnation'
        elif score < 85:
            return 'moderate_stagnation'
        else:
            return 'severe_stagnation'

    def _get_recommendation(self, score):
        """獲取建議"""
        if score < 50:
            return 'continue_training'
        elif score < 70:
            return 'consider_gentle_adjustment'
        elif score < 85:
            return 'aggressive_adjustment_recommended'
        else:
            return 'reset_adjustment_required'


if __name__ == '__main__':
    # 測試檢測器
    print("=" * 60)
    print("停滯檢測器測試")
    print("=" * 60)

    detector = StagnationDetector(config=None)

    # 讀取實際數據
    try:
        with open('checkpoints/training_history.json', 'r') as f:
            history = json.load(f)

        score = detector.calculate_score(history)
        report = detector.get_status_report(history)

        print(f"\n當前迭代: {history['iterations'][-1]}")
        print(f"停滯分數: {score:.1f}/100")
        print(f"狀態等級: {report['status']}")
        print(f"\n分項評分:")
        print(f"  策略損失: {report['breakdown']['policy_loss']}/40")
        print(f"  價值MAE:  {report['breakdown']['mae']}/30")
        print(f"  勝率:     {report['breakdown']['win_rate']}/20")
        print(f"  梯度:     {report['breakdown']['gradient']}/10")
        print(f"\n建議: {report['recommendation']}")

    except Exception as e:
        print(f"錯誤: {e}")
