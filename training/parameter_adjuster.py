"""參數調整器 - 根據停滯程度調整訓練參數"""

import json
import shutil
import os
from datetime import datetime


class ParameterAdjuster:
    """
    參數調整器

    提供三種調整策略:
    1. gentle: 溫和調整 (+20-50%)
    2. aggressive: 激進調整 (+100-200%)
    3. reset: 重置式調整 (大幅改變)
    """

    # 溫和調整規則 (適用於首次輕度停滯)
    GENTLE_ADJUSTMENTS = {
        'DIRICHLET_EPSILON': lambda x: min(x * 1.2, 0.5),      # +20%
        'LEARNING_RATE': lambda x: min(x * 1.5, 0.001),        # +50%
        'BATCH_SIZE': lambda x: max(int(x * 0.8), 256),        # -20%
        'TEMPERATURE': lambda x: min(x * 1.1, 1.5),            # +10%
    }

    # 激進調整規則 (適用於連續停滯或中度停滯)
    AGGRESSIVE_ADJUSTMENTS = {
        'DIRICHLET_EPSILON': 0.40,                    # 固定高探索
        'DIRICHLET_ALPHA': 0.50,                      # 增加探索多樣性
        'TEMPERATURE': 1.5,                           # 高溫度
        'LEARNING_RATE': lambda x: min(x * 2.0, 0.001),  # 翻倍
        'BATCH_SIZE': 512,                            # 減半
        'VALUE_LOSS_WEIGHT': 2.0,                     # 加強價值學習
        'CPUCT': 2.5,                                 # 增加MCTS探索
        'GRADIENT_CLIP_NORM': 5.0,                    # 放寬梯度裁剪
    }

    # 重置式調整 (適用於嚴重停滯)
    RESET_ADJUSTMENTS = {
        'LEARNING_RATE': 0.001,                       # 提升到10倍
        'TEMPERATURE': 2.0,                           # 極高探索
        'DIRICHLET_EPSILON': 0.5,                     # 最大探索
        'BATCH_SIZE': 512,                            # 較小批次
        'VALUE_LOSS_WEIGHT': 2.5,                     # 強化價值
        'EPOCHS_PER_ITERATION': 15,                   # 增加訓練強度
    }

    def __init__(self):
        self.adjustment_history = []
        self.backup_dir = 'checkpoints/parameter_adjustment_backups'
        os.makedirs(self.backup_dir, exist_ok=True)

    def adjust(self, strategy, current_config, iteration):
        """
        執行參數調整

        Args:
            strategy: 'gentle', 'aggressive', 'reset'
            current_config: 當前配置對象
            iteration: 當前迭代數

        Returns:
            (new_config, adjustments): 新配置和調整記錄
        """
        # 備份當前配置
        self._backup_config(current_config, iteration)

        # 根據策略執行調整
        if strategy == 'gentle':
            new_config, adjustments = self._apply_gentle(current_config)
        elif strategy == 'aggressive':
            new_config, adjustments = self._apply_aggressive(current_config)
        elif strategy == 'reset':
            new_config, adjustments = self._apply_reset(current_config)
        else:
            return current_config, {}

        # 記錄調整歷史
        self._log_adjustment(iteration, strategy, adjustments)

        return new_config, adjustments

    def _apply_gentle(self, config):
        """溫和調整"""
        adjustments = {}

        for param, adjuster in self.GENTLE_ADJUSTMENTS.items():
            if not hasattr(config, param):
                continue

            old_value = getattr(config, param)
            new_value = adjuster(old_value) if callable(adjuster) else adjuster

            # 計算變化百分比
            if isinstance(old_value, (int, float)) and old_value != 0:
                change_pct = ((new_value - old_value) / old_value) * 100
            else:
                change_pct = 0

            adjustments[param] = {
                'old': old_value,
                'new': new_value,
                'change': f'{change_pct:+.1f}%'
            }

            setattr(config, param, new_value)

        return config, adjustments

    def _apply_aggressive(self, config):
        """激進調整"""
        adjustments = {}

        for param, new_value_or_func in self.AGGRESSIVE_ADJUSTMENTS.items():
            if not hasattr(config, param):
                continue

            old_value = getattr(config, param)

            # 計算新值
            if callable(new_value_or_func):
                new_value = new_value_or_func(old_value)
            else:
                new_value = new_value_or_func

            # 計算變化百分比
            if isinstance(old_value, (int, float)) and old_value != 0:
                change_pct = ((new_value - old_value) / old_value) * 100
            else:
                change_pct = 0

            adjustments[param] = {
                'old': old_value,
                'new': new_value,
                'change': f'{change_pct:+.1f}%'
            }

            setattr(config, param, new_value)

        return config, adjustments

    def _apply_reset(self, config):
        """重置式調整 (最激進)"""
        adjustments = {}

        for param, new_value in self.RESET_ADJUSTMENTS.items():
            if not hasattr(config, param):
                continue

            old_value = getattr(config, param)

            # 計算變化百分比
            if isinstance(old_value, (int, float)) and old_value != 0:
                change_pct = ((new_value - old_value) / old_value) * 100
            else:
                change_pct = 0

            adjustments[param] = {
                'old': old_value,
                'new': new_value,
                'change': f'{change_pct:+.1f}%'
            }

            setattr(config, param, new_value)

        # 添加特殊動作標記
        adjustments['_special_actions'] = {
            'clear_buffer_50': True,      # 清空50%經驗池
            'reset_optimizer': False,     # 暫不重置優化器
        }

        return config, adjustments

    def _backup_config(self, config, iteration):
        """備份配置"""
        backup_file = os.path.join(
            self.backup_dir,
            f'config_iter{iteration}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
        )

        # 提取可序列化的配置
        config_dict = {}
        for attr in dir(config):
            if not attr.startswith('_') and not callable(getattr(config, attr)):
                value = getattr(config, attr)
                if isinstance(value, (int, float, str, bool)):
                    config_dict[attr] = value

        with open(backup_file, 'w') as f:
            json.dump(config_dict, f, indent=2)

        print(f"✅ 配置已備份: {backup_file}")

    def _log_adjustment(self, iteration, strategy, adjustments):
        """記錄調整歷史"""
        log_file = 'checkpoints/adjustment_log.json'

        # 讀取現有日誌
        if os.path.exists(log_file):
            with open(log_file, 'r') as f:
                log = json.load(f)
        else:
            log = {'adjustments': []}

        # 添加新記錄
        log['adjustments'].append({
            'iteration': iteration,
            'timestamp': datetime.now().isoformat(),
            'strategy': strategy,
            'adjustments': adjustments
        })

        # 寫回
        with open(log_file, 'w') as f:
            json.dump(log, f, indent=2, ensure_ascii=False)

    def get_adjustment_history(self):
        """獲取調整歷史"""
        log_file = 'checkpoints/adjustment_log.json'

        if not os.path.exists(log_file):
            return []

        with open(log_file, 'r') as f:
            log = json.load(f)

        return log.get('adjustments', [])


if __name__ == '__main__':
    # 測試調整器
    print("=" * 60)
    print("參數調整器測試")
    print("=" * 60)

    # 模擬配置對象
    class MockConfig:
        DIRICHLET_EPSILON = 0.25
        LEARNING_RATE = 0.0001
        BATCH_SIZE = 1024
        TEMPERATURE = 1.0
        VALUE_LOSS_WEIGHT = 1.0
        CPUCT = 2.0
        GRADIENT_CLIP_NORM = 1.0
        DIRICHLET_ALPHA = 0.3
        EPOCHS_PER_ITERATION = 5

    config = MockConfig()
    adjuster = ParameterAdjuster()

    # 測試溫和調整
    print("\n【溫和調整】")
    new_config, adjustments = adjuster.adjust('gentle', config, iteration=100)
    for param, values in adjustments.items():
        print(f"{param}: {values['old']} → {values['new']} ({values['change']})")

    # 測試激進調整
    print("\n【激進調整】")
    config2 = MockConfig()
    new_config2, adjustments2 = adjuster.adjust('aggressive', config2, iteration=200)
    for param, values in adjustments2.items():
        if not param.startswith('_'):
            print(f"{param}: {values['old']} → {values['new']} ({values['change']})")

    # 測試重置調整
    print("\n【重置調整】")
    config3 = MockConfig()
    new_config3, adjustments3 = adjuster.adjust('reset', config3, iteration=300)
    for param, values in adjustments3.items():
        if not param.startswith('_'):
            print(f"{param}: {values['old']} → {values['new']} ({values['change']})")
