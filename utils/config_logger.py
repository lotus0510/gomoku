"""配置變更記錄器 - 追蹤所有參數調整

記錄內容：
- 時間戳
- 迭代數
- 變更類型（手動/自動重置）
- 變更的參數
- 變更前後的值
- 變更原因

作者：Claude Code
日期：2025-12-27
"""
import json
import os
from datetime import datetime, timezone, timedelta


class ConfigChangeLogger:
    """配置變更記錄器"""

    def __init__(self, log_file='checkpoints/config_changes.json'):
        """初始化

        Args:
            log_file: 日誌文件路徑
        """
        self.log_file = log_file
        self.changes = []

        # 載入現有記錄
        if os.path.exists(log_file):
            try:
                with open(log_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.changes = data.get('changes', [])
            except Exception as e:
                print(f"⚠️ 載入配置變更記錄失敗: {e}")
                self.changes = []

    def log_change(self, iteration, change_type, changes, reason=None):
        """記錄一次配置變更

        Args:
            iteration: 當前迭代數
            change_type: 變更類型 ('manual', 'auto_reset', 'resume')
            changes: 變更的參數字典 {param_name: {'old': old_val, 'new': new_val}}
            reason: 變更原因
        """
        # UTC+8 時區
        tz = timezone(timedelta(hours=8))
        timestamp = datetime.now(tz).isoformat()

        change_record = {
            'timestamp': timestamp,
            'iteration': iteration,
            'change_type': change_type,
            'changes': changes,
            'reason': reason,
            'change_count': len(changes)
        }

        self.changes.append(change_record)
        self._save()

        # 打印摘要
        print(f"\n📝 配置變更已記錄:")
        print(f"   時間: {timestamp}")
        print(f"   迭代: {iteration}")
        print(f"   類型: {change_type}")
        print(f"   變更參數數量: {len(changes)}")

    def log_resume_changes(self, iteration, old_config, new_config, important_params=None):
        """記錄恢復訓練時的配置變更

        Args:
            iteration: 當前迭代數
            old_config: 舊配置字典
            new_config: 新配置對象
            important_params: 重要參數列表（如果為 None，檢查所有參數）
        """
        if important_params is None:
            # 默認檢查的重要參數
            important_params = [
                'LEARNING_RATE',
                'LR_DECAY_STEPS',
                'LR_DECAY_RATE',
                'BATCH_SIZE',
                'NUM_WORKERS',
                'DIRICHLET_EPSILON',
                'C_PUCT',
                'MCTS_SIMULATIONS',
                'VALUE_LOSS_WEIGHT',
                'POLICY_LOSS_WEIGHT',
                'GRADIENT_CLIP_NORM',
                'REPLAY_BUFFER_SIZE',
                'GAMES_PER_ITERATION',
                'EPOCHS_PER_ITERATION'
            ]

        changes = {}
        for param in important_params:
            old_val = old_config.get(param) if old_config else None
            new_val = getattr(new_config, param, None)

            if old_val is not None and new_val is not None and old_val != new_val:
                changes[param] = {
                    'old': old_val,
                    'new': new_val,
                    'change_percent': self._calc_change_percent(old_val, new_val)
                }

        if changes:
            self.log_change(
                iteration=iteration,
                change_type='resume',
                changes=changes,
                reason='恢復訓練時檢測到配置變更'
            )
            return True
        return False

    def _calc_change_percent(self, old_val, new_val):
        """計算變更百分比"""
        try:
            if isinstance(old_val, (int, float)) and isinstance(new_val, (int, float)):
                if old_val != 0:
                    return round((new_val / old_val - 1) * 100, 1)
        except:
            pass
        return None

    def _save(self):
        """保存到文件"""
        os.makedirs(os.path.dirname(self.log_file), exist_ok=True)

        data = {
            'metadata': {
                'version': '1.0',
                'description': '訓練配置變更歷史記錄',
                'total_changes': len(self.changes)
            },
            'changes': self.changes
        }

        # 原子性寫入
        temp_file = self.log_file + '.tmp'
        try:
            with open(temp_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)

            if os.path.exists(self.log_file):
                os.remove(self.log_file)
            os.rename(temp_file, self.log_file)
        except Exception as e:
            print(f"⚠️ 保存配置變更記錄失敗: {e}")
            if os.path.exists(temp_file):
                os.remove(temp_file)

    def get_summary(self):
        """獲取變更摘要"""
        if not self.changes:
            return "無配置變更記錄"

        summary = f"配置變更記錄摘要:\n"
        summary += f"  總變更次數: {len(self.changes)}\n"
        summary += f"  最後變更: {self.changes[-1]['timestamp']}\n"
        summary += f"  最後變更迭代: {self.changes[-1]['iteration']}\n"

        # 統計變更類型
        type_counts = {}
        for change in self.changes:
            t = change['change_type']
            type_counts[t] = type_counts.get(t, 0) + 1

        summary += f"  變更類型統計:\n"
        for t, count in type_counts.items():
            summary += f"    - {t}: {count}次\n"

        return summary


if __name__ == '__main__':
    # 測試
    logger = ConfigChangeLogger('test_config_changes.json')

    # 測試記錄手動變更
    logger.log_change(
        iteration=10,
        change_type='manual',
        changes={
            'LEARNING_RATE': {'old': 0.001, 'new': 0.0005, 'change_percent': -50.0},
            'BATCH_SIZE': {'old': 512, 'new': 768, 'change_percent': 50.0}
        },
        reason='手動調整學習率和批次大小'
    )

    print("\n" + logger.get_summary())

    # 清理測試文件
    if os.path.exists('test_config_changes.json'):
        os.remove('test_config_changes.json')
