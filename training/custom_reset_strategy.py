"""自定義重置策略 - 針對價值網絡崩潰的專用解決方案

這個策略專門設計用於處理價值網絡崩潰問題(U型MAE曲線)。

核心創新:
1. 早期檢測: Value Loss連續5次上升即觸發
2. 回滾模型: 加載迭代40的健康checkpoint
3. 清理新數據: 刪除崩潰期的套路化數據,保留健康期數據
4. 穩健調參: 小幅提升學習率(1.5x),避免震蕩

作者: 基於用戶提議優化
日期: 2025-12-26
"""

import os
import torch
import json
from datetime import datetime


class CustomResetStrategy:
    """
    自定義重置策略 - 增強版

    觸發條件:
    - Value Loss連續5次上升
    - MAE在惡化
    - MAE超過健康閾值(0.85)

    執行動作:
    - 回滾到迭代40的checkpoint
    - 清空最新50%的buffer數據
    - 調整參數以加強價值學習
    """

    def __init__(self, target_iteration=40, min_iteration=50, min_reset_interval=20):
        """
        初始化

        Args:
            target_iteration: 回滾目標迭代數 (預設40)
            min_iteration: 最小觸發迭代數 (預設50)
            min_reset_interval: 最小重置間隔 (預設20)
        """
        self.target_iteration = target_iteration
        self.min_iteration = min_iteration
        self.min_reset_interval = min_reset_interval

        self.reset_count = 0
        self.last_reset_iteration = 0
        self.reset_history = []

        # 日誌文件
        self.log_dir = 'checkpoints/reset_logs'
        os.makedirs(self.log_dir, exist_ok=True)

    def should_trigger(self, training_history, current_iteration):
        """
        判斷是否需要觸發重置

        Args:
            training_history: 訓練歷史字典
            current_iteration: 當前迭代數

        Returns:
            bool: 是否應該重置
        """
        # 條件1: 足夠的訓練數據
        if current_iteration < self.min_iteration:
            return False

        # 條件2: 避免頻繁重置
        if current_iteration - self.last_reset_iteration < self.min_reset_interval:
            return False

        # 條件3: 安全上限 (已重置3次則停止)
        if self.reset_count >= 3:
            print(f"\n⚠️ 已重置{self.reset_count}次,停止自動重置")
            print("   需要人工介入檢查配置")
            return False

        # 條件4: 檢查Value Loss連續上升
        value_losses = training_history.get('value_loss', [])
        if len(value_losses) < 5:
            return False

        recent_vl = value_losses[-5:]
        increases = sum(recent_vl[i] > recent_vl[i-1] for i in range(1, 5))

        # 條件5: 檢查MAE惡化
        value_mae = training_history.get('value_mae', [])
        if len(value_mae) < 5:
            return False

        recent_mae = value_mae[-5:]
        mae_worsening = recent_mae[-1] > recent_mae[0]
        mae_unhealthy = recent_mae[-1] > 0.85  # 健康閾值

        # 綜合判定
        should_reset = (
            increases >= 4 and          # Value Loss連續上升4次
            mae_worsening and           # MAE在惡化
            mae_unhealthy               # MAE超過健康值
        )

        if should_reset:
            # 記錄觸發信息
            self._log_trigger(training_history, current_iteration)

        return should_reset

    def execute_reset(self, config, buffer, model, current_iteration, training_history=None):
        """
        執行重置操作

        Args:
            config: 配置對象
            buffer: 經驗池對象
            model: 神經網絡模型
            current_iteration: 當前迭代數
            training_history: 訓練歷史（用於動態查找健康點）

        Returns:
            (config, buffer, model): 更新後的配置、經驗池、模型
        """
        self.reset_count += 1
        self.last_reset_iteration = current_iteration

        print(f"\n{'='*60}")
        print(f"🔄 執行價值崩潰重置策略 (第{self.reset_count}次)")
        print(f"{'='*60}")
        print(f"觸發迭代: {current_iteration}")
        print(f"{'='*60}\n")

        # 動作1: 回滾模型到健康checkpoint（動態查找）
        model = self._rollback_model(model, training_history)

        # 動作2: 清空最新50%的buffer數據
        buffer = self._clean_buffer(buffer)

        # 動作3: 調整訓練參數
        config = self._adjust_parameters(config)

        # 記錄重置
        self._save_reset_record(current_iteration, config)

        print(f"\n{'='*60}")
        print(f"✅ 重置完成! 從健康狀態重新訓練")
        print(f"{'='*60}\n")

        return config, buffer, model

    def _rollback_model(self, model, training_history=None):
        """回滾模型到最後一個健康的迭代

        Args:
            model: 當前模型
            training_history: 訓練歷史（用於動態查找健康點）

        Returns:
            回滾後的模型
        """
        # 動態查找最後一個健康的迭代
        target_iter = self._find_last_healthy_iteration(training_history)

        if target_iter is None:
            # 如果找不到健康點，使用固定目標
            target_iter = self.target_iteration
            print(f"⚠️ 無法從歷史找到健康點，使用預設目標: 迭代 {target_iter}")
        else:
            print(f"🔍 從訓練歷史找到最後健康點: 迭代 {target_iter}")

        # 嘗試加載目標checkpoint
        checkpoint_path = f'checkpoints/checkpoint_iter_{target_iter}.pth'

        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, weights_only=True)
            model.load_state_dict(checkpoint['model_state_dict'])
            print(f"✅ 模型已回滾到迭代 {target_iter}")
            print(f"   Checkpoint: {checkpoint_path}")
            return model

        # 如果目標不存在,往前搜尋可用的checkpoint
        print(f"⚠️ 目標checkpoint不存在: {checkpoint_path}")
        print(f"   正在往前搜尋可用checkpoint...")

        for i in range(target_iter, max(0, target_iter - 30), -1):
            alt_path = f'checkpoints/checkpoint_iter_{i}.pth'
            if os.path.exists(alt_path):
                checkpoint = torch.load(alt_path, weights_only=True)
                model.load_state_dict(checkpoint['model_state_dict'])
                print(f"✅ 使用替代checkpoint: 迭代 {i}")
                print(f"   Checkpoint: {alt_path}")
                return model

        print(f"❌ 未找到合適的checkpoint,保持當前模型")
        return model

    def _find_last_healthy_iteration(self, training_history):
        """從訓練歷史中找到最後一個健康的迭代

        健康標準:
        - MAE < 0.85
        - Value Loss 不在上升趨勢
        - 梯度範數 > 0.3

        Args:
            training_history: 訓練歷史字典

        Returns:
            最後健康迭代數，如果找不到則返回 None
        """
        if not training_history:
            return None

        iterations = training_history.get('iterations', [])
        value_mae = training_history.get('value_mae', [])
        value_loss = training_history.get('value_loss', [])
        gradient_norm = training_history.get('gradient_norm', [])

        if not iterations or not value_mae:
            return None

        # 從後往前查找最後一個健康點
        for i in range(len(iterations) - 1, -1, -1):
            mae = value_mae[i] if i < len(value_mae) else 1.0
            grad = gradient_norm[i] if i < len(gradient_norm) else 0.0

            # 健康條件
            is_healthy = (
                mae < 0.85 and          # MAE健康
                grad > 0.3              # 梯度正常
            )

            if is_healthy:
                # 找到了！再往前退5次作為安全邊際
                safe_iter = max(1, iterations[i] - 5)
                print(f"   健康點分析: 迭代 {iterations[i]} (MAE={mae:.3f}, Grad={grad:.2f})")
                print(f"   安全回滾: 迭代 {safe_iter} (往前退5次作為安全邊際)")
                return safe_iter

        # 找不到健康點
        return None

    def _clean_buffer(self, buffer):
        """清空最新50%的數據,保留健康期數據"""
        # 處理 PrioritizedReplayBuffer
        if hasattr(buffer, 'data'):
            original_size = len(buffer.data)

            if original_size == 0:
                print(f"⚠️ Buffer為空,跳過清理")
                return buffer

            mid = original_size // 2
            buffer.data = buffer.data[:mid]

            # 同步更新 priorities（如果有）
            if hasattr(buffer, 'priorities'):
                buffer.priorities = buffer.priorities[:mid]

            removed_count = original_size - mid
            print(f"✅ 已清空 {removed_count} 條新數據")
            print(f"   保留前 {mid} 條健康期數據 (保留率: {mid/original_size:.1%})")

            return buffer
        else:
            # 如果是簡單列表（不太可能）
            print(f"⚠️ Buffer類型不支持清理,跳過")
            return buffer

    def _adjust_parameters(self, config):
        """調整訓練參數"""
        print(f"\n📊 參數調整:")

        # 1. 學習率: 漸進式提升
        old_lr = config.LEARNING_RATE

        if self.reset_count == 1:
            new_lr = old_lr * 1.5          # 第1次: +50%
        elif self.reset_count == 2:
            new_lr = old_lr * 2.0          # 第2次: +100%
        else:
            new_lr = min(old_lr * 2.5, 0.0005)  # 第3次: +150%,上限0.0005

        config.LEARNING_RATE = new_lr
        print(f"   學習率: {old_lr:.6f} → {new_lr:.6f} ({(new_lr/old_lr-1)*100:+.0f}%)")

        # 2. 探索參數: 適度提升,避免重蹈覆轍
        old_epsilon = config.DIRICHLET_EPSILON
        new_epsilon = min(0.30 + self.reset_count * 0.05, 0.45)
        config.DIRICHLET_EPSILON = new_epsilon
        print(f"   探索率(ε): {old_epsilon:.2f} → {new_epsilon:.2f}")

        # 溫度控制（如果配置有 TEMP_THRESHOLD_MOVE）
        if hasattr(config, 'TEMP_THRESHOLD_MOVE'):
            # 延長探索期
            old_temp_threshold = config.TEMP_THRESHOLD_MOVE
            new_temp_threshold = min(old_temp_threshold + 5, 35)
            config.TEMP_THRESHOLD_MOVE = new_temp_threshold
            print(f"   溫度閾值: {old_temp_threshold} → {new_temp_threshold} 步")

        # 3. 價值損失權重: 加強價值學習
        old_weight = getattr(config, 'VALUE_LOSS_WEIGHT', 1.0)
        new_weight = 2.0 + (self.reset_count - 1) * 0.5  # 2.0, 2.5, 3.0
        config.VALUE_LOSS_WEIGHT = new_weight
        print(f"   價值權重: {old_weight:.1f} → {new_weight:.1f}")

        # 4. 批次大小: 可選 - 減小批次有助於擺脫局部最優
        if hasattr(config, 'BATCH_SIZE') and config.BATCH_SIZE > 512:
            old_batch = config.BATCH_SIZE
            new_batch = max(512, int(old_batch * 0.75))
            config.BATCH_SIZE = new_batch
            print(f"   批次大小: {old_batch} → {new_batch}")

        return config

    def _log_trigger(self, history, iteration):
        """記錄觸發信息"""
        recent_vl = history['value_loss'][-5:]
        recent_mae = history['value_mae'][-5:]

        print(f"\n{'='*60}")
        print(f"🚨 檢測到價值網絡崩潰!")
        print(f"{'='*60}")
        print(f"當前迭代: {iteration}")
        print(f"\nValue Loss (最近5次): {[f'{x:.4f}' for x in recent_vl]}")
        print(f"MAE (最近5次):        {[f'{x:.4f}' for x in recent_mae]}")
        print(f"\nMAE變化: {recent_mae[0]:.4f} → {recent_mae[-1]:.4f} "
              f"({(recent_mae[-1]/recent_mae[0]-1)*100:+.1f}%)")
        print(f"{'='*60}\n")

    def _save_reset_record(self, iteration, config):
        """保存重置記錄"""
        record = {
            'reset_number': self.reset_count,
            'timestamp': datetime.now().isoformat(),
            'trigger_iteration': iteration,
            'target_iteration': self.target_iteration,
            'new_config': {
                'LEARNING_RATE': config.LEARNING_RATE,
                'DIRICHLET_EPSILON': config.DIRICHLET_EPSILON,
                'TEMP_THRESHOLD_MOVE': getattr(config, 'TEMP_THRESHOLD_MOVE', 'N/A'),
                'VALUE_LOSS_WEIGHT': getattr(config, 'VALUE_LOSS_WEIGHT', 1.0),
                'BATCH_SIZE': getattr(config, 'BATCH_SIZE', 'N/A'),
            }
        }

        self.reset_history.append(record)

        # 保存到JSON
        log_file = os.path.join(self.log_dir, 'reset_history.json')
        with open(log_file, 'w', encoding='utf-8') as f:
            json.dump(self.reset_history, f, indent=2, ensure_ascii=False)

        print(f"📝 重置記錄已保存: {log_file}")

    def get_status(self):
        """獲取狀態信息"""
        return {
            'reset_count': self.reset_count,
            'last_reset_iteration': self.last_reset_iteration,
            'can_reset': self.reset_count < 3,
            'history': self.reset_history
        }


# 測試代碼
if __name__ == '__main__':
    print("=" * 60)
    print("自定義重置策略測試")
    print("=" * 60)

    strategy = CustomResetStrategy(target_iteration=40)

    # 模擬訓練歷史
    mock_history = {
        'value_loss': [1.0, 1.05, 1.1, 1.15, 1.2],  # 連續上升
        'value_mae': [0.85, 0.87, 0.89, 0.91, 0.93],  # MAE惡化且超過0.85
    }

    # 測試觸發檢測
    should_trigger = strategy.should_trigger(mock_history, current_iteration=55)

    print(f"\n測試迭代: 55")
    print(f"Value Loss: {mock_history['value_loss']}")
    print(f"MAE: {mock_history['value_mae']}")
    print(f"\n觸發重置? {should_trigger}")

    if should_trigger:
        print("\n✅ 檢測邏輯正常工作!")
        print(f"   將會回滾到迭代 {strategy.target_iteration}")
    else:
        print("\n❌ 未觸發 (檢查條件)")

    # 顯示狀態
    status = strategy.get_status()
    print(f"\n當前狀態: {status}")
