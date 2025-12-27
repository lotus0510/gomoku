# 自动参数调整系统 - 整合指南

**目的**: 自动检测训练停滞并调整参数
**类型**: 通用停滞检测（补充自定义重置策略）
**状态**: ✅ 代码已就绪

---

## 系统概述

本系统提供**通用停滞检测**，作为[自定义重置策略](CUSTOM_RESET_GUIDE.md)的补充。

### 两种策略对比

| 特性 | 自定义重置策略 ⭐ | 通用停滞检测 |
|------|-----------------|------------|
| **适用场景** | 价值网络崩溃 | 综合性停滞 |
| **检测方式** | Value Loss连续上升 | 多指标综合评分 |
| **调整方式** | 回滚+清理数据 | 调参继续训练 |
| **推荐度** | ⭐⭐⭐⭐⭐ 针对当前问题 | ⭐⭐⭐ 通用备选 |

**建议**: 优先使用[自定义重置策略](CUSTOM_RESET_GUIDE.md)，它对价值崩溃效果更好。

---

## 快速开始

### 文件检查

确认以下文件存在:
```bash
ls training/stagnation_detector.py
ls training/parameter_adjuster.py
```

### 整合步骤

#### 1. 导入模块

在 `train_pipeline_pytorch.py` 顶部添加:

```python
from training.stagnation_detector import StagnationDetector
from training.parameter_adjuster import ParameterAdjuster
```

#### 2. 初始化系统

在 `main()` 函数中:

```python
def main():
    config = TrainingConfig.get_optimized_config()

    # 初始化通用停滞检测
    detector = StagnationDetector(config)
    adjuster = ParameterAdjuster()
    print("✅ 自动参数调整系统已启用")

    # ... 其他初始化 ...
```

#### 3. 添加检查逻辑

在迭代循环末尾:

```python
for iteration in range(start_iteration, config.ITERATIONS):
    # ... 训练逻辑 ...

    # 检查是否需要调整参数
    should_adjust, strategy = detector.should_adjust(iteration)

    if should_adjust:
        print(f"\n{'='*60}")
        print(f"🔧 检测到训练停滞! (迭代 {iteration})")
        print(f"   停滞分数: {detector.history[-1]['score']:.1f}/100")
        print(f"   调整策略: {strategy}")
        print(f"{'='*60}\n")

        # 执行调整
        config, adjustments = adjuster.adjust(strategy, config, iteration)

        # 显示调整内容
        print("📊 参数调整详情:")
        for param, values in adjustments.items():
            if not param.startswith('_'):
                print(f"   {param}: {values['old']} → {values['new']}")

        # 如果需要清空buffer
        if adjustments.get('_special_actions', {}).get('clear_buffer_50'):
            mid = len(replay_buffer.data) // 2
            replay_buffer.data = replay_buffer.data[mid:]
            print(f"✅ 已清空50%旧数据")
```

---

## 核心机制

### 停滞检测

系统监控4个维度:

1. **策略损失**（40分）
   - 最近5次迭代改善 < 0.5%

2. **价值MAE**（30分）
   - 最近5次迭代改善 < 5%

3. **胜率**（20分）
   - 相比前10次迭代下降

4. **梯度**（10分）
   - 平均梯度 < 0.5

**停滞分数 >= 80** 时触发调整

### 三级调整策略

#### Level 1: Gentle（分数 80-89）
```
学习率: +10%
温度: +0.2
探索率: +0.05
```

#### Level 2: Aggressive（分数 90-99）
```
学习率: +50%
温度: +0.5
探索率: +0.15
价值权重: +50%
```

#### Level 3: Reset（分数 >= 100）
```
学习率: 10倍
温度: 2倍
清空50%旧数据
```

---

## 与自定义重置策略配合使用

### 推荐配置

如果同时使用两种策略:

```python
# 初始化
reset_strategy = CustomResetStrategy(target_iteration=40)  # 价值崩溃
detector = StagnationDetector(config)                       # 综合停滞
adjuster = ParameterAdjuster()

# 检查顺序：先检查价值崩溃，再检查综合停滞
if reset_strategy.should_trigger(training_history, iteration):
    # 价值崩溃 -> 回滚模型
    config, replay_buffer, model = reset_strategy.execute_reset(...)
elif detector.should_adjust(iteration)[0]:
    # 综合停滞 -> 调整参数
    should_adjust, strategy = detector.should_adjust(iteration)
    config, adjustments = adjuster.adjust(strategy, config, iteration)
```

### 避免冲突

- 自定义重置触发后，通用检测自动重置评分
- 两次调整至少间隔10次迭代
- 价值崩溃优先级高于综合停滞

---

## 监控与日志

### 查看调整历史

```bash
# 查看通用调整记录
cat checkpoints/adjustment_log.json

# 格式化输出
python -c "
import json
with open('checkpoints/adjustment_log.json') as f:
    log = json.load(f)
    for record in log:
        print(f'迭代{record[\"iteration\"]}: {record[\"strategy\"]} 调整')
"
```

### 停滞分数追踪

```python
# 在训练过程中查看停滞分数
status = detector.get_status()
print(f"当前停滞分数: {status['current_score']}")
print(f"历史调整次数: {status['adjustment_count']}")
```

---

## 配置选项

### 调整检测阈值

修改 `training/stagnation_detector.py`:

```python
# 降低触发阈值
self.threshold = 70  # 默认80，改为70更敏感

# 调整各维度权重
weights = {
    'policy_loss': 50,    # 默认40
    'value_mae': 30,      # 默认30
    'win_rate': 15,       # 默认20
    'gradient': 5         # 默认10
}
```

### 调整参数变化幅度

修改 `training/parameter_adjuster.py`:

```python
# Gentle策略更保守
'gentle': {
    'LEARNING_RATE': 1.05,    # 默认1.1，改为+5%
    'TEMPERATURE': 0.1,       # 默认0.2
    'DIRICHLET_EPSILON': 0.03 # 默认0.05
}
```

---

## 常见问题

### Q1: 通用检测和自定义重置有什么区别？

**A**:
- **自定义重置**: 专治价值崩溃，回滚模型到健康点
- **通用检测**: 监控综合指标，调参继续训练
- **推荐**: 优先用自定义重置，通用检测作为补充

### Q2: 可以只用通用检测吗？

**A**: 可以，但对于价值崩溃问题，自定义重置效果更好。通用检测更适合：
- 训练初期（没有健康checkpoint可回滚）
- 综合性停滞（多个指标都差）
- 探索新配置

### Q3: 两种策略会冲突吗？

**A**: 不会，如果按推荐顺序检查：
```python
# 1. 先检查价值崩溃
if reset_strategy.should_trigger(...):
    # 执行回滚
# 2. 再检查综合停滞
elif detector.should_adjust(...)[0]:
    # 执行调参
```

### Q4: 调整太频繁怎么办？

**A**:
```python
# 提高阈值
detector.threshold = 85  # 默认80

# 增加最小间隔
detector.min_interval = 15  # 默认10
```

---

## 性能对比

### 场景1: 价值网络崩溃

| 方案 | 检测时机 | 恢复时间 | 效果 |
|------|---------|---------|------|
| 自定义重置 | 迭代55 | 10次迭代 | ⭐⭐⭐⭐⭐ |
| 通用检测 | 迭代80 | 30次迭代 | ⭐⭐ |

### 场景2: 综合性停滞

| 方案 | 检测时机 | 恢复时间 | 效果 |
|------|---------|---------|------|
| 自定义重置 | 可能不触发 | N/A | ⭐⭐ |
| 通用检测 | 及时检测 | 15次迭代 | ⭐⭐⭐⭐ |

**结论**: 针对当前问题（价值崩溃），自定义重置是最佳选择。

---

## 相关文档

- [自定义重置策略完整指南](CUSTOM_RESET_GUIDE.md) ⭐ 推荐优先阅读
- [监控指标说明](MONITORING_METRICS.md)
- [训练故障诊断](../troubleshooting/TRAINING_FAILURE_REPORT.md)

---

## 总结

### 推荐方案

**针对当前价值崩溃问题**:
1. ✅ **优先使用**: [自定义重置策略](CUSTOM_RESET_GUIDE.md)
2. ⭐ **可选补充**: 本通用检测系统

### 实施建议

**最简配置（推荐）**:
```python
# 只使用自定义重置
reset_strategy = CustomResetStrategy(target_iteration=40)

if reset_strategy.should_trigger(training_history, iteration):
    config, replay_buffer, model = reset_strategy.execute_reset(...)
```

**完整配置（全面保护）**:
```python
# 双重检测
reset_strategy = CustomResetStrategy(target_iteration=40)
detector = StagnationDetector(config)
adjuster = ParameterAdjuster()

# 价值崩溃优先
if reset_strategy.should_trigger(...):
    # 回滚处理
elif detector.should_adjust(...)[0]:
    # 调参处理
```

---

**关键提示**: 如果你的问题是价值网络崩溃（MAE从0.77→1.04），请直接使用[自定义重置策略](CUSTOM_RESET_GUIDE.md)，效果更好！
