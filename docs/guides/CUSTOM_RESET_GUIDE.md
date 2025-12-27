# 自定义重置策略 - 完整指南

**目的**: 防止价值网络崩潰（MAE从0.77→1.04）
**效果**: 节省15-25小时训练时间，保持MAE在0.75-0.85健康范围
**状态**: ✅ 代码已就绪，仅需整合

---

## 目录

1. [快速开始](#快速开始)
2. [核心机制](#核心机制)
3. [整合步骤](#整合步骤)
4. [配置说明](#配置说明)
5. [启动检查清单](#启动检查清单)
6. [监控与调试](#监控与调试)
7. [vs原方案对比](#vs原方案对比)
8. [常见问题](#常见问题)

---

## 快速开始

### 3步骤快速整合（15分钟）

#### 步骤1: 导入策略（1分钟）

在 `train_pipeline_pytorch.py` 顶部添加:

```python
from training.custom_reset_strategy import CustomResetStrategy
```

#### 步骤2: 初始化策略（2分钟）

在 `main()` 函数中，config初始化后添加:

```python
def main():
    config = TrainingConfig.get_optimized_config()

    # 启用自定义重置策略
    reset_strategy = CustomResetStrategy(target_iteration=40)
    print("✅ 自定义重置策略已启用")

    # ... 其他初始化 ...
```

#### 步骤3: 添加检查逻辑（12分钟）

在迭代循环末尾，checkpoint保存后添加:

```python
for iteration in range(start_iteration, config.ITERATIONS):
    # ... 自我对弈、训练、保存checkpoint ...

    # 检查是否需要重置
    if reset_strategy.should_trigger(training_history, iteration):
        config, replay_buffer, model = reset_strategy.execute_reset(
            config, replay_buffer, model, iteration
        )
        # 同步优化器学习率
        for pg in optimizer.param_groups:
            pg['lr'] = config.LEARNING_RATE
```

---

## 核心机制

### 触发条件

当同时满足以下条件时触发重置:

1. ✅ 当前迭代 >= 50
2. ✅ 距上次重置 >= 20次迭代
3. ✅ Value Loss 连续上升4次（检查最近5次）
4. ✅ MAE 在恶化（最近5次，末尾>开头）
5. ✅ MAE 超过健康值（> 0.85）

### 重置动作

```
检测到价值网络崩潰 →
  1. 加载迭代40的健康模型权重
  2. 清空最新50%的崩潰数据
  3. 学习率提升50%（0.0001→0.00015）
  4. 价值权重翻倍（1.0→2.0）
  5. 从健康状态重新训练
```

### 预期收益

基于实际训练数据（迭代135，MAE从0.77崩到1.04）:

**不使用重置策略**:
- 迭代1→30: MAE 0.98→0.77 ✅ 健康
- 迭代30→135: MAE 0.77→1.04 ❌ 崩潰
- 浪费: 105次迭代 × 16分钟 = 28小时

**使用重置策略**:
- 迭代1→30: MAE 0.98→0.77 ✅ 健康
- 迭代30→55: MAE 0.77→0.87 ⚠️ 开始恶化
- 迭代55: 🔄 触发重置，回滚到迭代40
- 迭代40→90: MAE 0.82→0.75 ✅ 重新优化
- 节省: ~20小时

---

## 整合步骤

### 准备工作

#### 1. 确认文件已创建

```bash
ls training/custom_reset_strategy.py
```

#### 2. 测试策略代码

```bash
python training/custom_reset_strategy.py
```

预期输出:
```
============================================================
自定义重置策略测试
============================================================
...
✅ 检测逻辑正常工作!
```

### 修改训练脚本

#### 1. 备份原始文件（推荐）

```bash
cp train_pipeline_pytorch.py train_pipeline_pytorch.py.backup
```

#### 2. 添加import

在文件顶部添加:
```python
from training.custom_reset_strategy import CustomResetStrategy
```

#### 3. 初始化策略

在 `main()` 函数中:
```python
def main():
    config = TrainingConfig.get_optimized_config()

    # 初始化重置策略
    reset_strategy = CustomResetStrategy(
        target_iteration=40,      # 回滚目标
        min_iteration=50,         # 最早触发
        min_reset_interval=20     # 重置间隔
    )
    print("✅ 价值崩潰重置策略已启用")
    print(f"   回滚目标: 迭代 40")
    print(f"   触发条件: Value Loss连续5次上升 + MAE>0.85")

    # ... 其他初始化 ...
```

#### 4. 添加检查逻辑

在迭代循环末尾:
```python
for iteration in range(start_iteration, config.ITERATIONS):
    # ... 自我对弈 ...
    # ... 训练网絡 ...
    # ... 保存checkpoint ...
    # ... 更新history ...

    # 检查是否需要重置
    if reset_strategy.should_trigger(training_history, iteration):
        config, replay_buffer, model = reset_strategy.execute_reset(
            config, replay_buffer, model, iteration
        )
        # 同步优化器学习率
        for pg in optimizer.param_groups:
            pg['lr'] = config.LEARNING_RATE
```

#### 5. 验证语法

```bash
python -m py_compile train_pipeline_pytorch.py
```

---

## 配置说明

### 关键参数

| 参数 | 默认值 | 说明 | 调整建议 |
|------|-------|------|---------|
| `target_iteration` | 40 | 回滚目标迭代 | 30-50之间，选择MAE较低点 |
| `min_iteration` | 50 | 最早触发迭代 | 建议>=50，确保足够数据 |
| `min_reset_interval` | 20 | 重置间隔 | 避免频繁重置 |

### Checkpoint频率

确保在 `training/config.py` 中设置:

```python
CHECKPOINT_FREQUENCY = 5  # 或 1（推荐每次都保存）
```

### 参数调整

重置后会自动调整参数:

| 参数 | 原值 | 第1次重置 | 第2次重置 | 第3次重置 |
|------|------|----------|----------|----------|
| 学习率 | 0.0001 | 0.00015 (+50%) | 0.0002 (+100%) | 0.00025 (+150%) |
| 探索率(ε) | 0.25 | 0.35 | 0.40 | 0.45 |
| 温度(τ) | 1.0 | 1.15 | 1.30 | 1.45 |
| 价值权重 | 1.0 | 2.0 | 2.5 | 3.0 |

---

## 启动检查清单

### 阶段1: 准备工作（5分钟）

```
□ training/custom_reset_strategy.py 已存在
□ venv环境已激活
□ 策略测试通过
```

### 阶段2: 代码修改（10分钟）

```
□ 已添加import
□ 已初始化reset_strategy
□ 已添加检查逻辑
□ 语法检查通过
```

### 阶段3: 配置检查（3分钟）

```
□ CHECKPOINT_FREQUENCY >= 5
□ 使用优化配置
□ checkpoint目录已创建
```

### 阶段4: 启动训练

```bash
# 激活venv
venv\Scripts\activate

# 开始训练
python train_pipeline_pytorch.py
```

### 阶段5: 确认启动成功

应该看到:
```
============================================================
✅ 价值崩潰重置策略已启用
   回滚目标: 迭代 40
   触发条件: Value Loss连续5次上升 + MAE>0.85
   最早触发: 迭代 50
============================================================
```

---

## 监控与调试

### 观察训练进度

**关键指标**:
- 迭代10: MAE应该 < 0.95
- 迭代20: MAE应该 < 0.85
- 迭代30: MAE应该 < 0.80（目标: ~0.77）

### 等待首次重置

**预期时机**: 迭代50-60

**重置訊息**:
```
============================================================
🚨 检测到价值网絡崩潰!
============================================================
当前迭代: 55

Value Loss (最近5次): ['1.00', '1.02', '1.05', '1.08', '1.10']
MAE (最近5次):        ['0.85', '0.87', '0.89', '0.91', '0.93']

============================================================
🔄 执行价值崩潰重置策略 (第1次)
============================================================
触发迭代: 55
回滚目标: 迭代 40

✅ 模型已回滚到迭代 40
✅ 已清空 5000 条新数据
✅ 重置完成!
============================================================
```

### 查看重置记录

```bash
# 查看重置历史
cat checkpoints/reset_logs/reset_history.json

# 格式化输出
python -c "
import json
with open('checkpoints/reset_logs/reset_history.json') as f:
    history = json.load(f)
    print(f'重置次数: {len(history)}')
    for record in history:
        print(f'  迭代{record[\"trigger_iteration\"]} → {record[\"target_iteration\"]}')
"
```

### 检查重置效果

重置后10次迭代内:
- 迭代56: MAE应该回到 ~0.82（迭代40的水平）
- 迭代65: MAE应该改善到 ~0.78

---

## vs原方案对比

### 关键差异

| 方面 | 原报告方案 | 自定义方案 | 评价 |
|------|-----------|-----------|------|
| **触发时机** | 较晚（分数>=80） | 更早（VL连续上升） | ✅ 自定义更及时 |
| **模型处理** | 保持当前 | **回滚到健康点** | ✅ 自定义更优 |
| **数据清理** | 清舊留新 | **清新留舊** | ✅ 自定义更合理 |
| **学习率** | +900% | +50-150% | ✅ 自定义更稳定 |
| **适用性** | 通用停滞 | 价值崩潰 | ✅ 对症下药 |

### 为什么自定义方案更优？

**关键创新**:
1. 🎯 **回滚到健康checkpoint**（不是从崩潰状态继续）
2. 🧹 **清除崩潰期数据**（不是清除健康期数据）
3. 🎚️ **稳健调参**（不是激进10倍LR）

**您的情况**: 价值崩潰 → **自定义方案完胜** 🏆

---

## 常见问题

### Q1: 没有迭代40的checkpoint怎么办？

**A**: 策略会自动寻找最接近的checkpoint（35, 30, 25...）

确保设置:
```python
CHECKPOINT_FREQUENCY = 5  # 每5次迭代保存一次
# 或
CHECKPOINT_FREQUENCY = 1  # 每次都保存（推荐）
```

### Q2: 重置后训练会从迭代40重新开始吗？

**A**: 不会！
- 迭代计数器不变（如果在55触发，下一次仍是56）
- 只是模型权重回滚到迭代40的状态
- 训练历史保留，可以看到重置的效果

### Q3: 如何禁用自动重置？

**A**: 注释掉检查代码:
```python
# if reset_strategy.should_trigger(...):
#     ...
```

或添加开关:
```python
ENABLE_CUSTOM_RESET = False

if ENABLE_CUSTOM_RESET:
    if reset_strategy.should_trigger(...):
        # ...
```

### Q4: 重置会影响策略网絡吗？

**A**: 会，但影响很小

```
迭代55策略损失: 5.30
迭代40策略损失: 5.38（倒退1.5%）

但是:
  策略会在10-15次迭代内重新学回来
  价值网絡如果继续崩潰，需要50+次才能修复

短期倒退 vs 长期健康 → 值得！
```

### Q5: 触发太频繁怎么办？

**A**: 提高MAE阈值
```python
# 在 custom_reset_strategy.py 中修改
mae_unhealthy = recent_mae[-1] > 0.90  # 从0.85改为0.90
```

### Q6: 从未触发重置是正常的吗？

**A**: 如果MAE一直<0.85，这是好事！不需要重置。

检查MAE趋势:
```python
import json
with open('checkpoints/training_history.json') as f:
    h = json.load(f)
    mae = h['value_mae'][-10:]
    print('最近MAE:', mae)
```

---

## 安全机制

1. **重置上限**: 最多3次，避免无限循环
2. **间隔保护**: 两次重置至少间隔20次迭代
3. **Checkpoint检查**: 找不到目标checkpoint会自动寻找替代
4. **完整日志**: 所有重置记录自动保存到 `checkpoints/reset_logs/`

---

## 故障排除

### 问题1: 训练崩溃

**错误**: `RuntimeError: CUDA out of memory`

**解决**:
```python
# 在config.py中降低批次大小
BATCH_SIZE = 512  # 从1024降到512
```

### 问题2: 找不到模块

**错误**: `ModuleNotFoundError: No module named 'training.custom_reset_strategy'`

**解决**:
```bash
# 确认文件存在
ls training/custom_reset_strategy.py

# 确认在正确目录
pwd  # 应该在 .../gomoku 目录

# 确认venv已激活
which python  # 应该指向venv
```

### 问题3: 重置后仍然崩潰

**可能原因**:
1. 迭代40的checkpoint本身就不健康
2. 参数调整不够

**解决**:
```python
# 修改target_iteration为更早的迭代
reset_strategy = CustomResetStrategy(
    target_iteration=30,  # 从40改为30
)
```

---

## 相关文档

| 文档 | 用途 |
|------|------|
| `training/custom_reset_strategy.py` | 策略源代码 |
| `docs/troubleshooting/TRAINING_FAILURE_REPORT.md` | 训练故障诊断 |
| `docs/guides/MONITORING_METRICS.md` | 监控指标说明 |

---

## 成功标准

训练成功的标志:

1. ✅ **启动阶段**: 看到"✅ 价值崩潰重置策略已启用"
2. ✅ **训练初期**（1-30）: MAE从0.98→0.77
3. ✅ **首次重置**（50-60）: 成功触发并回滚
4. ✅ **重置后**: MAE回到0.82左右
5. ✅ **稳定期**（70+）: MAE保持在0.75-0.85
6. ✅ **游戏质量**: 长度50-70步，多样性好

---

**预祝训练成功！** 🎉

如果MAE保持在0.75-0.85，说明策略生效了！
