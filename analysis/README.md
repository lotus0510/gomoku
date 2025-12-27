# Analysis 分析工具

简洁的训练分析工具集。

---

## 🚀 使用方法

### 生成训练图表（最常用）

```bash
python analysis/plot_comprehensive.py
```

**输出**: `analysis/plots/` 目录下的10张详细图表

**查看结果**:
- 最重要：`plots/9_comprehensive_dashboard.png` - 综合仪表板
- 其他：`plots/1-10` 各种详细分析图表

---

## 📊 可用工具

### plot_comprehensive.py ⭐⭐⭐⭐⭐
**最重要的工具** - 生成10张详细训练分析图表

**输出图表**:
1. 训练指标总览（损失、梯度、学习率、胜率）
2. 损失分解分析（策略/价值/总损失）
3. 梯度健康度分析
4. 价值网络指标（MAE）
5. 游戏长度分布
6. 胜率趋势（黑白对比）
7. 策略演化（edge-rush问题）
8. 游戏模式分析
9. **综合仪表板**（一页总览）⭐
10. 相关性矩阵

---

### monitor_training.py ⭐⭐⭐
实时训练监控工具

**使用场景**: 训练进行中，在单独的终端运行

**特点**: 每5秒自动刷新，显示最新训练状态

```bash
python analysis/monitor_training.py
```

---

### diagnose_training.py ⭐⭐⭐
训练健康度诊断工具

**使用场景**: 发现训练异常时

**检查项**:
- 胜率平衡性（黑白棋平衡）
- 游戏长度异常（edge-rush）
- 策略质量
- 价值网络健康（MAE检查）
- 梯度健康（消失/爆炸）
- 训练稳定性

```bash
python analysis/diagnose_training.py
```

---

## 📁 目录结构

```
analysis/
├── plot_comprehensive.py       生成图表（最常用）
├── monitor_training.py         实时监控
├── diagnose_training.py        健康诊断
├── plots/                      图表输出目录
└── README.md                   本文件
```

---

## 📊 图表说明

### 重点关注的3张图表

| 图表 | 文件名 | 关键指标 |
|------|-------|---------|
| 综合仪表板 | 9_comprehensive_dashboard.png | 总体健康度（最重要）|
| 训练指标 | 1_training_metrics.png | 损失趋势 |
| 价值网络 | 4_value_network_metrics.png | MAE: 0.70-0.85 ✅ |

### 问题诊断图表

| 图表 | 诊断什么 |
|------|---------|
| 2_loss_breakdown.png | 哪种损失异常 |
| 3_gradient_analysis.png | 梯度消失/爆炸 |
| 5_game_length_distribution.png | 套路化（游戏过短）|
| 6_win_rate_trends.png | 黑白失衡 |
| 7_strategy_evolution.png | Edge-rush问题 |

---

## 🎯 推荐工作流

### 日常使用（训练后）

```bash
# 1. 生成图表
python analysis/plot_comprehensive.py

# 2. 查看综合仪表板
# Windows: start analysis/plots/9_comprehensive_dashboard.png
# Linux: xdg-open analysis/plots/9_comprehensive_dashboard.png

# 3. 完成！
```

### 训练中监控（可选）

```bash
# 在单独的终端运行
python analysis/monitor_training.py
```

### 发现问题时

```bash
# 运行诊断工具
python analysis/diagnose_training.py

# 查看对应的图表定位问题
```

---

## ⚠️ 快速问题诊断

| 症状 | 查看图表 | 健康标准 |
|------|---------|---------|
| 训练不稳定 | 1_training_metrics.png | 损失持续下降 |
| 价值崩溃 | 4_value_network_metrics.png | MAE: 0.70-0.85 |
| 套路化 | 5_game_length_distribution.png | 游戏长度50-70步 |
| 黑白失衡 | 6_win_rate_trends.png | 胜率45-55% |
| 梯度问题 | 3_gradient_analysis.png | 梯度0.5-5.0 |

---

## 💡 关键提示

**90%的时间**: 只需要运行 `plot_comprehensive.py`

**查看仪表板**: `plots/9_comprehensive_dashboard.png` 一目了然

**其他工具**: monitor 和 diagnose 按需使用

---

**最后更新**: 2025-12-26

**核心理念**: 简洁、直观、专注可视化
