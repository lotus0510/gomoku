# Shell工具脚本

这个目录包含用于快速数据查询和验证的Shell工具脚本。

## 脚本列表

### analyze_best.sh
**功能**: 从训练历史JSON中提取胜率和策略损失数据

**使用方法**:
```bash
cd ../../  # 回到项目根目录
bash scripts/tools/analyze_best.sh
```

**输出**: 显示最佳迭代的胜率和策略损失信息

---

### extract_latest.sh
**功能**: 提取最近20次迭代的策略损失数据

**使用方法**:
```bash
cd ../../
bash scripts/tools/extract_latest.sh
```

**输出**: 最新训练数据的快速预览

---

### verify_data.sh
**功能**: 验证特定迭代的训练数据完整性

**使用方法**:
```bash
cd ../../
bash scripts/tools/verify_data.sh [迭代数]
```

**输出**: 数据验证结果

---

## 注意事项

1. **执行位置**: 这些脚本需要在项目根目录执行，因为它们读取 `checkpoints/training_history.json`
2. **依赖**: 需要安装 `jq` 命令行JSON处理工具（部分脚本）
3. **用途**: 这些是轻量级快速查询工具，用于替代完整的Python分析脚本

## 替代方案

对于更详细的分析，推荐使用 `analysis/` 目录下的Python脚本：
- `analysis/check_training.py` - 统一训练检查工具
- `analysis/diagnose_training.py` - 完整训练诊断
- `analysis/plot_comprehensive.py` - 可视化分析

---

**最后更新**: 2025-12-26
