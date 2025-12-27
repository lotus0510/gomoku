#!/bin/bash
echo "=== 验证训练数据 ==="
echo ""

# 检查training_history.json中迭代125的数据
echo "迭代125的数据:"
python3 -c "
import json
with open('checkpoints/training_history.json', 'r') as f:
    h = json.load(f)
idx = 124  # 迭代125是索引124
print(f'策略损失: {h[\"policy_loss\"][idx]:.4f}')
print(f'总损失: {h[\"total_loss\"][idx]:.4f}')
print(f'价值损失: {h[\"value_loss\"][idx]:.4f}')
print(f'胜率: {h[\"win_rate_vs_random\"][idx]}')
" 2>&1 || echo "Python脚本执行失败"

echo ""
echo "=== 对比摘要文件 ==="
grep -A 3 "基本指標" checkpoints/summaries/iteration_0125_summary.txt

echo ""
echo "=== 检查最近的评估结果 ==="
for i in 120 125 130; do
    if [ -f "checkpoints/summaries/iteration_$(printf "%04d" $i)_summary.txt" ]; then
        echo "迭代$i:"
        grep "對隨機玩家勝率" "checkpoints/summaries/iteration_$(printf "%04d" $i)_summary.txt"
    fi
done
