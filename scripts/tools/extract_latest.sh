#!/bin/bash
# 提取最新20次迭代的关键数据
awk '
/"policy_loss":/ {flag=1; next}
flag && /\]/ {flag=0}
flag {gsub(/,/, ""); if (NF>0) print}
' checkpoints/training_history.json | tail -20 | nl -v 109

echo "---"
echo "策略损失（迭代109-128）"
