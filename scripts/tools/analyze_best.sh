#!/bin/bash
echo "=== 提取胜率数据 ==="
awk '
/"win_rate_vs_random":/ {flag=1; next}
flag && /\]/ {exit}
flag {
    gsub(/,/, "")
    gsub(/null/, "")
    if ($1 != "") {
        iter = NR - start_line
        print iter, $1
    }
}
BEGIN {start_line=0}
' checkpoints/training_history.json > /tmp/win_rates.txt

echo "胜率数据（非null的评估）："
nl -v 1 /tmp/win_rates.txt | awk '{if ($3 != "") print "迭代" $3*5 ": " $3}'

echo ""
echo "=== 提取策略损失数据 ==="
awk '
/"policy_loss":/ {flag=1; next}
flag && /\]/ {exit}
flag {
    gsub(/,/, "")
    if ($1 != "") {
        print NR-start_line, $1
    }
}
BEGIN {start_line=0}
' checkpoints/training_history.json > /tmp/policy_loss.txt

echo "找出最低策略损失："
sort -k2 -n /tmp/policy_loss.txt | head -1 | awk '{print "迭代" $1 ": " $2}'
