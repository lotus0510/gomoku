#!/usr/bin/env python3
"""分析訓練進展 - 判斷是否需要繼續訓練"""

import json
import pandas as pd
import os

print("=" * 80)
print("訓練進展分析")
print("=" * 80)
print()

# 載入數據
with open('checkpoints/training_history.json', 'r') as f:
    history = json.load(f)

df = pd.read_csv('logs/games/games_summary.csv')

# 基本信息
total_iters = len(history['iterations'])
print(f"📊 總迭代數: {total_iters}")
print(f"📊 總遊戲數: {len(df)}")
print()

# 最近10次迭代分析
last_n = min(10, total_iters)
iterations = history['iterations'][-last_n:]

print("=" * 80)
print(f"最近 {last_n} 次迭代詳細分析")
print("=" * 80)
print()

# 1. 梯度趨勢
print("🔍 梯度範數：")
grad_norms = history['gradient_norm'][-last_n:]
for it, grad in zip(iterations, grad_norms):
    if grad == grad:  # not nan
        status = "✅" if 0.5 <= grad <= 5.0 else ("⚠️" if grad > 5 else "🔴")
        print(f"  迭代 {it:3d}: {grad:6.2f}  {status}")
    else:
        print(f"  迭代 {it:3d}:    nan  🔴")

print()

# 2. 遊戲長度趨勢（關鍵！）
print("🎮 遊戲長度變化：")
recent_iters = df['iteration'].unique()[-last_n:]

lengths = []
for it in recent_iters:
    it_df = df[df['iteration'] == it]
    avg_len = it_df['num_moves'].mean()
    short_pct = len(it_df[it_df['num_moves'] < 15]) / len(it_df) * 100

    lengths.append(avg_len)
    status = "⚠️" if short_pct > 40 else "✅"
    print(f"  迭代 {it:3d}: {avg_len:5.1f} 步（{short_pct:4.1f}% <15步）{status}")

print()

# 趨勢判斷
if len(lengths) >= 5:
    early = sum(lengths[:3]) / 3
    late = sum(lengths[-3:]) / 3

    print("📈 遊戲長度趨勢：")
    print(f"  前3次平均: {early:.1f} 步")
    print(f"  後3次平均: {late:.1f} 步")
    print(f"  變化: {late - early:+.1f} 步")

    if late > early + 5:
        print("  ✅ 遊戲變長！策略正在進化")
    elif late < early - 5:
        print("  ⚠️  遊戲變短，策略可能退化")
    else:
        print("  ⏸️  遊戲長度穩定，等待突破")

print()

# 3. 價值網絡
value_mae = history.get('value_mae', [])[-last_n:]
if value_mae:
    print("📉 價值MAE變化：")
    print(f"  初始: {value_mae[0]:.4f}")
    print(f"  最新: {value_mae[-1]:.4f}")
    print(f"  改善: {value_mae[0] - value_mae[-1]:+.4f}")

    if value_mae[-1] < value_mae[0]:
        print("  ✅ 價值網絡正在改善")
    else:
        print("  ⚠️  價值網絡未改善")

print()

# 4. 總結建議
print("=" * 80)
print("💡 建議")
print("=" * 80)
print()

avg_grad = sum(g for g in grad_norms if g == g) / max(1, len([g for g in grad_norms if g == g]))

if avg_grad < 0.1:
    print("🔴 嚴重問題：梯度消失")
    print("   → 立即停止，降低學習率")
elif avg_grad > 10:
    print("🔴 嚴重問題：梯度爆炸")
    print("   → 立即停止，降低學習率")
elif total_iters < 20:
    print(f"🟡 當前迭代數太少（{total_iters}/建議200+）")
    print("   → 強烈建議繼續訓練")
    print(f"   → 預期在迭代 20-50 之間會看到策略改善")
    print("   → 梯度健康，基礎設施完善，可以放心訓練")
elif lengths[-1] > lengths[0]:
    print("🟢 遊戲長度正在增加")
    print("   → 策略正在進化，繼續訓練")
else:
    print("🟡 遊戲長度未明顯變化")
    print("   → 可能需要更多迭代才能突破")
    print(f"   → 建議至少訓練到 {total_iters + 30} 次迭代")

print()

# 5. 下一步
print("🎯 下一步行動：")
if avg_grad > 0.5 and avg_grad < 5:
    print("  1. ✅ 繼續訓練（梯度健康）")
    print(f"  2. 每 5-10 次迭代檢查一次進展")
    print(f"  3. 到迭代 50 時再做評估")
    print()
    print("  執行：")
    print("    python train_pipeline_pytorch.py")
    print()
    print("  監控（另開終端）：")
    print("    python analyze_progress.py")
else:
    print("  ⚠️  先修復梯度問題再繼續")

print()
print("=" * 80)
