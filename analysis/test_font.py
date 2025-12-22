#!/usr/bin/env python3
"""測試 matplotlib 中文字體顯示"""

import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

print("=" * 80)
print("Matplotlib 中文字體測試")
print("=" * 80)
print()

# 1. 列出可用的中文字體
print("【可用的中文字體】")
print("-" * 80)

chinese_fonts = []
for font in fm.fontManager.ttflist:
    font_name = font.name
    # 檢查是否包含中文相關關鍵字
    if any(keyword in font_name.lower() for keyword in
           ['chinese', 'cjk', 'simhei', 'simsun', 'yahei', 'microsoft', 'noto', 'wenquanyi', 'fang']):
        chinese_fonts.append(font_name)
        print(f"  ✓ {font_name}")

if not chinese_fonts:
    print("  ⚠️  未找到明確的中文字體")
    print("\n【所有可用字體（前20個）】")
    for i, font in enumerate(fm.fontManager.ttflist[:20]):
        print(f"  {i+1}. {font.name}")
else:
    print(f"\n找到 {len(chinese_fonts)} 個中文字體")

print()

# 2. 測試當前配置
print("【當前字體配置】")
print("-" * 80)
print(f"sans-serif: {plt.rcParams['font.sans-serif']}")
print(f"當前使用字體: {plt.rcParams['font.family']}")
print()

# 3. 生成測試圖表
print("【生成測試圖表】")
print("-" * 80)

fig, axes = plt.subplots(2, 2, figsize=(12, 10))
fig.suptitle('中文字體測試 - 訓練指標示例', fontsize=16, fontweight='bold')

# 測試不同字體
test_fonts = ['Microsoft YaHei', 'SimHei', 'Arial Unicode MS', 'sans-serif']

for idx, (ax, font_name) in enumerate(zip(axes.flat, test_fonts)):
    try:
        # 設置字體
        ax.set_title(f'使用字體: {font_name}', fontsize=12, fontweight='bold')

        # 繪製簡單圖表
        x = [1, 2, 3, 4, 5]
        y = [2, 4, 3, 5, 6]

        ax.plot(x, y, 'b-', linewidth=2, marker='o', markersize=8, label='訓練損失')
        ax.set_xlabel('迭代次數', fontsize=11, fontname=font_name if font_name != 'sans-serif' else None)
        ax.set_ylabel('損失值', fontsize=11, fontname=font_name if font_name != 'sans-serif' else None)
        ax.legend(prop={'size': 10})
        ax.grid(True, alpha=0.3)

        # 添加中文註解
        ax.text(3, 4.5, '中文測試 ✓', fontsize=14,
               bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5),
               fontname=font_name if font_name != 'sans-serif' else None)

    except Exception as e:
        ax.text(0.5, 0.5, f'字體載入失敗:\n{font_name}\n{str(e)}',
               ha='center', va='center', transform=ax.transAxes,
               fontsize=10, color='red')

plt.tight_layout()

# 保存測試圖表
output_path = 'analysis/font_test.png'
plt.savefig(output_path, dpi=150, bbox_inches='tight')
print(f"  ✓ 測試圖表已保存: {output_path}")
print()

# 4. 檢測結果
print("【檢測結果】")
print("-" * 80)

if chinese_fonts:
    print("  ✅ 找到中文字體")
    print("  ✅ 測試圖表已生成")
    print()
    print("  建議動作：")
    print("    1. 打開 analysis/font_test.png")
    print("    2. 檢查中文是否正常顯示")
    print("    3. 如果顯示正常，可以使用 plot_comprehensive.py")
    print("    4. 如果顯示異常（方框□），看下面的解決方案")
else:
    print("  ⚠️  未找到明確的中文字體")
    print()
    print("  建議動作：")
    print("    1. 先查看 font_test.png，確認是否真的有問題")
    print("    2. 如果有問題，看下面的解決方案")

print()
print("=" * 80)
