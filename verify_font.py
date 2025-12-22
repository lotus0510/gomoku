#!/usr/bin/env python3
"""快速驗證字體配置"""

import matplotlib.pyplot as plt
import seaborn as sns

# 模擬 plot_comprehensive.py 的字體設置
sns.set_style("whitegrid")
sns.set_palette("husl")

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimSun', 'Microsoft JhengHei']
plt.rcParams['axes.unicode_minus'] = False

# 檢查配置
print("=" * 80)
print("字體配置驗證")
print("=" * 80)
print(f"\n當前 font.sans-serif 配置: {plt.rcParams['font.sans-serif']}")
print(f"當前 axes.unicode_minus 配置: {plt.rcParams['axes.unicode_minus']}")

# 生成測試圖表
fig, ax = plt.subplots(figsize=(10, 6))
ax.set_title('測試中文標題 - 訓練指標', fontsize=16, fontweight='bold')
ax.set_xlabel('迭代次數', fontsize=12)
ax.set_ylabel('損失值', fontsize=12)
ax.plot([1, 2, 3, 4, 5], [5, 4, 3, 2, 1], label='總損失')
ax.legend()
ax.grid(True)

plt.savefig('verify_font.png', dpi=150, bbox_inches='tight')
print("\n✓ 測試圖表已保存: verify_font.png")
print("\n請打開 verify_font.png 檢查中文是否正常顯示")
print("=" * 80)
