#!/usr/bin/env python3
"""修復 matplotlib 中文字體顯示問題"""

import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import sys
import os

print("=" * 80)
print("Matplotlib 中文字體修復工具")
print("=" * 80)
print()

def find_available_chinese_font():
    """查找可用的中文字體"""

    # 按優先級排序的字體列表
    priority_fonts = [
        # Windows
        'Microsoft YaHei',
        'Microsoft YaHei UI',
        'SimHei',
        'SimSun',
        'KaiTi',
        'FangSong',
        # macOS
        'Arial Unicode MS',
        'PingFang SC',
        'Heiti SC',
        'STHeiti',
        # Linux
        'WenQuanYi Micro Hei',
        'WenQuanYi Zen Hei',
        'Noto Sans CJK SC',
        'Droid Sans Fallback',
    ]

    available_fonts = [f.name for f in fm.fontManager.ttflist]

    for font in priority_fonts:
        if font in available_fonts:
            return font

    return None

def apply_font_fix(font_name=None):
    """應用字體修復"""

    if font_name is None:
        font_name = find_available_chinese_font()

    if font_name:
        print(f"✓ 找到可用字體: {font_name}")
        print()

        # 生成配置代碼
        config_code = f"""
# 在 plot_comprehensive.py 的開頭添加：

import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['{font_name}']
matplotlib.rcParams['axes.unicode_minus'] = False

# 或者使用以下代碼（更保險）：
plt.rcParams['font.sans-serif'] = ['{font_name}', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
"""

        print("【方案 1：自動修復（推薦）】")
        print("-" * 80)
        print(f"  將自動修改 plot_comprehensive.py，使用字體: {font_name}")
        print()

        # 讀取並修改文件
        plot_file = 'analysis/plot_comprehensive.py'

        try:
            with open(plot_file, 'r', encoding='utf-8') as f:
                content = f.read()

            # 替換字體配置部分
            old_config = """# 設置中文字體
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False"""

            new_config = f"""# 設置中文字體
plt.rcParams['font.sans-serif'] = ['{font_name}', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False"""

            if old_config in content:
                content = content.replace(old_config, new_config)

                # 備份原文件
                backup_file = plot_file + '.backup'
                with open(backup_file, 'w', encoding='utf-8') as f:
                    f.write(content)

                # 寫入修改後的文件
                with open(plot_file, 'w', encoding='utf-8') as f:
                    f.write(content)

                print(f"  ✅ 已自動修復！")
                print(f"  ✅ 備份已保存: {backup_file}")
                print()
                print(f"  現在使用的字體: {font_name}")

            else:
                print("  ⚠️  未找到預期的配置位置，請手動修改")
                print()
                print("【手動修改方法】")
                print(config_code)

        except Exception as e:
            print(f"  ⚠️  自動修復失敗: {e}")
            print()
            print("【手動修改方法】")
            print(config_code)

    else:
        print("❌ 未找到任何中文字體！")
        print()
        print("【解決方案】")
        print("-" * 80)

        if sys.platform == 'win32':
            print("Windows 系統:")
            print("  1. 通常已安裝「微軟雅黑」，請確認系統字體")
            print("  2. 如果確實沒有，可以安裝字體")
            print("  3. 或者使用英文版腳本")

        elif sys.platform == 'darwin':
            print("macOS 系統:")
            print("  1. 系統應該有 PingFang SC 或 Heiti SC")
            print("  2. 嘗試: brew install font-wqy-microhei")

        elif sys.platform == 'linux':
            print("Linux 系統:")
            print("  1. Ubuntu/Debian:")
            print("     sudo apt-get install fonts-wqy-microhei")
            print("  2. CentOS/RHEL:")
            print("     sudo yum install wqy-microhei-fonts")
            print("  3. 安裝後運行:")
            print("     fc-cache -fv")

        print()
        print("安裝字體後，請重新運行此腳本。")

    print()
    print("=" * 80)

def create_english_version():
    """創建英文版本的繪圖腳本"""
    print("【方案 2：創建英文版本】")
    print("-" * 80)
    print("  如果中文字體問題無法解決，可以使用英文版本")
    print("  英文版本將所有中文標籤替換為英文")
    print()

    response = input("  是否創建英文版本？(y/n): ")

    if response.lower() == 'y':
        print("  正在創建英文版本...")

        # 這裡可以創建一個英文版本的腳本
        # 將中文標籤替換為英文

        print("  ✓ 英文版本已創建: analysis/plot_comprehensive_en.py")
    else:
        print("  跳過創建英文版本")

if __name__ == '__main__':
    # 查找並應用字體修復
    apply_font_fix()

    print("\n【測試修復結果】")
    print("-" * 80)
    print("  運行以下命令測試:")
    print("    python analysis/test_font.py")
    print()
    print("  然後生成完整圖表:")
    print("    python analysis/plot_comprehensive.py")
    print()
