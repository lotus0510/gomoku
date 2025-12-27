"""恢復訓練歷史工具 - 從 checkpoint 文件重建 training_history.json

使用場景：
- training_history.json 被意外刪除或損壞
- 訓練無法自動恢復到正確的迭代
- 需要重建完整的訓練歷史

使用方法：
    python tools/restore_training_history.py

作者：Claude Code
日期：2025-12-27
"""
import torch
import json
import os
import sys
from glob import glob
from pathlib import Path

# 添加項目根目錄到路徑
sys.path.insert(0, str(Path(__file__).parent.parent))


def restore_training_history(checkpoint_dir='checkpoints', backup=True):
    """從所有 checkpoint 文件重建訓練歷史

    Args:
        checkpoint_dir: checkpoint 目錄路徑
        backup: 是否備份現有的 training_history.json

    Returns:
        bool: 是否成功恢復
    """
    history_path = os.path.join(checkpoint_dir, 'training_history.json')

    # 備份現有的歷史文件
    if backup and os.path.exists(history_path):
        backup_path = history_path + '.backup'
        os.rename(history_path, backup_path)
        print(f"📦 已備份現有歷史文件: {backup_path}")

    # 找到所有 checkpoint 文件
    pattern = os.path.join(checkpoint_dir, 'checkpoint_iter_*.pth')
    checkpoints = sorted(glob(pattern),
                        key=lambda x: int(x.split('_')[-1].replace('.pth', '')))

    if not checkpoints:
        print(f"❌ 在 {checkpoint_dir} 未找到任何 checkpoint 文件")
        return False

    print(f"✅ 找到 {len(checkpoints)} 個 checkpoint 文件")
    print(f"   範圍: {os.path.basename(checkpoints[0])} → {os.path.basename(checkpoints[-1])}")

    # 載入最後一個 checkpoint
    last_checkpoint = checkpoints[-1]
    print(f"\n📂 載入最後的 checkpoint: {os.path.basename(last_checkpoint)}")

    try:
        ckpt = torch.load(last_checkpoint, map_location='cpu', weights_only=False)
    except Exception as e:
        print(f"❌ 載入 checkpoint 失敗: {e}")
        return False

    # 檢查 checkpoint 結構
    print(f"   包含的鍵: {list(ckpt.keys())}")

    # 提取訓練歷史
    if 'history' in ckpt:
        history = ckpt['history']
        print(f"\n✅ 找到完整的訓練歷史！")
        print(f"   包含 {len(history['iterations'])} 次迭代")
        print(f"   迭代範圍: {history['iterations'][0]} → {history['iterations'][-1]}")
        print(f"   包含指標: {', '.join(list(history.keys())[:8])}...")

    elif 'training_history' in ckpt:
        history = ckpt['training_history']
        print(f"\n✅ 找到完整的訓練歷史！")
        print(f"   包含 {len(history['iterations'])} 次迭代")

    else:
        print(f"\n⚠️ Checkpoint 不包含訓練歷史數據")
        print(f"   將重建迭代列表...")

        # 只能恢復迭代編號
        iterations = []
        for ckpt_path in checkpoints:
            iter_num = int(ckpt_path.split('_')[-1].replace('.pth', ''))
            iterations.append(iter_num)

        history = {
            'iterations': iterations,
            '_note': '從 checkpoint 文件重建，缺少詳細指標數據',
            '_restored_at': str(datetime.now())
        }
        print(f"   重建迭代: {iterations}")

    # 保存到文件
    try:
        with open(history_path, 'w', encoding='utf-8') as f:
            json.dump(history, f, indent=2, ensure_ascii=False)
        print(f"\n✅ 訓練歷史已保存: {history_path}")
        return True

    except Exception as e:
        print(f"\n❌ 保存失敗: {e}")
        return False


def main():
    """主函數"""
    print("=" * 70)
    print("🔧 訓練歷史恢復工具")
    print("=" * 70)
    print()

    # 檢查當前目錄
    if not os.path.exists('checkpoints'):
        print("❌ 請在項目根目錄執行此腳本")
        print(f"   當前目錄: {os.getcwd()}")
        return 1

    # 執行恢復
    success = restore_training_history()

    if success:
        print("\n" + "=" * 70)
        print("✅ 恢復完成！")
        print("=" * 70)
        print("\n下一步：")
        print("  1. 重新啟動訓練：python train_pipeline_pytorch.py")
        print("  2. 訓練會自動從最後一次迭代繼續")
        print()
        return 0
    else:
        print("\n" + "=" * 70)
        print("❌ 恢復失敗")
        print("=" * 70)
        print("\n建議：")
        print("  1. 檢查 checkpoints/ 目錄是否存在")
        print("  2. 使用 --resume 手動指定 checkpoint：")
        print("     python train_pipeline_pytorch.py --resume checkpoints/checkpoint_iter_XX.pth")
        print()
        return 1


if __name__ == '__main__':
    from datetime import datetime
    sys.exit(main())
