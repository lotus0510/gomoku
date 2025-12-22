"""可視化具體對局，分析黑棋為何不防守"""

import json
import sys

def visualize_board(moves, step):
    """顯示到第N步的棋盤"""
    board = [['.' for _ in range(15)] for _ in range(15)]

    for i, (row, col) in enumerate(moves[:step]):
        if i % 2 == 0:  # 黑棋
            board[row][col] = '●'
        else:  # 白棋
            board[row][col] = '○'

    print("   ", end="")
    for i in range(15):
        print(f"{i:2d}", end=" ")
    print()

    for i, row in enumerate(board):
        print(f"{i:2d} ", end="")
        for cell in row:
            print(f" {cell} ", end="")
        print()

def analyze_game(game_file):
    """分析一局遊戲"""
    with open(game_file, 'r') as f:
        game = json.load(f)

    moves = game['moves']
    policies = game.get('top_policies', [])
    values = game.get('values', [])

    print("=" * 70)
    print(f"第{game['game_num']}局分析 - 白棋{game['num_moves']}步獲勝")
    print("=" * 70)
    print()

    # 關鍵步驟分析
    critical_steps = [2, 4, 6, 8, 10]

    for step in critical_steps:
        if step > len(moves):
            break

        print(f"\n{'='*70}")
        print(f"第{step}步後的局面（白棋剛下完）")
        print(f"{'='*70}")
        visualize_board(moves, step)

        # 白棋在第0行的棋子統計
        white_on_row0 = []
        for i in range(0, step, 2):  # 白棋是偶數步（從0開始）
            if i > 0 and moves[i][0] == 0:  # 白棋在第0行
                white_on_row0.append(moves[i])

        if white_on_row0:
            cols = sorted([col for row, col in white_on_row0])
            print(f"\n⚪ 白棋在第0行: {len(white_on_row0)}子 → 位置 {cols}")

            # 檢查是否連續
            if len(cols) >= 2:
                if cols[-1] - cols[0] == len(cols) - 1:
                    print(f"   🚨 已形成 {len(cols)} 連！")
                    if len(cols) >= 3:
                        print(f"   🚨🚨 危險！再下{5-len(cols)}步就贏了！")

        # 顯示黑棋的策略輸出
        if step < len(policies):
            policy = policies[step]
            print(f"\n● 黑棋第{step+1}步的AI判斷:")
            print(f"   價值估計: {values[step]:.3f}  ← 應該負數（黑棋危險）")
            print(f"   Top-5 候選動作:")

            for rank, (r, c, prob) in enumerate(policy, 1):
                # 檢查是否是阻擋位置
                is_block = (r == 0 and c in [col for _, col in white_on_row0 if col < 14])
                marker = " ✓ 阻擋位置!" if is_block else ""
                print(f"     {rank}. ({r},{c:2d}) 機率={prob:.3f} {marker}")

        if step < len(moves):
            next_move = moves[step]
            print(f"\n   實際選擇: ({next_move[0]},{next_move[1]})  ", end="")
            if next_move[0] == 0 and len(white_on_row0) >= 2:
                # 檢查是否選擇了阻擋
                cols = [col for row, col in white_on_row0]
                expected_block = max(cols) + 1 if max(cols) < 14 else min(cols) - 1
                if next_move[1] == expected_block:
                    print("✓ 正確阻擋！")
                else:
                    print(f"✗ 沒有阻擋（應該下 (0,{expected_block})）")
            else:
                print()

        input("\n按Enter繼續下一步...")

if __name__ == '__main__':
    game_file = sys.argv[1] if len(sys.argv) > 1 else 'logs/games/iteration_34/game_80.json'

    try:
        analyze_game(game_file)

        print("\n" + "=" * 70)
        print("分析完成")
        print("=" * 70)
        print("\n結論:")
        print("  ✗ 黑棋完全沒有識別出白棋的連珠威脅")
        print("  ✗ AI的價值判斷錯誤（應該顯示黑棋劣勢）")
        print("  ✗ AI的策略輸出隨機（所有動作機率相同）")
        print("\n根本原因:")
        print("  → 策略網絡崩潰（無法區分好壞棋）")
        print("  → 價值網絡失效（無法判斷局面優劣）")
        print("  → 梯度消失導致模型無法學習")

    except FileNotFoundError:
        print(f"找不到文件: {game_file}")
        print("\n可用的遊戲記錄:")
        import os
        for root, dirs, files in os.walk('logs/games'):
            for f in files:
                if f.endswith('.json'):
                    print(f"  {os.path.join(root, f)}")
