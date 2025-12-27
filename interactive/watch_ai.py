import pygame
import sys
import os
import random
import glob
import json
import numpy as np
import torch

# 添加專案根目錄到 Python 路徑
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from game import GomokuGame
from core.neural_net import create_enhanced_model
from training.config import TrainingConfig

# --- 常數 ---
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
BOARD_COLOR = (218, 185, 123)
TEXT_COLOR = BLACK
STATUS_BAR_BG = (150, 150, 150)

BOARD_SIZE = 15
GRID_SIZE = 40
MARGIN = 20
STONE_RADIUS = GRID_SIZE // 2 - 2
STATUS_BAR_HEIGHT = 50


BOARD_WIDTH = (BOARD_SIZE - 1) * GRID_SIZE + 2 * MARGIN
BOARD_HEIGHT = (BOARD_SIZE - 1) * GRID_SIZE + 2 * MARGIN
WINDOW_WIDTH = BOARD_WIDTH
WINDOW_HEIGHT = BOARD_HEIGHT + STATUS_BAR_HEIGHT
BOARD_START_Y = STATUS_BAR_HEIGHT

# --- 繪圖函數 ---
def draw_grid_and_stars(surface):
    surface.fill(BOARD_COLOR)
    for i in range(BOARD_SIZE):
        pygame.draw.line(surface, BLACK, (MARGIN, MARGIN + i * GRID_SIZE), (BOARD_WIDTH - MARGIN, MARGIN + i * GRID_SIZE), 1)
        pygame.draw.line(surface, BLACK, (MARGIN + i * GRID_SIZE, MARGIN), (MARGIN + i * GRID_SIZE, BOARD_HEIGHT - MARGIN), 1)
    star_points = [(3, 3), (11, 3), (3, 11), (11, 11), (7, 7)]
    for i, j in star_points:
        pygame.draw.circle(surface, BLACK, (MARGIN + i * GRID_SIZE, MARGIN + j * GRID_SIZE), 5)

def draw_stones(surface, board_state):
    for row in range(BOARD_SIZE):
        for col in range(BOARD_SIZE):
            if board_state[row][col] != 0:
                center_pos = (MARGIN + col * GRID_SIZE, MARGIN + row * GRID_SIZE)
                stone_color = BLACK if board_state[row][col] == 1 else WHITE
                pygame.draw.circle(surface, stone_color, center_pos, STONE_RADIUS)

def display_message(screen, message, font, color, center_x, center_y):
    text_surface = font.render(message, True, color)
    text_rect = text_surface.get_rect(center=(center_x, center_y))
    screen.blit(text_surface, text_rect)

# --- AI 相關函數 ---
def prepare_input(board, turn, device):
    """準備神經網絡輸入 (PyTorch 格式: B, C, H, W)"""
    board_size = board.shape[0]
    
    # 通道 1: 我方
    player_channel = (board == turn).astype(float)
    # 通道 2: 對方
    opponent_turn = 2 if turn == 1 else 1
    opponent_channel = (board == opponent_turn).astype(float)
    # 通道 3: 回合
    turn_channel = np.ones((board_size, board_size), dtype=float) if turn == 1 else np.zeros((board_size, board_size), dtype=float)
    
    # 堆疊 (H, W, 3)
    input_numpy = np.stack([player_channel, opponent_channel, turn_channel], axis=-1)
    
    # 轉換為 Tensor (1, 3, H, W)
    input_tensor = torch.from_numpy(input_numpy).float().permute(2, 0, 1).unsqueeze(0)
    return input_tensor.to(device)

def find_best_loss_iteration():
    """從訓練歷史中找到策略損失最低的迭代（最強棋力）"""
    # 獲取項目根目錄
    script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    history_path = os.path.join(script_dir, 'checkpoints', 'training_history.json')

    if not os.path.exists(history_path):
        print(f"⚠️ 找不到 training_history.json: {history_path}")
        return None

    try:
        with open(history_path, 'r', encoding='utf-8') as f:
            history = json.load(f)

        # 找到策略損失最低的迭代（代表最強棋力）
        losses = history.get('policy_loss', [])
        if not losses:
            return None

        min_loss = min(losses)
        best_iteration = losses.index(min_loss) + 1  # +1 因為迭代從 1 開始

        # 同時獲取胜率信息（如果有）
        win_rates = history.get('win_rate_vs_random', [])
        win_rate = win_rates[best_iteration - 1] if best_iteration <= len(win_rates) else None

        if win_rate:
            print(f"🏆 找到最佳棋力: 策略損失 {min_loss:.4f}, 胜率 {win_rate:.0%} (迭代 {best_iteration})")
        else:
            print(f"🏆 找到最佳棋力: 策略損失 {min_loss:.4f} (迭代 {best_iteration})")
        return best_iteration

    except Exception as e:
        print(f"讀取訓練歷史失敗: {e}")
        return None

def load_model_by_iteration(iteration, device):
    """根據迭代次數加載特定模型"""
    # 獲取項目根目錄
    script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    checkpoints_dir = os.path.join(script_dir, 'checkpoints')

    # 嘗試加載指定迭代的檢查點（不使用前導零）
    checkpoint_path = os.path.join(checkpoints_dir, f'checkpoint_iter_{iteration}.pth')

    if not os.path.exists(checkpoint_path):
        print(f"⚠️ 找不到迭代 {iteration} 的檢查點: {checkpoint_path}")
        return None

    try:
        print(f"加載迭代 {iteration} 的模型...")
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        config_dict = checkpoint.get('config', {})

        model, _ = create_enhanced_model(
            board_size=config_dict.get('BOARD_SIZE', 15),
            num_res_blocks=config_dict.get('NUM_RES_BLOCKS', 10),
            num_filters=config_dict.get('NUM_FILTERS', 128),
            device=device
        )
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"✅ 成功加載模型 (迭代 {iteration})")
        return model

    except Exception as e:
        print(f"❌ 加載迭代 {iteration} 模型失敗: {e}")
        return None

def load_latest_model(device):
    """加載最新的模型"""
    # 獲取項目根目錄
    script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    checkpoints_dir = os.path.join(script_dir, 'checkpoints')

    # 找到所有檢查點文件
    checkpoint_files = glob.glob(os.path.join(checkpoints_dir, 'checkpoint_iter_*.pth'))

    if not checkpoint_files:
        print("⚠️ 找不到任何檢查點文件")
        return None

    # 按文件名排序，取最新的
    latest_checkpoint = max(checkpoint_files, key=lambda x: int(x.split('_iter_')[1].split('.')[0]))
    iteration = int(latest_checkpoint.split('_iter_')[1].split('.')[0])

    print(f"找到最新檢查點: 迭代 {iteration}")
    return load_model_by_iteration(iteration, device)

# --- 選擇對手函數 ---
def show_menu(screen, font):
    """顯示選擇對手的菜單"""
    menu_running = True
    selected_option = None

    options = [
        {"text": "1. 人類 vs AI", "value": "human_vs_ai"},
        {"text": "2. 觀看 AI vs 隨機", "value": "ai_vs_random"},
        {"text": "3. 觀看 AI vs AI", "value": "ai_vs_ai"}
    ]

    while menu_running:
        screen.fill(BOARD_COLOR)

        # 標題
        title = font.render("選擇遊戲模式", True, BLACK)
        title_rect = title.get_rect(center=(WINDOW_WIDTH // 2, 80))
        screen.blit(title, title_rect)

        # 選項
        for i, option in enumerate(options):
            option_text = font.render(option["text"], True, BLACK)
            option_rect = option_text.get_rect(center=(WINDOW_WIDTH // 2, 180 + i * 80))
            screen.blit(option_text, option_rect)

        # 提示
        hint_font = pygame.font.Font("C:/Windows/Fonts/msyh.ttc", 20) if os.path.exists("C:/Windows/Fonts/msyh.ttc") else pygame.font.Font(None, 20)
        hint = hint_font.render("請按 1, 2 或 3 選擇", True, (100, 100, 100))
        hint_rect = hint.get_rect(center=(WINDOW_WIDTH // 2, 400))
        screen.blit(hint, hint_rect)

        pygame.display.flip()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return None
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_1:
                    return "human_vs_ai"
                elif event.key == pygame.K_2:
                    return "ai_vs_random"
                elif event.key == pygame.K_3:
                    return "ai_vs_ai"
                elif event.key == pygame.K_ESCAPE:
                    return None

    return selected_option

# --- 主函數 ---
def watch_ai_play(move_delay=500):
    pygame.init()
    pygame.font.init()

    screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
    board_surface = pygame.Surface((BOARD_WIDTH, BOARD_HEIGHT))
    pygame.display.set_caption("五子棋對弈 (PyTorch)")

    try:
        font = pygame.font.Font("C:/Windows/Fonts/msyh.ttc", 36)
        small_font = pygame.font.Font("C:/Windows/Fonts/msyh.ttc", 24)
    except FileNotFoundError:
        font = pygame.font.Font(None, 36)
        small_font = pygame.font.Font(None, 24)

    # 顯示菜單選擇對手
    game_mode = show_menu(screen, font)
    if game_mode is None:
        pygame.quit()
        return

    # 設置設備
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用設備: {device}")

    # 根據選擇加載模型
    ai_model = None
    black_model = None
    white_model = None
    is_human_player = (game_mode == "human_vs_ai")

    if game_mode == "ai_vs_ai":
        # AI vs AI 模式
        print("\n" + "="*60)
        print("⚫ 黑棋：加載 loss 最低的模型")
        print("="*60)
        best_iteration = find_best_loss_iteration()
        if best_iteration:
            black_model = load_model_by_iteration(best_iteration, device)
        else:
            print("⚠️ 無法找到最佳 loss 模型，使用最新模型")
            black_model = load_latest_model(device)

        if black_model is None:
            print("❌ 無法加載黑棋模型，程序退出")
            return
        black_model.eval()

        print("\n" + "="*60)
        print("⚪ 白棋：加載最新模型")
        print("="*60)
        white_model = load_latest_model(device)
        if white_model is None:
            print("❌ 無法加載白棋模型，程序退出")
            return
        white_model.eval()

        print("\n" + "="*60)
        print("✅ AI vs AI 對戰設置完成！")
        print("⚫ 黑棋：Best Loss 模型")
        print("⚪ 白棋：Latest 模型")
        print("="*60 + "\n")

    elif game_mode == "ai_vs_random":
        # AI vs 隨機 模式
        print("\n" + "="*60)
        print("⚫ 黑棋：加載 loss 最低的模型")
        print("="*60)
        best_iteration = find_best_loss_iteration()
        if best_iteration:
            black_model = load_model_by_iteration(best_iteration, device)
        else:
            print("⚠️ 無法找到最佳 loss 模型，使用最新模型")
            black_model = load_latest_model(device)

        if black_model is None:
            print("❌ 無法加載 AI 模型，程序退出")
            return
        black_model.eval()

        print("\n" + "="*60)
        print("✅ AI vs 隨機對戰設置完成！")
        print("⚫ 黑棋：Best Loss 模型")
        print("⚪ 白棋：隨機玩家")
        print("="*60 + "\n")

    else:  # human_vs_ai
        # 人類 vs AI 模式
        print("\n" + "="*60)
        print("⚫ 你（黑棋）vs ⚪ AI（最新模型）")
        print("="*60)
        ai_model = load_latest_model(device)
        if ai_model is None:
            print("❌ 無法加載 AI 模型，程序退出")
            return
        ai_model.eval()
        print("✅ AI 模型加載成功！")
        print("="*60 + "\n")

    game = GomokuGame(board_size=BOARD_SIZE)
    running = True
    auto_play = (game_mode in ["ai_vs_ai", "ai_vs_random"])  # 觀看模式自動播放

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r:  # 按 'r' 重新開始
                    game.reset()
                elif event.key == pygame.K_SPACE and game_mode in ["ai_vs_ai", "ai_vs_random"]:  # 空格鍵僅在觀看模式有效
                    auto_play = not auto_play

            # 人類玩家點擊棋盤（僅在人機對戰模式）
            if event.type == pygame.MOUSEBUTTONDOWN and is_human_player:
                if not game.game_over and game.turn == 1:  # 黑棋是人類
                    mouse_x, mouse_y = event.pos
                    # 轉換為棋盤座標
                    board_x = (mouse_x - MARGIN + GRID_SIZE // 2) // GRID_SIZE
                    board_y = (mouse_y - STATUS_BAR_HEIGHT - MARGIN + GRID_SIZE // 2) // GRID_SIZE

                    if 0 <= board_x < BOARD_SIZE and 0 <= board_y < BOARD_SIZE:
                        if game.get_board_state()[board_y][board_x] == 0:
                            game.make_move(board_y, board_x)

        # 繪圖
        screen.fill(BOARD_COLOR)
        pygame.draw.rect(screen, STATUS_BAR_BG, (0, 0, WINDOW_WIDTH, STATUS_BAR_HEIGHT))

        board_state = game.get_board_state()
        draw_grid_and_stars(board_surface)
        draw_stones(board_surface, board_state)
        screen.blit(board_surface, (0, BOARD_START_Y))

        # AI 行動
        if not game.game_over:
            should_ai_move = False

            if game_mode == "ai_vs_ai":
                # AI vs AI 模式 - 雙方都是 AI
                should_ai_move = auto_play
            elif game_mode == "ai_vs_random":
                # AI vs 隨機模式 - 雙方都自動
                should_ai_move = auto_play
            elif is_human_player and game.turn == 2:
                # 人機對戰模式，輪到 AI（白棋）
                should_ai_move = True

            if should_ai_move:
                legal_moves = game.get_legal_moves()
                if not legal_moves:
                    game.game_over = True
                else:
                    # 根據模式和回合選擇移動
                    if game_mode == "ai_vs_ai":
                        # AI vs AI：雙方都用 AI
                        current_model = black_model if game.turn == 1 else white_model
                        input_tensor = prepare_input(board_state, game.turn, device)
                        with torch.no_grad():
                            policy, value = current_model(input_tensor)
                            policy = policy.cpu().numpy()[0]

                        legal_indices = [r * BOARD_SIZE + c for r, c in legal_moves]
                        masked_policy = np.zeros_like(policy)
                        masked_policy[legal_indices] = policy[legal_indices]

                        if masked_policy.sum() > 0:
                            best_move_idx = np.argmax(masked_policy)
                        else:
                            best_move_idx = random.choice(legal_indices)

                        row, col = divmod(best_move_idx, BOARD_SIZE)

                    elif game_mode == "ai_vs_random":
                        # AI vs 隨機
                        if game.turn == 1:
                            # 黑棋：AI
                            input_tensor = prepare_input(board_state, game.turn, device)
                            with torch.no_grad():
                                policy, value = black_model(input_tensor)
                                policy = policy.cpu().numpy()[0]

                            legal_indices = [r * BOARD_SIZE + c for r, c in legal_moves]
                            masked_policy = np.zeros_like(policy)
                            masked_policy[legal_indices] = policy[legal_indices]

                            if masked_policy.sum() > 0:
                                best_move_idx = np.argmax(masked_policy)
                            else:
                                best_move_idx = random.choice(legal_indices)

                            row, col = divmod(best_move_idx, BOARD_SIZE)
                        else:
                            # 白棋：隨機
                            row, col = random.choice(legal_moves)

                    else:  # human_vs_ai
                        # 人類 vs AI：白棋用 AI
                        input_tensor = prepare_input(board_state, game.turn, device)
                        with torch.no_grad():
                            policy, value = ai_model(input_tensor)
                            policy = policy.cpu().numpy()[0]

                        legal_indices = [r * BOARD_SIZE + c for r, c in legal_moves]
                        masked_policy = np.zeros_like(policy)
                        masked_policy[legal_indices] = policy[legal_indices]

                        if masked_policy.sum() > 0:
                            best_move_idx = np.argmax(masked_policy)
                        else:
                            best_move_idx = random.choice(legal_indices)

                        row, col = divmod(best_move_idx, BOARD_SIZE)

                    # 執行移動
                    game.make_move(row, col)

                    # 延遲（觀看模式）
                    if game_mode in ["ai_vs_ai", "ai_vs_random"]:
                        pygame.time.wait(move_delay)

        # 狀態顯示更新
        if game.game_over:
            if game.winner == 0:
                message = "平局! (按 R 重開)"
            else:
                if game_mode == "ai_vs_ai":
                    if game.winner == 1:
                        message = "⚫ 黑棋 (Best Loss) 獲勝! (按 R 重開)"
                    else:
                        message = "⚪ 白棋 (Latest) 獲勝! (按 R 重開)"
                elif game_mode == "ai_vs_random":
                    if game.winner == 1:
                        message = "⚫ AI 獲勝! (按 R 重開)"
                    else:
                        message = "⚪ 隨機玩家獲勝! (按 R 重開)"
                else:  # human_vs_ai
                    if game.winner == 1:
                        message = "⚫ 你獲勝了! (按 R 重開)"
                    else:
                        message = "⚪ AI 獲勝! (按 R 重開)"
            display_message(screen, message, small_font, TEXT_COLOR, WINDOW_WIDTH // 2, STATUS_BAR_HEIGHT // 2)
        else:
            if game_mode in ["ai_vs_ai", "ai_vs_random"]:
                # 觀看模式
                if not auto_play:
                    display_message(screen, "按空白鍵開始/暫停", small_font, TEXT_COLOR, WINDOW_WIDTH // 2, STATUS_BAR_HEIGHT // 2)
                else:
                    if game_mode == "ai_vs_ai":
                        current_player = "⚫ 黑棋 (Best Loss)" if game.turn == 1 else "⚪ 白棋 (Latest)"
                    else:  # ai_vs_random
                        current_player = "⚫ AI" if game.turn == 1 else "⚪ 隨機玩家"
                    display_message(screen, f"{current_player} 思考中...", small_font, TEXT_COLOR, WINDOW_WIDTH // 2, STATUS_BAR_HEIGHT // 2)
            else:  # human_vs_ai
                if game.turn == 1:
                    display_message(screen, "⚫ 你的回合 - 點擊棋盤下棋", small_font, TEXT_COLOR, WINDOW_WIDTH // 2, STATUS_BAR_HEIGHT // 2)
                else:
                    display_message(screen, "⚪ AI 思考中...", small_font, TEXT_COLOR, WINDOW_WIDTH // 2, STATUS_BAR_HEIGHT // 2)

        pygame.display.flip()

        # 限制幀率
        pygame.time.Clock().tick(30)

    pygame.quit()
    sys.exit()

if __name__ == '__main__':
    watch_ai_play()
