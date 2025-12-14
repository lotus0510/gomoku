import pygame
import sys
import os
import random
import numpy as np
from game import GomokuGame
from core.neural_net import create_enhanced_model

# --- 從 main.py 和 train.py 引入的常數和函數 ---

# 顏色
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
BOARD_COLOR = (218, 185, 123)
TEXT_COLOR = BLACK
STATUS_BAR_BG = (150, 150, 150)

# 尺寸和佈局
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
def prepare_input(board, turn):
    board_size = board.shape[0]
    player_channel = np.where(board == turn, 1, 0)
    opponent_turn = 2 if turn == 1 else 1
    opponent_channel = np.where(board == opponent_turn, 1, 0)
    turn_channel = np.ones((board_size, board_size), dtype=float) if turn == 1 else np.zeros((board_size, board_size), dtype=float)
    input_tensor = np.stack([player_channel, opponent_channel, turn_channel], axis=-1)
    return np.expand_dims(input_tensor, axis=0)

# --- 主函數 ---
def watch_ai_play(move_delay=500):
    pygame.init()
    pygame.font.init()

    screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
    board_surface = pygame.Surface((BOARD_WIDTH, BOARD_HEIGHT))
    pygame.display.set_caption("觀看 AI 對弈")

    try:
        font = pygame.font.Font("C:/Windows/Fonts/msyh.ttc", 36)
    except FileNotFoundError:
        font = pygame.font.Font(None, 36)

    # 載入 AI 模型
    print("載入 AI 模型...")

    # 嘗試載入權重（優先使用新系統的權重）
    weight_paths = [
        'checkpoints/latest_model.weights.h5',  # 新系統
        'gomoku_model.weights.h5',              # 舊系統（備用）
        'old_backup/gomoku_model.weights.h5'    # 備份
    ]

    # 不同的模型配置（按常見程度排序）
    model_configs = [
        {'num_res_blocks': 3, 'num_filters': 64, 'name': 'fast-test'},     # fast-test 配置
        {'num_res_blocks': 10, 'num_filters': 128, 'name': 'full'},        # 完整配置
        {'num_res_blocks': 10, 'num_filters': 64, 'name': 'custom'},       # 其他可能配置
    ]

    loaded = False
    model = None

    for weight_path in weight_paths:
        if not os.path.exists(weight_path):
            continue

        print(f"找到已儲存的權重：{weight_path}")

        # 嘗試不同的模型配置
        for config in model_configs:
            print(f"  嘗試載入 ({config['name']} 配置: {config['num_res_blocks']} blocks, {config['num_filters']} filters)...")
            try:
                model = create_enhanced_model(
                    board_size=BOARD_SIZE,
                    num_res_blocks=config['num_res_blocks'],
                    num_filters=config['num_filters']
                )
                model.load_weights(weight_path)
                loaded = True
                print(f"  ✓ 載入成功！使用 {config['name']} 配置")
                break
            except Exception as e:
                # 靜默失敗，嘗試下一個配置
                continue

        if loaded:
            break

    if not loaded:
        print("未找到已儲存的權重，AI 將使用隨機初始權重進行遊戲。")
        print("提示：先運行訓練以生成模型權重：")
        # 使用默認配置
        model = create_enhanced_model(
            board_size=BOARD_SIZE,
            num_res_blocks=3,
            num_filters=64
        )

    game = GomokuGame(board_size=BOARD_SIZE)
    running = True
    
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r: # 按 'r' 重新開始
                    game.reset()
        
        # 繪圖
        screen.fill(BOARD_COLOR)
        pygame.draw.rect(screen, STATUS_BAR_BG, (0, 0, WINDOW_WIDTH, STATUS_BAR_HEIGHT))
        
        board_state = game.get_board_state()
        draw_grid_and_stars(board_surface)
        draw_stones(board_surface, board_state)
        screen.blit(board_surface, (0, BOARD_START_Y))

        # 如果遊戲還沒結束，讓 AI 下一步棋
        if not game.game_over:
            # 準備輸入並預測
            input_tensor = prepare_input(board_state, game.turn)
            policy, value = model.predict(input_tensor, verbose=0)
            
            # 選擇最佳合法著法
            legal_moves = game.get_legal_moves()
            if legal_moves:
                legal_moves_indices = [r * BOARD_SIZE + c for r, c in legal_moves]
                masked_policy = np.zeros_like(policy[0])
                masked_policy[legal_moves_indices] = policy[0][legal_moves_indices]
                
                if np.sum(masked_policy) > 0:
                    best_move_index = np.argmax(masked_policy)
                    row, col = best_move_index // BOARD_SIZE, best_move_index % BOARD_SIZE
                    game.make_move(row, col)
                else: # 如果模型沒有給出任何有效建議，則隨機走一步
                    row, col = random.choice(legal_moves)
                    game.make_move(row, col)

                # 更新畫面以顯示新棋子
                draw_stones(board_surface, game.get_board_state())
                screen.blit(board_surface, (0, BOARD_START_Y))
                
                # 顯示狀態訊息
                current_player_name = '黑棋' if game.turn == 1 else '白棋'
                message = f"AI 評估價值: {value[0][0]:.2f} | 下一步: {current_player_name}"
                display_message(screen, message, font, TEXT_COLOR, WINDOW_WIDTH // 2, STATUS_BAR_HEIGHT // 2)
                
                pygame.display.flip()
                pygame.time.wait(move_delay) # 等待一段時間以便觀察
            else:
                game.game_over = True # 沒有合法著法，遊戲結束

        # 顯示最終結果
        if game.game_over:
            if game.winner == 0:
                message = "平局! (按 R 重新開始)"
            else:
                winner_name = '黑棋' if game.winner == 1 else '白棋'
                message = f"{winner_name} 獲勝! (按 R 重新開始)"
            display_message(screen, message, font, TEXT_COLOR, WINDOW_WIDTH // 2, STATUS_BAR_HEIGHT // 2)
        
        pygame.display.flip()

    pygame.quit()
    sys.exit()

if __name__ == '__main__':
    watch_ai_play()
