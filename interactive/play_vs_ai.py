import pygame
import sys
import os
import random
import glob
import numpy as np
import torch

from game import GomokuGame
from core.neural_net import create_enhanced_model

# --- 常數 ---
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
BOARD_COLOR = (218, 185, 123)
TEXT_COLOR = BLACK
STATUS_BAR_BG = (150, 150, 150)
HIGHLIGHT_COLOR = (255, 0, 0)  # 高亮最後一步

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

def draw_stones(surface, board_state, last_move=None):
    for row in range(BOARD_SIZE):
        for col in range(BOARD_SIZE):
            if board_state[row][col] != 0:
                center_pos = (MARGIN + col * GRID_SIZE, MARGIN + row * GRID_SIZE)
                stone_color = BLACK if board_state[row][col] == 1 else WHITE
                pygame.draw.circle(surface, stone_color, center_pos, STONE_RADIUS)
                
                # 高亮最後一步
                if last_move and last_move == (row, col):
                    pygame.draw.circle(surface, HIGHLIGHT_COLOR, center_pos, 4)

def display_message(screen, message, font, color, center_x, center_y):
    text_surface = font.render(message, True, color)
    text_rect = text_surface.get_rect(center=(center_x, center_y))
    screen.blit(text_surface, text_rect)

# --- AI 相關函數 (復用 watch_ai.py 的邏輯) ---
def prepare_input(board, turn, device):
    board_size = board.shape[0]
    player_channel = (board == turn).astype(float)
    opponent_turn = 2 if turn == 1 else 1
    opponent_channel = (board == opponent_turn).astype(float)
    turn_channel = np.ones((board_size, board_size), dtype=float) if turn == 1 else np.zeros((board_size, board_size), dtype=float)
    input_numpy = np.stack([player_channel, opponent_channel, turn_channel], axis=-1)
    input_tensor = torch.from_numpy(input_numpy).float().permute(2, 0, 1).unsqueeze(0)
    return input_tensor.to(device)

def load_best_model(device):
    checkpoints_dir = 'checkpoints'
    
    # 為了避免複雜的 weights_only 問題，我們優先使用 latest_model.pth (純權重)
    # 並配合配置猜測，這在現在看來是最穩健的
    latest_weights = os.path.join(checkpoints_dir, 'latest_model.pth')
    
    if os.path.exists(latest_weights):
        print(f"發現權重文件: {latest_weights}")
        param_guesses = [
            {'blocks': 3, 'filters': 64, 'name': 'Fast Test'}, # 用戶當前最可能的配置
            {'blocks': 10, 'filters': 128, 'name': 'Full Training'},
        ]
        
        for params in param_guesses:
            print(f"嘗試配置: {params['name']} ...")
            try:
                model, _ = create_enhanced_model(
                    board_size=15,
                    num_res_blocks=params['blocks'],
                    num_filters=params['filters'],
                    device=device
                )
                state_dict = torch.load(latest_weights, map_location=device, weights_only=True)
                model.load_state_dict(state_dict)
                print(f"成功匹配配置: {params['name']}")
                return model
            except Exception:
                continue
                
    print("❌未能加載模型，使用隨機初始化模型。")
    model, _ = create_enhanced_model(board_size=15, num_res_blocks=3, num_filters=64, device=device)
    return model

# --- 主函數 ---
def play_vs_ai():
    pygame.init()
    pygame.font.init()

    screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
    board_surface = pygame.Surface((BOARD_WIDTH, BOARD_HEIGHT))
    pygame.display.set_caption("人機對戰 (您是黑棋)")

    try:
        font = pygame.font.Font("C:/Windows/Fonts/msyh.ttc", 36)
        small_font = pygame.font.Font("C:/Windows/Fonts/msyh.ttc", 20)
    except FileNotFoundError:
        font = pygame.font.Font(None, 36)
        small_font = pygame.font.Font(None, 24)

    # 設置設備和模型
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用設備: {device}")
    model = load_best_model(device)
    model.eval()

    game = GomokuGame(board_size=BOARD_SIZE)
    
    # 用戶選擇先後手（簡單起見，目前默認人類先手黑棋）
    # 如果想讓 AI 先手，可以修改這裡
    human_turn = 1 
    ai_turn = 2
    
    running = True
    ai_thinking = False
    status_msg = "您的回合 (黑棋)"
    ai_eval = 0.0
    
    while running:
        # --- 事件處理 ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r: # 重置
                    game.reset()
                    ai_thinking = False
                    status_msg = "您的回合 (黑棋)"
            
            # 人類落子
            if not game.game_over and game.turn == human_turn and not ai_thinking:
                if event.type == pygame.MOUSEBUTTONDOWN:
                    x, y = event.pos
                    if y > BOARD_START_Y and y < BOARD_START_Y + BOARD_HEIGHT:
                        board_x, board_y = x, y - BOARD_START_Y
                        col = round((board_x - MARGIN) / GRID_SIZE)
                        row = round((board_y - MARGIN) / GRID_SIZE)
                        
                        # 嘗試落子
                        game_over, winner = game.make_move(row, col)
                        # 如果無效移動，make_move 內部會處理，這裡不用擔心
                        
                        if game.game_over:
                            if game.winner == human_turn: message = "恭喜獲勝！"
                            elif game.winner == ai_turn: message = "AI 獲勝！"
                            else: message = "平局！"
        
        # --- AI 邏輯 ---
        if not game.game_over and game.turn == ai_turn and not ai_thinking:
            ai_thinking = True
            status_msg = "AI 思考中..."
        
        if ai_thinking:
            # 為了不卡住 UI，可以考慮用線程，但簡單起見，我們就在這裡執行
            # 並強制渲染一次 "思考中"
            
            # 渲染思考中狀態
            screen.fill(BOARD_COLOR)
            pygame.draw.rect(screen, STATUS_BAR_BG, (0, 0, WINDOW_WIDTH, STATUS_BAR_HEIGHT))
            board_state = game.get_board_state()
            draw_grid_and_stars(board_surface)
            draw_stones(board_surface, board_state, game.last_move)
            screen.blit(board_surface, (0, BOARD_START_Y))
            display_message(screen, status_msg, font, TEXT_COLOR, WINDOW_WIDTH // 2, STATUS_BAR_HEIGHT // 2)
            pygame.display.flip()
            
            # 執行推理
            input_tensor = prepare_input(game.get_board_state(), game.turn, device)
            with torch.no_grad():
                policy, value = model(input_tensor)
                policy = policy.cpu().numpy()[0]
                ai_eval = value.item()
            
            legal_moves = game.get_legal_moves()
            if legal_moves:
                legal_indices = [r * BOARD_SIZE + c for r, c in legal_moves]
                masked_policy = np.zeros_like(policy)
                masked_policy[legal_indices] = policy[legal_indices]
                
                if masked_policy.sum() > 0:
                    best_move_idx = np.argmax(masked_policy)
                else:
                    best_move_idx = random.choice(legal_indices)
                
                row, col = divmod(best_move_idx, BOARD_SIZE)
                game.make_move(row, col)
            else:
                game.game_over = True
            
            ai_thinking = False
            status_msg = f"AI 評估: {ai_eval:.2f} | 您的回合"

        # --- 正常渲染循環 ---
        screen.fill(BOARD_COLOR)
        pygame.draw.rect(screen, STATUS_BAR_BG, (0, 0, WINDOW_WIDTH, STATUS_BAR_HEIGHT))
        
        board_state = game.get_board_state()
        draw_grid_and_stars(board_surface)
        draw_stones(board_surface, board_state, game.last_move)
        screen.blit(board_surface, (0, BOARD_START_Y))
        
        if game.game_over:
            if game.winner == 0:
                final_msg = "平局! (按 R 重開)"
                color = (0, 0, 255)
            elif game.winner == human_turn:
                final_msg = "你贏了! (按 R 重開)"
                color = (0, 100, 0)
            else:
                final_msg = "AI 贏了! (按 R 重開)"
                color = (255, 0, 0)
            display_message(screen, final_msg, font, color, WINDOW_WIDTH // 2, STATUS_BAR_HEIGHT // 2)
        else:
            display_message(screen, status_msg, font, TEXT_COLOR, WINDOW_WIDTH // 2, STATUS_BAR_HEIGHT // 2)

        pygame.display.flip()
        pygame.time.Clock().tick(60)

    pygame.quit()
    sys.exit()

if __name__ == '__main__':
    play_vs_ai()
