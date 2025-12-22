import pygame
import sys
import os
import random
import glob
import numpy as np
import torch

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

def load_best_model(device):
    """智能加載最佳模型（優先加載帶配置的 Checkpoint，其次是 latest_model）"""
    checkpoints_dir = 'checkpoints'
    
    # 1. 嘗試加載最新的完整 checkpoint (包含配置)
    checkpoint_files = glob.glob(os.path.join(checkpoints_dir, 'checkpoint_iter_*.pth'))
    if checkpoint_files:
        # 按迭代次數排序
        latest_checkpoint = max(checkpoint_files, key=os.path.getctime)
        print(f"發現完整檢查點: {latest_checkpoint}")
        
        try:
            checkpoint = torch.load(latest_checkpoint, map_location=device, weights_only=True)
            config_dict = checkpoint.get('config', {})
            
            # 從 checkpoint 恢復配置
            print("從檢查點讀取配置...")
            model, _ = create_enhanced_model(
                board_size=config_dict.get('BOARD_SIZE', 15),
                num_res_blocks=config_dict.get('NUM_RES_BLOCKS', 10),
                num_filters=config_dict.get('NUM_FILTERS', 128),
                device=device
            )
            model.load_state_dict(checkpoint['model_state_dict'])
            print(f"成功加載模型 (迭代 {checkpoint.get('iteration', '?')})")
            return model
        except Exception as e:
            print(f"加載檢查點失敗: {e}")
            print("嘗試加載 latest_model.pth ...")
    
    # 2. 嘗試加載 latest_model.pth (僅權重，需猜測配置)
    latest_weights = os.path.join(checkpoints_dir, 'latest_model.pth')
    if os.path.exists(latest_weights):
        print(f"發現權重文件: {latest_weights}")
        
        # 定義可能的配置列表
        param_guesses = [
            {'blocks': 3, 'filters': 64, 'name': 'Fast Test'},
            {'blocks': 10, 'filters': 128, 'name': 'Full Training'},
            {'blocks': 5, 'filters': 64, 'name': 'Medium'},
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
            except Exception as e:
                pass # 繼續嘗試下一個
                
    print("❌未能加載任何模型，將使用隨機初始化模型。")
    # 默認創建一個
    model, _ = create_enhanced_model(board_size=15, num_res_blocks=3, num_filters=64, device=device)
    return model

# --- 主函數 ---
def watch_ai_play(move_delay=500):
    pygame.init()
    pygame.font.init()

    screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
    board_surface = pygame.Surface((BOARD_WIDTH, BOARD_HEIGHT))
    pygame.display.set_caption("觀看 AI 對弈 (PyTorch)")

    try:
        font = pygame.font.Font("C:/Windows/Fonts/msyh.ttc", 36)
        small_font = pygame.font.Font("C:/Windows/Fonts/msyh.ttc", 24)
    except FileNotFoundError:
        font = pygame.font.Font(None, 36)
        small_font = pygame.font.Font(None, 24)

    # 設置設備
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用設備: {device}")
    
    # 加載模型
    model = load_best_model(device)
    model.eval()

    game = GomokuGame(board_size=BOARD_SIZE)
    running = True
    auto_play = False
    
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r: # 按 'r' 重新開始
                    game.reset()
                elif event.key == pygame.K_SPACE: # 按空格鍵切換自動/手動步進
                    auto_play = not auto_play

        # 繪圖
        screen.fill(BOARD_COLOR)
        pygame.draw.rect(screen, STATUS_BAR_BG, (0, 0, WINDOW_WIDTH, STATUS_BAR_HEIGHT))
        
        board_state = game.get_board_state()
        draw_grid_and_stars(board_surface)
        draw_stones(board_surface, board_state)
        screen.blit(board_surface, (0, BOARD_START_Y))

        # AI 思考與行動
        if not game.game_over:
            # PyTorch 推理
            input_tensor = prepare_input(board_state, game.turn, device)
            
            with torch.no_grad():
                policy, value = model(input_tensor)
                # 轉回 NumPy
                policy = policy.cpu().numpy()[0]
                value = value.item()
            
            # 獲取合法步驟
            legal_moves = game.get_legal_moves()
            if legal_moves:
                # Mask 非法移動
                legal_indices = [r * BOARD_SIZE + c for r, c in legal_moves]
                masked_policy = np.zeros_like(policy)
                masked_policy[legal_indices] = policy[legal_indices]
                
                # 選擇最佳移動
                if masked_policy.sum() > 0:
                    best_move_idx = np.argmax(masked_policy)
                    move_prob = masked_policy[best_move_idx]
                else:
                    best_move_idx = random.choice(legal_indices)
                    move_prob = 0.0
                
                row, col = divmod(best_move_idx, BOARD_SIZE)
                
                # 執行移動（如果是自動播放，或者這是第一步）
                if auto_play or game.last_move is None:
                    game.make_move(row, col)
                    # 延遲
                    if auto_play:
                        # 顯示更新
                        draw_stones(board_surface, game.get_board_state())
                        screen.blit(board_surface, (0, BOARD_START_Y))
                        
                        current_player_name = '黑棋' if game.turn == 1 else '白棋'
                        msg = f"評估: {value:.2f} | {current_player_name}"
                        display_message(screen, msg, font, TEXT_COLOR, WINDOW_WIDTH // 2, STATUS_BAR_HEIGHT // 2)
                        
                        pygame.display.flip()
                        pygame.time.wait(move_delay)
            else:
                game.game_over = True

        # 狀態顯示更新
        if game.game_over:
            if game.winner == 0:
                message = "平局! (按 R 重開)"
            else:
                winner_name = '黑棋' if game.winner == 1 else '白棋'
                message = f"{winner_name} 獲勝! (按 R 重開)"
            display_message(screen, message, font, TEXT_COLOR, WINDOW_WIDTH // 2, STATUS_BAR_HEIGHT // 2)
        else:
             if not auto_play:
                 display_message(screen, "按空白鍵開始/暫停", small_font, TEXT_COLOR, WINDOW_WIDTH // 2, STATUS_BAR_HEIGHT // 2)

        pygame.display.flip()
        
        # 限制幀率
        pygame.time.Clock().tick(30)

    pygame.quit()
    sys.exit()

if __name__ == '__main__':
    watch_ai_play()
