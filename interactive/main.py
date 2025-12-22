import pygame
import sys
from game import GomokuGame

# --- 常數 ---
# 顏色
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
BOARD_COLOR = (218, 185, 123)
TEXT_COLOR = BLACK
BUTTON_COLOR = (100, 100, 100)
BUTTON_TEXT_COLOR = WHITE
STATUS_BAR_BG = (150, 150, 150)
MENU_BG_COLOR = (200, 170, 110)

# 尺寸和佈局 (與 game.py 中的 board_size 同步)
BOARD_SIZE = 15
GRID_SIZE = 40
MARGIN = 20
STONE_RADIUS = GRID_SIZE // 2 - 2
STATUS_BAR_HEIGHT = 50
BUTTON_WIDTH = 120
BUTTON_HEIGHT = 40

# 棋盤自身的寬度和高度
BOARD_WIDTH = (BOARD_SIZE - 1) * GRID_SIZE + 2 * MARGIN
BOARD_HEIGHT = (BOARD_SIZE - 1) * GRID_SIZE + 2 * MARGIN

# 整個視窗的寬度和高度
WINDOW_WIDTH = BOARD_WIDTH
WINDOW_HEIGHT = BOARD_HEIGHT + STATUS_BAR_HEIGHT + BUTTON_HEIGHT + MARGIN

# 按鈕位置
NEW_GAME_BUTTON_RECT = pygame.Rect(
    (WINDOW_WIDTH - BUTTON_WIDTH) // 2,
    WINDOW_HEIGHT - BUTTON_HEIGHT - MARGIN // 2,
    BUTTON_WIDTH,
    BUTTON_HEIGHT
)
START_GAME_BUTTON_RECT = pygame.Rect(
    (WINDOW_WIDTH - 200) // 2,
    WINDOW_HEIGHT // 2,
    200,
    60
)

# 棋盤繪製的起始Y座標
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

def draw_button(screen, rect, text, font, button_color, text_color):
    pygame.draw.rect(screen, button_color, rect, border_radius=10)
    display_message(screen, text, font, text_color, rect.centerx, rect.centery)

def main():
    pygame.init()
    pygame.font.init()

    screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
    board_surface = pygame.Surface((BOARD_WIDTH, BOARD_HEIGHT))
    pygame.display.set_caption("五子棋 (Gomoku)")

    try:
        title_font = pygame.font.Font("C:/Windows/Fonts/msyh.ttc", 72)
        font = pygame.font.Font("C:/Windows/Fonts/msyh.ttc", 36)
        button_font = pygame.font.Font("C:/Windows/Fonts/msyh.ttc", 24)
        start_button_font = pygame.font.Font("C:/Windows/Fonts/msyh.ttc", 40)
    except FileNotFoundError:
        print("找不到字體 'msyh.ttc'，將使用預設字體。中文可能無法顯示。")
        title_font = pygame.font.Font(None, 80)
        font = pygame.font.Font(None, 36)
        button_font = pygame.font.Font(None, 24)
        start_button_font = pygame.font.Font(None, 48)

    # 初始化遊戲邏輯和UI狀態
    game = GomokuGame(board_size=BOARD_SIZE)
    game_state = 'main_menu'

    running = True
    while running:
        # --- 事件處理 ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            
            if game_state == 'main_menu':
                if event.type == pygame.MOUSEBUTTONDOWN:
                    if START_GAME_BUTTON_RECT.collidepoint(event.pos):
                        game.reset()
                        game_state = 'in_game'
            
            elif game_state == 'in_game':
                if event.type == pygame.MOUSEBUTTONDOWN:
                    x, y = event.pos
                    if NEW_GAME_BUTTON_RECT.collidepoint(x, y):
                        game.reset()
                    
                    elif y > BOARD_START_Y and y < BOARD_START_Y + BOARD_HEIGHT:
                        board_x, board_y = x, y - BOARD_START_Y
                        col = round((board_x - MARGIN) / GRID_SIZE)
                        row = round((board_y - MARGIN) / GRID_SIZE)
                        game.make_move(row, col)

        # --- 繪圖 ---
        if game_state == 'main_menu':
            screen.fill(MENU_BG_COLOR)
            display_message(screen, "五子棋", title_font, BLACK, WINDOW_WIDTH // 2, WINDOW_HEIGHT // 3)
            draw_button(screen, START_GAME_BUTTON_RECT, "開始遊戲", start_button_font, BUTTON_COLOR, BUTTON_TEXT_COLOR)
        
        elif game_state == 'in_game':
            board_state = game.get_board_state()
            draw_grid_and_stars(board_surface)
            draw_stones(board_surface, board_state)
            
            screen.fill(BOARD_COLOR)
            pygame.draw.rect(screen, STATUS_BAR_BG, (0, 0, WINDOW_WIDTH, STATUS_BAR_HEIGHT))
            draw_button(screen, NEW_GAME_BUTTON_RECT, "新遊戲", button_font, BUTTON_COLOR, BUTTON_TEXT_COLOR)
            
            screen.blit(board_surface, (0, BOARD_START_Y))
            
            if game.game_over:
                if game.winner == 0:
                    message = "平局!"
                else:
                    winner_name = '黑棋' if game.winner == 1 else '白棋'
                    message = f"{winner_name} 獲勝!"
                display_message(screen, message, font, TEXT_COLOR, WINDOW_WIDTH // 2, STATUS_BAR_HEIGHT // 2)
            else:
                current_player_name = '黑棋' if game.turn == 1 else '白棋'
                message = f"目前回合: {current_player_name}"
                display_message(screen, message, font, TEXT_COLOR, WINDOW_WIDTH // 2, STATUS_BAR_HEIGHT // 2)

        pygame.display.flip()

    pygame.quit()
    sys.exit()

if __name__ == '__main__':
    main()