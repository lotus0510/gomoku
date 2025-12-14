import matplotlib.pyplot as plt
import json
import os
import numpy as np # 需要用於尋找最低損失

def plot_training_history():
    """
    讀取 training_history.json 檔案並繪製損失圖表，包含訓練參數。
    """
    history_file = 'training_history.json'
    
    if not os.path.exists(history_file):
        print(f"錯誤：找不到歷史數據檔案 '{history_file}'。")
        print("請先執行 'train.py' 來生成數據。")
        return

    print(f"從 '{history_file}' 讀取數據...")
    with open(history_file, 'r') as f:
        history = json.load(f)
        
    total_loss = history.get('total_loss', [])
    policy_loss = history.get('policy_loss', [])
    value_loss = history.get('value_loss', [])
    eval_win = history.get('eval_win_rate', [])
    eval_draw = history.get('eval_draw_rate', [])
    eval_loss = history.get('eval_loss_rate', [])
    params = history.get('params', {}) # 讀取儲存的訓練參數
    runs = params.get('runs', []) if isinstance(params, dict) else []
    
    if not total_loss:
        print("錯誤：歷史數據為空，無法繪圖。請確認訓練已至少完成一個 epoch。")
        return

    num_iterations_logged = len(total_loss)
    iterations = np.arange(1, num_iterations_logged + 1)

    fig, (ax_loss, ax_eval) = plt.subplots(2, 1, figsize=(15, 12), gridspec_kw={'height_ratios': [2, 1]})

    # 繪製損失曲線
    ax_loss.plot(iterations, total_loss, label='Total Loss', linestyle='--', color='red', marker='x')
    ax_loss.plot(iterations, policy_loss, label='Policy Loss', marker='o')
    ax_loss.plot(iterations, value_loss, label='Value Loss', marker='o')
    
    # 計算統計資訊
    total_games_played = sum(
        (run.get('iterations', 0) or 0) * (run.get('num_games_per_iteration', 0) or 0)
        for run in runs
    )
    if total_games_played == 0:
        total_games_played = (params.get('iterations', 0) or 0) * (params.get('num_games_per_iteration', 0) or 0)

    latest_run = runs[-1] if runs else (params if isinstance(params, dict) else {})
    if not isinstance(latest_run, dict):
        latest_run = {}
    training_target = latest_run.get('training_target') or params.get('training_target') if isinstance(params, dict) else None
    training_target = training_target or "Gomoku 自我對弈 (Policy/Value Network)"
    avg_total_loss = float(np.mean(total_loss)) if total_loss else 0

    # 圖表標題與軸標籤
    title_str = (
        f"{training_target}\n"
        f"{num_iterations_logged} Iterations Logged, ~{total_games_played} Games"
    )
    ax_loss.set_title(title_str)
    ax_loss.set_xlabel('Training Iteration (1-based)')
    ax_loss.set_ylabel('Loss')
    ax_loss.set_xticks(iterations)
    ax_loss.legend()
    ax_loss.grid(True)

    # --- 訓練參數資訊框 ---
    last_total = total_loss[-1]
    last_policy = policy_loss[-1] if policy_loss else 0
    last_value = value_loss[-1] if value_loss else 0
    info_text = (
        f"Training: {training_target}\n"
        f"Logged Iters: {num_iterations_logged}\n"
        f"Total Games: {total_games_played}\n"
        f"Board Size: {latest_run.get('board_size', params.get('board_size', 'N/A'))}\n"
        f"Batch Size: {latest_run.get('batch_size', params.get('batch_size', 'N/A'))}\n"
        f"Start Time: {latest_run.get('start_time', params.get('start_time', 'N/A'))}\n"
        f"Latest Loss: {last_total:.4f} (P {last_policy:.4f}, V {last_value:.4f})\n"
        f"Avg Total Loss: {avg_total_loss:.4f}"
    )
    ax_loss.text(1.02, 0.95, info_text, transform=ax_loss.transAxes,
                 fontsize=10, verticalalignment='top', bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5))

    # --- 新增：標註最低 Total Loss ---
    if total_loss:
        min_loss_value = min(total_loss)
        min_loss_iter = np.argmin(total_loss)
        ax_loss.annotate(f'Min Total Loss: {min_loss_value:.4f} at Iter {min_loss_iter + 1}',
                         xy=(min_loss_iter + 1, min_loss_value), xycoords='data',
                         xytext=(min_loss_iter + 1.3, min_loss_value + 0.1), textcoords='data',
                         arrowprops=dict(facecolor='black', shrink=0.05),
                         horizontalalignment='left', verticalalignment='bottom')

    # --- 最近幾次訓練摘要（若有多次跑） ---
    if runs:
        recent_runs = runs[-3:]  # 只展示最近 3 次
        run_lines = ["Recent Runs:"]
        for idx, run in enumerate(reversed(recent_runs), start=1):
            run_lines.append(
                f"{len(runs) - idx + 1}: {run.get('iterations', '?')} iters x {run.get('num_games_per_iteration', '?')} games "
                f"(batch {run.get('batch_size', '?')}, board {run.get('board_size', '?')})"
            )
        run_lines.append(f"Target: {training_target}")
        run_text = "\n".join(run_lines)
        ax_loss.text(1.02, 0.45, run_text, transform=ax_loss.transAxes,
                     fontsize=9, verticalalignment='top', bbox=dict(boxstyle='round,pad=0.5', fc='lightblue', alpha=0.5))

    # --- 評估曲線（若有資料） ---
    eval_has_data = any(v is not None for v in eval_win) or any(v is not None for v in eval_draw) or any(v is not None for v in eval_loss)
    if eval_has_data:
        # 將 None 轉為 nan 方便繪製
        eval_win_np = np.array([np.nan if v is None else v for v in eval_win], dtype=float)
        eval_draw_np = np.array([np.nan if v is None else v for v in eval_draw], dtype=float)
        eval_loss_np = np.array([np.nan if v is None else v for v in eval_loss], dtype=float)

        ax_eval.plot(iterations, eval_win_np, label='Win Rate (vs fixed)', color='green', marker='o')
        ax_eval.plot(iterations, eval_draw_np, label='Draw Rate', color='gray', marker='s')
        ax_eval.plot(iterations, eval_loss_np, label='Loss Rate', color='red', marker='x')
        ax_eval.set_ylim(0, 1)
        ax_eval.set_ylabel('Rate')
        ax_eval.set_xlabel('Training Iteration (1-based)')
        ax_eval.grid(True)
        ax_eval.legend()
        ax_eval.set_title('固定對手評估 (Win/Draw/Loss)')
    else:
        ax_eval.set_visible(False)

    plt.tight_layout(rect=[0, 0, 0.75, 1]) # 調整佈局以避免資訊框重疊
    
    print("正在顯示圖表...")
    plt.show()

if __name__ == '__main__':
    plot_training_history()
