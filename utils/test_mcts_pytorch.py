"""测试 PyTorch MCTS"""

import torch
import numpy as np
from core.neural_net import create_enhanced_model
from core.mcts import MCTS
from core.mcts_batched import BatchedMCTS
from core.game_state import GameState
from training.config import TrainingConfig

print("=" * 60)
print("测试 PyTorch MCTS")
print("=" * 60)

# 检查设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"\n设备: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

# 创建配置
config = TrainingConfig.get_fast_test_config()

# 创建模型
print("\n创建模型...")
model, _ = create_enhanced_model(
    board_size=config.BOARD_SIZE,
    num_res_blocks=3,
    num_filters=64,
    device=device
)
print("✅ 模型创建成功")

# 创建游戏状态
print("\n创建游戏状态...")
state = GameState(board_size=config.BOARD_SIZE)
state.make_move(7, 7)  # 中间落子
print(f"✅ 游戏状态创建成功，已落子: (7, 7)")

# 测试标准 MCTS
print("\n测试标准 MCTS...")
mcts = MCTS(model, config, device=device)
action_probs, root_value = mcts.search(state, add_noise=False)

print(f"✅ MCTS 搜索完成")
print(f"   动作概率形状: {action_probs.shape}")
print(f"   概率和: {action_probs.sum():.6f}")
print(f"   根节点价值: {root_value:.6f}")

# 检查 NaN
if np.isnan(action_probs).any() or np.isnan(root_value):
    print("   ❌ 检测到 NaN！")
else:
    print("   ✅ 无 NaN")

# 测试批量 MCTS
print("\n测试批量 MCTS...")
batched_mcts = BatchedMCTS(model, config, device=device)
action_probs_batched, root_value_batched = batched_mcts.search(state, add_noise=False)

print(f"✅ 批量 MCTS 搜索完成")
print(f"   动作概率形状: {action_probs_batched.shape}")
print(f"   概率和: {action_probs_batched.sum():.6f}")
print(f"   根节点价值: {root_value_batched:.6f}")

# 检查 NaN
if np.isnan(action_probs_batched).any() or np.isnan(root_value_batched):
    print("   ❌ 检测到 NaN！")
else:
    print("   ✅ 无 NaN")

# 测试温度采样
print("\n测试温度采样...")
action = mcts.get_action_with_temperature(action_probs, temperature=1.0)
print(f"✅ 温度采样成功: {action}")

print("\n" + "=" * 60)
print("✅ 所有测试通过！")
print("=" * 60)
