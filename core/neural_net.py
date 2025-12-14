"""增强的神经网络架构 - SE-ResNet（Squeeze-and-Excitation Residual Network）"""

import os

# 靜音 TensorFlow C++ 層的 INFO 日誌，避免多進程刷屏
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Conv2D, Dense, Flatten, BatchNormalization,
    Activation, Add, GlobalAveragePooling2D, Reshape, Multiply, Dropout
)
from tensorflow.keras.regularizers import l2


def se_block(x, filters, ratio=8, name_prefix='se'):
    """
    Squeeze-and-Excitation 注意力块

    通过学习通道间的依赖关系，自适应地重新校准通道特征

    Args:
        x: 输入张量 (batch, height, width, filters)
        filters: 通道数
        ratio: 压缩比例
        name_prefix: 层名称前缀

    Returns:
        重新加权后的张量
    """
    # Squeeze: 全局平均池化 (batch, filters)
    squeeze = GlobalAveragePooling2D(name=f'{name_prefix}_gap')(x)

    # Excitation: 两层全连接
    # 第一层: 降维
    excitation = Dense(
        filters // ratio,
        activation='relu',
        kernel_initializer='he_normal',
        name=f'{name_prefix}_fc1'
    )(squeeze)

    # 第二层: 升维并sigmoid
    excitation = Dense(
        filters,
        activation='sigmoid',
        kernel_initializer='he_normal',
        name=f'{name_prefix}_fc2'
    )(excitation)

    # Reshape为 (batch, 1, 1, filters) 以便广播
    excitation = Reshape((1, 1, filters), name=f'{name_prefix}_reshape')(excitation)

    # Scale: 将注意力权重应用到原始特征
    scaled = Multiply(name=f'{name_prefix}_scale')([x, excitation])

    return scaled


def residual_block(x, filters, l2_reg=1e-4, se_ratio=8, block_id=0):
    """
    改进的残差块（带SE注意力）

    结构: Conv → BN → ReLU → Conv → BN → SE → Add → ReLU

    Args:
        x: 输入张量
        filters: 滤波器数量
        l2_reg: L2正则化系数
        se_ratio: SE注意力压缩比
        block_id: 块编号（用于命名）

    Returns:
        残差块输出
    """
    shortcut = x
    name_prefix = f'res_block_{block_id}'

    # 第一层卷积
    x = Conv2D(
        filters,
        kernel_size=3,
        padding='same',
        kernel_regularizer=l2(l2_reg),
        kernel_initializer='he_normal',
        name=f'{name_prefix}_conv1'
    )(x)
    x = BatchNormalization(name=f'{name_prefix}_bn1')(x)
    x = Activation('relu', name=f'{name_prefix}_relu1')(x)

    # 第二层卷积
    x = Conv2D(
        filters,
        kernel_size=3,
        padding='same',
        kernel_regularizer=l2(l2_reg),
        kernel_initializer='he_normal',
        name=f'{name_prefix}_conv2'
    )(x)
    x = BatchNormalization(name=f'{name_prefix}_bn2')(x)

    # SE注意力
    x = se_block(x, filters, ratio=se_ratio, name_prefix=f'{name_prefix}_se')

    # 残差连接
    x = Add(name=f'{name_prefix}_add')([shortcut, x])
    x = Activation('relu', name=f'{name_prefix}_relu2')(x)

    return x


def create_enhanced_model(
    board_size=15,
    num_res_blocks=10,
    num_filters=128,
    se_ratio=8,
    l2_reg=1e-4,
    dropout_rate=0.3,
    value_head_hidden=512
):
    """
    创建增强的SE-ResNet模型

    改进点：
    1. 更深的网络（10个残差块 vs 3个）
    2. 更多滤波器（128 vs 64）
    3. SE注意力机制
    4. 改进的策略头（2层卷积 + Dropout）
    5. 增强的价值头（更大的隐藏层）

    Args:
        board_size: 棋盘大小
        num_res_blocks: 残差块数量
        num_filters: 卷积滤波器数量
        se_ratio: SE注意力压缩比
        l2_reg: L2正则化系数
        dropout_rate: Dropout比率
        value_head_hidden: 价值头隐藏层大小

    Returns:
        Keras Model
    """

    # ===== 输入层 =====
    # 输入形状: (board_size, board_size, 3)
    # 3个通道: [我方棋子, 对方棋子, 当前回合]
    inputs = Input(shape=(board_size, board_size, 3), name='board_input')

    # ===== 初始卷积层 =====
    x = Conv2D(
        filters=num_filters,
        kernel_size=3,
        padding='same',
        kernel_regularizer=l2(l2_reg),
        kernel_initializer='he_normal',
        name='initial_conv'
    )(inputs)
    x = BatchNormalization(name='initial_bn')(x)
    x = Activation('relu', name='initial_relu')(x)

    # ===== 残差块塔 =====
    for i in range(num_res_blocks):
        x = residual_block(
            x,
            filters=num_filters,
            l2_reg=l2_reg,
            se_ratio=se_ratio,
            block_id=i
        )

    # ===== 策略头 (Policy Head) =====
    # 改进: 使用2个卷积层而非1个
    policy = Conv2D(
        filters=4,
        kernel_size=1,
        padding='same',
        kernel_regularizer=l2(l2_reg),
        kernel_initializer='he_normal',
        name='policy_conv1'
    )(x)
    policy = BatchNormalization(name='policy_bn1')(policy)
    policy = Activation('relu', name='policy_relu1')(policy)

    policy = Conv2D(
        filters=2,
        kernel_size=1,
        padding='same',
        kernel_regularizer=l2(l2_reg),
        kernel_initializer='he_normal',
        name='policy_conv2'
    )(policy)
    policy = BatchNormalization(name='policy_bn2')(policy)
    policy = Activation('relu', name='policy_relu2')(policy)

    policy = Flatten(name='policy_flatten')(policy)

    # 添加Dropout防止过拟合
    if dropout_rate > 0:
        policy = Dropout(dropout_rate, name='policy_dropout')(policy)

    policy_output = Dense(
        board_size * board_size,
        activation='softmax',
        kernel_regularizer=l2(l2_reg),
        name='policy_output'
    )(policy)

    # ===== 价值头 (Value Head) =====
    # 改进: 更大的隐藏层
    value = Conv2D(
        filters=2,
        kernel_size=1,
        padding='same',
        kernel_regularizer=l2(l2_reg),
        kernel_initializer='he_normal',
        name='value_conv'
    )(x)
    value = BatchNormalization(name='value_bn')(value)
    value = Activation('relu', name='value_relu')(value)

    value = Flatten(name='value_flatten')(value)

    # 第一层全连接
    value = Dense(
        value_head_hidden,
        activation='relu',
        kernel_regularizer=l2(l2_reg),
        kernel_initializer='he_normal',
        name='value_fc1'
    )(value)

    # 添加Dropout
    if dropout_rate > 0:
        value = Dropout(dropout_rate, name='value_dropout')(value)

    # 第二层全连接
    value = Dense(
        256,
        activation='relu',
        kernel_regularizer=l2(l2_reg),
        kernel_initializer='he_normal',
        name='value_fc2'
    )(value)

    # 输出层（tanh激活，输出范围[-1, 1]）
    value_output = Dense(
        1,
        activation='tanh',
        kernel_initializer='he_normal',
        name='value_output'
    )(value)

    # ===== 构建模型 =====
    model = Model(inputs=inputs, outputs=[policy_output, value_output], name='SE_ResNet_Gomoku')

    return model


if __name__ == '__main__':
    import numpy as np
    from core.game_state import GameState

    print("=== 测试增强的SE-ResNet模型 ===\n")

    # 创建模型（使用快速测试配置）
    print("创建模型（测试配置）...")
    test_model = create_enhanced_model(
        board_size=15,
        num_res_blocks=3,  # 测试用较小配置
        num_filters=64,
        se_ratio=8,
        l2_reg=1e-4,
        dropout_rate=0.3
    )

    # 打印模型摘要
    print("\n模型架构摘要：")
    test_model.summary()

    # 计算参数量
    total_params = test_model.count_params()
    print(f"\n总参数量: {total_params:,}")

    # 测试前向传播
    print("\n测试前向传播...")
    state = GameState(board_size=15)
    state.make_move(7, 7)  # 中间放一个黑棋
    state.make_move(8, 8)  # 附近放一个白棋

    test_input = np.expand_dims(state.to_input(), axis=0)  # 添加批次维度
    print(f"  输入形状: {test_input.shape}")

    policy, value = test_model.predict(test_input, verbose=0)
    print(f"  策略输出形状: {policy.shape}, 和: {policy.sum():.6f}")
    print(f"  价值输出形状: {value.shape}, 值: {value[0, 0]:.6f}")
    print(f"  策略最大概率位置: {np.unravel_index(policy.argmax(), (15, 15))}")

    print("\n✓ 模型创建和测试成功！")

    # 创建完整配置模型
    print("\n创建完整配置模型...")
    full_model = create_enhanced_model(
        board_size=15,
        num_res_blocks=10,
        num_filters=128,
        se_ratio=8
    )
    full_params = full_model.count_params()
    print(f"  完整模型参数量: {full_params:,}")
    print(f"  参数量增加: {full_params / total_params:.2f}x")
