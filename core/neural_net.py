"""增强的神经网络架构 - SE-ResNet (PyTorch 版本)"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SEBlock(nn.Module):
    """
    Squeeze-and-Excitation 注意力块

    通过学习通道间的依赖关系，自适应地重新校准通道特征
    """
    def __init__(self, channels, ratio=8):
        """
        Args:
            channels: 输入通道数
            ratio: 压缩比例
        """
        super(SEBlock, self).__init__()

        # Squeeze: 全局平均池化
        self.gap = nn.AdaptiveAvgPool2d(1)

        # Excitation: 两层全连接
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // ratio, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // ratio, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        """
        Args:
            x: (batch, channels, height, width)
        Returns:
            重新加权后的张量
        """
        b, c, _, _ = x.size()

        # Squeeze: (batch, channels, 1, 1)
        squeeze = self.gap(x)

        # Excitation: (batch, channels)
        excitation = squeeze.reshape(b, c)
        excitation = self.fc(excitation)

        # Scale: (batch, channels, 1, 1)
        excitation = excitation.reshape(b, c, 1, 1)

        return x * excitation


class ResidualBlock(nn.Module):
    """
    改进的残差块（带 SE 注意力）

    结构: Conv → BN → ReLU → Conv → BN → SE → Add → ReLU
    """
    def __init__(self, channels, se_ratio=8):
        """
        Args:
            channels: 通道数
            se_ratio: SE 注意力压缩比
        """
        super(ResidualBlock, self).__init__()

        # 第一层卷积
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)

        # 第二层卷积
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

        # SE 注意力
        self.se = SEBlock(channels, ratio=se_ratio)

        # ReLU
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        """
        Args:
            x: 输入张量
        Returns:
            残差块输出
        """
        residual = x

        # 第一层
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        # 第二层
        out = self.conv2(out)
        out = self.bn2(out)

        # SE 注意力
        out = self.se(out)

        # 残差连接
        out += residual
        out = self.relu(out)

        return out


class SEResNetGomoku(nn.Module):
    """
    SE-ResNet 五子棋模型

    改进点：
    1. 更深的网络（10 个残差块 vs 3 个）
    2. 更多滤波器（128 vs 64）
    3. SE 注意力机制
    4. 改进的策略头（2 层卷积 + Dropout）
    5. 增强的价值头（更大的隐藏层）
    """
    def __init__(
        self,
        board_size=15,
        num_res_blocks=10,
        num_filters=128,
        se_ratio=8,
        dropout_rate=0.3,
        value_head_hidden=512
    ):
        """
        Args:
            board_size: 棋盘大小
            num_res_blocks: 残差块数量
            num_filters: 卷积滤波器数量
            se_ratio: SE 注意力压缩比
            dropout_rate: Dropout 比率
            value_head_hidden: 价值头隐藏层大小
        """
        super(SEResNetGomoku, self).__init__()

        self.board_size = board_size
        self.num_filters = num_filters

        # ===== 初始卷积层 =====
        self.initial_conv = nn.Conv2d(3, num_filters, kernel_size=3, padding=1, bias=False)
        self.initial_bn = nn.BatchNorm2d(num_filters)
        self.initial_relu = nn.ReLU(inplace=True)

        # ===== 残差块塔 =====
        self.res_blocks = nn.ModuleList([
            ResidualBlock(num_filters, se_ratio=se_ratio)
            for _ in range(num_res_blocks)
        ])

        # ===== 策略头 (Policy Head) =====
        self.policy_conv1 = nn.Conv2d(num_filters, 4, kernel_size=1, bias=False)
        self.policy_bn1 = nn.BatchNorm2d(4)
        self.policy_relu1 = nn.ReLU(inplace=True)

        self.policy_conv2 = nn.Conv2d(4, 2, kernel_size=1, bias=False)
        self.policy_bn2 = nn.BatchNorm2d(2)
        self.policy_relu2 = nn.ReLU(inplace=True)

        self.policy_dropout = nn.Dropout(dropout_rate) if dropout_rate > 0 else nn.Identity()
        self.policy_fc = nn.Linear(2 * board_size * board_size, board_size * board_size)

        # ===== 价值头 (Value Head) =====
        self.value_conv = nn.Conv2d(num_filters, 2, kernel_size=1, bias=False)
        self.value_bn = nn.BatchNorm2d(2)
        self.value_relu = nn.ReLU(inplace=True)

        self.value_fc1 = nn.Linear(2 * board_size * board_size, value_head_hidden)
        self.value_dropout = nn.Dropout(dropout_rate) if dropout_rate > 0 else nn.Identity()
        self.value_fc2 = nn.Linear(value_head_hidden, 256)
        self.value_fc3 = nn.Linear(256, 1)

        # 初始化权重
        self._initialize_weights()

    def _initialize_weights(self):
        """He 初始化"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        """
        前向传播

        Args:
            x: (batch, 3, board_size, board_size)
               3 个通道: [我方棋子, 对方棋子, 当前回合]

        Returns:
            policy: (batch, board_size * board_size) - 策略分布
            value: (batch, 1) - 价值估计 [-1, 1]
        """
        # ===== 初始卷积 =====
        x = self.initial_conv(x)
        x = self.initial_bn(x)
        x = self.initial_relu(x)

        # ===== 残差块塔 =====
        for res_block in self.res_blocks:
            x = res_block(x)

        # ===== 策略头 =====
        policy = self.policy_conv1(x)
        policy = self.policy_bn1(policy)
        policy = self.policy_relu1(policy)

        policy = self.policy_conv2(policy)
        policy = self.policy_bn2(policy)
        policy = self.policy_relu2(policy)

        policy = policy.reshape(policy.size(0), -1)  # Flatten
        policy = self.policy_dropout(policy)
        policy = self.policy_fc(policy)
        policy = F.softmax(policy, dim=1)  # Softmax

        # ===== 价值头 =====
        value = self.value_conv(x)
        value = self.value_bn(value)
        value = self.value_relu(value)

        value = value.reshape(value.size(0), -1)  # Flatten
        value = self.value_fc1(value)
        value = F.relu(value)
        value = self.value_dropout(value)
        value = self.value_fc2(value)
        value = F.relu(value)
        value = self.value_fc3(value)
        value = torch.tanh(value)  # Tanh 激活 [-1, 1]

        return policy, value


def create_enhanced_model(
    board_size=15,
    num_res_blocks=10,
    num_filters=128,
    se_ratio=8,
    l2_reg=1e-4,
    dropout_rate=0.3,
    value_head_hidden=512,
    device='cuda' if torch.cuda.is_available() else 'cpu'
):
    """
    创建增强的 SE-ResNet 模型

    Args:
        board_size: 棋盘大小
        num_res_blocks: 残差块数量
        num_filters: 卷积滤波器数量
        se_ratio: SE 注意力压缩比
        l2_reg: L2 正则化系数（PyTorch 中通过 optimizer 的 weight_decay 实现）
        dropout_rate: Dropout 比率
        value_head_hidden: 价值头隐藏层大小
        device: 设备 ('cuda' 或 'cpu')

    Returns:
        model: SEResNetGomoku 模型
        l2_reg: L2 正则化系数（用于 optimizer）
    """
    model = SEResNetGomoku(
        board_size=board_size,
        num_res_blocks=num_res_blocks,
        num_filters=num_filters,
        se_ratio=se_ratio,
        dropout_rate=dropout_rate,
        value_head_hidden=value_head_hidden
    )

    model = model.to(device)

    return model, l2_reg


if __name__ == '__main__':
    import numpy as np

    print("=" * 60)
    print("测试 PyTorch SE-ResNet 模型")
    print("=" * 60)

    # 检查设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n设备: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # 创建模型（测试配置）
    print("\n创建模型（测试配置）...")
    model, l2_reg = create_enhanced_model(
        board_size=15,
        num_res_blocks=3,
        num_filters=64,
        se_ratio=8,
        dropout_rate=0.3,
        device=device
    )

    # 打印模型摘要
    print("\n模型架构：")
    print(model)

    # 计算参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n总参数量: {total_params:,}")
    print(f"可训练参数: {trainable_params:,}")

    # 测试前向传播
    print("\n测试前向传播...")
    batch_size = 4
    test_input = torch.randn(batch_size, 3, 15, 15).to(device)
    print(f"输入形状: {test_input.shape}")

    model.eval()
    with torch.no_grad():
        policy, value = model(test_input)

    print(f"策略输出形状: {policy.shape}, 和: {policy[0].sum():.6f}")
    print(f"价值输出形状: {value.shape}, 值范围: [{value.min():.6f}, {value.max():.6f}]")

    # 检查是否有 NaN
    if torch.isnan(policy).any() or torch.isnan(value).any():
        print("❌ 检测到 NaN！")
    else:
        print("✅ 无 NaN，数值稳定")

    # 创建完整配置模型
    print("\n创建完整配置模型...")
    full_model, _ = create_enhanced_model(
        board_size=15,
        num_res_blocks=10,
        num_filters=128,
        se_ratio=8,
        device=device
    )
    full_params = sum(p.numel() for p in full_model.parameters())
    print(f"完整模型参数量: {full_params:,}")
    print(f"参数量增加: {full_params / total_params:.2f}x")

    print("\n✓ 模型创建和测试成功！")
