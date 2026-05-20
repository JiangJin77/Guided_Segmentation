"""
阈值分割与可学习病理先验概率图模块

将get_threshold.py中的硬阈值分割和set_dataset.py中的多尺度卷积引导图
统一为可学习模块：

    ThresholdSegmentation:
        直接对应于get_threshold.py的阈值分割逻辑 (data >= lower) & (data <= upper)
        lower/upper作为可学习参数，训练时使用sigmoid近似保证可微，推理时使用硬阈值

    LearnablePriorMap:
        在ThresholdSegmentation的基础上增加多尺度卷积和后处理，生成精细化的病理先验概率图
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class ThresholdSegmentation(nn.Module):
    """可学习阈值分割模块
    直接对应于get_threshold.py中的阈值分割逻辑：
        threshold_array = ((data_array >= lower) & (data_array <= upper)).astype(np.float32)

    训练阶段使用sigmoid近似保证梯度回传，推理阶段使用硬阈值。
    lower和upper初始值由get_threshold.py的网格搜索结果确定（默认17.5~22.0）。
    """
    def __init__(self, init_lower=17.5, init_upper=22.0):
        super(ThresholdSegmentation, self).__init__()

        # 可学习阈值参数（通过softplus保证lower > 0）
        self.lower = nn.Parameter(torch.tensor(init_lower, dtype=torch.float32))
        # 学习upper-lower的差值，确保upper > lower
        self.offset = nn.Parameter(torch.tensor(init_upper - init_lower, dtype=torch.float32))

    def get_threshold(self):
        """返回当前的lower和upper值"""
        lower = F.softplus(self.lower)
        upper = lower + F.softplus(self.offset) + 1e-4
        return lower, upper

    def forward(self, x, hard=False):
        """阈值分割

        Args:
            x: 输入图像 (B, 1, H, W) 或 (B, C, H, W)
            hard: 是否使用硬阈值（推理时用）

        Returns:
            mask: 二值掩码 (B, 1, H, W)
            lower: 当前threshold下限
            upper: 当前threshold上限
        """
        lower, upper = self.get_threshold()

        if hard or not self.training:
            # 硬阈值 —— 与get_threshold.py完全一致
            mask = ((x >= lower) & (x <= upper)).to(x.dtype)
        else:
            # 可微阈值 —— sigmoid近似 (x >= lower) & (x <= upper)
            temperature = 0.1
            mask = torch.sigmoid((x - lower) / temperature) * \
                   torch.sigmoid((upper - x) / temperature)

        return mask, lower, upper


class LearnablePriorMap(nn.Module):
    """可学习的病理先验概率图
    在ThresholdSegmentation的基础上，对阈值分割结果进行多尺度卷积融合，
    生成精细化的先验概率图。

    Args:
        init_lower: 阈值下限初始值（来自grid search）
        init_upper: 阈值上限初始值（来自grid search）
        radii: 多尺度卷积核半径列表
    """
    def __init__(self, init_lower=17.5, init_upper=22.0, radii=None):
        super(LearnablePriorMap, self).__init__()
        if radii is None:
            radii = range(1, 4)
        self.radii = list(radii)

        # 显式的阈值分割模块
        self.threshold = ThresholdSegmentation(
            init_lower=init_lower,
            init_upper=init_upper,
        )

        # 可学习的多尺度卷积核权重（初始值由alpha=0.75, beta=0.25的衰减公式确定）
        init_weights = []
        for r in self.radii:
            w = math.e ** (-0.75 * (r ** 0.25))
            init_weights.append(w / ((2 * r + 1) ** 2))
        self.scale_weights = nn.Parameter(torch.tensor(init_weights, dtype=torch.float32))

        # 多尺度融合后处理
        self.post_conv = nn.Sequential(
            nn.Conv2d(len(radii), 8, kernel_size=3, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(8, 1, kernel_size=1),
        )

    def forward(self, x, hard=False):
        """生成病理先验概率图

        Args:
            x: 输入图像 (B, 1, H, W)
            hard: 是否使用硬阈值（推理时用）

        Returns:
            prior_map: 病理先验概率图 (B, 1, H, W)
            gate_map: 阈值门控掩码 (B, 1, H, W)
            lower: 当前阈值下限
            upper: 当前阈值上限
        """
        # 1. 阈值分割
        gate_map, lower, upper = self.threshold(x, hard=hard)

        # 2. 多尺度卷积（带可学习权重）
        multi_scale_maps = []
        for i, r in enumerate(self.radii):
            k = 2 * r + 1
            w = F.softplus(self.scale_weights[i])
            kernel = torch.ones(1, 1, k, k, device=x.device) * w
            conv_out = F.conv2d(gate_map, kernel, padding=r)
            multi_scale_maps.append(conv_out)

        # 3. 多尺度融合
        stacked = torch.cat(multi_scale_maps, dim=1)
        prior_map = self.post_conv(stacked)

        return prior_map, gate_map, lower, upper


if __name__ == '__main__':
    # 测试ThresholdSegmentation
    ts = ThresholdSegmentation()
    x = torch.randn(4, 1, 256, 256)
    mask, lower, upper = ts(x)
    print(f"阈值分割: lower={lower.item():.2f}, upper={upper.item():.2f}")
    print(f"掩码形状: {mask.shape}")

    # 测试LearnablePriorMap
    lpm = LearnablePriorMap()
    prior, gate, lo, up = lpm(x)
    print(f"\n先验概率图: {prior.shape}")
    print(f"门控掩码:   {gate.shape}")
    print(f"阈值:       lower={lo.item():.2f}, upper={up.item():.2f}")
    print(f"参数量:     {sum(p.numel() for p in lpm.parameters())}")

    # 测试硬阈值模式（推理）
    lpm.eval()
    prior_h, gate_h, lo_h, up_h = lpm(x, hard=True)
    print(f"\n[推理模式 - 硬阈值]")
    print(f"gate唯一值: {gate_h.unique()}")
    print(f"阈值:       lower={lo_h.item():.2f}, upper={up_h.item():.2f}")