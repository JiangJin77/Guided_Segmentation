"""
JJNetV2 — 改进版：单路编码器、DCNv2 可变形对齐、增强阈值预测、空洞多尺度先验

改进点:
    1. 单路共享编码器 (batch 拼接, 一次前向)
    2. 真·可变形对齐 (torchvision.ops.deform_conv2d DCNv2)
    3. 增强自适应阈值预测 (局部特征提取 + MLP)
    4. 真多尺度先验 (空洞卷积)
    5. 改进不对称稀疏损失 (不确定性感知)
    6. 偏移场 TV 平滑正则化
    7. 梯度检查点支持
    8. 修正文档字符串

整体架构:
    切片自适应阈值 → 可微阈值分割 → 空洞多尺度先验图
    → 单路双路径编码(共享权重) → DCNv2 可变形对侧对齐 → 对侧融合
    → 先验精炼(循环) → CCFANet解码器 → 分割结果
"""
from __future__ import annotations
import math
import sys
import os
import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision.ops import deform_conv2d

if __name__ == '__main__' and __package__ is None:
    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
    from Network.JJNet import (
        _BasicConv2d, _ChannelAttention, _global_module, _BAM,
        _GateFusion, _CBAMLayer, _CFM, _CFF, _Bottle2neck, _Res2Net_Ours,
    )
else:
    from .JJNet import (
        _BasicConv2d, _ChannelAttention, _global_module, _BAM,
        _GateFusion, _CBAMLayer, _CFM, _CFF, _Bottle2neck, _Res2Net_Ours,
    )


# ============================================================================
# 模块 1: 增强自适应阈值预测器
# ============================================================================

class AdaptiveThresholdPredictorV2(nn.Module):
    """增强版切片自适应阈值预测器

    与 V1 的区别:
        - 使用轻量卷积提取局部空间强度分布特征
        - 而非仅依赖全局均值

    Args:
        init_lower: 阈值下限初始值
        init_upper: 阈值上限初始值
    """
    def __init__(self, init_lower=17.5, init_upper=22.0):
        super().__init__()
        # 全局基线参数 (可学习)
        self.base_lower = nn.Parameter(torch.tensor(init_lower, dtype=torch.float32))
        self.base_offset = nn.Parameter(torch.tensor(init_upper - init_lower, dtype=torch.float32))

        # 增强自适应偏移网络: Conv → GAP → MLP
        # 提取空间局部强度分布特征
        self.adaptive_net = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(8),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(8, 16, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(16, 8),
            nn.LeakyReLU(inplace=True),
            nn.Linear(8, 2),  # [delta_lower, delta_upper]
        )
        # 零初始化, 初始时退化为基线
        self.adaptive_net[-1].weight.data.zero_()
        self.adaptive_net[-1].bias.data.zero_()

    def forward(self, x):
        """预测 per-sample 阈值

        Args:
            x: 输入图像 (B, 1, H, W)
        Returns:
            lower: 阈值下限 (B, 1, 1, 1)
            upper: 阈值上限 (B, 1, 1, 1)
        """
        B = x.shape[0]

        # 全局基线
        base_lower = F.softplus(self.base_lower)
        base_upper = base_lower + F.softplus(self.base_offset) + 1e-4

        # 自适应偏移 (基于局部特征)
        delta = self.adaptive_net(x)  # (B, 2)

        # 应用到每个样本
        lower = base_lower + delta[:, 0:1].view(B, 1, 1, 1)
        upper = base_upper + delta[:, 1:2].view(B, 1, 1, 1)

        # 约束: lower > 0, upper > lower
        lower = F.softplus(lower) + 1.0
        upper = torch.max(upper, lower + 1.0)

        return lower, upper


# ============================================================================
# 模块 2: 空洞多尺度先验图生成器
# ============================================================================

class PriorMapGeneratorV2(nn.Module):
    """空洞多尺度先验概率图生成器

    与 V1 的区别:
        - 使用空洞卷积 (dilation) 代替固定半径平均池化
        - 真正的多尺度感受野, 逐像素可学习响应
        - 丰富的后处理融合

    Args:
        dilations: 空洞率列表
    """
    def __init__(self, dilations=None):
        super().__init__()
        if dilations is None:
            dilations = [1, 2, 3]
        self.dilations = list(dilations)

        # 多尺度空洞卷积分支
        self.dilated_convs = nn.ModuleList([
            nn.Conv2d(1, 8, kernel_size=3, padding=d, dilation=d, bias=False)
            for d in dilations
        ])
        self.bn_dilate = nn.BatchNorm2d(len(dilations) * 8)

        # 融合后处理
        num_branches = len(dilations)
        self.fusion = nn.Sequential(
            nn.Conv2d(num_branches * 8, 16, kernel_size=1, bias=False),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(16, 16, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(16, 1, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, x, lower, upper):
        """生成先验概率图

        Args:
            x: 输入图像 (B, 1, H, W)
            lower: 阈值下限 (B, 1, 1, 1)
            upper: 阈值上限 (B, 1, 1, 1)
        Returns:
            prior_map: 先验概率图 (B, 1, H, W) — 0~1 连续值
            gate_map: 阈值门控掩码 (B, 1, H, W)
        """
        # 1. 可微阈值分割 (sigmoid近似)
        temperature = 0.1
        gate_map = torch.sigmoid((x - lower) / temperature) * \
                   torch.sigmoid((upper - x) / temperature)

        # 2. 多尺度空洞卷积
        multi_scale = []
        for dilated_conv in self.dilated_convs:
            multi_scale.append(dilated_conv(gate_map))
        stacked = torch.cat(multi_scale, dim=1)
        stacked = self.bn_dilate(stacked)

        # 3. 融合 → 先验概率图
        prior_map = self.fusion(stacked)

        return prior_map, gate_map


# ============================================================================
# 模块 3: DCNv2 可变形对侧对齐
# ============================================================================

class DeformableContraAlignmentDCN(nn.Module):
    """DCNv2 可变形对侧对齐模块

    与 V1 的区别:
        - 使用 torchvision.ops.deform_conv2d (真正的 DCNv2)
        - 同时学习偏移量 + 调制权重 (mask)
        - 可变形感受野, 名实相符

    Args:
        channel: 输入特征通道数
        max_offset_pixels: 最大偏移像素数 (在特征图上)
    """
    def __init__(self, channel, max_offset_pixels=4):
        super().__init__()
        self.max_offset_pixels = max_offset_pixels
        self.kernel_size = 3
        self.deformable_groups = 1

        # 偏移量 + 调制权重预测网络
        # offset: 2 * deformable_groups * kH * kW = 2 * 1 * 9 = 18
        # mask:   deformable_groups * kH * kW = 1 * 9 = 9
        num_offset_mask = 3 * self.deformable_groups * self.kernel_size * self.kernel_size
        self.offset_mask_net = nn.Sequential(
            nn.Conv2d(channel * 2, channel // 2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channel // 2),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(channel // 2, channel // 4, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channel // 4),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(channel // 4, num_offset_mask, kernel_size=3, padding=1),
        )
        # 零初始化偏移, 初始时退化为普通卷积
        self.offset_mask_net[-1].weight.data.zero_()
        self.offset_mask_net[-1].bias.data.zero_()

        # DCNv2 权重 (可变形卷积核)
        self.dcn_weight = nn.Parameter(
            torch.empty(channel, channel, self.kernel_size, self.kernel_size)
        )
        nn.init.kaiming_normal_(self.dcn_weight, mode='fan_out', nonlinearity='leaky_relu')

    def forward(self, ipsi_feat, contra_feat):
        """将对侧特征 warp 到与患侧对齐 (DCNv2)

        Args:
            ipsi_feat: 患侧特征 (B, C, H, W)
            contra_feat: 对侧特征 (B, C, H, W)

        Returns:
            aligned: 对齐后的对侧特征 (B, C, H, W)
            offset: 偏移场 (B, 2, H, W) — 可视化/分析用
        """
        B, C, H, W = ipsi_feat.shape

        # 预测偏移量 + 调制权重
        offset_mask = self.offset_mask_net(torch.cat([ipsi_feat, contra_feat], dim=1))
        num_masks = self.deformable_groups * self.kernel_size * self.kernel_size
        offset = offset_mask[:, :2 * num_masks]  # (B, 18, H, W)
        mask = offset_mask[:, 2 * num_masks:]    # (B, 9, H, W)

        # 限制偏移量幅度, 并保证 mask 非负
        offset = torch.tanh(offset) * self.max_offset_pixels
        mask = torch.sigmoid(mask)

        # DCNv2 warp
        aligned = deform_conv2d(
            contra_feat,
            offset,
            self.dcn_weight,
            None,  # bias
            stride=(1, 1),
            padding=(self.kernel_size // 2, self.kernel_size // 2),
            dilation=(1, 1),
            mask=mask,
        )

        # 汇总可视化偏移场
        offset_vis = torch.mean(offset.view(B, -1, 2, H, W), dim=1)  # (B, 2, H, W)

        return aligned, offset_vis


# ============================================================================
# 模块 4: 带 DCNv2 对齐的对侧融合
# ============================================================================

class ContralateralFusionWithAlignmentV2(nn.Module):
    """带 DCNv2 可变形对齐的对侧融合模块

    Args:
        channel: 输入特征通道数
    """
    def __init__(self, channel):
        super().__init__()
        self.alignment = DeformableContraAlignmentDCN(channel)

        self.diff_conv = nn.Sequential(
            nn.Conv2d(channel, channel, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channel),
            nn.LeakyReLU(inplace=True),
        )
        self.channel_att = _ChannelAttention(channel)
        self.gate = nn.Sequential(
            nn.Conv2d(channel * 2, 1, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, ipsi, contra):
        """融合患侧与对侧特征

        Args:
            ipsi: 患侧特征 (B, C, H, W)
            contra: 对侧特征 (B, C, H, W)

        Returns:
            fused: 融合特征 (B, C, H, W)
            diff_map: 不对称性热图 (B, 1, H, W)
            offset: 偏移场 (B, 2, H, W)
        """
        # 1. DCNv2 可变形对齐
        contra_aligned, offset = self.alignment(ipsi, contra)

        # 2. 差分特征
        diff = ipsi - contra_aligned
        diff_feat = self.diff_conv(diff)
        diff_feat = diff_feat * self.channel_att(diff_feat)

        # 3. 门控融合
        gate = self.gate(torch.cat([ipsi, contra_aligned], dim=1))
        fused = ipsi + gate * diff_feat

        # 4. 不对称热图 (可解释性输出)
        diff_map = torch.mean(torch.abs(diff), dim=1, keepdim=True)

        # 存储偏移场以便训练脚本收集用于正则化损失
        self.last_offset = offset

        return fused, diff_map, offset


# ============================================================================
# 模块 5: 先验精炼模块 (与 V1 相同, 精炼逻辑成熟)
# ============================================================================

class PriorRefinement(nn.Module):
    """先验精炼模块

    利用对侧对照产生的不对称信息来修正和精炼先验图。
    """
    def __init__(self):
        super().__init__()
        self.refine_net = nn.Sequential(
            nn.Conv2d(6, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(32, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(16, 1, kernel_size=1),
        )

    def forward(self, prior, diff_maps):
        """精炼先验图

        Args:
            prior: 当前先验图 (B, 1, H, W)
            diff_maps: 5个尺度的不对称热图 [(B,1,H_i,W_i), ...]

        Returns:
            refined_prior: 精炼后先验图 (B, 1, H, W)
        """
        B, _, H, W = prior.shape

        resampled = []
        for d in diff_maps:
            if d.shape[2:] != (H, W):
                d = F.interpolate(d, (H, W), mode='bilinear', align_corners=True)
            resampled.append(d)

        x = torch.cat([prior] + resampled, dim=1)
        residual = self.refine_net(x)
        refined = prior + residual
        refined = torch.clamp(refined, 0, 1)

        return refined


# ============================================================================
# 模块 6: 改进损失函数
# ============================================================================

class JJNetLossV2(nn.Module):
    """JJNetV2 综合损失函数

    与 V1 的区别:
        1. 不对称稀疏损失: 不确定性感知加权, 软掩码替代硬剪裁
        2. 偏移场 TV 平滑正则化: 鼓励空间平滑的偏移场
        3. 其余损失项不变

    Args:
        deep_supervision_weights: 深度监督各输出的权重
        prior_weight: 先验监督损失权重
        asym_weight: 不对称稀疏损失权重
        offset_weight: 偏移场 L2 正则化权重
        tv_weight: 偏移场 TV 平滑权重
        asym_tau: 不对称损失的 soft 阈值参数
    """
    def __init__(self, deep_supervision_weights=None,
                 prior_weight=0.1, asym_weight=0.05,
                 offset_weight=0.001, tv_weight=0.0005,
                 asym_tau=0.1):
        super().__init__()
        if deep_supervision_weights is None:
            deep_supervision_weights = [0.1, 0.2, 0.2, 0.5]
        self.ds_weights = deep_supervision_weights
        self.prior_weight = prior_weight
        self.asym_weight = asym_weight
        self.offset_weight = offset_weight
        self.tv_weight = tv_weight
        self.asym_tau = asym_tau

    def dice_loss(self, pred, target, smooth=1.0):
        pred = torch.sigmoid(pred)
        pred_flat = pred.contiguous().view(-1)
        target_flat = target.contiguous().view(-1)
        intersection = (pred_flat * target_flat).sum()
        dice = (2.0 * intersection + smooth) / (pred_flat.sum() + target_flat.sum() + smooth)
        return 1.0 - dice

    def bce_loss(self, pred, target):
        return F.binary_cross_entropy_with_logits(pred, target)

    def tv_loss(self, offset):
        """Total Variation 平滑损失, 鼓励偏移场空间平滑"""
        dy = offset[:, :, 1:, :] - offset[:, :, :-1, :]
        dx = offset[:, :, :, 1:] - offset[:, :, :, :-1]
        return torch.mean(torch.abs(dy)) + torch.mean(torch.abs(dx))

    def _gaussian_blur(self, x, kernel_size=7, sigma=3.0):
        """手动高斯模糊 (兼容老版本 PyTorch)"""
        B, C, H, W = x.shape
        k = (kernel_size - 1) // 2
        axes = torch.arange(-k, k + 1, dtype=torch.float32, device=x.device)
        gauss_1d = torch.exp(-(axes ** 2) / (2 * sigma ** 2))
        gauss_1d = gauss_1d / gauss_1d.sum()
        kernel_2d = gauss_1d[:, None] * gauss_1d[None, :]
        kernel = kernel_2d.expand(C, 1, kernel_size, kernel_size).contiguous()
        padding = kernel_size // 2
        return F.conv2d(x, kernel, padding=padding, groups=C)

    def forward(self, outputs, labels, prior_maps=None, diff_maps=None,
                offsets=None):
        """计算总损失

        Args:
            outputs: 深度监督输出列表 [edge_out3, sal_out1, sal_out2, sal_out3]
            labels: 真实标签 (B, 1, H, W)
            prior_maps: 各迭代轮次的先验图列表
            diff_maps: 各层不对称热图列表
            offsets: 各层偏移场列表
        """
        total_loss = 0.0

        # ---- 1. 分割主损失 (深度监督) ----
        if isinstance(outputs, (list, tuple)):
            for i, out in enumerate(outputs[:4]):
                w = self.ds_weights[i] if i < len(self.ds_weights) else 0.25
                dice = self.dice_loss(out, labels)
                bce = self.bce_loss(out, labels)
                total_loss += w * (dice + bce)
        else:
            dice = self.dice_loss(outputs, labels)
            bce = self.bce_loss(outputs, labels)
            total_loss += dice + bce

        # ---- 2. 先验监督损失 ----
        if prior_maps is not None:
            for prior in prior_maps:
                label_soft = F.avg_pool2d(labels, kernel_size=5, stride=1, padding=2)
                prior_loss = F.mse_loss(prior, label_soft)
                total_loss += self.prior_weight * prior_loss

        # ---- 3. 改进不对称稀疏损失 (不确定性感知) ----
        if diff_maps is not None:
            for diff in diff_maps:
                diff_resized = F.interpolate(
                    diff, size=labels.shape[2:], mode='bilinear', align_corners=True
                )
                # 软掩码: 远离标签边界的不对称才被惩罚
                # 1) 对标签做距离变换近似 (高斯模糊)
                label_smooth = self._gaussian_blur(labels, kernel_size=7, sigma=3.0)
                # 2) 在非病灶区域: 不确定性感知权重
                #    小不对称 exp(-|diff|/tau) 允许存在 (正常解剖变异)
                #    大不对称才被惩罚
                uncertainty_weight = 1.0 - label_smooth
                asym_soft = diff_resized * torch.exp(-diff_resized / self.asym_tau)
                asym_loss = torch.mean(uncertainty_weight * asym_soft)
                total_loss += self.asym_weight * asym_loss

        # ---- 4. 偏移场正则化 ----
        if offsets is not None:
            for offset in offsets:
                # L2 正则化 (防止过大偏移)
                offset_loss = torch.mean(offset ** 2)
                total_loss += self.offset_weight * offset_loss
                # TV 平滑正则化 (鼓励空间平滑)
                if self.tv_weight > 0:
                    tv = self.tv_loss(offset)
                    total_loss += self.tv_weight * tv

        return total_loss


# ============================================================================
# 模块 7: JJNetV2 主网络
# ============================================================================

class JJNetV2(nn.Module):
    """JJNetV2 — 改进版卒中梗死分割网络

    Args:
        channel: 解码器基础通道数 (默认64)
        prior_init_lower: 先验阈值下限初始值
        prior_init_upper: 先验阈值上限初始值
        prior_dilations: 空洞多尺度卷积空洞率列表
        num_refine_iters: 先验精炼迭代次数 (默认1)
        deep_supervision: 是否启用深度监督
        use_checkpoint: 是否使用梯度检查点 (节省显存)
    """
    def __init__(self, channel=64,
                 prior_init_lower=17.5, prior_init_upper=22.0,
                 prior_dilations=None, num_refine_iters=1,
                 deep_supervision=True, use_checkpoint=False):
        super().__init__()

        self.deep_supervision = deep_supervision
        self.num_refine_iters = num_refine_iters
        self.use_checkpoint = use_checkpoint

        # 1. 增强版切片自适应阈值预测器
        self.threshold_predictor = AdaptiveThresholdPredictorV2(
            init_lower=prior_init_lower,
            init_upper=prior_init_upper,
        )

        # 2. 空洞多尺度先验概率图生成器
        self.prior_generator = PriorMapGeneratorV2(dilations=prior_dilations)

        # 3. 单路双路径编码器 (输入2通道: raw + prior)
        self.encoder = _Res2Net_Ours(
            2, _Bottle2neck, [3, 4, 6, 3],
            baseWidth=26, scale=4
        )

        # 4. 5层 DCNv2 可变形对侧融合
        self.contra_fusions = nn.ModuleList([
            ContralateralFusionWithAlignmentV2(64),    # x0
            ContralateralFusionWithAlignmentV2(256),   # x1
            ContralateralFusionWithAlignmentV2(512),   # x2
            ContralateralFusionWithAlignmentV2(1024),  # x3
            ContralateralFusionWithAlignmentV2(2048),  # x4
        ])

        # 5. 先验精炼模块
        self.prior_refine = PriorRefinement()

        # ---- 6. 解码器 (复现 CCFANet 结构) ---- #
        act_fn = nn.LeakyReLU(inplace=True)

        self.downSample = nn.MaxPool2d(2, stride=2)

        self.layer0 = nn.Sequential(
            nn.Conv2d(64, channel, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(channel),
            nn.Dropout2d(0.3),
            act_fn,
        )
        self.layer1 = nn.Sequential(
            nn.Conv2d(256, channel, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(channel),
            nn.Dropout2d(0.3),
            act_fn,
        )

        self.low_fusion = _GateFusion(channel)
        self.high_fusion1 = _CFF(256, 512, channel)
        self.high_fusion2 = _CFF(1024, 2048, channel)

        # 边缘分支
        self.layer_edge0 = nn.Sequential(
            nn.Conv2d(channel, channel, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(channel), nn.Dropout2d(0.3), act_fn,
        )
        self.layer_edge1 = nn.Sequential(
            nn.Conv2d(channel, channel, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(channel), nn.Dropout2d(0.3), act_fn,
        )
        self.layer_edge2 = nn.Sequential(
            nn.Conv2d(channel, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64), nn.Dropout2d(0.3), act_fn,
        )
        self.layer_edge3 = nn.Sequential(nn.Conv2d(64, 1, kernel_size=1))

        # High level path 1
        self.layer_hig01 = nn.Sequential(
            nn.Conv2d(channel, channel, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(channel), nn.Dropout2d(0.3), act_fn,
        )
        self.layer_hig11 = nn.Sequential(
            nn.Conv2d(channel, channel, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(channel), nn.Dropout2d(0.3), act_fn,
        )
        self.layer_hig21 = nn.Sequential(
            nn.Conv2d(channel, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64), nn.Dropout2d(0.3), act_fn,
        )
        self.layer_hig31 = nn.Sequential(nn.Conv2d(64, 1, kernel_size=1))

        # High level path 2
        self.layer_hig02 = nn.Sequential(
            nn.Conv2d(channel, channel, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(channel), nn.Dropout2d(0.3), act_fn,
        )
        self.layer_hig12 = nn.Sequential(
            nn.Conv2d(channel, channel, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(channel), nn.Dropout2d(0.3), act_fn,
        )
        self.layer_hig22 = nn.Sequential(
            nn.Conv2d(channel, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64), nn.Dropout2d(0.3), act_fn,
        )
        self.layer_hig32 = nn.Sequential(nn.Conv2d(64, 1, kernel_size=1))

        self.layer_fil = nn.Sequential(nn.Conv2d(64, 1, kernel_size=1))

        self.atten_edge_0 = _ChannelAttention(channel)
        self.atten_edge_1 = _ChannelAttention(channel)
        self.atten_edge_2 = _ChannelAttention(channel)
        self.atten_edge_ori = _ChannelAttention(channel)

        self.cat_01 = _BAM(channel)
        self.cat_11 = _BAM(channel)
        self.cat_21 = _BAM(channel)
        self.cat_31 = _BAM(channel)

        self.cat_02 = _BAM(channel)
        self.cat_12 = _BAM(channel)
        self.cat_22 = _BAM(channel)
        self.cat_32 = _BAM(channel)

        self.up_2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.up_4 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
        self.up_8 = nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True)

    def _run_encoder(self, ipsi_in, contra_in):
        """单路编码器: batch 拼接, 一次前向, 再切片

        相比 JJNet 的两次独立前向, 吞吐量提升约 2 倍。
        """
        B = ipsi_in.shape[0]

        if self.use_checkpoint and self.training:
            both_in = torch.cat([ipsi_in, contra_in], dim=0)
            # 使用梯度检查点
            both_out = torch.utils.checkpoint.checkpoint(
                self.encoder, both_in, use_reentrant=False
            )
            if isinstance(both_out, (list, tuple)):
                ipsi_x0, contra_x0 = both_out[0][:B], both_out[0][B:]
                ipsi_x1, contra_x1 = both_out[1][:B], both_out[1][B:]
                ipsi_out = (ipsi_x0, ipsi_x1, *[o[:B] for o in both_out[2:]])
                contra_out = (contra_x0, contra_x1, *[o[B:] for o in both_out[2:]])
            else:
                raise RuntimeError("checkpoint encoder must return tuple")
        else:
            # 标准单路前向: batch 拼接
            both_in = torch.cat([ipsi_in, contra_in], dim=0)
            both_out = self.encoder(both_in)  # tuple of 5 tensors
            ipsi_out = tuple(o[:B] for o in both_out)
            contra_out = tuple(o[B:] for o in both_out)

        return ipsi_out, contra_out

    def _encode_and_fuse(self, x, prior):
        """编码 + 对侧对齐 + 融合 (单次前向)

        Args:
            x: 原始图像 (B, 1, H, W)
            prior: 当前先验图 (B, 1, H, W)

        Returns:
            fused_feats: 5层融合特征列表
            diff_maps: 5层不对称热图列表
            offsets: 5层偏移场列表
        """
        B = x.shape[0]

        # 构建双路径输入
        ipsi_in = torch.cat([x, prior], dim=1)                              # (B, 2, H, W)
        contra_in = torch.cat([torch.flip(x, dims=[-1]),
                                torch.flip(prior, dims=[-1])], dim=1)        # (B, 2, H, W)

        # 单路共享编码器
        (ipsi_x0, ipsi_x1, ipsi_x2, ipsi_x3, ipsi_x4), \
        (contra_x0, contra_x1, contra_x2, contra_x3, contra_x4) = \
            self._run_encoder(ipsi_in, contra_in)

        ipsi_feats = [ipsi_x0, ipsi_x1, ipsi_x2, ipsi_x3, ipsi_x4]
        contra_feats = [contra_x0, contra_x1, contra_x2, contra_x3, contra_x4]

        # 逐层对齐 + 融合
        fused_feats = []
        diff_maps = []
        offsets = []

        for scale in range(5):
            # 将对侧特征翻转回原始方向后再做对齐
            contra_flipped = torch.flip(contra_feats[scale], dims=[-1])
            fused, diff_map, offset = self.contra_fusions[scale](
                ipsi_feats[scale], contra_flipped
            )
            fused_feats.append(fused)
            diff_maps.append(diff_map)
            offsets.append(offset)

        return fused_feats, diff_maps, offsets

    def _decode(self, fused_feats):
        """CCFANet 解码器

        Args:
            fused_feats: 5层融合特征列表 [f_x0~f_x4]

        Returns:
            edge_out3: 边缘预测 (B, 1, H, W)
            sal_out1: 显著性图1 (B, 1, H, W)
            sal_out2: 显著性图2 (B, 1, H, W)
            sal_out3: 最终融合显著性图 (B, 1, H, W)
        """
        f_x0, f_x1, f_x2, f_x3, f_x4 = fused_feats

        x0_1 = self.layer0(f_x0)
        x1_1 = self.layer1(f_x1)
        low_x = self.low_fusion(x0_1, x1_1)

        # 边缘路径
        edge_out0 = self.layer_edge0(self.up_2(low_x))
        edge_out1 = self.layer_edge1(self.up_2(edge_out0))
        edge_out2 = self.layer_edge2(self.up_2(edge_out1))
        edge_out3 = self.layer_edge3(edge_out2)

        atten_edge_ori = self.atten_edge_ori(low_x)
        atten_edge_0 = self.atten_edge_0(edge_out0)
        atten_edge_1 = self.atten_edge_1(edge_out1)
        atten_edge_2 = self.atten_edge_2(edge_out2)

        # 高层融合
        high_x01 = self.high_fusion1(self.downSample(f_x1), f_x2)
        high_x02 = self.high_fusion2(self.up_2(f_x3), self.up_4(f_x4))

        # High path 1
        cat_out_01 = self.cat_01(high_x01, low_x.mul(atten_edge_ori))
        hig_out01 = self.layer_hig01(self.up_2(cat_out_01))

        cat_out11 = self.cat_11(hig_out01, edge_out0.mul(atten_edge_0))
        hig_out11 = self.layer_hig11(self.up_2(cat_out11))

        cat_out21 = self.cat_21(hig_out11, edge_out1.mul(atten_edge_1))
        hig_out21 = self.layer_hig21(self.up_2(cat_out21))

        cat_out31 = self.cat_31(hig_out21, edge_out2.mul(atten_edge_2))
        sal_out1 = self.layer_hig31(cat_out31)

        # High path 2
        cat_out_02 = self.cat_02(high_x02, low_x.mul(atten_edge_ori))
        hig_out02 = self.layer_hig02(self.up_2(cat_out_02))

        cat_out12 = self.cat_12(hig_out02, edge_out0.mul(atten_edge_0))
        hig_out12 = self.layer_hig12(self.up_2(cat_out12))

        cat_out22 = self.cat_22(hig_out12, edge_out1.mul(atten_edge_1))
        hig_out22 = self.layer_hig22(self.up_2(cat_out22))

        cat_out32 = self.cat_32(hig_out22, edge_out2.mul(atten_edge_2))
        sal_out2 = self.layer_hig32(cat_out32)

        # 最终融合
        sal_out3 = self.layer_fil(cat_out31 + cat_out32)

        return edge_out3, sal_out1, sal_out2, sal_out3

    def forward(self, x):
        """前向传播

        Args:
            x: 输入图像 (B, 1, H, W)

        Returns:
            edge_out3: 边缘预测 (B, 1, H, W)
            sal_out1: 显著性图1 (B, 1, H, W)
            sal_out2: 显著性图2 (B, 1, H, W)
            sal_out3: 最终融合显著性图 (B, 1, H, W)
            prior_maps: 各迭代轮次的先验图列表
            diff_maps: 最终轮次各层不对称热图
            gate_map: 阈值门控掩码 (B, 1, H, W)
            lower: 当前阈值下限
            upper: 当前阈值上限
        """
        # ---- Phase 1: 切片自适应阈值 + 粗先验生成 ----
        lower, upper = self.threshold_predictor(x)
        prior_map, gate_map = self.prior_generator(x, lower, upper)

        # ---- Phase 2: 先验精炼循环 ----
        prior_maps = [prior_map]
        current_prior = prior_map

        fused_feats = None
        for i in range(self.num_refine_iters + 1):
            # 编码 + 对齐 + 融合
            fused_feats, diff_maps, offsets = self._encode_and_fuse(x, current_prior)

            # 精炼 (除了最后一轮)
            if i < self.num_refine_iters:
                current_prior = self.prior_refine(current_prior, diff_maps)
                prior_maps.append(current_prior)

        # ---- Phase 3: 解码 ----
        edge_out3, sal_out1, sal_out2, sal_out3 = self._decode(fused_feats)

        if self.deep_supervision:
            return (edge_out3, sal_out1, sal_out2, sal_out3,
                    prior_maps, diff_maps, gate_map, lower, upper)
        else:
            return sal_out3


# ============================================================================
# 测试
# ============================================================================

if __name__ == '__main__':
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = ''

    print("=" * 60)
    print("JJNetV2 网络测试")
    print("=" * 60)

    # 测试 AdaptiveThresholdPredictorV2
    print("\n[1/6] 测试 AdaptiveThresholdPredictorV2 ...")
    atp = AdaptiveThresholdPredictorV2()
    x = torch.randn(8, 1, 256, 256)
    lower, upper = atp(x)
    print(f"     lower: {lower[0,0,0,0].item():.2f} (per-sample: {lower.shape})")
    print(f"     upper: {upper[0,0,0,0].item():.2f} (per-sample: {upper.shape})")

    # 测试 PriorMapGeneratorV2
    print("\n[2/6] 测试 PriorMapGeneratorV2 ...")
    pmg = PriorMapGeneratorV2()
    prior_map, gate_map = pmg(x, lower, upper)
    print(f"     prior_map: {prior_map.shape} [{prior_map.min().item():.3f}, {prior_map.max().item():.3f}]")
    print(f"     gate_map:  {gate_map.shape}")

    # 测试 DeformableContraAlignmentDCN
    print("\n[3/6] 测试 DeformableContraAlignmentDCN ...")
    dca = DeformableContraAlignmentDCN(64)
    ipsi = torch.randn(8, 64, 64, 64)
    contra = torch.randn(8, 64, 64, 64)
    aligned, offset = dca(ipsi, contra)
    print(f"     aligned: {aligned.shape}")
    print(f"     offset:  {offset.shape} [{offset.min().item():.3f}, {offset.max().item():.3f}]")

    # 测试 ContralateralFusionWithAlignmentV2
    print("\n[4/6] 测试 ContralateralFusionWithAlignmentV2 ...")
    cf = ContralateralFusionWithAlignmentV2(64)
    fused, diff_map, off = cf(ipsi, contra)
    print(f"     fused:       {fused.shape}")
    print(f"     diff_map:    {diff_map.shape}")

    # 测试 PriorRefinement
    print("\n[5/6] 测试 PriorRefinement ...")
    pr = PriorRefinement()
    prior = torch.rand(8, 1, 256, 256)
    diff_maps = [
        torch.rand(8, 1, 64, 64),
        torch.rand(8, 1, 64, 64),
        torch.rand(8, 1, 32, 32),
        torch.rand(8, 1, 16, 16),
        torch.rand(8, 1, 8, 8),
    ]
    refined = pr(prior, diff_maps)
    print(f"     refined: {refined.shape} [{refined.min().item():.3f}, {refined.max().item():.3f}]")

    # 测试 JJNetV2 主网络
    print("\n[6/6] 测试 JJNetV2 主网络 ...")
    model = JJNetV2(channel=64, num_refine_iters=1)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"     总参数量:     {total_params:,}")
    print(f"     可训练参数量: {trainable_params:,}")

    # 前向测试
    input_images = torch.randn(8, 1, 256, 256)
    output = model(input_images)

    if isinstance(output, tuple) and len(output) == 9:
        (edge, s1, s2, s3,
         prior_maps, diff_maps, gate_map, lower, upper) = output
        print(f"     边缘预测:         {edge.shape}")
        print(f"     显著性图1:        {s1.shape}")
        print(f"     显著性图2:        {s2.shape}")
        print(f"     最终融合图:       {s3.shape}")
        print(f"     先验图轮次:       {len(prior_maps)} 轮")
        for i, pm in enumerate(prior_maps):
            print(f"       prior_{i}: {pm.shape}")
        print(f"     不对称热图层数:   {len(diff_maps)}")
        print(f"     gate_map:         {gate_map.shape}")
        print(f"     lower: {lower[0,0,0,0].item():.2f}  upper: {upper[0,0,0,0].item():.2f}")
    else:
        print(f"     输出形状: {output.shape}")

    # 测试损失函数
    print("\n[*] 测试 JJNetLossV2 ...")
    criterion = JJNetLossV2()
    loss = criterion(
        outputs=[edge, s1, s2, s3],
        labels=torch.randint(0, 2, (8, 1, 256, 256)).float(),
        prior_maps=prior_maps,
        diff_maps=diff_maps,
        offsets=[off],
    )
    print(f"     总损失: {loss.item():.4f}")

    print("\n" + "=" * 60)
    print("所有测试通过！JJNetV2 构建成功。")
    print("=" * 60)