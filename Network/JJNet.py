"""
JJNet — 可学习病理先验 + 可变形对侧半脑对照的卒中梗死分割网络

创新点:
    1. AdaptiveThresholdPredictor: 切片自适应阈值预测, 替代全局固定阈值
    2. DeformableContraAlignment: 可变形对侧对齐, 解决大脑生理性不对称
    3. PriorRefinement: 先验-对侧交互精炼, 不对称信息反馈优化先验

整体架构:
    切片自适应阈值 → 可微阈值分割 → 多尺度先验图
    → 双路径编码(共享权重) → 可变形对侧对齐 → 对侧融合
    → 先验精炼(循环) → CCFANet解码器 → 分割结果
"""
import math
import sys
import os
import torch
import torch.nn as nn
import torch.nn.functional as F

if __name__ == '__main__' and __package__ is None:
    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

# ============================================================
# CCFANet 模块内联 (JJNet 独立依赖)
# ============================================================

class _BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super().__init__()
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.LeakyReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class _ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super().__init__()
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(in_planes, in_planes // 16, 1, bias=False)
        self.relu1 = nn.LeakyReLU()
        self.fc2 = nn.Conv2d(in_planes // 16, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        return self.sigmoid(max_out + avg_out)


class _global_module(nn.Module):
    def __init__(self, channels=64, r=4):
        super().__init__()
        out_channels = int(channels // r)
        self.global_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, out_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(inplace=True),
            nn.Dropout2d(0.3),
            nn.Conv2d(out_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )
        self.sig = nn.Sigmoid()

    def forward(self, x):
        return self.sig(self.global_att(x))


class _BAM(nn.Module):
    """Partial Decoder Component (Identification Module)"""
    def __init__(self, channel):
        super().__init__()
        self.relu = nn.LeakyReLU(True)
        self.global_att = _global_module(channel)
        self.conv_layer = _BasicConv2d(channel * 2, channel, 3, padding=1)
        self.dropout = nn.Dropout2d(0.3)

    def forward(self, x, x_boun_atten):
        out1 = self.conv_layer(torch.cat((x, x_boun_atten), dim=1))
        out1 = self.dropout(out1)
        out2 = self.global_att(out1)
        out3 = out1.mul(out2)
        return x + out3


class _GateFusion(nn.Module):
    def __init__(self, in_planes):
        super().__init__()
        self.gate_1 = nn.Conv2d(in_planes * 2, 1, kernel_size=1, bias=True)
        self.gate_2 = nn.Conv2d(in_planes * 2, 1, kernel_size=1, bias=True)
        self.softmax = nn.Softmax(dim=1)
        self.bn = nn.BatchNorm2d(in_planes)

    def forward(self, x1, x2):
        cat_fea = torch.cat([x1, x2], dim=1)
        att_vec_1 = self.gate_1(cat_fea)
        att_vec_2 = self.gate_2(cat_fea)
        att_vec_cat = torch.cat([att_vec_1, att_vec_2], dim=1)
        att_vec_soft = self.softmax(att_vec_cat)
        att_soft_1, att_soft_2 = att_vec_soft[:, 0:1, :, :], att_vec_soft[:, 1:2, :, :]
        x_fusion = x1 * att_soft_1 + x2 * att_soft_2
        return self.bn(x_fusion)


class _CBAMLayer(nn.Module):
    def __init__(self, channel, reduction=16, spatial_kernel=7):
        super().__init__()
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.mlp = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, bias=False),
        )
        self.conv = nn.Conv2d(2, 1, kernel_size=spatial_kernel,
                              padding=spatial_kernel // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_out = self.mlp(self.max_pool(x))
        avg_out = self.mlp(self.avg_pool(x))
        channel_out = self.sigmoid(max_out + avg_out)
        x = channel_out * x
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        avg_out = torch.mean(x, dim=1, keepdim=True)
        spatial_out = self.sigmoid(self.conv(torch.cat([max_out, avg_out], dim=1)))
        x = spatial_out * x
        return x


class _CFM(nn.Module):
    def __init__(self, inc):
        super().__init__()
        self.cbam = _CBAMLayer(2 * inc)
        self.use_cbam = True
        self.cbr = nn.Sequential(
            nn.Conv2d(inc, inc, 3, 1, 1, padding_mode='reflect', bias=False),
            nn.BatchNorm2d(inc),
            nn.LeakyReLU(inplace=True),
            nn.Dropout2d(0.3),
        )
        self.cc = nn.Conv2d(2 * inc, 2 * inc, 3, 1, 1, padding_mode='reflect', bias=False)
        self.bn_cc = nn.BatchNorm2d(2 * inc)

    def forward(self, fs, fu):
        fbs = self.cbr(fs)
        fbs = torch.add(fbs, fs)
        fbs = torch.concat((fbs, fu), dim=1)
        if self.use_cbam:
            fbs = self.cbam(fbs)
        fbs = torch.sigmoid(self.bn_cc(self.cc(fbs)))
        fbu = self.cbr(fu)
        fbu = torch.add(fbu, fu)
        fbu = torch.concat((fbu, fs), dim=1)
        if self.use_cbam:
            fbu = self.cbam(fbu)
        fbu = torch.sigmoid(self.bn_cc(self.cc(fbu)))
        return fbs, fbu


class _CFF(nn.Module):
    def __init__(self, in_channel1, in_channel2, out_channel):
        super().__init__()
        act_fn = nn.LeakyReLU(inplace=True)
        self.layer0 = _BasicConv2d(in_channel1, out_channel // 2, 1)
        self.layer1 = _BasicConv2d(in_channel2, out_channel // 2, 1)
        self.layer3_1 = nn.Sequential(
            nn.Conv2d(out_channel, out_channel // 2, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channel // 2), nn.Dropout2d(0.3), act_fn,
        )
        self.layer3_2 = nn.Sequential(
            nn.Conv2d(out_channel, out_channel // 2, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channel // 2), nn.Dropout2d(0.3), act_fn,
        )
        self.layer5_1 = nn.Sequential(
            nn.Conv2d(out_channel, out_channel // 2, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(out_channel // 2), nn.Dropout2d(0.3), act_fn,
        )
        self.layer5_2 = nn.Sequential(
            nn.Conv2d(out_channel, out_channel // 2, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(out_channel // 2), nn.Dropout2d(0.3), act_fn,
        )
        self.layer_out = nn.Sequential(
            nn.Conv2d(out_channel // 2, out_channel, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channel), act_fn,
        )
        self.cfm1 = _CFM(out_channel // 2)
        self.cfm2 = _CFM(out_channel // 2)
        self.channel_att = _ChannelAttention(out_channel)

    def forward(self, x0, x1):
        x0_1 = self.layer0(x0)
        x1_1 = self.layer1(x1)
        x_3_1, x_5_1 = self.cfm1(x0_1, x1_1)
        x_3_1 = self.layer3_1(x_3_1)
        x_5_1 = self.layer5_1(x_5_1)
        x_3_1, x_5_1 = self.cfm2(x_3_1, x_5_1)
        x_3_2 = self.layer3_2(x_3_1)
        x_5_2 = self.layer5_2(x_5_1)
        out = self.layer_out(x0_1 + x1_1 + torch.mul(x_3_2, x_5_2))
        return out * self.channel_att(out)


class _Bottle2neck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None,
                 baseWidth=26, scale=4, stype='normal'):
        super().__init__()
        width = int(math.floor(planes * (baseWidth / 64.0)))
        self.conv1 = nn.Conv2d(inplanes, width * scale, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(width * scale)
        self.nums = 1 if scale == 1 else scale - 1
        if stype == 'stage':
            self.pool = nn.AvgPool2d(kernel_size=3, stride=stride, padding=1)
        convs, bns = [], []
        for _ in range(self.nums):
            convs.append(nn.Conv2d(width, width, kernel_size=3, stride=stride, padding=1, bias=False))
            bns.append(nn.BatchNorm2d(width))
        self.convs = nn.ModuleList(convs)
        self.bns = nn.ModuleList(bns)
        self.conv3 = nn.Conv2d(width * scale, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.LeakyReLU(inplace=True)
        self.downsample = downsample
        self.stype = stype
        self.scale = scale
        self.width = width

    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        spx = torch.split(out, self.width, 1)
        for i in range(self.nums):
            sp = spx[i] if (i == 0 or self.stype == 'stage') else sp + spx[i]
            sp = self.relu(self.bns[i](self.convs[i](sp)))
            out = sp if i == 0 else torch.cat((out, sp), 1)
        if self.scale != 1 and self.stype == 'normal':
            out = torch.cat((out, spx[self.nums]), 1)
        elif self.scale != 1 and self.stype == 'stage':
            out = torch.cat((out, self.pool(spx[self.nums])), 1)
        out = self.bn3(self.conv3(out))
        if self.downsample is not None:
            residual = self.downsample(x)
        return self.relu(out + residual)


class _Res2Net_Ours(nn.Module):
    def __init__(self, resinc, block, layers, baseWidth=26, scale=4, num_classes=1000):
        super().__init__()
        self.inplanes = 64
        self.baseWidth = baseWidth
        self.scale = scale
        self.conv1 = nn.Sequential(
            nn.Conv2d(resinc, 32, 3, 2, 1, bias=False),
            nn.BatchNorm2d(32), nn.LeakyReLU(inplace=True), nn.Dropout2d(0.3),
            nn.Conv2d(32, 32, 3, 1, 1, bias=False),
            nn.BatchNorm2d(32), nn.LeakyReLU(inplace=True), nn.Dropout2d(0.3),
            nn.Conv2d(32, 64, 3, 1, 1, bias=False),
        )
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.LeakyReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.AvgPool2d(kernel_size=stride, stride=stride, ceil_mode=True, count_include_pad=False),
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )
        layers = [block(self.inplanes, planes, stride, downsample=downsample,
                        stype='stage', baseWidth=self.baseWidth, scale=self.scale)]
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, baseWidth=self.baseWidth, scale=self.scale))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x0 = self.maxpool(x)
        x1 = self.layer1(x0)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)
        return x0, x1, x2, x3, x4


# ============================================================
# JJNet 原始代码
# ============================================================


# ============================================================================
# 模块 1: 切片自适应阈值预测器
# ============================================================================

class AdaptiveThresholdPredictor(nn.Module):
    """切片自适应阈值预测器

    结合全局基线参数与 per-sample 自适应偏移, 为每个样本预测专属阈值。
    基线参数提供先验知识(grid search结果), 自适应偏移捕捉个体差异。

    Args:
        init_lower: 阈值下限初始值 (来自 grid search)
        init_upper: 阈值上限初始值 (来自 grid search)
    """
    def __init__(self, init_lower=17.5, init_upper=22.0):
        super().__init__()
        # 全局基线参数 (可学习)
        self.base_lower = nn.Parameter(torch.tensor(init_lower, dtype=torch.float32))
        self.base_offset = nn.Parameter(torch.tensor(init_upper - init_lower, dtype=torch.float32))

        # 自适应偏移网络 (轻量级 MLP)
        self.adaptive_net = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(1, 4),
            nn.LeakyReLU(inplace=True),
            nn.Linear(4, 2),  # [delta_lower, delta_upper]
        )
        # 零初始化, 初始时退化为基线
        self.adaptive_net[-1].weight.data.zero_()
        self.adaptive_net[-1].bias.data.zero_()

    def forward(self, x):
        """预测 per-sample 阈值

        Args:
            x: 输入图像 (B, 1, H, W)
        Returns:
            lower: 阈值下限 (B, 1, 1, 1) — per-sample
            upper: 阈值上限 (B, 1, 1, 1)
        """
        B = x.shape[0]

        # 全局基线
        base_lower = F.softplus(self.base_lower)
        base_upper = base_lower + F.softplus(self.base_offset) + 1e-4

        # 自适应偏移
        delta = self.adaptive_net(x)  # (B, 2)

        # 应用到每个样本
        lower = base_lower + delta[:, 0:1].view(B, 1, 1, 1)
        upper = base_upper + delta[:, 1:2].view(B, 1, 1, 1)

        # 约束: lower > 0, upper > lower
        lower = F.softplus(lower) + 1.0
        upper = torch.max(upper, lower + 1.0)

        return lower, upper


# ============================================================================
# 模块 2: 可微阈值分割 + 多尺度先验图生成
# ============================================================================

class PriorMapGenerator(nn.Module):
    """可学习病理先验概率图生成器

    在 LearnablePriorMap 基础上, 支持 per-sample 阈值输入。
    阈值分割 → 多尺度卷积融合 → 后处理 → 先验概率图

    Args:
        radii: 多尺度卷积半径列表
    """
    def __init__(self, radii=None):
        super().__init__()
        if radii is None:
            radii = range(1, 4)
        self.radii = list(radii)

        # 可学习的多尺度卷积核权重
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

        # 2. 多尺度卷积
        multi_scale_maps = []
        for i, r in enumerate(self.radii):
            k = 2 * r + 1
            w = F.softplus(self.scale_weights[i])
            kernel = torch.ones(1, 1, k, k, device=x.device) * w
            conv_out = F.conv2d(gate_map, kernel, padding=r)
            multi_scale_maps.append(conv_out)

        # 3. 融合 + 后处理
        stacked = torch.cat(multi_scale_maps, dim=1)
        prior_map = self.post_conv(stacked)

        return prior_map, gate_map


# ============================================================================
# 模块 3: 可变形对侧对齐
# ============================================================================

class DeformableContraAlignment(nn.Module):
    """可变形对侧对齐模块

    预测偏移场(offset field)并将对侧特征 warp 到与患侧对齐。
    解决大脑生理性不对称、占位效应、扫描倾斜等问题。

    Args:
        channel: 输入特征通道数
        max_offset: 最大归一化偏移量 (限制形变幅度, 防止过度扭曲)
    """
    def __init__(self, channel, max_offset=0.25):
        super().__init__()
        self.max_offset = max_offset

        # 偏移场预测网络: 拼接 ipsi + contra → (dx, dy)
        self.offset_net = nn.Sequential(
            nn.Conv2d(channel * 2, channel // 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(channel // 2),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(channel // 2, channel // 4, kernel_size=3, padding=1),
            nn.BatchNorm2d(channel // 4),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(channel // 4, 2, kernel_size=3, padding=1),
        )
        # 零初始化偏移
        self.offset_net[-1].weight.data.zero_()
        self.offset_net[-1].bias.data.zero_()

    def forward(self, ipsi_feat, contra_feat):
        """将对侧特征 warp 到与患侧对齐

        Args:
            ipsi_feat: 患侧特征 (B, C, H, W)
            contra_feat: 对侧特征 (B, C, H, W) — 已翻转/未翻转均可

        Returns:
            aligned: 对齐后的对侧特征 (B, C, H, W)
            offset: 偏移场 (B, 2, H, W) — 可视化/分析用
        """
        B, C, H, W = ipsi_feat.shape

        # 预测偏移场
        offset = self.offset_net(torch.cat([ipsi_feat, contra_feat], dim=1))
        offset = torch.tanh(offset) * self.max_offset

        # 生成标准网格
        grid_y, grid_x = torch.meshgrid(
            torch.linspace(-1, 1, H, device=ipsi_feat.device),
            torch.linspace(-1, 1, W, device=ipsi_feat.device),
            indexing='ij'
        )
        grid = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0).repeat(B, 1, 1, 1)

        # 偏移归一化到网格坐标 [-1, 1]
        offset_h = offset[:, 0:1] * (2.0 / H)
        offset_w = offset[:, 1:2] * (2.0 / W)
        offset_norm = torch.cat([offset_w, offset_h], dim=1).permute(0, 2, 3, 1)

        # Warp
        sample_grid = grid + offset_norm
        aligned = F.grid_sample(
            contra_feat, sample_grid,
            mode='bilinear', align_corners=False
        )

        return aligned, offset


# ============================================================================
# 模块 4: 带可变形对齐的对侧融合
# ============================================================================

class ContralateralFusionWithAlignment(nn.Module):
    """带可变形对齐的对侧融合模块

    对侧特征 warp 对齐 → 差分特征提取 → 通道注意力 → 门控融合

    Args:
        channel: 输入特征通道数
    """
    def __init__(self, channel):
        super().__init__()
        self.alignment = DeformableContraAlignment(channel)

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
        # 1. 可变形对齐
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
# 模块 5: 先验精炼模块
# ============================================================================

class PriorRefinement(nn.Module):
    """先验精炼模块

    利用对侧对照产生的不对称信息(5个尺度的diff_map)来修正和精炼先验图。
    不对称性强的区域 → 增强先验置信度
    不对称性弱的区域(对称假阳性) → 抑制先验置信度

    输入: prior + 5个上采样后的 diff_map (共6通道)
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

        # 将所有 diff_map 上采样到统一尺寸
        resampled = []
        for d in diff_maps:
            if d.shape[2:] != (H, W):
                d = F.interpolate(d, (H, W), mode='bilinear', align_corners=True)
            resampled.append(d)

        # 拼接: prior + 5个diff_map
        x = torch.cat([prior] + resampled, dim=1)
        residual = self.refine_net(x)

        # 残差连接: 保留原有先验结构, 仅修正
        refined = prior + residual
        refined = torch.clamp(refined, 0, 1)

        return refined


# ============================================================================
# 模块 6: 损失函数
# ============================================================================

class JJNetLoss(nn.Module):
    """JJNet 综合损失函数

    包含:
        1. 分割主损失 (Dice + BCE) — 深度监督
        2. 先验监督损失 — 引导先验图贴近标签的软版本
        3. 不对称稀疏损失 — 非病灶区域不对称值应为0
        4. 偏移场正则化 — 防止对齐过度形变

    Args:
        deep_supervision_weights: 深度监督各输出的权重 [edge, sal1, sal2, sal3]
        prior_weight: 先验监督损失权重
        asym_weight: 不对称稀疏损失权重
        offset_weight: 偏移场正则化权重
    """
    def __init__(self, deep_supervision_weights=None,
                 prior_weight=0.1, asym_weight=0.05, offset_weight=0.001):
        super().__init__()
        if deep_supervision_weights is None:
            deep_supervision_weights = [0.1, 0.2, 0.2, 0.5]
        self.ds_weights = deep_supervision_weights
        self.prior_weight = prior_weight
        self.asym_weight = asym_weight
        self.offset_weight = offset_weight

    def dice_loss(self, pred, target, smooth=1.0):
        """计算 Dice Loss"""
        pred = torch.sigmoid(pred)
        pred_flat = pred.contiguous().view(-1)
        target_flat = target.contiguous().view(-1)
        intersection = (pred_flat * target_flat).sum()
        dice = (2.0 * intersection + smooth) / (pred_flat.sum() + target_flat.sum() + smooth)
        return 1.0 - dice

    def bce_loss(self, pred, target):
        """计算 BCE Loss"""
        return F.binary_cross_entropy_with_logits(pred, target)

    def forward(self, outputs, labels, prior_maps=None, diff_maps=None,
                offsets=None):
        """计算总损失

        Args:
            outputs: 深度监督输出列表 [edge_out3, sal_out1, sal_out2, sal_out3]
                     或单个 sal_out3
            labels: 真实标签 (B, 1, H, W)
            prior_maps: 各迭代轮次的先验图列表 [prior_0, prior_1, ...]
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
                # 使用高斯模糊对标签做软化
                label_soft = F.avg_pool2d(labels, kernel_size=5, stride=1, padding=2)
                prior_loss = F.mse_loss(prior, label_soft)
                total_loss += self.prior_weight * prior_loss

        # ---- 3. 不对称稀疏损失 ----
        if diff_maps is not None:
            # 在非标签区域, 不对称值应接近0 (L1稀疏)
            for diff in diff_maps:
                diff_resized = F.interpolate(
                    diff, size=labels.shape[2:], mode='bilinear', align_corners=True
                )
                # 仅在非病灶区域计算 (1 - label)
                non_lesion = 1.0 - labels
                asym_loss = torch.mean(diff_resized * non_lesion)
                total_loss += self.asym_weight * asym_loss

        # ---- 4. 偏移场正则化 ----
        if offsets is not None:
            for offset in offsets:
                offset_loss = torch.mean(offset ** 2)
                total_loss += self.offset_weight * offset_loss

        return total_loss


# ============================================================================
# 模块 7: JJNet 主网络
# ============================================================================

class JJNet(nn.Module):
    """JJNet — 可学习病理先验 + 可变形对侧半脑对照的卒中梗死分割网络

    Args:
        channel: 解码器基础通道数 (默认64, 与原始CCFANet一致)
        prior_init_lower: 先验阈值下限初始值 (来自grid search)
        prior_init_upper: 先验阈值上限初始值
        prior_radii: 多尺度卷积半径列表
        num_refine_iters: 先验精炼迭代次数 (默认1)
        deep_supervision: 是否启用深度监督
    """
    def __init__(self, channel=64,
                 prior_init_lower=17.5, prior_init_upper=22.0,
                 prior_radii=None, num_refine_iters=1,
                 deep_supervision=True):
        super(JJNet, self).__init__()

        self.deep_supervision = deep_supervision
        self.num_refine_iters = num_refine_iters

        # 1. 切片自适应阈值预测器
        self.threshold_predictor = AdaptiveThresholdPredictor(
            init_lower=prior_init_lower,
            init_upper=prior_init_upper,
        )

        # 2. 可学习先验概率图生成器
        self.prior_generator = PriorMapGenerator(radii=prior_radii)

        # 3. 共享权重双路径编码器 (输入2通道: raw + prior)
        self.encoder = _Res2Net_Ours(
            2, _Bottle2neck, [3, 4, 6, 3],
            baseWidth=26, scale=4
        )

        # 4. 5层可变形对侧融合
        self.contra_fusions = nn.ModuleList([
            ContralateralFusionWithAlignment(64),    # x0
            ContralateralFusionWithAlignment(256),   # x1
            ContralateralFusionWithAlignment(512),   # x2
            ContralateralFusionWithAlignment(1024),  # x3
            ContralateralFusionWithAlignment(2048),  # x4
        ])

        # 5. 先验精炼模块
        self.prior_refine = PriorRefinement()

        # ---- 6. 解码器 (复用原始CCFANet结构) ---- #
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
        # 构建双路径输入
        ipsi_in = torch.cat([x, prior], dim=1)                          # (B, 2, H, W)
        contra_in = torch.cat([torch.flip(x, dims=[-1]),
                                torch.flip(prior, dims=[-1])], dim=1)    # (B, 2, H, W)

        # 共享权重双路径编码
        ipsi_x0, ipsi_x1, ipsi_x2, ipsi_x3, ipsi_x4 = self.encoder(ipsi_in)
        contra_x0, contra_x1, contra_x2, contra_x3, contra_x4 = self.encoder(contra_in)

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
            edge_out3: 边缘预测 (B, 3, H, W)
            sal_out1: 显著性图1
            sal_out2: 显著性图2
            sal_out3: 最终融合显著性图
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
            edge_out3: 边缘预测 (B, 3, H, W)
            sal_out1: 显著性图1 (B, 3, H, W)
            sal_out2: 显著性图2 (B, 3, H, W)
            sal_out3: 最终融合显著性图 (B, 3, H, W)
            prior_maps: 各迭代轮次的先验图列表 — 可解释性
            diff_maps: 最终轮次各层不对称热图 — 可解释性
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
    print("JJNet 网络测试")
    print("=" * 60)

    # 测试 AdaptiveThresholdPredictor
    print("\n[1/6] 测试 AdaptiveThresholdPredictor ...")
    atp = AdaptiveThresholdPredictor()
    x = torch.randn(8, 1, 256, 256)
    lower, upper = atp(x)
    print(f"     lower: {lower[0,0,0,0].item():.2f} (per-sample: {lower.shape})")
    print(f"     upper: {upper[0,0,0,0].item():.2f} (per-sample: {upper.shape})")

    # 测试 PriorMapGenerator
    print("\n[2/6] 测试 PriorMapGenerator ...")
    pmg = PriorMapGenerator()
    prior_map, gate_map = pmg(x, lower, upper)
    print(f"     prior_map: {prior_map.shape} [{prior_map.min().item():.3f}, {prior_map.max().item():.3f}]")
    print(f"     gate_map:  {gate_map.shape}")

    # 测试 DeformableContraAlignment
    print("\n[3/6] 测试 DeformableContraAlignment ...")
    dca = DeformableContraAlignment(64)
    ipsi = torch.randn(8, 64, 64, 64)
    contra = torch.randn(8, 64, 64, 64)
    aligned, offset = dca(ipsi, contra)
    print(f"     aligned: {aligned.shape}")
    print(f"     offset:  {offset.shape} [{offset.min().item():.3f}, {offset.max().item():.3f}]")

    # 测试 ContralateralFusionWithAlignment
    print("\n[4/6] 测试 ContralateralFusionWithAlignment ...")
    cf = ContralateralFusionWithAlignment(64)
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

    # 测试 JJNet 主网络
    print("\n[6/6] 测试 JJNet 主网络 ...")
    model = JJNet(channel=64, num_refine_iters=1)
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
    print("\n[*] 测试 JJNetLoss ...")
    criterion = JJNetLoss()
    loss = criterion(
        outputs=[edge, s1, s2, s3],
        labels=torch.randint(0, 2, (8, 1, 256, 256)).float(),
        prior_maps=prior_maps,
        diff_maps=diff_maps,
        offsets=[off],
    )
    print(f"     总损失: {loss.item():.4f}")

    print("\n" + "=" * 60)
    print("所有测试通过！JJNet 构建成功。")
    print("=" * 60)