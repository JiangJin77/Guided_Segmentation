"""
CCFANet增强版 —— 集成可学习病理先验概率图 + 对侧半脑对照融合

新增功能：
    1. LearnablePriorMap替代固定阈值预处理，生成可学习的病理先验概率图
    2. 将先验概率图与原始图像拼接作为网络输入
    3. 共享权重双路径编码器（患侧 + 对侧）实现半脑对照
    4. 多尺度对侧特征融合（ContralateralFusion）
"""
import torch
import torch.nn as nn

from .CCFANet import (Res2Net_Ours, Bottle2neck,
                      ChannelAttention, BAM,
                      CFF, GateFusion)
from .learnable_prior import LearnablePriorMap, ThresholdSegmentation


class ContralateralFusion(nn.Module):
    """对侧半脑对照融合模块
    融合患侧(ipsi)与对侧(contra)特征，突出不对称性（即病理区域）。
    采用减法与通道注意力结合的方式。
    """
    def __init__(self, channel):
        super(ContralateralFusion, self).__init__()

        self.diff_conv = nn.Sequential(
            nn.Conv2d(channel, channel, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channel),
            nn.LeakyReLU(inplace=True),
        )
        self.att = ChannelAttention(channel)
        self.gate = nn.Sequential(
            nn.Conv2d(channel * 2, 1, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, ipsi, contra):
        """返回融合后的特征

        Args:
            ipsi: 患侧特征 (B, C, H, W)
            contra: 对侧特征 (B, C, H, W)

        Returns:
            fused: 融合特征 (B, C, H, W)
            diff_map: 不对称性热图 (B, 1, H, W)
        """
        diff = ipsi - contra
        diff_feat = self.diff_conv(diff)
        diff_feat = diff_feat * self.att(diff_feat)

        gate = self.gate(torch.cat([ipsi, contra], dim=1))
        fused = ipsi + gate * diff_feat

        diff_map = torch.mean(torch.abs(diff), dim=1, keepdim=True)

        return fused, diff_map


class CCFANet_Enhanced(nn.Module):
    """CCFANet增强版，集成可学习病理先验概率图 + 对侧半脑对照

    Args:
        channel: 基础通道数（默认64，与原始CCFANet一致）
        prior_init_lower: 先验阈值下限初始值（默认17.5）
        prior_init_upper: 先验阈值上限初始值（默认22.0）
        prior_radii: 先验多尺度卷积半径列表（默认range(1,4)）
        deep_supervision: 是否启用深度监督
    """
    def __init__(self, channel=64,
                 prior_init_lower=17.5, prior_init_upper=22.0,
                 prior_radii=None, deep_supervision=True):
        super(CCFANet_Enhanced, self).__init__()

        self.deep_supervision = deep_supervision

        # 1. 显式的阈值分割模块 —— 直接对应get_threshold.py的网格搜索结果
        self.threshold_seg = ThresholdSegmentation(
            init_lower=prior_init_lower,
            init_upper=prior_init_upper,
        )

        # 2. 可学习的病理先验概率图（包含阈值分割 + 多尺度卷积）
        self.prior_map = LearnablePriorMap(
            init_lower=prior_init_lower,
            init_upper=prior_init_upper,
            radii=prior_radii,
        )

        # 患侧编码器（输入2通道：原始图像 + 先验概率图）
        self.ipsi_encoder = Res2Net_Ours(2, Bottle2neck, [3, 4, 6, 3], baseWidth=26, scale=4)

        # 对侧编码器（共享权重，只编码原始图像，水平翻转）
        self.contra_encoder = self.ipsi_encoder

        # 对侧融合模块（每个特征尺度）
        self.contra_fusion0 = ContralateralFusion(64)
        self.contra_fusion1 = ContralateralFusion(256)
        self.contra_fusion2 = ContralateralFusion(512)
        self.contra_fusion3 = ContralateralFusion(1024)
        self.contra_fusion4 = ContralateralFusion(2048)

        # ---- 以下使用原始CCFANet的解码器结构 ---- #
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

        self.low_fusion = GateFusion(channel)
        self.high_fusion1 = CFF(256, 512, channel)
        self.high_fusion2 = CFF(1024, 2048, channel)

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
        self.layer_edge3 = nn.Sequential(nn.Conv2d(64, 3, kernel_size=1))

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
        self.layer_hig31 = nn.Sequential(nn.Conv2d(64, 3, kernel_size=1))

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
        self.layer_hig32 = nn.Sequential(nn.Conv2d(64, 3, kernel_size=1))

        self.layer_fil = nn.Sequential(nn.Conv2d(64, 3, kernel_size=1))

        self.atten_edge_0 = ChannelAttention(channel)
        self.atten_edge_1 = ChannelAttention(channel)
        self.atten_edge_2 = ChannelAttention(channel)
        self.atten_edge_ori = ChannelAttention(channel)

        self.cat_01 = BAM(channel)
        self.cat_11 = BAM(channel)
        self.cat_21 = BAM(channel)
        self.cat_31 = BAM(channel)

        self.cat_02 = BAM(channel)
        self.cat_12 = BAM(channel)
        self.cat_22 = BAM(channel)
        self.cat_32 = BAM(channel)

        self.up_2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.up_4 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
        self.up_8 = nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True)

    def forward(self, xx):
        """前向传播

        Args:
            xx: 输入图像 (B, 1, H, W)

        Returns:
            edge_out3: 边缘预测 (B, 3, H, W)
            sal_out1: 显著性图1 (B, 3, H, W)
            sal_out2: 显著性图2 (B, 3, H, W)
            sal_out3: 最终融合显著性图 (B, 3, H, W)
            gate_map: 阈值门控掩码 (B, 1, H, W) —— 对应get_threshold.py的分割结果
            lower: 当前阈值下限 (scalar)
            upper: 当前阈值上限 (scalar)
            prior_map: 先验概率图 (B, 1, H, W) —— 多尺度卷积融合后的精细先验
            diff_maps: 各层不对称性热图列表 —— 用于可解释性
        """
        # 1. 显式阈值分割 —— 直接对应get_threshold.py中(data >= lower) & (data <= upper)
        threshold_mask, lower, upper = self.threshold_seg(xx)

        # 2. 生成可学习病理先验概率图（基于阈值分割 + 多尺度卷积）
        prior_map, gate_map, _, _ = self.prior_map(xx)

        # 2. 构建患侧输入 (B, 2, H, W)：[原始图像, 先验概率图]
        ipsi_input = torch.cat([xx, prior_map], dim=1)

        # 3. 构建对侧输入：水平翻转原始图像, 同样拼接先验
        #    注意：先验图跟着翻转，保持空间对应关系
        contra_input = torch.cat([torch.flip(xx, dims=[-1]),
                                  torch.flip(prior_map, dims=[-1])], dim=1)

        # 4. 双路径编码（共享权重）
        ipsi_x0, ipsi_x1, ipsi_x2, ipsi_x3, ipsi_x4 = self.ipsi_encoder(ipsi_input)
        contra_x0, contra_x1, contra_x2, contra_x3, contra_x4 = self.contra_encoder(contra_input)

        # 5. 对侧融合（融合后特征，并将对侧特征翻转回原始方向）
        f_x0, diff0 = self.contra_fusion0(ipsi_x0, torch.flip(contra_x0, dims=[-1]))
        f_x1, diff1 = self.contra_fusion1(ipsi_x1, torch.flip(contra_x1, dims=[-1]))
        f_x2, diff2 = self.contra_fusion2(ipsi_x2, torch.flip(contra_x2, dims=[-1]))
        f_x3, diff3 = self.contra_fusion3(ipsi_x3, torch.flip(contra_x3, dims=[-1]))
        f_x4, diff4 = self.contra_fusion4(ipsi_x4, torch.flip(contra_x4, dims=[-1]))

        diff_maps = [diff0, diff1, diff2, diff3, diff4]

        # 6. 以下复用原始CCFANet的解码器结构
        x0_1 = self.layer0(f_x0)
        x1_1 = self.layer1(f_x1)
        low_x = self.low_fusion(x0_1, x1_1)

        edge_out0 = self.layer_edge0(self.up_2(low_x))
        edge_out1 = self.layer_edge1(self.up_2(edge_out0))
        edge_out2 = self.layer_edge2(self.up_2(edge_out1))
        edge_out3 = self.layer_edge3(edge_out2)

        atten_edge_ori = self.atten_edge_ori(low_x)
        atten_edge_0 = self.atten_edge_0(edge_out0)
        atten_edge_1 = self.atten_edge_1(edge_out1)
        atten_edge_2 = self.atten_edge_2(edge_out2)

        high_x01 = self.high_fusion1(self.downSample(f_x1), f_x2)
        high_x02 = self.high_fusion2(self.up_2(f_x3), self.up_4(f_x4))

        # high path 1
        cat_out_01 = self.cat_01(high_x01, low_x.mul(atten_edge_ori))
        hig_out01 = self.layer_hig01(self.up_2(cat_out_01))

        cat_out11 = self.cat_11(hig_out01, edge_out0.mul(atten_edge_0))
        hig_out11 = self.layer_hig11(self.up_2(cat_out11))

        cat_out21 = self.cat_21(hig_out11, edge_out1.mul(atten_edge_1))
        hig_out21 = self.layer_hig21(self.up_2(cat_out21))

        cat_out31 = self.cat_31(hig_out21, edge_out2.mul(atten_edge_2))
        sal_out1 = self.layer_hig31(cat_out31)

        # high path 2
        cat_out_02 = self.cat_02(high_x02, low_x.mul(atten_edge_ori))
        hig_out02 = self.layer_hig02(self.up_2(cat_out_02))

        cat_out12 = self.cat_12(hig_out02, edge_out0.mul(atten_edge_0))
        hig_out12 = self.layer_hig12(self.up_2(cat_out12))

        cat_out22 = self.cat_22(hig_out12, edge_out1.mul(atten_edge_1))
        hig_out22 = self.layer_hig22(self.up_2(cat_out22))

        cat_out32 = self.cat_32(hig_out22, edge_out2.mul(atten_edge_2))
        sal_out2 = self.layer_hig32(cat_out32)

        sal_out3 = self.layer_fil(cat_out31 + cat_out32)

        if self.deep_supervision:
            return edge_out3, sal_out1, sal_out2, sal_out3, \
                   threshold_mask, gate_map, lower, upper, prior_map, diff_maps
        else:
            return sal_out3


if __name__ == '__main__':
    model = CCFANet_Enhanced(channel=64)
    input_images = torch.randn(8, 1, 256, 256)
    output = model(input_images)
    output = model(input_images)
    print("深度监督模式下返回数量:", len(output))
    (edge, s1, s2, s3,
     threshold_mask, gate, lower, upper, prior, diffs) = output
    print(f"边缘预测:         {edge.shape}")
    print(f"显著性图1:        {s1.shape}")
    print(f"显著性图2:        {s2.shape}")
    print(f"最终融合图:       {s3.shape}")
    print(f"阈值掩码:         {threshold_mask.shape}  (直接对应get_threshold.py)")
    print(f"门控掩码:         {gate.shape}")
    print(f"阈值lower:        {lower.item():.2f}  upper: {upper.item():.2f}")
    print(f"先验概率图:       {prior.shape}")
    print(f"不对称热图数:     {len(diffs)}, 每层形状: {[d.shape for d in diffs]}")
    print(f"总参数量:         {sum(p.numel() for p in model.parameters()):,}")