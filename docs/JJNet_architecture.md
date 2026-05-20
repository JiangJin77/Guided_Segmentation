# JJNet 网络架构图

```mermaid
%%{init: {'theme': 'neutral', 'flowchart': {'curve': 'basis'}}}%%
flowchart TB
    %% ===== 输入 =====
    Input(["Input Image (B, 1, H, W)"]):::input

    %% ===== Phase 1: 阈值 + 粗先验 =====
    subgraph Phase1["Phase 1: 切片自适应阈值 + 粗先验"]
        direction TB
        ATP["AdaptiveThresholdPredictor<br/><small>全局基线 + per-sample MLP偏移</small>"]:::phase1
        PMG["PriorMapGenerator<br/><small>可微阈值分割 → 多尺度卷积融合 → Sigmoid</small>"]:::phase1
        GateMap(["gate_map (B,1,H,W)"]):::tensor
        Prior0(["粗先验 prior₀ (B,1,H,W)"]):::tensor
        ATP -->|"lower, upper"| PMG
        PMG --> Prior0
        PMG --> GateMap
    end

    %% ===== Phase 2: 先验精炼循环 =====
    subgraph Phase2["Phase 2: 先验精炼循环 (num_refine_iters 次)"]
        direction TB

        subgraph Encoder["共享权重双路径编码器 (Res2Net)"]
            direction LR
            IpsiEnc["患侧分支<br/><small>cat(x, prior) → 5层特征</small>"]:::encoder
            ContraEnc["对侧分支<br/><small>cat(flip(x), flip(prior)) → 5层特征</small>"]:::encoder
            SharedWeight["<small>权重共享</small>"]:::note
            IpsiEnc --- SharedWeight --- ContraEnc
        end

        subgraph Align["5x 可变形对侧对齐 + 融合"]
            direction TB
            DCA1["DeformableContraAlignment<br/><small>偏移场预测 → grid_sample warp</small>"]:::align
            CFF1["ContralateralFusionWithAlignment<br/><small>差分 → 通道注意力 → 门控融合</small>"]:::align
            CFF1 --> DCA1
            DCA1 --> Fused_i(["融合特征 f_xᵢ (B,C_i,H_i,W_i)"]):::tensor
            DCA1 --> Diff_i(["不对称热图 diff_i (B,1,H_i,W_i)"]):::tensor
            DCA1 --> Off_i(["偏移场 offset_i (B,2,H_i,W_i)"]):::tensor
        end

        subgraph Refine["PriorRefinement"]
            PR["<b>PriorRefinement</b><br/><small>cat(prior, diff₀~diff₄)<br/>→ 4层卷积 → 残差修正</small>"]:::refine
            PriorNew(["精炼先验 prior_{t+1}"]):::tensor
            PR --> PriorNew
        end

        Encoder --> Align
        Align -->|"diff_maps"| Refine
        Refine -.->|"循环回到编码器"| Encoder
    end

    %% ===== Phase 3: 解码器 =====
    subgraph Phase3["Phase 3: CCFANet 解码器"]
        direction TB

        subgraph Decoder["CCFANet Decoder"]
            direction TB
            LowFusion["GateFusion<br/><small>低层融合 f_x₀ + f_x₁</small>"]:::decoder
            HighFusion1["CFF<br/><small>高层融合1 f_x₁ + f_x₂</small>"]:::decoder
            HighFusion2["CFF<br/><small>高层融合2 f_x₃ + f_x₄</small>"]:::decoder

            EdgePath["边缘路径<br/><small>4× 卷积 → upsampling</small>"]:::decoder
            AttnEdge["通道注意力<br/>(×4)"]:::decoder

            HighPath1["High Path 1<br/><small>逐步融合 + BAM ×4</small>"]:::decoder
            HighPath2["High Path 2<br/><small>逐步融合 + BAM ×4</small>"]:::decoder

            FinalFusion["layer_fil<br/><small>cat(H1, H2) → 1×1 conv</small>"]:::decoder

            LowFusion --> EdgePath
            EdgePath --> AttnEdge
            AttnEdge --> HighPath1
            AttnEdge --> HighPath2
            HighFusion1 --> HighPath1
            HighFusion2 --> HighPath2
            HighPath1 --> FinalFusion
            HighPath2 --> FinalFusion
        end

        EdgeOut(["edge_out (B,3,H,W)"]):::output
        Sal1(["sal_out₁ (B,3,H,W)"]):::output
        Sal2(["sal_out₂ (B,3,H,W)"]):::output
        Sal3(["sal_out₃ (B,3,H,W) —— 最终分割"]):::output

        EdgePath --> EdgeOut
        HighPath1 --> Sal1
        HighPath2 --> Sal2
        FinalFusion --> Sal3
    end

    %% ===== Loss =====
    subgraph Loss["损失函数 (JJNetLoss)"]
        direction TB
        DiceBCE["Dice + BCE<br/><small>深度监督(4路加权)</small>"]:::loss
        PriorLoss["先验监督 Loss<br/><small>MSE(prior, label_soft)</small>"]:::loss
        AsymLoss["不对称稀疏 Loss<br/><small>L1(非病灶区diff)</small>"]:::loss
        OffsetLoss["偏移正则化 Loss<br/><small>L2(offset)</small>"]:::loss
        TotalLoss["Total Loss"]:::loss
        DiceBCE --> TotalLoss
        PriorLoss --> TotalLoss
        AsymLoss --> TotalLoss
        OffsetLoss --> TotalLoss
    end

    %% ===== 主数据流 =====
    Input --> Phase1
    Phase1 -->|"prior₀"| Phase2
    Phase2 -->|"fused_feats + diff_maps"| Phase3
    Phase3 -->|"outputs"| Loss
    Phase2 -->|"diff_maps, offsets"| Loss
    Phase1 -->|"prior_maps"| Loss

    %% ===== 可解释性输出 =====
    Phase3 -.->|"edge, sal₁, sal₂, sal₃"| OutputGroup["输出"]
    Phase2 -.->|"diff_maps (5层不对称热图)"| OutputGroup
    Phase1 -.->|"gate_map, lower, upper"| OutputGroup

    %% ===== 样式 =====
    classDef input fill:#1a73e8,color:#fff,stroke:#fff,stroke-width:2px
    classDef phase1 fill:#e8f0fe,stroke:#1a73e8,stroke-width:1px
    classDef encoder fill:#fff3cd,stroke:#856404,stroke-width:1px
    classDef align fill:#d4edda,stroke:#155724,stroke-width:1px
    classDef refine fill:#f8d7da,stroke:#721c24,stroke-width:1px
    classDef decoder fill:#e8f4f8,stroke:#0c5460,stroke-width:1px
    classDef loss fill:#f3e5f5,stroke:#4a148c,stroke-width:1px
    classDef tensor fill:#fff,stroke:#6c757d,stroke-width:1px,stroke-dasharray:3 3
    classDef output fill:#d63384,color:#fff,stroke:#fff,stroke-width:2px
    classDef note fill:transparent,stroke:none,color:#666,font-style:italic
```

---

## 数据流简要说明

| 阶段 | 关键操作 | 输入 → 输出 |
|------|---------|-------------|
| **Phase 1** | `AdaptiveThresholdPredictor` → `PriorMapGenerator` | 图像 → per-sample 阈值 + 粗先验图(0~1概率图) |
| **Phase 2** | 编码器(共享权重) → 可变形对齐 → PriorRefinement (循环) | 粗先验 + 图像 → 精炼先验 + 5层融合特征 + 5层不对称热图 |
| **Phase 3** | GateFusion → CFF → 边缘路径 + 双High Path → BAM融合 | 5层特征 → edge_out + 3路显著性图 |
| **Loss** | Dice+BCE + MSE先验 + L1稀疏 + L2偏移正则 | 深度监督输出 + 中间产物 → 总损失 |

## 形状说明

- <span style="background:#1a73e8;color:#fff;padding:0 6px;border-radius:3px;">蓝色椭圆</span> — 输入
- <span style="background:#d63384;color:#fff;padding:0 6px;border-radius:3px;">粉色椭圆</span> — 输出
- <span style="background:#e8f0fe;border:1px solid #1a73e8;padding:0 6px;border-radius:3px;">蓝框</span> — Phase 1 模块
- <span style="background:#fff3cd;border:1px solid #856404;padding:0 6px;border-radius:3px;">黄框</span> — 编码器
- <span style="background:#d4edda;border:1px solid #155724;padding:0 6px;border-radius:3px;">绿框</span> — 对齐/融合
- <span style="background:#f8d7da;border:1px solid #721c24;padding:0 6px;border-radius:3px;">红框</span> — 先验精炼
- <span style="background:#f3e5f5;border:1px solid #4a148c;padding:0 6px;border-radius:3px;">紫框</span> — 损失函数
- <span style="border:1px dashed #6c757d;padding:0 6px;border-radius:3px;">虚框</span> — 中间张量