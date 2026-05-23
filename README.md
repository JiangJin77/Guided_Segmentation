# Guided_Segmentation

## 数据集划分
共 194 个 nii.gz 文件（1.nii.gz ~ 194.nii.gz），训练集：测试集=8：2：
训练集：1.nii.gz ~ 155.nii.gz
测试集：156.nii.gz ~ 194.nii.gz

### 原始 nii.gz
```
data/                     # 原始图像数据
  └── 1.nii.gz ~ 194.nii.gz
labelmap/                 # 标签
  └── 1.nii.gz ~ 194.nii.gz
```

### 切片后的 .h5 张量
每个 nii.gz 文件逐切片（沿 Z 轴）保存为独立的 .h5 文件：
```
dataset/
  └── slices/
        ├── images/          # 原始图像切片
        │     ├── 1_0.h5    # 1.nii.gz 的第0层切片
        │     ├── 1_1.h5    # 1.nii.gz 的第1层切片
        │     └── ...
        ├── masks/           # 标签切片 (torch.uint8)
        │     ├── 1_0.h5    
        │     ├── 1_1.h5    
        │     └── ...
        └── thresholds/       # 引导图切片 (卷积生成)
        │     ├── 1_0.h5    
        │     ├── 1_1.h5    
        │     └── ...
```

- images: `torch.float32`, shape：`(H, W)`
- masks: `torch.uint8`, shape：`(H, W)`，值范围：`{0, 1, 2}`
- thresholds: `torch.float32`, shape：`(H, W)`

### 阈值引导图生成
使用 `Tools/set_dataset.py` 中的 `add_conv` 函数：
1. 对原始图像做阈值分割（lower=17.5, upper=22.0）
2. 用不同尺寸的卷积核做多尺度卷积并加权求和
