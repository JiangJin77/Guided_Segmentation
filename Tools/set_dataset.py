"""
将预处理过的.nii.gz文件转换为.pt格式的切片张量并保存，同时生成引导图并保存为.pt文件
    1. weight_radio函数用于计算卷积核权重的衰减值
    2. _build_kernels函数根据给定的半径列表构建卷积核，并应用权重衰减
    3. threshold_segmentation_numpy函数对输入图像进行阈值分割，生成二值掩码
    4. add_conv函数对输入图像进行多尺度卷积操作并加权求和，生成引导图
    5. get_slice_tensor函数将.nii.gz文件转换为切片张量并保存为.pt文件，根据flag参数决定处理方式
"""
import os
import math
import h5py
import numpy as np
import SimpleITK as sitk
import torch
from torch.nn import functional as F
from typing import Iterable, List, Tuple


def weight_radio(radius, alpha=0.5, beta=0.5):
    """计算卷积核权重的衰减值，e^(-alpha * (radius ** beta))"""
    return math.e ** (-alpha* (radius ** beta))


def _build_kernels(radii: Iterable[int], alpha=0.5, beta=0.5) -> List[Tuple[int, torch.Tensor]]:
    """根据给定的半径列表构建卷积核，并应用权重衰减"""
    kernels = []
    for r in radii:
        k = 2 * r + 1
        weight_value = torch.ones(k, k).unsqueeze(0).unsqueeze(0) * weight_radio(r, alpha, beta)
        kernels.append((r, weight_value))
    return kernels


def threshold_segmentation_numpy(image_array, lower, upper):
    '''对输入图像进行阈值分割，生成二值掩码
    :param image_array: 输入图像的numpy数组
    :param lower: 阈值下限
    :param upper: 阈值上限
    :return: 二值掩码的torch张量
    '''
    threshold = ((image_array > lower) & (image_array < upper)).astype(np.float32)
    return torch.from_numpy(threshold)


def add_conv(image, alpha=0.5, beta=0.5, radii=None, lower=17.5, upper=22):
    """将阈值分割的图片进行不同尺寸的卷积，再经过权重衰减，最后点加"""
    if radii is None:
        radii = range(1, 4)

    image_array = sitk.GetArrayFromImage(image)
    threshold_tensor = threshold_segmentation_numpy(image_array, lower=lower, upper=upper)
    batch = threshold_tensor.unsqueeze(1)  # (D, H, W) -> (D, 1, H, W)
    result = torch.zeros_like(batch)

    kernels = _build_kernels(radii=radii, alpha=alpha, beta=beta)
    with torch.no_grad():
        for r, weight in kernels:
            result += F.conv2d(batch, weight, bias=None, stride=1, padding=r)

    return result.squeeze(1)    # (D, H, W)


def get_slice_tensor(file_path, save_file, flag='data', lower=17.5, upper=22, radii=None, alpha=0.5, beta=0.5):     
    """将nii.gz文件转换为切片张量并保存
        :param file_path: 输入数据集路径
        :param save_file: 输出数据集路径
        :param flag: 数据集类型
        :param lower: 阈值下限
        :param upper: 阈值上限
        :param radii: 卷积核的半径列表
        :param alpha: 卷积核的权重衰减系数alpha
        :param beta: 卷积核的权重衰减系数beta
    """
    if radii is None:
        radii = range(1, 4)

    if flag not in {'data', 'label', 'threshold'}:
        raise ValueError(f"无效标签: {flag}")
    
    os.makedirs(save_file, exist_ok=True)
    files = [f for f in os.listdir(file_path) if f.endswith(".nii.gz")]
    for file in files:
        image = sitk.ReadImage(os.path.join(file_path, file))    # 获取数据
        if flag == 'data':   
            image_array = sitk.GetArrayFromImage(image)    # 转换为numpy数组 （Z, Y, X) 
            image_tensor = torch.from_numpy(image_array)     # 转换为张量   （Z, Y, X) = (D, H, W)
        elif flag == 'label':
            image_array = sitk.GetArrayFromImage(image)
            image_tensor = torch.from_numpy(image_array).to(torch.uint8)# 转换为 uint8 保存
        elif flag == 'threshold':
            image_tensor = add_conv(image, alpha=alpha, beta=beta, radii=radii, lower=lower, upper=upper)

        for i in range(image_tensor.shape[0]):
            slice_tensor = image_tensor[i, :, :]
            slice_save_path = os.path.join(save_file, f"{file.replace('.nii.gz', '')}_{i}.h5")
            with h5py.File(slice_save_path, 'w') as f:
                f.create_dataset('data', data=slice_tensor.numpy(), compression='gzip')   


if __name__ == "__main__":
    # 生成5折交叉总的数据集
    a, b, k = 0.75, 0.25, 1
    data_path = r"E:\JiangJin\dataset\train_test\data"
    label_path = r"E:\JiangJin\dataset\train_test\labelmap"
    data_save_path = r"E:\My_vscode_project\Guidance\Project_1\dataset\5folds\tensor\images"
    label_save_path = r"E:\My_vscode_project\Guidance\Project_1\dataset\5folds\tensor\masks"
    threshold_save_path = r"E:\My_vscode_project\Guidance\Project_1\dataset\5folds\tensor\thresholds"
    get_slice_tensor(file_path=data_path, 
                     save_file=data_save_path, 
                     flag='data')
    get_slice_tensor(file_path=label_path, 
                     save_file=label_save_path,
                     flag='label')
    get_slice_tensor(file_path=data_path, 
                     save_file=threshold_save_path,
                     flag='threshold',
                     alpha=a, beta=b, radii=range(1, k + 1),
                     lower=17.5, upper=22)
    print("数据集处理完成！")