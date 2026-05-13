# 将.npy文件转换为.nii.gz文件
import numpy as np
import nibabel as nib
import os
import re
def get_original_nifti_path(sample_name, search_dirs):
    """
    根据样本名称在指定目录中查找原始的 .nii.gz 文件
    """
    for dir_path in search_dirs:
        # 尝试常见命名格式: sample_name.nii.gz 或 sample_name/img.nii.gz
        target_path = os.path.join(dir_path, f"{sample_name}.nii.gz")
        if os.path.exists(target_path):
            return target_path
        
    return None


def convert_npy_to_nifti(npy_path, nii_gz_path, original_data_dir=None):
    """
    将 .npy 文件转换为 .nii.gz 文件
    npy_path (str): 输入的 .npy 文件路径
    nii_gz_path (str): 输出的 .nii.gz 文件路径
    affine (np.ndarray, optional): 仿射变换矩阵 (4x4)。如果为 None，则使用单位矩阵。
    """
    # 1. 加载 .npy 数据
    if not os.path.exists(npy_path):
        raise FileNotFoundError(f"文件不存在: {npy_path}")
    
    data = np.load(npy_path)    #(Z, Y, X) 维度顺序
    data_1 = np.copy(data)
    data_1[data_1 == 1] = 2
    print(f"原始数据唯一值: {np.unique(data)}")
    print(f"修改后数据唯一值: {np.unique(data_1)}")
    affine = np.eye(4)
    if original_data_dir:
        # 从 npy 文件名提取样本名 (例如: "156_pred.npy" -> "156")
        base_name = os.path.basename(npy_path)
        # 去除 _pred.npy 或 _target.npy 后缀
        sample_name = re.sub(r'_(pred|target)\.npy$', '', base_name)
        
        # 查找原始图像
        original_nifti_path = get_original_nifti_path(sample_name, [original_data_dir])
        
        if original_nifti_path:
            try:
                ref_img = nib.load(original_nifti_path)
                affine = ref_img.affine
                header = ref_img.header
                # 获取体素间距，用于打印验证
                zooms = header.get_zooms()
                print(f"找到参考图像: {original_nifti_path}")
                print(f"参考图像 Affine:\n{affine}")
                print(f"参考图像体素间距 (X,Y,Z): {zooms}")
            except Exception as e:
                print(f"警告: 无法加载参考图像 {original_nifti_path}, 使用单位矩阵. 错误: {e}")
        else:
            print(f"警告: 在 {original_data_dir} 中未找到样本 {sample_name} 的原始图像，使用单位矩阵。")

    data_1 = np.transpose(data_1, (2, 1, 0))  # 转换为 (D, H, W) 维度顺序，符合 NIfTI 标准
    # 3. 创建 NIfTI 图像对象
    # nibabel 期望数据维度通常为 (x, y, z) 或 (x, y, z, t) 等
    nifti_img = nib.Nifti1Image(data_1.astype(np.int16), affine)
    print(f"转换后数据唯一值: {np.unique(nifti_img)}")
    # 4. 保存为 .nii.gz
    # 确保输出目录存在
    output_dir = os.path.dirname(nii_gz_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    nib.save(nifti_img, nii_gz_path)
    print(f"成功将 {npy_path} 转换为 {nii_gz_path}")
    

# 使用示例
if __name__ == "__main__":
    # 请替换为你的实际文件路径
    input_npy = r"E:\My_vscode_project\Guidance\Project_1\对比实验\Outputs\CFANet_C_1\fold_2_123_1\predictions_3d\158_pred.npy"       # 输入文件
    output_nii = r"E:\My_vscode_project\Guidance\Project_1\158_pred.nii.gz"   # 输出文件
    original_data_dir = r"E:\My_vscode_project\Guidance\Project_1\dataset\niigz\test\data"      # 源数据目录
    try:
        convert_npy_to_nifti(input_npy, output_nii, original_data_dir)
    except Exception as e:
        print(f"转换失败: {e}")