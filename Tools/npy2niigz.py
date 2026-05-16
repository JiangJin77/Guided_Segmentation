# 将.npy文件转换为.nii.gz文件
import re
import numpy as np
import nibabel as nib
from pathlib import Path


def get_original_nifti_path(sample_name: str, search_dirs: list[str]) -> Path | None:
    """ 根据样本名称在指定目录中查找原始的 .nii.gz 文件 """
    for dir_path in search_dirs:
        # sample_name.nii.gz 或 sample_name/img.nii.gz
        target = Path(dir_path) / f"{sample_name}.nii.gz"
        if target.exists():
            return target
    return None


def convert_npy_to_nifti(
    npy_path: str, 
    nii_gz_path: str, 
    original_data_dir: str | None = None
    ):
    """ 将 .npy 文件转换为 .nii.gz 文件
        npy_path: 输入的 .npy 文件路径
        nii_gz_path: 输出的 .nii.gz 文件路径
        original_data_dir: 原始 NIfTI 数据目录（用于继承 affine/header）。如果为 None，则使用单位矩阵。
    """
    npy_path = Path(npy_path)
    if not npy_path.exists():
        raise FileNotFoundError(f"文件不存在: {npy_path}")
    
    # 1. 加载 .npy 数据   (Z, Y, X)
    data = np.load(str(npy_path))    #(Z, Y, X) 维度顺序
    data_mod = np.where(data == 1, 2, data)     # 标签1 -> 标签2
    print(f"原始数据唯一值: {np.unique(data)}")
    print(f"修改后数据唯一值: {np.unique(data_mod)}")

    # 2. 确定affine矩阵
    affine = np.eye(4)
    if original_data_dir:
        # 从npy文件名中提取样本名称，假设命名格式为 sample_pred.npy 或 sample_target.npy
        base_name = npy_path.name   
        sample_name = re.sub(r'_(pred|target)\.npy$', '', base_name)    # 去除 _pred.npy 或 _target.npy 后缀  
        original_nifti_path = get_original_nifti_path(sample_name, [original_data_dir]) # 查找原始图像
        
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
    
    # 3. (Z, Y, X) -> (X, Y, Z)，符合 NIfTI 标准
    data_mod = np.transpose(data_mod, (2, 1, 0)) 

    # 4. 创建 NIfTI 图像
    nifti_img = nib.Nifti1Image(data_mod.astype(np.int16), affine)
    print(f"转换后数据唯一值: {np.unique(nifti_img)}")

    # 确保输出目录存在
    output_path = Path(nii_gz_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(nifti_img, str(output_path))
    print(f"成功将 {npy_path} 转换为 {output_path}")
    

if __name__ == "__main__":
    input_npy = r"E:\My_vscode_project\Guidance\Project_1\对比实验\Outputs\CFANet_C_1\fold_2_123_1\predictions_3d\158_pred.npy"       # 输入文件
    output_nii = r"E:\My_vscode_project\Guidance\Project_1\158_pred.nii.gz"   # 输出文件
    original_data_dir = r"E:\My_vscode_project\Guidance\Project_1\dataset\niigz\test\data"      # 源数据目录
    try:
        convert_npy_to_nifti(input_npy, output_nii, original_data_dir)
    except Exception as e:
        print(f"转换失败: {e}")