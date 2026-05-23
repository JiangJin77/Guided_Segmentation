# 统计标签中没有类别2的数量及名字
import os
import nibabel as nib
import numpy as np
def count_no_class2_labels(label_dir):
    '''统计标签中没有类别2的数量及名字
    Args:
        label_dir (str): 标签文件夹路径
        return: (int, list)
    '''
    names = []
    names_list = os.listdir(label_dir)   # 获取文件夹中的所有文件名
    names_list.sort(key=lambda x: int(x.replace('.nii.gz', '')) if x.endswith('.nii.gz') else float('inf'))

    for name in names_list:   
        if name.endswith('.nii.gz'):
            label_path = os.path.join(label_dir, name)
            label_img = nib.load(label_path)
            label = label_img.get_fdata()
            if not np.any(label == 2):
                names.append(name)
                print(name)
                
    return len(names), names


if __name__ == '__main__':
    label_directory = r'E:\My_vscode_project\Dataset\data\niigz\labels' # 标签文件夹路径
    count, names = count_no_class2_labels(label_directory)
    print(f'不含标签2的个数: {count}')