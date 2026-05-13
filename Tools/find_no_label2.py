# 统计标签中没有类别2的数量及名字
import os
import nibabel as nib

def count_no_class_2_labels(label_dir):
    count = 0
    names = []
    names_dir = os.listdir(label_dir)
    names_dir.sort(key=lambda x: int(x.replace('.nii.gz', '')) if x.endswith('.nii.gz') else float('inf'))
    # print(names_dir)
    for name in names_dir:   
        if name.endswith('.nii.gz'):
            label_path = os.path.join(label_dir, name)
            label_img = nib.load(label_path)
            label = label_img.get_fdata()
            if 2 not in label:
                count += 1
                names.append(name)
                print(name)
                
    return count, names

label_directory = r'E:\My_vscode_project\Dataset\data\niigz\labels'
count, names = count_no_class_2_labels(label_directory)
print(f'不含标签2的个数: {count}')