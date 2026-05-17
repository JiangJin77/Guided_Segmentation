'''
网格搜索法获取阈值
    load_dataset:预加载数据集到内存中
    get_dice:计算Dice系数
    single_epoch_dice:计算单个阈值组合的平均损失和Dice系数
    grid_search_threshold:执行网格搜索，寻找最佳阈值组合
'''
import os
import numpy as np
import SimpleITK as sitk
from tqdm import tqdm

def load_dataset(data_path, label_path):
    """预加载数据集到内存中"""
    data_list = []
    label_list = []
    data_files = os.listdir(data_path)
    data_files.sort(key=lambda x: int(x.replace('.nii.gz', '')) if x.endswith('.nii.gz') else float('inf'))

    for data_file in data_files:
        data_image = sitk.ReadImage(os.path.join(data_path, data_file))
        label_image = sitk.ReadImage(os.path.join(label_path, data_file))
        data_list.append(sitk.GetArrayFromImage(data_image))    
        label_list.append(sitk.GetArrayFromImage(label_image))
    return data_list, label_list 


def get_dice(pred, target, smooth=1e-5):
    """计算Dice系数"""
    intersection = (pred * target).sum()
    dice = (2. * intersection + smooth) / (pred.sum() + target.sum() + smooth)
    return dice


def single_epoch_dice(node, data_list, label_list, whether_positive=False, alpha=1, beta=1):

    bl = 0  # 背景Dice
    fl = 0  # 前景Dice
    loss_total = 0  # 损失
    num_not_zero = 0    # 非零数据个数
    lower, upper = node
    
    for data_array, label_array in zip(data_list, label_list):
        threshold_array = ((data_array >= lower) & (data_array <= upper)).astype(np.float32)

        # 标签处理逻辑
        fore_label = (label_array == 2).astype(np.float32)  # 前景标签（CSF区域）
        fore_dice = get_dice(threshold_array, fore_label)    # dice(分割二值图，梗死区)
        
        back_label = (label_array != 0).astype(np.float32)  # 背景标签（大脑区，包含CSF区域）
        back_dice = get_dice(threshold_array, back_label)    # dice(分割二值图，大脑区)

        if whether_positive and np.max(label_array) == 1:
            loss = 0
        else:
            loss = alpha * back_dice - beta * fore_dice
            num_not_zero += 1
            bl += back_dice
            fl += fore_dice
        loss_total += loss

    if num_not_zero == 0:
        return 0.0, 0.0, 0.0
    return loss_total / num_not_zero, alpha * bl, beta * fl


def grid_search_threshold(datapath = r"E:\JiangJin\dataset\train_test\data",
                          labelpath = r"E:\JiangJin\dataset\train_test\labelmap",
                          lower_min=0.0, 
                          lower_max=50.5, 
                          upper_min=0.0, 
                          upper_max=50.5,
                          step=0.5,
                          whether_positive=False,                         
                          alpha=1, beta=1):
    # 预加载数据集
    print("正在预加载数据集...")
    data_list, label_list = load_dataset(datapath, labelpath)

    # 定义搜索范围和步长
    lower_range = np.arange(lower_min, lower_max + step, step)  
    upper_range = np.arange(upper_min, upper_max + step, step)  
    
    best_loss = float('inf')
    best_params = None
    results = []
    # 生成有效组合列表（lower < upper）
    valid_params = [(l, u) for l in lower_range for u in upper_range if l < u]
    valid_combinations = len(valid_params)
    total_combinations = len(lower_range) * len(upper_range)  
    print(f"开始网格搜索，总共 {total_combinations} 种组合，其中有 {valid_combinations} 个有效组合")
    
    for lower, upper in tqdm(valid_params, desc="搜索中", unit="组合"):    
        try:
            loss, back_dice_sum, fore_dice_sum = single_epoch_dice([lower, upper], 
                                                                       data_list, label_list,
                                                                       whether_positive, alpha, beta)
            results.append({
                'lower': lower,
                'upper': upper, 
                'loss': loss,
                'back_dice_sum': back_dice_sum,
                'fore_dice_sum': fore_dice_sum
            })
                
            if loss < best_loss:
                best_loss = loss
                best_params = [lower, upper]
     
        except Exception as e:
            tqdm.write(f"组合 (lower={lower}, upper={upper}) 计算失败: {e}")
            continue

    print(f"\n网格搜索完成！")
    
    if best_params is not None:
        print(f"最佳参数: [{best_params[0]:.2f}, {best_params[1]:.2f}]")
        print(f"最佳损失: {best_loss:.6f}")
    else:
        print("未找到有效参数组合")
    
    return best_params, best_loss, results


if __name__ == '__main__':
    # 运行网格搜索
    best_params, best_loss, results = grid_search_threshold()
