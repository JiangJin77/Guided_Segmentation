import logging
from pathlib import Path

import h5py
import torch
from torch.utils.data import Dataset


logger = logging.getLogger(__name__)    # 创建一个日志记录器，用于输出数据加载和处理的相关信息


class TensorFlipAugment:
    """对图像、掩码和阈值张量执行同步翻转增强"""

    FLIP_MODES = {
        0: None,      # 原始
        1: [2],       # 水平翻转
        2: [1],       # 垂直翻转
        3: [1, 2],    # 水平+垂直翻转
    }

    def __init__(self, mode: int):
        self.mode = mode

    def __call__(self, image: torch.Tensor, mask: torch.Tensor, threshold: torch.Tensor):
        dims = self.FLIP_MODES.get(self.mode)
        if dims is None:
            return image, mask, threshold
        return torch.flip(image, dims=dims), torch.flip(mask, dims=dims), torch.flip(threshold, dims=dims)


class DataSetFFM(Dataset):
    """基于 images/masks/thresholds 子目录的 PyTorch Dataset。"""

    def __init__(self, data_path, augment: bool = True, verbose: bool = True):
        self.data_path = Path(data_path)
        self.augment = augment
        self.flip_modes = sorted(TensorFlipAugment.FLIP_MODES.keys()) if augment else [0]
        self.augment_factor = len(self.flip_modes)

        self.data_list = self._load_data_list()
        self.total_len = len(self.data_list) * self.augment_factor

        if verbose:     
            logger.info('从 %s 中导入 %d 个文件', self.data_path, len(self.data_list))
            logger.info('数据总数: %d', self.total_len)

    def _load_data_list(self):
        """加载 images、masks、thresholds 三个目录中的公共文件名。"""
        images_dir = self.data_path / 'images'
        masks_dir = self.data_path / 'masks'
        thresholds_dir = self.data_path / 'thresholds'

        for directory in (images_dir, masks_dir, thresholds_dir):
            if not directory.exists():
                logger.warning('目录不存在: %s', directory)
                return []

        image_names = {path.stem for path in images_dir.glob('*.h5')}
        mask_names = {path.stem for path in masks_dir.glob('*.h5')}
        threshold_names = {path.stem for path in thresholds_dir.glob('*.h5')}

        common_names = sorted(image_names & mask_names & threshold_names)
        return [
            {
                'base_name': name,
                'image': images_dir / f'{name}.h5',
                'mask': masks_dir / f'{name}.h5',
                'threshold': thresholds_dir / f'{name}.h5',
            }
            for name in common_names
        ]
    
    # 确保张量的数据类型和形状
    @staticmethod
    def _load_h5(path):
        with h5py.File(path, 'r') as f:
            return torch.from_numpy(f['data'][()])

    # 确保张量的数据类型正确，并且不包含梯度信息
    @staticmethod
    def _ensure_tensor(value, dtype):
        tensor = torch.as_tensor(value)
        if tensor.dtype != dtype:
            tensor = tensor.to(dtype)
        return tensor.detach()
    
    # 如果图像或掩码是二维或三维的，添加一个通道维度
    @staticmethod
    def _add_channel_dim(tensor: torch.Tensor):
        if tensor.ndim in {2, 3}:
            return tensor.unsqueeze(0)
        return tensor
    
    # 验证掩码张量的值
    @staticmethod
    def _validate_mask(mask: torch.Tensor):
        if mask.min() < 0 or mask.max() > 2:
            logger.warning('Mask contains values outside [0, 1, 2]: %s', torch.unique(mask))
            return mask.clamp(0, 2)
        return mask

    def __len__(self):
        return self.total_len

    def __getitem__(self, idx):
        if not self.data_list:
            raise IndexError('数据集为空，无法索引。')

        data_idx = idx // self.augment_factor
        aug_mode = idx % self.augment_factor
        paths = self.data_list[data_idx]

        image = self._load_h5(paths['image'])
        mask = self._load_h5(paths['mask'])
        threshold = self._load_h5(paths['threshold'])

        image = self._ensure_tensor(image, torch.float32)
        mask = self._ensure_tensor(mask, torch.long)
        threshold = self._ensure_tensor(threshold, torch.float32)

        mask = self._validate_mask(mask)

        image = self._add_channel_dim(image)
        threshold = self._add_channel_dim(threshold)
        mask = self._add_channel_dim(mask)

        if self.augment and aug_mode != 0:
            image, mask, threshold = TensorFlipAugment(mode=aug_mode)(image, mask, threshold)

        file_name = f"{paths['base_name']}_aug{aug_mode}"
        return image, mask, threshold, file_name

    def __repr__(self):
        return f'{self.__class__.__name__}(data_path={self.data_path}, augment={self.augment}, total_len={self.total_len})'
