"""
JJNet 训练脚本
"""
import csv
import logging
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from Network.JJNet import JJNet, JJNetLoss


# ============================================================================
# 数据集 (仅加载 images / masks, 无需预计算 threshold)
# ============================================================================
class JJNetDataSet(Dataset):
    """JJNet 数据集 —— 仅需要 CT 图像和标签掩码"""

    def __init__(self, data_path, augment=True):
        self.data_path = Path(data_path)
        self.augment = augment
        self.flip_modes = [0, 1, 2, 3] if augment else [0]
        self.augment_factor = len(self.flip_modes)

        images_dir = self.data_path / 'images'
        masks_dir = self.data_path / 'masks'
        if not images_dir.exists() or not masks_dir.exists():
            raise FileNotFoundError(f'目录不存在: images 或 masks 在 {data_path}')

        image_names = {p.stem for p in images_dir.glob('*.h5')}
        mask_names = {p.stem for p in masks_dir.glob('*.h5')}
        common = sorted(image_names & mask_names)

        self.data_list = [
            {'base_name': name,
             'image': images_dir / f'{name}.h5',
             'mask': masks_dir / f'{name}.h5'}
            for name in common
        ]
        self.total_len = len(self.data_list) * self.augment_factor
        logging.info('从 %s 加载 %d 个样本 (增强后 %d)', data_path, len(self.data_list), self.total_len)

    @staticmethod
    def _load_h5(path):
        with h5py.File(path, 'r') as f:
            return torch.from_numpy(f['data'][()])

    def __len__(self):
        return self.total_len

    def __getitem__(self, idx):
        data_idx = idx // self.augment_factor
        aug_mode = idx % self.augment_factor
        paths = self.data_list[data_idx]

        image = self._load_h5(paths['image']).float()
        mask = self._load_h5(paths['mask']).long()

        # 添加通道维度
        if image.ndim == 2:
            image = image.unsqueeze(0)
        if mask.ndim == 2:
            mask = mask.unsqueeze(0)

        # 标签转为二值 (病灶 vs 背景)
        mask = (mask > 0).float()

        # 翻转增强
        if self.augment and aug_mode != 0:
            dims = {1: [2], 2: [1], 3: [1, 2]}[aug_mode]
            image = torch.flip(image, dims=dims)
            mask = torch.flip(mask, dims=dims)

        return image, mask


# ============================================================================
# 工具函数
# ============================================================================
def compute_dice(pred, target, smooth=1.0):
    """计算 Dice 系数 (pred 和 target 均为二值)"""
    pred_flat = pred.contiguous().view(-1)
    target_flat = target.contiguous().view(-1)
    intersection = (pred_flat * target_flat).sum()
    return (2.0 * intersection + smooth) / (pred_flat.sum() + target_flat.sum() + smooth)


def compute_iou(pred, target, smooth=1.0):
    """计算 IoU"""
    pred_flat = pred.contiguous().view(-1)
    target_flat = target.contiguous().view(-1)
    intersection = (pred_flat * target_flat).sum()
    union = pred_flat.sum() + target_flat.sum() - intersection
    return (intersection + smooth) / (union + smooth)


# ============================================================================
# 训练主函数
# ============================================================================
def train(
    # 数据
    train_data_dir='/root/autodl-tmp/py_01/dataset/tensor/train',
    val_data_dir='/root/autodl-tmp/py_01/dataset/tensor/val',
    # 训练超参数
    batch_size=8,
    learning_rate=0.001,
    num_epochs=50,
    device=None,
    # 网络参数
    channel=64,
    prior_init_lower=17.5,  
    prior_init_upper=22.0,
    prior_radii=None,
    num_refine_iters=1,
    # 损失权重
    ds_weights=None,
    prior_weight=0.1,
    asym_weight=0.05,
    offset_weight=0.001,
    # 训练策略
    patience=8,
    scheduler_patience=3,
    # 输出
    output_dir='./outputs/JJNet',
    resume=False,
):
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if prior_radii is None:
        prior_radii = range(1, 4)
    if ds_weights is None:
        ds_weights = [0.1, 0.2, 0.2, 0.5]

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f'结果保存至: {output_dir}')

    # ---- 数据加载 ----
    train_dataset = JJNetDataSet(train_data_dir, augment=True)
    val_dataset = JJNetDataSet(val_data_dir, augment=False)
    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                              shuffle=True, num_workers=8, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size,
                            shuffle=False, num_workers=8, pin_memory=True)

    # ---- 模型、损失、优化器 ----
    model = JJNet(
        channel=channel,
        prior_init_lower=prior_init_lower,
        prior_init_upper=prior_init_upper,
        prior_radii=prior_radii,
        num_refine_iters=num_refine_iters,
        deep_supervision=True,
    ).to(device)

    criterion = JJNetLoss(
        deep_supervision_weights=ds_weights,
        prior_weight=prior_weight,
        asym_weight=asym_weight,
        offset_weight=offset_weight,
    )

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=scheduler_patience, verbose=True
    )
    scaler = torch.amp.GradScaler('cuda') if device.type == 'cuda' else None

    # ---- 保存路径 ----
    save_path = output_dir / 'model.pth'
    checkpoint_path = output_dir / 'checkpoint.pth.tar'

    # ---- 训练记录 ----
    train_losses = []
    val_losses = []
    val_dices = []
    val_ious = []
    best_dice = 0.0
    start_epoch = 0
    epoch_no_improve = 0

    # ---- 恢复检查点 ----
    if resume and checkpoint_path.exists():
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        train_losses = checkpoint['train_losses']
        val_losses = checkpoint['val_losses']
        val_dices = checkpoint['val_dices']
        val_ious = checkpoint['val_ious']
        best_dice = checkpoint['best_dice']
        epoch_no_improve = checkpoint['epoch_no_improve']
        if scaler and 'scaler' in checkpoint:
            scaler.load_state_dict(checkpoint['scaler'])
        print(f'恢复至 Epoch {start_epoch}, 最佳 Dice: {best_dice:.4f}')
    else:
        print('从头开始训练...')

    # ---- 训练循环 ----
    for epoch in range(start_epoch, num_epochs):
        model.train()
        train_loss = 0.0

        loop = tqdm(train_loader, leave=False, desc=f'Epoch [{epoch + 1}/{num_epochs}]')
        for image, mask in loop:
            image, mask = image.to(device), mask.to(device)

            # JJNet forward
            if scaler:
                with torch.amp.autocast('cuda'):
                    outputs = model(image)
                    (edge, sal1, sal2, sal3,
                     prior_maps, diff_maps, gate_map, lower, upper) = outputs

                    # 收集偏移场用于损失
                    offsets = []
                    for name, module in model.named_modules():
                        if 'contra_fusions' in name and hasattr(module, 'last_offset'):
                            offsets.append(module.last_offset)

                    loss = criterion([edge, sal1, sal2, sal3], mask,
                                     prior_maps=prior_maps,
                                     diff_maps=diff_maps,
                                     offsets=offsets if offsets else None)

                optimizer.zero_grad()
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(image)
                (edge, sal1, sal2, sal3,
                 prior_maps, diff_maps, gate_map, lower, upper) = outputs

                offsets = []
                for name, module in model.named_modules():
                    if 'contra_fusions' in name and hasattr(module, 'last_offset'):
                        offsets.append(module.last_offset)

                loss = criterion([edge, sal1, sal2, sal3], mask,
                                 prior_maps=prior_maps,
                                 diff_maps=diff_maps,
                                 offsets=offsets if offsets else None)

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            train_loss += loss.item()
            loop.set_postfix({'loss': f'{loss.item():.4f}'})

        train_loss /= len(train_loader)
        train_losses.append(train_loss)

        # ---- 验证 ----
        model.eval()
        val_loss = 0.0
        val_dice_sum = 0.0
        val_iou_sum = 0.0

        with torch.no_grad():
            for image, mask in tqdm(val_loader, leave=False, desc='验证'):
                image, mask = image.to(device), mask.to(device)

                outputs = model(image)
                (edge, sal1, sal2, sal3,
                 prior_maps, diff_maps, gate_map, lower, upper) = outputs

                offsets = []
                for name, module in model.named_modules():
                    if 'contra_fusions' in name and hasattr(module, 'last_offset'):
                        offsets.append(module.last_offset)

                loss = criterion([edge, sal1, sal2, sal3], mask,
                                 prior_maps=prior_maps,
                                 diff_maps=diff_maps,
                                 offsets=offsets if offsets else None)
                val_loss += loss.item()

                # 评估指标 (使用融合输出 sal3)
                pred = (torch.sigmoid(sal3) > 0.5).float()
                val_dice_sum += compute_dice(pred, mask).item()
                val_iou_sum += compute_iou(pred, mask).item()

        val_loss /= len(val_loader)
        val_dice = val_dice_sum / len(val_loader)
        val_iou = val_iou_sum / len(val_loader)
        val_losses.append(val_loss)
        val_dices.append(val_dice)
        val_ious.append(val_iou)

        scheduler.step(val_dice)

        print(f'Epoch [{epoch + 1}/{num_epochs}]  '
              f'Train Loss: {train_loss:.4f}  '
              f'Val Loss: {val_loss:.4f}  '
              f'Dice: {val_dice:.4f}  '
              f'IoU: {val_iou:.4f}  '
              f'lower: {lower.item():.2f}  upper: {upper.item():.2f}')

        # ---- 保存最佳模型 ----
        if val_dice > best_dice:
            best_dice = val_dice
            epoch_no_improve = 0
            torch.save(model.state_dict(), save_path)
            print(f'  -> 最佳模型已保存 (Dice: {val_dice:.4f})')
        else:
            epoch_no_improve += 1
            if epoch_no_improve >= patience:
                print(f'早停: Epoch {epoch + 1}')
                break

        # ---- 保存检查点 ----
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_losses': train_losses,
            'val_losses': val_losses,
            'val_dices': val_dices,
            'val_ious': val_ious,
            'best_dice': best_dice,
            'epoch_no_improve': epoch_no_improve,
            'scaler': scaler.state_dict() if scaler else None,
        }
        torch.save(checkpoint, checkpoint_path)

    # ---- 保存训练曲线 ----
    csv_path = output_dir / 'train_log.csv'
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Epoch', 'Train Loss', 'Val Loss', 'Dice', 'IoU'])
        for i in range(len(train_losses)):
            writer.writerow([i + 1, train_losses[i], val_losses[i], val_dices[i], val_ious[i]])

    plt.figure(figsize=(14, 5))

    plt.subplot(1, 2, 1)
    plt.plot(range(1, len(train_losses) + 1), train_losses, label='Train Loss', linewidth=2)
    plt.plot(range(1, len(val_losses) + 1), val_losses, label='Val Loss', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training & Validation Loss')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)

    plt.subplot(1, 2, 2)
    plt.plot(range(1, len(val_dices) + 1), val_dices, label='Dice', linewidth=2)
    plt.plot(range(1, len(val_ious) + 1), val_ious, label='IoU', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Score')
    plt.title('Validation Metrics')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.savefig(output_dir / 'training_curve.png', dpi=300, bbox_inches='tight')
    plt.close()

    print(f'\n训练完成! 最佳 Dice: {best_dice:.4f}')
    print(f'模型: {save_path}')
    print(f'日志: {csv_path}')


def main():
    train()


if __name__ == '__main__':
    main()