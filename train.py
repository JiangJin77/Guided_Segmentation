import csv
import torch
from pathlib import Path
from tqdm import tqdm
from torch import nn, optim
from torch.utils.data import DataLoader
from data import DataSetFFM
import matplotlib.pyplot as plt
from Network.CCFANet import CCFANet


def prepare_input(image, threshold, channel):
    if channel == 1:
        return image
    return torch.cat((image, threshold), dim=1)


def get_model_output(model, input_tensor):
    outputs = model(input_tensor)
    if isinstance(outputs, tuple):
        return outputs[-1]
    return outputs


def compute_metrics(conf_intersection, conf_union, num_classes):
    """从累积的 confusion matrix 计算 mIoU 和 mDice"""
    ious, dices = [], []
    for cls in range(num_classes):
        inter = conf_intersection[cls].item()
        union = conf_union[cls].item()
        if union == 0:
            continue
        ious.append(inter / union)
        dices.append(2 * inter / (union + inter))
    miou = sum(ious) / len(ious) if ious else 0.0
    mdice = sum(dices) / len(dices) if dices else 0.0
    return miou, mdice, ious, dices


# =====================================================================
# 全局默认超参数 
# =====================================================================
BATCH_SIZE = 8  
LEARNING_RATE = 0.001
NUM_EPOCHS = 50
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
CHANNEL = 2  
BASE_OUTPUT_DIR = '/root/autodl-tmp/py_01/寻参实验/Outputs' # 基础输出目录
EARLY_STOPPING_PATIENCE = 8  # 早停值
SCHEDULER_PATIENCE = 3  # 学习率调度器的早停值

def train(num_classes=3, 
          channel=CHANNEL, 
          batch_size=BATCH_SIZE,
          learning_rate=LEARNING_RATE,
          num_epochs=NUM_EPOCHS,
          device=DEVICE,
          output_dir=BASE_OUTPUT_DIR,
          patience=EARLY_STOPPING_PATIENCE,
          scheduler_patience=SCHEDULER_PATIENCE,
          resume=False):
    
    # 创建文件夹
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    print(f"结果保存至: {output_dir}")

    # 2. 数据加载
    train_dataset = DataSetFFM(data_path='/root/autodl-tmp/py_01/dataset/tensor/train', augment=True)
    val_dataset = DataSetFFM(data_path='/root/autodl-tmp/py_01/dataset/tensor/val', augment=False)
    data_loader = DataLoader(train_dataset, 
                            batch_size=batch_size, 
                            num_workers=8,
                            pin_memory=True,
                            shuffle=True)
    val_loader = DataLoader(val_dataset, 
                            batch_size=batch_size, 
                            num_workers=8,
                            pin_memory=True,
                            shuffle=False)

    # 模型、优化器、损失函数
    model = CCFANet(resinc=channel).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=scheduler_patience, verbose=True)
    criterion = nn.CrossEntropyLoss(weight=torch.tensor([1.0, 2.0, 7.0], dtype=torch.float32).to(device))
    scaler = torch.amp.GradScaler('cuda') if device.type == 'cuda' else None    # 混合精度训练
    
    # 模型、检查点保存路径
    save_path = Path(output_dir) / 'model.pth'
    checkpoint_path = Path(output_dir) / 'checkpoint.pth.tar'

    # 训练记录初始化
    train_losses = []
    val_losses = []
    val_mious = []
    val_dices = []
    val_cls2_ious = []
    best_loss = float('inf')
    best_cls2_iou = 0.0
    start_epoch = 0
    epoch_no_improve = 0

    # 尝试加载检查点以恢复训练
    if resume and checkpoint_path.exists():
        print(f'Loading checkpoint from {checkpoint_path}...')
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        train_losses = checkpoint['train_losses']
        val_losses = checkpoint['val_losses']
        val_mious = checkpoint.get('val_mious', [])
        val_dices = checkpoint.get('val_dices', [])
        val_cls2_ious = checkpoint.get('val_cls2_ious', [])
        best_loss = checkpoint['best_loss']
        best_cls2_iou = checkpoint.get('best_cls2_iou', 0.0)
        epoch_no_improve = checkpoint['epoch_no_improve']
        if scaler and 'scaler' in checkpoint:
            scaler.load_state_dict(checkpoint['scaler'])
        print(f"成功恢复至Epoch：{start_epoch}, 最佳Loss: {best_loss:.4f}, 最佳cls2 IoU: {best_cls2_iou:.4f}")
    else:
        print("未找到检查点，从头开始训练...")
    
    # ===== 2.训练循环 =====
    for epoch in range(start_epoch, num_epochs):
        model.train()
        train_loss = 0

        loop = tqdm(data_loader, leave=False, desc=f'Epoch[{epoch+1}/{num_epochs}]')
        for i, (image, mask, threshold, _) in enumerate(loop):
            image, mask, threshold = image.to(device), mask.to(device), threshold.to(device)

            input_tensor = prepare_input(image, threshold, channel)
            out_image = get_model_output(model, input_tensor)

            mask = mask.squeeze(1).long()

            if scaler:
                with torch.amp.autocast('cuda'):
                    loss = criterion(out_image, mask)
                optimizer.zero_grad()
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss = criterion(out_image, mask)
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            train_loss += loss.item()
            loop.set_postfix({'loss': f'{loss.item():.4f}'})

        train_loss /= len(data_loader)
        train_losses.append(train_loss)

        # 验证阶段
        model.eval()
        val_loss = 0
        conf_intersection = torch.zeros(num_classes, device=device) 
        conf_union = torch.zeros(num_classes, device=device)     
        with torch.no_grad():
            for i, (image, mask, threshold, _) in enumerate(val_loader):
                image, mask, threshold = image.to(device), mask.to(device), threshold.to(device)

                input_tensor = prepare_input(image, threshold, channel)
                out_image = get_model_output(model, input_tensor)
                mask = mask.squeeze(1).long()
                loss = criterion(out_image, mask)
                val_loss += loss.item()

                # 累积 confusion matrix 以计算 mIoU / Dice
                pred = out_image.argmax(dim=1)
                for cls in range(num_classes):
                    pred_mask = (pred == cls)
                    true_mask = (mask == cls)
                    conf_intersection[cls] += (pred_mask & true_mask).sum()
                    conf_union[cls] += (pred_mask | true_mask).sum()

        val_loss /= len(val_loader)
        val_losses.append(val_loss)

        miou, mdice, ious, _ = compute_metrics(conf_intersection, conf_union, num_classes)
        cls2_iou = ious[2] if 2 < len(ious) else 0.0
        val_mious.append(miou)
        val_dices.append(mdice)
        val_cls2_ious.append(cls2_iou)
        scheduler.step(val_loss)

        cls_ious = ', '.join([f'cls{i}: {v:.4f}' for i, v in enumerate(ious)])
        print(f'Epoch [{epoch + 1}/{num_epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, mIoU: {miou:.4f}, mDice: {mdice:.4f}, cls2 IoU: {cls2_iou:.4f}')
        print(f'  IoU per class: {cls_ious}')

        # 根据类别2的IoU判断是否保存最优模型
        if cls2_iou > best_cls2_iou:
            best_cls2_iou = cls2_iou
            epoch_no_improve = 0
            torch.save(model.state_dict(), save_path)
            print(f'Epoch [{epoch + 1}/{num_epochs}], cls2 IoU: {cls2_iou:.4f} (Best Model Saved)')
        else:
            epoch_no_improve += 1
            print(f'Epoch [{epoch + 1}/{num_epochs}], cls2 IoU: {cls2_iou:.4f}')
            if epoch_no_improve >= patience:
                print(f'Early stopping at epoch {epoch + 1}')
                break

        # 保存检查点
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_losses': train_losses,
            'val_losses': val_losses,
            'val_mious': val_mious,
            'val_dices': val_dices,
            'val_cls2_ious': val_cls2_ious,
            'best_loss': best_loss,
            'best_cls2_iou': best_cls2_iou,
            'epoch_no_improve': epoch_no_improve,
            'scaler': scaler.state_dict() if scaler else None,
        }
        torch.save(checkpoint, checkpoint_path)

    # 保存结果 (CSV 和 图片)
    csv_filename = 'train_losses.csv'
    png_filename = 'loss_curve.png'
    
    csv_path = Path(output_dir) / csv_filename
    png_path = Path(output_dir) / png_filename

    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Epoch', 'Training Loss', 'Validation Loss', 'mIoU', 'mDice', 'cls2 IoU'])
        for i in range(len(train_losses)):
            writer.writerow([i + 1, train_losses[i], val_losses[i], val_mious[i], val_dices[i], val_cls2_ious[i]])

    # 绘制并保存损失曲线图片
    plt.figure(figsize=(14, 5))

    plt.subplot(1, 2, 1)
    plt.plot(range(1, len(train_losses) + 1), train_losses, label='Training Loss', color='blue', linewidth=2)
    plt.plot(range(1, len(val_losses) + 1), val_losses, label='Validation Loss', color='red', linewidth=2)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.title('Training and Validation Loss', fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)

    plt.subplot(1, 2, 2)
    plt.plot(range(1, len(val_mious) + 1), val_mious, label='mIoU', color='green', linewidth=2)
    plt.plot(range(1, len(val_dices) + 1), val_dices, label='mDice', color='orange', linewidth=2)
    plt.plot(range(1, len(val_cls2_ious) + 1), val_cls2_ious, label='cls2 IoU', color='red', linewidth=2, linestyle='--')
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Score', fontsize=12)
    plt.title('Validation Metrics', fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.savefig(png_path, dpi=300, bbox_inches='tight')
    plt.close()  
    
    print("训练完成！")
    print(f"模型保存至：{save_path}")
    print(f"损失曲线已保存至：{png_path}")
    
    return train_losses, val_mious, val_dices, val_cls2_ious, save_path, best_cls2_iou

def main():
    train()

if __name__ == '__main__':
    main()