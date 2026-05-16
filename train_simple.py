import os
import csv
import torch
from tqdm import tqdm
from torch import nn, optim
from torch.utils.data import DataLoader
from data import DataSetFFM
import matplotlib.pyplot as plt
import argparse
from Network.CCFANet import CFANet_C_1

# 默认参数
DEFAULT_BATCH_SIZE = 8
DEFAULT_LEARNING_RATE = 0.001
DEFAULT_NUM_EPOCHS = 80
DEFAULT_CHANNEL = 2
DEFAULT_OUTPUT_DIR = './outputs'
DEFAULT_PATIENCE = 10
DEFAULT_NUM_WORKERS = 8

def train(train_data_path='./dataset/train',
          val_data_path='./dataset/val',
          batch_size=DEFAULT_BATCH_SIZE,
          learning_rate=DEFAULT_LEARNING_RATE,
          num_epochs=DEFAULT_NUM_EPOCHS,
          channel=DEFAULT_CHANNEL,
          output_dir=DEFAULT_OUTPUT_DIR,
          patience=DEFAULT_PATIENCE,
          num_workers=DEFAULT_NUM_WORKERS,
          resume=False):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    print(f"结果保存至: {output_dir}")

    # 数据加载
    train_dataset = DataSetFFM(data_path=train_data_path, augment=True)
    val_dataset = DataSetFFM(data_path=val_data_path, augment=False)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=True, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=True, shuffle=False)

    # 模型初始化 (占位符)
    model = CFANet_C_1(resinc=channel).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=patience, verbose=True)
    criterion = nn.CrossEntropyLoss(weight=torch.tensor([1.0, 2.0, 7.0], dtype=torch.float32).to(device))

    save_path = os.path.join(output_dir, 'model_single.pth' if channel == 1 else 'model.pth')
    checkpoint_path = os.path.join(output_dir, 'checkpoint.pth.tar')

    # 初始化训练记录
    train_losses = []
    val_losses = []
    best_loss = float('inf')
    start_epoch = 0
    epoch_no_improve = 0

    # 加载检查点
    if resume and os.path.exists(checkpoint_path):
        print(f'加载检查点: {checkpoint_path}')
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        train_losses = checkpoint['train_losses']
        val_losses = checkpoint['val_losses']
        best_loss = checkpoint['best_loss']
        epoch_no_improve = checkpoint['epoch_no_improve']
        print(f"恢复至 Epoch {start_epoch}, 最佳 Loss: {best_loss:.4f}")
    else:
        print("从头开始训练")

    # 训练循环
    for epoch in range(start_epoch, num_epochs):
        model.train()
        train_loss = 0.0

        loop = tqdm(train_loader, leave=False, desc=f'Epoch [{epoch+1}/{num_epochs}]')
        for images, masks, thresholds, _ in loop:
            images, masks, thresholds = images.to(device), masks.to(device), thresholds.to(device)

            if channel == 1:
                inputs = images
            else:
                inputs = torch.cat((images, thresholds), dim=1)

            outputs = model(inputs)
            if isinstance(outputs, tuple):
                outputs = outputs[-1]

            masks = masks.squeeze(1).long()
            loss = criterion(outputs, masks)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss += loss.item()
            loop.set_postfix({'loss': f'{loss.item():.4f}'})

        train_loss /= len(train_loader)
        train_losses.append(train_loss)

        # 验证
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, masks, thresholds, _ in loop:
                images, masks, thresholds = images.to(device), masks.to(device), thresholds.to(device)

                if channel == 1:
                    inputs = images
                else:
                    inputs = torch.cat((images, thresholds), dim=1)

                outputs = model(inputs)
                if isinstance(outputs, tuple):
                    outputs = outputs[-1]

                masks = masks.squeeze(1).long()
                loss = criterion(outputs, masks)
                val_loss += loss.item()

        val_loss /= len(val_loader)
        val_losses.append(val_loss)
        scheduler.step(val_loss)

        print(f'Epoch [{epoch + 1}/{num_epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')

        # 保存最佳模型
        if val_loss < best_loss:
            best_loss = val_loss
            epoch_no_improve = 0
            torch.save(model.state_dict(), save_path)
            print(f'最佳模型已保存, Loss: {val_loss:.4f}')
        else:
            epoch_no_improve += 1
            if epoch_no_improve >= patience:
                print(f'早停于 Epoch {epoch + 1}')
                break

        # 保存检查点
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_losses': train_losses,
            'val_losses': val_losses,
            'best_loss': best_loss,
            'epoch_no_improve': epoch_no_improve
        }
        torch.save(checkpoint, checkpoint_path)

    # 保存结果
    csv_path = os.path.join(output_dir, 'train_losses.csv')
    png_path = os.path.join(output_dir, 'loss_curve.png')

    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Epoch', 'Training Loss', 'Validation Loss'])
        for i in range(len(train_losses)):
            writer.writerow([i + 1, train_losses[i], val_losses[i]])

    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(train_losses) + 1), train_losses, label='Training Loss', color='blue', linewidth=2)
    plt.plot(range(1, len(val_losses) + 1), val_losses, label='Validation Loss', color='red', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss Curve')
    plt.legend()
    plt.grid(True)
    plt.savefig(png_path, dpi=300, bbox_inches='tight')
    plt.close()

    print("训练完成！")
    print(f"模型保存至: {save_path}")
    print(f"损失曲线保存至: {png_path}")

    return train_losses, save_path, best_loss

def main():
    parser = argparse.ArgumentParser(description='训练脚本')
    parser.add_argument('--train_data_path', type=str, default='./dataset/train', help='训练数据路径')
    parser.add_argument('--val_data_path', type=str, default='./dataset/val', help='验证数据路径')
    parser.add_argument('--output_dir', type=str, default='./outputs', help='输出目录')
    parser.add_argument('--batch_size', type=int, default=DEFAULT_BATCH_SIZE, help='批大小')
    parser.add_argument('--learning_rate', type=float, default=DEFAULT_LEARNING_RATE, help='学习率')
    parser.add_argument('--num_epochs', type=int, default=DEFAULT_NUM_EPOCHS, help='训练轮数')
    parser.add_argument('--channel', type=int, default=DEFAULT_CHANNEL, help='通道数')
    parser.add_argument('--patience', type=int, default=DEFAULT_PATIENCE, help='早停耐心值')
    parser.add_argument('--num_workers', type=int, default=DEFAULT_NUM_WORKERS, help='数据加载器工作进程数')
    parser.add_argument('--resume', action='store_true', help='是否恢复训练')

    args = parser.parse_args()
    train(**vars(args))

if __name__ == '__main__':
    main()