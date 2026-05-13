import os
import csv
import torch
from tqdm import tqdm
from torch import nn, optim
from torch.utils.data import DataLoader
from data import DataSetFFM
import matplotlib.pyplot as plt
from CFANet_C_1 import CFANet_C_1


# 全局默认超参数
BATCH_SIZE = 8  
LEARNING_RATE = 0.001
NUM_EPOCHS = 80
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
CHANNEL = 2  
BASE_OUTPUT_DIR = '/root/autodl-tmp/py_01/寻参实验/Outputs' # 基础输出目录
EARLY_STOPPING_PATIENCE = 10  # 早停值

def train(num_classes=3, 
          channel=CHANNEL, 
          batch_size=BATCH_SIZE,
          learning_rate=LEARNING_RATE,
          num_epochs=NUM_EPOCHS,
          device=DEVICE,
          output_dir=BASE_OUTPUT_DIR,
          patience=EARLY_STOPPING_PATIENCE,
          resume=False):
    
    # 创建文件夹
    os.makedirs(output_dir, exist_ok=True)
    print(f"结果保存至: {output_dir}")

    # ==================================================================
    # 2. 数据加载 (保持不变)
    # ==================================================================
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
    model = CFANet_C_1(resinc=channel).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=patience, verbose=True)
    criterion = nn.CrossEntropyLoss(weight=torch.tensor([1.0, 2.0, 7.0], dtype=torch.float32).to(device))
    save_path = os.path.join(output_dir, 'model_single.pth') if channel == 1 else os.path.join(output_dir, 'model.pth')
    checkpoint_path = os.path.join(output_dir, 'checkpoint.pth.tar')
    
    # 训练记录初始化
    train_losses = []
    val_losses = []
    best_loss = float('inf')
    start_epoch = 0
    epoch_no_improve = 0

    # ===== 1.尝试加载检查点以恢复训练 =====
    if resume and os.path.exists(checkpoint_path):
        print(f'Loading checkpoint from {checkpoint_path}...')
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        train_losses = checkpoint['train_losses']
        val_losses = checkpoint['val_losses']
        best_loss = checkpoint['best_loss']
        epoch_no_improve = checkpoint['epoch_no_improve']
        print(f"成功恢复至Epoch：{start_epoch}, 最佳Loss: {best_loss:.4f}")
    else:
        print("未找到检查点，从头开始训练...")
    
    # ===== 2.训练循环 =====
    for epoch in range(start_epoch, num_epochs):
        model.train()
        train_loss = 0
        
        # 使用 desc 显示当前进度
        loop = tqdm(data_loader, leave=False, desc=f'Epoch[{epoch+1}/{num_epochs}]') #leave=False 训练完成后不保留进度条
        for i, (image, mask, threshold, _) in enumerate(loop):
            image, mask, threshold = image.to(device), mask.to(device), threshold.to(device)

            if channel == 1:   # 单通道
                input_tensor = image
            else:             # 双通道
                input_tensor = torch.cat((image, threshold), dim=1)
            
            # 假设模型返回多个值，取最后一个作为输出
            outputs = model(input_tensor)
            if isinstance(outputs, tuple):
                out_image = outputs[-1]
            else:
                out_image = outputs

            mask = mask.squeeze(1).long()
            loss = criterion(out_image, mask)
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()    
            
            train_loss += loss.item()
            loop.set_postfix({'loss': f'{loss.item():.4f}'})    # 更新进度条

        train_loss /= len(data_loader)
        train_losses.append(train_loss)

        # 验证阶段 
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for i, (image, mask, threshold, _) in enumerate(val_loader):
                image, mask, threshold = image.to(device), mask.to(device), threshold.to(device)

                if channel == 1:   # 单通道
                    input_tensor = image
                else:             # 双通道
                    input_tensor = torch.cat((image, threshold), dim=1)

                # 假设模型返回多个值，取最后一个作为输出    
                outputs = model(input_tensor)
                if isinstance(outputs, tuple):
                    out_image = outputs[-1]
                else:
                    out_image = outputs
                mask = mask.squeeze(1).long()
                loss = criterion(out_image, mask)   
                val_loss += loss.item()
        val_loss /= len(val_loader)
        val_losses.append(val_loss)
        scheduler.step(val_loss)  # 根据验证损失调整学习率
        print(f'Epoch [{epoch + 1}/{num_epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')


        # 判断是否保存最优模型
        if val_loss < best_loss:
            best_loss = val_loss
            epoch_no_improve = 0
            torch.save(model.state_dict(), save_path)
            print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {val_loss:.4f} (Best Model Saved)')
        else:
            epoch_no_improve += 1
            print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {val_loss:.4f}')
            # 判断是否早停
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
            'best_loss': best_loss,
            'epoch_no_improve': epoch_no_improve
        }
        torch.save(checkpoint, checkpoint_path)

    # 保存结果 (CSV 和 图片)
    csv_filename = 'train_losses.csv'
    png_filename = 'loss_curve.png'
    
    csv_path = os.path.join(output_dir, csv_filename)
    png_path = os.path.join(output_dir, png_filename)

    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Epoch', 'Training Loss', 'Validation Loss'])
        for i in range(len(train_losses)):
            writer.writerow([i + 1, train_losses[i], val_losses[i]])

    # 绘制并保存损失曲线图片
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(train_losses) + 1), train_losses, label='Training Loss', color='blue', linewidth=2)
    plt.plot(range(1, len(val_losses) + 1), val_losses, label='Validation Loss', color='red', linewidth=2)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.title(f'Training and Validation Loss Curve', fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.savefig(png_path, dpi=300, bbox_inches='tight')
    plt.close()  
    
    print("训练完成！")
    print(f"模型保存至：{save_path}")
    print(f"损失曲线已保存至：{png_path}")
    
    return train_losses, save_path, best_loss

def main():
    train()

if __name__ == '__main__':
    main()