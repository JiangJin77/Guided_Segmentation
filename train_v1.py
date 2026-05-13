import argparse
import csv
import logging
import os
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from data import DataSetFFM

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class Trainer:
    """训练器类，封装训练逻辑。"""

    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self._setup_directories()
        self._setup_model()
        self._setup_optimizer_and_scheduler()
        self._setup_loss()
        self._load_checkpoint()

    def _setup_directories(self):
        """设置输出目录。"""
        self.output_dir = Path(self.config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"结果保存至: {self.output_dir}")

    def _setup_model(self):
        """初始化模型。"""
        # 假设模型已定义，这里用占位符
        # self.model = CFANet_C_1(resinc=self.config.channel).to(self.device)
        self.model = nn.Identity()  # 占位符，实际使用时替换
        logger.info(f"模型初始化完成，使用设备: {self.device}")

    def _setup_optimizer_and_scheduler(self):
        """设置优化器和调度器。"""
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.config.learning_rate)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=self.config.patience, verbose=True
        )

    def _setup_loss(self):
        """设置损失函数。"""
        weights = torch.tensor([1.0, 2.0, 7.0], dtype=torch.float32).to(self.device)
        self.criterion = nn.CrossEntropyLoss(weight=weights)

    def _load_checkpoint(self):
        """加载检查点。"""
        self.checkpoint_path = self.output_dir / 'checkpoint.pth.tar'
        self.save_path = self.output_dir / ('model_single.pth' if self.config.channel == 1 else 'model.pth')

        self.train_losses = []
        self.val_losses = []
        self.best_loss = float('inf')
        self.start_epoch = 0
        self.epoch_no_improve = 0

        if self.config.resume and self.checkpoint_path.exists():
            logger.info(f'加载检查点: {self.checkpoint_path}')
            checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.start_epoch = checkpoint['epoch'] + 1
            self.train_losses = checkpoint['train_losses']
            self.val_losses = checkpoint['val_losses']
            self.best_loss = checkpoint['best_loss']
            self.epoch_no_improve = checkpoint['epoch_no_improve']
            logger.info(f"恢复至 Epoch {self.start_epoch}, 最佳 Loss: {self.best_loss:.4f}")
        else:
            logger.info("未找到检查点，从头开始训练")

    def _get_data_loaders(self):
        """获取数据加载器。"""
        train_dataset = DataSetFFM(data_path=self.config.train_data_path, augment=True)
        val_dataset = DataSetFFM(data_path=self.config.val_data_path, augment=False)

        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            num_workers=self.config.num_workers,
            pin_memory=True,
            shuffle=True
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.batch_size,
            num_workers=self.config.num_workers,
            pin_memory=True,
            shuffle=False
        )
        return train_loader, val_loader

    def _train_epoch(self, train_loader):
        """训练一个 epoch。"""
        self.model.train()
        total_loss = 0.0

        loop = tqdm(train_loader, leave=False, desc=f'Epoch [{self.current_epoch+1}/{self.config.num_epochs}]')
        for images, masks, thresholds, _ in loop:
            images, masks, thresholds = images.to(self.device), masks.to(self.device), thresholds.to(self.device)

            if self.config.channel == 1:
                inputs = images
            else:
                inputs = torch.cat((images, thresholds), dim=1)

            outputs = self.model(inputs)
            if isinstance(outputs, tuple):
                outputs = outputs[-1]

            masks = masks.squeeze(1).long()
            loss = self.criterion(outputs, masks)

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            total_loss += loss.item()
            loop.set_postfix({'loss': f'{loss.item():.4f}'})

        return total_loss / len(train_loader)

    def _validate_epoch(self, val_loader):
        """验证一个 epoch。"""
        self.model.eval()
        total_loss = 0.0

        with torch.no_grad():
            for images, masks, thresholds, _ in val_loader:
                images, masks, thresholds = images.to(self.device), masks.to(self.device), thresholds.to(self.device)

                if self.config.channel == 1:
                    inputs = images
                else:
                    inputs = torch.cat((images, thresholds), dim=1)

                outputs = self.model(inputs)
                if isinstance(outputs, tuple):
                    outputs = outputs[-1]

                masks = masks.squeeze(1).long()
                loss = self.criterion(outputs, masks)
                total_loss += loss.item()

        return total_loss / len(val_loader)

    def _save_checkpoint(self):
        """保存检查点。"""
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'best_loss': self.best_loss,
            'epoch_no_improve': self.epoch_no_improve
        }
        torch.save(checkpoint, self.checkpoint_path)

    def _save_results(self):
        """保存训练结果。"""
        # 保存 CSV
        csv_path = self.output_dir / 'train_losses.csv'
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Epoch', 'Training Loss', 'Validation Loss'])
            for i, (train_loss, val_loss) in enumerate(zip(self.train_losses, self.val_losses), 1):
                writer.writerow([i, train_loss, val_loss])

        # 保存损失曲线
        png_path = self.output_dir / 'loss_curve.png'
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, len(self.train_losses) + 1), self.train_losses, label='Training Loss', color='blue', linewidth=2)
        plt.plot(range(1, len(self.val_losses) + 1), self.val_losses, label='Validation Loss', color='red', linewidth=2)
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('Loss', fontsize=12)
        plt.title('Training and Validation Loss Curve', fontsize=14)
        plt.legend(fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.savefig(png_path, dpi=300, bbox_inches='tight')
        plt.close()

        logger.info(f"模型保存至: {self.save_path}")
        logger.info(f"损失曲线保存至: {png_path}")

    def train(self):
        """执行训练。"""
        train_loader, val_loader = self._get_data_loaders()

        for self.current_epoch in range(self.start_epoch, self.config.num_epochs):
            train_loss = self._train_epoch(train_loader)
            val_loss = self._validate_epoch(val_loader)

            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)

            self.scheduler.step(val_loss)

            logger.info(f'Epoch [{self.current_epoch + 1}/{self.config.num_epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')

            if val_loss < self.best_loss:
                self.best_loss = val_loss
                self.epoch_no_improve = 0
                torch.save(self.model.state_dict(), self.save_path)
                logger.info(f'最佳模型已保存，Loss: {val_loss:.4f}')
            else:
                self.epoch_no_improve += 1
                if self.epoch_no_improve >= self.config.patience:
                    logger.info(f'早停于 Epoch {self.current_epoch + 1}')
                    break

            self._save_checkpoint()

        self._save_results()
        logger.info("训练完成！")


def main():
    parser = argparse.ArgumentParser(description='训练脚本')
    parser.add_argument('--train_data_path', type=str, default='./dataset/train', help='训练数据路径')
    parser.add_argument('--val_data_path', type=str, default='./dataset/val', help='验证数据路径')
    parser.add_argument('--output_dir', type=str, default='./outputs', help='输出目录')
    parser.add_argument('--batch_size', type=int, default=8, help='批大小')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='学习率')
    parser.add_argument('--num_epochs', type=int, default=80, help='训练轮数')
    parser.add_argument('--channel', type=int, default=2, help='通道数')
    parser.add_argument('--patience', type=int, default=10, help='早停耐心值')
    parser.add_argument('--num_workers', type=int, default=8, help='数据加载器工作进程数')
    parser.add_argument('--resume', action='store_true', help='是否恢复训练')

    config = parser.parse_args()
    trainer = Trainer(config)
    trainer.train()


if __name__ == '__main__':
    main()