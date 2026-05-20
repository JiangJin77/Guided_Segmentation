"""
JJNet 预测结果可视化脚本
将预测的分割 mask 叠加到原始 CT 图像上，生成直观的分割效果图。

用法:
    python Tools/visualize.py --checkpoint outputs/JJNet/model.pth --data_dir /path/to/data --output_dir ./vis_results

依赖:
    pip install matplotlib numpy torch h5py
"""
import argparse
import logging
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from Network.JJNet import JJNet
from train_JJNet import JJNetDataSet, compute_dice, compute_iou

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def visualize_sample(
    image: np.ndarray,
    mask: np.ndarray,
    pred: np.ndarray,
    save_path: Path,
    sample_name: str = '',
    dice: float = None,
    iou: float = None,
    dpi: int = 200,
    prior_maps: list = None,
    diff_maps: list = None,
    gate_map: np.ndarray = None,
):
    """
    为单个样本生成可视化图，包含原始图像、GT、预测、以及 overlay 对比。

    Args:
        image: (H, W) 原始 CT 图像 (float32)
        mask: (H, W) Ground truth 二值 mask (0 或 1)
        pred: (H, W) 预测二值 mask (0 或 1)
        save_path: 输出 PNG 文件路径
        sample_name: 样本名称
        dice: 可选的 Dice 系数（显示在图标题中）
        iou: 可选的 IoU（显示在图标题中）
        dpi: 输出图片分辨率
        prior_maps: 先验图列表 (每个元素为 (H, W) float32)
        diff_maps: 不对称热力图列表 (每个元素为 (H, W) float32)
        gate_map: 门控图 (H, W) float32
    """
    # 归一化原始图像到 [0, 1] 用于显示
    img_min, img_max = image.min(), image.max()
    if img_max > img_min:
        image_display = (image - img_min) / (img_max - img_min)
    else:
        image_display = np.zeros_like(image)

    # 确定子图数量: 基础 4 张 + 额外 prior/diff
    num_extra = 0
    if prior_maps is not None:
        num_extra += len(prior_maps)
    if diff_maps is not None:
        num_extra += len(diff_maps)
    if gate_map is not None:
        num_extra += 1

    total_plots = 1 + 3 + num_extra  # raw + (gt, pred, overlay) + extras
    cols = min(4, total_plots)
    rows = (total_plots + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 5 * rows))
    axes = axes.flatten() if total_plots > 1 else [axes]

    plot_idx = 0

    # 1. 原始 CT 图像
    axes[plot_idx].imshow(image_display, cmap='gray')
    axes[plot_idx].set_title(f'{sample_name} - CT Image')
    axes[plot_idx].axis('off')
    plot_idx += 1

    # 2. Ground Truth
    axes[plot_idx].imshow(image_display, cmap='gray')
    axes[plot_idx].imshow(mask, cmap='Reds', alpha=0.5, vmin=0, vmax=1)
    axes[plot_idx].set_title(f'Ground Truth (lesion area: {mask.sum():.0f} px)')
    axes[plot_idx].axis('off')
    plot_idx += 1

    # 3. 预测结果
    axes[plot_idx].imshow(image_display, cmap='gray')
    axes[plot_idx].imshow(pred, cmap='Reds', alpha=0.5, vmin=0, vmax=1)
    title = f'Prediction'
    if dice is not None:
        title += f' | Dice: {dice:.4f}'
    if iou is not None:
        title += f' | IoU: {iou:.4f}'
    axes[plot_idx].set_title(title)
    axes[plot_idx].axis('off')
    plot_idx += 1

    # 4. GT vs Pred 对比 overlay
    overlay = np.zeros((*image.shape, 3), dtype=np.float32)
    overlay[..., 0] = image_display  # R channel
    overlay[..., 1] = image_display  # G channel
    overlay[..., 2] = image_display  # B channel
    # GT: green, Pred: red
    overlay[..., 1] = np.where(mask > 0, 0.8, overlay[..., 1])   # green for GT
    overlay[..., 0] = np.where(pred > 0, 0.8, overlay[..., 0])   # red for pred
    # overlap = yellow (both red and green)
    overlap = (mask > 0) & (pred > 0)
    overlay[overlap, 0] = 1.0  # yellow = R+G
    overlay[overlap, 1] = 1.0
    overlay[overlap, 2] = 0.0
    axes[plot_idx].imshow(overlay)
    axes[plot_idx].set_title('GT (green) vs Pred (red) | Overlap (yellow)')
    axes[plot_idx].axis('off')
    plot_idx += 1

    # 5. Prior maps
    if prior_maps is not None:
        for i, pmap in enumerate(prior_maps):
            if plot_idx >= len(axes):
                break
            axes[plot_idx].imshow(pmap, cmap='hot', vmin=0, vmax=1)
            axes[plot_idx].set_title(f'Prior Map {i}')
            axes[plot_idx].axis('off')
            plot_idx += 1

    # 6. Diff maps (asymmetry heatmaps)
    if diff_maps is not None:
        for i, dmap in enumerate(diff_maps):
            if plot_idx >= len(axes):
                break
            axes[plot_idx].imshow(dmap, cmap='coolwarm', vmin=0, vmax=1)
            axes[plot_idx].set_title(f'Asymmetry Map {i} ({dmap.shape})')
            axes[plot_idx].axis('off')
            plot_idx += 1

    # 7. Gate map
    if gate_map is not None:
        if plot_idx < len(axes):
            axes[plot_idx].imshow(gate_map, cmap='viridis', vmin=0, vmax=1)
            axes[plot_idx].set_title('Gate Map')
            axes[plot_idx].axis('off')
            plot_idx += 1

    # 隐藏未使用的子图
    for i in range(plot_idx, len(axes)):
        axes[i].axis('off')

    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
    plt.close(fig)


def inference_and_visualize(
    data_dir: str,
    checkpoint_path: str,
    output_dir: str,
    device: str = 'cuda',
    batch_size: int = 8,
    max_samples: int = None,
    channel: int = 64,
    prior_init_lower: float = 17.5,
    prior_init_upper: float = 22.0,
    prior_radii=None,
    num_refine_iters: int = 1,
):
    """
    加载模型，执行推理，可视化所有样本。

    Args:
        data_dir: 数据目录，需包含 images/ 和 masks/ 子目录
        checkpoint_path: 模型权重文件路径 (.pth)
        output_dir: 输出目录
        device: 运行设备
        batch_size: 推理批次大小
        max_samples: 最多处理的样本数 (None=全部)
        channel: 模型通道数
        prior_init_lower: 先验下界初始值
        prior_init_upper: 先验上界初始值
        prior_radii: 先验半径范围
        num_refine_iters: 先验细化迭代次数
    """
    if prior_radii is None:
        prior_radii = range(1, 4)

    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    logger.info(f'Using device: {device}')

    # 加载模型
    logger.info(f'Loading model from {checkpoint_path}...')
    model = JJNet(
        channel=channel,
        prior_init_lower=prior_init_lower,
        prior_init_upper=prior_init_upper,
        prior_radii=prior_radii,
        num_refine_iters=num_refine_iters,
        deep_supervision=True,
    ).to(device)

    state_dict = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()
    logger.info('Model loaded successfully')

    # 加载数据
    dataset = JJNetDataSet(data_dir, augment=False)
    if max_samples is not None and max_samples < len(dataset):
        dataset.data_list = dataset.data_list[:max_samples]
        dataset.total_len = len(dataset.data_list) * dataset.augment_factor
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 收集所有 metrics
    all_dice = []
    all_iou = []

    with torch.no_grad():
        for idx, (image, mask) in enumerate(tqdm(loader, desc='推理与可视化')):
            image = image.to(device)
            outputs = model(image)
            (edge, sal1, sal2, sal3,
             prior_maps, diff_maps, gate_map, lower, upper) = outputs

            pred = (torch.sigmoid(sal3) > 0.5).float()

            # 逐样本处理
            for batch_idx in range(image.size(0)):
                sample_idx = idx * batch_size + batch_idx
                if sample_idx >= len(dataset.data_list):
                    break

                base_name = dataset.data_list[sample_idx]['base_name']
                img_np = image[batch_idx, 0].cpu().numpy()
                mask_np = mask[batch_idx, 0].cpu().numpy().astype(np.uint8)
                pred_np = pred[batch_idx, 0].cpu().numpy().astype(np.uint8)

                d = compute_dice(pred[batch_idx:batch_idx+1], mask[batch_idx:batch_idx+1]).item()
                i = compute_iou(pred[batch_idx:batch_idx+1], mask[batch_idx:batch_idx+1]).item()
                all_dice.append(d)
                all_iou.append(i)

                # 收集 prior 和 diff maps
                prior_np = [p[batch_idx, 0].cpu().numpy() for p in prior_maps]
                diff_np = [d[batch_idx, 0].cpu().numpy() for d in diff_maps]
                gate_np = gate_map[batch_idx, 0].cpu().numpy()

                save_path = output_dir / f'{base_name}.png'
                visualize_sample(
                    image=img_np,
                    mask=mask_np,
                    pred=pred_np,
                    save_path=save_path,
                    sample_name=base_name,
                    dice=d,
                    iou=i,
                    prior_maps=prior_np,
                    diff_maps=diff_np,
                    gate_map=gate_np,
                )

    # 输出统计
    if all_dice:
        logger.info(f'Mean Dice: {np.mean(all_dice):.4f} ± {np.std(all_dice):.4f}')
        logger.info(f'Mean IoU: {np.mean(all_iou):.4f} ± {np.std(all_iou):.4f}')
        logger.info(f'Results saved to {output_dir}')


def main():
    parser = argparse.ArgumentParser(description='JJNet 预测可视化')
    parser.add_argument('--checkpoint', type=str, required=True, help='模型权重路径 (.pth)')
    parser.add_argument('--data_dir', type=str, required=True, help='数据目录 (包含 images/ 和 masks/)')
    parser.add_argument('--output_dir', type=str, default='./vis_results', help='输出目录')
    parser.add_argument('--device', type=str, default='cuda', help='运行设备')
    parser.add_argument('--batch_size', type=int, default=8, help='推理批次大小')
    parser.add_argument('--max_samples', type=int, default=None, help='最多处理的样本数')
    parser.add_argument('--channel', type=int, default=64, help='模型通道数')
    parser.add_argument('--num_refine_iters', type=int, default=1, help='先验细化迭代次数')
    args = parser.parse_args()

    inference_and_visualize(
        data_dir=args.data_dir,
        checkpoint_path=args.checkpoint,
        output_dir=args.output_dir,
        device=args.device,
        batch_size=args.batch_size,
        max_samples=args.max_samples,
        channel=args.channel,
        num_refine_iters=args.num_refine_iters,
    )


if __name__ == '__main__':
    main()