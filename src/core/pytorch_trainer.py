"""
PyTorch训练器模块
负责模型训练的核心功能
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import numpy as np
from pathlib import Path
import json
import time
import logging
from typing import Dict, List, Tuple, Optional, Callable
from tqdm import tqdm
import matplotlib.pyplot as plt

from .model_factory import ModelFactory
from src.utils.config_manager import config_manager

logger = logging.getLogger(__name__)


class ClothingDataset(Dataset):
    """服装图片数据集"""
    
    def __init__(self, data_dir: str, transform=None, classes=None):
        """
        初始化数据集
        
        Args:
            data_dir: 数据目录路径
            transform: 图像变换
            classes: 类别列表
        """
        self.data_dir = Path(data_dir)
        self.transform = transform
        self.classes = classes or ['主图', '细节', '吊牌']
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        
        # 收集所有图片文件
        self.samples = []
        self._load_samples()
        
        logger.info(f"数据集加载完成: {len(self.samples)} 个样本")
        for cls, count in self._count_samples().items():
            logger.info(f"  {cls}: {count} 张")
    
    def _load_samples(self):
        """加载样本数据"""
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
        
        for class_name in self.classes:
            class_dir = self.data_dir / class_name
            if not class_dir.exists():
                logger.warning(f"类别目录不存在: {class_dir}")
                continue
            
            class_idx = self.class_to_idx[class_name]
            for img_path in class_dir.iterdir():
                if img_path.suffix.lower() in image_extensions:
                    self.samples.append((str(img_path), class_idx))
    
    def _count_samples(self) -> Dict[str, int]:
        """统计各类别样本数量"""
        counts = {cls: 0 for cls in self.classes}
        for _, class_idx in self.samples:
            class_name = self.classes[class_idx]
            counts[class_name] += 1
        return counts
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, class_idx = self.samples[idx]
        
        # 加载图像
        image = Image.open(img_path).convert('RGB')
        
        # 应用变换
        if self.transform:
            image = self.transform(image)

        return image, class_idx


class EarlyStopping:
    """早停机制 - 当验证损失不再下降时停止训练"""

    def __init__(self, patience: int = 10, min_delta: float = 0.001):
        """
        Args:
            patience: 容忍轮数，验证损失连续多少轮未改善则停止
            min_delta: 最小改善量，小于此值视为未改善
        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None

    def __call__(self, val_loss: float) -> bool:
        """检查是否应该停止训练"""
        if self.best_loss is None:
            self.best_loss = val_loss
            return False

        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            return False

        self.counter += 1
        return self.counter >= self.patience


class ClothingTrainer:
    """服装分类模型训练器"""

    def __init__(self,
                 model_name: Optional[str] = None,
                 num_classes: Optional[int] = None,
                 device: str = 'auto',
                 input_size: Optional[int] = None,
                 amp_enabled: bool = False):
        """
        初始化训练器

        Args:
            model_name: 模型名称（默认读取 config/model_config.yaml）
            num_classes: 分类数量（默认读取 config/model_config.yaml）
            device: 计算设备
            input_size: 输入图像尺寸（默认读取 config/model_config.yaml）
            amp_enabled: 是否启用混合精度训练
        """
        model_settings = config_manager.get_model_settings()

        resolved_model_name = model_name or model_settings["name"]
        resolved_num_classes = int(num_classes or model_settings["num_classes"])
        resolved_input_size = int(input_size or model_settings["image_size"])

        if resolved_input_size <= 0:
            raise ValueError(f"输入尺寸必须大于0，当前值: {resolved_input_size}")

        self.model_name = ModelFactory.normalize_model_name(resolved_model_name)
        self.num_classes = resolved_num_classes
        self.device = self._setup_device(device)
        self.input_size = resolved_input_size

        # 混合精度训练
        self.amp_enabled = amp_enabled and torch.cuda.is_available()
        self.scaler = torch.cuda.amp.GradScaler() if self.amp_enabled else None

        # 训练状态
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.criterion = None

        # 训练历史
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'learning_rate': []
        }

        # 回调函数
        self.callbacks = []

        # 停止标志和暂停标志
        self.stop_flag = False
        self.pause_flag = False
        self.progress_callback = None  # 签名: (batch_idx, total_batches, loss, acc) -> None

        logger.info(f"训练器初始化完成:")
        logger.info(f"  模型: {self.model_name}")
        logger.info(f"  设备: {self.device}")
        logger.info(f"  类别数: {self.num_classes}")
        logger.info(f"  输入尺寸: {self.input_size}x{self.input_size}")
        logger.info(f"  混合精度: {'启用' if self.amp_enabled else '禁用'}")

        # 清理GPU内存
        self._cleanup_gpu_memory()
    
    def _cleanup_gpu_memory(self):
        """清理GPU内存"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            logger.info("GPU内存已清理")
    
    def _get_gpu_memory_info(self):
        """获取GPU内存信息"""
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1024**3  # GB
            cached = torch.cuda.memory_reserved() / 1024**3      # GB
            total = torch.cuda.get_device_properties(0).total_memory / 1024**3  # GB
            return f"GPU内存: {allocated:.1f}GB已分配, {cached:.1f}GB已缓存, {total:.1f}GB总计"
        return "GPU不可用"
    
    def _setup_device(self, device: str) -> torch.device:
        """设置计算设备"""
        if device == 'auto':
            if torch.cuda.is_available():
                device_obj = torch.device('cuda')
                logger.info(f"自动选择GPU: {torch.cuda.get_device_name(0)}")
            else:
                device_obj = torch.device('cpu')
                logger.info("GPU不可用，使用CPU")
        else:
            device_obj = torch.device(device)
            logger.info(f"使用指定设备: {device}")
        
        return device_obj
    
    def add_callback(self, callback: Callable):
        """添加训练回调函数"""
        self.callbacks.append(callback)
    
    def _call_callbacks(self, event: str, **kwargs):
        """调用回调函数"""
        for callback in self.callbacks:
            if hasattr(callback, event):
                getattr(callback, event)(**kwargs)
    
    def prepare_data(self, train_dir: str, val_dir: str = None, 
                    batch_size: int = 32, num_workers: int = 4) -> Tuple[DataLoader, DataLoader]:
        """
        准备训练数据
        
        Args:
            train_dir: 训练数据目录
            val_dir: 验证数据目录
            batch_size: 批次大小
            num_workers: 数据加载进程数
            
        Returns:
            训练和验证数据加载器
        """
        # 数据增强
        train_transform = transforms.Compose([
            transforms.Resize((self.input_size, self.input_size)),
            transforms.RandomHorizontalFlip(0.5),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        val_transform = transforms.Compose([
            transforms.Resize((self.input_size, self.input_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        # 创建数据集
        train_dataset = ClothingDataset(train_dir, transform=train_transform)
        
        if val_dir:
            val_dataset = ClothingDataset(val_dir, transform=val_transform)
        else:
            # 从训练集中分割验证集
            total_size = len(train_dataset)
            val_size = int(0.2 * total_size)
            train_size = total_size - val_size
            
            train_dataset, val_dataset = torch.utils.data.random_split(
                train_dataset, [train_size, val_size]
            )
            # 为验证集设置变换
            val_dataset.dataset.transform = val_transform
        
        # 创建数据加载器（训练时减少num_workers避免内存问题）
        safe_num_workers = min(num_workers, 2)  # 限制worker数量
        train_loader = DataLoader(
            train_dataset, 
            batch_size=batch_size, 
            shuffle=True, 
            num_workers=safe_num_workers,
            pin_memory=True if self.device.type == 'cuda' else False,
            persistent_workers=False  # 避免持久worker占用内存
        )
        
        val_loader = DataLoader(
            val_dataset, 
            batch_size=batch_size, 
            shuffle=False, 
            num_workers=safe_num_workers,
            pin_memory=True if self.device.type == 'cuda' else False,
            persistent_workers=False
        )
        
        logger.info(f"数据准备完成:")
        logger.info(f"  训练样本: {len(train_dataset)}")
        logger.info(f"  验证样本: {len(val_dataset)}")
        logger.info(f"  批次大小: {batch_size}")
        
        return train_loader, val_loader
    
    def build_model(self, pretrained: bool = True):
        """构建模型"""
        self.model = ModelFactory.create_model(
            self.model_name, 
            num_classes=self.num_classes, 
            pretrained=pretrained
        )
        self.model.to(self.device)
        
        # 损失函数
        self.criterion = nn.CrossEntropyLoss()
        
        logger.info("模型构建完成")
        return self.model
    
    def setup_optimizer(self, lr: float = 0.001, weight_decay: float = 1e-4):
        """设置优化器"""
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )
        
        # 学习率调度器
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=50, eta_min=1e-6
        )
        
        logger.info(f"优化器设置完成: lr={lr}, weight_decay={weight_decay}")
    
    def create_data_loaders(self, data_dir: str, batch_size: int = 32, 
                           val_split: float = 0.2, num_workers: int = 0) -> Tuple[DataLoader, DataLoader]:
        """
        创建数据加载器
        
        Args:
            data_dir: 数据目录
            batch_size: 批次大小
            val_split: 验证集比例
            num_workers: 数据加载进程数
            
        Returns:
            训练和验证数据加载器
        """
        # 数据增强
        train_transform = transforms.Compose([
            transforms.Resize((self.input_size, self.input_size)),
            transforms.RandomHorizontalFlip(0.5),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        val_transform = transforms.Compose([
            transforms.Resize((self.input_size, self.input_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        # 创建完整数据集
        full_dataset = ClothingDataset(data_dir, transform=train_transform)
        
        # 分割数据集
        total_size = len(full_dataset)
        val_size = int(val_split * total_size)
        train_size = total_size - val_size
        
        train_dataset, val_dataset = torch.utils.data.random_split(
            full_dataset, [train_size, val_size]
        )
        
        # 为验证集单独设置变换
        # 创建验证集的副本并设置不同的变换
        val_dataset_copy = ClothingDataset(data_dir, transform=val_transform)
        
        # 获取验证集的索引
        val_indices = val_dataset.indices
        val_samples = [val_dataset_copy.samples[i] for i in val_indices]
        val_dataset_copy.samples = val_samples
        
        # 创建数据加载器（训练时减少num_workers避免内存问题）
        safe_num_workers = min(num_workers, 2)  # 限制worker数量
        train_loader = DataLoader(
            train_dataset, 
            batch_size=batch_size, 
            shuffle=True, 
            num_workers=safe_num_workers,
            pin_memory=True if self.device.type == 'cuda' else False,
            persistent_workers=False  # 避免持久worker占用内存
        )
        
        val_loader = DataLoader(
            val_dataset_copy, 
            batch_size=batch_size, 
            shuffle=False, 
            num_workers=safe_num_workers,
            pin_memory=True if self.device.type == 'cuda' else False,
            persistent_workers=False
        )
        
        logger.info(f"数据加载器创建完成:")
        logger.info(f"  训练样本: {len(train_dataset)}")
        logger.info(f"  验证样本: {len(val_dataset)}")
        logger.info(f"  批次大小: {batch_size}")
        
        return train_loader, val_loader
    
    def train_epoch(self, train_loader: DataLoader) -> Tuple[float, float]:
        """训练一个epoch

        Returns:
            Tuple[float, float]: (loss, accuracy) 或 (None, None) 如果被停止
        """
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        total_batches = len(train_loader)

        # 显示训练开始时的GPU内存状态
        logger.info(f"训练开始: {self._get_gpu_memory_info()}")

        with tqdm(train_loader, desc="训练中") as pbar:
            for batch_idx, (images, labels) in enumerate(pbar):
                # 检查暂停和停止标志
                while getattr(self, 'pause_flag', False):
                    if self.stop_flag:
                        logger.info("训练被用户停止")
                        return None, None
                    time.sleep(0.5)

                if self.stop_flag:
                    logger.info("训练被用户停止")
                    return None, None

                images, labels = images.to(self.device), labels.to(self.device)

                self.optimizer.zero_grad()

                # 混合精度训练
                if self.amp_enabled:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(images)
                        loss = self.criterion(outputs, labels)
                    self.scaler.scale(loss).backward()
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    outputs = self.model(images)
                    loss = self.criterion(outputs, labels)
                    loss.backward()
                    self.optimizer.step()

                # 统计
                running_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

                current_acc = 100. * correct / total if total > 0 else 0

                # 调用进度回调 - 每个 batch 都调用以确保实时更新
                if self.progress_callback:
                    try:
                        self.progress_callback(batch_idx + 1, total_batches, loss.item(), current_acc)
                    except Exception as e:
                        logger.warning(f"进度回调错误: {e}")

                # 定期清理GPU内存
                if batch_idx % 50 == 0 and batch_idx > 0:
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

                # 更新进度条
                pbar.set_postfix({
                    'Loss': f'{loss.item():.4f}',
                    'Acc': f'{current_acc:.2f}%'
                })

        epoch_loss = running_loss / len(train_loader)
        epoch_acc = 100. * correct / total

        return epoch_loss, epoch_acc
    
    def validate_epoch(self, val_loader: DataLoader) -> Tuple[float, float]:
        """验证一个epoch

        Returns:
            Tuple[float, float]: (loss, accuracy) 或 (None, None) 如果被停止
        """
        # 检查停止标志
        if self.stop_flag:
            return None, None

        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            with tqdm(val_loader, desc="验证中") as pbar:
                for images, labels in pbar:
                    # 检查暂停和停止标志
                    while getattr(self, 'pause_flag', False):
                        if self.stop_flag:
                            return None, None
                        time.sleep(0.5)

                    if self.stop_flag:
                        return None, None

                    images, labels = images.to(self.device), labels.to(self.device)

                    outputs = self.model(images)
                    loss = self.criterion(outputs, labels)

                    running_loss += loss.item()
                    _, predicted = outputs.max(1)
                    total += labels.size(0)
                    correct += predicted.eq(labels).sum().item()

                    pbar.set_postfix({
                        'Loss': f'{loss.item():.4f}',
                        'Acc': f'{100.*correct/total:.2f}%'
                    })

        epoch_loss = running_loss / len(val_loader)
        epoch_acc = 100. * correct / total

        return epoch_loss, epoch_acc
    
    def train(self, 
              train_loader: DataLoader, 
              val_loader: DataLoader,
              epochs: int = 50,
              save_dir: str = 'models/checkpoints',
              save_best: bool = True) -> Dict:
        """
        训练模型
        
        Args:
            train_loader: 训练数据加载器
            val_loader: 验证数据加载器
            epochs: 训练轮数
            save_dir: 模型保存目录
            save_best: 是否保存最佳模型
            
        Returns:
            训练历史
        """
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        
        best_acc = 0.0
        start_time = time.time()
        
        self._call_callbacks('on_train_begin', epochs=epochs)
        
        for epoch in range(epochs):
            epoch_start = time.time()
            
            self._call_callbacks('on_epoch_begin', epoch=epoch)
            
            # 训练阶段
            train_loss, train_acc = self.train_epoch(train_loader)
            
            # 验证阶段
            val_loss, val_acc = self.validate_epoch(val_loader)
            
            # 更新学习率
            if self.scheduler:
                self.scheduler.step()
                current_lr = self.scheduler.get_last_lr()[0]
            else:
                current_lr = self.optimizer.param_groups[0]['lr']
            
            # 记录历史
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            self.history['learning_rate'].append(current_lr)
            
            # 保存最佳模型
            if save_best and val_acc > best_acc:
                best_acc = val_acc
                self.save_model(save_path / 'best_model.pth', epoch, val_acc)
                logger.info(f"💾 保存最佳模型: 验证准确率 {val_acc:.2f}%")
            
            # 定期保存检查点
            if (epoch + 1) % 10 == 0:
                self.save_model(save_path / f'checkpoint_epoch_{epoch+1}.pth', epoch, val_acc)
            
            epoch_time = time.time() - epoch_start
            
            # 打印进度
            logger.info(f"Epoch [{epoch+1}/{epochs}] - "
                       f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% - "
                       f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}% - "
                       f"LR: {current_lr:.6f} - Time: {epoch_time:.2f}s")
            
            self._call_callbacks('on_epoch_end', 
                               epoch=epoch, 
                               train_loss=train_loss,
                               train_acc=train_acc,
                               val_loss=val_loss,
                               val_acc=val_acc,
                               lr=current_lr)
        
        total_time = time.time() - start_time
        logger.info(f"🎉 训练完成! 总用时: {total_time/60:.2f}分钟, 最佳验证准确率: {best_acc:.2f}%")
        
        self._call_callbacks('on_train_end', best_acc=best_acc, total_time=total_time)
        
        return self.history
    
    def save_model(self, path: str, epoch: int, accuracy: float):
        """保存模型"""
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict() if self.optimizer else None,
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'model_name': self.model_name,
            'num_classes': self.num_classes,
            'input_size': self.input_size,
            'epoch': epoch,
            'accuracy': accuracy,
            'history': self.history,
            'timestamp': time.time()
        }
        
        torch.save(checkpoint, path)
        logger.info(f"模型已保存: {path}")
    
    def load_model(self, path: str):
        """加载模型"""
        try:
            checkpoint = torch.load(path, map_location=self.device)
            
            # 重建模型
            if not self.model:
                self.build_model(pretrained=False)
            
            # 加载模型权重
            self.model.load_state_dict(checkpoint['model_state_dict'])
            
            # 加载优化器状态（可选）
            if self.optimizer and 'optimizer_state_dict' in checkpoint:
                try:
                    self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                    logger.info("优化器状态已恢复")
                except Exception as e:
                    logger.warning(f"优化器状态恢复失败: {e}")
            
            # 加载调度器状态（可选）
            if self.scheduler and 'scheduler_state_dict' in checkpoint:
                try:
                    self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                    logger.info("学习率调度器状态已恢复")
                except Exception as e:
                    logger.warning(f"学习率调度器状态恢复失败: {e}")
            
            # 加载训练历史（可选）
            if 'history' in checkpoint:
                self.history = checkpoint['history']
                logger.info("训练历史已恢复")
            
            logger.info(f"模型已加载: {path}")
            logger.info(f"  Epoch: {checkpoint.get('epoch', 'unknown')}")
            accuracy = checkpoint.get('accuracy', 'unknown')
            if accuracy != 'unknown':
                logger.info(f"  准确率: {accuracy:.2f}%")
            else:
                logger.info(f"  准确率: {accuracy}")
            
            return checkpoint
            
        except Exception as e:
            logger.error(f"模型加载失败: {e}")
            raise
    
    def plot_history(self, save_path: str = None):
        """绘制训练历史"""
        if not self.history['train_loss']:
            logger.warning("没有训练历史数据")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        
        # 损失曲线
        axes[0, 0].plot(self.history['train_loss'], label='Train Loss')
        axes[0, 0].plot(self.history['val_loss'], label='Val Loss')
        axes[0, 0].set_title('Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid()
        
        # 准确率曲线
        axes[0, 1].plot(self.history['train_acc'], label='Train Acc')
        axes[0, 1].plot(self.history['val_acc'], label='Val Acc')
        axes[0, 1].set_title('Accuracy')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy (%)')
        axes[0, 1].legend()
        axes[0, 1].grid()
        
        # 学习率曲线
        axes[1, 0].plot(self.history['learning_rate'])
        axes[1, 0].set_title('Learning Rate')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Learning Rate')
        axes[1, 0].grid()
        
        # 验证准确率详细
        axes[1, 1].plot(self.history['val_acc'], 'g-', linewidth=2)
        axes[1, 1].set_title('Validation Accuracy Detail')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Accuracy (%)')
        axes[1, 1].grid()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"训练历史图表已保存: {save_path}")
        
        return fig


if __name__ == "__main__":
    # 测试代码
    import logging
    logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')
    
    print("🧪 测试训练器...")
    
    # 创建训练器
    trainer = ClothingTrainer(
        model_name='tf_efficientnetv2_s',
        num_classes=3,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    
    # 构建模型
    model = trainer.build_model()
    print(f"✅ 模型构建成功: {type(model).__name__}")
    
    # 设置优化器
    trainer.setup_optimizer(lr=0.001)
    print("✅ 优化器设置成功")
    
    print("✅ 训练器测试完成")
