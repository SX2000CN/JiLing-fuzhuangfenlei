"""
训练工作线程 - 在后台执行模型训练任务
"""

import os
import sys
import time
from pathlib import Path
from typing import Optional, Dict, Any

from PySide6.QtCore import QObject, Signal

# 获取项目根目录
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent

# 添加项目路径到 sys.path
_src_path = str(PROJECT_ROOT / "src")
if _src_path not in sys.path:
    sys.path.insert(0, _src_path)

# 后端模块延迟导入
torch = None
ClothingTrainer = None
BACKEND_AVAILABLE = False


def _init_backend():
    """初始化后端模块"""
    global torch, ClothingTrainer, BACKEND_AVAILABLE
    if BACKEND_AVAILABLE:
        return True
    try:
        import torch as _torch
        from core.pytorch_trainer import ClothingTrainer as _ClothingTrainer
        torch = _torch
        ClothingTrainer = _ClothingTrainer
        BACKEND_AVAILABLE = True
        return True
    except ImportError:
        return False


class TrainingWorker(QObject):
    """
    训练工作线程

    在后台执行模型训练任务，通过信号与主线程通信。

    Signals:
        progress_updated: (int, str, dict) - 进度百分比、状态消息、指标字典
        training_completed: (bool, str) - 是否成功、结果消息
        epoch_completed: (int, dict) - 当前轮次、指标字典
    """
    progress_updated = Signal(int, str, dict)
    training_completed = Signal(bool, str)
    epoch_completed = Signal(int, dict)

    def __init__(self, trainer_config: Dict[str, Any], training_params: Dict[str, Any]):
        super().__init__()
        self.trainer_config = trainer_config
        self.training_params = training_params
        self.should_stop = False
        self.trainer = None

    def start_training(self) -> None:
        """开始训练任务"""
        if not _init_backend():
            self.training_completed.emit(False, "后端模块不可用，无法训练")
            return

        try:
            # 强制清理GPU内存
            self.progress_updated.emit(0, "清理GPU内存...", {})
            if torch.cuda.is_available():
                import gc
                gc.collect()
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                gc.collect()
                torch.cuda.empty_cache()

            # 创建训练器
            self.progress_updated.emit(0, "创建训练器...", {})
            self.trainer = ClothingTrainer(**self.trainer_config)

            # 设置进度回调
            num_epochs = self.training_params['num_epochs']
            current_epoch = [0]
            total_batches_all = [0]

            def batch_progress_callback(batch_idx, total_batches, loss, acc):
                total_batches_all[0] = total_batches
                completed_batches = current_epoch[0] * total_batches + batch_idx
                total_batches_overall = num_epochs * total_batches
                progress = int((completed_batches / total_batches_overall) * 99)
                message = f"Epoch {current_epoch[0]+1}/{num_epochs} - Batch {batch_idx}/{total_batches}"
                self.progress_updated.emit(progress, message, {
                    'batch_loss': loss,
                    'batch_acc': acc / 100.0
                })

            self.trainer.progress_callback = batch_progress_callback

            # 构建模型
            self.progress_updated.emit(0, "构建模型中...", {})
            self.trainer.build_model(pretrained=self.training_params.get('pretrained', True))

            # 加载基础模型（如果指定）
            base_model_path = self.training_params.get('base_model_path')
            if base_model_path and os.path.exists(base_model_path):
                self.progress_updated.emit(0, "加载基础模型...", {})
                self.trainer.load_model(base_model_path)

            # 设置优化器
            self.progress_updated.emit(0, "设置优化器...", {})
            self.trainer.setup_optimizer(lr=self.training_params['learning_rate'])

            # 创建数据加载器
            self.progress_updated.emit(0, "准备数据集...", {})
            train_loader, val_loader = self.trainer.create_data_loaders(
                data_dir=self.training_params['data_path'],
                batch_size=self.training_params['batch_size'],
                val_split=self.training_params['val_split']
            )

            self.progress_updated.emit(0, "开始训练...", {})

            # 训练循环
            for epoch in range(num_epochs):
                current_epoch[0] = epoch

                if self.should_stop or self.trainer.stop_flag:
                    break

                if epoch % 5 == 0 and epoch > 0 and torch.cuda.is_available():
                    torch.cuda.empty_cache()

                train_loss, train_acc = self.trainer.train_epoch(train_loader)

                if train_loss is None:
                    self.should_stop = True
                    break

                val_loss, val_acc = self.trainer.validate_epoch(val_loader)

                if val_loss is None:
                    self.should_stop = True
                    break

                progress = int((epoch + 1) / num_epochs * 99)

                metrics = {
                    'train_loss': train_loss,
                    'train_acc': train_acc / 100.0,
                    'val_loss': val_loss,
                    'val_acc': val_acc / 100.0,
                    'lr': self.trainer.optimizer.param_groups[0]['lr'] if self.trainer.optimizer else 0.001
                }

                message = f"Epoch {epoch+1}/{num_epochs} 完成"
                self.progress_updated.emit(progress, message, metrics)
                self.epoch_completed.emit(epoch + 1, metrics)

                if self.trainer.scheduler:
                    self.trainer.scheduler.step()

            if not self.should_stop and not self.trainer.stop_flag:
                self.progress_updated.emit(99, "保存模型...", {})
                os.makedirs("models", exist_ok=True)

                model_save_path = f"models/JiLing_model_{int(time.time())}.pth"
                final_metrics = self.trainer.history.get('val_acc', [0])
                final_acc = final_metrics[-1] if final_metrics else 0
                self.trainer.save_model(model_save_path, num_epochs, final_acc)

                self._cleanup_gpu_memory(self.trainer)

                self.progress_updated.emit(100, "训练完成！", {})
                self.training_completed.emit(True, f"模型已保存到 {model_save_path}")
            else:
                self._cleanup_gpu_memory(self.trainer)
                self.training_completed.emit(False, "训练被用户停止")

        except Exception as e:
            self.training_completed.emit(False, f"训练错误: {str(e)}")

    def _cleanup_gpu_memory(self, trainer: Optional[Any] = None) -> None:
        """清理GPU内存"""
        try:
            import gc

            if trainer:
                for attr in ['model', 'optimizer', 'scheduler', 'criterion']:
                    if hasattr(trainer, attr) and getattr(trainer, attr):
                        obj = getattr(trainer, attr)
                        if hasattr(obj, 'cpu'):
                            obj.cpu()
                        delattr(trainer, attr)

            gc.collect()

            if torch and torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()

        except Exception:
            pass

    def stop_training(self) -> None:
        """停止训练"""
        self.should_stop = True
        if self.trainer:
            self.trainer.stop_flag = True
