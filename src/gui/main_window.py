#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
服装分类系统 - 主GUI界面
支持图像分类和模型训练功能
"""
import sys
import os
import json
import time
from pathlib import Path
from threading import Thread
from datetime import datetime
from typing import List, Dict, Optional, Any

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QTabWidget, QLabel, QPushButton, QLineEdit, QTextEdit,
    QFileDialog, QProgressBar, QComboBox, QSpinBox, QDoubleSpinBox,
    QCheckBox, QGroupBox, QGridLayout, QListWidget, QListWidgetItem,
    QMessageBox, QSplitter, QFrame, QScrollArea, QTableWidget,
    QTableWidgetItem, QHeaderView, QTreeWidget, QTreeWidgetItem,
    QInputDialog, QDialog
)
from PySide6.QtCore import Qt, QThread, QObject, Signal, QTimer, QSize, QSettings
from PySide6.QtGui import QPixmap, QFont, QIcon, QPalette, QColor

# 添加项目路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# 可选依赖导入
try:
    import GPUtil
    GPU_UTIL_AVAILABLE = True
except ImportError:
    GPU_UTIL_AVAILABLE = False

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

from core.model_factory import ModelFactory
from core.pytorch_classifier import ClothingClassifier
from core.pytorch_trainer import ClothingTrainer


class TrainingWorker(QObject):
    """训练工作线程"""
    progress_updated = Signal(int, str, dict)  # progress, message, metrics
    training_completed = Signal(bool, str)  # success, message
    epoch_completed = Signal(int, dict)  # epoch, metrics
    
    def __init__(self, trainer_config, training_params):
        super().__init__()
        self.trainer_config = trainer_config
        self.training_params = training_params
        self.should_stop = False
        
    def start_training(self):
        """开始训练"""
        try:
            # 清理GPU内存
            self.progress_updated.emit(0, "清理GPU内存...", {})
            import torch
            import os  # 明确导入os模块
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                print("[OK] GPU memory cleared")

            # 创建训练器
            trainer = ClothingTrainer(**self.trainer_config)

            # 构建模型
            self.progress_updated.emit(5, "构建模型中...", {})
            model = trainer.build_model(pretrained=self.training_params.get('pretrained', True))

            # 加载基础模型（如果指定）
            base_model_path = self.training_params.get('base_model_path')
            if base_model_path and os.path.exists(base_model_path):
                self.progress_updated.emit(8, "加载基础模型...", {})
                trainer.load_model(base_model_path)
                print(f"[OK] Base model loaded: {base_model_path}")
            
            # 设置优化器
            self.progress_updated.emit(10, "设置优化器...", {})
            trainer.setup_optimizer(
                lr=self.training_params['learning_rate']
            )
            
            # 创建数据加载器
            self.progress_updated.emit(15, "准备数据集...", {})
            train_loader, val_loader = trainer.create_data_loaders(
                data_dir=self.training_params['data_path'],
                batch_size=self.training_params['batch_size'],
                val_split=self.training_params['val_split']
            )
            
            self.progress_updated.emit(20, "开始训练...", {})
            
            # 实际训练过程
            num_epochs = self.training_params['num_epochs']
            for epoch in range(num_epochs):
                if self.should_stop:
                    break
                
                # 每5个epoch清理一次GPU内存
                if epoch % 5 == 0 and epoch > 0:
                    import torch
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        print(f"[OK] Epoch {epoch}: GPU memory cleared")
                
                # 训练一个epoch
                train_loss, train_acc = trainer.train_epoch(train_loader)
                
                # 验证
                val_loss, val_acc = trainer.validate_epoch(val_loader)
                
                # 更新进度
                progress = 20 + (epoch + 1) * 75 // num_epochs
                
                # 当前指标
                metrics = {
                    'train_loss': train_loss,
                    'train_acc': train_acc,
                    'val_loss': val_loss,
                    'val_acc': val_acc,
                    'lr': trainer.optimizer.param_groups[0]['lr'] if trainer.optimizer else 0.001
                }
                
                message = f"训练中... Epoch {epoch+1}/{num_epochs}"
                self.progress_updated.emit(progress, message, metrics)
                self.epoch_completed.emit(epoch + 1, metrics)
                
                # 学习率调度
                if trainer.scheduler:
                    trainer.scheduler.step()
            
            if not self.should_stop:
                # 保存模型
                self.progress_updated.emit(95, "保存模型...", {})
                
                # 确保models目录存在
                os.makedirs("models", exist_ok=True)
                
                model_save_path = f"models/JiLing_baiditu_{int(time.time())}.pth"
                # 获取最后一个epoch的验证准确率
                final_metrics = trainer.history.get('val_acc', [0])
                final_acc = final_metrics[-1] if final_metrics else 0
                trainer.save_model(model_save_path, num_epochs, final_acc)
                
                # 清理GPU内存
                self.progress_updated.emit(98, "清理GPU内存...", {})
                self._cleanup_gpu_memory(trainer)
                
                self.progress_updated.emit(100, "训练完成！", {})
                self.training_completed.emit(True, f"模型训练成功完成，已保存到 {model_save_path}")
            else:
                # 训练被停止，也要清理内存
                self._cleanup_gpu_memory(trainer)
                self.training_completed.emit(False, "训练被用户停止")
                
        except Exception as e:
            self.training_completed.emit(False, f"训练错误: {str(e)}")
    
    def _cleanup_gpu_memory(self, trainer=None):
        """清理GPU内存"""
        try:
            import torch
            import gc
            
            # 删除训练器中的大对象
            if trainer:
                if hasattr(trainer, 'model') and trainer.model:
                    trainer.model.cpu()  # 移动到CPU
                    del trainer.model
                if hasattr(trainer, 'optimizer') and trainer.optimizer:
                    del trainer.optimizer
                if hasattr(trainer, 'scheduler') and trainer.scheduler:
                    del trainer.scheduler
                if hasattr(trainer, 'criterion') and trainer.criterion:
                    del trainer.criterion
            
            # 清理Python垃圾回收
            gc.collect()
            
            # 清空GPU缓存
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            
        except Exception as e:
            print(f"清理GPU内存时出错: {e}")
    
    def stop_training(self):
        """停止训练"""
        self.should_stop = True


class ClassificationWorker(QObject):
    """分类工作线程"""
    progress_updated = Signal(int, str)
    classification_completed = Signal(list)
    
    def __init__(self, image_paths, classifier=None, config_path="config.json"):
        super().__init__()
        self.image_paths = image_paths
        self.classifier = classifier
        self.config_path = config_path
        
    def start_classification(self):
        """批量分类并移动图片到对应文件夹"""
        start_time = time.time()
        print(f"ClassificationWorker: 开始分类任务 - {datetime.now().strftime('%H:%M:%S')}")
        try:
            import os
            import torch
            from pathlib import Path
            from PIL import Image
            import numpy as np
            
            classifier = self.classifier
            if classifier is None:
                raise Exception("未加载分类器")
            
            total_images = len(self.image_paths)
            results = []
            
            init_time = time.time()
            print(f"[TIME] Init done, time: {init_time - start_time:.3f}s")

            # 读取输出文件夹
            output_folder = None
            config_path = getattr(self, 'config_path', 'config.json')
            if os.path.exists(config_path):
                import json
                with open(config_path, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                output_folder = config.get("paths", {}).get("output_folder", None)

            if not output_folder:
                # 默认用图片所在文件夹
                output_folder = os.path.dirname(self.image_paths[0])

            output_folder = Path(output_folder)

            # 创建类别文件夹
            folder_start = time.time()
            for class_name in classifier.classes:
                (output_folder / class_name).mkdir(parents=True, exist_ok=True)
            folder_time = time.time()
            print(f"[TIME] Folder creation done, time: {folder_time - folder_start:.3f}s")

            # 智能批次大小优化 - 根据GPU显存和图片数量动态调整
            gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3 if torch.cuda.is_available() else 4

            # RTX 3060 12GB的优化批次大小策略
            if gpu_memory_gb >= 10:  # RTX 3060 12GB
                base_batch_size = 160  # 增大批次，提升GPU利用率
            elif gpu_memory_gb >= 6:  # RTX 3060 6GB
                base_batch_size = 128
            else:  # 其他GPU
                base_batch_size = 64

            # 根据图片总数调整 - 小批次处理小数据集更高效
            if total_images <= 50:
                batch_size = min(base_batch_size, total_images)
            else:
                batch_size = base_batch_size

            print(f"[GPU] Optimized mode - batch size {batch_size} (GPU: {gpu_memory_gb:.1f}GB) processing {total_images} images")
            print(f"[TIME] Batch processing started - {datetime.now().strftime('%H:%M:%S')}")

            total_preprocess_time = 0
            total_inference_time = 0
            total_file_move_time = 0

            for batch_start in range(0, total_images, batch_size):
                batch_start_time = time.time()
                batch_end = min(batch_start + batch_size, total_images)
                batch_paths = self.image_paths[batch_start:batch_end]

                print(f"[BATCH] Processing batch {batch_start//batch_size + 1}, images: {batch_start+1}-{batch_end}")
                
                # 更新进度
                progress = int((batch_end) * 100 / total_images)
                self.progress_updated.emit(progress, f"批量分类中... {batch_end}/{total_images}")
                
                # 批量预处理图像 - 多线程并行优化（增加线程数）
                preprocess_start = time.time()
                batch_tensors = []
                valid_paths = []
                
                # Optimal preprocess function - verified best performance version
                def preprocess_single_image(image_path):
                    try:
                        # 使用PIL + 原生transform - 最佳性能平衡
                        image = Image.open(image_path).convert('RGB')
                        input_tensor = classifier.transform(image)
                        return image_path, input_tensor
                    except Exception as e:
                        return image_path, None, str(e)

                # Optimal thread config - verified best performance
                import concurrent.futures

                # 20线程 - 经过系统测试验证的最优配置 (29.48张/秒)
                optimal_workers = 20  # 最优20线程配置

                print(f"[PERF] Using {optimal_workers} threads (verified best: 29.48 img/s)")
                
                with concurrent.futures.ThreadPoolExecutor(max_workers=optimal_workers) as executor:
                    parallel_results = list(executor.map(preprocess_single_image, batch_paths))
                
                # 收集成功处理的结果
                for result in parallel_results:
                    if len(result) == 2:  # 成功
                        image_path, tensor = result
                        batch_tensors.append(tensor)
                        valid_paths.append(image_path)
                    else:  # 失败
                        image_path, _, error = result
                        print(f"预处理失败 {image_path}: {error}")
                        results.append({
                            'path': image_path,
                            'result': {'predicted_class': 'error', 'confidence': 0.0, 'error': error}
                        })
                
                preprocess_end = time.time()
                preprocess_time = preprocess_end - preprocess_start
                total_preprocess_time += preprocess_time
                print(f"[TIME] Batch preprocess done, {len(batch_tensors)} images, time: {preprocess_time:.3f}s")
                
                if not batch_tensors:
                    continue
                
                # 高效GPU推理 - 专注于速度优化
                inference_start = time.time()
                try:
                    batch_tensor = torch.stack(batch_tensors).to(classifier.device, non_blocking=True)
                    
                    # 单轮高效推理，专注于速度而非GPU占用率
                    with torch.no_grad():
                        with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                            outputs = classifier.model(batch_tensor)
                            probabilities = torch.softmax(outputs, dim=1)
                            confidences, predicted = torch.max(probabilities, 1)
                    
                    inference_end = time.time()
                    inference_time = inference_end - inference_start
                    total_inference_time += inference_time
                    print(f"[TIME] GPU inference done, {len(batch_tensors)} images, time: {inference_time:.3f}s")
                    
                    # 立即处理批量结果，减少GPU内存占用时间
                    file_move_start = time.time()
                    for i, (image_path, confidence, predicted_idx) in enumerate(zip(valid_paths, confidences, predicted)):
                        predicted_class = classifier.classes[predicted_idx.item()]
                        confidence_score = confidence.item()
                        
                        # 移动文件
                        dest_folder = output_folder / predicted_class
                        dest_file = dest_folder / Path(image_path).name
                        
                        if not os.path.exists(dest_file):
                            os.rename(image_path, dest_file)
                        
                        # 构造结果
                        all_probs = probabilities[i].cpu().numpy()
                        class_probs = {
                            class_name: float(prob) 
                            for class_name, prob in zip(classifier.classes, all_probs)
                        }
                        
                        result = {
                            'predicted_class': predicted_class,
                            'confidence': confidence_score,
                            'class_probabilities': class_probs,
                            'image_path': str(dest_file)
                        }
                        
                        results.append({
                            'path': str(dest_file),
                            'result': result
                        })
                        
                        print(f"批量处理: {Path(image_path).name} -> {predicted_class} ({confidence_score:.2f})")
                    
                    file_move_end = time.time()
                    file_move_time = file_move_end - file_move_start
                    total_file_move_time += file_move_time
                    
                    batch_total_time = time.time() - batch_start_time
                    print(f"[TIME] Batch file move done, {len(valid_paths)} images, time: {file_move_time:.3f}s")
                    print(f"[STATS] Batch total: {batch_total_time:.3f}s (preprocess:{preprocess_time:.3f}s + inference:{inference_time:.3f}s + move:{file_move_time:.3f}s)")
                    print(f"[PERF] Average per image: {batch_total_time/len(valid_paths):.3f}s/img")
                    print("-" * 60)
                
                except Exception as e:
                    print(f"批量推理失败: {e}")
                    # 降级到单张处理
                    for image_path in valid_paths:
                        try:
                            predicted_class, confidence, result = classifier.predict_single(image_path)
                            # 移动文件
                            dest_folder = output_folder / predicted_class
                            dest_file = dest_folder / Path(image_path).name
                            if not os.path.exists(dest_file):
                                os.rename(image_path, dest_file)
                            results.append({
                                'path': str(dest_file),
                                'result': result
                            })
                        except Exception as e2:
                            results.append({
                                'path': image_path,
                                'result': {'predicted_class': 'error', 'confidence': 0.0, 'error': str(e2)}
                            })
            
            # 总体时间统计
            total_end_time = time.time()
            total_time = total_end_time - start_time
            
            print("=" * 60)
            print(f"[DONE] Classification complete! - {datetime.now().strftime('%H:%M:%S')}")
            print(f"[STATS] Overall performance:")
            print(f"   - Total time: {total_time:.3f}s")
            print(f"   - Images processed: {len(results)}")
            print(f"   - Average speed: {len(results)/total_time:.2f} img/s")
            print(f"   - Preprocess total: {total_preprocess_time:.3f}s ({total_preprocess_time/total_time*100:.1f}%)")
            print(f"   - GPU inference total: {total_inference_time:.3f}s ({total_inference_time/total_time*100:.1f}%)")
            print(f"   - File move total: {total_file_move_time:.3f}s ({total_file_move_time/total_time*100:.1f}%)")
            print(f"   - Other: {total_time-total_preprocess_time-total_inference_time-total_file_move_time:.3f}s")
            print("=" * 60)
            
            print(f"ClassificationWorker: 分类完成，处理了 {len(results)} 张图片")
            self.classification_completed.emit(results)
            
        except Exception as e:
            print(f"ClassificationWorker: 分类失败: {str(e)}")
            self.progress_updated.emit(0, f"分类失败: {str(e)}")
            self.classification_completed.emit([])


class MainWindow(QMainWindow):
    """主窗口"""
    
    def __init__(self):
        super().__init__()
        # 初始化设置存储
        self.settings = QSettings("JiLing", "ClothingClassifier")
        
        self.init_ui()
        self.load_config()
        self.training_worker = None
        self.training_thread = None
        self.classification_worker = None
        self.classification_thread = None
        self.current_classifier = None  # 当前加载的分类器
        
        # 加载记忆的路径
        self.load_remembered_paths()
        
    def init_ui(self):
        """初始化UI"""
        self.setWindowTitle("JiLing 服装分类系统 v2.0 (PyTorch)")
        self.setGeometry(100, 100, 1200, 800)
        
        # 设置样式
        self.setStyleSheet("""
            QMainWindow {
                background-color: #f5f5f5;
            }
            QTabWidget::pane {
                border: 1px solid #c0c0c0;
                background-color: white;
            }
            QTabBar::tab {
                background-color: #e0e0e0;
                padding: 8px 16px;
                margin: 2px;
                border-radius: 4px;
                color: #333333;
            }
            QTabBar::tab:selected {
                background-color: #4a90e2;
                color: white;
            }
            QPushButton {
                background-color: #4a90e2;
                color: white;
                border: none;
                padding: 8px 16px;
                border-radius: 4px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #357abd;
            }
            QPushButton:pressed {
                background-color: #2968a3;
            }
            QPushButton:disabled {
                background-color: #cccccc;
                color: #666666;
            }
            QLabel {
                color: #333333;
            }
            QLineEdit {
                background-color: white;
                border: 1px solid #ccc;
                padding: 5px;
                border-radius: 3px;
                color: #333333;
            }
            QGroupBox {
                font-weight: bold;
                color: #333333;
                border: 1px solid #ccc;
                border-radius: 5px;
                margin: 5px;
                padding-top: 10px;
            }
            QGroupBox::title {
                color: #333333;
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px 0 5px;
            }
            QTextEdit {
                background-color: white;
                color: #333333;
                border: 1px solid #ccc;
                border-radius: 3px;
            }
            QTableWidget {
                background-color: white;
                color: #333333;
                border: 1px solid #ccc;
                gridline-color: #ddd;
            }
            QTableWidget::item:selected {
                background-color: #4a90e2;
                color: white;
            }
            QComboBox {
                background-color: white;
                color: #333333;
                border: 1px solid #ccc;
                padding: 5px;
                border-radius: 3px;
            }
            QSpinBox, QDoubleSpinBox {
                background-color: white;
                color: #333333;
                border: 1px solid #ccc;
                padding: 5px;
                border-radius: 3px;
            }
            QCheckBox {
                color: #333333;
            }
        """)
        
        # 创建中央布局
        central_layout = QVBoxLayout()
        
        # 创建标题
        title_label = QLabel("JiLing 服装分类系统")
        title_label.setAlignment(Qt.AlignCenter)
        title_label.setFont(QFont("微软雅黑", 16, QFont.Bold))
        title_label.setStyleSheet("color: #333; margin: 10px; padding: 10px;")
        central_layout.addWidget(title_label)
        
        # 创建选项卡
        self.tab_widget = QTabWidget()
        central_layout.addWidget(self.tab_widget)
        
        # 创建各个选项卡
        self.create_classification_tab()
        self.create_training_tab()
        self.create_model_tab()
        
        # 设置中央布局
        central_widget = QWidget()
        central_widget.setLayout(central_layout)
        self.setCentralWidget(central_widget)
        
        # 创建底部状态栏和设置按钮
        self.create_bottom_bar()
        
        # 初始化系统状态组件
        self._init_system_status_components()
        
        # 更新系统状态
        self.update_system_status()
        
        # 加载自动加载设置
        self._load_auto_load_settings()
        
        # 启动时自动加载模型
        self._auto_load_model_on_startup()
        
    def create_bottom_bar(self):
        """创建底部状态栏和设置按钮"""
        # 创建状态栏
        self.status_bar = self.statusBar()
        
        # 创建设置按钮
        self.settings_button = QPushButton("Settings")
        self.settings_button.setFixedSize(80, 30)
        self.settings_button.setStyleSheet("""
            QPushButton {
                background-color: #4a90e2;
                color: white;
                border: none;
                border-radius: 4px;
                font-weight: bold;
                font-size: 12px;
            }
            QPushButton:hover {
                background-color: #357abd;
            }
            QPushButton:pressed {
                background-color: #2968a3;
            }
        """)
        self.settings_button.clicked.connect(self.show_settings_dialog)
        
        # 将设置按钮添加到状态栏右侧
        self.status_bar.addPermanentWidget(self.settings_button)
        
        # 添加一些状态信息
        self.status_label = QLabel("就绪")
        self.status_bar.addWidget(self.status_label)
        
    def _init_system_status_components(self):
        """初始化系统状态组件"""
        # 创建系统状态文本框
        self.system_status_text = QTextEdit()
        self.system_status_text.setReadOnly(True)
        self.system_status_text.setMaximumHeight(120)
        # 注意：这个组件不会被添加到UI中，但会在update_system_status中使用
        
        # 创建配置编辑器组件（用于配置管理功能）
        self.config_edit = QTextEdit()
        self.config_edit.setFont(QFont("Consolas", 10))
        # 注意：这个组件不会被添加到UI中，但会在配置管理方法中使用
        
    def show_settings_dialog(self):
        """显示设置对话框"""
        if not hasattr(self, 'settings_dialog'):
            self.create_settings_dialog()
        self.settings_dialog.show()
        self.settings_dialog.raise_()
        self.settings_dialog.activateWindow()
        
    def create_settings_dialog(self):
        """创建设置对话框"""
        from PySide6.QtWidgets import QDialog, QVBoxLayout, QHBoxLayout, QGroupBox, QLabel, QComboBox, QPushButton, QTabWidget
        
        self.settings_dialog = QDialog(self)
        self.settings_dialog.setWindowTitle("系统设置")
        self.settings_dialog.setModal(True)
        self.settings_dialog.resize(600, 400)
        
        layout = QVBoxLayout(self.settings_dialog)
        
        # 创建选项卡
        tab_widget = QTabWidget()
        
        # 基本信息选项卡
        basic_tab = self._create_basic_info_tab()
        tab_widget.addTab(basic_tab, "Basic Info")
        
        # 主题设置选项卡
        theme_tab = QWidget()
        theme_layout = QVBoxLayout(theme_tab)
        
        theme_group = QGroupBox("主题设置")
        theme_group_layout = QVBoxLayout(theme_group)
        
        # 主题选择
        theme_select_layout = QHBoxLayout()
        theme_select_layout.addWidget(QLabel("界面主题:"))
        self.theme_combo = QComboBox()
        self.theme_combo.addItems(["浅色主题", "深色主题"])
        self.theme_combo.setCurrentText("浅色主题")
        self.theme_combo.currentTextChanged.connect(self._apply_theme)
        theme_select_layout.addWidget(self.theme_combo)
        theme_select_layout.addStretch()
        theme_group_layout.addLayout(theme_select_layout)
        
        theme_layout.addWidget(theme_group)
        theme_layout.addStretch()
        
        tab_widget.addTab(theme_tab, "Theme")
        
        layout.addWidget(tab_widget)
        
        # 按钮区域
        button_layout = QHBoxLayout()
        button_layout.addStretch()
        
        apply_button = QPushButton("应用")
        apply_button.clicked.connect(self._apply_theme_settings)
        button_layout.addWidget(apply_button)
        
        close_button = QPushButton("关闭")
        close_button.clicked.connect(self.settings_dialog.close)
        button_layout.addWidget(close_button)
        
        layout.addLayout(button_layout)
        
        # 加载当前主题设置
        self.load_theme_settings()
        
    def _apply_theme_settings(self):
        """应用主题设置"""
        theme = self.theme_combo.currentText()
        self._apply_theme(theme)
        QMessageBox.information(self, "成功", f"主题 '{theme}' 已应用！")
        
    def load_theme_settings(self):
        """加载主题设置"""
        current_theme = self.settings.value("theme", "浅色主题")
        self.theme_combo.setCurrentText(current_theme)
        self._apply_theme(current_theme)
        
    def _apply_theme(self, theme):
        """应用主题"""
        if theme == "深色主题":
            self.apply_dark_theme()
        else:
            self.apply_light_theme()
            
        # 保存设置
        self.settings.setValue("theme", theme)
        
    def apply_light_theme(self):
        """应用浅色主题"""
        self.setStyleSheet("""
            QMainWindow {
                background-color: #f5f5f5;
            }
            QTabWidget::pane {
                border: 1px solid #c0c0c0;
                background-color: white;
            }
            QTabBar::tab {
                background-color: #e0e0e0;
                padding: 8px 16px;
                margin: 2px;
                border-radius: 4px;
                color: #333333;
            }
            QTabBar::tab:selected {
                background-color: #4a90e2;
                color: white;
            }
            QPushButton {
                background-color: #4a90e2;
                color: white;
                border: none;
                padding: 8px 16px;
                border-radius: 4px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #357abd;
            }
            QPushButton:pressed {
                background-color: #2968a3;
            }
            QLabel {
                color: #333333;
            }
            QLineEdit {
                background-color: white;
                border: 1px solid #ccc;
                padding: 5px;
                border-radius: 3px;
                color: #333333;
            }
            QGroupBox {
                font-weight: bold;
                color: #333333;
                border: 1px solid #ccc;
                border-radius: 5px;
                margin: 5px;
                padding-top: 10px;
            }
            QGroupBox::title {
                color: #333333;
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px 0 5px;
            }
            QTextEdit {
                background-color: white;
                color: #333333;
                border: 1px solid #ccc;
                border-radius: 3px;
            }
            QTableWidget {
                background-color: white;
                color: #333333;
                border: 1px solid #ccc;
                gridline-color: #ddd;
            }
            QTableWidget::item:selected {
                background-color: #4a90e2;
                color: white;
            }
            QComboBox {
                background-color: white;
                color: #333333;
                border: 1px solid #ccc;
                padding: 5px;
                border-radius: 3px;
            }
            QSpinBox, QDoubleSpinBox {
                background-color: white;
                color: #333333;
                border: 1px solid #ccc;
                padding: 5px;
                border-radius: 3px;
            }
            QCheckBox {
                color: #333333;
            }
        """)
        
    def apply_dark_theme(self):
        """应用深色主题"""
        self.setStyleSheet("""
            QMainWindow {
                background-color: #2b2b2b;
            }
            QTabWidget::pane {
                border: 1px solid #555;
                background-color: #3a3a3a;
            }
            QTabBar::tab {
                background-color: #4a4a4a;
                padding: 8px 16px;
                margin: 2px;
                border-radius: 4px;
                color: #ffffff;
            }
            QTabBar::tab:selected {
                background-color: #5a90e2;
                color: white;
            }
            QPushButton {
                background-color: #5a90e2;
                color: white;
                border: none;
                padding: 8px 16px;
                border-radius: 4px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #4a7abd;
            }
            QPushButton:pressed {
                background-color: #3a68a3;
            }
            QLabel {
                color: #ffffff;
            }
            QLineEdit {
                background-color: #4a4a4a;
                border: 1px solid #666;
                padding: 5px;
                border-radius: 3px;
                color: #ffffff;
            }
            QGroupBox {
                font-weight: bold;
                color: #ffffff;
                border: 1px solid #666;
                border-radius: 5px;
                margin: 5px;
                padding-top: 10px;
            }
            QGroupBox::title {
                color: #ffffff;
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px 0 5px;
            }
            QTextEdit {
                background-color: #4a4a4a;
                color: #ffffff;
                border: 1px solid #666;
                border-radius: 3px;
            }
            QTableWidget {
                background-color: #4a4a4a;
                color: #ffffff;
                border: 1px solid #666;
                gridline-color: #666;
            }
            QTableWidget::item:selected {
                background-color: #5a90e2;
                color: white;
            }
            QComboBox {
                background-color: #4a4a4a;
                color: #ffffff;
                border: 1px solid #666;
                padding: 5px;
                border-radius: 3px;
            }
            QSpinBox, QDoubleSpinBox {
                background-color: #4a4a4a;
                color: #ffffff;
                border: 1px solid #666;
                padding: 5px;
                border-radius: 3px;
            }
            QCheckBox {
                color: #ffffff;
            }
        """)
        
    def create_classification_tab(self):
        """创建分类选项卡"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # 文件选择部分
        file_group = QGroupBox("图像选择")
        file_layout = QVBoxLayout(file_group)
        
        # 单文件选择
        single_layout = QHBoxLayout()
        single_layout.addWidget(QLabel("单个文件:"))
        self.single_file_edit = QLineEdit()
        self.single_file_edit.setPlaceholderText("选择单个图像文件...")
        single_layout.addWidget(self.single_file_edit)
        
        single_browse_btn = QPushButton("浏览")
        single_browse_btn.clicked.connect(self.browse_single_file)
        single_layout.addWidget(single_browse_btn)
        file_layout.addLayout(single_layout)
        
        # 文件夹选择
        folder_layout = QHBoxLayout()
        folder_layout.addWidget(QLabel("文件夹:"))
        self.folder_edit = QLineEdit()
        self.folder_edit.setPlaceholderText("选择包含图像的文件夹...")
        folder_layout.addWidget(self.folder_edit)
        
        folder_browse_btn = QPushButton("浏览")
        folder_browse_btn.clicked.connect(self.browse_folder)
        folder_layout.addWidget(folder_browse_btn)
        
        # 添加"使用上次路径"按钮
        last_path_btn = QPushButton("上次路径")
        last_path_btn.setToolTip("使用上次选择的文件夹路径")
        last_path_btn.clicked.connect(self.use_last_classification_path)
        folder_layout.addWidget(last_path_btn)
        
        file_layout.addLayout(folder_layout)
        
        layout.addWidget(file_group)
        
        # 模型选择部分
        model_group = QGroupBox("模型选择")
        model_layout = QVBoxLayout(model_group)
        
        # 模型文件选择
        model_file_layout = QHBoxLayout()
        model_file_layout.addWidget(QLabel("模型文件:"))
        self.model_file_edit = QLineEdit()
        self.model_file_edit.setPlaceholderText("选择预训练模型文件 (.pth)...")
        model_file_layout.addWidget(self.model_file_edit)
        
        model_browse_btn = QPushButton("浏览")
        model_browse_btn.clicked.connect(self.browse_model_file)
        model_file_layout.addWidget(model_browse_btn)
        model_layout.addLayout(model_file_layout)
        
        # 模型信息显示
        model_info_layout = QHBoxLayout()
        self.model_status_label = QLabel("状态: 未加载模型")
        self.model_status_label.setStyleSheet("color: #666666; font-style: italic;")
        model_info_layout.addWidget(self.model_status_label)
        
        # 加载模型按钮
        self.load_model_btn = QPushButton("加载模型")
        self.load_model_btn.clicked.connect(self.load_model)
        self.load_model_btn.setEnabled(False)
        model_info_layout.addWidget(self.load_model_btn)
        
        # 使用默认模型按钮
        self.use_default_btn = QPushButton("使用默认模型")
        self.use_default_btn.clicked.connect(self.use_default_model)
        model_info_layout.addWidget(self.use_default_btn)
        
        model_layout.addLayout(model_info_layout)
        layout.addWidget(model_group)
        
        # 分类控制
        control_layout = QHBoxLayout()
        self.classify_btn = QPushButton("开始分类")
        self.classify_btn.clicked.connect(self.start_classification)
        control_layout.addWidget(self.classify_btn)
        
        self.clear_results_btn = QPushButton("清空结果")
        self.clear_results_btn.clicked.connect(self.clear_classification_results)
        control_layout.addWidget(self.clear_results_btn)
        
        control_layout.addStretch()
        layout.addLayout(control_layout)
        
        # 进度条
        self.classification_progress = QProgressBar()
        self.classification_progress.setVisible(False)
        layout.addWidget(self.classification_progress)
        
        # 结果显示
        results_group = QGroupBox("分类结果")
        results_layout = QVBoxLayout(results_group)
        
        self.results_table = QTableWidget()
        self.results_table.setColumnCount(4)
        self.results_table.setHorizontalHeaderLabels(["文件名", "分类结果", "置信度", "路径"])
        
        # 设置表格列宽
        header = self.results_table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(1, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(2, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(3, QHeaderView.Stretch)
        
        results_layout.addWidget(self.results_table)
        layout.addWidget(results_group)
        
        self.tab_widget.addTab(tab, "图像分类")
        
        return tab
        
    def create_training_tab(self):
        """创建训练选项卡"""
        tab = QWidget()
        layout = QHBoxLayout(tab)
        
        # 左侧参数设置
        params_widget = QWidget()
        params_layout = QVBoxLayout(params_widget)
        params_widget.setMaximumWidth(350)
        
        # 模型设置
        model_group = QGroupBox("模型设置")
        model_layout = QGridLayout(model_group)
        
        # 训练模式选择
        model_layout.addWidget(QLabel("训练模式:"), 0, 0)
        self.train_mode_combo = QComboBox()
        self.train_mode_combo.addItems([
            "从预训练权重开始", 
            "从已有模型继续训练",
            "Fine-tuning已有模型"
        ])
        self.train_mode_combo.currentTextChanged.connect(self.on_train_mode_changed)
        model_layout.addWidget(self.train_mode_combo, 0, 1)
        
        # 基础模型文件选择
        model_layout.addWidget(QLabel("基础模型:"), 1, 0)
        base_model_layout = QHBoxLayout()
        self.base_model_edit = QLineEdit()
        self.base_model_edit.setPlaceholderText("选择基础模型文件...")
        self.base_model_edit.setEnabled(False)  # 默认禁用
        base_model_layout.addWidget(self.base_model_edit)
        
        self.base_model_browse_btn = QPushButton("浏览")
        self.base_model_browse_btn.clicked.connect(self.browse_base_model)
        self.base_model_browse_btn.setEnabled(False)  # 默认禁用
        base_model_layout.addWidget(self.base_model_browse_btn)
        model_layout.addLayout(base_model_layout, 1, 1)
        
        model_layout.addWidget(QLabel("模型类型:"), 2, 0)
        self.model_combo = QComboBox()
        self.model_combo.addItems([
            "tf_efficientnetv2_s", "convnext_tiny", "resnet50",
            "vit_base_patch16_224", "swin_tiny_patch4_window7_224"
        ])
        model_layout.addWidget(self.model_combo, 2, 1)
        
        model_layout.addWidget(QLabel("预训练:"), 3, 0)
        self.pretrained_checkbox = QCheckBox()
        self.pretrained_checkbox.setChecked(True)
        model_layout.addWidget(self.pretrained_checkbox, 3, 1)
        
        params_layout.addWidget(model_group)
        
        # 训练参数
        train_group = QGroupBox("训练参数")
        train_layout = QGridLayout(train_group)
        
        train_layout.addWidget(QLabel("训练轮数:"), 0, 0)
        self.epochs_spin = QSpinBox()
        self.epochs_spin.setRange(1, 1000)
        self.epochs_spin.setValue(10)
        train_layout.addWidget(self.epochs_spin, 0, 1)
        
        train_layout.addWidget(QLabel("批次大小:"), 1, 0)
        self.batch_size_spin = QSpinBox()
        self.batch_size_spin.setRange(1, 256)
        self.batch_size_spin.setValue(16)  # 降低训练默认batch_size
        self.batch_size_spin.setToolTip("训练推荐16-32，避免GPU内存不足")
        train_layout.addWidget(self.batch_size_spin, 1, 1)
        
        train_layout.addWidget(QLabel("学习率:"), 2, 0)
        self.lr_spin = QDoubleSpinBox()
        self.lr_spin.setRange(0.0001, 1.0)
        self.lr_spin.setDecimals(6)
        self.lr_spin.setValue(0.001)
        train_layout.addWidget(self.lr_spin, 2, 1)
        
        train_layout.addWidget(QLabel("验证比例:"), 3, 0)
        self.val_split_spin = QDoubleSpinBox()
        self.val_split_spin.setRange(0.1, 0.5)
        self.val_split_spin.setDecimals(2)
        self.val_split_spin.setValue(0.2)
        train_layout.addWidget(self.val_split_spin, 3, 1)
        
        # 内存使用提醒
        memory_tip = QLabel("[TIP] Batch size 16 recommended to avoid GPU OOM")
        memory_tip.setStyleSheet("color: #666; font-size: 12px;")
        memory_tip.setWordWrap(True)
        train_layout.addWidget(memory_tip, 4, 0, 1, 2)
        
        params_layout.addWidget(train_group)
        
        # 数据设置
        data_group = QGroupBox("数据设置")
        data_layout = QVBoxLayout(data_group)
        
        data_path_layout = QHBoxLayout()
        data_path_layout.addWidget(QLabel("数据路径:"))
        self.data_path_edit = QLineEdit()
        self.data_path_edit.setPlaceholderText("选择训练数据文件夹...")
        data_path_layout.addWidget(self.data_path_edit)
        
        data_browse_btn = QPushButton("浏览")
        data_browse_btn.clicked.connect(self.browse_data_folder)
        data_path_layout.addWidget(data_browse_btn)
        data_layout.addLayout(data_path_layout)
        
        params_layout.addWidget(data_group)
        
        # 训练控制
        control_group = QGroupBox("训练控制")
        control_layout = QVBoxLayout(control_group)
        
        self.start_train_btn = QPushButton("开始训练")
        self.start_train_btn.clicked.connect(self.start_training)
        control_layout.addWidget(self.start_train_btn)
        
        self.stop_train_btn = QPushButton("停止训练")
        self.stop_train_btn.clicked.connect(self.stop_training)
        self.stop_train_btn.setEnabled(False)
        control_layout.addWidget(self.stop_train_btn)
        
        params_layout.addWidget(control_group)
        params_layout.addStretch()
        
        layout.addWidget(params_widget)
        
        # 右侧监控面板
        monitor_widget = QWidget()
        monitor_layout = QVBoxLayout(monitor_widget)
        
        # 进度显示
        progress_group = QGroupBox("训练进度")
        progress_layout = QVBoxLayout(progress_group)
        
        self.train_progress = QProgressBar()
        progress_layout.addWidget(self.train_progress)
        
        self.train_status_label = QLabel("准备就绪")
        self.train_status_label.setAlignment(Qt.AlignCenter)
        progress_layout.addWidget(self.train_status_label)
        
        monitor_layout.addWidget(progress_group)
        
        # 指标显示
        metrics_group = QGroupBox("训练指标")
        metrics_layout = QVBoxLayout(metrics_group)
        
        self.metrics_table = QTableWidget()
        self.metrics_table.setColumnCount(2)
        self.metrics_table.setHorizontalHeaderLabels(["指标", "数值"])
        self.metrics_table.setMaximumHeight(200)
        metrics_layout.addWidget(self.metrics_table)
        
        monitor_layout.addWidget(metrics_group)
        
        # 训练日志
        log_group = QGroupBox("训练日志")
        log_layout = QVBoxLayout(log_group)
        
        self.train_log = QTextEdit()
        self.train_log.setReadOnly(True)
        self.train_log.setMaximumHeight(200)
        log_layout.addWidget(self.train_log)
        
        monitor_layout.addWidget(log_group)
        
        layout.addWidget(monitor_widget)
        
        self.tab_widget.addTab(tab, "模型训练")
        
    def create_model_tab(self):
        """创建模型管理选项卡 - 重设计版本"""
        print("DEBUG: create_model_tab 被调用")
        tab = QWidget()
        layout = QVBoxLayout(tab)

        # 创建分割器
        splitter = QSplitter(Qt.Horizontal)
        layout.addWidget(splitter)

        # 左侧面板 - 模型文件管理器
        left_panel = self._create_model_file_manager()
        splitter.addWidget(left_panel)

        # 右侧面板 - 模型信息和工具
        right_panel = self._create_model_info_panel()
        splitter.addWidget(right_panel)

        # 设置分割器比例
        splitter.setSizes([400, 600])

        self.tab_widget.addTab(tab, "模型管理")

    def _create_model_file_manager(self):
        """创建模型文件管理器"""
        panel = QWidget()
        layout = QVBoxLayout(panel)

        # 标题
        title_label = QLabel("Model File Manager")
        title_label.setFont(QFont("微软雅黑", 12, QFont.Bold))
        layout.addWidget(title_label)

        # 工具栏
        toolbar = self._create_model_toolbar()
        layout.addWidget(toolbar)

        # 模型文件树形列表
        self.model_tree = QTreeWidget()
        self.model_tree.setHeaderLabels(["Filename", "Size", "Modified", "Status"])
        self.model_tree.setColumnWidth(0, 200)
        self.model_tree.setColumnWidth(1, 80)
        self.model_tree.setColumnWidth(2, 120)
        self.model_tree.setColumnWidth(3, 80)
        self.model_tree.itemDoubleClicked.connect(self._on_model_double_clicked)
        layout.addWidget(self.model_tree)

        # 刷新按钮
        refresh_btn = QPushButton("Refresh")
        refresh_btn.clicked.connect(self._refresh_model_list)
        layout.addWidget(refresh_btn)

        # 初始加载模型列表
        self._refresh_model_list()

        return panel

    def _create_model_toolbar(self):
        """创建模型管理工具栏"""
        toolbar = QWidget()
        layout = QHBoxLayout(toolbar)

        # 加载模型按钮
        load_btn = QPushButton("Load")
        load_btn.clicked.connect(self._load_selected_model)
        layout.addWidget(load_btn)

        # 删除模型按钮
        delete_btn = QPushButton("Delete")
        delete_btn.clicked.connect(self._delete_selected_model)
        layout.addWidget(delete_btn)

        # 重命名按钮
        rename_btn = QPushButton("Rename")
        rename_btn.clicked.connect(self._rename_selected_model)
        layout.addWidget(rename_btn)

        # 导出按钮
        export_btn = QPushButton("Export")
        export_btn.clicked.connect(self._export_selected_model)
        layout.addWidget(export_btn)

        layout.addStretch()
        return toolbar

    def _create_model_info_panel(self):
        """创建模型信息面板"""
        panel = QWidget()
        layout = QVBoxLayout(panel)

        # 创建选项卡式信息面板
        info_tabs = QTabWidget()

        # 基本信息选项卡
        basic_info_tab = self._create_basic_info_tab()
        info_tabs.addTab(basic_info_tab, "Basic Info")

        # 性能监控选项卡
        performance_tab = self._create_performance_tab()
        info_tabs.addTab(performance_tab, "Performance")

        # 优化工具选项卡
        optimization_tab = self._create_optimization_tab()
        info_tabs.addTab(optimization_tab, "Optimization")

        layout.addWidget(info_tabs)

        return panel

    def _create_basic_info_tab(self):
        """创建基本信息选项卡"""
        tab = QWidget()
        layout = QVBoxLayout(tab)

        # 当前模型信息
        current_model_group = QGroupBox("当前加载模型")
        current_layout = QVBoxLayout(current_model_group)

        self.current_model_info = QTextEdit()
        self.current_model_info.setReadOnly(True)
        self.current_model_info.setMaximumHeight(150)
        self.current_model_info.setPlainText("未加载模型")
        current_layout.addWidget(self.current_model_info)

        layout.addWidget(current_model_group)

        # 模型统计信息
        stats_group = QGroupBox("模型统计")
        stats_layout = QGridLayout(stats_group)

        self.total_models_label = QLabel("总模型数: 0")
        self.total_size_label = QLabel("总大小: 0 MB")
        self.recent_models_label = QLabel("最近使用: 无")

        stats_layout.addWidget(self.total_models_label, 0, 0)
        stats_layout.addWidget(self.total_size_label, 0, 1)
        stats_layout.addWidget(self.recent_models_label, 1, 0, 1, 2)

        layout.addWidget(stats_group)

        # 自动加载设置
        print("DEBUG: 创建自动加载控件")
        auto_load_group = QGroupBox("自动加载设置")
        auto_load_layout = QVBoxLayout(auto_load_group)

        # 启用自动加载
        self.auto_load_checkbox = QCheckBox("启动时自动加载模型")
        self.auto_load_checkbox.setChecked(True)  # 默认启用
        self.auto_load_checkbox.stateChanged.connect(self._on_auto_load_changed)
        auto_load_layout.addWidget(self.auto_load_checkbox)

        # 自动加载模型选择
        model_select_layout = QHBoxLayout()
        model_select_layout.addWidget(QLabel("自动加载模型:"))
        
        self.auto_load_model_combo = QComboBox()
        self.auto_load_model_combo.addItem("最新训练模型", "latest")
        self.auto_load_model_combo.addItem("最佳性能模型", "best")
        self.auto_load_model_combo.addItem("指定模型文件", "custom")
        self.auto_load_model_combo.setCurrentText("最新训练模型")
        self.auto_load_model_combo.currentTextChanged.connect(self._on_auto_load_model_changed)
        model_select_layout.addWidget(self.auto_load_model_combo)
        model_select_layout.addStretch()
        auto_load_layout.addLayout(model_select_layout)

        # 自定义模型路径（当选择指定模型文件时显示）
        self.custom_model_widget = QWidget()
        self.custom_model_layout = QHBoxLayout(self.custom_model_widget)
        self.custom_model_layout.addWidget(QLabel("模型路径:"))
        
        self.custom_model_edit = QLineEdit()
        self.custom_model_edit.setPlaceholderText("选择模型文件路径...")
        self.custom_model_layout.addWidget(self.custom_model_edit)
        
        custom_browse_btn = QPushButton("浏览")
        custom_browse_btn.clicked.connect(self._browse_custom_model)
        self.custom_model_layout.addWidget(custom_browse_btn)
        
        auto_load_layout.addWidget(self.custom_model_widget)
        self.custom_model_widget.setVisible(False)  # 默认隐藏

        # 保存设置按钮
        save_auto_load_btn = QPushButton("Save Settings")
        save_auto_load_btn.clicked.connect(self._save_auto_load_settings)
        auto_load_layout.addWidget(save_auto_load_btn)

        layout.addWidget(auto_load_group)

        # 系统状态
        system_group = QGroupBox("系统状态")
        system_layout = QVBoxLayout(system_group)

        self.system_status_text = QTextEdit()
        self.system_status_text.setReadOnly(True)
        self.system_status_text.setMaximumHeight(120)
        system_layout.addWidget(self.system_status_text)

        layout.addWidget(system_group)

        # 支持的模型列表
        supported_group = QGroupBox("支持的模型类型")
        supported_layout = QVBoxLayout(supported_group)

        self.supported_models_text = QTextEdit()
        self.supported_models_text.setReadOnly(True)
        self._update_supported_models()
        supported_layout.addWidget(self.supported_models_text)

        layout.addWidget(supported_group)

        return tab

    def _create_performance_tab(self):
        """创建性能监控选项卡"""
        tab = QWidget()
        layout = QVBoxLayout(tab)

        # 实时性能指标
        metrics_group = QGroupBox("实时性能指标")
        metrics_layout = QGridLayout(metrics_group)

        # 推理速度
        self.inference_speed_label = QLabel("推理速度: -- FPS")
        self.inference_speed_label.setStyleSheet("font-weight: bold; color: #2e7d32;")

        # 内存使用
        self.memory_usage_label = QLabel("内存使用: -- MB")
        self.memory_usage_label.setStyleSheet("font-weight: bold; color: #1976d2;")

        # GPU利用率
        self.gpu_usage_label = QLabel("GPU利用率: --%")
        self.gpu_usage_label.setStyleSheet("font-weight: bold; color: #f57c00;")

        # 温度
        self.temperature_label = QLabel("温度: --°C")
        self.temperature_label.setStyleSheet("font-weight: bold; color: #d32f2f;")

        metrics_layout.addWidget(self.inference_speed_label, 0, 0)
        metrics_layout.addWidget(self.memory_usage_label, 0, 1)
        metrics_layout.addWidget(self.gpu_usage_label, 1, 0)
        metrics_layout.addWidget(self.temperature_label, 1, 1)

        layout.addWidget(metrics_group)

        # 性能图表区域
        chart_group = QGroupBox("性能趋势图")
        chart_layout = QVBoxLayout(chart_group)

        self.performance_chart_placeholder = QLabel("Performance Chart Area\n(requires matplotlib)")
        self.performance_chart_placeholder.setAlignment(Qt.AlignCenter)
        self.performance_chart_placeholder.setStyleSheet("""
            QLabel {
                background-color: #f5f5f5;
                border: 2px dashed #ccc;
                border-radius: 5px;
                padding: 20px;
                color: #666;
            }
        """)
        chart_layout.addWidget(self.performance_chart_placeholder)

        layout.addWidget(chart_group)

        # 控制按钮
        control_layout = QHBoxLayout()

        self.start_monitoring_btn = QPushButton("Start Monitor")
        self.start_monitoring_btn.clicked.connect(self._start_performance_monitoring)
        control_layout.addWidget(self.start_monitoring_btn)

        self.stop_monitoring_btn = QPushButton("Stop Monitor")
        self.stop_monitoring_btn.clicked.connect(self._stop_performance_monitoring)
        self.stop_monitoring_btn.setEnabled(False)
        control_layout.addWidget(self.stop_monitoring_btn)

        control_layout.addStretch()
        layout.addLayout(control_layout)

        return tab

    def _create_optimization_tab(self):
        """创建优化工具选项卡"""
        tab = QWidget()
        layout = QVBoxLayout(tab)

        # 模型优化选项
        optimization_group = QGroupBox("模型优化工具")
        optimization_layout = QVBoxLayout(optimization_group)

        # 量化选项
        quant_layout = QHBoxLayout()
        quant_layout.addWidget(QLabel("量化精度:"))
        self.quantization_combo = QComboBox()
        self.quantization_combo.addItems(["FP32 (原始)", "FP16 (半精度)", "INT8 (量化)"])
        quant_layout.addWidget(self.quantization_combo)

        quantize_btn = QPushButton("Quantize")
        quantize_btn.clicked.connect(self._quantize_model)
        quant_layout.addWidget(quantize_btn)

        optimization_layout.addLayout(quant_layout)

        # 导出选项
        export_layout = QHBoxLayout()
        export_layout.addWidget(QLabel("导出格式:"))
        self.export_format_combo = QComboBox()
        self.export_format_combo.addItems(["ONNX", "TensorRT", "OpenVINO"])
        export_layout.addWidget(self.export_format_combo)

        export_btn = QPushButton("Export")
        export_btn.clicked.connect(self._export_model)
        export_layout.addWidget(export_btn)

        optimization_layout.addLayout(export_layout)

        # 压缩选项
        compress_layout = QHBoxLayout()
        compress_layout.addWidget(QLabel("压缩级别:"))
        self.compression_combo = QComboBox()
        self.compression_combo.addItems(["无压缩", "轻度压缩", "深度压缩"])
        compress_layout.addWidget(self.compression_combo)

        compress_btn = QPushButton("Compress")
        compress_btn.clicked.connect(self._compress_model)
        compress_layout.addWidget(compress_btn)

        optimization_layout.addLayout(compress_layout)

        layout.addWidget(optimization_group)

        # 优化结果显示
        result_group = QGroupBox("优化结果")
        result_layout = QVBoxLayout(result_group)

        self.optimization_result = QTextEdit()
        self.optimization_result.setReadOnly(True)
        self.optimization_result.setMaximumHeight(150)
        self.optimization_result.setPlainText("优化结果将在这里显示...")
        result_layout.addWidget(self.optimization_result)

        layout.addWidget(result_group)

        return tab

    def _create_performance_tab(self):
        """创建性能监控选项卡"""
        tab = QWidget()
        layout = QVBoxLayout(tab)

        # 实时性能指标
        metrics_group = QGroupBox("实时性能指标")
        metrics_layout = QGridLayout(metrics_group)

        # 推理速度
        self.inference_speed_label = QLabel("推理速度: -- FPS")
        self.inference_speed_label.setStyleSheet("font-weight: bold; color: #2e7d32;")

        # 内存使用
        self.memory_usage_label = QLabel("内存使用: -- MB")
        self.memory_usage_label.setStyleSheet("font-weight: bold; color: #1976d2;")

        # GPU利用率
        self.gpu_usage_label = QLabel("GPU利用率: --%")
        self.gpu_usage_label.setStyleSheet("font-weight: bold; color: #f57c00;")

        # 温度
        self.temperature_label = QLabel("温度: --°C")
        self.temperature_label.setStyleSheet("font-weight: bold; color: #d32f2f;")

        metrics_layout.addWidget(self.inference_speed_label, 0, 0)
        metrics_layout.addWidget(self.memory_usage_label, 0, 1)
        metrics_layout.addWidget(self.gpu_usage_label, 1, 0)
        metrics_layout.addWidget(self.temperature_label, 1, 1)

        layout.addWidget(metrics_group)

        # 性能图表区域
        chart_group = QGroupBox("性能趋势图")
        chart_layout = QVBoxLayout(chart_group)

        self.performance_chart_placeholder = QLabel("Performance Chart Area\n(requires matplotlib)")
        self.performance_chart_placeholder.setAlignment(Qt.AlignCenter)
        self.performance_chart_placeholder.setStyleSheet("""
            QLabel {
                background-color: #f5f5f5;
                border: 2px dashed #ccc;
                border-radius: 5px;
                padding: 20px;
                color: #666;
            }
        """)
        chart_layout.addWidget(self.performance_chart_placeholder)

        layout.addWidget(chart_group)

        # 控制按钮
        control_layout = QHBoxLayout()

        self.start_monitoring_btn = QPushButton("Start Monitor")
        self.start_monitoring_btn.clicked.connect(self._start_performance_monitoring)
        control_layout.addWidget(self.start_monitoring_btn)

        self.stop_monitoring_btn = QPushButton("Stop Monitor")
        self.stop_monitoring_btn.clicked.connect(self._stop_performance_monitoring)
        self.stop_monitoring_btn.setEnabled(False)
        control_layout.addWidget(self.stop_monitoring_btn)

        control_layout.addStretch()
        layout.addLayout(control_layout)

        return tab

    def _on_auto_load_changed(self):
        """刷新模型统计信息"""
        try:
            models_dir = Path("models")
            if not models_dir.exists():
                models_dir.mkdir(exist_ok=True)

            total_size = 0
            model_count = 0

            for model_file in models_dir.glob("*.pth"):
                if model_file.is_file():
                    file_size = model_file.stat().st_size / (1024 * 1024)  # MB
                    total_size += file_size
                    model_count += 1

            self.model_count_label.setText(str(model_count))
            self.total_size_label.setText(".1f")

        except Exception as e:
            QMessageBox.warning(self, "错误", f"刷新统计信息失败: {str(e)}")

    def _on_auto_load_model_changed(self):
        """打开模型目录"""
        models_dir = Path("models")
        if not models_dir.exists():
            models_dir.mkdir(exist_ok=True)

        import subprocess
        import platform
        try:
            if platform.system() == "Windows":
                subprocess.run(["explorer", str(models_dir)])
            elif platform.system() == "Darwin":  # macOS
                subprocess.run(["open", str(models_dir)])
            else:  # Linux
                subprocess.run(["xdg-open", str(models_dir)])
        except Exception as e:
            QMessageBox.warning(self, "错误", f"无法打开目录: {str(e)}")

    def _browse_custom_model(self):
        """打开数据目录"""
        data_dir = Path("data")
        if not data_dir.exists():
            data_dir.mkdir(exist_ok=True)

        import subprocess
        import platform
        try:
            if platform.system() == "Windows":
                subprocess.run(["explorer", str(data_dir)])
            elif platform.system() == "Darwin":  # macOS
                subprocess.run(["open", str(data_dir)])
            else:  # Linux
                subprocess.run(["xdg-open", str(data_dir)])
        except Exception as e:
            QMessageBox.warning(self, "错误", f"无法打开目录: {str(e)}")

    def _save_auto_load_settings(self):
        """打开输出目录"""
        output_dir = Path("outputs")
        if not output_dir.exists():
            output_dir.mkdir(exist_ok=True)

        import subprocess
        import platform
        try:
            if platform.system() == "Windows":
                subprocess.run(["explorer", str(output_dir)])
            elif platform.system() == "Darwin":  # macOS
                subprocess.run(["open", str(output_dir)])
            else:  # Linux
                subprocess.run(["xdg-open", str(output_dir)])
        except Exception as e:
            QMessageBox.warning(self, "错误", f"无法打开目录: {str(e)}")

    def _refresh_model_list(self):
        """刷新模型文件列表"""
        try:
            self.model_tree.clear()

            # 扫描models目录
            models_dir = Path("models")
            if not models_dir.exists():
                models_dir.mkdir(exist_ok=True)

            total_size = 0
            model_count = 0

            for model_file in models_dir.glob("*.pth"):
                if model_file.is_file():
                    # 获取文件信息
                    file_size = model_file.stat().st_size / (1024 * 1024)  # MB
                    mod_time = model_file.stat().st_mtime
                    mod_time_str = time.strftime("%Y-%m-%d %H:%M", time.localtime(mod_time))

                    # 创建树节点
                    item = QTreeWidgetItem(self.model_tree)
                    item.setText(0, model_file.name)
                    item.setText(1, ".1f")
                    item.setText(2, mod_time_str)
                    item.setText(3, "可用")

                    # 存储文件路径
                    item.setData(0, Qt.UserRole, str(model_file))

                    total_size += file_size
                    model_count += 1

            # 更新统计信息
            if hasattr(self, 'total_models_label'):
                self.total_models_label.setText(f"总模型数: {model_count}")
            if hasattr(self, 'total_size_label'):
                self.total_size_label.setText(".1f")

        except Exception as e:
            QMessageBox.warning(self, "错误", f"刷新模型列表失败: {str(e)}")

    def _on_model_double_clicked(self, item, column):
        """双击模型文件时的处理"""
        if item:
            model_path = item.data(0, Qt.UserRole)
            if model_path:
                self._load_model_from_path(model_path)

    def _load_selected_model(self):
        """加载选中的模型"""
        current_item = self.model_tree.currentItem()
        if current_item:
            model_path = current_item.data(0, Qt.UserRole)
            if model_path:
                self._load_model_from_path(model_path)
        else:
            QMessageBox.information(self, "提示", "请先选择一个模型文件")

    def _load_model_from_path(self, model_path):
        """从指定路径加载模型"""
        try:
            self.model_file_edit.setText(model_path)
            self.load_model()
            QMessageBox.information(self, "成功", f"模型加载成功: {Path(model_path).name}")
        except Exception as e:
            QMessageBox.warning(self, "错误", f"加载模型失败: {str(e)}")

    def _delete_selected_model(self):
        """删除选中的模型"""
        current_item = self.model_tree.currentItem()
        if not current_item:
            QMessageBox.information(self, "提示", "请先选择要删除的模型文件")
            return

        model_path = current_item.data(0, Qt.UserRole)
        if not model_path:
            return

        # 确认删除
        reply = QMessageBox.question(
            self, "确认删除",
            f"确定要删除模型文件吗？\n{Path(model_path).name}\n\n此操作不可恢复！",
            QMessageBox.Yes | QMessageBox.No
        )

        if reply == QMessageBox.Yes:
            try:
                Path(model_path).unlink()
                self._refresh_model_list()
                QMessageBox.information(self, "成功", "模型文件已删除")
            except Exception as e:
                QMessageBox.warning(self, "错误", f"删除失败: {str(e)}")

    def _rename_selected_model(self):
        """重命名选中的模型"""
        current_item = self.model_tree.currentItem()
        if not current_item:
            QMessageBox.information(self, "提示", "请先选择要重命名的模型文件")
            return

        model_path = current_item.data(0, Qt.UserRole)
        if not model_path:
            return

        old_name = Path(model_path).name
        new_name, ok = QInputDialog.getText(self, "重命名模型", "新文件名:", text=old_name)

        if ok and new_name and new_name != old_name:
            try:
                new_path = Path(model_path).parent / new_name
                if new_path.exists():
                    QMessageBox.warning(self, "错误", "目标文件名已存在")
                    return

                Path(model_path).rename(new_path)
                self._refresh_model_list()
                QMessageBox.information(self, "成功", "模型文件重命名成功")
            except Exception as e:
                QMessageBox.warning(self, "错误", f"重命名失败: {str(e)}")

    def _export_selected_model(self):
        """导出选中的模型"""
        current_item = self.model_tree.currentItem()
        if not current_item:
            QMessageBox.information(self, "提示", "请先选择要导出的模型文件")
            return

        model_path = current_item.data(0, Qt.UserRole)
        if not model_path:
            return

        # 选择导出路径
        export_path, _ = QFileDialog.getSaveFileName(
            self, "导出模型", "", "PyTorch模型 (*.pth);;所有文件 (*)"
        )

        if export_path:
            try:
                import shutil
                shutil.copy2(model_path, export_path)
                QMessageBox.information(self, "成功", f"模型已导出到: {export_path}")
            except Exception as e:
                QMessageBox.warning(self, "错误", f"导出失败: {str(e)}")

    def _update_supported_models(self):
        """更新支持的模型列表"""
        try:
            from core.model_factory import ModelFactory
            factory = ModelFactory()

            models_info = "Supported pretrained model architectures:\n\n"
            for model_name in factory.get_supported_models():
                models_info += f"- {model_name}\n"

            models_info += "\n[TIP] Different models have trade-offs between accuracy and speed"
            self.supported_models_text.setPlainText(models_info)
        except Exception as e:
            self.supported_models_text.setPlainText(f"Failed to load model info: {str(e)}")

    def _start_performance_monitoring(self):
        """开始性能监控"""
        self.start_monitoring_btn.setEnabled(False)
        self.stop_monitoring_btn.setEnabled(True)

        # 这里可以启动定时器来更新性能指标
        if not hasattr(self, 'performance_timer'):
            self.performance_timer = QTimer()
            self.performance_timer.timeout.connect(self._update_performance_metrics)

        self.performance_timer.start(1000)  # 每秒更新一次
        QMessageBox.information(self, "提示", "性能监控已启动")

    def _stop_performance_monitoring(self):
        """停止性能监控"""
        self.start_monitoring_btn.setEnabled(True)
        self.stop_monitoring_btn.setEnabled(False)

        if hasattr(self, 'performance_timer'):
            self.performance_timer.stop()

        QMessageBox.information(self, "提示", "性能监控已停止")

    def _update_performance_metrics(self):
        """更新性能指标"""
        try:
            import torch

            # GPU信息
            if torch.cuda.is_available():
                if GPU_UTIL_AVAILABLE:
                    gpu = GPUtil.getGPUs()[0]
                    self.gpu_usage_label.setText(".1f")
                    self.temperature_label.setText(f"温度: {gpu.temperature}°C")
                else:
                    self.gpu_usage_label.setText("GPU利用率: 需要GPUtil")
                    self.temperature_label.setText("温度: 需要GPUtil")

                # 内存使用
                memory_allocated = torch.cuda.memory_allocated() / 1024 / 1024  # MB
                self.memory_usage_label.setText(".1f")
            else:
                self.gpu_usage_label.setText("GPU利用率: N/A")
                self.temperature_label.setText("温度: N/A")
                self.memory_usage_label.setText("内存使用: N/A")

            # CPU信息
            if PSUTIL_AVAILABLE:
                cpu_percent = psutil.cpu_percent()
                memory_percent = psutil.virtual_memory().percent
            else:
                cpu_percent = 0
                memory_percent = 0

            # 这里可以显示推理速度等指标
            # 暂时显示占位符
            self.inference_speed_label.setText("推理速度: -- FPS")

        except Exception as e:
            print(f"更新性能指标失败: {e}")
            self.inference_speed_label.setText("推理速度: 错误")
            self.memory_usage_label.setText("内存使用: 错误")
            self.gpu_usage_label.setText("GPU利用率: 错误")
            self.temperature_label.setText("温度: 错误")

    def _quantize_model(self):
        """量化模型"""
        QMessageBox.information(self, "提示", "模型量化功能正在开发中...\n\n此功能将支持:\n• FP16半精度量化\n• INT8量化\n• 动态量化")

    def _export_model(self):
        """导出模型"""
        export_format = self.export_format_combo.currentText()
        QMessageBox.information(self, "提示", f"{export_format}导出功能正在开发中...\n\n此功能将支持:\n• ONNX格式导出\n• TensorRT优化\n• OpenVINO部署")

    def _compress_model(self):
        """压缩模型"""
        compression_level = self.compression_combo.currentText()
        QMessageBox.information(self, "提示", f"{compression_level}功能正在开发中...\n\n此功能将支持:\n• 模型权重压缩\n• 结构化剪枝\n• 知识蒸馏")

    def load_config(self):
        """加载配置"""
        try:
            config_path = "config.json"
            if os.path.exists(config_path):
                with open(config_path, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                self.config_edit.setPlainText(json.dumps(config, indent=2, ensure_ascii=False))
            else:
                self.config_edit.setPlainText("配置文件不存在")
        except Exception as e:
            QMessageBox.warning(self, "错误", f"加载配置失败: {str(e)}")
    
    def save_config(self):
        """保存配置"""
        try:
            config_text = self.config_edit.toPlainText()
            config = json.loads(config_text)
            
            with open("config.json", 'w', encoding='utf-8') as f:
                json.dump(config, f, indent=2, ensure_ascii=False)
                
            QMessageBox.information(self, "成功", "配置保存成功！")
        except Exception as e:
            QMessageBox.warning(self, "错误", f"保存配置失败: {str(e)}")
    
    def reset_config(self):
        """重置配置"""
        default_config = {
            "model_config": {
                "model_name": "tf_efficientnetv2_s",
                "model_path": "models/JiLing_baiditu_1755873239.pth",
                "num_classes": 3,
                "input_size": 384
            },
            "paths": {
                "input_folder": "data/test",
                "output_folder": "outputs",
                "log_folder": "logs"
            },
            "classification": {
                "batch_size": 32,
                "confidence_threshold": 0.5,
                "classes": ["主图", "细节", "吊牌"]
            },
            "processing": {
                "move_files": True,
                "save_statistics": True,
                "create_subfolders": True
            },
            "device": {
                "preferred": "auto",
                "fallback": "cpu"
            }
        }
        self.config_edit.setPlainText(json.dumps(default_config, indent=2, ensure_ascii=False))
    
    def update_system_status(self):
        """更新系统状态"""
        import torch
        
        status = f"""系统状态信息:

PyTorch版本: {torch.__version__}
CUDA可用: {'是' if torch.cuda.is_available() else '否'}
"""
        
        if torch.cuda.is_available():
            status += f"""CUDA版本: {torch.version.cuda}
GPU数量: {torch.cuda.device_count()}
当前GPU: {torch.cuda.get_device_name(0)}
GPU内存: {torch.cuda.get_device_properties(0).total_memory // 1024**3} GB
"""
        
        self.system_status_text.setPlainText(status)
    
    def browse_single_file(self):
        """浏览单个文件"""
        # 获取上次使用的路径
        last_folder = self.settings.value("last_classification_folder", "")
        start_dir = last_folder if last_folder and os.path.exists(last_folder) else ""
        
        file_path, _ = QFileDialog.getOpenFileName(
            self, "选择图像文件", start_dir, "图像文件 (*.jpg *.jpeg *.png *.bmp *.tiff)")
        if file_path:
            self.single_file_edit.setText(file_path)
            # 保存文件所在目录到设置中
            file_dir = os.path.dirname(file_path)
            self.settings.setValue("last_classification_folder", file_dir)
    
    def use_last_classification_path(self):
        """使用上次的分类路径"""
        last_folder = self.settings.value("last_classification_folder", "")
        if last_folder and os.path.exists(last_folder):
            self.folder_edit.setText(last_folder)
            QMessageBox.information(self, "路径已设置", f"已设置为上次使用的路径:\n{last_folder}")
        else:
            QMessageBox.warning(self, "路径不存在", "上次使用的路径不存在或未设置")
    
    def browse_folder(self):
        """浏览文件夹"""
        # 获取上次使用的路径
        last_folder = self.settings.value("last_classification_folder", "")
        start_dir = last_folder if last_folder and os.path.exists(last_folder) else ""
        
        folder_path = QFileDialog.getExistingDirectory(self, "选择图像文件夹", start_dir)
        if folder_path:
            self.folder_edit.setText(folder_path)
            # 保存到设置中
            self.settings.setValue("last_classification_folder", folder_path)
    
    def browse_data_folder(self):
        """浏览训练数据文件夹"""
        # 获取上次使用的路径
        last_folder = self.settings.value("last_training_folder", "")
        start_dir = last_folder if last_folder and os.path.exists(last_folder) else ""
        
        folder_path = QFileDialog.getExistingDirectory(self, "选择训练数据文件夹", start_dir)
        if folder_path:
            self.data_path_edit.setText(folder_path)
            # 保存到设置中
            self.settings.setValue("last_training_folder", folder_path)
    
    def browse_base_model(self):
        """浏览基础模型文件"""
        # 获取上次使用的路径
        last_model_dir = self.settings.value("last_model_folder", "models")
        start_dir = last_model_dir if last_model_dir and os.path.exists(last_model_dir) else "models"
        
        file_path, _ = QFileDialog.getOpenFileName(
            self, "选择基础模型文件", start_dir, "PyTorch模型文件 (*.pth *.pt)")
        if file_path:
            self.base_model_edit.setText(file_path)
            # 保存模型文件夹到设置中
            model_dir = os.path.dirname(file_path)
            self.settings.setValue("last_model_folder", model_dir)
    
    def on_train_mode_changed(self, mode):
        """训练模式改变时的处理"""
        if mode == "从预训练权重开始":
            # 禁用基础模型选择
            self.base_model_edit.setEnabled(False)
            self.base_model_browse_btn.setEnabled(False)
            self.base_model_edit.clear()
            self.pretrained_checkbox.setEnabled(True)
            self.pretrained_checkbox.setChecked(True)
        else:
            # 启用基础模型选择
            self.base_model_edit.setEnabled(True)
            self.base_model_browse_btn.setEnabled(True)
            if mode == "从已有模型继续训练":
                self.pretrained_checkbox.setEnabled(False)
                self.pretrained_checkbox.setChecked(False)
            else:  # Fine-tuning已有模型
                self.pretrained_checkbox.setEnabled(False)
                self.pretrained_checkbox.setChecked(False)
    
    def browse_model_file(self):
        """浏览模型文件"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "选择模型文件", "", "PyTorch模型文件 (*.pth *.pt)")
        if file_path:
            self.model_file_edit.setText(file_path)
            self.load_model_btn.setEnabled(True)
            self.model_status_label.setText("状态: 模型文件已选择，点击加载")
            self.model_status_label.setStyleSheet("color: #ff9500; font-style: italic;")
    
    def load_model(self):
        """加载模型"""
        model_path = self.model_file_edit.text().strip()
        if not model_path:
            QMessageBox.warning(self, "警告", "请选择模型文件！")
            return
        
        # 处理相对路径
        if not os.path.isabs(model_path):
            # 相对路径，相对于项目根目录
            project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
            model_path = os.path.join(project_root, model_path)
        
        if not os.path.exists(model_path):
            QMessageBox.warning(self, "警告", f"模型文件不存在：{model_path}")
            return
        
        try:
            # 验证模型文件
            import torch
            checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
            
            # 更新配置
            config_path = "config.json"
            if os.path.exists(config_path):
                with open(config_path, 'r', encoding='utf-8') as f:
                    config = json.load(f)
            else:
                config = {
                    "model_name": "tf_efficientnetv2_s",
                    "num_classes": 3,
                    "class_names": ["主图", "细节", "吊牌"],
                    "input_size": [224, 224],
                    "device": "auto",
                    "image_extensions": [".jpg", ".jpeg", ".png", ".bmp", ".tiff"]
                }
            
            # 更新模型路径
            config["model_config"]["model_path"] = model_path
            
            # 保存配置
            with open(config_path, 'w', encoding='utf-8') as f:
                json.dump(config, f, indent=2, ensure_ascii=False)
            
            # 创建新的分类器实例，使用正确的参数
            self.current_classifier = ClothingClassifier(
                model_path=model_path,
                device='auto',
                model_name=config.get("model_config", {}).get("model_name", "efficientnetv2_s")
            )
            
            self.model_status_label.setText(f"状态: 已加载 {os.path.basename(model_path)}")
            self.model_status_label.setStyleSheet("color: #28a745; font-weight: bold;")
            
            QMessageBox.information(self, "成功", f"模型加载成功！\n{os.path.basename(model_path)}")
            
        except Exception as e:
            self.model_status_label.setText("状态: 模型加载失败")
            self.model_status_label.setStyleSheet("color: #dc3545; font-style: italic;")
            self.current_classifier = None
            QMessageBox.critical(self, "错误", f"模型加载失败: {str(e)}")
    
    def use_default_model(self):
        """使用默认模型"""
        try:
            # 获取项目根目录
            project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))

            # 定义可能的模型路径和优先级
            possible_models = [
                # 优先使用最新的JiLing训练模型
                ("models/JiLing_baiditu_1755873239.pth", "最新训练的JiLing模型"),
                # 其他可能的JiLing模型（按时间戳降序）
                ("models/JiLing_baiditu_1755749592.pth", "JiLing训练模型"),
                # saved_models目录中的最佳模型
                ("models/saved_models/best_model.pth", "最佳训练模型"),
                # 默认模型
                ("models/clothing_classifier.pth", "默认分类模型"),
                # 演示模型
                ("models/demo_model.pth", "演示模型")
            ]

            # 查找存在的模型文件
            for model_path, model_desc in possible_models:
                model_full_path = os.path.join(project_root, model_path)
                if os.path.exists(model_full_path):
                    print(f"找到模型文件: {model_path} ({model_desc})")
                    self.model_file_edit.setText(model_path)
                    self.load_model()
                    return

            # 如果没有找到任何模型，询问是否创建演示模型
            reply = QMessageBox.question(
                self, "创建演示模型",
                "未找到任何可用的模型文件，是否创建演示模型用于测试？",
                QMessageBox.Yes | QMessageBox.No
            )
            if reply == QMessageBox.Yes:
                self.create_demo_model()
        except Exception as e:
            QMessageBox.critical(self, "错误", f"使用默认模型失败: {str(e)}")
    
    def create_demo_model(self):
        """创建演示模型"""
        try:
            from core.model_factory import ModelFactory
            import torch
            
            # 创建模型
            factory = ModelFactory()
            model = factory.create_model("tf_efficientnetv2_s", num_classes=3, pretrained=True)
            
            # 确保models目录存在
            os.makedirs("models", exist_ok=True)
            
            # 保存模型
            demo_model_path = "models/demo_model.pth"
            torch.save({
                'model_state_dict': model.state_dict(),
                'model_name': 'tf_efficientnetv2_s',
                'num_classes': 3,
                'class_names': ['主图', '细节', '吊牌']
            }, demo_model_path)
            
            self.model_file_edit.setText(demo_model_path)
            self.load_model()
            
        except Exception as e:
            QMessageBox.critical(self, "错误", f"创建演示模型失败: {str(e)}")
    
    def load_remembered_paths(self):
        """加载记忆的路径"""
        try:
            # 加载分类文件夹路径
            last_classification_folder = self.settings.value("last_classification_folder", "")
            if last_classification_folder and os.path.exists(last_classification_folder):
                self.folder_edit.setText(last_classification_folder)
            
            # 加载训练数据文件夹路径
            last_training_folder = self.settings.value("last_training_folder", "")
            if last_training_folder and os.path.exists(last_training_folder):
                if hasattr(self, 'data_path_edit'):
                    self.data_path_edit.setText(last_training_folder)
            
            print(f"[OK] Path memory loaded:")
            print(f"  Classification folder: {last_classification_folder}")
            print(f"  Training folder: {last_training_folder}")

        except Exception as e:
            print(f"[WARN] Failed to load path memory: {e}")
    
    def save_current_paths(self):
        """保存当前路径"""
        try:
            # 保存当前分类文件夹路径
            current_folder = self.folder_edit.text().strip()
            if current_folder:
                self.settings.setValue("last_classification_folder", current_folder)
            
            # 保存当前训练文件夹路径
            if hasattr(self, 'data_path_edit'):
                current_training_folder = self.data_path_edit.text().strip()
                if current_training_folder:
                    self.settings.setValue("last_training_folder", current_training_folder)

        except Exception as e:
            print(f"[WARN] Failed to save path: {e}")
    
    def closeEvent(self, event):
        """窗口关闭事件"""
        # 保存当前路径
        self.save_current_paths()
        event.accept()
    
    
    def start_classification(self):
        """开始分类"""
        print("start_classification 函数被调用")  # 调试信息
        
        # 保存当前使用的路径
        self.save_current_paths()
        
        # 收集图像路径
        image_paths = []
        
        # 单个文件
        single_file = self.single_file_edit.text().strip()
        print(f"单个文件路径: {single_file}")  # 调试信息
        if single_file and os.path.exists(single_file):
            image_paths.append(single_file)
        
        # 文件夹
        folder_path = self.folder_edit.text().strip()
        print(f"文件夹路径: {folder_path}")  # 调试信息
        if folder_path and os.path.exists(folder_path):
            extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
            for ext in extensions:
                image_paths.extend(Path(folder_path).glob(f"*{ext}"))
                image_paths.extend(Path(folder_path).glob(f"*{ext.upper()}"))
            
            # 去重，因为可能有重复的文件
            image_paths = list(set(image_paths))
        
        print(f"找到图像文件数量: {len(image_paths)}")  # 调试信息
        
        if not image_paths:
            QMessageBox.warning(self, "警告", "请选择要分类的图像文件或文件夹！")
            return
        
        # 检查是否有加载的模型
        print(f"当前分类器状态: {self.current_classifier}")  # 调试信息
        if self.current_classifier is None:
            print("分类器为空，询问是否使用默认模型")  # 调试信息
            reply = QMessageBox.question(
                self, "未加载模型",
                "未检测到已加载的模型，是否使用默认模型？",
                QMessageBox.Yes | QMessageBox.No
            )
            if reply == QMessageBox.Yes:
                print("用户选择使用默认模型")  # 调试信息
                self.use_default_model()
                print(f"使用默认模型后分类器状态: {self.current_classifier}")  # 调试信息
                if self.current_classifier is None:
                    print("默认模型加载失败，返回")  # 调试信息
                    return
            else:
                print("用户选择不使用默认模型，返回")  # 调试信息
                return
        
        print("开始准备分类线程")  # 调试信息
        
        # 转换为字符串路径
        image_paths = [str(p) for p in image_paths]
        print(f"转换后的图像路径数量: {len(image_paths)}")  # 调试信息
        
        # 禁用按钮，显示进度条
        print("禁用按钮，显示进度条")  # 调试信息
        self.classify_btn.setEnabled(False)
        self.classification_progress.setVisible(True)
        self.classification_progress.setValue(0)
        
        # 创建分类工作线程，传入当前分类器
        print("创建分类工作线程")  # 调试信息
        self.classification_worker = ClassificationWorker(image_paths, self.current_classifier)
        self.classification_thread = QThread()
        self.classification_worker.moveToThread(self.classification_thread)
        
        # 连接信号
        print("连接信号")  # 调试信息
        self.classification_worker.progress_updated.connect(self.update_classification_progress)
        self.classification_worker.classification_completed.connect(self.classification_completed)
        self.classification_thread.started.connect(self.classification_worker.start_classification)
        
        # 启动线程
        print("启动线程")  # 调试信息
        self.classification_thread.start()
        print("线程启动完成")  # 调试信息
    
    def update_classification_progress(self, progress, message):
        """更新分类进度"""
        self.classification_progress.setValue(progress)
        self.train_status_label.setText(message)
    
    def classification_completed(self, results):
        """分类完成"""
        # 更新结果表格
        self.results_table.setRowCount(len(results))
        
        for i, result in enumerate(results):
            file_path = result['path']
            classification_result = result['result']
            
            # 文件名
            filename_item = QTableWidgetItem(os.path.basename(file_path))
            self.results_table.setItem(i, 0, filename_item)
            
            # 分类结果
            if classification_result:
                class_name = classification_result.get('class_name', 'Unknown')
                confidence = classification_result.get('confidence', 0.0)
                
                class_item = QTableWidgetItem(class_name)
                confidence_item = QTableWidgetItem(f"{confidence:.3f}")
            else:
                class_item = QTableWidgetItem("分类失败")
                confidence_item = QTableWidgetItem("N/A")
            
            self.results_table.setItem(i, 1, class_item)
            self.results_table.setItem(i, 2, confidence_item)
            
            # 路径
            path_item = QTableWidgetItem(file_path)
            self.results_table.setItem(i, 3, path_item)
        
        # 恢复UI状态
        self.classify_btn.setEnabled(True)
        self.classification_progress.setVisible(False)
        self.train_status_label.setText("分类完成")
        
        # 清理线程
        self.classification_thread.quit()
        self.classification_thread.wait()
    
    def clear_classification_results(self):
        """清空分类结果"""
        self.results_table.setRowCount(0)
    
    def start_training(self):
        """开始训练"""
        # 检查数据路径
        data_path = self.data_path_edit.text().strip()
        if not data_path or not os.path.exists(data_path):
            QMessageBox.warning(self, "警告", "请选择有效的训练数据路径！")
            return
        
        # 检查基础模型路径（如果需要）
        train_mode = self.train_mode_combo.currentText()
        base_model_path = self.base_model_edit.text().strip()
        
        if train_mode != "从预训练权重开始":
            if not base_model_path or not os.path.exists(base_model_path):
                QMessageBox.warning(self, "警告", "请选择有效的基础模型文件！")
                return
        
        # 准备训练配置
        trainer_config = {
            'model_name': self.model_combo.currentText(),
            'num_classes': 3,  # 固定为3类
            'device': 'auto',
            'input_size': 580  # 与分类器保持一致的580x580甜蜜点尺寸
        }
        
        # 训练参数（这些不是构造函数参数）
        training_params = {
            'data_path': data_path,
            'num_epochs': self.epochs_spin.value(),
            'batch_size': self.batch_size_spin.value(),
            'learning_rate': self.lr_spin.value(),
            'val_split': self.val_split_spin.value(),
            'pretrained': self.pretrained_checkbox.isChecked(),
            'train_mode': train_mode,
            'base_model_path': base_model_path if train_mode != "从预训练权重开始" else None
        }
        
        # 更新UI状态
        self.start_train_btn.setEnabled(False)
        self.stop_train_btn.setEnabled(True)
        self.train_progress.setValue(0)
        self.train_status_label.setText("准备训练...")
        self.train_log.clear()
        
        # 创建训练工作线程，传递两个参数组
        self.training_worker = TrainingWorker(trainer_config, training_params)
        self.training_thread = QThread()
        self.training_worker.moveToThread(self.training_thread)
        
        # 连接信号
        self.training_worker.progress_updated.connect(self.update_training_progress)
        self.training_worker.training_completed.connect(self.training_completed)
        self.training_worker.epoch_completed.connect(self.epoch_completed)
        self.training_thread.started.connect(self.training_worker.start_training)
        
        # 启动线程
        self.training_thread.start()
        
        # 记录日志
        self.add_training_log("开始训练...")
        self.add_training_log(f"训练模式: {training_params['train_mode']}")
        if training_params.get('base_model_path'):
            self.add_training_log(f"基础模型: {os.path.basename(training_params['base_model_path'])}")
        self.add_training_log(f"模型: {trainer_config['model_name']}")
        self.add_training_log(f"训练轮数: {training_params['num_epochs']}")
        self.add_training_log(f"批次大小: {training_params['batch_size']}")
        self.add_training_log(f"学习率: {training_params['learning_rate']}")
        self.add_training_log(f"数据路径: {training_params['data_path']}")
    
    def stop_training(self):
        """停止训练"""
        if self.training_worker:
            self.training_worker.stop_training()
            self.add_training_log("正在停止训练...")
    
    def update_training_progress(self, progress, message, metrics):
        """更新训练进度"""
        self.train_progress.setValue(progress)
        self.train_status_label.setText(message)
        
        # 更新指标表格
        if metrics:
            self.metrics_table.setRowCount(len(metrics))
            for i, (key, value) in enumerate(metrics.items()):
                key_item = QTableWidgetItem(key)
                value_item = QTableWidgetItem(f"{value:.6f}" if isinstance(value, float) else str(value))
                self.metrics_table.setItem(i, 0, key_item)
                self.metrics_table.setItem(i, 1, value_item)
    
    def epoch_completed(self, epoch, metrics):
        """训练轮次完成"""
        log_message = f"Epoch {epoch} 完成 - "
        log_message += " | ".join([f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}" 
                                  for k, v in metrics.items()])
        self.add_training_log(log_message)
    
    def training_completed(self, success, message):
        """训练完成"""
        # 更新UI状态
        self.start_train_btn.setEnabled(True)
        self.stop_train_btn.setEnabled(False)
        
        # 记录日志
        self.add_training_log(f"训练{'成功' if success else '失败'}: {message}")
        
        # 显示消息
        if success:
            QMessageBox.information(self, "训练完成", message)
        else:
            QMessageBox.warning(self, "训练失败", message)
        
        # 清理线程
        if self.training_thread:
            self.training_thread.quit()
            self.training_thread.wait()
    
    def add_training_log(self, message):
        """添加训练日志"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        log_entry = f"[{timestamp}] {message}"
        self.train_log.append(log_entry)
    
    def _on_auto_load_changed(self, state):
        """自动加载设置改变"""
        enabled = state == Qt.Checked
        self.auto_load_model_combo.setEnabled(enabled)
        self.custom_model_edit.setEnabled(enabled)
        
        if hasattr(self, 'custom_model_widget'):
            # 只有当选择自定义模型时才显示自定义路径
            if enabled and self.auto_load_model_combo.currentData() == "custom":
                self.custom_model_widget.setVisible(True)
            else:
                self.custom_model_widget.setVisible(False)
    
    def _on_auto_load_model_changed(self, text):
        """自动加载模型选择改变"""
        model_type = self.auto_load_model_combo.currentData()
        if model_type == "custom":
            self.custom_model_widget.setVisible(True)
        else:
            self.custom_model_widget.setVisible(False)
    
    def _browse_custom_model(self):
        """浏览自定义模型文件"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "选择模型文件", "", "PyTorch模型文件 (*.pth);;所有文件 (*)"
        )
        if file_path:
            self.custom_model_edit.setText(file_path)
    
    def _save_auto_load_settings(self):
        """保存自动加载设置"""
        try:
            settings = {
                'auto_load_enabled': self.auto_load_checkbox.isChecked(),
                'auto_load_model_type': self.auto_load_model_combo.currentData(),
                'custom_model_path': self.custom_model_edit.text().strip()
            }
            
            # 保存到QSettings
            self.settings.setValue("auto_load_enabled", settings['auto_load_enabled'])
            self.settings.setValue("auto_load_model_type", settings['auto_load_model_type'])
            self.settings.setValue("custom_model_path", settings['custom_model_path'])
            
            QMessageBox.information(self, "成功", "自动加载设置已保存！")
            
        except Exception as e:
            QMessageBox.warning(self, "错误", f"保存设置失败: {str(e)}")
    
    def _load_auto_load_settings(self):
        """加载自动加载设置"""
        try:
            # 检查控件是否已创建
            if not hasattr(self, 'auto_load_checkbox'):
                print("自动加载控件尚未创建，跳过设置加载")
                print("调试信息: 检查对象属性...")
                attrs = [attr for attr in dir(self) if 'auto' in attr.lower()]
                print(f"包含'auto'的属性: {attrs}")
                return
            
            # 从QSettings加载设置
            auto_load_enabled = self.settings.value("auto_load_enabled", True, type=bool)
            auto_load_model_type = self.settings.value("auto_load_model_type", "latest")
            custom_model_path = self.settings.value("custom_model_path", "")
            
            # 应用设置到UI
            self.auto_load_checkbox.setChecked(auto_load_enabled)
            self.auto_load_model_combo.setCurrentText(
                "最新训练模型" if auto_load_model_type == "latest" else
                "最佳性能模型" if auto_load_model_type == "best" else
                "指定模型文件"
            )
            self.custom_model_edit.setText(custom_model_path)
            
            # 根据设置调整UI状态
            self._on_auto_load_changed(Qt.Checked if auto_load_enabled else Qt.Unchecked)
            
        except Exception as e:
            print(f"加载自动加载设置失败: {e}")
    
    def _auto_load_model_on_startup(self):
        """启动时自动加载模型"""
        try:
            print("DEBUG: 开始执行自动加载")
            # 检查控件是否已创建
            if not hasattr(self, 'auto_load_checkbox'):
                print("自动加载控件尚未创建，跳过自动加载")
                return
            
            if not self.auto_load_checkbox.isChecked():
                print("自动加载未启用")
                return
            
            model_type = self.auto_load_model_combo.currentData()
            print(f"DEBUG: 自动加载模型类型: {model_type}")
            
            if model_type == "latest":
                print("DEBUG: 调用 use_default_model()")
                # 自动加载最新训练的模型
                self.use_default_model()
                print(f"DEBUG: use_default_model() 执行完成, 分类器状态: {self.current_classifier is not None}")
            elif model_type == "best":
                # 自动加载最佳性能模型
                self._load_best_model()
            elif model_type == "custom":
                # 自动加载指定模型
                custom_path = self.custom_model_edit.text().strip()
                if custom_path and os.path.exists(custom_path):
                    self.model_file_edit.setText(custom_path)
                    self.load_model()
                    
        except Exception as e:
            print(f"自动加载模型失败: {e}")
    
    def _load_best_model(self):
        """加载最佳性能模型"""
        try:
            # 查找最佳模型（这里可以根据验证准确率或其他指标选择）
            models_dir = Path("models")
            if not models_dir.exists():
                return
            
            # 简单策略：选择最新的模型作为"最佳"模型
            model_files = list(models_dir.glob("*.pth"))
            if model_files:
                best_model = max(model_files, key=lambda x: x.stat().st_mtime)
                self.model_file_edit.setText(str(best_model))
                self.load_model()
                
        except Exception as e:
            print(f"加载最佳模型失败: {e}")


def main():
    """主函数"""
    app = QApplication(sys.argv)
    
    # 设置应用程序属性
    app.setApplicationName("JiLing服装分类系统")
    app.setApplicationVersion("2.0")
    app.setOrganizationName("JiLing")
    
    # 创建并显示主窗口
    window = MainWindow()
    window.show()
    
    # 运行应用程序
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
