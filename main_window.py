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
    QTableWidgetItem, QHeaderView
)
from PySide6.QtCore import Qt, QThread, QObject, Signal, QTimer, QSize, QSettings
from PySide6.QtGui import QPixmap, QFont, QIcon, QPalette, QColor

# 添加项目路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

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
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                print("🧹 GPU内存已清理")
            
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
                print(f"✅ 已加载基础模型: {base_model_path}")
            
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
                        print(f"🧹 Epoch {epoch}: GPU内存已清理")
                
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
                import os
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
            print(f"⏱️ 初始化完成，耗时: {init_time - start_time:.3f}秒")
            
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
            print(f"⏱️ 文件夹创建完成，耗时: {folder_time - folder_start:.3f}秒")
            
            # 🎯 智能批次大小优化 - 根据GPU显存和图片数量动态调整
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
            
            print(f"ClassificationWorker: ⭐ GPU优化模式 - 批次大小 {batch_size} (GPU: {gpu_memory_gb:.1f}GB) 处理 {total_images} 张图片")
            print(f"⏱️ 开始批次处理 - {datetime.now().strftime('%H:%M:%S')}")
            
            total_preprocess_time = 0
            total_inference_time = 0
            total_file_move_time = 0
            
            for batch_start in range(0, total_images, batch_size):
                batch_start_time = time.time()
                batch_end = min(batch_start + batch_size, total_images)
                batch_paths = self.image_paths[batch_start:batch_end]
                
                print(f"📦 处理批次 {batch_start//batch_size + 1}, 图片: {batch_start+1}-{batch_end}")
                
                # 更新进度
                progress = int((batch_end) * 100 / total_images)
                self.progress_updated.emit(progress, f"批量分类中... {batch_end}/{total_images}")
                
                # 批量预处理图像 - 多线程并行优化（增加线程数）
                preprocess_start = time.time()
                batch_tensors = []
                valid_paths = []
                
                # 🏆 最优预处理函数 - 经过验证的最佳性能版本
                def preprocess_single_image(image_path):
                    try:
                        # 使用PIL + 原生transform - 最佳性能平衡
                        image = Image.open(image_path).convert('RGB')
                        input_tensor = classifier.transform(image)
                        return image_path, input_tensor
                    except Exception as e:
                        return image_path, None, str(e)
                
                # � 最优线程配置 - 经过测试验证的最佳性能
                import concurrent.futures
                
                # 20线程 - 经过系统测试验证的最优配置 (29.48张/秒)
                optimal_workers = 20  # 最优20线程配置
                
                print(f"� 启用{optimal_workers}线程最优配置 (已验证最佳性能: 29.48张/秒)")
                
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
                print(f"⏱️ 批次预处理完成，{len(batch_tensors)}张图片，耗时: {preprocess_time:.3f}秒")
                
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
                    print(f"⏱️ GPU推理完成，{len(batch_tensors)}张图片，耗时: {inference_time:.3f}秒")
                    
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
                    print(f"⏱️ 批次文件移动完成，{len(valid_paths)}张图片，耗时: {file_move_time:.3f}秒")
                    print(f"📊 批次总耗时: {batch_total_time:.3f}秒 (预处理:{preprocess_time:.3f}s + 推理:{inference_time:.3f}s + 移动:{file_move_time:.3f}s)")
                    print(f"⚡ 平均每张图片: {batch_total_time/len(valid_paths):.3f}秒/张")
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
            print(f"🎉 分类任务完成！ - {datetime.now().strftime('%H:%M:%S')}")
            print(f"📊 总体性能统计:")
            print(f"   • 总耗时: {total_time:.3f}秒")
            print(f"   • 处理图片: {len(results)}张")
            print(f"   • 平均速度: {len(results)/total_time:.2f}张/秒")
            print(f"   • 预处理总耗时: {total_preprocess_time:.3f}秒 ({total_preprocess_time/total_time*100:.1f}%)")
            print(f"   • GPU推理总耗时: {total_inference_time:.3f}秒 ({total_inference_time/total_time*100:.1f}%)")
            print(f"   • 文件移动总耗时: {total_file_move_time:.3f}秒 ({total_file_move_time/total_time*100:.1f}%)")
            print(f"   • 其他耗时: {total_time-total_preprocess_time-total_inference_time-total_file_move_time:.3f}秒")
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
        
        # 创建中央部件
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # 创建布局
        layout = QVBoxLayout(central_widget)
        
        # 创建标题
        title_label = QLabel("JiLing 服装分类系统")
        title_label.setAlignment(Qt.AlignCenter)
        title_label.setFont(QFont("微软雅黑", 16, QFont.Bold))
        title_label.setStyleSheet("color: #333; margin: 10px; padding: 10px;")
        layout.addWidget(title_label)
        
        # 创建选项卡
        self.tab_widget = QTabWidget()
        layout.addWidget(self.tab_widget)
        
        # 创建各个选项卡
        self.create_classification_tab()
        self.create_training_tab()
        self.create_model_tab()
        self.create_settings_tab()
        
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
        memory_tip = QLabel("💡 内存优化提醒：批次大小16可避免GPU内存不足")
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
        """创建模型管理选项卡"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # 模型信息
        info_group = QGroupBox("支持的模型")
        info_layout = QVBoxLayout(info_group)
        
        factory = ModelFactory()
        models_text = QTextEdit()
        models_text.setReadOnly(True)
        
        models_info = "支持的预训练模型:\n\n"
        for model_name in factory.get_supported_models():
            models_info += f"• {model_name}\n"
        
        models_text.setPlainText(models_info)
        info_layout.addWidget(models_text)
        
        layout.addWidget(info_group)
        
        # GPU状态
        gpu_group = QGroupBox("系统状态")
        gpu_layout = QVBoxLayout(gpu_group)
        
        self.system_status_text = QTextEdit()
        self.system_status_text.setReadOnly(True)
        self.update_system_status()
        gpu_layout.addWidget(self.system_status_text)
        
        layout.addWidget(gpu_group)
        
        self.tab_widget.addTab(tab, "模型管理")
        
    def create_settings_tab(self):
        """创建设置选项卡"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # 配置编辑
        config_group = QGroupBox("配置设置")
        config_layout = QVBoxLayout(config_group)
        
        self.config_edit = QTextEdit()
        config_layout.addWidget(self.config_edit)
        
        # 配置控制
        config_control_layout = QHBoxLayout()
        
        load_config_btn = QPushButton("加载配置")
        load_config_btn.clicked.connect(self.load_config)
        config_control_layout.addWidget(load_config_btn)
        
        save_config_btn = QPushButton("保存配置")
        save_config_btn.clicked.connect(self.save_config)
        config_control_layout.addWidget(save_config_btn)
        
        reset_config_btn = QPushButton("重置配置")
        reset_config_btn.clicked.connect(self.reset_config)
        config_control_layout.addWidget(reset_config_btn)
        
        config_control_layout.addStretch()
        config_layout.addLayout(config_control_layout)
        
        layout.addWidget(config_group)
        layout.addStretch()
        
        self.tab_widget.addTab(tab, "系统设置")
    
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
            "model_name": "tf_efficientnetv2_s",
            "num_classes": 3,
            "class_names": ["主图", "细节", "吊牌"],
            "input_size": [224, 224],
            "device": "auto",
            "model_path": "models/JiLing_baiditu_1755873239.pth",
            "image_extensions": [".jpg", ".jpeg", ".png", ".bmp", ".tiff"]
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
            
            print(f"✅ 路径记忆加载完成:")
            print(f"  分类文件夹: {last_classification_folder}")
            print(f"  训练文件夹: {last_training_folder}")
            
        except Exception as e:
            print(f"⚠️ 加载记忆路径失败: {e}")
    
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
            print(f"⚠️ 保存路径失败: {e}")
    
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
