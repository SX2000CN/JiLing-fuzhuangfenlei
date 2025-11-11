#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
现代化服装分类系统 - 主GUI界面
采用现代化设计风格，类似web版本的布局
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
    QLabel, QPushButton, QLineEdit, QTextEdit, QFileDialog, QProgressBar,
    QComboBox, QSpinBox, QDoubleSpinBox, QCheckBox, QGroupBox, QGridLayout,
    QListWidget, QListWidgetItem, QMessageBox, QSplitter, QFrame,
    QScrollArea, QTableWidget, QTableWidgetItem, QHeaderView,
    QStackedWidget, QSizePolicy
)
from PySide6.QtCore import Qt, QThread, QObject, Signal, QTimer, QSize, QSettings
from PySide6.QtGui import QPixmap, QFont, QIcon, QPalette, QColor, QPainter, QPen

# 添加项目路径
project_root = Path(__file__).parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# 导入核心模块
try:
    from src.core.model_factory import ModelFactory
    from src.core.pytorch_classifier import ClothingClassifier
    from src.core.pytorch_trainer import ClothingTrainer
except ImportError:
    from core.model_factory import ModelFactory
    from core.pytorch_classifier import ClothingClassifier
    from core.pytorch_trainer import ClothingTrainer


class ModernSidebar(QWidget):
    """现代化侧边栏 - Ant Design风格"""
    menuChanged = Signal(str)
    
    def __init__(self):
        super().__init__()
        self.current_menu = "classification"
        self.collapsed = False
        self.setup_ui()
        
    def setup_ui(self):
        self.setFixedWidth(250)
        self.setStyleSheet("""
            QWidget {
                background-color: #001529;
                color: rgba(255, 255, 255, 0.85);
            }
        """)
        
        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        
        # 标题区域 - 与web版本一致
        title_widget = QWidget()
        title_widget.setFixedHeight(64)
        title_widget.setStyleSheet("""
            background-color: #002140; 
            border-bottom: 1px solid #1d3344;
            color: rgba(255, 255, 255, 0.85);
        """)
        title_layout = QHBoxLayout(title_widget)
        title_layout.setContentsMargins(24, 0, 24, 0)
        
        title_label = QLabel("JiLing 服装分类")
        title_label.setFont(QFont("Microsoft YaHei", 16, QFont.Weight.Bold))
        title_label.setStyleSheet("color: rgba(255, 255, 255, 0.85); border: none;")
        title_layout.addWidget(title_label)
        
        layout.addWidget(title_widget)
        
        # 菜单项
        menu_widget = QWidget()
        menu_layout = QVBoxLayout(menu_widget)
        menu_layout.setContentsMargins(0, 16, 0, 0)
        menu_layout.setSpacing(4)
        
        # 菜单项定义 - 使用Unicode图标
        menu_items = [
            ("classification", "�️", "图像分类"),
            ("training", "🚀", "模型训练"),  
            ("results", "📊", "结果查看"),
            ("settings", "⚙️", "系统设置"),
        ]
        
        self.menu_buttons = {}
        for key, icon, text in menu_items:
            btn = self.create_menu_button(key, icon, text)
            self.menu_buttons[key] = btn
            menu_layout.addWidget(btn)
            
        menu_layout.addStretch()
        layout.addWidget(menu_widget)
        
        self.setLayout(layout)
        self.set_active_menu("classification")
        
    def create_menu_button(self, key: str, icon: str, text: str) -> QPushButton:
        btn = QPushButton(f"  {icon}  {text}")
        btn.setFixedHeight(48)
        btn.clicked.connect(lambda: self.set_active_menu(key))
        btn.setStyleSheet("""
            QPushButton {
                text-align: left;
                padding: 12px 24px;
                border: none;
                color: rgba(255, 255, 255, 0.65);
                font-size: 14px;
                background-color: transparent;
                border-radius: 0px;
            }
            QPushButton:hover {
                color: rgba(255, 255, 255, 0.85);
                background-color: rgba(255, 255, 255, 0.06);
            }
            QPushButton:pressed {
                background-color: rgba(255, 255, 255, 0.12);
            }
        """)
        return btn
        
    def set_active_menu(self, key: str):
        # 重置所有按钮样式
        for btn_key, btn in self.menu_buttons.items():
            if btn_key == key:
                # Ant Design的主色调 - #1890ff
                btn.setStyleSheet("""
                    QPushButton {
                        text-align: left;
                        padding: 12px 24px;
                        border: none;
                        color: rgba(255, 255, 255, 0.85);
                        font-size: 14px;
                        background-color: #1890ff;
                        border-radius: 0px;
                        margin: 0px 8px;
                    }
                    QPushButton:hover {
                        background-color: #40a9ff;
                    }
                """)
            else:
                btn.setStyleSheet("""
                    QPushButton {
                        text-align: left;
                        padding: 12px 24px;
                        border: none;
                        color: rgba(255, 255, 255, 0.65);
                        font-size: 14px;
                        background-color: transparent;
                        border-radius: 0px;
                    }
                    QPushButton:hover {
                        color: rgba(255, 255, 255, 0.85);
                        background-color: rgba(255, 255, 255, 0.06);
                    }
                """)
        
        self.current_menu = key
        self.menuChanged.emit(key)


class ModernHeader(QWidget):
    """现代化顶部栏 - Ant Design风格"""
    
    def __init__(self):
        super().__init__()
        self.setup_ui()
        
    def setup_ui(self):
        self.setFixedHeight(64)
        self.setStyleSheet("""
            QWidget {
                background-color: #ffffff;
                border-bottom: 1px solid #f0f0f0;
            }
        """)
        
        layout = QHBoxLayout()
        layout.setContentsMargins(24, 0, 24, 0)
        
        # 页面标题
        self.title_label = QLabel("图像分类")
        self.title_label.setFont(QFont("Microsoft YaHei", 18, QFont.Weight.Bold))
        self.title_label.setStyleSheet("color: #262626; border: none;")
        layout.addWidget(self.title_label)
        
        layout.addStretch()
        
        # 状态指示器组 - Web UI风格
        status_container = QWidget()
        status_container_layout = QHBoxLayout(status_container)
        status_container_layout.setContentsMargins(16, 8, 16, 8)
        status_container_layout.setSpacing(12)
        
        # 连接状态
        self.connection_status = QLabel("● 已连接")
        self.connection_status.setStyleSheet("""
            QLabel {
                color: #52c41a;
                font-size: 12px;
                font-weight: 500;
                padding: 4px 8px;
                background-color: #f6ffed;
                border: 1px solid #b7eb8f;
                border-radius: 4px;
            }
        """)
        
        # 模型状态
        self.model_status = QLabel("🧠 模型: 已加载" if hasattr(self, 'classifier') and self.classifier else "🧠 模型: 未加载")
        self.model_status.setStyleSheet("""
            QLabel {
                color: #1890ff;
                font-size: 12px;
                font-weight: 500;
                padding: 4px 8px;
                background-color: #e6f7ff;
                border: 1px solid #91d5ff;
                border-radius: 4px;
            }
        """)
        
        # GPU状态
        self.gpu_status = QLabel("🖥️ GPU: 可用")
        self.gpu_status.setStyleSheet("""
            QLabel {
                color: #722ed1;
                font-size: 12px;
                font-weight: 500;
                padding: 4px 8px;
                background-color: #f9f0ff;
                border: 1px solid #d3adf7;
                border-radius: 4px;
            }
        """)
        
        status_container_layout.addWidget(self.connection_status)
        status_container_layout.addWidget(self.model_status)
        status_container_layout.addWidget(self.gpu_status)
        status_container_layout.addStretch()
        
        layout.addWidget(status_container)
        
        self.setLayout(layout)
        
    def set_title(self, title: str):
        self.title_label.setText(title)
        
    def set_status(self, status: str, color: str = "#52c41a"):
        status_configs = {
            "normal": {"color": "#52c41a", "bg": "#f6ffed", "border": "#b7eb8f", "text": "系统正常"},
            "warning": {"color": "#faad14", "bg": "#fffbe6", "border": "#ffe58f", "text": "系统警告"}, 
            "error": {"color": "#ff4d4f", "bg": "#fff2f0", "border": "#ffccc7", "text": "系统错误"},
            "loading": {"color": "#1890ff", "bg": "#e6f7ff", "border": "#91d5ff", "text": "系统加载中"}
        }
        
        config = status_configs.get(status, status_configs["normal"])
        
        self.status_dot.setStyleSheet(f"color: {config['color']}; font-size: 12px; border: none;")
        self.status_text.setStyleSheet(f"color: {config['color']}; font-size: 14px; border: none; font-weight: 500;")
        self.status_text.setText(config['text'])
        
        # 更新父容器样式
        parent = self.status_dot.parent()
        if parent:
            parent.setStyleSheet(f"""
                background-color: {config['bg']}; 
                border: 1px solid {config['border']}; 
                border-radius: 6px;
            """)


class ModernCard(QFrame):
    """现代化卡片组件 - Ant Design风格"""
    
    def __init__(self, title: str = "", parent=None):
        super().__init__(parent)
        self.title = title
        self.setup_ui()
        
    def setup_ui(self):
        self.setFrameStyle(QFrame.Shape.Box)
        self.setStyleSheet("""
            QFrame {
                background-color: #ffffff;
                border: 1px solid #f0f0f0;
                border-radius: 6px;
                padding: 0px;
            }
            QFrame:hover {
                border-color: #d9d9d9;
            }
        """)
        
        layout = QVBoxLayout()
        layout.setContentsMargins(24, 20, 24, 24)
        layout.setSpacing(16)
        
        if self.title:
            title_label = QLabel(self.title)
            title_label.setFont(QFont("Microsoft YaHei", 16, QFont.Weight.Bold))
            title_label.setStyleSheet("color: #262626; border: none; padding: 0px;")
            layout.addWidget(title_label)
            
        self.content_layout = QVBoxLayout()
        layout.addLayout(self.content_layout)
        
        self.setLayout(layout)
        
    def add_content(self, widget):
        self.content_layout.addWidget(widget)


class ModernButton(QPushButton):
    """现代化按钮组件 - Ant Design风格"""
    
    def __init__(self, text: str, button_type: str = "default", parent=None):
        super().__init__(text, parent)
        self.button_type = button_type
        self.setup_style()
        
    def setup_style(self):
        base_style = """
            QPushButton {
                font-size: 14px;
                font-weight: 400;
                border-radius: 6px;
                padding: 4px 15px;
                height: 32px;
                border: 1px solid;
                outline: none;
            }
            QPushButton:focus {
                outline: none;
            }
        """
        
        if self.button_type == "primary":
            self.setStyleSheet(base_style + """
                QPushButton {
                    background-color: #1890ff;
                    border-color: #1890ff;
                    color: #ffffff;
                }
                QPushButton:hover {
                    background-color: #40a9ff;
                    border-color: #40a9ff;
                }
                QPushButton:pressed {
                    background-color: #096dd9;
                    border-color: #096dd9;
                }
                QPushButton:disabled {
                    background-color: #f5f5f5;
                    border-color: #d9d9d9;
                    color: rgba(0, 0, 0, 0.25);
                }
            """)
        elif self.button_type == "success":
            self.setStyleSheet(base_style + """
                QPushButton {
                    background-color: #52c41a;
                    border-color: #52c41a;
                    color: #ffffff;
                }
                QPushButton:hover {
                    background-color: #73d13d;
                    border-color: #73d13d;
                }
                QPushButton:pressed {
                    background-color: #389e0d;
                    border-color: #389e0d;
                }
            """)
        elif self.button_type == "danger":
            self.setStyleSheet(base_style + """
                QPushButton {
                    background-color: #ff4d4f;
                    border-color: #ff4d4f;
                    color: #ffffff;
                }
                QPushButton:hover {
                    background-color: #ff7875;
                    border-color: #ff7875;
                }
                QPushButton:pressed {
                    background-color: #d9363e;
                    border-color: #d9363e;
                }
            """)
        else:  # default
            self.setStyleSheet(base_style + """
                QPushButton {
                    background-color: #ffffff;
                    border-color: #d9d9d9;
                    color: rgba(0, 0, 0, 0.85);
                }
                QPushButton:hover {
                    border-color: #40a9ff;
                    color: #40a9ff;
                }
                QPushButton:pressed {
                    border-color: #096dd9;
                    color: #096dd9;
                }
                QPushButton:disabled {
                    background-color: #f5f5f5;
                    border-color: #d9d9d9;
                    color: rgba(0, 0, 0, 0.25);
                }
            """)


class ClassificationPage(QWidget):
    """图像分类页面 - 完整功能版本"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.classifier = None
        self.current_image_path = None
        self.current_folder_path = None
        self.classification_results = []
        self.settings = QSettings()
        self.setup_ui()
        self.load_model()
        self.update_warning_visibility()
        
    def setup_ui(self):
        layout = QVBoxLayout()
        layout.setContentsMargins(24, 24, 24, 24)
        layout.setSpacing(24)
        
        # 模型警告提示框
        self.warning_widget = QWidget()
        warning_layout = QHBoxLayout(self.warning_widget)
        warning_layout.setContentsMargins(16, 12, 16, 12)
        warning_layout.setSpacing(12)
        
        # 警告图标
        warning_icon = QLabel("⚠️")
        warning_icon.setStyleSheet("font-size: 16px; color: #faad14;")
        
        # 警告文本
        warning_text = QLabel("请先加载模型")
        warning_text.setStyleSheet("color: #262626; font-size: 14px; font-weight: 500;")
        
        # 子文本
        warning_subtext = QLabel("在设置页面选择并加载一个模型后再进行分类")
        warning_subtext.setStyleSheet("color: #8c8c8c; font-size: 12px;")
        
        warning_layout.addWidget(warning_icon)
        warning_layout.addWidget(warning_text)
        warning_layout.addWidget(warning_subtext)
        warning_layout.addStretch()
        
        self.warning_widget.setStyleSheet("""
            QWidget {
                background-color: #fffbe6;
                border: 1px solid #ffe58f;
                border-radius: 6px;
            }
        """)
        
        layout.addWidget(self.warning_widget)
        
        # 第一行：文件选择
        file_card = ModernCard("文件选择")
        file_content = QWidget()
        file_layout = QVBoxLayout(file_content)
        
        # 单个文件选择
        single_layout = QHBoxLayout()
        single_layout.addWidget(QLabel("单个文件:"))
        self.single_file_edit = QLineEdit()
        self.single_file_edit.setPlaceholderText("选择单个图像文件...")
        self.single_file_edit.textChanged.connect(self.on_file_path_changed)
        single_layout.addWidget(self.single_file_edit)
        
        single_browse_btn = ModernButton("� 浏览", "default")
        single_browse_btn.clicked.connect(self.browse_single_file)
        single_layout.addWidget(single_browse_btn)
        file_layout.addLayout(single_layout)
        
        # 文件夹选择
        folder_layout = QHBoxLayout()
        folder_layout.addWidget(QLabel("文件夹:"))
        self.folder_edit = QLineEdit()
        self.folder_edit.setPlaceholderText("选择包含图像的文件夹...")
        self.folder_edit.textChanged.connect(self.on_folder_path_changed)
        folder_layout.addWidget(self.folder_edit)
        
        folder_browse_btn = ModernButton("📂 浏览", "default")
        folder_browse_btn.clicked.connect(self.browse_folder)
        folder_layout.addWidget(folder_browse_btn)
        
        # 添加"使用上次路径"按钮
        last_path_btn = ModernButton("⏮️ 上次路径", "default")
        last_path_btn.setToolTip("使用上次选择的文件夹路径")
        last_path_btn.clicked.connect(self.use_last_classification_path)
        folder_layout.addWidget(last_path_btn)
        
        file_layout.addLayout(folder_layout)
        file_card.add_content(file_content)
        layout.addWidget(file_card)
        
        # 第二行：模型选择
        model_card = ModernCard("模型选择")
        model_content = QWidget()
        model_layout = QVBoxLayout(model_content)
        
        # 模型文件选择
        model_file_layout = QHBoxLayout()
        model_file_layout.addWidget(QLabel("模型文件:"))
        self.model_file_edit = QLineEdit()
        self.model_file_edit.setPlaceholderText("选择预训练模型文件 (.pth)...")
        self.model_file_edit.textChanged.connect(self.on_model_path_changed)
        model_file_layout.addWidget(self.model_file_edit)
        
        model_browse_btn = ModernButton("📁 浏览", "default")
        model_browse_btn.clicked.connect(self.browse_model_file)
        model_file_layout.addWidget(model_browse_btn)
        
        model_layout.addLayout(model_file_layout)
        
        # 模型信息和控制
        model_info_layout = QHBoxLayout()
        self.model_status_label = QLabel("状态: 未加载模型")
        self.model_status_label.setStyleSheet("color: #8c8c8c; font-style: italic;")
        model_info_layout.addWidget(self.model_status_label)
        
        model_info_layout.addStretch()
        
        # 模型控制按钮
        self.load_model_btn = ModernButton("🔄 加载模型", "default")
        self.load_model_btn.clicked.connect(self.load_model_from_file)
        self.load_model_btn.setEnabled(False)
        model_info_layout.addWidget(self.load_model_btn)
        
        self.use_default_btn = ModernButton("🤖 使用默认模型", "primary")
        self.use_default_btn.clicked.connect(self.use_default_model)
        model_info_layout.addWidget(self.use_default_btn)
        
        model_layout.addLayout(model_info_layout)
        model_card.add_content(model_content)
        layout.addWidget(model_card)
        
        # 第三行：图片预览和分类控制
        row3 = QHBoxLayout()
        
        # 图片预览
        preview_card = ModernCard("图片预览")
        self.image_label = QLabel()
        self.image_label.setMinimumSize(300, 300)
        self.image_label.setMaximumSize(400, 400)
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_label.setStyleSheet("""
            QLabel {
                border: 2px dashed #d9d9d9;
                border-radius: 6px;
                background-color: #fafafa;
                color: #8c8c8c;
            }
        """)
        self.image_label.setText("�️\n\n点击选择图片进行预览")
        preview_card.add_content(self.image_label)
        row3.addWidget(preview_card, 1)
        
        # 分类控制
        control_card = ModernCard("分类控制")
        control_content = QWidget()
        control_layout = QVBoxLayout(control_content)
        
        # 操作按钮
        action_layout = QVBoxLayout()
        action_layout.setSpacing(12)
        
        self.classify_btn = ModernButton("🚀 开始分类", "primary")
        self.classify_btn.clicked.connect(self.start_classification)
        action_layout.addWidget(self.classify_btn)
        
        self.clear_results_btn = ModernButton("🗑️ 清空结果", "default")
        self.clear_results_btn.clicked.connect(self.clear_classification_results)
        action_layout.addWidget(self.clear_results_btn)
        
        control_layout.addLayout(action_layout)
        
        # 进度条
        self.classification_progress = QProgressBar()
        self.classification_progress.setVisible(False)
        self.classification_progress.setStyleSheet("""
            QProgressBar {
                border: 1px solid #d9d9d9;
                border-radius: 3px;
                text-align: center;
                font-size: 12px;
                height: 22px;
            }
            QProgressBar::chunk {
                background-color: #1890ff;
                border-radius: 2px;
            }
        """)
        control_layout.addWidget(self.classification_progress)
        
        control_layout.addStretch()
        control_card.add_content(control_content)
        row3.addWidget(control_card, 1)
        
        layout.addLayout(row3)
        
        # 第四行：结果显示
        results_card = ModernCard("分类结果")
        self.results_table = QTableWidget()
        self.results_table.setColumnCount(4)
        self.results_table.setHorizontalHeaderLabels(["文件名", "分类结果", "置信度", "路径"])
        
        # 设置表格样式
        self.results_table.setStyleSheet("""
            QTableWidget {
                border: 1px solid #f0f0f0;
                border-radius: 6px;
                background-color: #ffffff;
                gridline-color: #f0f0f0;
                font-size: 14px;
            }
            QHeaderView::section {
                background-color: #fafafa;
                border: none;
                border-bottom: 1px solid #f0f0f0;
                border-right: 1px solid #f0f0f0;
                padding: 8px 12px;
                font-weight: 500;
                color: #262626;
            }
            QTableWidget::item {
                padding: 8px 12px;
                border-bottom: 1px solid #f0f0f0;
            }
            QTableWidget::item:selected {
                background-color: #e6f7ff;
                color: #262626;
            }
        """)
        
        # 设置表格列宽
        header = self.results_table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(3, QHeaderView.ResizeMode.Stretch)
        
        results_card.add_content(self.results_table)
        layout.addWidget(results_card)
        
        self.setLayout(layout)
        
    def load_model(self):
        """自动加载最新模型"""
        try:
            # 查找可用的模型文件
            models_dir = Path("models")
            if not models_dir.exists():
                self.model_status_label.setText("❌ 模型目录不存在")
                return
                
            model_files = list(models_dir.glob("*.pth"))
            if not model_files:
                self.model_status_label.setText("❌ 未找到模型文件")
                return
                
            # 加载最新的模型
            latest_model = max(model_files, key=lambda x: x.stat().st_mtime)
            
            # 使用模型路径初始化分类器，使用支持的模型名称
            self.classifier = ClothingClassifier(
                model_path=str(latest_model),
                model_name='tf_efficientnetv2_s'  # 使用支持的模型名称
            )
            self.model_status_label.setText("✅ 模型已加载")
            self.model_status_label.setStyleSheet("color: #52c41a; font-weight: 500;")
            self.model_file_edit.setText(str(latest_model))
            print(f"模型加载成功: {latest_model}")
            self.update_warning_visibility()
                
        except Exception as e:
            self.model_status_label.setText(f"❌ 模型加载错误: {str(e)}")
            self.model_status_label.setStyleSheet("color: #ff4d4f; font-weight: 500;")
            print(f"加载模型时出错: {e}")
            self.classifier = None
            self.update_warning_visibility()
    
    def update_warning_visibility(self):
        """根据模型状态更新警告显示"""
        if hasattr(self, 'warning_widget'):
            if self.classifier is None:
                self.warning_widget.show()
            else:
                self.warning_widget.hide()
            
    def on_file_path_changed(self):
        """文件路径变化时的处理"""
        path = self.single_file_edit.text().strip()
        if path and Path(path).exists():
            self.current_image_path = path
            self.load_image_preview(path)
            self.clear_folder_selection()
        
    def on_folder_path_changed(self):
        """文件夹路径变化时的处理"""
        path = self.folder_edit.text().strip()
        if path and Path(path).exists():
            self.current_folder_path = path
            self.clear_single_file_selection()
            
    def on_model_path_changed(self):
        """模型路径变化时的处理"""
        path = self.model_file_edit.text().strip()
        self.load_model_btn.setEnabled(bool(path and Path(path).exists()))
        
    def browse_single_file(self):
        """浏览单个文件"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "选择图片文件", "",
            "图片文件 (*.jpg *.jpeg *.png *.bmp *.tiff *.webp)"
        )
        if file_path:
            self.single_file_edit.setText(file_path)
            
    def browse_folder(self):
        """浏览文件夹"""
        folder_path = QFileDialog.getExistingDirectory(self, "选择图片文件夹")
        if folder_path:
            self.folder_edit.setText(folder_path)
            # 保存路径到设置
            self.settings.setValue("last_classification_folder", folder_path)
            
    def browse_model_file(self):
        """浏览模型文件"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "选择模型文件", "models",
            "PyTorch模型 (*.pth *.pt)"
        )
        if file_path:
            self.model_file_edit.setText(file_path)
            
    def use_last_classification_path(self):
        """使用上次分类路径"""
        last_path = self.settings.value("last_classification_folder", "")
        if last_path and Path(last_path).exists():
            self.folder_edit.setText(last_path)
        else:
            QMessageBox.information(self, "提示", "没有找到上次使用的路径")
            
    def clear_single_file_selection(self):
        """清除单文件选择"""
        self.single_file_edit.clear()
        self.current_image_path = None
        self.clear_image_preview()
        
    def clear_folder_selection(self):
        """清除文件夹选择"""
        self.folder_edit.clear()
        self.current_folder_path = None
        
    def load_image_preview(self, image_path: str):
        """加载图片预览"""
        try:
            pixmap = QPixmap(image_path)
            if not pixmap.isNull():
                # 缩放图片以适应预览区域
                scaled_pixmap = pixmap.scaled(
                    self.image_label.size(),
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation
                )
                self.image_label.setPixmap(scaled_pixmap)
                self.image_label.setStyleSheet("""
                    QLabel {
                        border: 1px solid #d9d9d9;
                        border-radius: 6px;
                        background-color: #ffffff;
                    }
                """)
        except Exception as e:
            print(f"加载图片预览失败: {e}")
            
    def clear_image_preview(self):
        """清除图片预览"""
        self.image_label.setPixmap(QPixmap())
        self.image_label.setText("📷\n\n点击或拖拽图片到此区域上传\n\n支持 JPG、PNG、BMP 等格式")
        self.image_label.setStyleSheet("""
            QLabel {
                border: 2px dashed #d9d9d9;
                border-radius: 6px;
                background-color: #fafafa;
                color: #8c8c8c;
            }
        """)
        
    def use_default_model(self):
        """使用默认模型"""
        self.load_model()
        
    def load_model_from_file(self):
        """从文件加载模型"""
        model_path = self.model_file_edit.text().strip()
        if not model_path or not Path(model_path).exists():
            QMessageBox.warning(self, "警告", "请选择有效的模型文件")
            return
            
        try:
            self.classifier = ClothingClassifier(
                model_path=model_path,
                model_name='tf_efficientnetv2_s'
            )
            self.model_status_label.setText("✅ 模型已加载")
            self.model_status_label.setStyleSheet("color: #52c41a; font-weight: 500;")
            print(f"模型加载成功: {model_path}")
            self.update_warning_visibility()
        except Exception as e:
            self.model_status_label.setText(f"❌ 模型加载失败: {str(e)}")
            self.model_status_label.setStyleSheet("color: #ff4d4f; font-weight: 500;")
            self.classifier = None
            self.update_warning_visibility()
            
    def start_classification(self):
        """开始分类"""
        if not self.classifier:
            QMessageBox.warning(self, "警告", "请先加载模型")
            return
            
        # 检查输入
        if self.current_image_path:
            # 单文件分类
            self.classify_single_file()
        elif self.current_folder_path:
            # 批量分类
            self.classify_folder()
        else:
            QMessageBox.warning(self, "警告", "请选择要分类的文件或文件夹")
            
    def classify_single_file(self):
        """分类单个文件"""
        if not self.current_image_path or not Path(self.current_image_path).exists():
            return
            
        try:
            self.classification_progress.setVisible(True)
            self.classification_progress.setRange(0, 0)  # 不确定进度
            self.classify_btn.setEnabled(False)
            
            # 执行分类
            results = self.classifier.classify_image(self.current_image_path)
            
            if results:
                # 添加到结果表格
                row = self.results_table.rowCount()
                self.results_table.insertRow(row)
                
                file_name = Path(self.current_image_path).name
                best_class, best_confidence = results[0]
                
                self.results_table.setItem(row, 0, QTableWidgetItem(file_name))
                self.results_table.setItem(row, 1, QTableWidgetItem(best_class))
                self.results_table.setItem(row, 2, QTableWidgetItem(f"{best_confidence*100:.2f}%"))
                self.results_table.setItem(row, 3, QTableWidgetItem(self.current_image_path))
                
                QMessageBox.information(self, "分类完成", 
                    f"分类结果：{best_class}\n置信度：{best_confidence*100:.2f}%")
            else:
                QMessageBox.warning(self, "分类失败", "无法分类该图片")
                
        except Exception as e:
            QMessageBox.critical(self, "错误", f"分类过程中出错：{str(e)}")
            
        finally:
            self.classification_progress.setVisible(False)
            self.classify_btn.setEnabled(True)
            
    def classify_folder(self):
        """批量分类文件夹"""
        # TODO: 实现批量分类功能
        QMessageBox.information(self, "提示", "批量分类功能正在开发中")
        
    def clear_classification_results(self):
        """清空分类结果"""
        self.results_table.setRowCount(0)
        self.classification_results.clear()
        self.clear_single_file_selection()
        self.clear_folder_selection()


class TrainingPage(QWidget):
    """模型训练页面"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setup_ui()
        
    def setup_ui(self):
        layout = QVBoxLayout()
        layout.setContentsMargins(24, 24, 24, 24)
        layout.setSpacing(24)
        
        # 训练配置卡片
        config_card = ModernCard("训练配置")
        config_content = QWidget()
        config_layout = QGridLayout(config_content)
        
        # 数据集路径
        config_layout.addWidget(QLabel("数据集路径:"), 0, 0)
        self.dataset_path_edit = QLineEdit()
        self.dataset_path_edit.setPlaceholderText("选择训练数据集文件夹")
        config_layout.addWidget(self.dataset_path_edit, 0, 1)
        
        dataset_btn = ModernButton("📂 浏览", "default")
        dataset_btn.clicked.connect(self.select_dataset)
        config_layout.addWidget(dataset_btn, 0, 2)
        
        # 训练参数
        config_layout.addWidget(QLabel("学习率:"), 1, 0)
        self.learning_rate_spin = QDoubleSpinBox()
        self.learning_rate_spin.setRange(0.0001, 1.0)
        self.learning_rate_spin.setValue(0.001)
        self.learning_rate_spin.setDecimals(4)
        config_layout.addWidget(self.learning_rate_spin, 1, 1)
        
        config_layout.addWidget(QLabel("批次大小:"), 2, 0)
        self.batch_size_spin = QSpinBox()
        self.batch_size_spin.setRange(1, 128)
        self.batch_size_spin.setValue(32)
        config_layout.addWidget(self.batch_size_spin, 2, 1)
        
        config_layout.addWidget(QLabel("训练轮数:"), 3, 0)
        self.epochs_spin = QSpinBox()
        self.epochs_spin.setRange(1, 1000)
        self.epochs_spin.setValue(50)
        config_layout.addWidget(self.epochs_spin, 3, 1)
        
        config_card.add_content(config_content)
        layout.addWidget(config_card)
        
        # 训练控制卡片
        control_card = ModernCard("训练控制")
        control_content = QWidget()
        control_layout = QVBoxLayout(control_content)
        
        btn_layout = QHBoxLayout()
        self.start_training_btn = ModernButton("🚀 开始训练", "primary")
        self.start_training_btn.clicked.connect(self.start_training)
        
        self.stop_training_btn = ModernButton("⏹️ 停止训练", "danger")
        self.stop_training_btn.setEnabled(False)
        
        btn_layout.addWidget(self.start_training_btn)
        btn_layout.addWidget(self.stop_training_btn)
        btn_layout.addStretch()
        control_layout.addLayout(btn_layout)
        
        # 训练进度
        self.training_progress = QProgressBar()
        self.training_progress.setStyleSheet("""
            QProgressBar {
                border: 1px solid #d9d9d9;
                border-radius: 4px;
                text-align: center;
                font-size: 12px;
                height: 24px;
            }
            QProgressBar::chunk {
                background-color: #52c41a;
                border-radius: 3px;
            }
        """)
        control_layout.addWidget(self.training_progress)
        
        control_card.add_content(control_content)
        layout.addWidget(control_card)
        
        # 训练日志卡片
        log_card = ModernCard("训练日志")
        self.training_log = QTextEdit()
        self.training_log.setStyleSheet("""
            QTextEdit {
                border: 1px solid #d9d9d9;
                border-radius: 6px;
                padding: 12px;
                font-family: 'Consolas', monospace;
                font-size: 12px;
                background-color: #fafafa;
            }
        """)
        self.training_log.setText("等待开始训练...")
        log_card.add_content(self.training_log)
        layout.addWidget(log_card)
        
        self.setLayout(layout)
        
    def select_dataset(self):
        """选择数据集文件夹"""
        folder_path = QFileDialog.getExistingDirectory(self, "选择训练数据集文件夹")
        if folder_path:
            self.dataset_path_edit.setText(folder_path)
            
    def start_training(self):
        """开始训练"""
        dataset_path = self.dataset_path_edit.text()
        if not dataset_path:
            QMessageBox.warning(self, "警告", "请先选择数据集文件夹")
            return
            
        if not Path(dataset_path).exists():
            QMessageBox.warning(self, "警告", "数据集文件夹不存在")
            return
            
        # TODO: 实现实际的训练逻辑
        self.training_log.setText("🚀 训练即将开始...\n")
        self.start_training_btn.setEnabled(False)
        self.stop_training_btn.setEnabled(True)
        
        # 模拟训练过程
        self.training_log.append(f"📂 数据集路径: {dataset_path}")
        self.training_log.append(f"⚙️ 学习率: {self.learning_rate_spin.value()}")
        self.training_log.append(f"📦 批次大小: {self.batch_size_spin.value()}")
        self.training_log.append(f"🔄 训练轮数: {self.epochs_spin.value()}")
        self.training_log.append("💡 训练功能待完善...")


class ResultsPage(QWidget):
    """结果查看页面"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setup_ui()
        
    def setup_ui(self):
        layout = QVBoxLayout()
        layout.setContentsMargins(24, 24, 24, 24)
        
        # 结果展示卡片
        results_card = ModernCard("分类结果历史")
        
        # 结果表格
        self.results_table = QTableWidget()
        self.results_table.setColumnCount(4)
        self.results_table.setHorizontalHeaderLabels(["时间", "图片", "分类结果", "置信度"])
        self.results_table.horizontalHeader().setStretchLastSection(True)
        self.results_table.setStyleSheet("""
            QTableWidget {
                border: 1px solid #d9d9d9;
                border-radius: 6px;
                background-color: #ffffff;
                gridline-color: #f0f0f0;
            }
            QHeaderView::section {
                background-color: #fafafa;
                border: none;
                border-bottom: 1px solid #f0f0f0;
                padding: 8px 12px;
                font-weight: 500;
            }
        """)
        
        results_card.add_content(self.results_table)
        layout.addWidget(results_card)
        
        self.setLayout(layout)


class SettingsPage(QWidget):
    """系统设置页面"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setup_ui()
        
    def setup_ui(self):
        layout = QVBoxLayout()
        layout.setContentsMargins(24, 24, 24, 24)
        layout.setSpacing(24)
        
        # 模型设置卡片
        model_card = ModernCard("模型设置")
        model_content = QWidget()
        model_layout = QGridLayout(model_content)
        
        model_layout.addWidget(QLabel("当前模型:"), 0, 0)
        self.current_model_label = QLabel("未加载")
        self.current_model_label.setStyleSheet("color: #8c8c8c;")
        model_layout.addWidget(self.current_model_label, 0, 1)
        
        load_model_btn = ModernButton("📁 加载模型", "default")
        model_layout.addWidget(load_model_btn, 0, 2)
        
        model_card.add_content(model_content)
        layout.addWidget(model_card)
        
        # 系统设置卡片
        system_card = ModernCard("系统设置")
        system_content = QWidget()
        system_layout = QGridLayout(system_content)
        
        # GPU设置
        system_layout.addWidget(QLabel("使用GPU:"), 0, 0)
        self.gpu_checkbox = QCheckBox("启用GPU加速")
        system_layout.addWidget(self.gpu_checkbox, 0, 1)
        
        # 日志级别
        system_layout.addWidget(QLabel("日志级别:"), 1, 0)
        self.log_level_combo = QComboBox()
        self.log_level_combo.addItems(["DEBUG", "INFO", "WARNING", "ERROR"])
        self.log_level_combo.setCurrentText("INFO")
        system_layout.addWidget(self.log_level_combo, 1, 1)
        
        system_card.add_content(system_content)
        layout.addWidget(system_card)
        
        # 关于信息卡片
        about_card = ModernCard("关于")
        about_content = QWidget()
        about_layout = QVBoxLayout(about_content)
        
        about_text = QLabel("""
        <h3>JiLing 服装分类系统</h3>
        <p>版本: 2.0.0</p>
        <p>基于深度学习的智能服装分类系统</p>
        <p>支持多种服装类型的自动识别和分类</p>
        """)
        about_text.setStyleSheet("color: #595959; line-height: 1.6;")
        about_layout.addWidget(about_text)
        
        about_card.add_content(about_content)
        layout.addWidget(about_card)
        
        layout.addStretch()
        self.setLayout(layout)


class ModernMainWindow(QMainWindow):
    """现代化主窗口"""
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("JiLing 服装分类系统 - 现代版")
        self.setMinimumSize(1200, 800)
        self.setup_ui()
        self.setup_style()
        
    def setup_ui(self):
        # 主窗口部件
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        
        # 主布局
        main_layout = QHBoxLayout(main_widget)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)
        
        # 侧边栏
        self.sidebar = ModernSidebar()
        self.sidebar.menuChanged.connect(self.on_menu_changed)
        main_layout.addWidget(self.sidebar)
        
        # 右侧内容区域
        content_widget = QWidget()
        content_layout = QVBoxLayout(content_widget)
        content_layout.setContentsMargins(0, 0, 0, 0)
        content_layout.setSpacing(0)
        
        # 顶部栏
        self.header = ModernHeader()
        content_layout.addWidget(self.header)
        
        # 页面内容区域
        self.content_stack = QStackedWidget()
        self.content_stack.setStyleSheet("background-color: #f5f5f5;")
        
        # 添加各个页面
        self.classification_page = ClassificationPage()
        self.training_page = TrainingPage()
        self.results_page = ResultsPage()
        self.settings_page = SettingsPage()
        
        self.content_stack.addWidget(self.classification_page)
        self.content_stack.addWidget(self.training_page)
        self.content_stack.addWidget(self.results_page)
        self.content_stack.addWidget(self.settings_page)
        
        content_layout.addWidget(self.content_stack)
        main_layout.addWidget(content_widget)
        
        # 默认显示分类页面
        self.on_menu_changed("classification")
        
    def setup_style(self):
        """设置全局样式"""
        self.setStyleSheet("""
            QMainWindow {
                background-color: #f5f5f5;
            }
            QWidget {
                font-family: "Microsoft YaHei", "Segoe UI", sans-serif;
            }
        """)
        
    def on_menu_changed(self, menu_key: str):
        """菜单切换事件"""
        page_map = {
            "classification": (0, "图像分类"),
            "training": (1, "模型训练"),
            "results": (2, "结果查看"),
            "settings": (3, "系统设置"),
        }
        
        if menu_key in page_map:
            page_index, title = page_map[menu_key]
            self.content_stack.setCurrentIndex(page_index)
            self.header.set_title(title)


def main():
    """主函数"""
    app = QApplication(sys.argv)
    
    # 设置应用程序信息
    app.setApplicationName("JiLing 服装分类系统")
    app.setApplicationVersion("2.0.0")
    app.setOrganizationName("JiLing Technology")
    
    # 创建并显示主窗口
    window = ModernMainWindow()
    window.show()
    
    # 运行应用程序
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
