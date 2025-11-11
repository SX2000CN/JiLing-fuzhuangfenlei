#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
JiLing 服装分类系统 - Finder风格界面
完全模仿 macOS Finder 的视觉效果和交互方式
"""
import sys
import os
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QLineEdit, QFileDialog, QProgressBar,
    QListWidget, QListWidgetItem, QSplitter, QFrame, QTreeWidget,
    QTreeWidgetItem, QScrollArea, QStackedWidget, QToolBar, QStatusBar,
    QMenu, QGraphicsDropShadowEffect, QGridLayout
)
from PySide6.QtCore import Qt, Signal, QSize, QTimer, QPropertyAnimation, QEasingCurve
from PySide6.QtGui import QPixmap, QFont, QIcon, QColor, QPalette, QAction

# 添加项目路径
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# 导入核心模块
try:
    from src.core.model_factory import ModelFactory
    from src.core.pytorch_classifier import ClothingClassifier
    from src.core.pytorch_trainer import ClothingTrainer
except:
    ModelFactory = None
    ClothingClassifier = None
    ClothingTrainer = None


# ==================== Finder配色方案 ====================
class FinderColors:
    """macOS Finder配色"""
    # 背景色 - 浅灰色调
    WINDOW_BG = "#ECECEC"
    SIDEBAR_BG = "#E8E8E8"
    TOOLBAR_BG = "#F6F6F6"
    CONTENT_BG = "#FFFFFF"
    
    # 边框
    BORDER = "#D1D1D1"
    DIVIDER = "#C8C8C8"
    
    # 文字
    TEXT_PRIMARY = "#000000"
    TEXT_SECONDARY = "#6B6B6B"
    TEXT_DISABLED = "#ACACAC"
    
    # 选中状态 - Finder蓝色
    SELECTION_BG = "#3B7FFF"
    SELECTION_TEXT = "#FFFFFF"
    
    # 悬停状态
    HOVER_BG = "#DCDCDC"
    
    # 侧边栏图标颜色
    ICON_BLUE = "#007AFF"
    ICON_PURPLE = "#AF52DE"
    ICON_GREEN = "#34C759"
    ICON_ORANGE = "#FF9500"
    ICON_RED = "#FF3B30"


# ==================== Finder风格侧边栏 ====================
class FinderSidebar(QWidget):
    """Finder侧边栏 - 左侧导航"""
    itemClicked = Signal(str, str)  # (category, item_name)
    
    def __init__(self):
        super().__init__()
        self.setup_ui()
        
    def setup_ui(self):
        self.setFixedWidth(200)
        self.setStyleSheet(f"""
            QWidget {{
                background-color: {FinderColors.SIDEBAR_BG};
                border-right: 1px solid {FinderColors.BORDER};
            }}
        """)
        
        layout = QVBoxLayout()
        layout.setContentsMargins(0, 10, 0, 0)
        layout.setSpacing(0)
        
        # 收藏夹区域
        self.add_section("收藏", [
            ("📱", "分类任务", "classification"),
            ("🚀", "训练任务", "training"),
            ("📊", "结果分析", "results"),
        ], layout)
        
        # 位置区域
        self.add_section("位置", [
            ("💻", "本地模型", "models"),
            ("📁", "数据集", "datasets"),
            ("🗂️", "输出结果", "outputs"),
        ], layout)
        
        # 标签区域
        self.add_section("标签", [
            ("🔴", "重要", "important"),
            ("🟡", "进行中", "inprogress"),
            ("🟢", "已完成", "completed"),
        ], layout)
        
        layout.addStretch()
        self.setLayout(layout)
    
    def add_section(self, title: str, items: List[tuple], parent_layout):
        """添加侧边栏分组"""
        # 分组标题
        title_label = QLabel(title.upper())
        title_label.setStyleSheet(f"""
            QLabel {{
                color: {FinderColors.TEXT_SECONDARY};
                font-size: 11px;
                font-weight: 600;
                padding: 8px 12px 4px 12px;
                background-color: transparent;
            }}
        """)
        parent_layout.addWidget(title_label)
        
        # 分组项
        for icon, text, key in items:
            item = self.create_sidebar_item(icon, text, key)
            parent_layout.addWidget(item)
    
    def create_sidebar_item(self, icon: str, text: str, key: str) -> QPushButton:
        """创建侧边栏项"""
        btn = QPushButton(f"  {icon}  {text}")
        btn.setFixedHeight(28)
        btn.setCursor(Qt.CursorShape.PointingHandCursor)
        btn.clicked.connect(lambda: self.itemClicked.emit("navigation", key))
        
        btn.setStyleSheet(f"""
            QPushButton {{
                text-align: left;
                padding-left: 12px;
                border: none;
                background-color: transparent;
                color: {FinderColors.TEXT_PRIMARY};
                font-size: 13px;
                border-radius: 5px;
                margin: 0px 6px;
            }}
            QPushButton:hover {{
                background-color: {FinderColors.HOVER_BG};
            }}
            QPushButton:pressed {{
                background-color: {FinderColors.SELECTION_BG};
                color: {FinderColors.SELECTION_TEXT};
            }}
        """)
        return btn


# ==================== Finder风格工具栏 ====================
class FinderToolbar(QWidget):
    """Finder顶部工具栏"""
    actionTriggered = Signal(str)
    
    def __init__(self):
        super().__init__()
        self.setup_ui()
        
    def setup_ui(self):
        self.setFixedHeight(52)
        self.setStyleSheet(f"""
            QWidget {{
                background-color: {FinderColors.TOOLBAR_BG};
                border-bottom: 1px solid {FinderColors.BORDER};
            }}
        """)
        
        layout = QHBoxLayout()
        layout.setContentsMargins(10, 0, 10, 0)
        layout.setSpacing(8)
        
        # 导航按钮组
        nav_layout = QHBoxLayout()
        nav_layout.setSpacing(0)
        
        back_btn = self.create_toolbar_button("◀", "后退", "back")
        forward_btn = self.create_toolbar_button("▶", "前进", "forward")
        
        nav_layout.addWidget(back_btn)
        nav_layout.addWidget(forward_btn)
        layout.addLayout(nav_layout)
        
        layout.addSpacing(12)
        
        # 视图切换按钮
        view_layout = QHBoxLayout()
        view_layout.setSpacing(0)
        
        icon_view_btn = self.create_toolbar_button("⊞", "图标视图", "icon_view")
        list_view_btn = self.create_toolbar_button("☰", "列表视图", "list_view")
        column_view_btn = self.create_toolbar_button("⫴", "列视图", "column_view")
        
        view_layout.addWidget(icon_view_btn)
        view_layout.addWidget(list_view_btn)
        view_layout.addWidget(column_view_btn)
        layout.addLayout(view_layout)
        
        layout.addSpacing(12)
        
        # 操作按钮
        action_btn = self.create_toolbar_button("⚙", "操作", "actions")
        share_btn = self.create_toolbar_button("↗", "分享", "share")
        
        layout.addWidget(action_btn)
        layout.addWidget(share_btn)
        
        layout.addStretch()
        
        # 搜索框 - Finder样式
        self.search_box = QLineEdit()
        self.search_box.setPlaceholderText("搜索")
        self.search_box.setFixedWidth(200)
        self.search_box.setStyleSheet(f"""
            QLineEdit {{
                padding: 6px 28px 6px 28px;
                border: 1px solid {FinderColors.BORDER};
                border-radius: 10px;
                background-color: {FinderColors.CONTENT_BG};
                font-size: 13px;
            }}
            QLineEdit:focus {{
                border: 1px solid {FinderColors.SELECTION_BG};
            }}
        """)
        layout.addWidget(self.search_box)
        
        self.setLayout(layout)
    
    def create_toolbar_button(self, icon: str, tooltip: str, action: str) -> QPushButton:
        """创建工具栏按钮"""
        btn = QPushButton(icon)
        btn.setToolTip(tooltip)
        btn.setFixedSize(36, 28)
        btn.setCursor(Qt.CursorShape.PointingHandCursor)
        btn.clicked.connect(lambda: self.actionTriggered.emit(action))
        
        btn.setStyleSheet(f"""
            QPushButton {{
                border: 1px solid transparent;
                border-radius: 5px;
                background-color: transparent;
                color: {FinderColors.TEXT_PRIMARY};
                font-size: 16px;
                padding: 0px;
            }}
            QPushButton:hover {{
                background-color: {FinderColors.HOVER_BG};
                border: 1px solid {FinderColors.BORDER};
            }}
            QPushButton:pressed {{
                background-color: {FinderColors.SELECTION_BG};
                color: {FinderColors.SELECTION_TEXT};
            }}
        """)
        return btn


# ==================== Finder风格内容区 ====================
class FinderContentArea(QWidget):
    """Finder主内容区域"""
    
    def __init__(self):
        super().__init__()
        self.current_view = "icon"
        self.setup_ui()
        
    def setup_ui(self):
        self.setStyleSheet(f"""
            QWidget {{
                background-color: {FinderColors.CONTENT_BG};
            }}
        """)
        
        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        
        # 路径导航栏 (面包屑)
        self.breadcrumb = QLabel("JiLing 服装分类系统  ›  图像分类")
        self.breadcrumb.setStyleSheet(f"""
            QLabel {{
                padding: 10px 20px;
                background-color: {FinderColors.TOOLBAR_BG};
                border-bottom: 1px solid {FinderColors.BORDER};
                color: {FinderColors.TEXT_SECONDARY};
                font-size: 12px;
            }}
        """)
        layout.addWidget(self.breadcrumb)
        
        # 主内容区
        self.content_stack = QStackedWidget()
        
        # 图标视图页面
        self.icon_view = self.create_icon_view()
        self.content_stack.addWidget(self.icon_view)
        
        # 列表视图页面
        self.list_view = self.create_list_view()
        self.content_stack.addWidget(self.list_view)
        
        layout.addWidget(self.content_stack)
        
        self.setLayout(layout)
    
    def create_icon_view(self) -> QWidget:
        """创建图标视图 - 类似Finder的图标排列"""
        widget = QWidget()
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        
        content = QWidget()
        grid = QGridLayout(content)
        grid.setSpacing(30)
        grid.setContentsMargins(30, 30, 30, 30)
        
        # 添加一些示例项目
        items = [
            ("📁", "模型文件", "folder"),
            ("🖼️", "测试图片", "folder"),
            ("📊", "分类结果", "folder"),
            ("⚙️", "配置文件", "file"),
            ("📄", "训练日志", "file"),
        ]
        
        row, col = 0, 0
        for icon, name, item_type in items:
            item_widget = self.create_icon_item(icon, name)
            grid.addWidget(item_widget, row, col)
            col += 1
            if col >= 4:
                col = 0
                row += 1
        
        grid.setRowStretch(row + 1, 1)
        
        scroll.setWidget(content)
        
        main_layout = QVBoxLayout(widget)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.addWidget(scroll)
        
        return widget
    
    def create_icon_item(self, icon: str, name: str) -> QWidget:
        """创建图标项"""
        widget = QFrame()
        widget.setFixedSize(120, 120)
        widget.setCursor(Qt.CursorShape.PointingHandCursor)
        widget.setStyleSheet(f"""
            QFrame {{
                background-color: transparent;
                border-radius: 8px;
            }}
            QFrame:hover {{
                background-color: {FinderColors.HOVER_BG};
            }}
        """)
        
        layout = QVBoxLayout(widget)
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.setSpacing(8)
        
        # 图标
        icon_label = QLabel(icon)
        icon_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        icon_label.setStyleSheet("""
            QLabel {
                font-size: 48px;
                background-color: transparent;
            }
        """)
        layout.addWidget(icon_label)
        
        # 文件名
        name_label = QLabel(name)
        name_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        name_label.setWordWrap(True)
        name_label.setStyleSheet(f"""
            QLabel {{
                font-size: 12px;
                color: {FinderColors.TEXT_PRIMARY};
                background-color: transparent;
            }}
        """)
        layout.addWidget(name_label)
        
        return widget
    
    def create_list_view(self) -> QWidget:
        """创建列表视图 - 类似Finder的详细列表"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(0, 0, 0, 0)
        
        # 创建列表
        list_widget = QListWidget()
        list_widget.setStyleSheet(f"""
            QListWidget {{
                background-color: {FinderColors.CONTENT_BG};
                border: none;
                outline: none;
                font-size: 13px;
            }}
            QListWidget::item {{
                padding: 6px 20px;
                border: none;
            }}
            QListWidget::item:hover {{
                background-color: {FinderColors.HOVER_BG};
            }}
            QListWidget::item:selected {{
                background-color: {FinderColors.SELECTION_BG};
                color: {FinderColors.SELECTION_TEXT};
            }}
        """)
        
        # 添加示例项
        items = [
            ("📁", "模型文件", "今天 下午3:24", "2.3 GB"),
            ("🖼️", "测试图片", "今天 下午2:15", "156 MB"),
            ("📊", "分类结果", "昨天 上午10:30", "45 KB"),
            ("⚙️", "配置文件", "2天前", "12 KB"),
            ("📄", "训练日志", "3天前", "890 KB"),
        ]
        
        for icon, name, date, size in items:
            item = QListWidgetItem(f"  {icon}  {name:<30}  {date:<20}  {size}")
            item.setFont(QFont("SF Pro Text", 13))
            list_widget.addItem(item)
        
        layout.addWidget(list_widget)
        return widget
    
    def switch_view(self, view_type: str):
        """切换视图类型"""
        if view_type == "icon_view":
            self.content_stack.setCurrentIndex(0)
        elif view_type == "list_view":
            self.content_stack.setCurrentIndex(1)


# ==================== Finder风格状态栏 ====================
class FinderStatusBar(QWidget):
    """Finder底部状态栏"""
    
    def __init__(self):
        super().__init__()
        self.setup_ui()
        
    def setup_ui(self):
        self.setFixedHeight(22)
        self.setStyleSheet(f"""
            QWidget {{
                background-color: {FinderColors.TOOLBAR_BG};
                border-top: 1px solid {FinderColors.BORDER};
            }}
        """)
        
        layout = QHBoxLayout()
        layout.setContentsMargins(12, 0, 12, 0)
        
        self.info_label = QLabel("5 个项目，4.2 GB 可用")
        self.info_label.setStyleSheet(f"""
            QLabel {{
                color: {FinderColors.TEXT_SECONDARY};
                font-size: 11px;
            }}
        """)
        layout.addWidget(self.info_label)
        
        layout.addStretch()
        
        self.setLayout(layout)
    
    def update_info(self, text: str):
        """更新状态信息"""
        self.info_label.setText(text)


# ==================== Finder主窗口 ====================
class FinderMainWindow(QMainWindow):
    """Finder风格主窗口"""
    
    def __init__(self):
        super().__init__()
        self.setup_ui()
        
    def setup_ui(self):
        self.setWindowTitle("JiLing 服装分类系统")
        self.setMinimumSize(1000, 700)
        
        # 设置窗口样式 - 类似macOS
        self.setStyleSheet(f"""
            QMainWindow {{
                background-color: {FinderColors.WINDOW_BG};
            }}
        """)
        
        # 中心部件
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        layout = QVBoxLayout(central_widget)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        
        # 工具栏
        self.toolbar = FinderToolbar()
        self.toolbar.actionTriggered.connect(self.handle_toolbar_action)
        layout.addWidget(self.toolbar)
        
        # 主内容区 - 分割器
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setHandleWidth(1)
        splitter.setStyleSheet(f"""
            QSplitter::handle {{
                background-color: {FinderColors.BORDER};
            }}
        """)
        
        # 侧边栏
        self.sidebar = FinderSidebar()
        self.sidebar.itemClicked.connect(self.handle_sidebar_click)
        splitter.addWidget(self.sidebar)
        
        # 内容区
        self.content_area = FinderContentArea()
        splitter.addWidget(self.content_area)
        
        # 设置初始比例
        splitter.setSizes([200, 800])
        
        layout.addWidget(splitter)
        
        # 状态栏
        self.status_bar = FinderStatusBar()
        layout.addWidget(self.status_bar)
    
    def handle_toolbar_action(self, action: str):
        """处理工具栏动作"""
        print(f"工具栏动作: {action}")
        
        if action in ["icon_view", "list_view", "column_view"]:
            self.content_area.switch_view(action)
            self.status_bar.update_info(f"切换到 {action} 视图")
    
    def handle_sidebar_click(self, category: str, item: str):
        """处理侧边栏点击"""
        print(f"侧边栏: {category} - {item}")
        self.content_area.breadcrumb.setText(f"JiLing 服装分类系统  ›  {item}")
        self.status_bar.update_info(f"正在查看: {item}")


# ==================== 主程序 ====================
def main():
    app = QApplication(sys.argv)
    
    # 设置应用样式 - 使用Fusion获得更好的跨平台效果
    app.setStyle("Fusion")
    
    # 设置字体
    if sys.platform == "darwin":  # macOS
        app.setFont(QFont("SF Pro Text", 13))
    else:
        app.setFont(QFont("Segoe UI", 10))
    
    window = FinderMainWindow()
    window.show()
    
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
