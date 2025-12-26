"""
训练页面 - 模型训练和监控
"""

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QFrame, QSizePolicy
)
from PySide6.QtCore import Qt, Signal

from ..theme import StyleSheet, theme_manager, ColorTokens
from ..utils import FontManager
from ..components import SliderRow, ComboRow, FileRow, ParamCard, TerminalOutput, WindowControlBar


class TrainingPage(QWidget):
    """训练页面 - 模型训练和监控"""

    # 信号
    start_training_requested = Signal()
    stop_training_requested = Signal()
    # 窗口控制信号
    minimize_requested = Signal()
    maximize_requested = Signal()
    close_requested = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._init_ui()

        # 监听主题变化
        theme_manager.theme_changed.connect(self._on_theme_changed)

    def _on_theme_changed(self, theme: str):
        """主题变化时更新样式"""
        self._param_area.setStyleSheet(StyleSheet.param_area())
        self._terminal_area.setStyleSheet(StyleSheet.terminal_area())
        self._btn_start.setStyleSheet(StyleSheet.btn_start())
        self._btn_stop.setStyleSheet(StyleSheet.btn_stop())
        for divider in self._dividers:
            divider.setStyleSheet(StyleSheet.divider_h())
        # 更新标题样式
        self._page_title.setStyleSheet(StyleSheet.page_title())
        self._terminal_title.setStyleSheet(StyleSheet.terminal_title())
        # 更新按钮分割线
        self._btn_divider.setStyleSheet(f"background-color: {theme_manager.get_color(ColorTokens.BORDER_MUTED)};")

    def _init_ui(self):
        """初始化 UI"""
        self._dividers = []

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # 左侧参数区 (占比 40%)
        self._param_area = self._create_param_area()
        layout.addWidget(self._param_area, 4)

        # 右侧终端区 (占比 60%)
        self._terminal_area = self._create_terminal_area()
        layout.addWidget(self._terminal_area, 6)

    def _create_param_area(self) -> QWidget:
        """创建参数设置区域"""
        area = QWidget()
        area.setStyleSheet(StyleSheet.param_area())
        area.setObjectName("paramArea")

        layout = QVBoxLayout(area)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # 页面标题区域
        title_bar = QWidget()
        title_bar.setFixedHeight(35)
        title_layout = QHBoxLayout(title_bar)
        title_layout.setContentsMargins(20, 0, 20, 0)

        self._page_title = QLabel("MODEL TRAINING")
        self._page_title.setStyleSheet(StyleSheet.page_title())
        self._page_title.setFont(FontManager.title_font())
        title_layout.addWidget(self._page_title)
        layout.addWidget(title_bar)

        # 内容区域
        content_widget = QWidget()
        content_layout = QVBoxLayout(content_widget)
        content_layout.setContentsMargins(0, 10, 0, 10)
        content_layout.setSpacing(0)

        # 参数卡片1: 训练模式、模型类型、模型位置
        card1 = ParamCard()

        self.mode_combo = ComboRow("训练模式", [
            "从预训练权重开始",
            "从已有模型继续训练",
            "Fine-tuning已有模型"
        ])
        card1.add_row(self.mode_combo)

        self.model_combo = ComboRow("模型类型", [
            "tf_efficientnetv2_s",
            "convnext_tiny",
            "resnet50",
            "vit_base_patch16_224",
            "swin_tiny_patch4_window7_224"
        ])
        card1.add_row(self.model_combo)

        self.base_model_row = FileRow("基础模型", "", is_folder=False)
        card1.add_row(self.base_model_row)

        content_layout.addWidget(card1)

        # 分割线
        divider1 = self._create_divider()
        content_layout.addWidget(divider1)

        # 参数卡片2: 滑块参数
        card2 = ParamCard()

        # 训练轮数: 范围1-100, 默认10
        self.epochs_slider = SliderRow("训练轮数", 1, 100, 10)
        card2.add_row(self.epochs_slider)

        # 批次大小: 范围1-32, 默认8
        self.batch_slider = SliderRow("批次大小", 1, 32, 8)
        card2.add_row(self.batch_slider)

        # 学习率: 滑块值1-100, scale=0.0001, 实际范围0.0001-0.01
        self.lr_slider = SliderRow("学习率", 1, 100, 10,
            display_format="lr", scale=0.0001,
            min_label_text="0.0001", max_label_text="0.01",
            value_width=60)
        card2.add_row(self.lr_slider)

        # 验证比例: 范围10-50, 默认20
        self.val_slider = SliderRow("验证比例", 10, 50, 20,
            display_format="percent", scale=1.0,
            min_label_text="10%", max_label_text="50%",
            value_width=60)
        card2.add_row(self.val_slider)

        content_layout.addWidget(card2)

        # 分割线
        divider2 = self._create_divider()
        content_layout.addWidget(divider2)

        # 参数卡片3: 数据路径
        card3 = ParamCard()
        self.data_path_row = FileRow("数据路径", "", is_folder=True)
        card3.add_row(self.data_path_row)
        content_layout.addWidget(card3)

        content_layout.addStretch()
        layout.addWidget(content_widget)

        return area

    def _create_divider(self) -> QFrame:
        """创建分割线"""
        divider = QFrame()
        divider.setStyleSheet(f"background-color: {theme_manager.get_color(ColorTokens.DIVIDER)}; margin: 10px 20px;")
        divider.setFixedHeight(1)
        self._dividers.append(divider)
        return divider

    def _create_terminal_area(self) -> QWidget:
        """创建终端输出区域"""
        area = QWidget()
        area.setStyleSheet(StyleSheet.terminal_area())
        area.setObjectName("terminalArea")

        layout = QVBoxLayout(area)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # 顶部区域（72px）：窗口控制按钮 + 标题
        top_bar = QWidget()
        top_bar.setFixedHeight(72)
        top_outer_layout = QVBoxLayout(top_bar)
        top_outer_layout.setContentsMargins(0, 0, 0, 0)
        top_outer_layout.setSpacing(0)

        # 窗口控制按钮行（32px）- 贴顶靠右
        self._window_controls = WindowControlBar()
        self._window_controls.minimize_clicked.connect(self.minimize_requested.emit)
        self._window_controls.maximize_clicked.connect(self.maximize_requested.emit)
        self._window_controls.close_clicked.connect(self.close_requested.emit)
        top_outer_layout.addWidget(self._window_controls)

        # 标题行
        title_row = QWidget()
        title_row_layout = QHBoxLayout(title_row)
        title_row_layout.setContentsMargins(24, 0, 24, 0)

        self._terminal_title = QLabel("终端")
        self._terminal_title.setStyleSheet(StyleSheet.terminal_title())
        self._terminal_title.setFont(FontManager.header_font())
        title_row_layout.addWidget(self._terminal_title)
        title_row_layout.addStretch()

        top_outer_layout.addWidget(title_row)
        layout.addWidget(top_bar)

        # 分割线
        divider1 = QFrame()
        divider1.setStyleSheet(StyleSheet.divider_h())
        layout.addWidget(divider1)
        self._dividers.append(divider1)

        # 终端输出
        self.terminal = TerminalOutput()
        layout.addWidget(self.terminal, 1)

        # 分割线
        divider2 = QFrame()
        divider2.setStyleSheet(StyleSheet.divider_h())
        layout.addWidget(divider2)
        self._dividers.append(divider2)

        # 底部按钮栏
        bottom_bar = QWidget()
        bottom_bar.setFixedHeight(80)
        bottom_layout = QHBoxLayout(bottom_bar)
        bottom_layout.setContentsMargins(0, 0, 0, 0)
        bottom_layout.setSpacing(0)

        self._btn_start = QPushButton("开始训练")
        self._btn_start.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self._btn_start.setStyleSheet(StyleSheet.btn_start())
        self._btn_start.setFont(FontManager.action_button_font())
        self._btn_start.clicked.connect(self.start_training_requested.emit)
        bottom_layout.addWidget(self._btn_start)

        # 按钮分割线
        self._btn_divider = QFrame()
        self._btn_divider.setFixedWidth(1)
        self._btn_divider.setStyleSheet(f"background-color: {theme_manager.get_color(ColorTokens.BORDER_MUTED)};")
        bottom_layout.addWidget(self._btn_divider)

        self._btn_stop = QPushButton("停止训练")
        self._btn_stop.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self._btn_stop.setStyleSheet(StyleSheet.btn_stop())
        self._btn_stop.setFont(FontManager.action_button_font())
        self._btn_stop.clicked.connect(self.stop_training_requested.emit)
        bottom_layout.addWidget(self._btn_stop)

        layout.addWidget(bottom_bar)

        return area

    # ========== 公开接口 ==========

    def get_training_params(self) -> dict:
        """获取训练参数"""
        return {
            "mode": self.mode_combo.currentText(),
            "model_type": self.model_combo.currentText(),
            "base_model_path": self.base_model_row.path(),
            "epochs": self.epochs_slider.value(),
            "batch_size": self.batch_slider.value(),
            "learning_rate": self.lr_slider.value() * self.lr_slider.scale,
            "val_split": self.val_slider.value() / 100.0,
            "data_path": self.data_path_row.path(),
        }

    def set_training_state(self, is_training: bool):
        """设置训练状态"""
        self._btn_start.setEnabled(not is_training)
        self._btn_stop.setEnabled(is_training)

    def update_maximize_state(self, is_maximized: bool):
        """更新最大化状态时的样式"""
        # 内部圆角 = 窗口圆角(10) - 边框宽度(1) = 9
        corner_radius = 0 if is_maximized else 9
        self._btn_stop.setStyleSheet(StyleSheet.btn_stop(corner_radius=corner_radius))

    def append_log(self, text: str):
        """追加日志"""
        self.terminal.append(text)

    def clear_log(self):
        """清空日志"""
        self.terminal.clear()
