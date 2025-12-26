"""
分类页面 - 图像分类和结果展示
"""

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QFrame, QProgressBar, QSizePolicy
)
from PySide6.QtCore import Qt, Signal

from ..theme import StyleSheet, theme_manager, ColorTokens
from ..utils import FontManager
from ..components import ParamCard, FileRow, WindowControlBar
from ..widgets import ClassificationResultTable


class ClassificationPage(QWidget):
    """分类页面 - 图像分类和结果展示"""

    # 信号
    load_model_requested = Signal()
    use_default_model_requested = Signal()
    start_classification_requested = Signal()
    clear_results_requested = Signal()
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
        # 更新区域样式
        self._param_area.setStyleSheet(StyleSheet.param_area())
        self._result_area.setStyleSheet(StyleSheet.terminal_area())
        # 更新按钮样式
        self._btn_load_model.setStyleSheet(StyleSheet.primary_button())
        self._btn_use_default.setStyleSheet(StyleSheet.secondary_button())
        self._btn_classify.setStyleSheet(StyleSheet.action_button_classify())
        self._btn_clear.setStyleSheet(StyleSheet.action_button_clear())
        # 更新分割线
        for divider in self._dividers:
            divider.setStyleSheet(StyleSheet.divider_h())
        # 更新标题样式
        self._page_title.setStyleSheet(StyleSheet.page_title())
        self._result_title.setStyleSheet(StyleSheet.terminal_title())
        # 更新按钮分割线
        self._btn_divider.setStyleSheet(f"background-color: {theme_manager.get_color(ColorTokens.BORDER_MUTED)};")
        # 更新统计标签
        self.stats_label.setStyleSheet(f"color: {theme_manager.get_color(ColorTokens.TEXT_PRIMARY)}; font-size: 12px;")
        # 更新模型状态标签
        self._model_status_label.setStyleSheet(StyleSheet.param_label())

    def _init_ui(self):
        """初始化 UI"""
        self._dividers = []

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # 左侧参数区 (占比 40%)
        self._param_area = self._create_param_area()
        layout.addWidget(self._param_area, 4)

        # 右侧结果区 (占比 60%)
        self._result_area = self._create_result_area()
        layout.addWidget(self._result_area, 6)

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

        self._page_title = QLabel("IMAGE CLASSIFICATION")
        self._page_title.setStyleSheet(StyleSheet.page_title())
        self._page_title.setFont(FontManager.title_font())
        title_layout.addWidget(self._page_title)
        layout.addWidget(title_bar)

        # 内容区域
        content_widget = QWidget()
        content_layout = QVBoxLayout(content_widget)
        content_layout.setContentsMargins(0, 10, 0, 10)
        content_layout.setSpacing(0)

        # 参数卡片1: 图像选择
        card1 = ParamCard()

        self.single_file_row = FileRow("单个文件", "", is_folder=False)
        card1.add_row(self.single_file_row)

        self.folder_row = FileRow("文件夹", "", is_folder=True)
        card1.add_row(self.folder_row)

        content_layout.addWidget(card1)

        # 分割线
        divider1 = self._create_divider()
        content_layout.addWidget(divider1)

        # 参数卡片2: 模型选择
        card2 = ParamCard()

        self.model_file_row = FileRow("模型文件", "", is_folder=False)
        card2.add_row(self.model_file_row)

        # 模型状态显示行
        status_row = self._create_model_status_row()
        card2.add_row(status_row)

        content_layout.addWidget(card2)

        # 分割线
        divider2 = self._create_divider()
        content_layout.addWidget(divider2)

        # 参数卡片3: 快速操作按钮
        card3 = ParamCard()
        btn_row = self._create_button_row()
        card3.add_row(btn_row)
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

    def _create_model_status_row(self) -> QWidget:
        """创建模型状态显示行"""
        row = QWidget()
        row.setFixedHeight(40)
        layout = QHBoxLayout(row)
        layout.setContentsMargins(20, 0, 20, 0)

        self._model_status_label = QLabel("模型状态")
        self._model_status_label.setStyleSheet(StyleSheet.param_label())
        self._model_status_label.setFont(FontManager.label_font())
        layout.addWidget(self._model_status_label)

        layout.addStretch()

        self.model_status_indicator = QLabel("未加载")
        self._update_model_status_style(False)
        self.model_status_indicator.setFont(FontManager.input_font())
        layout.addWidget(self.model_status_indicator)

        return row

    def _update_model_status_style(self, loaded: bool):
        """更新模型状态样式"""
        if loaded:
            color = "#89D185"  # 绿色
            bg = "rgba(137, 209, 133, 0.1)"
        else:
            color = "#F48771"  # 红色
            bg = "rgba(244, 135, 113, 0.1)"

        self.model_status_indicator.setStyleSheet(f"""
            QLabel {{
                color: {color};
                font-size: 12px;
                padding: 2px 8px;
                background-color: {bg};
                border-radius: 2px;
            }}
        """)

    def _create_button_row(self) -> QWidget:
        """创建按钮行"""
        row = QWidget()
        row.setFixedHeight(40)
        layout = QHBoxLayout(row)
        layout.setContentsMargins(20, 0, 20, 0)

        self._btn_load_model = QPushButton("加载模型")
        self._btn_load_model.setFixedHeight(26)
        self._btn_load_model.setStyleSheet(StyleSheet.primary_button())
        self._btn_load_model.setFont(FontManager.button_font())
        self._btn_load_model.clicked.connect(self.load_model_requested.emit)
        layout.addWidget(self._btn_load_model)

        self._btn_use_default = QPushButton("使用默认模型")
        self._btn_use_default.setFixedHeight(26)
        self._btn_use_default.setStyleSheet(StyleSheet.secondary_button())
        self._btn_use_default.setFont(FontManager.button_font())
        self._btn_use_default.clicked.connect(self.use_default_model_requested.emit)
        layout.addWidget(self._btn_use_default)

        layout.addStretch()

        return row

    def _create_result_area(self) -> QWidget:
        """创建结果显示区域"""
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
        title_row_layout.setContentsMargins(20, 0, 20, 0)

        self._result_title = QLabel("分类结果")
        self._result_title.setStyleSheet(StyleSheet.terminal_title())
        self._result_title.setFont(FontManager.header_font())
        title_row_layout.addWidget(self._result_title)
        title_row_layout.addStretch()

        # 统计信息
        self.stats_label = QLabel("共 0 张图片")
        self.stats_label.setStyleSheet(f"color: {theme_manager.get_color(ColorTokens.TEXT_PRIMARY)}; font-size: 12px;")
        self.stats_label.setFont(FontManager.input_font())
        title_row_layout.addWidget(self.stats_label)

        top_outer_layout.addWidget(title_row)
        layout.addWidget(top_bar)

        # 分割线
        divider1 = QFrame()
        divider1.setStyleSheet(StyleSheet.divider_h())
        layout.addWidget(divider1)
        self._dividers.append(divider1)

        # 进度条
        self.progress_bar = QProgressBar()
        self.progress_bar.setStyleSheet(StyleSheet.progress_bar())
        self.progress_bar.setTextVisible(False)
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)

        # 分类结果表格
        self.result_table = ClassificationResultTable()
        layout.addWidget(self.result_table)

        # 分割线
        divider2 = QFrame()
        divider2.setStyleSheet(StyleSheet.divider_h())
        layout.addWidget(divider2)
        self._dividers.append(divider2)

        # 底部按钮
        bottom_bar = QWidget()
        bottom_bar.setFixedHeight(80)
        bottom_layout = QHBoxLayout(bottom_bar)
        bottom_layout.setContentsMargins(0, 0, 0, 0)
        bottom_layout.setSpacing(0)

        self._btn_classify = QPushButton("开始分类")
        self._btn_classify.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self._btn_classify.setStyleSheet(StyleSheet.action_button_classify())
        self._btn_classify.setFont(FontManager.action_button_font())
        self._btn_classify.clicked.connect(self.start_classification_requested.emit)
        bottom_layout.addWidget(self._btn_classify)

        # 按钮分割线
        self._btn_divider = QFrame()
        self._btn_divider.setFixedWidth(1)
        self._btn_divider.setStyleSheet(f"background-color: {theme_manager.get_color(ColorTokens.BORDER_MUTED)};")
        bottom_layout.addWidget(self._btn_divider)

        self._btn_clear = QPushButton("清空结果")
        self._btn_clear.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self._btn_clear.setStyleSheet(StyleSheet.action_button_clear())
        self._btn_clear.setFont(FontManager.action_button_font())
        self._btn_clear.clicked.connect(self.clear_results_requested.emit)
        bottom_layout.addWidget(self._btn_clear)

        layout.addWidget(bottom_bar)

        return area

    # ========== 公开接口 ==========

    def set_model_status(self, loaded: bool, text: str = None):
        """设置模型加载状态"""
        if text:
            self.model_status_indicator.setText(text)
        else:
            self.model_status_indicator.setText("已加载" if loaded else "未加载")
        self._update_model_status_style(loaded)

    def set_progress(self, value: int, visible: bool = True):
        """设置进度条"""
        self.progress_bar.setVisible(visible)
        self.progress_bar.setValue(value)

    def set_stats(self, count: int):
        """设置统计信息"""
        self.stats_label.setText(f"共 {count} 张图片")

    def get_single_file_path(self) -> str:
        """获取单个文件路径"""
        return self.single_file_row.path()

    def get_folder_path(self) -> str:
        """获取文件夹路径"""
        return self.folder_row.path()

    def get_model_file_path(self) -> str:
        """获取模型文件路径"""
        return self.model_file_row.path()

    def clear_results(self):
        """清空结果"""
        self.result_table.setRowCount(0)
        self.set_stats(0)
        self.set_progress(0, False)

    def update_maximize_state(self, is_maximized: bool):
        """更新最大化状态时的样式"""
        # 内部圆角 = 窗口圆角(10) - 边框宽度(1) = 9
        corner_radius = 0 if is_maximized else 9
        self._btn_clear.setStyleSheet(StyleSheet.action_button_clear(corner_radius=corner_radius))
