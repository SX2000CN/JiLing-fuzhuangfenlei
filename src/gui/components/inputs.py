"""
输入组件 - 滑块行、下拉框行、文件选择行
匹配 native_ui.py 的水平布局风格
"""

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QSlider,
    QComboBox, QLineEdit, QPushButton, QFileDialog
)
from PySide6.QtCore import Qt, Signal

from ..theme import StyleSheet, theme_manager, ColorTokens
from ..utils import FontManager


class ParamRow(QWidget):
    """参数行基类 - 标签在左，控件在右"""

    def __init__(self, label_text: str, parent=None):
        super().__init__(parent)
        self.setFixedHeight(40)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(20, 0, 20, 0)
        layout.setAlignment(Qt.AlignVCenter)

        # 标签
        self._label = QLabel(label_text)
        self._label.setStyleSheet(StyleSheet.param_label())
        self._label.setFont(FontManager.label_font())
        layout.addWidget(self._label)

        layout.addStretch()

        # 输入区域（子类实现）
        self.input_layout = QHBoxLayout()
        self.input_layout.setSpacing(0)
        self.input_layout.setAlignment(Qt.AlignVCenter)
        layout.addLayout(self.input_layout)

        # 监听主题变化
        theme_manager.theme_changed.connect(self._on_theme_changed)

    def _on_theme_changed(self, theme: str):
        self._label.setStyleSheet(StyleSheet.param_label())


class ComboRow(ParamRow):
    """下拉框行"""

    currentIndexChanged = Signal(int)

    def __init__(self, label: str, items: list, parent=None):
        super().__init__(label, parent)

        self.combo = QComboBox()
        self.combo.addItems(items)
        self.combo.setFixedWidth(220)
        self.combo.setStyleSheet(StyleSheet.input_box())
        self.combo.setFont(FontManager.input_font())
        self.combo.currentIndexChanged.connect(self.currentIndexChanged.emit)
        self.input_layout.addWidget(self.combo)

    def _on_theme_changed(self, theme: str):
        super()._on_theme_changed(theme)
        self.combo.setStyleSheet(StyleSheet.input_box())

    def currentIndex(self) -> int:
        return self.combo.currentIndex()

    def setCurrentIndex(self, index: int):
        self.combo.setCurrentIndex(index)

    def currentText(self) -> str:
        return self.combo.currentText()


class FileRow(ParamRow):
    """文件选择行"""

    pathChanged = Signal(str)

    def __init__(self, label: str, placeholder: str = "", is_folder: bool = False, parent=None):
        super().__init__(label, parent)
        self._is_folder = is_folder

        self.path_edit = QLineEdit()
        self.path_edit.setPlaceholderText(placeholder)
        self.path_edit.setFixedWidth(190)
        self.path_edit.setReadOnly(True)
        self.path_edit.setStyleSheet(StyleSheet.input_box_narrow())
        self.path_edit.setFont(FontManager.input_font())
        self.path_edit.textChanged.connect(self.pathChanged.emit)
        self.input_layout.addWidget(self.path_edit)

        self.input_layout.addSpacing(4)

        # 浏览按钮 - 26x26px
        self.browse_btn = QPushButton("...")
        self.browse_btn.setFixedSize(26, 26)
        self.browse_btn.setStyleSheet(StyleSheet.browse_button())
        self.browse_btn.clicked.connect(self._browse)
        self.input_layout.addWidget(self.browse_btn)

    def _on_theme_changed(self, theme: str):
        super()._on_theme_changed(theme)
        self.path_edit.setStyleSheet(StyleSheet.input_box_narrow())
        self.browse_btn.setStyleSheet(StyleSheet.browse_button())

    def _browse(self):
        if self._is_folder:
            path = QFileDialog.getExistingDirectory(self, "选择文件夹")
        else:
            path, _ = QFileDialog.getOpenFileName(self, "选择文件", "", "模型文件 (*.pth *.pt);;所有文件 (*)")
        if path:
            self.path_edit.setText(path)

    def path(self) -> str:
        return self.path_edit.text()

    def setPath(self, path: str):
        self.path_edit.setText(path)


class SliderRow(QWidget):
    """滑块行 - 支持多种显示格式"""

    valueChanged = Signal(int)

    def __init__(self, label: str, min_val: int, max_val: int, default_val: int,
                 suffix: str = "", display_format: str = "int", scale: float = 1.0,
                 min_label_text: str = None, max_label_text: str = None,
                 value_width: int = 50, parent=None):
        """
        Args:
            label: 标签文本
            min_val: 滑块最小值
            max_val: 滑块最大值
            default_val: 默认值
            suffix: 后缀
            display_format: 显示格式 - "int", "float", "percent", "lr" (学习率)
            scale: 缩放系数 (实际值 = 滑块值 * scale)
            min_label_text: 左侧刻度标签 (None则自动生成)
            max_label_text: 右侧刻度标签 (None则自动生成)
            value_width: 数值框宽度
        """
        super().__init__(parent)
        self.setFixedHeight(40)
        self.display_format = display_format
        self.suffix = suffix
        self.scale = scale

        layout = QHBoxLayout(self)
        layout.setContentsMargins(20, 0, 20, 0)
        layout.setSpacing(8)
        layout.setAlignment(Qt.AlignVCenter)

        # 标签
        self._label = QLabel(label)
        self._label.setStyleSheet(StyleSheet.param_label())
        self._label.setFont(FontManager.label_font())
        layout.addWidget(self._label)

        layout.addStretch()

        # 滑块容器
        slider_container = QWidget()
        slider_container.setFixedWidth(177)
        slider_layout = QVBoxLayout(slider_container)
        slider_layout.setContentsMargins(0, 0, 0, 0)
        slider_layout.setSpacing(2)

        # 滑块
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setMinimum(min_val)
        self.slider.setMaximum(max_val)
        self.slider.setValue(default_val)
        self.slider.setStyleSheet(StyleSheet.slider())
        self.slider.valueChanged.connect(self._on_value_changed)
        slider_layout.addWidget(self.slider)

        # 刻度标签
        labels_layout = QHBoxLayout()
        labels_layout.setContentsMargins(0, 0, 0, 0)

        min_text = min_label_text if min_label_text else self._format_value(min_val)
        max_text = max_label_text if max_label_text else self._format_value(max_val)

        self._min_label = QLabel(min_text)
        self._min_label.setStyleSheet(StyleSheet.slider_label())
        self._min_label.setFont(FontManager.slider_tick_font())
        self._max_label = QLabel(max_text)
        self._max_label.setStyleSheet(StyleSheet.slider_label())
        self._max_label.setFont(FontManager.slider_tick_font())
        labels_layout.addWidget(self._min_label)
        labels_layout.addStretch()
        labels_layout.addWidget(self._max_label)
        slider_layout.addLayout(labels_layout)

        layout.addWidget(slider_container)

        # 数值框
        self.value_label = QLabel()
        self.value_label.setStyleSheet(StyleSheet.value_box(value_width))
        self.value_label.setFont(FontManager.input_font())
        self.value_label.setAlignment(Qt.AlignCenter)
        self.value_label.setFixedSize(value_width, 26)
        self._update_value_display(default_val)
        layout.addWidget(self.value_label)

        # 监听主题变化
        theme_manager.theme_changed.connect(self._on_theme_changed)

    def _on_theme_changed(self, theme: str):
        self._label.setStyleSheet(StyleSheet.param_label())
        self.slider.setStyleSheet(StyleSheet.slider())
        self._min_label.setStyleSheet(StyleSheet.slider_label())
        self._max_label.setStyleSheet(StyleSheet.slider_label())
        self.value_label.setStyleSheet(StyleSheet.value_box(self.value_label.width()))

    def _on_value_changed(self, value: int):
        self._update_value_display(value)
        self.valueChanged.emit(value)

    def _format_value(self, value: int) -> str:
        """根据格式类型格式化数值"""
        real_val = value * self.scale

        if self.display_format == "int":
            return f"{int(real_val)}{self.suffix}"
        elif self.display_format == "float":
            return f"{real_val:.2f}{self.suffix}"
        elif self.display_format == "percent":
            return f"{real_val:.1f}%"
        elif self.display_format == "lr":
            # 学习率使用科学计数法或小数
            if real_val >= 0.01:
                return f"{real_val:.3f}"
            else:
                return f"{real_val:.4f}"
        else:
            return f"{real_val}{self.suffix}"

    def _update_value_display(self, value: int):
        self.value_label.setText(self._format_value(value))

    def value(self) -> int:
        """返回滑块原始值"""
        return self.slider.value()

    def setValue(self, value: int):
        """设置滑块值"""
        self.slider.setValue(value)
        self._update_value_display(value)
