"""
设置页组件 - VS Code 风格
包含 SettingsRow 和 SettingsCheckRow

关键修复：高亮边距问题
- 使用正确的 padding 让高亮背景与文字有视觉间距
- scroll_layout 提供外部间距 (12px)
- 组件内部提供 padding (12px)
- 总效果：高亮边缘与文字间有 12px 可见间距
"""

from PySide6.QtWidgets import QWidget, QHBoxLayout, QVBoxLayout, QLabel
from PySide6.QtCore import Qt
from PySide6.QtGui import QPainter, QColor

from ..theme import StyleSheet, theme_manager, ColorTokens
from ..utils import FontManager
from .checkbox import VSCheckBox


class SettingsRow(QWidget):
    """
    设置行组件 - VS Code 风格
    每行包含：标题、描述、输入控件
    鼠标悬停时整行高亮

    布局说明：
    - 外部 scroll_layout 边距: (12, 0, 12, 24) 推动本组件向内
    - 内部 layout 边距: (12, 10, 12, 10) 让文字与高亮边缘有间距
    - 高亮填充整个 widget，但文字有 12px 内边距
    """

    def __init__(self, title: str, description: str = "", parent=None):
        super().__init__(parent)
        self.setMinimumHeight(50)
        self.setCursor(Qt.PointingHandCursor)

        self._hovered = False

        # 主布局 - 内部边距让高亮背景与文字有舒适间距
        layout = QHBoxLayout(self)
        layout.setContentsMargins(12, 10, 12, 10)
        layout.setSpacing(16)

        # 左侧文字区
        text_area = QWidget()
        text_layout = QVBoxLayout(text_area)
        text_layout.setContentsMargins(0, 0, 0, 0)
        text_layout.setSpacing(2)

        # 标题
        self.title_label = QLabel(title)
        self.title_label.setStyleSheet(StyleSheet.settings_row_label())
        self.title_label.setFont(FontManager.label_font())
        text_layout.addWidget(self.title_label)

        # 描述（可选）
        if description:
            self.desc_label = QLabel(description)
            self.desc_label.setStyleSheet(StyleSheet.settings_row_desc())
            self.desc_label.setFont(FontManager.small_font())
            self.desc_label.setWordWrap(True)
            text_layout.addWidget(self.desc_label)

        layout.addWidget(text_area, 1)

        # 右侧控件区
        self.control_layout = QHBoxLayout()
        self.control_layout.setContentsMargins(0, 0, 0, 0)
        self.control_layout.setSpacing(8)
        layout.addLayout(self.control_layout)

        # 保存控件引用
        self._control = None

        # 监听主题变化
        theme_manager.theme_changed.connect(self._on_theme_changed)

    def _on_theme_changed(self, theme: str):
        """主题变化时更新样式"""
        self.title_label.setStyleSheet(StyleSheet.settings_row_label())
        if hasattr(self, 'desc_label'):
            self.desc_label.setStyleSheet(StyleSheet.settings_row_desc())
        self.update()

    def set_control(self, control: QWidget):
        """设置右侧的输入控件"""
        self._control = control
        self.control_layout.addWidget(control)

    def get_control(self):
        """获取控件"""
        return self._control

    def enterEvent(self, event):
        """鼠标进入时高亮"""
        self._hovered = True
        self.update()
        super().enterEvent(event)

    def leaveEvent(self, event):
        """鼠标离开时恢复"""
        self._hovered = False
        self.update()
        super().leaveEvent(event)

    def paintEvent(self, event):
        """绘制背景（悬停高亮）"""
        if self._hovered:
            painter = QPainter(self)
            painter.setRenderHint(QPainter.Antialiasing, True)
            hover_color = QColor(theme_manager.get_color(ColorTokens.HOVER_BG))
            painter.setBrush(hover_color)
            painter.setPen(Qt.NoPen)
            painter.drawRoundedRect(self.rect(), 3, 3)
            painter.end()
        super().paintEvent(event)


class SettingsCheckRow(QWidget):
    """
    设置复选框行 - VS Code 风格
    复选框在最左侧，使用自定义 VSCheckBox
    """

    def __init__(self, title: str, description: str = "", parent=None):
        super().__init__(parent)
        self.setMinimumHeight(40)
        self.setCursor(Qt.PointingHandCursor)

        self._hovered = False

        # 主布局 - 内部边距让高亮背景与文字有舒适间距
        layout = QHBoxLayout(self)
        layout.setContentsMargins(12, 10, 12, 10)
        layout.setSpacing(10)

        # 复选框 - 使用自定义 VSCheckBox
        self.checkbox = VSCheckBox()
        layout.addWidget(self.checkbox)

        # 文字区
        text_area = QWidget()
        text_layout = QVBoxLayout(text_area)
        text_layout.setContentsMargins(0, 0, 0, 0)
        text_layout.setSpacing(2)

        # 标题
        self.title_label = QLabel(title)
        self.title_label.setStyleSheet(StyleSheet.settings_row_label())
        self.title_label.setFont(FontManager.label_font())
        text_layout.addWidget(self.title_label)

        # 描述（可选）
        if description:
            self.desc_label = QLabel(description)
            self.desc_label.setStyleSheet(StyleSheet.settings_row_desc())
            self.desc_label.setFont(FontManager.small_font())
            self.desc_label.setWordWrap(True)
            text_layout.addWidget(self.desc_label)

        layout.addWidget(text_area, 1)

        # 监听主题变化
        theme_manager.theme_changed.connect(self._on_theme_changed)

    def _on_theme_changed(self, theme: str):
        """主题变化时更新样式"""
        self.title_label.setStyleSheet(StyleSheet.settings_row_label())
        if hasattr(self, 'desc_label'):
            self.desc_label.setStyleSheet(StyleSheet.settings_row_desc())
        self.update()

    def isChecked(self) -> bool:
        return self.checkbox.isChecked()

    def setChecked(self, checked: bool):
        self.checkbox.setChecked(checked)

    def enterEvent(self, event):
        """鼠标进入时高亮"""
        self._hovered = True
        self.update()
        super().enterEvent(event)

    def leaveEvent(self, event):
        """鼠标离开时恢复"""
        self._hovered = False
        self.update()
        super().leaveEvent(event)

    def mousePressEvent(self, event):
        """点击整行切换复选框"""
        if event.button() == Qt.LeftButton:
            self.checkbox.toggle()
        super().mousePressEvent(event)

    def paintEvent(self, event):
        """绘制背景（悬停高亮）"""
        if self._hovered:
            painter = QPainter(self)
            painter.setRenderHint(QPainter.Antialiasing, True)
            hover_color = QColor(theme_manager.get_color(ColorTokens.HOVER_BG))
            painter.setBrush(hover_color)
            painter.setPen(Qt.NoPen)
            painter.drawRoundedRect(self.rect(), 3, 3)
            painter.end()
        super().paintEvent(event)
