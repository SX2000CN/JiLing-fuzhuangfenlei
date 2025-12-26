"""
卡片和容器组件
"""

from PySide6.QtWidgets import QWidget, QFrame, QVBoxLayout, QHBoxLayout, QLabel
from PySide6.QtCore import Qt
from PySide6.QtGui import QPainter, QPainterPath, QColor

from ..theme import StyleSheet, theme_manager, ColorTokens
from ..utils import FontManager


class ParamCard(QFrame):
    """参数卡片容器"""

    def __init__(self, title: str = "", parent=None):
        super().__init__(parent)
        self.setStyleSheet(StyleSheet.param_card())

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 16)
        layout.setSpacing(8)

        if title:
            title_label = QLabel(title)
            title_label.setStyleSheet(StyleSheet.page_title())
            title_label.setFont(FontManager.title_font())
            layout.addWidget(title_label)

        self._content_layout = layout

        # 监听主题变化
        theme_manager.theme_changed.connect(self._on_theme_changed)

    def _on_theme_changed(self, theme: str):
        self.setStyleSheet(StyleSheet.param_card())

    def add_row(self, widget: QWidget):
        self._content_layout.addWidget(widget)


class RoundedContainer(QWidget):
    """
    圆角容器组件 - 用于终端区等需要圆角的区域
    通过 paintEvent 绘制圆角矩形遮罩
    """

    def __init__(self, parent=None, corner_radius: int = 10, corners: str = "all"):
        super().__init__(parent)
        self._corner_radius = corner_radius
        self._corners = corners  # "all", "left", "right", "top", "bottom"

        # 监听主题变化
        theme_manager.theme_changed.connect(self._on_theme_changed)

    def _on_theme_changed(self, theme: str):
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing, True)

        bg_color = QColor(theme_manager.get_color(ColorTokens.BG_PRIMARY))
        painter.setBrush(bg_color)
        painter.setPen(Qt.NoPen)

        path = QPainterPath()
        r = self._corner_radius
        rect = self.rect()

        if self._corners == "all":
            path.addRoundedRect(rect, r, r)
        elif self._corners == "right":
            path.moveTo(rect.topLeft())
            path.lineTo(rect.topRight().x() - r, rect.topRight().y())
            path.arcTo(rect.topRight().x() - 2*r, rect.topRight().y(), 2*r, 2*r, 90, -90)
            path.lineTo(rect.bottomRight().x(), rect.bottomRight().y() - r)
            path.arcTo(rect.bottomRight().x() - 2*r, rect.bottomRight().y() - 2*r, 2*r, 2*r, 0, -90)
            path.lineTo(rect.bottomLeft())
            path.closeSubpath()
        elif self._corners == "left":
            path.moveTo(rect.topRight())
            path.lineTo(rect.topLeft().x() + r, rect.topLeft().y())
            path.arcTo(rect.topLeft().x(), rect.topLeft().y(), 2*r, 2*r, 90, 90)
            path.lineTo(rect.bottomLeft().x(), rect.bottomLeft().y() - r)
            path.arcTo(rect.bottomLeft().x(), rect.bottomLeft().y() - 2*r, 2*r, 2*r, 180, 90)
            path.lineTo(rect.bottomRight())
            path.closeSubpath()
        else:
            path.addRect(rect)

        painter.drawPath(path)
        super().paintEvent(event)
