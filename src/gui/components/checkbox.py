"""
自定义复选框组件 - VS Code 风格
"""

from PySide6.QtWidgets import QWidget
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QPainter, QColor, QPen, QPainterPath

from ..theme import theme_manager, ColorTokens


class VSCheckBox(QWidget):
    """
    自定义复选框 - VS Code 风格
    使用 QPainter 绘制勾选标记
    """

    toggled = Signal(bool)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._checked = False
        self._hovered = False
        self.setFixedSize(18, 18)
        self.setCursor(Qt.PointingHandCursor)

        # 监听主题变化
        theme_manager.theme_changed.connect(self._on_theme_changed)

    def _on_theme_changed(self, theme: str):
        """主题变化时更新"""
        self.update()

    def isChecked(self) -> bool:
        return self._checked

    def setChecked(self, checked: bool):
        if self._checked != checked:
            self._checked = checked
            self.update()
            self.toggled.emit(checked)

    def toggle(self):
        self.setChecked(not self._checked)

    def enterEvent(self, event):
        self._hovered = True
        self.update()
        super().enterEvent(event)

    def leaveEvent(self, event):
        self._hovered = False
        self.update()
        super().leaveEvent(event)

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.toggle()
        super().mousePressEvent(event)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing, True)
        painter.setRenderHint(QPainter.SmoothPixmapTransform, True)

        # 背景和边框颜色
        if self._checked:
            accent = theme_manager.get_color(ColorTokens.ACCENT)
            accent_hover = theme_manager.get_color(ColorTokens.ACCENT_HOVER)
            bg_color = QColor(accent_hover) if self._hovered else QColor(accent)
            border_color = bg_color
        else:
            bg_color = QColor(theme_manager.get_color(ColorTokens.BG_INPUT))
            accent = theme_manager.get_color(ColorTokens.ACCENT)
            border = theme_manager.get_color(ColorTokens.BORDER)
            border_color = QColor(accent) if self._hovered else QColor(border)

        # 绘制背景
        painter.setPen(QPen(border_color, 1))
        painter.setBrush(bg_color)
        painter.drawRoundedRect(self.rect().adjusted(0, 0, -1, -1), 3, 3)

        # 绘制勾选标记
        if self._checked:
            painter.setPen(QPen(QColor("#FFFFFF"), 2, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))
            path = QPainterPath()
            path.moveTo(4, 9)
            path.lineTo(7, 12)
            path.lineTo(14, 5)
            painter.drawPath(path)

        painter.end()
