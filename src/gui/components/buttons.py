"""
侧边栏按钮组件 - VS Code Activity Bar Style
"""

from PySide6.QtWidgets import QPushButton, QApplication
from PySide6.QtCore import QSize, QByteArray, Qt
from PySide6.QtGui import QIcon, QPixmap, QPainter
from PySide6.QtSvg import QSvgRenderer

from ..theme import StyleSheet, theme_manager, ColorTokens


class SidebarButton(QPushButton):
    """
    侧边栏按钮 - VS Code Activity Bar Style

    设计规格:
    - 尺寸: 50x50px
    - 默认: 透明背景, 图标颜色
    - Hover: 透明背景, 高亮图标
    - Active: 左侧 2px 边框
    """

    def __init__(self, icon_text: str = None, svg_template: str = None, icon_size: tuple = (24, 24), parent=None):
        super().__init__(parent)
        self.setFixedSize(50, 50)
        self.setCheckable(True)
        self.setStyleSheet(StyleSheet.sidebar_btn())

        self._svg_template = svg_template
        self._icon_text = icon_text
        self._icon_size = icon_size

        if svg_template:
            self._update_icon(theme_manager.get_color(ColorTokens.ICON_DEFAULT))
        elif icon_text:
            self.setText(icon_text)

        # 监听主题变化
        theme_manager.theme_changed.connect(self._on_theme_changed)

    def _on_theme_changed(self, theme: str):
        """主题变化时更新样式"""
        self.setStyleSheet(StyleSheet.sidebar_btn())
        if self._svg_template:
            self._update_icon(theme_manager.get_color(ColorTokens.ICON_DEFAULT))

    def _update_icon(self, color: str):
        """更新 SVG 图标颜色 - 支持高DPI渲染"""
        if not self._svg_template:
            return

        svg_data = self._svg_template.replace("{color}", color)
        renderer = QSvgRenderer(QByteArray(svg_data.encode()))
        w, h = self._icon_size

        # 获取设备像素比以支持高DPI显示
        dpr = QApplication.primaryScreen().devicePixelRatio() if QApplication.primaryScreen() else 1.0

        # 创建高分辨率 pixmap
        pixmap = QPixmap(int(w * dpr), int(h * dpr))
        pixmap.setDevicePixelRatio(dpr)
        pixmap.fill(Qt.transparent)

        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.Antialiasing, True)
        painter.setRenderHint(QPainter.SmoothPixmapTransform, True)
        renderer.render(painter)
        painter.end()

        self.setIcon(QIcon(pixmap))
        self.setIconSize(QSize(w, h))

    def enterEvent(self, event):
        """鼠标进入时高亮图标"""
        if self._svg_template:
            self._update_icon(theme_manager.get_color(ColorTokens.ICON_HOVER))
        super().enterEvent(event)

    def leaveEvent(self, event):
        """鼠标离开时恢复图标颜色"""
        if self._svg_template:
            self._update_icon(theme_manager.get_color(ColorTokens.ICON_DEFAULT))
        super().leaveEvent(event)
