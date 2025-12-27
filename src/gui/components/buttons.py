"""
侧边栏按钮组件 - VS Code Activity Bar Style + macOS 动画
"""

from PySide6.QtWidgets import QPushButton, QApplication
from PySide6.QtCore import QSize, QByteArray, Qt, QPropertyAnimation, Property
from PySide6.QtGui import QIcon, QPixmap, QPainter
from PySide6.QtSvg import QSvgRenderer

from ..theme import StyleSheet, theme_manager, ColorTokens
from ..animations import animation_manager


class SidebarButton(QPushButton):
    """
    侧边栏按钮 - VS Code Activity Bar Style + macOS 动画

    设计规格:
    - 尺寸: 50x50px
    - 默认: 透明背景, 图标颜色
    - Hover: 透明背景, 高亮图标, 图标放大
    - Press: 图标缩小
    - Active: 左侧 2px 边框
    """

    # 预渲染的最大缩放比例（用于高质量缩放）
    _RENDER_SCALE = 1.2

    def __init__(self, icon_text: str = None, svg_template: str = None, icon_size: tuple = (24, 24), parent=None):
        super().__init__(parent)
        self.setFixedSize(50, 50)
        self.setCheckable(True)
        self.setStyleSheet(StyleSheet.sidebar_btn())

        self._svg_template = svg_template
        self._icon_text = icon_text
        self._base_icon_size = icon_size  # 基础图标大小
        self._current_color = theme_manager.get_color(ColorTokens.ICON_DEFAULT)
        self._cached_pixmap = None  # 缓存渲染好的 pixmap

        # 动画相关 - 使用图标大小实现缩放效果
        self._icon_scale = 1.0
        self._scale_anim = None

        if svg_template:
            self._render_icon(self._current_color)  # 预渲染
            self._apply_icon_scale()
        elif icon_text:
            self.setText(icon_text)

        # 监听主题变化
        theme_manager.theme_changed.connect(self._on_theme_changed)

    # 图标缩放属性（用于动画）
    def _get_icon_scale(self) -> float:
        return self._icon_scale

    def _set_icon_scale(self, value: float):
        self._icon_scale = value
        self._apply_icon_scale()

    icon_scale = Property(float, _get_icon_scale, _set_icon_scale)

    def _apply_icon_scale(self):
        """应用图标缩放 - 只改变显示大小，不重新渲染"""
        if self._cached_pixmap:
            base_w, base_h = self._base_icon_size
            w = int(base_w * self._icon_scale)
            h = int(base_h * self._icon_scale)
            self.setIconSize(QSize(w, h))

    def _on_theme_changed(self, theme: str):
        """主题变化时更新样式"""
        self.setStyleSheet(StyleSheet.sidebar_btn())
        if self._svg_template:
            self._current_color = theme_manager.get_color(ColorTokens.ICON_DEFAULT)
            self._render_icon(self._current_color)
            self._apply_icon_scale()

    def _render_icon(self, color: str):
        """渲染 SVG 图标到缓存 - 只在颜色变化时调用"""
        if not self._svg_template:
            return

        self._current_color = color
        svg_data = self._svg_template.replace("{color}", color)
        renderer = QSvgRenderer(QByteArray(svg_data.encode()))

        # 以较大尺寸渲染，确保缩放时质量
        base_w, base_h = self._base_icon_size
        render_w = int(base_w * self._RENDER_SCALE)
        render_h = int(base_h * self._RENDER_SCALE)

        # 获取设备像素比以支持高DPI显示
        dpr = QApplication.primaryScreen().devicePixelRatio() if QApplication.primaryScreen() else 1.0

        # 创建高分辨率 pixmap
        pixmap = QPixmap(int(render_w * dpr), int(render_h * dpr))
        pixmap.setDevicePixelRatio(dpr)
        pixmap.fill(Qt.transparent)

        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.Antialiasing, True)
        painter.setRenderHint(QPainter.SmoothPixmapTransform, True)
        renderer.render(painter)
        painter.end()

        self._cached_pixmap = pixmap
        self.setIcon(QIcon(pixmap))

    def _start_scale_animation(self, target_scale: float, duration: int, use_spring: bool = False):
        """启动图标缩放动画"""
        if not animation_manager.enabled:
            return

        # 停止之前的动画
        if self._scale_anim and self._scale_anim.state() == QPropertyAnimation.Running:
            self._scale_anim.stop()

        self._scale_anim = QPropertyAnimation(self, b"icon_scale")
        self._scale_anim.setDuration(animation_manager._get_duration(duration))
        self._scale_anim.setStartValue(self._icon_scale)
        self._scale_anim.setEndValue(target_scale)

        if use_spring:
            self._scale_anim.setEasingCurve(animation_manager.create_spring_curve())
        else:
            self._scale_anim.setEasingCurve(animation_manager.create_ease_out_curve())

        self._scale_anim.start()

    def enterEvent(self, event):
        """鼠标进入时高亮图标 + 放大动画"""
        if self._svg_template:
            self._render_icon(theme_manager.get_color(ColorTokens.ICON_HOVER))
        self._start_scale_animation(animation_manager.SCALE_HOVER, animation_manager.DURATION_NORMAL, use_spring=True)
        super().enterEvent(event)

    def leaveEvent(self, event):
        """鼠标离开时恢复图标颜色 + 缩小动画"""
        if self._svg_template:
            self._render_icon(theme_manager.get_color(ColorTokens.ICON_DEFAULT))
        # 离开时不用 spring，避免过冲导致的"抽搐"效果
        self._start_scale_animation(1.0, animation_manager.DURATION_FAST)
        super().leaveEvent(event)

    def mousePressEvent(self, event):
        """鼠标按下 - 缩小动画"""
        if event.button() == Qt.LeftButton:
            self._start_scale_animation(animation_manager.SCALE_PRESS, animation_manager.DURATION_FAST)
        super().mousePressEvent(event)

    def mouseReleaseEvent(self, event):
        """鼠标释放 - spring 弹回动画"""
        if event.button() == Qt.LeftButton:
            # 如果鼠标还在按钮上，弹回到 hover 状态；否则直接回到正常状态
            if self.underMouse():
                self._start_scale_animation(animation_manager.SCALE_HOVER, animation_manager.DURATION_NORMAL, use_spring=True)
            else:
                self._start_scale_animation(1.0, animation_manager.DURATION_FAST)
        super().mouseReleaseEvent(event)
