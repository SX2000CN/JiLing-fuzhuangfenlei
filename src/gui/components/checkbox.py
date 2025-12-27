"""
自定义复选框组件 - VS Code 风格 + macOS 动画
"""

from PySide6.QtWidgets import QWidget
from PySide6.QtCore import Qt, Signal, Property, QPropertyAnimation, QTimer
from PySide6.QtGui import QPainter, QColor, QPen, QPainterPath

from ..theme import theme_manager, ColorTokens
from ..animations import animation_manager


class VSCheckBox(QWidget):
    """
    自定义复选框 - VS Code 风格 + macOS 动画

    动画效果:
    - 勾选: 背景色渐变 + 勾选标记绘制动画
    - 取消: 勾选标记淡出 + 背景色渐变
    """

    toggled = Signal(bool)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._checked = False
        self._hovered = False
        self._check_progress = 0.0  # 勾选动画进度 0-1
        self._check_anim = None
        self.setFixedSize(18, 18)
        self.setCursor(Qt.PointingHandCursor)

        # 监听主题变化
        theme_manager.theme_changed.connect(self._on_theme_changed)

    # 勾选进度属性（用于动画）
    def _get_check_progress(self) -> float:
        return self._check_progress

    def _set_check_progress(self, value: float):
        self._check_progress = value
        self.update()

    check_progress = Property(float, _get_check_progress, _set_check_progress)

    def _on_theme_changed(self, theme: str):
        """主题变化时更新"""
        self.update()

    def isChecked(self) -> bool:
        return self._checked

    def setChecked(self, checked: bool, animate: bool = True):
        if self._checked != checked:
            self._checked = checked
            if animate and animation_manager.enabled:
                self._start_check_animation(checked)
            else:
                self._check_progress = 1.0 if checked else 0.0
                self.update()
            self.toggled.emit(checked)

    def toggle(self):
        self.setChecked(not self._checked)

    def _start_check_animation(self, checking: bool):
        """启动勾选动画"""
        if self._check_anim and self._check_anim.state() == QPropertyAnimation.Running:
            self._check_anim.stop()

        self._check_anim = QPropertyAnimation(self, b"check_progress")

        if checking:
            self._check_anim.setDuration(animation_manager._get_duration(200))
            self._check_anim.setStartValue(0.0)
            self._check_anim.setEndValue(1.0)
            self._check_anim.setEasingCurve(animation_manager.create_ease_out_curve())
        else:
            self._check_anim.setDuration(animation_manager._get_duration(150))
            self._check_anim.setStartValue(1.0)
            self._check_anim.setEndValue(0.0)
            self._check_anim.setEasingCurve(animation_manager.create_ease_in_curve())

        self._check_anim.start()

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

        # 计算动画插值的背景色
        accent = QColor(theme_manager.get_color(ColorTokens.ACCENT))
        accent_hover = QColor(theme_manager.get_color(ColorTokens.ACCENT_HOVER))
        bg_input = QColor(theme_manager.get_color(ColorTokens.BG_INPUT))
        border = QColor(theme_manager.get_color(ColorTokens.BORDER))

        # 根据动画进度插值背景色
        progress = self._check_progress

        if progress > 0:
            # 选中状态的颜色
            target_bg = accent_hover if self._hovered else accent
            # 插值背景色
            bg_color = self._interpolate_color(bg_input, target_bg, progress)
            border_color = bg_color
        else:
            bg_color = bg_input
            border_color = accent if self._hovered else border

        # 绘制背景
        painter.setPen(QPen(border_color, 1))
        painter.setBrush(bg_color)
        painter.drawRoundedRect(self.rect().adjusted(0, 0, -1, -1), 3, 3)

        # 绘制勾选标记（带动画）
        if progress > 0:
            painter.setPen(QPen(QColor("#FFFFFF"), 2, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))

            # 勾选标记的三个点
            p1 = (4, 9)
            p2 = (7, 12)
            p3 = (14, 5)

            # 计算路径总长度
            len1 = ((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)**0.5
            len2 = ((p3[0] - p2[0])**2 + (p3[1] - p2[1])**2)**0.5
            total_len = len1 + len2

            # 根据进度绘制路径
            drawn_len = progress * total_len

            path = QPainterPath()
            path.moveTo(*p1)

            if drawn_len <= len1:
                # 只绘制第一段
                t = drawn_len / len1
                x = p1[0] + t * (p2[0] - p1[0])
                y = p1[1] + t * (p2[1] - p1[1])
                path.lineTo(x, y)
            else:
                # 绘制完整第一段 + 部分第二段
                path.lineTo(*p2)
                t = (drawn_len - len1) / len2
                x = p2[0] + t * (p3[0] - p2[0])
                y = p2[1] + t * (p3[1] - p2[1])
                path.lineTo(x, y)

            painter.drawPath(path)

        painter.end()

    def _interpolate_color(self, c1: QColor, c2: QColor, t: float) -> QColor:
        """颜色插值"""
        return QColor(
            int(c1.red() + t * (c2.red() - c1.red())),
            int(c1.green() + t * (c2.green() - c1.green())),
            int(c1.blue() + t * (c2.blue() - c1.blue())),
            int(c1.alpha() + t * (c2.alpha() - c1.alpha()))
        )
