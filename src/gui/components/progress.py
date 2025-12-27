"""
带动画的进度条组件
"""

from PySide6.QtWidgets import QProgressBar
from PySide6.QtCore import Property, QPropertyAnimation

from ..animations import animation_manager


class AnimatedProgressBar(QProgressBar):
    """
    带平滑动画的进度条

    数值变化时自动平滑过渡
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self._animated_value = 0
        self._value_anim = None

    def _get_animated_value(self) -> int:
        return self._animated_value

    def _set_animated_value(self, value: int):
        self._animated_value = value
        super().setValue(value)

    animated_value = Property(int, _get_animated_value, _set_animated_value)

    def setValue(self, value: int):
        """设置值（带动画）"""
        if not animation_manager.enabled:
            super().setValue(value)
            self._animated_value = value
            return

        # 停止之前的动画
        if self._value_anim and self._value_anim.state() == QPropertyAnimation.Running:
            self._value_anim.stop()

        self._value_anim = QPropertyAnimation(self, b"animated_value")
        self._value_anim.setDuration(animation_manager._get_duration(animation_manager.DURATION_NORMAL))
        self._value_anim.setStartValue(self._animated_value)
        self._value_anim.setEndValue(value)
        self._value_anim.setEasingCurve(animation_manager.create_ease_out_curve())
        self._value_anim.start()

    def setValueImmediate(self, value: int):
        """立即设置值（无动画）"""
        if self._value_anim and self._value_anim.state() == QPropertyAnimation.Running:
            self._value_anim.stop()
        self._animated_value = value
        super().setValue(value)
