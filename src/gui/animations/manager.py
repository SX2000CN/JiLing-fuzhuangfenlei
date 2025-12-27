"""
动画管理器 - 统一管理所有 UI 动画

macOS 风格：流畅弹性，300ms 过渡，spring 缓动
"""

from typing import Optional, Any
from PySide6.QtCore import (
    QObject, QPropertyAnimation, QEasingCurve, QParallelAnimationGroup,
    QSequentialAnimationGroup, Property, Signal
)
from PySide6.QtWidgets import QWidget, QGraphicsOpacityEffect


class AnimationManager(QObject):
    """动画管理器单例 - 统一管理所有动画"""

    _instance: Optional['AnimationManager'] = None

    # 预设时长 (macOS 风格)
    DURATION_FAST = 150      # 快速反馈（按下、淡出）
    DURATION_NORMAL = 300    # 标准过渡（悬停、页面切换）
    DURATION_SLOW = 450      # 复杂动画（展开面板）

    # 缩放参数 (减小幅度让动画更细腻)
    SCALE_HOVER = 1.03       # 悬停缩放
    SCALE_PRESS = 0.97       # 按下缩放

    # Spring 过冲系数
    SPRING_OVERSHOOT = 0.3   # OutBack 过冲量

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        super().__init__()
        self._initialized = True
        self._enabled = True
        self._duration_multiplier = 1.0
        self._active_animations: list = []

    @property
    def enabled(self) -> bool:
        return self._enabled

    @enabled.setter
    def enabled(self, value: bool):
        self._enabled = value

    @property
    def duration_multiplier(self) -> float:
        return self._duration_multiplier

    @duration_multiplier.setter
    def duration_multiplier(self, value: float):
        self._duration_multiplier = max(0.0, value)

    def _get_duration(self, base_duration: int) -> int:
        """获取实际动画时长"""
        if not self._enabled:
            return 0
        return int(base_duration * self._duration_multiplier)

    def create_spring_curve(self) -> QEasingCurve:
        """创建 macOS 风格的 spring 缓动曲线"""
        curve = QEasingCurve(QEasingCurve.OutBack)
        curve.setOvershoot(self.SPRING_OVERSHOOT)
        return curve

    def create_ease_out_curve(self) -> QEasingCurve:
        """创建 easeOut 缓动曲线"""
        return QEasingCurve(QEasingCurve.OutCubic)

    def create_ease_in_curve(self) -> QEasingCurve:
        """创建 easeIn 缓动曲线"""
        return QEasingCurve(QEasingCurve.InCubic)

    def create_animation(
        self,
        target: QObject,
        property_name: bytes,
        start_value: Any,
        end_value: Any,
        duration: int = DURATION_NORMAL,
        easing: Optional[QEasingCurve] = None
    ) -> QPropertyAnimation:
        """创建属性动画"""
        anim = QPropertyAnimation(target, property_name)
        anim.setDuration(self._get_duration(duration))
        anim.setStartValue(start_value)
        anim.setEndValue(end_value)
        anim.setEasingCurve(easing or self.create_ease_out_curve())
        return anim

    def fade_in(self, widget: QWidget, duration: int = DURATION_NORMAL) -> QPropertyAnimation:
        """淡入动画"""
        effect = widget.graphicsEffect()
        if not isinstance(effect, QGraphicsOpacityEffect):
            effect = QGraphicsOpacityEffect(widget)
            widget.setGraphicsEffect(effect)

        anim = self.create_animation(
            effect, b"opacity", 0.0, 1.0, duration, self.create_ease_out_curve()
        )
        return anim

    def fade_out(self, widget: QWidget, duration: int = DURATION_FAST) -> QPropertyAnimation:
        """淡出动画"""
        effect = widget.graphicsEffect()
        if not isinstance(effect, QGraphicsOpacityEffect):
            effect = QGraphicsOpacityEffect(widget)
            widget.setGraphicsEffect(effect)

        anim = self.create_animation(
            effect, b"opacity", 1.0, 0.0, duration, self.create_ease_in_curve()
        )
        return anim

    def slide_in(
        self,
        widget: QWidget,
        direction: str = "left",
        duration: int = DURATION_NORMAL
    ) -> QPropertyAnimation:
        """滑入动画"""
        offset = widget.width() if direction in ("left", "right") else widget.height()
        start_x = -offset if direction == "left" else (offset if direction == "right" else 0)
        start_y = -offset if direction == "up" else (offset if direction == "down" else 0)

        if direction in ("left", "right"):
            anim = self.create_animation(
                widget, b"x", widget.x() + start_x, widget.x(),
                duration, self.create_spring_curve()
            )
        else:
            anim = self.create_animation(
                widget, b"y", widget.y() + start_y, widget.y(),
                duration, self.create_spring_curve()
            )
        return anim

    def expand_width(
        self,
        widget: QWidget,
        target_width: int,
        duration: int = DURATION_NORMAL
    ) -> QParallelAnimationGroup:
        """展开宽度动画"""
        group = QParallelAnimationGroup()

        min_anim = self.create_animation(
            widget, b"minimumWidth", widget.minimumWidth(), target_width,
            duration, self.create_spring_curve()
        )
        max_anim = self.create_animation(
            widget, b"maximumWidth", widget.maximumWidth(), target_width,
            duration, self.create_spring_curve()
        )

        group.addAnimation(min_anim)
        group.addAnimation(max_anim)
        return group

    def collapse_width(
        self,
        widget: QWidget,
        target_width: int = 0,
        duration: int = DURATION_FAST
    ) -> QParallelAnimationGroup:
        """收起宽度动画"""
        group = QParallelAnimationGroup()

        min_anim = self.create_animation(
            widget, b"minimumWidth", widget.minimumWidth(), target_width,
            duration, self.create_ease_out_curve()
        )
        max_anim = self.create_animation(
            widget, b"maximumWidth", widget.maximumWidth(), target_width,
            duration, self.create_ease_out_curve()
        )

        group.addAnimation(min_anim)
        group.addAnimation(max_anim)
        return group


# 全局单例
animation_manager = AnimationManager()
