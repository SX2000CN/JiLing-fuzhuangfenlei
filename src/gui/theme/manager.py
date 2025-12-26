"""
主题管理器 - 单例模式，管理应用主题切换
"""

import sys
from typing import Dict, Optional
from PySide6.QtCore import QObject, Signal
from PySide6.QtWidgets import QApplication
from PySide6.QtGui import QPalette, QColor

from .tokens import ColorTokens
from .dark import DARK_THEME
from .light import LIGHT_THEME


class ThemeManager(QObject):
    """
    主题管理器单例

    支持:
    - 深色模式 (dark)
    - 浅色模式 (light)
    - 跟随系统 (system)
    """

    _instance: Optional['ThemeManager'] = None

    # 主题变化信号
    theme_changed = Signal(str)

    # 主题常量
    THEME_DARK = "dark"
    THEME_LIGHT = "light"
    THEME_SYSTEM = "system"

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

        self._themes: Dict[str, Dict] = {
            self.THEME_DARK: DARK_THEME,
            self.THEME_LIGHT: LIGHT_THEME,
        }

        self._current_theme = self.THEME_DARK
        self._theme_mode = self.THEME_DARK  # 用户选择的模式 (dark/light/system)

        # 缓存计算后的颜色
        self._color_cache: Dict[str, str] = {}

    @property
    def current_theme(self) -> str:
        """获取当前实际使用的主题 (dark 或 light)"""
        return self._current_theme

    @property
    def theme_mode(self) -> str:
        """获取用户选择的主题模式 (dark/light/system)"""
        return self._theme_mode

    def get_color(self, token: str) -> str:
        """
        获取颜色值

        Args:
            token: 颜色 token (如 ColorTokens.BG_PRIMARY)

        Returns:
            颜色值字符串 (如 "#1E1E1E")
        """
        theme = self._themes.get(self._current_theme, DARK_THEME)
        return theme.get(token, "#FF00FF")  # 返回品红色表示缺失的颜色

    def get_colors(self) -> Dict[str, str]:
        """获取当前主题的所有颜色"""
        return self._themes.get(self._current_theme, DARK_THEME).copy()

    def set_theme(self, theme_mode: str) -> None:
        """
        设置主题模式

        Args:
            theme_mode: "dark", "light", 或 "system"
        """
        self._theme_mode = theme_mode

        if theme_mode == self.THEME_SYSTEM:
            # 检测系统主题
            actual_theme = self._detect_system_theme()
        else:
            actual_theme = theme_mode

        if actual_theme != self._current_theme:
            self._current_theme = actual_theme
            self._color_cache.clear()
            self.theme_changed.emit(actual_theme)

    def _detect_system_theme(self) -> str:
        """检测系统主题"""
        try:
            app = QApplication.instance()
            if app:
                palette = app.palette()
                # 如果窗口背景色较暗，则为深色模式
                bg_color = palette.color(QPalette.ColorRole.Window)
                # 使用亮度判断：L = 0.299*R + 0.587*G + 0.114*B
                luminance = 0.299 * bg_color.red() + 0.587 * bg_color.green() + 0.114 * bg_color.blue()
                return self.THEME_DARK if luminance < 128 else self.THEME_LIGHT
        except Exception:
            pass

        # 默认深色
        return self.THEME_DARK

    def is_dark_theme(self) -> bool:
        """当前是否为深色主题"""
        return self._current_theme == self.THEME_DARK

    def toggle_theme(self) -> None:
        """切换主题（在 dark 和 light 之间）"""
        new_theme = self.THEME_LIGHT if self._current_theme == self.THEME_DARK else self.THEME_DARK
        self.set_theme(new_theme)


# 全局单例实例
theme_manager = ThemeManager()
