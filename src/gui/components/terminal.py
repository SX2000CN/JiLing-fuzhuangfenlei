"""
终端输出组件 - 显示日志和训练进度
"""

from datetime import datetime
from PySide6.QtWidgets import QTextEdit
from PySide6.QtGui import QTextCursor

from ..theme import StyleSheet, theme_manager, ColorTokens


class TerminalOutput(QTextEdit):
    """
    终端输出组件 - VS Code 终端风格
    支持彩色日志输出
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setReadOnly(True)
        self.setStyleSheet(StyleSheet.terminal_output())

        # 监听主题变化
        theme_manager.theme_changed.connect(self._on_theme_changed)

    def _on_theme_changed(self, theme: str):
        self.setStyleSheet(StyleSheet.terminal_output())

    def log(self, message: str, level: str = "INFO"):
        """
        添加日志消息

        Args:
            message: 日志内容
            level: 日志级别 (DEBUG, INFO, WARNING, ERROR, SUCCESS, METRIC, HIGHLIGHT)
        """
        colors = StyleSheet.COLORS
        color = colors.get(level, colors['INFO'])

        timestamp = datetime.now().strftime("%H:%M:%S")

        # 构造 HTML 格式的日志
        html = f'<span style="color: {color};">[{timestamp}] [{level}] {message}</span><br>'

        # 追加到文本
        cursor = self.textCursor()
        cursor.movePosition(QTextCursor.End)
        cursor.insertHtml(html)

        # 滚动到底部
        self.verticalScrollBar().setValue(self.verticalScrollBar().maximum())

    def log_debug(self, message: str):
        self.log(message, "DEBUG")

    def log_info(self, message: str):
        self.log(message, "INFO")

    def log_warning(self, message: str):
        self.log(message, "WARNING")

    def log_error(self, message: str):
        self.log(message, "ERROR")

    def log_success(self, message: str):
        self.log(message, "SUCCESS")

    def log_metric(self, message: str):
        self.log(message, "METRIC")

    def log_highlight(self, message: str):
        self.log(message, "HIGHLIGHT")

    def clear_log(self):
        """清空日志"""
        self.clear()
