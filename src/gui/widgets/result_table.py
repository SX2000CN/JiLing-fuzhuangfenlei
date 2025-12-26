"""
分类结果表格组件
"""

from PySide6.QtWidgets import QTableWidget, QTableWidgetItem, QHeaderView
from PySide6.QtCore import Qt

from ..theme import StyleSheet, theme_manager
from ..utils import FontManager


class ClassificationResultTable(QTableWidget):
    """分类结果表格 - VS Code 风格"""

    def __init__(self, parent=None):
        super().__init__(parent)

        # 设置列
        self.setColumnCount(4)
        self.setHorizontalHeaderLabels(["文件名", "分类结果", "置信度", "状态"])

        # 表头设置
        header = self.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.Stretch)
        header.setSectionResizeMode(1, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(2, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(3, QHeaderView.ResizeToContents)

        # 样式
        self.setStyleSheet(StyleSheet.result_table())
        self.setFont(FontManager.input_font())

        # 行为
        self.setAlternatingRowColors(False)
        self.setSelectionBehavior(QTableWidget.SelectRows)
        self.setEditTriggers(QTableWidget.NoEditTriggers)
        self.verticalHeader().setVisible(False)

        # 监听主题变化
        theme_manager.theme_changed.connect(self._on_theme_changed)

    def _on_theme_changed(self, theme: str):
        self.setStyleSheet(StyleSheet.result_table())

    def add_result(self, filename: str, predicted_class: str, confidence: float, status: str = "完成"):
        """添加一条分类结果"""
        row = self.rowCount()
        self.insertRow(row)

        self.setItem(row, 0, QTableWidgetItem(filename))
        self.setItem(row, 1, QTableWidgetItem(predicted_class))
        self.setItem(row, 2, QTableWidgetItem(f"{confidence:.2%}"))
        self.setItem(row, 3, QTableWidgetItem(status))

        # 滚动到新行
        self.scrollToBottom()

    def clear_results(self):
        """清空所有结果"""
        self.setRowCount(0)
