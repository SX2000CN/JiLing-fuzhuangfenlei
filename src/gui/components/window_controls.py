"""
窗口控制组件 - 最小化、最大化、关闭按钮
"""

from PySide6.QtWidgets import QWidget, QHBoxLayout, QPushButton, QApplication
from PySide6.QtCore import Signal, QSize, Qt, QRect
from PySide6.QtGui import QIcon, QPixmap, QPainter, QColor, QPen

from ..theme import StyleSheet, theme_manager, ColorTokens


class WindowControlBar(QWidget):
    """
    窗口控制栏 - 最小化、最大化、关闭按钮
    """

    minimize_clicked = Signal()
    maximize_clicked = Signal()
    close_clicked = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedHeight(32)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addStretch()

        # 最小化按钮
        self.btn_min = QPushButton()
        self.btn_min.setFixedSize(46, 32)
        self.btn_min.setIcon(self._create_minimize_icon())
        self.btn_min.setIconSize(QSize(12, 12))
        self.btn_min.setStyleSheet(StyleSheet.control_btn())
        self.btn_min.clicked.connect(self.minimize_clicked.emit)
        layout.addWidget(self.btn_min)

        # 最大化按钮
        self.btn_max = QPushButton()
        self.btn_max.setFixedSize(46, 32)
        self.btn_max.setIcon(self._create_maximize_icon())
        self.btn_max.setIconSize(QSize(12, 12))
        self.btn_max.setStyleSheet(StyleSheet.control_btn())
        self.btn_max.clicked.connect(self.maximize_clicked.emit)
        layout.addWidget(self.btn_max)

        # 关闭按钮
        self.btn_close = QPushButton()
        self.btn_close.setFixedSize(46, 32)
        self.btn_close.setIcon(self._create_close_icon())
        self.btn_close.setIconSize(QSize(12, 12))
        self.btn_close.setStyleSheet(StyleSheet.close_btn())
        self.btn_close.clicked.connect(self.close_clicked.emit)
        layout.addWidget(self.btn_close)

        # 监听主题变化
        theme_manager.theme_changed.connect(self._on_theme_changed)

    def _on_theme_changed(self, theme: str):
        """主题变化时更新图标和样式"""
        self.btn_min.setIcon(self._create_minimize_icon())
        self.btn_min.setStyleSheet(StyleSheet.control_btn())
        # 更新最大化按钮图标（需要检查当前窗口状态）
        self.btn_max.setIcon(self._create_maximize_icon())
        self.btn_max.setStyleSheet(StyleSheet.control_btn())
        self.btn_close.setIcon(self._create_close_icon())
        self.btn_close.setStyleSheet(StyleSheet.close_btn())

    def update_maximize_button(self, is_maximized: bool):
        """根据窗口状态更新最大化按钮图标和关闭按钮样式"""
        if is_maximized:
            self.btn_max.setIcon(self._create_restore_icon())
            # 最大化时关闭按钮无圆角
            self.btn_close.setStyleSheet(StyleSheet.close_btn(corner_radius=0))
        else:
            self.btn_max.setIcon(self._create_maximize_icon())
            # 正常状态关闭按钮有圆角（内部圆角 = 窗口圆角 - 边框宽度）
            self.btn_close.setStyleSheet(StyleSheet.close_btn(corner_radius=9))

    def _get_icon_color(self) -> QColor:
        """获取图标颜色"""
        color_hex = theme_manager.get_color(ColorTokens.TEXT_PRIMARY)
        return QColor(color_hex)

    def _get_corner_color(self) -> QColor:
        """获取图标圆角颜色"""
        color_hex = theme_manager.get_color(ColorTokens.ICON_CORNER)
        return QColor(color_hex)

    def _create_minimize_icon(self) -> QIcon:
        """创建最小化图标 - 像素级绘制，10px宽1px高的横线"""
        size = 12
        pixmap = QPixmap(size, size)
        pixmap.fill(Qt.transparent)

        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.Antialiasing, False)
        color = self._get_icon_color()
        painter.fillRect(1, 6, 10, 1, color)  # x=1, y=6, width=10, height=1
        painter.end()

        return QIcon(pixmap)

    def _create_maximize_icon(self) -> QIcon:
        """创建最大化图标 - 像素级绘制，10x10方框，边框1px，四角圆角"""
        size = 12
        pixmap = QPixmap(size, size)
        pixmap.fill(Qt.transparent)

        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.Antialiasing, False)
        color = self._get_icon_color()
        corner_color = self._get_corner_color()  # 使用主题颜色

        # 绘制方框边框 (避开四个角)
        # 上边 (避开左上角和右上角)
        painter.fillRect(2, 1, 8, 1, color)
        # 下边 (避开左下角和右下角)
        painter.fillRect(2, 10, 8, 1, color)
        # 左边 (避开左上角和左下角)
        painter.fillRect(1, 2, 1, 8, color)
        # 右边 (避开右上角和右下角)
        painter.fillRect(10, 2, 1, 8, color)

        # 绘制四个角 (圆角效果)
        painter.fillRect(1, 1, 1, 1, corner_color)    # 左上角
        painter.fillRect(10, 1, 1, 1, corner_color)   # 右上角
        painter.fillRect(1, 10, 1, 1, corner_color)   # 左下角
        painter.fillRect(10, 10, 1, 1, corner_color)  # 右下角

        painter.end()

        return QIcon(pixmap)

    def _create_restore_icon(self) -> QIcon:
        """创建恢复图标 - 两个重叠矩形效果"""
        size = 12
        pixmap = QPixmap(size, size)
        pixmap.fill(Qt.transparent)

        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.Antialiasing, False)
        color = self._get_icon_color()
        corner_color = self._get_corner_color()  # 使用主题颜色

        # === 前面的8x8矩形 (从(1,4)到(8,11)) ===
        # 上边 (避开两角)
        painter.fillRect(2, 4, 6, 1, color)
        # 下边 (避开两角)
        painter.fillRect(2, 11, 6, 1, color)
        # 左边 (避开两角)
        painter.fillRect(1, 5, 1, 6, color)
        # 右边 (避开两角)
        painter.fillRect(8, 5, 1, 6, color)
        # 四个角 - 圆角
        painter.fillRect(1, 4, 1, 1, corner_color)   # 左上
        painter.fillRect(8, 4, 1, 1, corner_color)   # 右上
        painter.fillRect(1, 11, 1, 1, corner_color)  # 左下
        painter.fillRect(8, 11, 1, 1, corner_color)  # 右下

        # === 后面矩形的露出部分 (上2右2) ===
        # 横线: y=2, x从3到10
        painter.fillRect(4, 2, 6, 1, color)  # 避开左上角和右上角
        # 竖线: x=10, y从2到9
        painter.fillRect(10, 3, 1, 6, color)  # 避开右上角和右下角

        # 三个角 - 圆角 (左上、右上、右下)
        painter.fillRect(3, 2, 1, 1, corner_color)   # 左上
        painter.fillRect(10, 2, 1, 1, corner_color)  # 右上
        painter.fillRect(10, 9, 1, 1, corner_color)  # 右下

        # 两个结构的右上相交位置
        painter.fillRect(9, 3, 1, 1, corner_color)

        painter.end()

        return QIcon(pixmap)

    def _create_close_icon(self) -> QIcon:
        """创建关闭图标 - 像素级绘制X形状，1px宽"""
        size = 12
        pixmap = QPixmap(size, size)
        pixmap.fill(Qt.transparent)

        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.Antialiasing, False)
        color = self._get_icon_color()

        # 像素级绘制X形状 (两条对角线，每条1px宽)
        # 左上到右下对角线
        for i in range(10):
            painter.fillRect(i + 1, i + 1, 1, 1, color)

        # 右上到左下对角线
        for i in range(10):
            painter.fillRect(10 - i, i + 1, 1, 1, color)

        painter.end()

        return QIcon(pixmap)
