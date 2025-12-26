"""
GPU 状态监控组件
"""

from PySide6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QLabel, QProgressBar
from PySide6.QtCore import QTimer

from ..theme import StyleSheet, theme_manager, ColorTokens
from ..utils import FontManager

# 可选依赖
try:
    import GPUtil
    GPU_UTIL_AVAILABLE = True
except ImportError:
    GPU_UTIL_AVAILABLE = False


class GPUStatusWidget(QWidget):
    """GPU 状态监控组件"""

    def __init__(self, parent=None):
        super().__init__(parent)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        # 标题
        title = QLabel("GPU 状态")
        title.setStyleSheet(StyleSheet.page_title())
        title.setFont(FontManager.title_font())
        layout.addWidget(title)

        # GPU 使用率
        usage_row = QHBoxLayout()
        usage_label = QLabel("使用率:")
        usage_label.setStyleSheet(StyleSheet.param_label())
        usage_label.setFont(FontManager.label_font())
        usage_row.addWidget(usage_label)

        self.usage_bar = QProgressBar()
        self.usage_bar.setStyleSheet(StyleSheet.progress_bar())
        self.usage_bar.setTextVisible(True)
        self.usage_bar.setFormat("%p%")
        usage_row.addWidget(self.usage_bar, 1)
        layout.addLayout(usage_row)

        # 显存使用
        memory_row = QHBoxLayout()
        memory_label = QLabel("显存:")
        memory_label.setStyleSheet(StyleSheet.param_label())
        memory_label.setFont(FontManager.label_font())
        memory_row.addWidget(memory_label)

        self.memory_bar = QProgressBar()
        self.memory_bar.setStyleSheet(StyleSheet.progress_bar())
        self.memory_bar.setTextVisible(True)
        self.memory_bar.setFormat("%p%")
        memory_row.addWidget(self.memory_bar, 1)
        layout.addLayout(memory_row)

        # 状态标签
        self.status_label = QLabel("检测中...")
        self.status_label.setStyleSheet(StyleSheet.settings_row_desc())
        self.status_label.setFont(FontManager.small_font())
        layout.addWidget(self.status_label)

        # 定时更新
        self.timer = QTimer()
        self.timer.timeout.connect(self._update_status)
        self.timer.start(2000)  # 每2秒更新

        self._update_status()

        # 监听主题变化
        theme_manager.theme_changed.connect(self._on_theme_changed)

    def _on_theme_changed(self, theme: str):
        self.usage_bar.setStyleSheet(StyleSheet.progress_bar())
        self.memory_bar.setStyleSheet(StyleSheet.progress_bar())
        self.status_label.setStyleSheet(StyleSheet.settings_row_desc())

    def _update_status(self):
        """更新 GPU 状态"""
        if not GPU_UTIL_AVAILABLE:
            self.status_label.setText("GPUtil 未安装")
            self.usage_bar.setValue(0)
            self.memory_bar.setValue(0)
            return

        try:
            gpus = GPUtil.getGPUs()
            if gpus:
                gpu = gpus[0]
                self.usage_bar.setValue(int(gpu.load * 100))
                self.memory_bar.setValue(int(gpu.memoryUtil * 100))
                self.status_label.setText(f"{gpu.name} - {gpu.memoryUsed:.0f}MB / {gpu.memoryTotal:.0f}MB")
            else:
                self.status_label.setText("未检测到 GPU")
                self.usage_bar.setValue(0)
                self.memory_bar.setValue(0)
        except Exception as e:
            self.status_label.setText(f"读取失败: {str(e)}")
