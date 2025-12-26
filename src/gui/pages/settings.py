"""
设置页面 - 应用程序配置
"""

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QFrame,
    QScrollArea, QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox
)
from PySide6.QtCore import Qt, Signal

from ..theme import StyleSheet, theme_manager, ColorTokens
from ..utils import FontManager
from ..components import SettingsRow, SettingsCheckRow, WindowControlBar


class SettingsPage(QWidget):
    """设置页面 - 应用程序配置"""

    # 信号
    theme_changed = Signal(str)  # 主题变化
    settings_changed = Signal(dict)  # 设置变化
    # 窗口控制信号
    minimize_requested = Signal()
    maximize_requested = Signal()
    close_requested = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._init_ui()

        # 监听主题变化
        theme_manager.theme_changed.connect(self._on_theme_changed)

    def _on_theme_changed(self, theme: str):
        """主题变化时更新样式"""
        self.setStyleSheet(StyleSheet.settings_area())
        for divider in self._dividers:
            divider.setStyleSheet(StyleSheet.divider_h())
        self._scroll_area.setStyleSheet(StyleSheet.settings_scroll())
        # 更新标题样式
        self._title_label.setStyleSheet(StyleSheet.terminal_title())
        # 更新所有输入框样式
        for line_edit in [self.model_dir_input, self.dataset_dir_input,
                          self.log_dir_input, self.export_dir_input]:
            line_edit.setStyleSheet(StyleSheet.input_box())
        for spinbox in [self.confidence_spin, self.workers_spin, self.max_batch_spin,
                        self.default_epochs_spin, self.patience_spin,
                        self.checkpoint_freq_spin, self.log_days_spin]:
            spinbox.setStyleSheet(StyleSheet.settings_spinbox())
        # 更新下拉框样式
        for combo in [self.theme_combo, self.scale_combo, self.device_combo,
                      self.precision_combo, self.log_level_combo]:
            combo.setStyleSheet(StyleSheet.input_box())
        # 更新分组标题样式
        for section_title in self._section_titles:
            section_title.setStyleSheet(f"""
                QLabel {{
                    color: {theme_manager.get_color(ColorTokens.TEXT_SECONDARY)};
                    font-size: 11px;
                    font-weight: 600;
                    padding: 20px 12px 8px 12px;
                    text-transform: uppercase;
                    letter-spacing: 1px;
                }}
            """)

    def _init_ui(self):
        """初始化 UI"""
        self._dividers = []
        self._section_titles = []

        self.setStyleSheet(StyleSheet.settings_area())
        self.setObjectName("settingsArea")

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # 顶部标题栏
        top_bar = self._create_top_bar()
        layout.addWidget(top_bar)

        # 分割线
        divider = QFrame()
        divider.setStyleSheet(StyleSheet.divider_h())
        layout.addWidget(divider)
        self._dividers.append(divider)

        # 可滚动的设置内容区域
        self._scroll_area = QScrollArea()
        self._scroll_area.setWidgetResizable(True)
        self._scroll_area.setStyleSheet(StyleSheet.settings_scroll())
        self._scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        scroll_content = QWidget()
        scroll_layout = QVBoxLayout(scroll_content)
        # 左右边距推动 SettingsRow 向内，使高亮背景不贴边
        scroll_layout.setContentsMargins(12, 0, 12, 24)
        scroll_layout.setSpacing(0)

        # 添加设置项
        self._create_appearance_settings(scroll_layout)
        self._create_model_settings(scroll_layout)
        self._create_path_settings(scroll_layout)
        self._create_performance_settings(scroll_layout)
        self._create_training_defaults(scroll_layout)
        self._create_log_settings(scroll_layout)
        self._create_other_settings(scroll_layout)

        scroll_layout.addStretch()

        self._scroll_area.setWidget(scroll_content)
        layout.addWidget(self._scroll_area)

    def _create_top_bar(self) -> QWidget:
        """创建顶部标题栏"""
        top_bar = QWidget()
        top_bar.setFixedHeight(72)

        top_outer_layout = QVBoxLayout(top_bar)
        top_outer_layout.setContentsMargins(0, 0, 0, 0)
        top_outer_layout.setSpacing(0)

        # 窗口控制按钮行（32px）- 贴顶靠右
        self._window_controls = WindowControlBar()
        self._window_controls.minimize_clicked.connect(self.minimize_requested.emit)
        self._window_controls.maximize_clicked.connect(self.maximize_requested.emit)
        self._window_controls.close_clicked.connect(self.close_requested.emit)
        top_outer_layout.addWidget(self._window_controls)

        # 标题行
        title_row = QWidget()
        title_row_layout = QHBoxLayout(title_row)
        title_row_layout.setContentsMargins(24, 0, 24, 0)

        self._title_label = QLabel("设置")
        self._title_label.setStyleSheet(StyleSheet.terminal_title())
        self._title_label.setFont(FontManager.header_font())
        title_row_layout.addWidget(self._title_label)
        title_row_layout.addStretch()

        top_outer_layout.addWidget(title_row)

        return top_bar

    def _create_section_title(self, text: str) -> QLabel:
        """创建分组标题"""
        label = QLabel(text)
        label.setStyleSheet(f"""
            QLabel {{
                color: {theme_manager.get_color(ColorTokens.TEXT_SECONDARY)};
                font-size: 11px;
                font-weight: 600;
                padding: 20px 12px 8px 12px;
                text-transform: uppercase;
                letter-spacing: 1px;
            }}
        """)
        label.setFont(FontManager.small_font())
        self._section_titles.append(label)
        return label

    def _create_appearance_settings(self, layout: QVBoxLayout):
        """创建外观设置"""
        layout.addWidget(self._create_section_title("外观设置"))

        # 主题
        row_theme = SettingsRow("主题", "选择应用程序的外观主题")
        self.theme_combo = QComboBox()
        self.theme_combo.addItems(["深色模式", "浅色模式", "跟随系统"])
        self.theme_combo.setStyleSheet(StyleSheet.input_box())
        self.theme_combo.setFont(FontManager.input_font())
        self.theme_combo.setFixedWidth(180)
        self.theme_combo.currentTextChanged.connect(self._on_theme_combo_changed)
        row_theme.set_control(self.theme_combo)
        layout.addWidget(row_theme)

        # 界面缩放
        row_scale = SettingsRow("界面缩放", "调整界面元素的大小")
        self.scale_combo = QComboBox()
        self.scale_combo.addItems(["100%", "125%", "150%", "175%", "200%"])
        self.scale_combo.setStyleSheet(StyleSheet.input_box())
        self.scale_combo.setFont(FontManager.input_font())
        self.scale_combo.setFixedWidth(180)
        row_scale.set_control(self.scale_combo)
        layout.addWidget(row_scale)

    def _create_model_settings(self, layout: QVBoxLayout):
        """创建模型设置"""
        layout.addWidget(self._create_section_title("模型设置"))

        # 推理设备
        row_device = SettingsRow("推理设备", "选择用于模型推理的计算设备")
        self.device_combo = QComboBox()
        self.device_combo.addItems(["自动检测", "CPU", "GPU (CUDA)"])
        self.device_combo.setStyleSheet(StyleSheet.input_box())
        self.device_combo.setFont(FontManager.input_font())
        self.device_combo.setFixedWidth(180)
        row_device.set_control(self.device_combo)
        layout.addWidget(row_device)

        # 推理精度
        row_precision = SettingsRow("推理精度", "FP16 可加速推理但可能降低精度")
        self.precision_combo = QComboBox()
        self.precision_combo.addItems(["FP32 (默认)", "FP16 (加速)"])
        self.precision_combo.setStyleSheet(StyleSheet.input_box())
        self.precision_combo.setFont(FontManager.input_font())
        self.precision_combo.setFixedWidth(180)
        row_precision.set_control(self.precision_combo)
        layout.addWidget(row_precision)

        # 置信度阈值
        row_conf = SettingsRow("置信度阈值", "低于此阈值的分类结果将被标记为不确定")
        self.confidence_spin = QDoubleSpinBox()
        self.confidence_spin.setRange(0.1, 0.99)
        self.confidence_spin.setValue(0.5)
        self.confidence_spin.setSingleStep(0.05)
        self.confidence_spin.setDecimals(2)
        self.confidence_spin.setStyleSheet(StyleSheet.settings_spinbox())
        self.confidence_spin.setFont(FontManager.input_font())
        row_conf.set_control(self.confidence_spin)
        layout.addWidget(row_conf)

    def _create_path_settings(self, layout: QVBoxLayout):
        """创建路径配置"""
        layout.addWidget(self._create_section_title("路径配置"))

        # 默认模型目录
        row_model_dir = SettingsRow("默认模型目录", "模型文件的存储位置")
        self.model_dir_input = QLineEdit("models/saved_models")
        self.model_dir_input.setStyleSheet(StyleSheet.input_box())
        self.model_dir_input.setFont(FontManager.input_font())
        self.model_dir_input.setFixedWidth(280)
        row_model_dir.set_control(self.model_dir_input)
        layout.addWidget(row_model_dir)

        # 数据集目录
        row_data_dir = SettingsRow("数据集目录", "训练和验证数据的存储位置")
        self.dataset_dir_input = QLineEdit("data")
        self.dataset_dir_input.setStyleSheet(StyleSheet.input_box())
        self.dataset_dir_input.setFont(FontManager.input_font())
        self.dataset_dir_input.setFixedWidth(280)
        row_data_dir.set_control(self.dataset_dir_input)
        layout.addWidget(row_data_dir)

        # 日志目录
        row_log_dir = SettingsRow("日志目录", "运行日志的存储位置")
        self.log_dir_input = QLineEdit("logs")
        self.log_dir_input.setStyleSheet(StyleSheet.input_box())
        self.log_dir_input.setFont(FontManager.input_font())
        self.log_dir_input.setFixedWidth(280)
        row_log_dir.set_control(self.log_dir_input)
        layout.addWidget(row_log_dir)

        # 导出目录
        row_export_dir = SettingsRow("导出目录", "分类结果的导出位置")
        self.export_dir_input = QLineEdit("outputs")
        self.export_dir_input.setStyleSheet(StyleSheet.input_box())
        self.export_dir_input.setFont(FontManager.input_font())
        self.export_dir_input.setFixedWidth(280)
        row_export_dir.set_control(self.export_dir_input)
        layout.addWidget(row_export_dir)

    def _create_performance_settings(self, layout: QVBoxLayout):
        """创建性能设置"""
        layout.addWidget(self._create_section_title("性能设置"))

        # 数据加载线程
        row_workers = SettingsRow("数据加载线程", "用于加载数据的并行线程数")
        self.workers_spin = QSpinBox()
        self.workers_spin.setRange(1, 16)
        self.workers_spin.setValue(4)
        self.workers_spin.setStyleSheet(StyleSheet.settings_spinbox())
        self.workers_spin.setFont(FontManager.input_font())
        row_workers.set_control(self.workers_spin)
        layout.addWidget(row_workers)

        # 批处理上限
        row_batch = SettingsRow("批处理上限", "单次处理的最大图片数量")
        self.max_batch_spin = QSpinBox()
        self.max_batch_spin.setRange(1, 256)
        self.max_batch_spin.setValue(64)
        self.max_batch_spin.setStyleSheet(StyleSheet.settings_spinbox())
        self.max_batch_spin.setFont(FontManager.input_font())
        row_batch.set_control(self.max_batch_spin)
        layout.addWidget(row_batch)

        # 混合精度训练
        self.amp_check = SettingsCheckRow("启用混合精度训练 (AMP)", "使用 FP16 加速训练，同时保持模型精度")
        self.amp_check.setChecked(True)
        layout.addWidget(self.amp_check)

        # 内存锁定
        self.pin_memory_check = SettingsCheckRow("启用内存锁定 (Pin Memory)", "锁定内存可加速 GPU 数据传输")
        self.pin_memory_check.setChecked(True)
        layout.addWidget(self.pin_memory_check)

    def _create_training_defaults(self, layout: QVBoxLayout):
        """创建训练默认值设置"""
        layout.addWidget(self._create_section_title("训练默认值"))

        # 默认训练轮数
        row_epochs = SettingsRow("默认训练轮数", "新建训练任务的默认轮数")
        self.default_epochs_spin = QSpinBox()
        self.default_epochs_spin.setRange(1, 1000)
        self.default_epochs_spin.setValue(50)
        self.default_epochs_spin.setStyleSheet(StyleSheet.settings_spinbox())
        self.default_epochs_spin.setFont(FontManager.input_font())
        row_epochs.set_control(self.default_epochs_spin)
        layout.addWidget(row_epochs)

        # 早停耐心值
        row_patience = SettingsRow("早停耐心值", "验证损失不再下降后等待的轮数")
        self.patience_spin = QSpinBox()
        self.patience_spin.setRange(1, 100)
        self.patience_spin.setValue(10)
        self.patience_spin.setStyleSheet(StyleSheet.settings_spinbox())
        self.patience_spin.setFont(FontManager.input_font())
        row_patience.set_control(self.patience_spin)
        layout.addWidget(row_patience)

        # 检查点保存间隔
        row_ckpt = SettingsRow("检查点保存间隔", "每隔多少轮保存一次模型检查点")
        self.checkpoint_freq_spin = QSpinBox()
        self.checkpoint_freq_spin.setRange(1, 100)
        self.checkpoint_freq_spin.setValue(5)
        self.checkpoint_freq_spin.setStyleSheet(StyleSheet.settings_spinbox())
        self.checkpoint_freq_spin.setFont(FontManager.input_font())
        row_ckpt.set_control(self.checkpoint_freq_spin)
        layout.addWidget(row_ckpt)

        # 仅保存最佳模型
        self.save_best_check = SettingsCheckRow("仅保存最佳模型", "只保留验证精度最高的模型")
        self.save_best_check.setChecked(True)
        layout.addWidget(self.save_best_check)

    def _create_log_settings(self, layout: QVBoxLayout):
        """创建日志与调试设置"""
        layout.addWidget(self._create_section_title("日志与调试"))

        # 日志级别
        row_log_level = SettingsRow("日志级别", "控制日志输出的详细程度")
        self.log_level_combo = QComboBox()
        self.log_level_combo.addItems(["DEBUG", "INFO", "WARNING", "ERROR"])
        self.log_level_combo.setCurrentIndex(1)
        self.log_level_combo.setStyleSheet(StyleSheet.input_box())
        self.log_level_combo.setFont(FontManager.input_font())
        self.log_level_combo.setFixedWidth(180)
        row_log_level.set_control(self.log_level_combo)
        layout.addWidget(row_log_level)

        # 保留日志天数
        row_log_days = SettingsRow("保留日志天数", "自动清理超过指定天数的日志文件")
        self.log_days_spin = QSpinBox()
        self.log_days_spin.setRange(1, 365)
        self.log_days_spin.setValue(30)
        self.log_days_spin.setStyleSheet(StyleSheet.settings_spinbox())
        self.log_days_spin.setFont(FontManager.input_font())
        row_log_days.set_control(self.log_days_spin)
        layout.addWidget(row_log_days)

        # 详细日志
        self.verbose_check = SettingsCheckRow("启用详细日志", "记录更多调试信息，用于问题排查")
        layout.addWidget(self.verbose_check)

    def _create_other_settings(self, layout: QVBoxLayout):
        """创建其他设置"""
        layout.addWidget(self._create_section_title("其他设置"))

        # 自动加载模型
        self.auto_load_check = SettingsCheckRow("启动时自动加载默认模型", "程序启动时自动加载上次使用的模型")
        self.auto_load_check.setChecked(True)
        layout.addWidget(self.auto_load_check)

        # 记住窗口位置
        self.remember_window_check = SettingsCheckRow("记住窗口位置和大小", "下次启动时恢复窗口状态")
        self.remember_window_check.setChecked(True)
        layout.addWidget(self.remember_window_check)

        # 任务完成通知
        self.notify_check = SettingsCheckRow("任务完成后发送系统通知", "训练或分类完成时显示桌面通知")
        self.notify_check.setChecked(True)
        layout.addWidget(self.notify_check)

    def _on_theme_combo_changed(self, text: str):
        """主题选择变化"""
        theme_map = {"深色模式": "dark", "浅色模式": "light", "跟随系统": "system"}
        if text in theme_map:
            self.theme_changed.emit(theme_map[text])

    # ========== 公开接口 ==========

    def get_settings(self) -> dict:
        """获取所有设置"""
        return {
            "theme": self.theme_combo.currentText(),
            "scale": self.scale_combo.currentText(),
            "device": self.device_combo.currentText(),
            "precision": self.precision_combo.currentText(),
            "confidence_threshold": self.confidence_spin.value(),
            "model_dir": self.model_dir_input.text(),
            "dataset_dir": self.dataset_dir_input.text(),
            "log_dir": self.log_dir_input.text(),
            "export_dir": self.export_dir_input.text(),
            "workers": self.workers_spin.value(),
            "max_batch": self.max_batch_spin.value(),
            "amp_enabled": self.amp_check.isChecked(),
            "pin_memory": self.pin_memory_check.isChecked(),
            "default_epochs": self.default_epochs_spin.value(),
            "patience": self.patience_spin.value(),
            "checkpoint_freq": self.checkpoint_freq_spin.value(),
            "save_best_only": self.save_best_check.isChecked(),
            "log_level": self.log_level_combo.currentText(),
            "log_days": self.log_days_spin.value(),
            "verbose": self.verbose_check.isChecked(),
            "auto_load_model": self.auto_load_check.isChecked(),
            "remember_window": self.remember_window_check.isChecked(),
            "notify_on_complete": self.notify_check.isChecked(),
        }

    def set_settings(self, settings: dict):
        """设置所有设置"""
        if "theme" in settings:
            index = self.theme_combo.findText(settings["theme"])
            if index >= 0:
                self.theme_combo.setCurrentIndex(index)

        if "scale" in settings:
            index = self.scale_combo.findText(settings["scale"])
            if index >= 0:
                self.scale_combo.setCurrentIndex(index)

        if "device" in settings:
            index = self.device_combo.findText(settings["device"])
            if index >= 0:
                self.device_combo.setCurrentIndex(index)

        if "precision" in settings:
            index = self.precision_combo.findText(settings["precision"])
            if index >= 0:
                self.precision_combo.setCurrentIndex(index)

        if "confidence_threshold" in settings:
            self.confidence_spin.setValue(settings["confidence_threshold"])

        if "model_dir" in settings:
            self.model_dir_input.setText(settings["model_dir"])

        if "dataset_dir" in settings:
            self.dataset_dir_input.setText(settings["dataset_dir"])

        if "log_dir" in settings:
            self.log_dir_input.setText(settings["log_dir"])

        if "export_dir" in settings:
            self.export_dir_input.setText(settings["export_dir"])

        if "workers" in settings:
            self.workers_spin.setValue(settings["workers"])

        if "max_batch" in settings:
            self.max_batch_spin.setValue(settings["max_batch"])

        if "amp_enabled" in settings:
            self.amp_check.setChecked(settings["amp_enabled"])

        if "pin_memory" in settings:
            self.pin_memory_check.setChecked(settings["pin_memory"])

        if "default_epochs" in settings:
            self.default_epochs_spin.setValue(settings["default_epochs"])

        if "patience" in settings:
            self.patience_spin.setValue(settings["patience"])

        if "checkpoint_freq" in settings:
            self.checkpoint_freq_spin.setValue(settings["checkpoint_freq"])

        if "save_best_only" in settings:
            self.save_best_check.setChecked(settings["save_best_only"])

        if "log_level" in settings:
            index = self.log_level_combo.findText(settings["log_level"])
            if index >= 0:
                self.log_level_combo.setCurrentIndex(index)

        if "log_days" in settings:
            self.log_days_spin.setValue(settings["log_days"])

        if "verbose" in settings:
            self.verbose_check.setChecked(settings["verbose"])

        if "auto_load_model" in settings:
            self.auto_load_check.setChecked(settings["auto_load_model"])

        if "remember_window" in settings:
            self.remember_window_check.setChecked(settings["remember_window"])

        if "notify_on_complete" in settings:
            self.notify_check.setChecked(settings["notify_on_complete"])
