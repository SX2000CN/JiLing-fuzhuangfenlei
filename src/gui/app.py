"""
应用入口 - 模块化主窗口

新的 VS Code 风格界面，支持：
- 模块化页面（训练、分类、设置）
- 主题切换（深色、浅色、跟随系统）
- 响应式布局
"""

import sys
import logging
import ctypes
import ctypes.wintypes
from pathlib import Path
from typing import Optional, Any

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QStackedWidget, QButtonGroup, QMessageBox
)
from PySide6.QtCore import Qt, QThread, QSettings, QEvent
from PySide6.QtGui import QPainter, QColor, QPen

# Windows 常量
WM_NCHITTEST = 0x0084
WM_NCCALCSIZE = 0x0083
HTCLIENT = 1
HTCAPTION = 2
HTMAXBUTTON = 9  # 最大化按钮（用于 Snap 布局预览）
HTLEFT = 10
HTRIGHT = 11
HTTOP = 12
HTTOPLEFT = 13
HTTOPRIGHT = 14
HTBOTTOM = 15
HTBOTTOMLEFT = 16
HTBOTTOMRIGHT = 17

# Windows 窗口样式常量
GWL_STYLE = -16
WS_THICKFRAME = 0x00040000  # 允许调整大小边框
WS_CAPTION = 0x00C00000     # 标题栏（用于 Snap 功能）
WS_MAXIMIZEBOX = 0x00010000  # 最大化按钮（Snap 布局预览需要）

# 获取项目根目录
PROJECT_ROOT = Path(__file__).parent.parent.parent

# 添加项目路径
_src_path = str(PROJECT_ROOT / "src")
if _src_path not in sys.path:
    sys.path.insert(0, _src_path)

from .theme import StyleSheet, theme_manager, ColorTokens
from .utils import FontManager, IconSvg, SettingsManager
from .components import SidebarButton, WindowControlBar
from .widgets import GPUStatusWidget
from .pages import TrainingPage, ClassificationPage, SettingsPage
from .workers import TrainingWorker, ClassificationWorker
from core import ClothingClassifier

logger = logging.getLogger(__name__)


class MainWindow(QMainWindow):
    """主窗口 - 无边框 + 细边框装饰"""

    SIDEBAR_WIDTH = 72
    CORNER_RADIUS = 10  # 圆角半径
    RESIZE_MARGIN = 5   # 边缘调整大小的检测区域
    BORDER_WIDTH = 1    # 边框宽度（行业规范）

    def __init__(self):
        super().__init__()
        self.setWindowTitle("JiLing 服装分类系统")
        self.setMinimumSize(800, 500)
        self.resize(990, 660)

        # 无边框窗口
        self.setWindowFlags(Qt.FramelessWindowHint)
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setMouseTracking(True)

        # 拖拽相关
        self._drag_pos = None
        self._was_maximized = False
        self._drag_start_pos = None

        # 调整大小相关
        self._resize_edge = None
        self._resize_start_pos = None
        self._resize_start_geometry = None

        # 页面切换相关
        self._param_panel_expanded_width: dict[int, int] = {}
        self._param_panel_anim_group = None
        self._param_panel_anim_panel: Optional[QWidget] = None
        self._param_panel_anim_target: Optional[int] = None

        # 工作线程引用
        self.training_worker: Optional[TrainingWorker] = None
        self.training_thread: Optional[QThread] = None
        self.classification_worker: Optional[ClassificationWorker] = None
        self.classification_thread: Optional[QThread] = None
        self.current_classifier: Optional[Any] = None

        # 设置存储
        self.settings = QSettings("JiLing", "FuzhuangFenlei")
        self.settings_manager = SettingsManager()

        # 版本检查
        settings_version = self.settings.value("settings/version", 0, type=int)
        if settings_version < 2:
            self.settings.clear()
            self.settings.setValue("settings/version", 2)
            self.settings.sync()

        self._setup_ui()
        self._connect_signals()
        self._load_settings()

        # 设置 Windows 样式以支持 Snap 功能
        self._setup_windows_style()

        # 监听主题变化
        theme_manager.theme_changed.connect(self._on_theme_changed)

    def _on_theme_changed(self, theme: str):
        """主题变化时更新"""
        self._sidebar.setStyleSheet(StyleSheet.sidebar())
        # 更新边框颜色
        if not self.isMaximized():
            self._update_border_style()
        self.update()

    def _update_border_style(self):
        """更新边框样式（使用当前主题颜色）"""
        border_color = theme_manager.get_color(ColorTokens.BORDER_MUTED)
        self._border_style = f"""
            QWidget#centralWidget {{
                border: {self.BORDER_WIDTH}px solid {border_color};
                border-radius: {self.CORNER_RADIUS}px;
                background: transparent;
            }}
        """
        central = self.centralWidget()
        if central:
            central.setStyleSheet(self._border_style)

    def _setup_windows_style(self):
        """设置 Windows 窗口样式以支持 Snap 功能"""
        hwnd = int(self.winId())
        # 获取当前样式
        style = ctypes.windll.user32.GetWindowLongW(hwnd, GWL_STYLE)
        # 添加 WS_THICKFRAME 和 WS_CAPTION 以支持 Windows Snap
        # WS_MAXIMIZEBOX 用于支持悬停最大化按钮时的 Snap 布局预览
        style |= WS_THICKFRAME | WS_CAPTION | WS_MAXIMIZEBOX
        ctypes.windll.user32.SetWindowLongW(hwnd, GWL_STYLE, style)

    def paintEvent(self, event):
        """绘制窗口背景（边框通过 stylesheet 实现）"""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing, True)

        w, h = self.width(), self.height()
        sidebar_width = self._sidebar.width() if hasattr(self, "_sidebar") else self.SIDEBAR_WIDTH

        if self.isMaximized():
            # 最大化：无圆角，直接填充
            sidebar_color = theme_manager.get_color(ColorTokens.BG_SIDEBAR)
            painter.fillRect(0, 0, sidebar_width, h, QColor(sidebar_color))

            bg_color = theme_manager.get_color(ColorTokens.BG_PRIMARY)
            painter.fillRect(sidebar_width, 0, w - sidebar_width, h, QColor(bg_color))
        else:
            # 正常状态：圆角背景
            from PySide6.QtGui import QPainterPath

            radius = self.CORNER_RADIUS

            # 创建圆角路径用于裁剪
            clip_path = QPainterPath()
            clip_path.addRoundedRect(0, 0, w, h, radius, radius)
            painter.setClipPath(clip_path)

            # 绘制背景
            bg_color = theme_manager.get_color(ColorTokens.BG_PRIMARY)
            painter.fillRect(0, 0, w, h, QColor(bg_color))

            sidebar_color = theme_manager.get_color(ColorTokens.BG_SIDEBAR)
            painter.fillRect(0, 0, sidebar_width, h, QColor(sidebar_color))

        super().paintEvent(event)

    def _setup_ui(self):
        """设置 UI 布局"""
        central = QWidget()
        self.setCentralWidget(central)
        central.setObjectName("centralWidget")

        # 通过 stylesheet 设置边框（VS Code 风格）
        self._update_border_style()

        # 保存 main_layout 引用以便在最大化时修改边距
        b = self.BORDER_WIDTH
        self._main_layout = QHBoxLayout(central)
        self._main_layout.setContentsMargins(b, b, b, b)  # 为边框留出空间
        self._main_layout.setSpacing(0)

        # 侧边栏
        self._sidebar = self._create_sidebar()
        self._main_layout.addWidget(self._sidebar)

        # 页面堆栈
        self.page_stack = QStackedWidget()

        # 训练页面
        self.training_page = TrainingPage()
        self.page_stack.addWidget(self.training_page)

        # 分类页面
        self.classification_page = ClassificationPage()
        self.page_stack.addWidget(self.classification_page)

        # 设置页面
        self.settings_page = SettingsPage()
        self.page_stack.addWidget(self.settings_page)

        self._main_layout.addWidget(self.page_stack)

    def _toggle_maximize(self):
        """切换最大化/还原（带缩放动画）"""
        from .animations import animation_manager
        from PySide6.QtCore import QPropertyAnimation, QRect
        from PySide6.QtGui import QScreen

        if not animation_manager.enabled:
            self._do_toggle_maximize()
            return

        # 获取当前 geometry
        current_geo = self.geometry()
        is_maximized = self.isMaximized()

        if is_maximized:
            # 恢复：从最大化到之前保存的正常大小
            target_geo = getattr(self, '_saved_normal_geometry', current_geo)
            # 先退出最大化状态
            self.setWindowState(self.windowState() & ~Qt.WindowMaximized)
            # 准备恢复样式
            self._prepare_restore_style()
        else:
            # 最大化：从当前大小到全屏
            # 保存当前大小（用于恢复）
            self._saved_normal_geometry = current_geo
            # 获取屏幕可用区域
            screen = self.screen() or QApplication.primaryScreen()
            target_geo = screen.availableGeometry()
            # 先执行最大化的样式变化
            self._prepare_maximize_style()

        # 创建 geometry 动画
        self._geo_anim = QPropertyAnimation(self, b"geometry")
        self._geo_anim.setDuration(animation_manager._get_duration(200))
        self._geo_anim.setStartValue(current_geo)
        self._geo_anim.setEndValue(target_geo)
        self._geo_anim.setEasingCurve(animation_manager.create_ease_out_curve())

        # 动画完成后的处理
        def on_finished():
            if is_maximized:
                # 恢复完成
                pass
            else:
                # 最大化完成，应用最大化状态
                self._finalize_maximize()

        self._geo_anim.finished.connect(on_finished)
        self._geo_anim.start()

    def _prepare_maximize_style(self):
        """准备最大化样式（不调用 showMaximized）"""
        central = self.centralWidget()
        self.setAttribute(Qt.WA_TranslucentBackground, False)
        self._main_layout.setContentsMargins(0, 0, 0, 0)
        central.setStyleSheet("")
        self._update_maximize_icons(True)

    def _prepare_restore_style(self):
        """准备恢复样式"""
        central = self.centralWidget()
        b = self.BORDER_WIDTH
        self.setAttribute(Qt.WA_TranslucentBackground, True)
        self._main_layout.setContentsMargins(b, b, b, b)
        self._update_border_style()
        self._update_maximize_icons(False)

    def _finalize_maximize(self):
        """完成最大化"""
        # 标记为最大化状态（通过设置 window state）
        self.setWindowState(self.windowState() | Qt.WindowMaximized)

    def _do_toggle_maximize(self):
        """实际执行最大化/还原"""
        central = self.centralWidget()
        b = self.BORDER_WIDTH

        if self.isMaximized():
            # 还原：恢复边框和边距
            self.showNormal()
            self._update_maximize_icons(False)
            self.setAttribute(Qt.WA_TranslucentBackground, True)
            self._main_layout.setContentsMargins(b, b, b, b)
            self._update_border_style()
        else:
            # 最大化：移除边框和边距
            self.setAttribute(Qt.WA_TranslucentBackground, False)
            self._main_layout.setContentsMargins(0, 0, 0, 0)
            central.setStyleSheet("")
            self.showMaximized()
            self._update_maximize_icons(True)
        self.update()

    def _update_maximize_icons(self, is_maximized: bool):
        """更新所有页面的最大化按钮图标"""
        self.training_page._window_controls.update_maximize_button(is_maximized)
        self.classification_page._window_controls.update_maximize_button(is_maximized)
        self.settings_page._window_controls.update_maximize_button(is_maximized)
        self.training_page.update_maximize_state(is_maximized)
        self.classification_page.update_maximize_state(is_maximized)

    def changeEvent(self, event):
        """监听窗口状态变化（处理 Windows Snap 触发的最大化/还原）"""
        if event.type() == QEvent.WindowStateChange:
            central = self.centralWidget()
            b = self.BORDER_WIDTH
            is_maximized = self.isMaximized()

            if is_maximized:
                # 最大化：移除边框和边距
                self.setAttribute(Qt.WA_TranslucentBackground, False)
                self._main_layout.setContentsMargins(0, 0, 0, 0)
                central.setStyleSheet("")
            else:
                # 还原：恢复边框和边距
                self.setAttribute(Qt.WA_TranslucentBackground, True)
                self._main_layout.setContentsMargins(b, b, b, b)
                self._update_border_style()

            self._update_maximize_icons(is_maximized)
            self.update()

        super().changeEvent(event)

    def _create_sidebar(self) -> QWidget:
        """创建侧边栏"""
        sidebar = QWidget()
        sidebar.setObjectName("sidebar")
        sidebar.setFixedWidth(self.SIDEBAR_WIDTH)
        sidebar.setStyleSheet(StyleSheet.sidebar())

        layout = QVBoxLayout(sidebar)
        layout.setContentsMargins(0, 10, 0, 10)
        layout.setSpacing(10)

        # 训练按钮
        self.btn_train = SidebarButton(svg_template=IconSvg.TRAIN, icon_size=(24, 24))
        self.btn_train.setFixedSize(self.SIDEBAR_WIDTH, 50)
        self.btn_train.setChecked(True)
        layout.addWidget(self.btn_train)

        # 分类按钮
        self.btn_classify = SidebarButton(svg_template=IconSvg.CLASSIFY, icon_size=(24, 24))
        self.btn_classify.setFixedSize(self.SIDEBAR_WIDTH, 50)
        layout.addWidget(self.btn_classify)

        layout.addStretch()

        # 性能状态（紧凑模式）- 位于设置按钮上方
        self.gpu_status = GPUStatusWidget(compact=True)
        self.gpu_status.setFixedWidth(self.SIDEBAR_WIDTH - 6)
        layout.addWidget(self.gpu_status)

        # 设置按钮
        self.btn_settings = SidebarButton(svg_template=IconSvg.SETTINGS, icon_size=(28, 28))
        self.btn_settings.setFixedSize(self.SIDEBAR_WIDTH, 50)
        layout.addWidget(self.btn_settings)

        # 按钮组
        self.sidebar_group = QButtonGroup(self)
        self.sidebar_group.addButton(self.btn_train, 0)
        self.sidebar_group.addButton(self.btn_classify, 1)
        self.sidebar_group.addButton(self.btn_settings, 2)
        self.sidebar_group.setExclusive(True)

        return sidebar

    def _connect_signals(self):
        """连接信号"""
        # 侧边栏按钮
        self.btn_train.clicked.connect(lambda: self._switch_page(0))
        self.btn_classify.clicked.connect(lambda: self._switch_page(1))
        self.btn_settings.clicked.connect(lambda: self._switch_page(2))

        # 训练页面信号
        self.training_page.start_training_requested.connect(self._start_training)
        self.training_page.pause_training_requested.connect(self._pause_training)
        self.training_page.resume_training_requested.connect(self._resume_training)
        self.training_page.stop_training_requested.connect(self._stop_training)
        self.training_page.minimize_requested.connect(self.showMinimized)
        self.training_page.maximize_requested.connect(self._toggle_maximize)
        self.training_page.close_requested.connect(self.close)

        # 分类页面信号
        self.classification_page.load_model_requested.connect(self._load_classify_model)
        self.classification_page.use_default_model_requested.connect(self._use_default_model)
        self.classification_page.start_classification_requested.connect(self._start_classification)
        self.classification_page.clear_results_requested.connect(self._clear_classification_results)
        self.classification_page.minimize_requested.connect(self.showMinimized)
        self.classification_page.maximize_requested.connect(self._toggle_maximize)
        self.classification_page.close_requested.connect(self.close)

        # 设置页面信号
        self.settings_page.theme_changed.connect(self._on_theme_setting_changed)
        self.settings_page.settings_changed.connect(self._on_settings_changed)
        self.settings_page.minimize_requested.connect(self.showMinimized)
        self.settings_page.maximize_requested.connect(self._toggle_maximize)
        self.settings_page.close_requested.connect(self.close)

    def _switch_page(self, page_index: int):
        """切换页面（对齐非模块版的参数区收展逻辑）"""
        from .animations import animation_manager

        current_index = self.page_stack.currentIndex()
        if current_index == page_index:
            return

        self._sync_sidebar_buttons(page_index)
        self.settings.setValue("ui/current_page", page_index)
        self.settings.sync()

        if not animation_manager.enabled:
            self._apply_page_index(page_index)
            self._restore_page_param_area(page_index, animate=False)
            return

        current_param = self._get_page_param_area(current_index)

        # 对齐非模块版：切到设置页时，先收起参数区再切换
        if page_index == 2 and current_param is not None:
            current_width = current_param.width()
            if current_width > 0:
                self._param_panel_expanded_width[current_index] = current_width

            def switch_to_settings():
                self._apply_page_index(page_index)

            self._animate_page_param_width(current_param, 0, switch_to_settings)
            return

        # 先切页
        self._apply_page_index(page_index)

        # 从设置页返回时执行展开动画；普通页面互切时若参数区被意外收起则立即恢复
        self._restore_page_param_area(page_index, animate=(current_index == 2))

    def _apply_page_index(self, page_index: int):
        """应用页面索引并同步左侧状态"""
        self.page_stack.setCurrentIndex(page_index)
        self._sync_sidebar_buttons(page_index)

    def _get_page_param_area(self, page_index: int) -> Optional[QWidget]:
        """获取页面参数区控件（训练/分类页有，设置页无）"""
        page = self.page_stack.widget(page_index)
        param_area = getattr(page, "_param_area", None)
        if isinstance(param_area, QWidget):
            return param_area
        return None

    def _restore_page_param_area(self, page_index: int, animate: bool):
        """恢复训练/分类页参数区，避免切页后出现终端独占"""
        target_param = self._get_page_param_area(page_index)
        if target_param is None:
            return

        expanded_width = self._param_panel_expanded_width.get(page_index, 380)
        is_collapsed = (
            (not target_param.isVisible())
            or target_param.maximumWidth() == 0
            or target_param.width() == 0
        )

        if not is_collapsed and not animate:
            return

        target_param.setVisible(True)

        if animate:
            target_param.setMinimumWidth(0)
            target_param.setMaximumWidth(0)
            self._animate_page_param_width(target_param, expanded_width)
            return

        target_param.setMinimumWidth(0)
        target_param.setMaximumWidth(16777215)
        target_param.updateGeometry()

    def _finalize_param_panel_state(self, panel: Optional[QWidget], target_width: Optional[int]):
        """将参数区收敛到稳定状态，避免动画中断后卡在异常宽度"""
        if panel is None or target_width is None:
            return

        if target_width == 0:
            panel.setMinimumWidth(0)
            panel.setMaximumWidth(0)
            panel.setVisible(False)
            return

        panel.setVisible(True)
        panel.setMinimumWidth(0)
        panel.setMaximumWidth(16777215)
        panel.updateGeometry()

    def _animate_page_param_width(self, panel: QWidget, target_width: int, on_finished=None):
        """动画改变页面参数区宽度"""
        from .animations import animation_manager
        from PySide6.QtCore import QPropertyAnimation, QParallelAnimationGroup

        if self._param_panel_anim_group:
            self._param_panel_anim_group.stop()
            self._finalize_param_panel_state(self._param_panel_anim_panel, self._param_panel_anim_target)
            self._param_panel_anim_group = None
            self._param_panel_anim_panel = None
            self._param_panel_anim_target = None

        current_width = panel.width()
        if current_width == target_width:
            self._finalize_param_panel_state(panel, target_width)
            if on_finished:
                on_finished()
            return

        panel.setVisible(True)
        panel.setMinimumWidth(current_width)
        panel.setMaximumWidth(current_width)

        min_anim = QPropertyAnimation(panel, b"minimumWidth")
        min_anim.setDuration(animation_manager._get_duration(200))
        min_anim.setStartValue(current_width)
        min_anim.setEndValue(target_width)
        min_anim.setEasingCurve(animation_manager.create_ease_out_curve())

        max_anim = QPropertyAnimation(panel, b"maximumWidth")
        max_anim.setDuration(animation_manager._get_duration(200))
        max_anim.setStartValue(current_width)
        max_anim.setEndValue(target_width)
        max_anim.setEasingCurve(animation_manager.create_ease_out_curve())

        group = QParallelAnimationGroup(self)
        group.addAnimation(min_anim)
        group.addAnimation(max_anim)

        self._param_panel_anim_panel = panel
        self._param_panel_anim_target = target_width

        def _cleanup():
            self._finalize_param_panel_state(panel, target_width)
            self._param_panel_anim_group = None
            self._param_panel_anim_panel = None
            self._param_panel_anim_target = None
            if on_finished:
                on_finished()

        group.finished.connect(_cleanup)
        group.start()
        self._param_panel_anim_group = group

    def _sync_sidebar_buttons(self, page_index: int):
        """同步左侧栏按钮勾选状态"""
        self.btn_train.setChecked(page_index == 0)
        self.btn_classify.setChecked(page_index == 1)
        self.btn_settings.setChecked(page_index == 2)

    def _on_theme_setting_changed(self, theme: str):
        """处理主题设置变化"""
        theme_manager.set_theme(theme)
        # 保存主题设置
        self.settings.setValue("appearance/theme", theme)

    def _on_settings_changed(self, gui_settings: dict):
        """处理设置变化 - 保存到 SettingsManager"""
        self.settings_manager.from_gui_settings(gui_settings)
        self.settings_manager.save()

    # ========== 训练功能 ==========

    def _start_training(self):
        """开始训练"""
        params = self.training_page.get_training_params()

        if not params.get("data_path"):
            self.training_page.append_log("[ERROR] 请先选择数据路径")
            return

        self.training_page.append_log("[INFO] 开始训练...")
        self.training_page.set_training_state(True)

        # 从设置读取性能参数
        num_workers = self.settings_manager.get("performance.workers", 4)
        device = self.settings_manager.get("model.device", "auto")
        patience = self.settings_manager.get("training.patience", 10)
        amp_enabled = self.settings_manager.get("performance.amp_enabled", False)

        # 训练器配置
        trainer_config = {
            'model_name': params['model_type'],
            'device': device,
            'amp_enabled': amp_enabled
        }

        # 训练参数
        training_params = {
            'data_path': params['data_path'],
            'num_epochs': params['epochs'],
            'batch_size': params['batch_size'],
            'learning_rate': params['learning_rate'],
            'val_split': 0.2,
            'pretrained': True,
            'num_workers': num_workers,
            'patience': patience
        }

        self.training_page.append_log(f"[INFO] 模型类型: {params['model_type']}")
        self.training_page.append_log(f"[INFO] 训练轮数: {params['epochs']}")
        self.training_page.append_log(f"[INFO] 批次大小: {params['batch_size']}")
        self.training_page.append_log(f"[INFO] 学习率: {params['learning_rate']}")
        self.training_page.append_log(f"[INFO] 工作线程: {num_workers}")
        self.training_page.append_log(f"[INFO] 早停耐心值: {patience}")
        self.training_page.append_log(f"[INFO] 混合精度: {'启用' if amp_enabled else '禁用'}")

        # 创建工作线程
        self.training_worker = TrainingWorker(trainer_config, training_params)
        self.training_thread = QThread()
        self.training_worker.moveToThread(self.training_thread)

        # 连接信号
        self.training_thread.started.connect(self.training_worker.start_training)
        self.training_worker.progress_updated.connect(self._on_training_progress)
        self.training_worker.training_completed.connect(self._on_training_completed)
        self.training_worker.epoch_completed.connect(self._on_epoch_completed)

        self.training_thread.start()

    def _on_training_progress(self, progress: int, message: str, metrics: dict):
        """训练进度更新"""
        if hasattr(self.training_page, 'set_progress'):
            self.training_page.set_progress(progress)
        if message:
            self.training_page.append_log(f"[INFO] {message}")

    def _on_training_completed(self, success: bool, message: str):
        """训练完成"""
        self.training_page.set_training_state(False)
        log_level = "INFO" if success else "ERROR"
        self.training_page.append_log(f"[{log_level}] {message}")

        # 清理线程
        if self.training_thread:
            self.training_thread.quit()
            self.training_thread.wait()

    def _on_epoch_completed(self, epoch: int, metrics: dict):
        """轮次完成"""
        train_loss = metrics.get('train_loss', 0)
        train_acc = metrics.get('train_acc', 0)
        val_loss = metrics.get('val_loss', 0)
        val_acc = metrics.get('val_acc', 0)
        self.training_page.append_log(
            f"[INFO] Epoch {epoch}: train_loss={train_loss:.4f}, train_acc={train_acc:.2%}, "
            f"val_loss={val_loss:.4f}, val_acc={val_acc:.2%}"
        )

    def _stop_training(self):
        """停止训练"""
        if self.training_worker:
            self.training_worker.stop_training()
        self.training_page.append_log("[INFO] 停止训练...")
        self.training_page.set_training_state(False)

    def _pause_training(self):
        """暂停训练"""
        if self.training_worker:
            self.training_worker.pause_training()
        self.training_page.append_log("[INFO] 暂停训练...")

    def _resume_training(self):
        """继续训练"""
        if self.training_worker:
            self.training_worker.resume_training()
        self.training_page.append_log("[INFO] 继续训练...")

    # ========== 分类功能 ==========

    def _load_classify_model(self):
        """加载分类模型"""
        model_path = self.classification_page.get_model_file_path()
        if not model_path:
            QMessageBox.warning(self, "警告", "请选择模型文件")
            return

        self.classification_page.set_model_status(True, "加载中...")
        try:
            device = self.settings_manager.get("model.device", "auto")
            self.current_classifier = ClothingClassifier(model_path, device=device)
            self.classification_page.set_model_status(True, "已加载")
        except Exception as e:
            self.current_classifier = None
            self.classification_page.set_model_status(False, "加载失败")
            QMessageBox.critical(self, "错误", f"模型加载失败: {e}")

    def _use_default_model(self):
        """使用默认模型"""
        default_path = PROJECT_ROOT / "models" / "saved_models" / "best_model.pth"
        if not default_path.exists():
            QMessageBox.warning(self, "警告", f"默认模型不存在: {default_path}")
            return

        self.classification_page.set_model_status(True, "加载中...")
        try:
            device = self.settings_manager.get("model.device", "auto")
            self.current_classifier = ClothingClassifier(str(default_path), device=device)
            self.classification_page.set_model_status(True, "已加载（默认）")
        except Exception as e:
            self.current_classifier = None
            self.classification_page.set_model_status(False, "加载失败")
            QMessageBox.critical(self, "错误", f"默认模型加载失败: {e}")

    def _start_classification(self):
        """开始分类"""
        folder_path = self.classification_page.get_folder_path()
        single_file = self.classification_page.get_single_file_path()

        if not folder_path and not single_file:
            QMessageBox.warning(self, "警告", "请选择图片文件或文件夹")
            return

        if not self.current_classifier:
            QMessageBox.warning(self, "警告", "请先加载模型")
            return

        # 收集图片路径
        image_paths = []
        if folder_path:
            folder = Path(folder_path)
            for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.webp']:
                image_paths.extend(folder.glob(ext))
                image_paths.extend(folder.glob(ext.upper()))
        if single_file:
            image_paths.append(Path(single_file))

        if not image_paths:
            QMessageBox.warning(self, "警告", "未找到图片文件")
            return

        # 从设置读取置信度阈值
        confidence_threshold = self.settings_manager.get("model.confidence_threshold", 0.5)

        self.classification_page.set_progress(0, True)

        # 创建工作线程
        self.classification_worker = ClassificationWorker(
            image_paths=[str(p) for p in image_paths],
            classifier=self.current_classifier,
            output_folder=folder_path,
            confidence_threshold=confidence_threshold
        )
        self.classification_thread = QThread()
        self.classification_worker.moveToThread(self.classification_thread)

        # 连接信号
        self.classification_thread.started.connect(self.classification_worker.start_classification)
        self.classification_worker.progress_updated.connect(self._on_classification_progress)
        self.classification_worker.classification_completed.connect(self._on_classification_completed)

        self.classification_thread.start()

    def _on_classification_progress(self, progress: int, message: str):
        """分类进度更新"""
        self.classification_page.set_progress(progress, True)

    def _on_classification_completed(self, results: list):
        """分类完成"""
        self.classification_page.set_progress(100, False)
        self.classification_page.set_results(results)

        # 生成统计摘要
        if results:
            class_counts = {}
            total_confidence = 0
            uncertain_count = 0

            for item in results:
                result = item.get('result', {})
                pred_class = result.get('predicted_class', 'unknown')
                confidence = result.get('confidence', 0)
                uncertain = result.get('uncertain', False)

                class_counts[pred_class] = class_counts.get(pred_class, 0) + 1
                total_confidence += confidence
                if uncertain:
                    uncertain_count += 1

            avg_confidence = total_confidence / len(results) if results else 0

            # 显示统计信息
            summary = f"分类完成: {len(results)} 张图片\n"
            summary += f"平均置信度: {avg_confidence:.1%}\n"
            summary += f"低置信度项: {uncertain_count} 张\n"
            summary += "各类别数量:\n"
            for cls, count in sorted(class_counts.items()):
                summary += f"  - {cls}: {count} 张\n"

            self.classification_page.show_summary(summary)

        # 清理线程
        if self.classification_thread:
            self.classification_thread.quit()
            self.classification_thread.wait()

    def _clear_classification_results(self):
        """清空分类结果"""
        self.classification_page.clear_results()

    # ========== 设置 ==========

    def _load_settings(self):
        """加载设置"""
        from .animations import animation_manager

        # 加载主题
        theme = self.settings.value("appearance/theme", "dark", type=str)
        theme_manager.set_theme(theme)

        # 从 SettingsManager 加载设置并应用到设置页面
        self.settings_manager.load()
        gui_settings = self.settings_manager.to_gui_settings()
        self.settings_page.set_settings(gui_settings)

        # 应用动画设置
        animations_enabled = self.settings_manager.get("appearance.animations_enabled", True)
        animation_manager.enabled = animations_enabled

        # 恢复左侧栏当前页状态
        saved_page = self.settings.value("ui/current_page", 0, type=int)
        if saved_page not in (0, 1, 2):
            saved_page = 0
        self.page_stack.setCurrentIndex(saved_page)
        self._sync_sidebar_buttons(saved_page)

    def _save_settings(self):
        """保存设置"""
        gui_settings = self.settings_page.get_settings()
        self.settings_manager.from_gui_settings(gui_settings)
        self.settings_manager.save()
        self.settings.setValue("ui/current_page", self.page_stack.currentIndex())
        self.settings.sync()

    # ========== 窗口拖拽和调整大小 ==========

    def _get_resize_edge(self, pos):
        """检测鼠标位置对应的调整边缘"""
        if self.isMaximized():
            return None

        x, y = pos.x(), pos.y()
        w, h = self.width(), self.height()
        m = self.RESIZE_MARGIN

        edges = []
        if x <= m:
            edges.append('left')
        elif x >= w - m:
            edges.append('right')
        if y <= m:
            edges.append('top')
        elif y >= h - m:
            edges.append('bottom')

        if edges:
            return '-'.join(edges)
        return None

    def _update_cursor(self, edge):
        """根据边缘更新鼠标光标"""
        if edge is None:
            self.setCursor(Qt.ArrowCursor)
        elif edge in ('left', 'right'):
            self.setCursor(Qt.SizeHorCursor)
        elif edge in ('top', 'bottom'):
            self.setCursor(Qt.SizeVerCursor)
        elif edge in ('left-top', 'top-left', 'right-bottom', 'bottom-right'):
            self.setCursor(Qt.SizeFDiagCursor)
        elif edge in ('right-top', 'top-right', 'left-bottom', 'bottom-left'):
            self.setCursor(Qt.SizeBDiagCursor)

    def mousePressEvent(self, event):
        """鼠标按下 - 开始拖拽或调整大小"""
        if event.button() == Qt.LeftButton:
            pos = event.position().toPoint()
            edge = self._get_resize_edge(pos)

            if edge:
                self._resize_edge = edge
                self._resize_start_pos = event.globalPosition().toPoint()
                self._resize_start_geometry = self.geometry()
            else:
                self._drag_pos = event.globalPosition().toPoint() - self.frameGeometry().topLeft()
                self._was_maximized = self.isMaximized()
                self._drag_start_pos = event.globalPosition().toPoint()

    def mouseMoveEvent(self, event):
        """鼠标移动 - 执行拖拽或调整大小"""
        if self._resize_edge and event.buttons() == Qt.LeftButton:
            self._do_resize(event.globalPosition().toPoint())
            return

        if self._drag_pos and event.buttons() == Qt.LeftButton:
            if self._was_maximized:
                # 从最大化拖拽还原：恢复边框和边距
                self._was_maximized = False
                mouse_x_ratio = self._drag_start_pos.x() / self.width() if self.width() > 0 else 0.5
                self.showNormal()
                self._update_maximize_icons(False)
                self.setAttribute(Qt.WA_TranslucentBackground, True)
                b = self.BORDER_WIDTH
                self._main_layout.setContentsMargins(b, b, b, b)
                self._update_border_style()
                self.update()
                new_x = int(event.globalPosition().toPoint().x() - self.width() * mouse_x_ratio)
                new_y = event.globalPosition().toPoint().y() - 16
                self.move(new_x, new_y)
                self._drag_pos = event.globalPosition().toPoint() - self.frameGeometry().topLeft()
            else:
                self.move(event.globalPosition().toPoint() - self._drag_pos)
            return

        pos = event.position().toPoint()
        edge = self._get_resize_edge(pos)
        self._update_cursor(edge)

    def _do_resize(self, global_pos):
        """执行调整大小"""
        if not self._resize_start_geometry:
            return

        dx = global_pos.x() - self._resize_start_pos.x()
        dy = global_pos.y() - self._resize_start_pos.y()
        geo = self._resize_start_geometry

        new_x, new_y = geo.x(), geo.y()
        new_w, new_h = geo.width(), geo.height()

        if 'left' in self._resize_edge:
            new_x = geo.x() + dx
            new_w = geo.width() - dx
        if 'right' in self._resize_edge:
            new_w = geo.width() + dx
        if 'top' in self._resize_edge:
            new_y = geo.y() + dy
            new_h = geo.height() - dy
        if 'bottom' in self._resize_edge:
            new_h = geo.height() + dy

        min_w, min_h = self.minimumSize().width(), self.minimumSize().height()
        if new_w < min_w:
            if 'left' in self._resize_edge:
                new_x = geo.x() + geo.width() - min_w
            new_w = min_w
        if new_h < min_h:
            if 'top' in self._resize_edge:
                new_y = geo.y() + geo.height() - min_h
            new_h = min_h

        self.setGeometry(new_x, new_y, new_w, new_h)

    def mouseReleaseEvent(self, event):
        """鼠标释放"""
        if event.button() == Qt.LeftButton:
            if self._drag_pos:
                mouse_y = event.globalPosition().toPoint().y()
                if mouse_y <= 10 and not self.isMaximized():
                    # 拖到顶部最大化：移除边框和边距
                    central = self.centralWidget()
                    self.setAttribute(Qt.WA_TranslucentBackground, False)
                    self._main_layout.setContentsMargins(0, 0, 0, 0)
                    central.setStyleSheet("")
                    self.showMaximized()
                    self._update_maximize_icons(True)
                    self.update()

            self._drag_pos = None
            self._was_maximized = False
            self._resize_edge = None
            self._resize_start_pos = None
            self._resize_start_geometry = None

    def nativeEvent(self, eventType, message):
        """处理 Windows 原生消息"""
        if eventType == b"windows_generic_MSG":
            msg = ctypes.wintypes.MSG.from_address(int(message))

            if msg.message == WM_NCHITTEST:
                x = msg.lParam & 0xFFFF
                y = (msg.lParam >> 16) & 0xFFFF

                if x > 32767:
                    x -= 65536
                if y > 32767:
                    y -= 65536

                local_x = x - self.frameGeometry().x()
                local_y = y - self.frameGeometry().y()

                w, h = self.width(), self.height()
                m = self.RESIZE_MARGIN

                # 控制按钮区域（让 Qt 处理）
                controls_width = 140
                in_controls_area = local_x >= w - controls_width and local_y < 32

                if self.isMaximized():
                    if in_controls_area:
                        return False, 0
                    if local_y < 32:
                        return True, HTCAPTION
                    return False, 0

                if in_controls_area:
                    return False, 0

                on_left = local_x <= m
                on_right = local_x >= w - m
                on_top = local_y <= m
                on_bottom = local_y >= h - m

                if on_left and on_top:
                    return True, HTTOPLEFT
                if on_right and on_top:
                    return True, HTTOPRIGHT
                if on_left and on_bottom:
                    return True, HTBOTTOMLEFT
                if on_right and on_bottom:
                    return True, HTBOTTOMRIGHT

                if on_left:
                    return True, HTLEFT
                if on_right:
                    return True, HTRIGHT
                if on_top:
                    return True, HTTOP
                if on_bottom:
                    return True, HTBOTTOM

                if local_y < 32:
                    return True, HTCAPTION

            elif msg.message == WM_NCCALCSIZE:
                return True, 0

        return super().nativeEvent(eventType, message)

    def closeEvent(self, event):
        """关闭事件 - 保存设置"""
        self._save_settings()
        super().closeEvent(event)


def main():
    """应用主入口"""
    try:
        logger.info("[1] 创建 QApplication...")
        app = QApplication(sys.argv)
        app.setApplicationName("JiLing 服装分类系统")
        app.setOrganizationName("JiLing")
        logger.info("    QApplication 创建完成")

        logger.info("[2] 加载字体...")
        FontManager.load_fonts()
        logger.info("    字体加载完成")

        logger.info("[3] 创建 MainWindow...")
        window = MainWindow()
        logger.info("    MainWindow 创建完成")

        logger.info("[4] 显示窗口...")
        window.show()
        logger.info("    窗口已显示")

        logger.info("[5] 进入事件循环...")
        sys.exit(app.exec())
    except Exception as e:
        logger.exception("应用启动失败: %s", e)
        sys.exit(1)


# 保留旧入口的兼容性
def legacy_main():
    """旧版入口 - 使用 native_ui.py"""
    from .native_ui import main as _legacy_main
    _legacy_main()


if __name__ == "__main__":
    main()

