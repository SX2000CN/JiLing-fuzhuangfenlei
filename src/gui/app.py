"""
应用入口 - 模块化主窗口

新的 VS Code 风格界面，支持：
- 模块化页面（训练、分类、设置）
- 主题切换（深色、浅色、跟随系统）
- 响应式布局
"""

import sys
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
from .utils import FontManager, IconSvg
from .components import SidebarButton, WindowControlBar
from .pages import TrainingPage, ClassificationPage, SettingsPage
from .workers import TrainingWorker, ClassificationWorker


class MainWindow(QMainWindow):
    """主窗口 - 无边框 + 细边框装饰"""

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

        # 工作线程引用
        self.training_worker: Optional[TrainingWorker] = None
        self.training_thread: Optional[QThread] = None
        self.classification_worker: Optional[ClassificationWorker] = None
        self.classification_thread: Optional[QThread] = None
        self.current_classifier: Optional[Any] = None

        # 设置存储
        self.settings = QSettings("JiLing", "FuzhuangFenlei")

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
        sidebar_width = 50

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
        """切换最大化/还原"""
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
        sidebar.setFixedWidth(50)
        sidebar.setStyleSheet(StyleSheet.sidebar())

        layout = QVBoxLayout(sidebar)
        layout.setContentsMargins(0, 10, 0, 10)
        layout.setSpacing(10)

        # 训练按钮
        self.btn_train = SidebarButton(svg_template=IconSvg.TRAIN, icon_size=(24, 24))
        self.btn_train.setFixedSize(50, 50)
        self.btn_train.setChecked(True)
        layout.addWidget(self.btn_train)

        # 分类按钮
        self.btn_classify = SidebarButton(svg_template=IconSvg.CLASSIFY, icon_size=(24, 24))
        self.btn_classify.setFixedSize(50, 50)
        layout.addWidget(self.btn_classify)

        layout.addStretch()

        # 设置按钮
        self.btn_settings = SidebarButton(svg_template=IconSvg.SETTINGS, icon_size=(28, 28))
        self.btn_settings.setFixedSize(50, 50)
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
        self.settings_page.minimize_requested.connect(self.showMinimized)
        self.settings_page.maximize_requested.connect(self._toggle_maximize)
        self.settings_page.close_requested.connect(self.close)

    def _switch_page(self, page_index: int):
        """切换页面"""
        self.page_stack.setCurrentIndex(page_index)

    def _on_theme_setting_changed(self, theme: str):
        """处理主题设置变化"""
        theme_manager.set_theme(theme)
        # 保存主题设置
        self.settings.setValue("appearance/theme", theme)

    # ========== 训练功能 ==========

    def _start_training(self):
        """开始训练"""
        params = self.training_page.get_training_params()

        if not params.get("data_path"):
            self.training_page.append_log("[ERROR] 请先选择数据路径")
            return

        self.training_page.append_log("[INFO] 开始训练...")
        self.training_page.set_training_state(True)

        # TODO: 实现实际训练逻辑
        self.training_page.append_log(f"[INFO] 模型类型: {params['model_type']}")
        self.training_page.append_log(f"[INFO] 训练轮数: {params['epochs']}")
        self.training_page.append_log(f"[INFO] 批次大小: {params['batch_size']}")
        self.training_page.append_log(f"[INFO] 学习率: {params['learning_rate']}")

    def _stop_training(self):
        """停止训练"""
        if self.training_worker:
            self.training_worker.should_stop = True
        self.training_page.append_log("[INFO] 停止训练...")
        self.training_page.set_training_state(False)

    # ========== 分类功能 ==========

    def _load_classify_model(self):
        """加载分类模型"""
        model_path = self.classification_page.get_model_file_path()
        if not model_path:
            QMessageBox.warning(self, "警告", "请选择模型文件")
            return

        self.classification_page.set_model_status(True, "加载中...")
        # TODO: 实现实际加载逻辑
        self.classification_page.set_model_status(True, "已加载")

    def _use_default_model(self):
        """使用默认模型"""
        self.classification_page.set_model_status(True, "默认模型")
        # TODO: 实现默认模型加载逻辑

    def _start_classification(self):
        """开始分类"""
        folder_path = self.classification_page.get_folder_path()
        single_file = self.classification_page.get_single_file_path()

        if not folder_path and not single_file:
            QMessageBox.warning(self, "警告", "请选择图片文件或文件夹")
            return

        self.classification_page.set_progress(0, True)
        # TODO: 实现实际分类逻辑

    def _clear_classification_results(self):
        """清空分类结果"""
        self.classification_page.clear_results()

    # ========== 设置 ==========

    def _load_settings(self):
        """加载设置"""
        theme = self.settings.value("appearance/theme", "dark", type=str)
        theme_manager.set_theme(theme)

    def _save_settings(self):
        """保存设置"""
        settings = self.settings_page.get_settings()
        for key, value in settings.items():
            self.settings.setValue(f"settings/{key}", value)
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
    import traceback

    try:
        print("[1] 创建 QApplication...")
        app = QApplication(sys.argv)
        app.setApplicationName("JiLing 服装分类系统")
        app.setOrganizationName("JiLing")
        print("    QApplication 创建完成")

        print("[2] 加载字体...")
        FontManager.load_fonts()
        print("    字体加载完成")

        print("[3] 创建 MainWindow...")
        window = MainWindow()
        print("    MainWindow 创建完成")

        print("[4] 显示窗口...")
        window.show()
        print("    窗口已显示")

        print("[5] 进入事件循环...")
        sys.exit(app.exec())
    except Exception as e:
        print(f"\n错误: {type(e).__name__}: {e}")
        traceback.print_exc()
        input("按回车键退出...")
        sys.exit(1)


# 保留旧入口的兼容性
def legacy_main():
    """旧版入口 - 使用 native_ui.py"""
    from .native_ui import main as _legacy_main
    _legacy_main()


if __name__ == "__main__":
    main()
